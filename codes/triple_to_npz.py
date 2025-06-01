import os
import torch
import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from acestep.pipeline_ace_step import ACEStepPipeline
from loguru import logger
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Convert triple files (audio + prompt + lyrics) to .npz files with prompt embeddings from local checkpoints")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing triple files (.mp3/.wav, _prompt.txt, _lyrics.txt)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints (e.g., /mnt/f/ACE/ACE_checkpoints_3.5B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed .npz files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run preprocessing (cuda or cpu)"
    )
    parser.add_argument(
        "--max_ssl_len",
        type=int,
        default=4096,
        help="Maximum length for SSL hidden states after downsampling"
    )
    parser.add_argument(
        "--max_latent_len",
        type=int,
        default=3188,
        help="Maximum length for target_latents (corresponding to 240s)"
    )
    parser.add_argument(
        "--max_lyric_len",
        type=int,
        default=1291,
        help="Maximum length for lyric token IDs"
    )
    return parser.parse_args()

def downsample_ssl(ssl_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Downsample SSL features using linear interpolation.
    Args:
        ssl_tensor: Tensor [T, D]
        target_len: Target number of timesteps
    Returns:
        Tensor [target_len, D]
    """
    if ssl_tensor.shape[0] <= target_len:
        return ssl_tensor[:target_len]
    ssl_tensor = ssl_tensor.unsqueeze(0).transpose(1, 2)  # [1, D, T]
    ssl_tensor = F.interpolate(ssl_tensor, size=target_len, mode='linear', align_corners=False)
    return ssl_tensor.transpose(1, 2).squeeze(0)  # [target_len, D]

def infer_mert_ssl(target_wavs, wav_lengths, mert_model, mert_resampler, device, max_ssl_len):
    #logger.debug(f"MERT input shape: {target_wavs.shape}, wav_lengths: {wav_lengths}")
    mert_input_wavs_mono_24k = mert_resampler(target_wavs.mean(dim=1).to(torch.float32))
    bsz = target_wavs.shape[0]
    actual_lengths_24k = wav_lengths // 2

    #logger.debug(f"mert_input_wavs_mono_24k shape: {mert_input_wavs_mono_24k.shape}, actual_lengths_24k: {actual_lengths_24k}")

    means = torch.stack(
        [mert_input_wavs_mono_24k[i, :actual_lengths_24k[i]].mean() for i in range(bsz)]
    )
    vars = torch.stack(
        [mert_input_wavs_mono_24k[i, :actual_lengths_24k[i]].var() for i in range(bsz)]
    )
    mert_input_wavs_mono_24k = (mert_input_wavs_mono_24k - means.view(-1, 1)) / torch.sqrt(vars.view(-1, 1) + 1e-7)

    chunk_size = 24000 * 5
    num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size

    all_chunks = []
    chunk_actual_lengths = []
    for i in range(bsz):
        audio = mert_input_wavs_mono_24k[i]
        actual_length = actual_lengths_24k[i]
        for start in range(0, actual_length, chunk_size):
            end = min(start + chunk_size, actual_length)
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            all_chunks.append(chunk)
            chunk_actual_lengths.append(end - start)

    all_chunks = torch.stack(all_chunks, dim=0).to(device).to(torch.bfloat16)
    #logger.debug(f"all_chunks shape: {all_chunks.shape}")
    with torch.no_grad():
        mert_ssl_hidden_states = mert_model(all_chunks).last_hidden_state

    chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]
    chunk_hidden_states = [
        mert_ssl_hidden_states[i, :chunk_num_features[i], :] for i in range(len(all_chunks))
    ]

    mert_ssl_hidden_states_list = []
    chunk_idx = 0
    for i in range(bsz):
        audio_chunks = chunk_hidden_states[chunk_idx:chunk_idx + num_chunks_per_audio[i]]
        audio_hidden = torch.cat(audio_chunks, dim=0)
        audio_hidden = downsample_ssl(audio_hidden, max_ssl_len)
        mert_ssl_hidden_states_list.append(audio_hidden.to(torch.float32).cpu().numpy())
        chunk_idx += num_chunks_per_audio[i]
    return mert_ssl_hidden_states_list

def infer_mhubert_ssl(target_wavs, wav_lengths, mhubert_model, mhubert_resampler, device, max_ssl_len):
    #logger.debug(f"mHuBERT input shape: {target_wavs.shape}, wav_lengths: {wav_lengths}")
    mhubert_input_wavs_mono_16k = mhubert_resampler(target_wavs.mean(dim=1).to(torch.float32))
    bsz = target_wavs.shape[0]
    actual_lengths_16k = wav_lengths // 3

    #logger.debug(f"mhubert_input_wavs_mono_16k shape: {mhubert_input_wavs_mono_16k.shape}, actual_lengths_16k: {actual_lengths_16k}")

    means = torch.stack(
        [mhubert_input_wavs_mono_16k[i, :actual_lengths_16k[i]].mean() for i in range(bsz)]
    )
    vars = torch.stack(
        [mhubert_input_wavs_mono_16k[i, :actual_lengths_16k[i]].var() for i in range(bsz)]
    )
    mhubert_input_wavs_mono_16k = (mhubert_input_wavs_mono_16k - means.view(-1, 1)) / torch.sqrt(vars.view(-1, 1) + 1e-7)

    chunk_size = 16000 * 30
    num_chunks_per_audio = (actual_lengths_16k + chunk_size - 1) // chunk_size

    all_chunks = []
    chunk_actual_lengths = []
    for i in range(bsz):
        audio = mhubert_input_wavs_mono_16k[i]
        actual_length = actual_lengths_16k[i]
        for start in range(0, actual_length, chunk_size):
            end = min(start + chunk_size, actual_length)
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            all_chunks.append(chunk)
            chunk_actual_lengths.append(end - start)

    all_chunks = torch.stack(all_chunks, dim=0).to(device).to(torch.bfloat16)
    #logger.debug(f"all_chunks shape: {all_chunks.shape}")
    with torch.no_grad():
        mhubert_ssl_hidden_states = mhubert_model(all_chunks).last_hidden_state

    chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]
    chunk_hidden_states = [
        mhubert_ssl_hidden_states[i, :chunk_num_features[i], :] for i in range(len(all_chunks))
    ]

    mhuburt_ssl_hidden_states_list = []
    chunk_idx = 0
    for i in range(bsz):
        audio_chunks = chunk_hidden_states[chunk_idx:chunk_idx + num_chunks_per_audio[i]]
        audio_hidden = torch.cat(audio_chunks, dim=0)
        audio_hidden = downsample_ssl(audio_hidden, max_ssl_len)
        mhuburt_ssl_hidden_states_list.append(audio_hidden.to(torch.float32).cpu().numpy())
        chunk_idx += num_chunks_per_audio[i]
    return mhuburt_ssl_hidden_states_list

def check_model_paths(checkpoint_dir):
    """Check if required model directories exist in checkpoint_dir."""
    required_models = {
        'music_dcae': os.path.join(checkpoint_dir, 'music_dcae_f8c8'),
        'ace_step_transformer': os.path.join(checkpoint_dir, 'ace_step_transformer'),
        'umt5': os.path.join(checkpoint_dir, 'umt5-base'),
        'mert': os.path.join(checkpoint_dir, 'MERT-v1-330M'),
        'mhubert': os.path.join(checkpoint_dir, 'mHuBERT-147')
    }
    for model_name, path in required_models.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        logger.info(f"Found {model_name} model at: {path}")
    return required_models

def main():
    args = parse_args()
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    device = args.device
    max_ssl_len = args.max_ssl_len
    max_latent_len = args.max_latent_len
    max_lyric_len = args.max_lyric_len

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"NPZ directory created at: {output_dir}")

    # Check model paths
    model_paths = check_model_paths(checkpoint_dir)

    # Initialize pipeline
    pipeline = ACEStepPipeline(
        checkpoint_dir=checkpoint_dir,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=False,
        device=device
    )
    pipeline.load_checkpoint(checkpoint_dir=checkpoint_dir)
    pipeline.music_dcae.to(device).to(torch.bfloat16)
    logger.info("Initialized pipeline and moved music_dcae to device")

    # Load MERT and mHuBERT models from local paths
    mert_model = AutoModel.from_pretrained(
        model_paths['mert'],
        trust_remote_code=True,
        local_files_only=True
    ).eval().to(device).to(torch.bfloat16)
    mert_resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=24000).to(device)
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_paths['mert'],
        trust_remote_code=True,
        local_files_only=True
    )

    mhubert_model = AutoModel.from_pretrained(
        model_paths['mhubert'],
        trust_remote_code=True,
        local_files_only=True
    ).eval().to(device).to(torch.bfloat16)
    mhubert_resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(device)
    mhubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_paths['mhubert'],
        trust_remote_code=True,
        local_files_only=True
    )
    logger.info("Loaded MERT and mHuBERT models from local paths")

    # Scan input directory for triple files
    data_path = Path(data_dir)
    sample_idx = 0
    audio_extensions = [".mp3", ".wav"]

    for audio_path in tqdm(data_path.glob("*"), desc="Processing audio files"):
        if audio_path.suffix.lower() not in audio_extensions:
            continue
        
        prompt_path = str(audio_path).replace(audio_path.suffix, "_prompt.txt")
        lyric_path = str(audio_path).replace(audio_path.suffix, "_lyrics.txt")
        
        try:
            if not (os.path.exists(prompt_path) and os.path.exists(lyric_path)):
                logger.warning(f"Skipping {audio_path}: Missing prompt or lyrics file")
                continue

            # Read prompt and lyrics
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()

            keys = audio_path.stem

            # Load and preprocess audio
            waveform, sr = torchaudio.load(audio_path)
            #logger.debug(f"Loaded {audio_path}, waveform shape: {waveform.shape}, sample rate: {sr}")
            if sr != 48000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(waveform)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
            
            max_samples = 48000 * 240  # 240s at 48kHz
            if waveform.shape[-1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            waveform = waveform.to(device).to(torch.bfloat16)
            wav_lengths = torch.tensor([waveform.shape[-1]], dtype=torch.long).to(device)

            # Encode audio to latents
            with torch.no_grad():
                latents, _ = pipeline.music_dcae.encode(waveform.unsqueeze(0), wav_lengths)
            latents = latents[0].to(torch.float32).cpu().numpy()
            if latents.shape[-1] > max_latent_len:
                latents = latents[:, :, :max_latent_len]
            attention_mask = np.ones(latents.shape[-1], dtype=np.float32)

            # Compute SSL hidden states
            # Pass waveform with batch dimension [1, channels, samples]
            mert_ssl_hidden_states = infer_mert_ssl(
                waveform.unsqueeze(0), wav_lengths, mert_model, mert_resampler, device, max_ssl_len
            )[0]
            mhuburt_ssl_hidden_states = infer_mhubert_ssl(
                waveform.unsqueeze(0), wav_lengths, mhubert_model, mhubert_resampler, device, max_ssl_len
            )[0]

            # Create text embeddings for prompt
            with torch.no_grad():
                prompt_text_hidden_states, prompt_text_attention_mask = pipeline.get_text_embeddings([prompt], device=device)
            prompt_text_hidden_states = prompt_text_hidden_states[0].to(torch.float32).cpu().numpy()
            prompt_text_attention_mask = prompt_text_attention_mask[0].to(torch.float32).cpu().numpy()

            # Tokenize lyrics
            lyric_token_ids = pipeline.tokenize_lyrics(lyrics)
            lyric_mask = np.ones(len(lyric_token_ids), dtype=np.int32)
            lyric_token_ids = np.array(lyric_token_ids, dtype=np.int32)
            if len(lyric_token_ids) < max_lyric_len:
                lyric_token_ids = np.pad(
                    lyric_token_ids, (0, max_lyric_len - len(lyric_token_ids)), mode="constant"
                )
                lyric_mask = np.pad(
                    lyric_mask, (0, max_lyric_len - len(lyric_mask)), mode="constant", constant_values=1
                )
            lyric_token_ids = lyric_token_ids[:max_lyric_len]
            lyric_mask = lyric_mask[:max_lyric_len]

            # Save to .npz file
            npz_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.npz")
            np.savez_compressed(
                npz_path,
                keys=keys,
                target_latents=latents,
                attention_mask=attention_mask,
                prompt_text_hidden_states=prompt_text_hidden_states,
                prompt_text_attention_mask=prompt_text_attention_mask,
                lyric_token_ids=lyric_token_ids,
                lyric_mask=lyric_mask,
                mert_ssl_hidden_states=mert_ssl_hidden_states,
                mhuburt_ssl_hidden_states=mhuburt_ssl_hidden_states
            )
            logger.info(f"Saved sample {sample_idx} to {npz_path}")

            sample_idx += 1

            # Clear VRAM
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            continue

    # Final VRAM cleanup
    torch.cuda.empty_cache()
    logger.info(f"Processing complete. {sample_idx} samples saved to {output_dir}")

if __name__ == "__main__":
    main()