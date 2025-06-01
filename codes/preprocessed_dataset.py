import torch
import numpy as np
from glob import glob
from pathlib import Path
import os
from loguru import logger
from my_codes.my_gpu import print_memory_usage
import time

class PreprocessedText2MusicDataset(torch.utils.data.Dataset):
    def __init__(self, npz_dir, device="cpu", max_len=None):
        super().__init__()
        self.npz_dir = npz_dir
        self.device = device
        logger.debug(f"Attempting to load dataset from directory: {self.npz_dir}") 
        
        try:
            self.npz_files = list(Path(npz_dir).rglob("*.npz"))
            logger.info(f"Found {len(self.npz_files)} .npz files in {npz_dir}")
        except Exception as e:
            logger.error(f"Error when trying to list .npz files in {npz_dir}: {e}") # Thêm log lỗi này
            self.npz_files = [] # Đảm bảo self.npz_files được khởi tạo
        
        self.max_len = max_len


    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        try:
            with np.load(npz_path) as npz:
                target_latents = npz["target_latents"]
                attention_mask = npz["attention_mask"]
                #encoder_text_hidden_states = npz["encoder_text_hidden_states"]
                #text_attention_mask = npz["text_attention_mask"]
                encoder_text_hidden_states = npz["prompt_text_hidden_states"]
                text_attention_mask = npz["prompt_text_attention_mask"]                
                lyric_token_ids = npz["lyric_token_ids"]
                lyric_mask = npz["lyric_mask"]
                mert_ssl_hidden_states = npz["mert_ssl_hidden_states"]
                mhuburt_ssl_hidden_states = npz["mhuburt_ssl_hidden_states"]
                speaker_embds = npz.get("speaker_embds", np.zeros((512,), dtype=np.float32))

            # Kiểm tra NaN/Inf
            for name, arr in [
                ("target_latents", target_latents),
                ("mert_ssl_hidden_states", mert_ssl_hidden_states),
                ("mhuburt_ssl_hidden_states", mhuburt_ssl_hidden_states),
                ("encoder_text_hidden_states", encoder_text_hidden_states),
            ]:
                if np.isnan(arr).any() or np.isinf(arr).any():
                    logger.error(f"NaN/Inf detected in {name} for {npz_path}")
                    raise ValueError(f"Invalid data in {npz_path}")


            target_latents_tensor = torch.from_numpy(target_latents).float()
            attention_mask_tensor = torch.from_numpy(attention_mask).float()
            if attention_mask_tensor.sum() == 0:
                logger.warning(f"Attention mask all zeros for {npz_path}, setting to ones")
                attention_mask_tensor = torch.ones_like(attention_mask_tensor)
            encoder_text_hidden_states_tensor = torch.from_numpy(encoder_text_hidden_states).float()
            text_attention_mask_tensor = torch.from_numpy(text_attention_mask).float()
            lyric_token_ids_tensor = torch.from_numpy(lyric_token_ids).long()
            lyric_mask_tensor = torch.from_numpy(lyric_mask).float()
            mert_ssl_tensor = torch.from_numpy(mert_ssl_hidden_states).float()
            mhuburt_ssl_tensor = torch.from_numpy(mhuburt_ssl_hidden_states).float()
            speaker_embds_tensor = torch.from_numpy(speaker_embds).float()

            if self.max_len is not None:
                target_latents_tensor = target_latents_tensor[:, :, :self.max_len]
                attention_mask_tensor = attention_mask_tensor[:self.max_len]
                if attention_mask_tensor.sum() == 0:
                    logger.warning(f"Attention mask all zeros after max_len for {npz_path}, setting to ones")
                    attention_mask_tensor = torch.ones_like(attention_mask_tensor)

            return {
                "keys": npz_path,
                "target_latents": target_latents_tensor,
                "attention_mask": attention_mask_tensor,
                "encoder_text_hidden_states": encoder_text_hidden_states_tensor,
                "text_attention_mask": text_attention_mask_tensor,
                "lyric_token_ids": lyric_token_ids_tensor,
                "lyric_mask": lyric_mask_tensor,
                "mert_ssl_hidden_states": mert_ssl_tensor,
                "mhuburt_ssl_hidden_states": mhuburt_ssl_tensor,
                "speaker_embds": speaker_embds_tensor,
            }
        except Exception as e:
            logger.error(f"Error loading {npz_path}: {str(e)}")
            raise


    def pad_and_stack(self, tensor_list, max_len=None, pad_value=0):
        if not tensor_list:
            return torch.tensor([])
        
        # Kiểm tra số chiều tensor
        dims = {t.dim() for t in tensor_list}
        if len(dims) > 1:
            raise ValueError(f"Inconsistent tensor dimensions: {dims}")
        
        # Xử lý tensor 1D (attention_mask, text_attention_mask)
        if all(t.dim() == 1 for t in tensor_list):
            if max_len is None:
                max_len = max(t.size(0) for t in tensor_list)
            padded = []
            for t in tensor_list:
                if t.size(0) < max_len:
                    pad = torch.full((max_len - t.size(0),), 
                                   pad_value, 
                                   dtype=t.dtype, 
                                   device=t.device)
                    padded.append(torch.cat([t, pad]))
                else:
                    padded.append(t[:max_len])
            return torch.stack(padded)
        
        # Xử lý tensor 2D (encoder_text_hidden_states)
        elif all(t.dim() == 2 for t in tensor_list):
            if max_len is None:
                max_len = max(t.size(0) for t in tensor_list)
            padded = []
            for t in tensor_list:
                if t.size(0) < max_len:
                    pad = torch.full((max_len - t.size(0), t.size(1)), 
                                   pad_value, 
                                   dtype=t.dtype, 
                                   device=t.device)
                    padded.append(torch.cat([t, pad], dim=0))
                else:
                    padded.append(t[:max_len])
            return torch.stack(padded)
        
        # Xử lý tensor 3D (target_latents)
        elif all(t.dim() == 3 for t in tensor_list):
            if max_len is None:
                max_len = max(t.size(-1) for t in tensor_list)
            padded = []
            for t in tensor_list:
                if t.size(-1) < max_len:
                    pad = torch.full(t.size()[:-1] + (max_len - t.size(-1),), 
                                   pad_value, 
                                   dtype=t.dtype, 
                                   device=t.device)
                    padded.append(torch.cat([t, pad], dim=-1))
                else:
                    padded.append(t[..., :max_len])
            return torch.stack(padded)
        
        else:
            raise ValueError(f"Unsupported tensor dimensions: {dims}")


    def collate_fn(self, batch):
        # Xác định max_len cho từng loại tensor
        max_len_latents = max(item["target_latents"].shape[-1] for item in batch)
        max_len_text = max(item["encoder_text_hidden_states"].shape[0] for item in batch)
        max_len_attention = max(item["attention_mask"].shape[0] for item in batch)
        max_len_lyric = max(item["lyric_token_ids"].shape[0] for item in batch)
        max_len_ssl = max(item["mert_ssl_hidden_states"].shape[0] for item in batch)


        keys = [item["keys"] for item in batch]
        
        # Pad và stack các tensor
        target_latents = self.pad_and_stack(
            [item["target_latents"] for item in batch],
            max_len=max_len_latents
        )
        
        attention_mask = self.pad_and_stack(
            [item["attention_mask"] for item in batch],
            max_len=max_len_attention
        )
        
        encoder_text_hidden_states = self.pad_and_stack(
            [item["encoder_text_hidden_states"] for item in batch],
            max_len=max_len_text
        )
        
        text_attention_mask = self.pad_and_stack(
            [item["text_attention_mask"] for item in batch],
            max_len=max_len_text
        )
        
        lyric_token_ids = self.pad_and_stack(
            [item["lyric_token_ids"] for item in batch],
            max_len=max_len_lyric
        )
        
        lyric_mask = self.pad_and_stack(
            [item["lyric_mask"] for item in batch],
            max_len=max_len_lyric
        )

        # Pad và stack SSL hidden states
        mert_ssl = self.pad_and_stack(
            [item["mert_ssl_hidden_states"] for item in batch],
            max_len=max_len_ssl
        )
        
        mhuburt_ssl = self.pad_and_stack(
            [item["mhuburt_ssl_hidden_states"] for item in batch],
            max_len=max_len_ssl
        )
            
        # Stack các tensor không cần pad
        speaker_embds = torch.stack([item["speaker_embds"] for item in batch])

        return {
            "keys": keys,
            "target_latents": target_latents,
            "attention_mask": attention_mask,
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "lyric_token_ids": lyric_token_ids,
            "lyric_mask": lyric_mask,
            "speaker_embds": speaker_embds,
            "mert_ssl_hidden_states": mert_ssl,
            "mhuburt_ssl_hidden_states": mhuburt_ssl,
        }


