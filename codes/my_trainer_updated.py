from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from preprocessed_dataset import PreprocessedText2MusicDataset
from loguru import logger
from diffusers.utils.torch_utils import randn_tensor
import os
import json
import math
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel
from peft import LoraConfig
import sys
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
import wandb


from my_codes.my_gpu import print_memory_usage

# Giữ log level ERROR, thêm log quan trọng và debug
logger.remove()
logger.add(sys.stderr, level="ERROR")
logger.add(sys.stderr, level="DEBUG")

torch.set_float32_matmul_precision("high")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

###########################################################################################

class SaveLoRAAdapterCallback(Callback):
    def __init__(self, save_dir, adapter_name="default", every_n_steps=10):
        super().__init__()
        self.save_dir = save_dir
        self.adapter_name = adapter_name
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step > 0 and trainer.is_global_zero:
            save_path = os.path.join(self.save_dir, f"step={trainer.global_step}_lora_adapter")
            os.makedirs(save_path, exist_ok=True)
            pl_module.transformers.save_lora_adapter(save_path, adapter_name=self.adapter_name)
            pl_module.transformers.peft_config[self.adapter_name].save_pretrained(save_path)

            #logger.info(f"[LoRA] ✅ Saved adapter at step={trainer.global_step} → {save_path}")
            #logger.info(f"💾 Checkpoint saved at step {trainer.global_step} to {save_path}")
            #print(f"[LoRA] 💾 Saved adapter at step={trainer.global_step} → {save_path}")
            tqdm.write(f"[LoRA] 💾 Saved adapter at step={trainer.global_step} → {save_path}")



###########################################################################################

class Pipeline(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 2,  # Giảm mặc định
        batch_size: int = 2,
        train: bool = True,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0,
        checkpoint_dir=None,
        max_steps: int = 200000,
        warmup_steps: int = 100,
        dataset_path: str = "./data/preprocessed_dataset",
        val_dataset_path: str = "./data/val_preprocessed",
        lora_config_path: str = None,
        adapter_name: str = "lora_adapter",
        use_gradient_checkpointing: bool = True,
        max_len: int = None,
        accumulate_grad_batches: int = 1,
        log_train_loss_every_n_steps: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()        
        self.is_train = train
        self.T = T
        self.ssl_coeff = ssl_coeff
        self.automatic_optimization = False
        self.validation_step_outputs = []
        self.train_loss_buffer = []
        self.log_train_loss_every_n_steps = log_train_loss_every_n_steps


        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

        logger.info(f"🔄 Loading base ACE Step Transformer ~ 2 minutes")
        self.transformers = ACEStepTransformer2DModel.from_pretrained(
            os.path.join(checkpoint_dir, "ace_step_transformer"),
            torch_dtype=torch.float32
        ).to("cuda")
        if self.hparams.use_gradient_checkpointing:
            self.transformers.enable_gradient_checkpointing()
            logger.debug(f"🚩 Gradient checkpointing enabled: {self.transformers.gradient_checkpointing}")

        logger.info(f"🔄 Loading LoRA ACE Step Transformer")
        assert lora_config_path is not None, "Please provide a LoRA config path"
        with open(lora_config_path, "r", encoding="utf-8") as f:
            lora_config = json.load(f)
        lora_config = LoraConfig(**lora_config)
        self.transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
        self.transformers.set_adapter(adapter_name)
        self.adapter_name = adapter_name

        # Debug LoRA setup
        logger.info(f"LoRA adapter name: {adapter_name}")
        logger.info(f"Available adapters: {self.transformers.peft_config.keys()}")

        # Kích hoạt gradient cho các tham số LoRA
        for name, param in self.transformers.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                #logger.debug(f"Enabled grad for: {name}")
        
        print_memory_usage()
        print('-'*100)

        if self.is_train:
            self.transformers.train()

    def preprocess(self, batch, train=True):
        keys = batch["keys"]
        target_latents = batch["target_latents"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        encoder_text_hidden_states = batch["encoder_text_hidden_states"].to("cuda")
        text_attention_mask = batch["text_attention_mask"].to("cuda")        
        speaker_embds = batch.get("speaker_embds", torch.zeros(batch["target_latents"].shape[0], 512, dtype=torch.float32, device=target_latents.device)).to("cuda")
        lyric_token_ids = batch["lyric_token_ids"].to("cuda")
        lyric_mask = batch["lyric_mask"].to("cuda")
        mert_ssl_hidden_states = batch["mert_ssl_hidden_states"].to("cuda")
        mhuburt_ssl_hidden_states = batch["mhuburt_ssl_hidden_states"].to("cuda")

        if train:
            bsz = target_latents.shape[0]
            device = target_latents.device
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bsz,), device=device) < 0.15),
                torch.zeros(size=(bsz,), device=device),
                torch.ones(size=(bsz,), device=device),
            ).long()
            encoder_text_hidden_states = torch.where(
                full_cfg_condition_mask.unsqueeze(1).unsqueeze(1).bool(),
                encoder_text_hidden_states.clone(),
                torch.zeros_like(encoder_text_hidden_states),
            ).clone()

            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bsz,), device=device) < 0.50),
                torch.zeros(size=(bsz,), device=device),
                torch.ones(size=(bsz,), device=device),
            ).long()
            speaker_embds = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                speaker_embds.clone(),
                torch.zeros_like(speaker_embds),
            ).clone()

            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bsz,), device=device) < 0.15),
                torch.zeros(size=(bsz,), device=device),
                torch.ones(size=(bsz,), device=device),
            ).long()
            lyric_token_ids = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_token_ids.clone(),
                torch.zeros_like(lyric_token_ids),
            ).clone()
            lyric_mask = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_mask.clone(),
                torch.zeros_like(lyric_mask),
            ).clone()

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhuburt_ssl_hidden_states,
        )
    
    ####################################################
    
    def train_dataloader(self):
        self.train_dataset = PreprocessedText2MusicDataset(
            npz_dir=self.hparams.dataset_path,
            device="cpu",
            max_len=self.hparams.max_len,
        )
        logger.debug(f"Dataset size: {len(self.train_dataset)} .npz files")
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.hparams.batch_size,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )


    def val_dataloader(self):
        self.val_dataset = PreprocessedText2MusicDataset(
            npz_dir=self.hparams.val_dataset_path,
            device="cpu",
            max_len=self.hparams.max_len,
        )
        logger.debug(f"Validation dataset size: {len(self.val_dataset)} .npz files")
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            persistent_workers=True,
        )


    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            logger.warning("No validation outputs collected")
            return
        
        try:
            # Lọc ra các loss hợp lệ
            valid_losses = [x['val/loss'] for x in self.validation_step_outputs 
                           if x['val/loss'] is not None and not torch.isnan(x['val/loss'])]
            
            if not valid_losses:
                logger.warning("All validation losses were invalid")
                return
                
            avg_val_loss = torch.mean(torch.stack(valid_losses))
            
            # Log lên progress bar và WandB
            self.log("val/avg_loss", avg_val_loss, 
                    prog_bar=True, 
                    logger=True,
                    sync_dist=True)
            
            logger.info(f"Validation completed - avg_loss: {avg_val_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error in validation metrics: {str(e)}")
        finally:
            self.validation_step_outputs.clear()



    ####################################################

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)
        return timesteps

    def run_step(self, batch, batch_idx, is_training=True):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhuburt_ssl_hidden_states,
        ) = self.preprocess(batch)

        # Kiểm tra NaN/Inf
        for name, tensor in [
            ("target_latents", target_latents),
            ("mert_ssl_hidden_states", mert_ssl_hidden_states),
            ("mhuburt_ssl_hidden_states", mhuburt_ssl_hidden_states),
            ("encoder_text_hidden_states", encoder_text_hidden_states),
        ]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.debug(f"NaN detected in {name}: {tensor}")
                return None

        # Chuẩn hóa
        target_latents = torch.clamp(target_latents, -10.0, 10.0)
        mert_ssl_hidden_states = torch.clamp(mert_ssl_hidden_states, -10.0, 10.0)
        mhuburt_ssl_hidden_states = torch.clamp(mhuburt_ssl_hidden_states, -10.0, 10.0)
        encoder_text_hidden_states = torch.clamp(encoder_text_hidden_states, -10.0, 10.0)

        target_image = target_latents
        device = target_image.device
        dtype = target_image.dtype
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = (sigmas * noise + (1.0 - sigmas) * target_image).clone()
        target = target_image.clone()

        if attention_mask.max().item() == 0:
            return None

        if target.max().item() == 0 and target.min().item() == 0:
            return None

        transformer_output = self.transformers(
            hidden_states=noisy_image,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device).to(dtype),
            ssl_hidden_states=[mert_ssl_hidden_states, mhuburt_ssl_hidden_states],
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        final_model_pred = (model_pred * (-sigmas) + noisy_image).clone()

        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )

        selected_model_pred = (final_model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        with torch.no_grad():
            mask_flat = mask.reshape(bsz, -1)
            mask_mean = mask_flat.mean(1)
        
        mse_per_pixel = F.mse_loss(selected_model_pred, selected_target.detach(), reduction="none")
        mse_per_sample = mse_per_pixel.mean(1)
        weighted_loss = mse_per_sample * mask_mean
        denoising_loss = weighted_loss.mean()

        if torch.isnan(denoising_loss) or torch.isinf(denoising_loss):
            return None

        # Tính total_proj_loss sớm
        proj_loss_values = []
        for k, v in proj_losses or []:
            if torch.isnan(v) or torch.isinf(v):
                continue
            proj_loss_values.append(v)
        total_proj_loss = torch.stack(proj_loss_values).mean() if proj_loss_values else torch.tensor(0.0, device=target_image.device, dtype=target_image.dtype)

        # Tính loss trước khi log
        loss = denoising_loss + total_proj_loss * self.ssl_coeff

        # Log và detach sau khi tính loss
        for k, v in proj_losses or []:
            if torch.isnan(v) or torch.isinf(v):
                continue
            self.log(f"train/{k}_loss", v.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            v_item = v.detach().item()
        self.log(f"train/denoising_loss", denoising_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        denoising_loss_item = denoising_loss.detach().item()
        self.log(f"train/total_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        total_loss_item = loss.detach().item()

        if is_training:
            # Backward thủ công
            optimizer = self.optimizers()
            self.manual_backward(loss)

            # Chỉ kiểm tra gradient sau khi hoàn thành tích lũy
            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
                for name, param in self.transformers.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.error(f"NaN gradient detected in {name}")
                    if 'lora' in name and param.grad is None and param.requires_grad:
                        logger.error(f"{name} has no grad after backward")

            # Gradient accumulation và clipping thủ công
            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.transformers.parameters() if p.requires_grad],
                    max_norm=0.5
                )
                optimizer.step()
                optimizer.zero_grad()

                scheduler = self.lr_schedulers()
                scheduler.step()
                #print(f"[LR STEP] step={self.global_step} | lr={scheduler.get_last_lr()[0]:.8f}")


        # Log learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "train/learning_rate",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        return loss


    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch, batch_idx, is_training=True)

        if loss is not None:
            self.train_loss_buffer.append(loss.detach())

            # Log trung bình mỗi N step
            if self.global_step % self.log_train_loss_every_n_steps == 0 and self.train_loss_buffer:
                avg_loss = torch.stack(self.train_loss_buffer).mean()
                self.log("train/avg_loss", avg_loss, prog_bar=True, logger=True, on_step=True)
                self.train_loss_buffer.clear()

        return loss



    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.run_step(batch, batch_idx, is_training=False)
        
        if loss is not None:            
            #return {"val/loss": loss}
            self.validation_step_outputs.append({"val/loss": loss.detach()})
        return None


    def on_save_checkpoint(self, checkpoint):
        return {}  # Không lưu state_dict

    ####################################################

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformers.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params=[{"params": trainable_params}],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )

        lr_lambda = get_custom_lr_lambda(
            warmup_steps=self.hparams.warmup_steps,
            max_steps=self.hparams.max_steps,   # max_steps là total_steps
            num_decay_intervals=getattr(self.hparams, "num_decay_intervals", 10),
            initial_lr=self.hparams.learning_rate,
            end_lr=getattr(self.hparams, "end_learning_rate", 1e-6),
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


###########################################################################################

def old_get_custom_lr_lambda(warmup_steps, max_steps, num_decay_intervals, initial_lr, end_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        decay_steps = max_steps - warmup_steps
        interval_steps = max(1, decay_steps // num_decay_intervals)
        step_in_decay = current_step - warmup_steps
        current_interval = min(step_in_decay // interval_steps, num_decay_intervals - 1)

        ratio = current_interval / (num_decay_intervals - 1)
        lr_scale = 1.0 - ratio * (1.0 - end_lr / initial_lr)

        return max(end_lr / initial_lr, lr_scale)
    
    
    print(f'-'*100)
    print('✅ get_custom_lr_lambda(warmup_steps, max_steps, num_decay_intervals, initial_lr, end_lr):')
    print(f'[✔] initial_lr: {initial_lr}')
    print(f'[✔] end_lr: {end_lr}')
    print(f'[✔] max_steps: {max_steps}')
    print(f'[✔] num_decay_intervals: {num_decay_intervals}')
    print(f'[✔] warmup_steps: {warmup_steps}')    
    print(f'-'*100)
    
    return lr_lambda

# Giảm LR liên tục, tuyến tính, Mềm mại, biểu đồ mượt hơn, Không cần num_decay_intervals 
def get_custom_lr_lambda(warmup_steps, max_steps, num_decay_intervals, initial_lr, end_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            scale = float(current_step) / float(max(1, warmup_steps))
            #logger.debug(f"Step {current_step}: Warmup, LR scale = {scale:.6f}")
            return scale

        decay_steps = max_steps - warmup_steps
        step_in_decay = current_step - warmup_steps
        # Tính ratio để đạt end_lr đúng tại max_steps
        ratio = min(step_in_decay / decay_steps, 1.0)
        lr_scale = 1.0 - ratio * (1.0 - end_lr / initial_lr)
        #logger.debug(f"Step {current_step}: Decay, ratio={ratio:.4f}, LR scale = {lr_scale:.6f}")
        return max(end_lr / initial_lr, lr_scale)

    print(f'-'*100)
    print('✅ get_custom_lr_lambda(warmup_steps, max_steps, num_decay_intervals, initial_lr, end_lr):')
    print(f'[✔] initial_lr: {initial_lr}')
    print(f'[✔] end_lr: {end_lr}')
    print(f'[✔] max_steps: {max_steps}')
    #print(f'[✔] num_decay_intervals: {num_decay_intervals}')  # Vẫn in nhưng không dùng
    print(f'[✔] warmup_steps: {warmup_steps}')
    print(f'-'*100)

    return lr_lambda

###########################################################################################

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=1)
        self._disable_validation = False
    
    def on_validation_start(self, trainer, pl_module):
        self._disable_validation = True
    
    # tắt log validation khỏi progress bar bằng cờ _disable_validation
    def on_validation_end(self, trainer, pl_module):
        self._disable_validation = False
        avg_val_loss = trainer.callback_metrics.get("val/avg_loss", torch.tensor(0.0))
        #print(f"\nValidation completed - avg_loss: {avg_val_loss:.4f}")

    def get_metrics(self, trainer, pl_module):
        if self._disable_validation:
            return {}
        
        items = super().get_metrics(trainer, pl_module)
        display_metrics = {
            "loss": items.get("train/total_loss", torch.tensor(0.0)),
            "g_step": trainer.global_step  # ✅ Thêm dòng này
        }
        return display_metrics

            
###########################################################################################

def main(args):

    # Load dataset tạm để tính số bước mỗi epoch
    temp_dataset = PreprocessedText2MusicDataset(
        npz_dir=args.dataset_path,
        device="cpu",
        max_len=args.max_len,
    )
    steps_per_epoch = math.ceil(len(temp_dataset) / args.batch_size)

    # Tính số bước dựa trên số epoch
    steps_from_epochs = args.epochs * steps_per_epoch if args.epochs > 0 else float("inf")

    # Nếu không truyền max_steps thì mặc định là rất lớn → ưu tiên số bước từ epochs
    if args.max_steps is None:
        args.max_steps = int(1e6) # default max_steps = 1000000

    # Tính số bước huấn luyện thực tế
    total_steps = min(args.max_steps, steps_from_epochs)

    # Kiểm tra hợp lệ: warmup phải nhỏ hơn total
    if total_steps <= args.warmup_steps:
        raise ValueError(f"[❌] warmup_steps={args.warmup_steps} must be smaller than total_steps={total_steps}")

    print('-'*100)
    print('✅ Find total_steps:')
    print(f"[✔] num_epochs = {args.epochs}")
    print(f"[✔] num_samples = {len(temp_dataset)}")
    print(f"[✔] batch_size = {args.batch_size}")
    print(f"[✔] steps_per_epoch = {steps_per_epoch}")
    print(f"[✔] total_steps = {total_steps} (min of steps_from_epochs={steps_from_epochs}, max_steps={args.max_steps})")
    print('-'*100)




    model = Pipeline(
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shift=args.shift,
        max_steps=total_steps,
        every_plot_step=args.every_plot_step,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        adapter_name=args.exp_name,
        lora_config_path=args.lora_config_path,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        max_len=args.max_len,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_train_loss_every_n_steps=args.log_train_loss_every_n_steps,
    )

    os.makedirs(args.checkpoint_save, exist_ok=True)

    # Khởi tạo WandB logger với cấu hình tốt hơn
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.exp_name,
        save_dir=args.logger_dir,
        config=args,
        log_model=False,  # Không tự động log model
        settings=wandb.Settings(
            console="off",  # Giảm log ra console
            disable_job_creation=True  # Tắt tính năng không cần thiết
        )
    )
    
    # Định nghĩa metrics cho WandB
    wandb_logger.experiment.define_metric("val/avg_loss", summary="last")
    wandb_logger.experiment.define_metric("train/learning_rate", summary="last")
    wandb_logger.experiment.define_metric("train/total_loss", summary="last")
    wandb_logger.experiment.define_metric("train/denoising_loss", summary="last")


    if args.max_steps is None:
        args.max_steps = -1
    
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        strategy="auto",
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        log_every_n_steps=10,
        logger=wandb_logger,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=False,  # Tắt checkpoint mặc định
        callbacks=[
            CustomProgressBar(),
            SaveLoRAAdapterCallback(
                save_dir=args.checkpoint_save,
                adapter_name=args.exp_name,
                every_n_steps=args.every_n_train_steps
            ),
        ],
    )
    
    trainer.fit(model, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for LR schedule")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--end_learning_rate", type=float, default=1e-6, help="Final learning rate after decay")
    parser.add_argument("--num_decay_intervals", type=int, default=10, help="Number of decay steps after warmup")    
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--every_n_train_steps", type=int, default=2000)
    parser.add_argument("--every_plot_step", type=int, default=2000)
    parser.add_argument("--log_train_loss_every_n_steps", type=int, default=10, help="Log train avg loss every N steps")
    parser.add_argument("--dataset_path", type=str, default="./data/train_preprocessed")
    parser.add_argument("--val_dataset_path", type=str, default="./data/val_preprocessed")    
    parser.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    parser.add_argument("--project_name", type=str, default="My ACE-Step project")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--logger_dir", type=str, default="./exps/logs/")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="/mnt/f/ACE/ACE_checkpoints_3.5B")
    parser.add_argument("--checkpoint_save", type=str, default="./exps/logs/")
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=0)
    parser.add_argument("--lora_config_path", type=str, default="config/zh_rap_lora_config.json")
    parser.add_argument("--use_gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--val_check_interval", type=int, default=1000, help="Run validation every N steps")    
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    args = parser.parse_args()
    main(args)