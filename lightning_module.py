"""Lightning Module for ResShift."""

from lightning import LightningModule
import torch
from torch import Tensor
from typing import Any
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
import torch.nn.functional as F

# TODOs:
# 1. Implement ResShiftLightning
# 2. Can we reproduce just the inference results with the different model parts? (YES)
# 3. Try another Autoencoder from diffusers, VAE Encoder and VQGAN Encoder
# 3. Add training and validation steps to the Lightning Module.


class ResShiftLightning(LightningModule):
    """Lightning Module for ResShift."""

    def __init__(
        self,
        base_diffusion: nn.Module,
        model: nn.Module,
        autoencoder: nn.Module | None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        """Initialize ResShiftLightning.

        Args:
            base_diffusion_model: Base model that has the diffusion logic implemented
            model: The Unet model doing the denoising
            autoencoder: Autoencoder model, recommended to use Stable Diffusion VAE
        """
        super().__init__()

        self.base_diffusion = base_diffusion
        self.model = model
        self.autoencoder = autoencoder

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        
        batch = torch.load("/mnt/SSD2/nils/ResShift/example_batch.pth", map_location=self.device)
        LR, HR = batch["LR"], batch["HR"]
        # LR = LR.to(self.device)
        # HR = HR.to(self.device)

        batch_size = LR.shape[0]

        tt = torch.randint(
            0,
            self.base_diffusion.num_timesteps,
            size=(batch_size,),
            device=LR.device,
        )  # shape microbatchsize

        noise_chn = self.autoencoder.embed_dim
        latent_downsamping_sf = 2**(len(self.autoencoder.ch_mult) - 1)
        latent_resolution = HR.shape[-1] // latent_downsamping_sf

        noise = torch.randn(
                size= (LR.shape[0], noise_chn,) + (latent_resolution, ) * 2,
                device=LR.device,
                ) # [micro B, noise_chn, latent_resolution, latent_resolution]
        model_kwargs = {"lq": LR}
        # with torch.no_grad():
        losses, _, _ = self.base_diffusion.training_losses(
            model=self.model,
            x_start=HR,
            y=LR,
            t=tt,
            first_stage_model=self.autoencoder,
            model_kwargs=model_kwargs,
            noise=noise,
        )

        # individual losses when using diffusers later on
        
        # if "diffusers" in self.autoencoder.__class__.__module__:
        #     z_y = self.autoencoder.encode(F.interpolate(LR, scale_factor=self.base_diffusion.sf, mode='bicubic')).latent_dist.sample()
        # else:
        #     z_y = self.autoencoder.encode(F.interpolate(LR, scale_factor=self.base_diffusion.sf, mode='bicubic'))
        # z_start = self.autoencoder.encode(HR)

        # z_t = self.base_diffusion.q_sample(z_start, z_y, tt, noise=noise)

        # model_output = self.model(z_t, tt, **model_kwargs)

        # my_loss = F.mse_loss(model_output, z_start)
        loss = losses["mse"].mean()

        self.log("train_loss", loss, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Pass"""
        pass

    def predict_step(self, y0: Tensor, mask: Tensor | None = None) -> Tensor:
        """Perform inference on the input tensor.

        Args:
            y0: Input tensor, of low quality image, normalized to [-1, 1] with
                shape [B, C, H, W]
            mask: Mask tensor, of the same shape as y0, with 1s for known pixels

        Returns:
            torch.Tensor: Output tensor.
        """
        if mask is not None:
            model_kwargs = {"lq": y0, "mask": mask}
        else:
            model_kwargs = {
                "lq": y0,
            }

        results = self.base_diffusion.p_sample_loop(
            y=y0,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=None,
            noise_repeat=False,
            clip_denoised=(self.autoencoder is None),
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
        )  # This has included the decoding for latent space

        # make a custom implementation to work with diffusers encoders/decoders

        # upsampling factor
        # upsample image to desired final resolution
        # TODO actually this is only a problem with the current model, can change the model
        # to accept another resolution when training
        # y0 = F.interpolate(y0, scale_factor=self.base_diffusion.sf, mode='bicubic')
        # # encode

        # if "diffusers" in self.autoencoder.__class__.__module__:
        #     z_y = self.autoencoder.encode(y0).latent_dist.sample()
        # else:
        #     z_y = self.autoencoder.encode(y0)

        # # add initial diffusion noise
        # z_sample = self.base_diffusion.prior_sample(z_y, torch.rand_like(z_y))

        # indices = tqdm(list(range(self.base_diffusion.num_timesteps))[::-1])

        # with torch.no_grad():
        #     for i in indices:
        #         t = torch.tensor([i] * y0.shape[0], device=y0.device)
        #         out = self.base_diffusion.p_sample(
        #             self.model,
        #             z_sample,
        #             z_y,
        #             t,
        #             clip_denoised=False,
        #             denoised_fn=None,
        #             model_kwargs=model_kwargs,
        #             noise_repeat=False,
        #         )
        #         z_sample = out["sample"]

        # # decode
        # with torch.no_grad():
        #     if "diffusers" in self.autoencoder.__class__.__module__:
        #         results = self.autoencoder.decode(z_sample).sample
        #     else:
        #         results = self.autoencoder.decode(z_sample)

        return results.clamp_(-1.0, 1.0) * 0.5 + 0.5

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        # we only train the model not the autoencoder
        optimizer = self.optimizer(self.model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "train_loss"},
            }
        else:
            return {"optimizer": optimizer}


def reload_model(model: nn.Module, ckpt: dict[str, Any]) -> None:
    """Reload model with checkpoint.

    Args:
        model: The model to reload the checkpoint into
        ckpt: The checkpoint to load into the model
    """
    module_flag = list(ckpt.keys())[0].startswith("module.")
    compile_flag = "_orig_mod" in list(ckpt.keys())[0]

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not "_orig_mod." in source_key):
            target_key = "_orig_mod." + target_key
        if module_flag and (not source_key.startswith("module")):
            target_key = "module." + target_key

        assert target_key in ckpt
        source_value.copy_(ckpt[target_key])


def resshift_with_checkpoint(
    base_diffusion,
    model,
    autoencoder,
    model_ckpt: str | None = None,
    autoencoder_ckpt: str | None = None,
) -> ResShiftLightning:
    """Load LightningModule with checkpoint."""
    # model checkpoint
    if model_ckpt is not None:
        state_dict = torch.load(model_ckpt)
        reload_model(model, state_dict)

    if autoencoder_ckpt is not None:
        state_dict = torch.load(autoencoder_ckpt)
        autoencoder.load_state_dict(state_dict)

    resshift = ResShiftLightning(
        base_diffusion=base_diffusion,
        model=model,
        autoencoder=autoencoder,
    )

    return resshift
