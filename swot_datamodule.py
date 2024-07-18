"""Lightning Data Module."""

import torch
import torchvision
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import xarray as xr
from xrpatcher import XRDAPatcher
from typing import Any
from torch import Tensor
from torchvision.transforms.functional import resize

class SimulatedSWOTDataset(Dataset):

    def __init__(self, lr_data: xr.DataArray, hr_data: xr.DataArray, scale_factor: int, patcher_args: dict[str, Any]):
        """Initialize a new instance of SWOT Dataset.
        
        Args:
            lr_data: lower resolution dataset
            high_res_xr: higher resolution dataset
            factor: super resolution factor
        """
        super().__init__()
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.scale_factor = scale_factor

        self.patcher = XRDAPatcher(self.lr_data, **patcher_args)

    def __len__(self):
        """Return length of dataset."""
        return len(self.patcher)
    
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return item from dataset."""
        patch = self.patcher[idx].load().values

        # normalize data

        # return appropriate data with input and target variables
        return {"input": patch[self.input_variables], "target": patch[self.target_variables]}


    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get LR/HR sample.
        
        Args:
            idx (int): Index of sample
        """
        lr_patch = self.patcher[idx]

        lat_min = lr_patch.lat.min().values
        lat_max = lr_patch.lat.max().values
        lon_min = lr_patch.lon.min().values
        lon_max = lr_patch.lon.max().values
        
        hr_patch = self.hr_data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max), time=lr_patch.time.values[0])
       

        lr_patch = torch.from_numpy(lr_patch.load().values)
        hr_patch = torch.from_numpy(hr_patch.load().values).unsqueeze(0)

        # normalize data

        # replace nan values with 0
        lr_patch[torch.isnan(lr_patch)] = 0
        hr_patch[torch.isnan(hr_patch)] = 0

        if hr_patch.shape[-1] != lr_patch.shape[-1] * self.scale_factor:
            hr_patch = resize(hr_patch, lr_patch.shape[-1] * self.scale_factor, antialias=False)

        return {"lr": lr_patch, "hr": hr_patch}
    
class SWOTDatamodule(LightningDataModule):

    def __init__(self, lr_path: str, hr_path: str,  train_patcher_kw, val_patcher_kw, batch_size: int = 64, num_workers: int = 0) -> None:
        super().__init__()

        self.lr_path = lr_path
        self.hr_path = hr_path
        self.train_patcher_kw = train_patcher_kw
        self.val_patcher_kw = val_patcher_kw

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        """Setup data module."""
        lr_xr = xr.open_dataset(self.lr_path)["sossheig"]
        hr_xr = xr.open_dataset(self.hr_path)["sossheig"]

        # define sequential train val split over time dimension
        split = int(0.8 * len(lr_xr.time))
        train_lr = lr_xr.isel(time=slice(0, split))
        val_lr = lr_xr.isel(time=slice(split, None))

        train_hr = hr_xr.isel(time=slice(0, split))
        val_hr = hr_xr.isel(time=slice(split, None))

        # 0-1 min max normalization on training data
        self.train_min = train_lr.min()
        self.train_max = train_lr.max()
        
        train_lr = (train_lr - self.train_min) / (self.train_max - self.train_min)
        train_hr = (train_hr - self.train_min) / (self.train_max - self.train_min)

        val_lr = (val_lr - self.train_min) / (self.train_max - self.train_min)
        val_hr = (val_hr - self.train_min) / (self.train_max - self.train_min)

        # train dataset
        self.train_ds = SimulatedSWOTDataset(
            lr_data=train_lr, 
            hr_data=train_hr, 
            scale_factor=8, 
            patcher_args=self.train_patcher_kw
        )

        # val dataset
        self.val_ds = SimulatedSWOTDataset(
            lr_data=val_lr, 
            hr_data=val_hr, 
            scale_factor=8, 
            patcher_args=self.val_patcher_kw
        )

    def train_dataloader(self) -> Any:
        """Return training dataloader."""
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> Any:
        """Return validation dataloader."""
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)



# patcher_kw = dict(patches={'time': 1, 'lat': 32, 'lon': 32}, strides={'time': 1, 'lat': 8, 'lon': 8})
# ds = SimulatedSWOTDataset(
#     lr_data=xr.open_dataset("/mnt/SSD2/nils/datasets/swot_simulated/dc_ref_low_res_8.nc")["sossheig"], 
#     hr_data=xr.open_dataset("/mnt/SSD2/nils/datasets/swot_simulated/dc_ref.nc")["sossheig"], 
#     scale_factor=8, 
#     patcher_args=patcher_kw
# )

# sample = ds[0]

# dm = SWOTDatamodule(
#     lr_path="/mnt/SSD2/nils/datasets/swot_simulated/dc_ref_low_res_8.nc",
#     hr_path="/mnt/SSD2/nils/datasets/swot_simulated/dc_ref.nc",
#     train_patcher_kw=patcher_kw,
#     val_patcher_kw=patcher_kw,
#     test_patcher_kw=patcher_kw,
#     batch_size=64,
#     num_workers=0
# )

# dm.setup("fit")

# train_dl = dm.train_dataloader()

# batch = next(iter(train_dl))

# print(batch["lr"].shape)
# print(batch["hr"].shape)

# import pdb
# pdb.set_trace()
# print(0)

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(batch["lr"][0, 0])
# axs[1].imshow(batch["hr"][0, 0])
# plt.show()
# fig.savefig("sample.png")

# import pdb
# pdb.set_trace()
# print(0)
