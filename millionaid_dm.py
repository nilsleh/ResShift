from torchgeo.datasets import MillionAID
from torchgeo.transforms import AugmentationSequential
import kornia.augmentation as K
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Any
from glob import glob
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor



class MyMillionAid(MillionAID):
    resize = K.Resize((224, 224))
    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        # normalization and resizing
        img = sample["image"] / 255.
        img = self.resize(img).squeeze(0).contiguous()

        return {"input": img, "target": sample["label"]}
    

class MillionAidTest(VisionDataset):
    resize = K.Resize((224, 224))
    def __init__(self, root: str):
        super().__init__(root)

        self.fpaths = sorted(glob(root + '/*.jpg', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')


        img = pil_to_tensor(img) / 255.
        img = self.resize(img).squeeze(0)

        return {"input": img}


class MillionAIDDataModule(LightningDataModule):

    """DataModule for MillionAID dataset."""
    def __init__(self, root: str, split="test", batch_size: int = 64, num_workers: int = 4):
        super().__init__()

        self.root = root
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.aug = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input"],
        )

    def setup(self, stage: str | None = None):
        if self.split == "test":
            self.train = MillionAidTest(root=self.root)
        else:
            self.train = MyMillionAid(root=self.root)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def on_after_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:
        """Apply augmentations to the batch."""
        return self.aug(batch)
    

# import torch
# dm = MillionAIDDataModule(root="/mnt/SSD2/nils/ocean_bench_exps/diffusion/data/million/test/test", batch_size=16)
# dm.setup("fit")
# batch = next(iter(dm.train_dataloader()))

# torch.save(batch, "millionaid_batch.pth")
# import pdb
# pdb.set_trace()

# print(0)