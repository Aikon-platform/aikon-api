import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from typing import List
from .utils import AllTranspose
from ...shared.utils.logging import console


class FileListDataset(Dataset):
    def __init__(
        self,
        data_paths,
        transform=None,
        device="cpu",
        transpositions: List[AllTranspose] = [AllTranspose.NONE],
    ):
        self.device = device
        self.data_paths = data_paths
        self.rotations = transpositions

        self.tensor_transforms = self._get_tensor_transforms(transform) if transform else None
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_paths) * len(self.rotations)

    def __getitem__(self, idx):
        zeros = torch.zeros(3, 224, 224).to(self.device)

        try:
            img_path = self.data_paths[idx]
        except IndexError as e:
            console(
                f"[FileListDataset.__getitem__] Index out of bounds: {idx}",
                e=e, color="yellow",
            )
            return zeros

        try:
            idx, rot = divmod(idx, len(self.rotations))
            try:
                im = Image.open(img_path)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
            except UnidentifiedImageError as e:
                console(
                    f"[FileListDataset.__getitem__] Could not identify image {img_path}",
                    e=e, color="yellow",
                )
                return zeros

            rot = self.rotations[rot]
            if rot != AllTranspose.NONE:
                im = im.transpose(rot.value)

            img = self.to_tensor(im)
            if self.tensor_transforms is not None:
                img = self.tensor_transforms(img)

            return img.to(self.device)

        except Exception as e:
            console(
                f"[FileListDataset.__getitem__] Error processing image {img_path}",
                e=e, color="yellow",
            )
            return zeros

    def get_image_paths(self):
        return self.data_paths

    @staticmethod
    def _get_tensor_transforms(transform):
        """Extract only the tensor-compatible transforms."""
        if not hasattr(transform, 'transforms'):
            return transform

        tensor_transforms = []
        for t in transform.transforms:
            if isinstance(t, (transforms.Normalize, transforms.Resize)):
                tensor_transforms.append(t)

        return transforms.Compose(tensor_transforms) if tensor_transforms else None
