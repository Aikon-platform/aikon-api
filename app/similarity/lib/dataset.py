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
        self.transforms = transform

        self._tensor_transforms = None
        self._pil_transforms = None
        self._target_size = None
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_paths) * len(self.rotations)

    def __getitem__(self, idx):
        zeros = torch.zeros(3, self.target_size[0], self.target_size[1]).to(self.device)
        idx, rot = divmod(idx, len(self.rotations))

        try:
            img_path = self.data_paths[idx]
        except IndexError as e:
            console(
                f"[FileListDataset.__getitem__] Index out of bounds: {idx}",
                e=e, color="yellow",
            )
            return zeros

        try:
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

            im = transforms.Resize(self.target_size, antialias=True)(im)

            if self.pil_transforms:
                im = self.pil_transforms(im)

            img = self.to_tensor(im)
            if self.tensor_transforms:
                img = self.tensor_transforms(img)

            return img.to(self.device)

        except Exception as e:
            console(
                f"[FileListDataset.__getitem__] Error processing image {img_path}",
                e=e, color="yellow",
            )
            return zeros

    @property
    def target_size(self):
        default_size = (224, 224)
        if self._target_size is not None:
            return self._target_size

        self._target_size = default_size

        if not self.transforms or not hasattr(self.transforms, 'transforms'):
            return self._target_size

        for t in self.transforms.transforms:
            if isinstance(t, transforms.Resize):
                self._target_size = (t.size, t.size) if isinstance(t.size, int) else t.size
        return self._target_size

    @property
    def tensor_transforms(self):
        if self._tensor_transforms is None:
            self._split_transforms()
        return self._tensor_transforms

    @property
    def pil_transforms(self):
        if self._pil_transforms is None:
            self._split_transforms()
        return self._pil_transforms

    def _split_transforms(self) -> None:
        if self.transforms is None or not hasattr(self.transforms, 'transforms'):
            self._pil_transforms = transforms.Compose([])
            self._tensor_transforms = self.transforms or []
            return

        pil_transforms = []
        tensor_transforms = []

        for t in self.transforms.transforms:
            if isinstance(t, transforms.Normalize):
                tensor_transforms.append(t)
            elif not isinstance(t, transforms.ToTensor):
                pil_transforms.append(t)

        self._pil_transforms = transforms.Compose(pil_transforms)
        self._tensor_transforms = transforms.Compose(tensor_transforms)
