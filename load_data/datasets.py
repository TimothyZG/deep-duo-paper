from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torch.utils.data import Dataset
from torchvision.datasets import Caltech256
from torch.utils.data import DataLoader
import torch

class BaseDataset(Dataset):
    def __init__(self, dataset_name, root_dir, download=True):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.download = download
        self.dataset = get_dataset(dataset=self.dataset_name, root_dir=self.root_dir, download=self.download)
        
    def get_splits(self):
        raise NotImplementedError("Subclasses should implement this method.")

class NoMetaDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.collate = getattr(base_dataset, "collate", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, *_ = self.base_dataset[idx]
        return x, y

    
class IWildCamDataset(BaseDataset):
    num_classes = 182
    def __init__(self, root_dir, download=True):
        super().__init__('iwildcam', root_dir, download)
        # self.num_classes = self.dataset.n_classes

    def get_splits(self, transforms, batch_size, num_workers):
        def wrap(subset, transform):
            orig = self.dataset.get_subset(subset, transform=transform)
            return NoMetaDataset(orig)
        return {
            "train": get_train_loader('standard', wrap('train', transforms['train']), batch_size=batch_size, num_workers=num_workers),
            "val": get_eval_loader('standard', wrap('id_val', transforms['val']), batch_size=batch_size, num_workers=num_workers),
            "test": get_eval_loader('standard', wrap('id_test', transforms['test']), batch_size=batch_size, num_workers=num_workers),
            "ood_test": get_eval_loader('standard', wrap('test', transforms['test']), batch_size=batch_size, num_workers=num_workers),
        }

class Caltech256Dataset(Dataset):
    num_classes = 257
    def __init__(self, root_dir, download=False):
        self.root_dir = root_dir
        self.download = download
        # self.num_classes=257

    def get_splits(self, transforms, batch_size, num_workers,train_perc = 0.7,val_perc = 0.15):
        full_dataset = Caltech256(root=self.root_dir, transform=None, download=self.download)
        train_size = int(train_perc * len(full_dataset))
        val_size = int(val_perc * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        splits = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=generator
        )
        train_dataset, val_dataset, test_dataset = splits

        # Wrap subsets with custom transform behavior
        train_dataset.dataset.transform = transforms["train"]
        val_dataset.dataset.transform = transforms["val"]
        test_dataset.dataset.transform = transforms["test"]

        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False),
            "ood_test": None,
        }