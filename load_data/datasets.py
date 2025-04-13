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

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, *_ = self.base_dataset[idx]
        return x, y
    
class IWildCamDataset(BaseDataset):
    def __init__(self, root_dir, download=True):
        super().__init__('iwildcam', root_dir, download)

    def get_splits(self, transforms, batch_size, num_workers):
        # Get the data subsets
        train_data = NoMetaDataset(self.dataset.get_subset('train', transform=transforms['train']))
        val_data = NoMetaDataset(self.dataset.get_subset('id_val', transform=transforms['val']))
        test_data = NoMetaDataset(self.dataset.get_subset('id_test', transform=transforms['test']))
        ood_test_data = NoMetaDataset(self.dataset.get_subset('test', transform=transforms['test']))
        # Create data loaders
        train_loader = get_train_loader('standard', train_data, batch_size=batch_size, num_workers=num_workers)
        val_loader = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=num_workers)
        test_loader = get_eval_loader('standard', test_data, batch_size=batch_size, num_workers=num_workers)
        ood_test_loader = get_eval_loader('standard', ood_test_data, batch_size=batch_size, num_workers=num_workers)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'ood_test': ood_test_loader
        }

class Caltech256Dataset(Dataset):
    def __init__(self, root_dir, download=False):
        self.root_dir = root_dir
        self.download = download

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