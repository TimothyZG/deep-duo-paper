# multi_transform_dataloader.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MultiTransformDataset(Dataset):
    """
    Dataset wrapper that applies different transforms to the same data
    """
    def __init__(self, base_dataset, transforms_dict):
        """
        Args:
            base_dataset: Original dataset
            transforms_dict: Dict with keys like 'teacher', 'student' and transform values
        """
        self.base_dataset = base_dataset
        self.transforms_dict = transforms_dict
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original data (assuming it returns (image, label))
        image, label = self.base_dataset[idx]
        
        result = {'labels': label}
        
        # Apply different transforms
        for key, transform in self.transforms_dict.items():
            # print(f"{key=}, {type(transform)=}, {transform=}")
            if transform is not None:
                result[f'{key}_inputs'] = transform(image)
            else:
                result[f'{key}_inputs'] = image
        
        return result