from .datasets import IWildCamDataset, Caltech256Dataset

def get_dataloaders(dataset_name, root_dir, batch_size, num_workers, transforms):
    if dataset_name.lower() == 'iwildcam':
        dataset = IWildCamDataset(root_dir=root_dir)
    elif dataset_name.lower() == 'caltech256':
        dataset = Caltech256Dataset(root_dir=root_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    data_loaders = dataset.get_splits(
        transforms=transforms,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return data_loaders

    