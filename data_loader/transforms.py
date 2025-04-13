from torchvision import transforms

def get_transforms(dataset_name, resize=224):
    # Training transforms for iWildCam with RandAugment
    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    
    # Validation and test transforms
    val_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform  # Usually same as validation transforms
    }
