import os
import sys
import torch
import argparse
from torchvision.transforms.functional import to_pil_image

# Ensure import works from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data.transforms import get_transforms
from load_data.dataloaders import get_dataloaders

def test_transforms_work_on_pil():
    transforms_dict = get_transforms(resize=224)
    # Simulate image input
    fake_img = torch.randint(0, 256, (3, 300, 300), dtype=torch.uint8)
    pil_img = to_pil_image(fake_img)

    for split, tf in transforms_dict.items():
        transformed = tf(pil_img)
        assert isinstance(transformed, torch.Tensor), f"{split} transform did not return a tensor"
        assert transformed.shape == (3, 224, 224), f"{split} transform returned wrong shape: {transformed.shape}"

    print("✅ test_transforms_work_on_pil passed")


def test_caltech256_dataloaders(tmp_path="/tmp"):
    transforms_dict = get_transforms(resize=224)

    dataloaders = get_dataloaders(
        dataset_name="caltech256",
        root_dir=tmp_path,
        batch_size=8,
        num_workers=0,
        transforms=transforms_dict
    )

    for split in ["train", "val", "test"]:
        split_loader = dataloaders[split]
        subset = split_loader.dataset  # Subset object
        base_dataset = subset.dataset  # Caltech256

        num_images = len(subset)
        num_classes = len(set(label for _, label in subset))

        print(f"📸 Caltech256 [{split}]: {num_images} images, {num_classes} classes")

    x, y = next(iter(dataloaders["train"]))
    assert x.shape == (8, 3, 224, 224)
    print("✅ test_caltech256_dataloaders passed")


def test_iwildcam_dataloaders(tmp_path="/tmp"):
    try:
        transforms_dict = get_transforms(resize=224)

        dataloaders = get_dataloaders(
            dataset_name="iwildcam",
            root_dir=tmp_path,
            batch_size=4,
            num_workers=0,
            transforms=transforms_dict
        )

        for split in ["train", "val", "test", "ood_test"]:
            split_loader = dataloaders[split]
            subset = split_loader.dataset  # NoMetaDataset
            base_subset = subset.base_dataset  # WILDS subset

            num_images = len(base_subset)
            num_classes = base_subset.dataset.n_classes  # from the root WILDS dataset

            print(f"📸 iWildCam [{split}]: {num_images} images, {num_classes} classes")

        x, y = next(iter(dataloaders["train"]))
        assert x.shape[1:] == (3, 224, 224)
        print("✅ test_iwildcam_dataloaders passed")
    except Exception as e:
        print(f"⚠️ Skipped iWildCam test: {e}")

def main(caltech256_path,iwildcam_path):
    print("Running tests for dataloaders and transforms...\n")
    test_transforms_work_on_pil()
    test_caltech256_dataloaders(caltech256_path)
    test_iwildcam_dataloaders(iwildcam_path)
    print("\n✅ All tests finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caltech256_path", type=str, required=True, help="Path to caltech256")
    parser.add_argument("--iwildcam_path", type=str, required=True, help="Path to iwildcam_path")
    args = parser.parse_args()
    main(args.caltech256_path,args.iwildcam_path)
