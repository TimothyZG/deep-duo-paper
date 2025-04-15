import os
import sys
import torch
import argparse
from torchvision.transforms.functional import to_pil_image

# Ensure import works from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_models.transforms import get_transforms
from load_data.dataloaders import get_dataloaders

test_passed = True  # Global flag

def test_transforms_work_on_pil():
    global test_passed
    try:
        transforms_dict = get_transforms(resize=224)
        fake_img = torch.randint(0, 256, (3, 300, 300), dtype=torch.uint8)
        pil_img = to_pil_image(fake_img)

        for split, tf in transforms_dict.items():
            transformed = tf(pil_img)
            assert isinstance(transformed, torch.Tensor), f"{split} transform did not return a tensor"
            assert transformed.shape == (3, 224, 224), f"{split} transform returned wrong shape: {transformed.shape}"

        print("✅ test_transforms_work_on_pil passed")
    except Exception as e:
        print(f"❌ test_transforms_work_on_pil failed: {e}")
        test_passed = False


def test_caltech256_dataloaders(tmp_path="/tmp"):
    global test_passed
    try:
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
            subset = split_loader.dataset
            base_dataset = subset.dataset

            num_images = len(subset)
            num_classes = len(set(label for _, label in subset))
            print(f"📸 Caltech256 [{split}]: {num_images} images, {num_classes} classes")

        x, y = next(iter(dataloaders["train"]))
        assert x.shape == (8, 3, 224, 224)
        print("✅ test_caltech256_dataloaders passed")
    except Exception as e:
        print(f"❌ test_caltech256_dataloaders failed: {e}")
        test_passed = False


def test_iwildcam_dataloaders(tmp_path="/tmp"):
    global test_passed
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
            subset = split_loader.dataset
            base_subset = subset.base_dataset

            num_images = len(base_subset)
            num_classes = base_subset.dataset.n_classes
            print(f"📸 iWildCam [{split}]: {num_images} images, {num_classes} classes")

        x, y = next(iter(dataloaders["train"]))
        assert x.shape[1:] == (3, 224, 224)
        print("✅ test_iwildcam_dataloaders passed")
    except Exception as e:
        print(f"❌ test_iwildcam_dataloaders failed: {e}")
        test_passed = False


def main(caltech256_path, iwildcam_path):
    print("Running tests for dataloaders and transforms...\n")
    test_transforms_work_on_pil()
    test_caltech256_dataloaders(caltech256_path)
    test_iwildcam_dataloaders(iwildcam_path)

    if test_passed:
        print("\n✅ All tests finished.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caltech256_path", type=str, required=True, help="Path to caltech256")
    parser.add_argument("--iwildcam_path", type=str, required=True, help="Path to iwildcam dataset")
    args = parser.parse_args()
    main(args.caltech256_path, args.iwildcam_path)
