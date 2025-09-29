from pathlib import Path
from PIL import Image

def sanity_check(root="Dataset"):
    for split in ["train", "val", "test"]:
        img_dir = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"

        imgs = sorted(img_dir.glob("*"))
        masks = sorted(mask_dir.glob("*"))

        print(f"\n{split.upper()} -> {len(imgs)} images, {len(masks)} masks")

        for img, mask in zip(imgs, masks):
            if img.stem != mask.stem:
                print(f"⚠️ Name mismatch: {img.name} vs {mask.name}")
            if Image.open(img).size != Image.open(mask).size:
                print(f"⚠️ Size mismatch: {img.name}")

sanity_check("Dataset")



def check_data_quality(images, masks, dataset_name):
    """Perform quality checks on the dataset"""
    print(f"\n=== {dataset_name} Data Quality Check ===")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
    
    oil_percentage = (np.sum(masks == 1) / masks.size) * 100
    print(f"Oil spill pixels: {oil_percentage:.2f}% of total pixels")

check_data_quality(X_train, y_train, "Training")
check_data_quality(X_val, y_val, "Validation")
check_data_quality(X_test, y_test, "Test")