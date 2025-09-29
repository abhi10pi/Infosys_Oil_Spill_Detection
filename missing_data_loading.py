# Add this as a new cell in your notebook and run it before model training

def load_dataset(dataset_path, split, apply_augmentation=False):
    """Load complete dataset for a given split"""
    images_path = os.path.join(dataset_path, split, "images")
    masks_path = os.path.join(dataset_path, split, "masks")
    
    image_files = sorted([f for f in os.listdir(images_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Loading {len(image_files)} samples from {split} set...")
    
    images, masks = [], []
    
    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        img_path = os.path.join(images_path, img_file)
        mask_path = os.path.join(masks_path, mask_file)
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        image = load_image(img_path, IMAGE_TYPE)
        mask = load_mask(mask_path)
        
        if apply_augmentation:
            image, mask = geometric_augmentation(image, mask)
            image = photometric_augmentation(image, IMAGE_TYPE)
        
        images.append(image)
        masks.append(mask)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_files)} samples")
    
    return np.array(images), np.array(masks)

# Load the data
print("Loading training data...")
X_train, y_train = load_dataset(DATASET_PATH, "train", apply_augmentation=True)

print("\nLoading validation data...")
X_val, y_val = load_dataset(DATASET_PATH, "val", apply_augmentation=False)

print("\nLoading test data...")
X_test, y_test = load_dataset(DATASET_PATH, "test", apply_augmentation=False)

print(f"\nData shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")