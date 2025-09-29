# =========================================================
# AI SpillGuard - Oil Spill Detection (U-Net Pipeline)
# =========================================================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Disable MKL oneDNN for stability

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import tensorflow as tf
from tensorflow.keras import layers, models
from glob import glob
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. Dataset Configuration
# -------------------------------
DATASET_PATH = "DataSet"
TRAIN_PATH   = os.path.join(DATASET_PATH, "train")
VAL_PATH     = os.path.join(DATASET_PATH, "val")
TEST_PATH    = os.path.join(DATASET_PATH, "test")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 128, 128, 3   # smaller = lighter
IMAGE_TYPE = 'RGB'   # or 'SAR'

print(f"Dataset Path: {DATASET_PATH}")
print(f"Image Type: {IMAGE_TYPE}")
print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")

# -------------------------------
# 2. SAR & RGB Preprocessing
# -------------------------------
def lee_filter(img, window_size=5):
    """Lee speckle filter for SAR images"""
    img = img.astype(np.float32)
    img_mean = uniform_filter(img, window_size)
    img_sqr_mean = uniform_filter(img**2, window_size)
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    weights = img_variance / (img_variance + overall_variance + 1e-10)
    return img_mean + weights * (img - img_mean)

def preprocess_sar(image):
    image = lee_filter(image, window_size=5)
    image = np.log(image + 1e-10)
    image = (image - image.mean()) / (image.std() + 1e-10)
    image = (image - image.min()) / (image.max() - image.min() + 1e-10)
    return image.astype(np.float32)

def preprocess_rgb(image):
    return image.astype(np.float32) / 255.0

# -------------------------------
# 3. Data Loader
# -------------------------------
def load_image(img_path, mask_path, image_type="RGB"):
    # --- Image ---
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image_type == "SAR":
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = preprocess_sar(img.astype(np.float32)/255.0)
        img = np.stack([img, img, img], axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = preprocess_rgb(img)

    # --- Mask ---
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return img, mask

def build_dataset(img_dir, mask_dir, batch_size=1, shuffle=True, image_type="RGB"):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif")
    img_files, mask_files = [], []
    for ext in exts:
        img_files.extend(sorted(glob(os.path.join(img_dir, ext))))
        mask_files.extend(sorted(glob(os.path.join(mask_dir, ext))))

    print(f"[DEBUG] {img_dir}: {len(img_files)} images, {len(mask_files)} masks")

    if len(img_files) == 0 or len(mask_files) == 0:
        raise ValueError(f"No images or masks found in {img_dir} or {mask_dir}")

    def _generator():
        for img_path, mask_path in zip(img_files, mask_files):
            img, mask = load_image(img_path, mask_path, image_type)
            yield img, mask

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([IMG_HEIGHT, IMG_WIDTH, 3], [IMG_HEIGHT, IMG_WIDTH, 1])
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 4. Lightweight U-Net Model
# -------------------------------
def simple_unet(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 128)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    bn = conv_block(p4, 256)
    bn = layers.Dropout(0.5)(bn)

    def up_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(filters, (2, 2), padding="same")(x)
        x = layers.concatenate([x, skip])
        x = conv_block(x, filters)
        return x

    u6 = up_block(bn, c4, 128)
    u7 = up_block(u6, c3, 64)
    u8 = up_block(u7, c2, 32)
    u9 = up_block(u8, c1, 16)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u9)
    return models.Model(inputs, outputs)

# -------------------------------
# 5. Loss & Metrics
# -------------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + (1 - dice_coefficient(y_true, y_pred))

# -------------------------------
# 6. Build Datasets
# -------------------------------
train_ds = build_dataset(TRAIN_PATH + "/images", TRAIN_PATH + "/masks", batch_size=1, shuffle=True, image_type=IMAGE_TYPE)
val_ds   = build_dataset(VAL_PATH + "/images", VAL_PATH + "/masks", batch_size=1, shuffle=False, image_type=IMAGE_TYPE)
test_ds  = build_dataset(TEST_PATH + "/images", TEST_PATH + "/masks", batch_size=1, shuffle=False, image_type=IMAGE_TYPE)

print("Train batches:", len(train_ds))
print("Val batches:", len(val_ds))
print("Test batches:", len(test_ds))

# -------------------------------
# 7. Compile & Train Model
# -------------------------------
model = simple_unet(input_size=(IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=bce_dice_loss,
              metrics=["accuracy", dice_coefficient, iou_score])

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

# -------------------------------
# 8. Evaluation
# -------------------------------
test_loss, test_acc, test_dice, test_iou = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}")
