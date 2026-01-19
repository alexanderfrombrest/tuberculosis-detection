import os
import shutil
import zipfile
import random
import cv2
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
import kagglehub

# --- CONFIGURATION ---
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PATH_NORMAL = os.path.join(DATA_PROCESSED_DIR, "normal")
PATH_TB = os.path.join(DATA_PROCESSED_DIR, "tb")

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
RANDOM_SEED = 42

def download_and_setup_data():
    """
    Downloads datasets if not present and formats them into the processed directory.
    """
    if os.path.exists(PATH_NORMAL) and os.path.exists(PATH_TB) and len(os.listdir(PATH_NORMAL)) > 0:
        return

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(PATH_NORMAL, exist_ok=True)
    os.makedirs(PATH_TB, exist_ok=True)

    # --- 1. DOWNLOAD KAGGLE DATA ---
    try:
        kaggle_path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")       
        kaggle_normal_src = os.path.join(kaggle_path, "TB_Chest_Radiography_Database", "Normal")
        kaggle_tb_src = os.path.join(kaggle_path, "TB_Chest_Radiography_Database", "Tuberculosis")
        
        process_images(kaggle_normal_src, PATH_NORMAL, prefix='kaggle')
        process_images(kaggle_tb_src, PATH_TB, prefix='kaggle')   
    except Exception as e:
        print(f"Error downloading Kaggle data: {e}")
        return

    # --- 2. DOWNLOAD MENDELEY DATA (Optional / Manual) ---
    mendeley_zip = os.path.join(DATA_RAW_DIR, "mendeley.zip")
    mendeley_extract_path = os.path.join(DATA_RAW_DIR, "mendeley_extracted")
    
    # URL might need updating as tokens expire
    mendeley_url = "https://data.mendeley.com/public-api/zip/8j2g3csprk/download/2"
    
    if not os.path.exists(mendeley_zip):
        os.system(f"wget -O {mendeley_zip} {mendeley_url}")

    if os.path.exists(mendeley_zip):
        with zipfile.ZipFile(mendeley_zip, 'r') as zip_ref:
            zip_ref.extractall(mendeley_extract_path)
            
        # The structure inside the zip is nested
        mendeley_tb_src = os.path.join(mendeley_extract_path, "Dataset of Tuberculosis Chest X-rays Images", "TB Chest X-rays")
        if os.path.exists(mendeley_tb_src):
            process_images(mendeley_tb_src, PATH_TB, prefix='mendeley')
    else:
        print("Mendeley zip not found.")

    # --- 3. BALANCE DATASET ---
    files_normal = os.listdir(PATH_NORMAL)
    files_tb = os.listdir(PATH_TB)
    
    count_normal = len(files_normal)
    count_tb = len(files_tb)
    
    print(f"Before balancing: Normal={count_normal}, TB={count_tb}")

    if count_normal > count_tb:
        diff = count_normal - count_tb
        random.seed(RANDOM_SEED)
        files_to_delete = random.sample(files_normal, diff)
        for f in files_to_delete:
            os.remove(os.path.join(PATH_NORMAL, f))
            
    print(f"Final counts: Normal={len(os.listdir(PATH_NORMAL))}, TB={len(os.listdir(PATH_TB))}")

def process_images(source_path, dest_path, prefix):
    """
    Reads images, resizes them, and saves them to the destination.
    """
    if not os.path.exists(source_path):
        return

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(source_path) if f.lower().endswith(valid_exts)]
    print(f"Processing {len(files)} images from {source_path}...")
    
    for file in tqdm(files):
        try:
            img = cv2.imread(os.path.join(source_path, file))
            if img is None: continue 
            
            img = cv2.resize(img, IMG_SIZE)
            
            # Save with unique name
            new_name = f"{prefix}_{file}"
            cv2.imwrite(os.path.join(dest_path, new_name), img)
        except Exception as e:
            print(f"Error processing {file}: {e}")

def build_model():
    """
    Constructs the MobileNetV2 with the custom 'Deep Head'.
    """
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    
    # Augmentation
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base Model
    x = base_model(x, training=False)

    # Custom Head (The "Deep" version)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model

def main():
    # 1. Prepare Data
    download_and_setup_data()
    
    # 2. Load Datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PROCESSED_DIR,
        validation_split=0.2,
        subset='training',
        seed=RANDOM_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PROCESSED_DIR,
        validation_split=0.2,
        subset='validation',
        seed=RANDOM_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Optimize
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 3. Build & Train
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "models/tb_model.keras",  # Save directly to models folder
            save_best_only=True, 
            monitor="val_accuracy",
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.25,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    print("Training Complete. Model saved to models/tb_model.keras")

if __name__ == "__main__":
    main()