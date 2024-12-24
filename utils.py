# utils.py
import cv2
import numpy as np
import pandas as pd
import yaml
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from tqdm import tqdm
from config import *
import matplotlib.pyplot as plt

## Set up

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def setup_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Create necessary directories for processed data"""
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"
   
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
   
    return train_dir, val_dir, test_dir

## Standardization

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image maintaining aspect ratio with padding"""
    if image is None:
        raise ValueError("Image is None")
    
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError(f"Invalid target_size: {target_size}. Expected tuple of length 2")
        
    # Ensure image is 3D array with shape (height, width, channels)
    if len(image.shape) != 3:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 3 dimensions")
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Ensure array bounds are not exceeded
    if y_offset + new_h > target_h:
        new_h = target_h - y_offset
    if x_offset + new_w > target_w:
        new_w = target_w - x_offset
        
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized[:new_h, :new_w]
    
    return padded

def normalize_image(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """Normalize image using provided mean and std"""
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image

def create_yaml_file(path: Path, data: Dict) -> None:
    """Create YAML file with dataset configuration"""
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

## Augmentation

def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply data augmentation per research proposal"""
    angle = random.uniform(-45, 45)
    matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        
    image = cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-30, 30))
    
    return image

def save_split_data(split_dir: Path, images: List[Tuple[str, np.ndarray]], labels: List[str]) -> None:
    """Save split data to disk"""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
   
    for (img_stem, img), label in zip(images, labels):
        cv2.imwrite(
            str(images_dir / f"{img_stem}.jpg"),
            cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        
        with open(labels_dir / f"{img_stem}.txt", 'w') as f:
            f.write(label)

def count_images_per_class(labels_dir: Path) -> Dict[str, int]:
    """Count number of images per class"""
    class_counts = {}
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return class_counts

def save_metadata_csv(metadata: Dict[str, Dict[str, int]], output_path: Path, dataset_type: str) -> None:
    """Save metadata about splits to a CSV file
    
    Args:
        metadata: Dictionary containing split information
        output_path: Path to save the CSV
        dataset_type: Either 'taco' or 'trashnet'
    """
    df = pd.DataFrame.from_dict(metadata, orient='index').reset_index()
    
    # Set columns based on dataset type
    if dataset_type == 'trashnet':
        columns = ['Fold', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    else:  # taco
        columns = ['Fold'] + TACO_CLASSES
        
    # Ensure all columns exist
    for col in columns[1:]:  # Skip 'Fold' column
        if col not in df.columns:
            df[col] = 0
            
    # Reorder columns
    df = df.reindex(columns=columns)
    df.to_csv(output_path, index=False)

def rotate_box(x_center: float, y_center: float, width: float, height: float, 
               angle: float, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """Rotate a bounding box around image center
    
    Args:
        x_center, y_center: Box center coordinates (normalized 0-1)
        width, height: Box dimensions (normalized 0-1)
        angle: Rotation angle in degrees
        image_width, image_height: Image dimensions
        
    Returns:
        new_x, new_y, new_w, new_h: Rotated box parameters
    """
    # Convert to pixel coordinates
    x = x_center * image_width
    y = y_center * image_height
    w = width * image_width
    h = height * image_height
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate image center
    cx = image_width / 2
    cy = image_height / 2
    
    # Rotate box center around image center
    x_shifted = x - cx
    y_shifted = y - cy
    
    new_x = cx + (x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad))
    new_y = cy + (x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad))
    
    # Convert back to normalized coordinates
    new_x_norm = new_x / image_width
    new_y_norm = new_y / image_height
    new_w_norm = width  # Width/height don't change for rotation around center
    new_h_norm = height
    
    return new_x_norm, new_y_norm, new_w_norm, new_h_norm

def augment_image_and_label(image: np.ndarray, label: str) -> Tuple[np.ndarray, str]:
    """Apply augmentation to both image and its label
    
    Args:
        image: Input image
        label: YOLO format label string "class_id x_center y_center width height"
        
    Returns:
        augmented_image, augmented_label
    """
    # Parse label
    parts = label.strip().split()
    class_id = int(parts[0])
    x_center, y_center = float(parts[1]), float(parts[2])
    width, height = float(parts[3]), float(parts[4])
    
    # Random rotation
    angle = random.uniform(-45, 45)
    matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    
    # Rotate bounding box
    new_x, new_y, new_w, new_h = rotate_box(
        x_center, y_center, width, height,
        angle, image.shape[1], image.shape[0]
    )
    
    # Apply other augmentations to image only
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # horizontal
        new_x = 1 - new_x  # Flip x coordinate
        
    if random.random() > 0.5:
        image = cv2.flip(image, 0)  # vertical
        new_y = 1 - new_y  # Flip y coordinate
        
    # Color augmentation
    image = cv2.convertScaleAbs(image, 
                               alpha=random.uniform(0.8, 1.2),
                               beta=random.uniform(-30, 30))
    
    # Create new label string
    new_label = f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"
    
    return image, new_label


def process_split_files(files: List[Tuple[str, np.ndarray]], labels: List[str], 
                       split_dir: Path, img_size: Tuple[int, int], 
                       mean: List[float], std: List[float]) -> None:
    """Process and save image files without augmentation
    
    Args:
        files: List of tuples (stem, image_array)
        labels: List of label strings
        split_dir: Output directory
        img_size: Target image size (width, height)
        mean: Normalization mean values
        std: Normalization standard deviation values
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for (stem, img), label in zip(files, labels):
        try:
            output_img_path = images_dir / f"{stem}.jpg"
            output_label_path = labels_dir / f"{stem}.txt"
            
            # Process image
            if isinstance(img_size, tuple):
                target_size = img_size
            else:
                target_size = (img_size, img_size)
            
            # Only resize and convert color space for validation/test
            img = resize_image(img, target_size)
            
            # Save processed image
            cv2.imwrite(
                str(output_img_path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            
            # Save label
            with open(output_label_path, 'w') as f:
                f.write(label)
                
            logger.debug(f"Successfully processed {stem}")
        except Exception as e:
            logger.error(f"Error processing {stem}: {e}", exc_info=True)

## Cross validation 

def create_cross_validation_folds(dataset_dir: Path, dataset_type: str, cv_folds: int):
    """Create k-fold cross-validation splits"""
    logger.info(f"Creating cross-validation folds for {dataset_type} dataset...")
    
    cv_dir = dataset_dir / "cv_splits"
    
    if cv_dir.exists() and len(list(cv_dir.glob('fold_*/dataset.yaml'))) == cv_folds:
        logger.info(f"Cross-validation folds already exist in {cv_dir}")
        return
        
    cv_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    
    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        logger.error(f"Training directories not found in {dataset_dir}")
        return
        
    image_files = sorted(list(train_images_dir.glob('*.jpg')))
    
    if not image_files:
        logger.error(f"No training images found in {train_images_dir}")
        return
        
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}")
        
        fold_dir = cv_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]
        
        for split_name, files in [('train', train_files), ('val', val_files)]:
            split_dir = fold_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
            
            for img_file in tqdm(files, desc=f"Processing {split_name} split for fold {fold_idx}"):
                try:
                    label_file = train_labels_dir / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        logger.warning(f"Label file missing for {img_file.name}")
                        continue
                    
                    output_img_path = split_dir / "images" / img_file.name
                    output_label_path = split_dir / "labels" / f"{img_file.stem}.txt"
                    
                    img = cv2.imread(str(img_file))
                    if img is None:
                        logger.warning(f"Could not read image: {img_file}")
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = resize_image(img, (IMG_SIZE, IMG_SIZE))
                    
                    if split_name == 'train' and AUGMENTATION_ENABLED:
                        img = augment_image(img)
                    
                    cv2.imwrite(
                        str(output_img_path),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    )
                    
                    import shutil
                    shutil.copy(label_file, output_label_path)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
            
            metadata[f"fold_{fold_idx}_{split_name}"] = count_images_per_class(split_dir / "labels")
        
        yaml_data = {
            'path': str(fold_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(TRASHNET_CLASSES if dataset_type == 'trashnet' else TACO_CLASSES),
            'names': TRASHNET_CLASSES if dataset_type == 'trashnet' else TACO_CLASSES
        }
        create_yaml_file(fold_dir / "dataset.yaml", yaml_data)
    
    save_metadata_csv(metadata, cv_dir / "fold_distribution.csv", dataset_type)
    logger.info(f"Created {cv_folds} cross-validation folds in {cv_dir}")

def log_dataset_stats(dataset_dir: Path, dataset_type: str):
    """Log dataset statistics"""
    total_images = len(list((dataset_dir / "train" / "images").glob("*.jpg")))
    total_images += len(list((dataset_dir / "val" / "images").glob("*.jpg")))
    total_images += len(list((dataset_dir / "test" / "images").glob("*.jpg")))
    logger.info(f"{dataset_type} dataset processed: {total_images} total images")

def save_dataset_metadata(dataset_dir: Path, dataset_type: str):
    """Save dataset statistics for analysis"""
    metadata = {
        'total_images': 0,
        'images_per_class': {},
        'class_distribution': {}
    }
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        metadata[f'{split}_images'] = len(list((split_dir / 'images').glob('*.jpg')))
        metadata['total_images'] += metadata[f'{split}_images']
        
        class_counts = count_images_per_class(split_dir / 'labels')
        metadata[f'{split}_class_distribution'] = class_counts
    
    output_file = dataset_dir.parent / f"{dataset_type}_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

## Visualization 

def visualize_sample_images(dataset_dir: Path, classes: List[str], samples_per_class: int = 2, 
                          title: str = "Dataset Samples") -> None:
    """Visualize sample images from each class in the dataset
    
    Args:
        dataset_dir: Path to dataset directory
        classes: List of class names
        samples_per_class: Number of samples to show per class
        title: Title for the plot
    """
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(15, 3*n_classes))
    fig.suptitle(title, fontsize=16)

    for i, class_name in enumerate(classes):
        # Get all images for this class
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
            
        image_paths = list(class_dir.glob('*.jpg'))
        if not image_paths:
            continue
            
        # Randomly sample images
        selected_images = random.sample(image_paths, min(len(image_paths), samples_per_class))
        
        for j, img_path in enumerate(selected_images):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if samples_per_class == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
                
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{class_name}')

    plt.tight_layout()
    plt.show()

def visualize_augmentations(image: np.ndarray, n_augmentations: int = 3) -> None:
    """Visualize original image and its augmentations
    
    Args:
        image: Original image
        n_augmentations: Number of augmented versions to show
    """
    fig, axes = plt.subplots(1, n_augmentations + 1, figsize=(15, 3))
    
    # Show original
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmentations
    for i in range(n_augmentations):
        aug_img = augment_image(image)
        axes[i+1].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f'Augmented {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
        


# Model and evaluation functions (placeholder implementations)
def load_model():
    """Load trained model"""
    raise NotImplementedError

def save_model():
    """Save trained model"""
    raise NotImplementedError

def calculate_metrics():
    """Calculate model evaluation metrics"""
    raise NotImplementedError