# preprocessing.py
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Tuple
from utils import (
    setup_logging, create_cross_validation_folds, log_dataset_stats,
    save_dataset_metadata, setup_directories, process_split_files,
    create_yaml_file, augment_image
)
import cv2
import numpy as np
from config import *

logger = setup_logging()

class DatasetPreprocessor:
    def __init__(self, save_examples: int = 5):
        """Initialize the dataset preprocessor
        
        Args:
            save_examples: Number of example images to save for visualization
        """
        self.processed_dir = OUTPUT_DIR
        self.taco_dir = self.processed_dir / "taco"
        self.trashnet_dir = self.processed_dir / "trashnet"
        self.example_dir = self.processed_dir / "examples"
        self.example_dir.mkdir(parents=True, exist_ok=True)
        self.save_examples = save_examples

        # Set up directories for both datasets
        self.taco_train_dir, self.taco_val_dir, self.taco_test_dir = setup_directories(self.taco_dir)
        self.trashnet_train_dir, self.trashnet_val_dir, self.trashnet_test_dir = setup_directories(self.trashnet_dir)

    def process_dataset(self, dataset_type: str) -> bool:
        """Check if dataset is already processed
        
        Args:
            dataset_type: Either 'taco' or 'trashnet'
            
        Returns:
            bool: True if dataset is already processed, False otherwise
        """
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        logger.info(f"Checking if {dataset_type} dataset is already processed in {dataset_dir}")
        
        if dataset_dir.exists():
            train_img_dir = dataset_dir / "train" / "images"
            train_lbl_dir = dataset_dir / "train" / "labels"
            val_img_dir = dataset_dir / "val" / "images"
            val_lbl_dir = dataset_dir / "val" / "labels"
            test_img_dir = dataset_dir / "test" / "images"
            test_lbl_dir = dataset_dir / "test" / "labels"
            yaml_file = dataset_dir / "dataset.yaml"
            
            if (train_img_dir.exists() and train_lbl_dir.exists() and 
                val_img_dir.exists() and val_lbl_dir.exists() and
                test_img_dir.exists() and test_lbl_dir.exists() and
                yaml_file.exists() and 
                len(list(train_img_dir.glob('*.jpg'))) > 0):
                
                logger.info(f"{dataset_type} dataset already processed in {dataset_dir}")
                return True
                
        logger.info(f"{dataset_type} dataset needs processing")
        return False

    
    def augment_data(self, image: np.ndarray, label: str) -> List[Tuple[np.ndarray, str]]:
        """Generate augmented versions of an image and its label"""
        augmented_pairs = []
        for _ in range(AUGMENTATION_FACTOR):
            aug_img, aug_label = augment_image_and_label(image, label)
            augmented_pairs.append((aug_img, aug_label))
        return augmented_pairs  

    
    

    def process_image_batch(self, image_paths: List[Path], labels: List[str], 
                            split_dir: Path, is_training: bool = False) -> None:
        """Process a batch of images with optional augmentation for training only
        
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            split_dir: Directory to save processed images
            is_training: Whether this is training data (for augmentation)
        """
        processed_images = []
        processed_labels = []
        
        for img_path, label in zip(image_paths, labels):
            # Read and convert image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Add original image
            processed_images.append((img_path.stem, img))  # Use stem as identifier
            processed_labels.append(label)
            
            # Add augmented versions for training set
            if is_training and AUGMENTATION_ENABLED:
                try:
                    augmented_imgs = self.augment_data(img)
                    for idx, aug_img in enumerate(augmented_imgs):
                        processed_images.append((f"{img_path.stem}_aug_{idx}", aug_img))
                        processed_labels.append(label)
                except Exception as e:
                    logger.error(f"Error augmenting {img_path}: {e}")
                    continue
        
        # Process and save all images
        process_split_files(processed_images, processed_labels, split_dir, 
                        IMG_SIZE, COLOR_MEAN, COLOR_STD)

    def process_trashnet(self):
        """Process TrashNet dataset with augmentation"""
        logger.info("Processing TrashNet dataset...")

        if self.process_dataset('trashnet'):
            return

        image_paths, labels = [], []

        # Collect image paths and labels
        for category in TRASHNET_CLASSES:
            category_dir = TRASHNET_DIR / category
            if not category_dir.exists():
                logger.warning(f"Directory not found: {category_dir}")
                continue

            for img_path in category_dir.glob('*.jpg'):
                class_id = TRASHNET_CLASSES.index(category)
                label = f"{class_id} 0.5 0.5 1.0 1.0\n"
                image_paths.append(img_path)
                labels.append(label)

        # Split data
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, shuffle=True
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_STATE, shuffle=True
        )

        # Process each split
        logger.info("Processing training split with augmentation...")
        self.process_image_batch(train_paths, train_labels, self.trashnet_train_dir, is_training=True)
        
        logger.info("Processing validation split...")
        self.process_image_batch(val_paths, val_labels, self.trashnet_val_dir, is_training=False)
        
        logger.info("Processing test split...")
        self.process_image_batch(test_paths, test_labels, self.trashnet_test_dir, is_training=False)

        # Save YAML
        yaml_data = {
            'path': str(self.trashnet_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(TRASHNET_CLASSES),
            'names': TRASHNET_CLASSES
        }
        create_yaml_file(self.trashnet_dir / 'dataset.yaml', yaml_data)

    def process_taco(self):
        """Process TACO dataset with augmentation"""
        logger.info("Processing TACO dataset...")

        if self.process_dataset('taco'):
            return

        split_mapping = {'train': 'train', 'valid': 'val', 'test': 'test'}

        for input_split, output_split in split_mapping.items():
            logger.info(f"Processing TACO {input_split} split...")
            split_dir = TACO_DIR / input_split
            if not split_dir.exists():
                logger.warning(f"Directory not found: {split_dir}")
                continue

            output_dir = getattr(self, f'taco_{output_split}_dir')
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'

            logger.info(f"Reading images from {images_dir}")
            image_paths, labels = [], []

            for img_path in images_dir.glob('*.jpg'):
                logger.debug(f"Processing image: {img_path.name}")
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    logger.warning(f"Label file missing for image: {img_path.name}")
                    continue

                try:
                    with open(label_path, "r") as f:
                        label_content = f.read()
                        labels.append(label_content)
                        image_paths.append(img_path)
                except Exception as e:
                    logger.error(f"Error reading label file {label_path}: {e}")
                    continue

            logger.info(f"Found {len(image_paths)} images in {input_split} split")

            # Process with augmentation only for training split
            is_training = (input_split == 'train')
            logger.info(f"Starting batch processing for {input_split} split {'with' if is_training else 'without'} augmentation")
            self.process_image_batch(image_paths, labels, output_dir, is_training=is_training)
            logger.info(f"Completed processing {input_split} split")

        # Save YAML
        logger.info("Creating YAML configuration file...")
        yaml_data = {
            'path': str(self.taco_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(TACO_CLASSES),
            'names': TACO_CLASSES
        }
        create_yaml_file(self.taco_dir / 'dataset.yaml', yaml_data)
        logger.info("TACO dataset processing complete")

    def create_cross_validation_folds(self, dataset_type: str):
        """Create k-fold cross-validation splits"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        create_cross_validation_folds(dataset_dir, dataset_type, CV_FOLDS)

    def log_dataset_stats(self, dataset_type: str):
        """Log dataset statistics"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        log_dataset_stats(dataset_dir, dataset_type)

    def save_dataset_metadata(self, dataset_type: str):
        """Save dataset statistics"""
        dataset_dir = getattr(self, f"{dataset_type}_dir")
        save_dataset_metadata(dataset_dir, dataset_type)