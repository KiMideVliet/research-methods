import logging
from pathlib import Path
from config import OUTPUT_DIR
from preprocessing import DatasetPreprocessor
from utils import setup_logging


## Logger set up
logger = setup_logging()

def main():
    """Main preprocessing pipeline"""
    try:
        # Initialize preprocessor
        logger.info("Initializing data preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        # Process TrashNet dataset
        logger.info("Processing TrashNet dataset...")
        preprocessor.process_trashnet()
        preprocessor.create_cross_validation_folds('trashnet')
        preprocessor.log_dataset_stats('trashnet')
        preprocessor.save_dataset_metadata('trashnet')
        
        # Process TACO dataset
        logger.info("Processing TACO dataset...")
        preprocessor.process_taco()
        preprocessor.create_cross_validation_folds('taco')
        preprocessor.log_dataset_stats('taco')
        preprocessor.save_dataset_metadata('taco')
        
        logger.info(f"Preprocessing complete. Output saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()