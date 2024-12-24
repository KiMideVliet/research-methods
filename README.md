# Assessing YOLOv8’s Capability in Multi-Class Household Waste Detection and Classification Under Real-World Conditions

This project implements a preprocessing pipeline for training YOLOv8 on household waste classification using the TrashNet and TACO datasets. The project aims to evaluate YOLOv8's capability in multi-class household waste detection and classification under real-world conditions.

## Project Structure
```
.
├── data/
│   ├── data_trashnet/     # TrashNet dataset
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── data_taco/         # TACO dataset
│       ├── train/
│       ├── valid/
│       └── test/
├── preprocessed_data/      # Output directory
│   ├── taco/
│   └── trashnet/
├── config.py              # Configuration settings
├── main.py               # Main execution script
├── preprocessing.py      # Dataset preprocessing
├── utils.py             # Utility functions
└── README.md            # This file
```

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Pandas
- scikit-learn
- tqdm
- matplotlib
- PyYAML

Install requirements:
```bash
pip install -r requirements.txt
```

## Datasets
The project uses two datasets:
1. **TrashNet**: Baseline dataset with 2,527 images in six waste categories
   - Categories: glass, paper, metal, plastic, cardboard, trash
   - Controlled conditions with uniform lighting and background

2. **TACO** (Trash Annotations in Context): 1,500 images with rich environmental context
   - 60 detailed categories mapped to TrashNet's six categories
   - Varied outdoor settings (streets, beaches, forests)

## Configuration
Key settings in `config.py`:
```python
IMG_SIZE = 640           # YOLOv8 optimal size
TRAIN_RATIO = 0.7       # Training set ratio
VAL_RATIO = 0.15        # Validation set ratio
TEST_RATIO = 0.15       # Test set ratio
CV_FOLDS = 3            # Number of cross-validation folds
AUGMENTATION_ENABLED = True
AUGMENTATION_FACTOR = 2  # Augmented versions per image
```

## Features
- Data preprocessing and standardization
- Dataset augmentation
  - Random rotations (±45 degrees)
  - Horizontal and vertical flips
  - Color jittering
- Cross-validation fold creation
- Class distribution balancing
- Dataset visualization tools
- Proper bounding box rotation handling
- Support for both Google Colab and local execution

## Usage

1. **Setup Datasets**
   ```bash
   # Create data directories
   mkdir -p data/data_trashnet data/data_taco
   
   # Place datasets in respective directories
   # TrashNet: data/data_trashnet/
   # TACO: data/data_taco/
   ```

2. **Run Preprocessing**
   ```bash
   python main.py
   ```
   This will:
   - Process both datasets
   - Apply augmentations
   - Create cross-validation folds
   - Generate statistics and visualizations

3. **Google Colab Usage**
   - Mount your Google Drive
   - Set up the directory structure
   - Run the preprocessing pipeline
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Output
The preprocessing pipeline generates:
- Processed images and labels
- Cross-validation folds
- Dataset statistics
- Class distribution reports
- Sample visualizations

## Data Augmentation
The pipeline includes:
- Rotation with proper bounding box adjustment
- Random flips (horizontal/vertical)
- Color jittering
- Maintains label accuracy during transformations

## Class Balance
- Monitors class distribution across splits
- Generates distribution reports
- Helps identify potential imbalances

## Authors
- Eva Koenders
- Jishnu Harinandansingh
- Michel Marien

## Acknowledgments
- TrashNet dataset creators
- TACO dataset team
- Ultralytics for YOLOv8
