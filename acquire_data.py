#!/usr/bin/env python3
"""
Data acquisition script for cat vs dog image classification.
Downloads images from the Kaggle dataset and prepares them for training.
"""

import os
import logging
import requests
import zipfile
import shutil
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
DATABASE_DIR = BASE_DIR / 'data' / 'database'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Database setup
DATABASE_PATH = DATABASE_DIR / 'images.db'
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Image(Base):
    """SQLAlchemy model for image metadata."""
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    label = Column(String, nullable=False)
    split = Column(String, nullable=False)  # 'train', 'validation', or 'test'
    width = Column(Integer)
    height = Column(Integer)
    size_kb = Column(Float)
    
    def __repr__(self):
        return f"<Image(filename='{self.filename}', label='{self.label}', split='{self.split}')>"


def download_dataset():
    """
    Download the cats vs dogs dataset.
    
    For this example, we'll use the Microsoft Cats vs Dogs dataset,
    but in a real scenario, you might use the Kaggle API.
    """
    logger.info("Downloading dataset...")
    
    # URL for the dataset (this is a placeholder - in a real scenario, use Kaggle API)
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    
    # Path to save the downloaded zip file
    zip_path = RAW_DATA_DIR / "cats_vs_dogs.zip"
    
    # Download the dataset
    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Dataset downloaded to {zip_path}")
        
        # Extract the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        
        logger.info(f"Dataset extracted to {RAW_DATA_DIR}")
        
        # Remove the zip file to save space
        os.remove(zip_path)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def process_dataset():
    """
    Process the downloaded dataset and organize it into train/validation/test splits.
    """
    logger.info("Processing dataset...")
    
    # Paths for the extracted dataset
    extracted_dir = RAW_DATA_DIR / "PetImages"
    cat_dir = extracted_dir / "Cat"
    dog_dir = extracted_dir / "Dog"
    
    # Create directories for processed data
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "validation"
    test_dir = PROCESSED_DATA_DIR / "test"
    
    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(exist_ok=True)
        (directory / "cat").mkdir(exist_ok=True)
        (directory / "dog").mkdir(exist_ok=True)
    
    # Process cat images
    process_category(cat_dir, "cat", train_dir, val_dir, test_dir)
    
    # Process dog images
    process_category(dog_dir, "dog", train_dir, val_dir, test_dir)
    
    logger.info("Dataset processing completed")


def process_category(source_dir, category, train_dir, val_dir, test_dir):
    """
    Process images for a specific category (cat/dog) and split them into train/val/test.
    
    Args:
        source_dir: Directory containing the source images
        category: 'cat' or 'dog'
        train_dir: Directory for training images
        val_dir: Directory for validation images
        test_dir: Directory for test images
    """
    # Get all image files
    image_files = list(source_dir.glob("*.jpg"))
    
    # Calculate split indices (80% train, 10% validation, 10% test)
    n_images = len(image_files)
    n_train = int(0.8 * n_images)
    n_val = int(0.1 * n_images)
    
    # Split the images
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Copy files to their respective directories and store metadata
    metadata = []
    
    # Process training images
    for img_file in train_files:
        dest_file = train_dir / category / img_file.name
        shutil.copy(img_file, dest_file)
        metadata.append(get_image_metadata(img_file, category, "train"))
    
    # Process validation images
    for img_file in val_files:
        dest_file = val_dir / category / img_file.name
        shutil.copy(img_file, dest_file)
        metadata.append(get_image_metadata(img_file, category, "validation"))
    
    # Process test images
    for img_file in test_files:
        dest_file = test_dir / category / img_file.name
        shutil.copy(img_file, dest_file)
        metadata.append(get_image_metadata(img_file, category, "test"))
    
    # Store metadata in database
    store_metadata(metadata)
    
    logger.info(f"Processed {n_images} {category} images")


def get_image_metadata(img_path, label, split):
    """
    Extract metadata from an image file.
    
    Args:
        img_path: Path to the image file
        label: 'cat' or 'dog'
        split: 'train', 'validation', or 'test'
        
    Returns:
        Dictionary containing image metadata
    """
    try:
        from PIL import Image as PILImage
        
        # Get file size in KB
        size_kb = img_path.stat().st_size / 1024
        
        # Get image dimensions
        with PILImage.open(img_path) as img:
            width, height = img.size
        
        return {
            'filename': img_path.name,
            'label': label,
            'split': split,
            'width': width,
            'height': height,
            'size_kb': size_kb
        }
    except Exception as e:
        logger.warning(f"Error extracting metadata for {img_path}: {e}")
        return {
            'filename': img_path.name,
            'label': label,
            'split': split,
            'width': None,
            'height': None,
            'size_kb': None
        }


def store_metadata(metadata_list):
    """
    Store image metadata in the SQLite database.
    
    Args:
        metadata_list: List of dictionaries containing image metadata
    """
    # Create database tables if they don't exist
    Base.metadata.create_all(engine)
    
    # Create a session
    session = Session()
    
    try:
        # Add all metadata records
        for metadata in metadata_list:
            image = Image(**metadata)
            session.add(image)
        
        # Commit the changes
        session.commit()
        logger.info(f"Stored metadata for {len(metadata_list)} images in the database")
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing metadata in database: {e}")
    finally:
        session.close()


def create_metadata_csv():
    """
    Create a CSV file with image metadata from the database.
    """
    # Create a session
    session = Session()
    
    try:
        # Query all images
        images = session.query(Image).all()
        
        # Convert to DataFrame
        data = [{
            'id': img.id,
            'filename': img.filename,
            'label': img.label,
            'split': img.split,
            'width': img.width,
            'height': img.height,
            'size_kb': img.size_kb
        } for img in images]
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = PROCESSED_DATA_DIR / 'metadata.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Metadata CSV created at {csv_path}")
    except Exception as e:
        logger.error(f"Error creating metadata CSV: {e}")
    finally:
        session.close()


def main():
    """Main function to run the data acquisition and processing pipeline."""
    try:
        download_dataset()
        process_dataset()
        create_metadata_csv()
        logger.info("Data acquisition and processing completed successfully")
    except Exception as e:
        logger.error(f"Error in data acquisition pipeline: {e}")


if __name__ == "__main__":
    main() 