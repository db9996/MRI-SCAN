import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# Define the dataset directory
data_dir = '/Users/jagathguru/Desktop/5C/data'  # Path to the data folder

# Load the CSV file
metadata = pd.read_csv(os.path.join(data_dir, 'data_mask.csv'))

# Function to load the image and mask based on the CSV metadata
def load_image_and_mask(row):
    image_path = os.path.join(data_dir, row['image_path'])
    mask_path = os.path.join(data_dir, row['mask_path'])
    
    print(f"Loading image from: {image_path}")  # Debugging line
    print(f"Loading mask from: {mask_path}")    # Debugging line
    
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    return image, mask

# Function to apply CLAHE
def apply_clahe(image):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a CLAHE object and apply it
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

# Function to normalize images and masks
def normalize_image(image):
    return image / 255.0  # Normalize to [0, 1]

def normalize_mask(mask):
    return mask / 255.0  # Assuming mask values are [0, 255]

# Load all images and masks
images = []
masks = []
for _, row in metadata.iterrows():
    image, mask = load_image_and_mask(row)
    
    # Apply CLAHE to the image
    enhanced_image = apply_clahe(image)
    
    # Normalize the images and masks
    normalized_image = normalize_image(enhanced_image)
    normalized_mask = normalize_mask(mask)

    images.append(normalized_image)
    masks.append(normalized_mask)

# Print shapes of the first image and mask
print("First Image shape:", images[0].shape)
print("First Mask shape:", masks[0].shape)

# Check the total number of images and masks loaded
print("Total images loaded:", len(images))
print("Total masks loaded:", len(masks))