import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score

def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess the input image for model prediction.
    Args:
        image: The input image as a PIL Image.
        target_size: The target size to resize the image to.
    Returns:
        A preprocessed image array ready for prediction.
    """
    image = image.resize(target_size)  # Resize to the target size
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def calculate_dice_score(y_true, y_pred, smooth=1e-6):
    """
    Calculate the DICE score between the ground truth and predicted masks.
    Args:
        y_true: Ground truth mask (binary).
        y_pred: Predicted mask (binary).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        The DICE score.
    """
    y_true_flat = tf.flatten(y_true)
    y_pred_flat = tf.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

def load_image_and_mask(image_path, mask_path):
    """
    Load an image and its corresponding mask.
    Args:
        image_path: Path to the image file.
        mask_path: Path to the mask file.
    Returns:
        A tuple of the image and mask as numpy arrays.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load the image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask in grayscale
    return image, mask

def display_results(original_image, ground_truth_mask, predicted_mask):
    """
    Display the original image, ground truth mask, and predicted mask.
    Args:
        original_image: The original input image.
        ground_truth_mask: The ground truth segmentation mask.
        predicted_mask: The predicted segmentation mask.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.show()