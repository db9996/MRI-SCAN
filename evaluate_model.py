import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the best model
unet_model = load_model('best_unet_model.h5')  # Adjust the path as necessary

# Load your preprocessed validation data
# Assuming X_val and y_val are defined

# Generate predictions
predictions = unet_model.predict(X_val)

# Binarize the predictions
binary_predictions = (predictions > 0.5).astype(np.uint8)

# Calculate DICE Score
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.flatten(y_true)
    y_pred_f = tf.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

dice_score = dice_coefficient(y_val, binary_predictions)
print(f'DICE Score: {dice_score.numpy():.4f}')

# Visualize results
def visualize_results(images, masks, predictions, num_images=5):
    for i in range(num_images):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i], cmap='gray')
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.show()

# Visualize the results
visualize_results(X_val, y_val, binary_predictions)