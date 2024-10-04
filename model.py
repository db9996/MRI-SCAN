import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from preprocessing import load_image_and_mask, apply_clahe, normalize_image, normalize_mask

# Define Nested U-Net (U-Net++)
def nested_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid for binary segmentation

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Example of creating and compiling the models
input_shape = (256, 256, 1)  # Adjust as necessary
unet_model = nested_unet(input_shape)

# Compile the model
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Nested U-Net model compiled successfully!")

# Load your preprocessed data
data_dir = '/Users/jagathguru/Desktop/5C/data'  # Adjust path if necessary
metadata = pd.read_csv(os.path.join(data_dir, 'data_mask.csv'))

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

# Prepare your data for training
X = np.array(images)  # Preprocessed images
y = np.array(masks)   # Preprocessed masks

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Callbacks for training
checkpoint = ModelCheckpoint('src/best_unet_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Nested U-Net model
print("Training Nested U-Net model...")
history_unet = unet_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                               epochs=50, batch_size=16, callbacks=[checkpoint, early_stopping], verbose=1)

# Save the final model
unet_model.save('src/nested_unet_model_final.keras')

print("Model trained and saved successfully!")