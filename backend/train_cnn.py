"""
train_cnn.py — CNN Model Training for Face Emotion Recognition

Architecture:
Input (48x48x1) → Conv2D(64) → Conv2D(128) → Conv2D(512) → Conv2D(512) → Dense(256) → Dense(7)

Each Conv block: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
BatchNorm: Normalizes layer inputs (faster training, less overfitting)
Dropout: Randomly drops neurons (prevents overfitting)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


def create_cnn_model():
    """Create the CNN architecture for emotion classification."""
    model = Sequential([
        # Block 1: 64 filters
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2: 128 filters
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3: 512 filters
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4: 512 filters
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output: 7 emotion classes with softmax
        Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(data_dir='data/fer2013', batch_size=64, epochs=25):
    """Train the CNN on FER-2013 dataset."""
    print("Creating data generators...")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    print(f"Classes: {train_generator.class_indices}")
    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Create and train model
    model = create_cnn_model()
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    print("\nTraining... (this will take 30-60 minutes on CPU)")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/emotion_model.h5')
    
    # Save class indices
    import json
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    # Plot training history
    plot_history(history)
    
    print("\nTraining complete! Model saved to models/emotion_model.h5")
    return model, history


def plot_history(history):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150)
    print("Training plots saved to models/training_history.png")
    plt.close()


if __name__ == "__main__":
    train_model()