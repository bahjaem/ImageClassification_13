"""
CNN Image Classification on CIFAR-10
=====================================
Project: Deep Learning Image Classification
Dataset: CIFAR-10 (60,000 images, 10 classes)
Model  : Convolutional Neural Network (CNN)
Author : [Your Name] | Roll No: [Your Roll No]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('screenshots', exist_ok=True)
os.makedirs('saved_model', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Print versions
# ─────────────────────────────────────────────
print("=" * 50)
print("  CNN Image Classification - CIFAR-10")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")

# ─────────────────────────────────────────────
# STEP 2: Load Dataset
# ─────────────────────────────────────────────
print("\n[1/7] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"  Training samples : {x_train.shape[0]}")
print(f"  Test samples     : {x_test.shape[0]}")
print(f"  Image shape      : {x_train.shape[1:]}")
print(f"  Classes          : {class_names}")

# ─────────────────────────────────────────────
# STEP 3: Visualize Sample Images
# ─────────────────────────────────────────────
print("\n[2/7] Saving sample images...")
plt.figure(figsize=(12, 5))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]], fontsize=7)
    plt.axis('off')
plt.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/sample_images.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/sample_images.png")

# ─────────────────────────────────────────────
# STEP 4: Preprocess Data
# ─────────────────────────────────────────────
print("\n[3/7] Preprocessing data...")
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test,  10)
print("  Normalization done. Pixel range: [0.0, 1.0]")
print("  Labels one-hot encoded.")

# ─────────────────────────────────────────────
# STEP 5: Build CNN Model
# ─────────────────────────────────────────────
print("\n[4/7] Building CNN architecture...")

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name='data_augmentation')

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_CIFAR10')
    return model

model = build_cnn_model()
model.summary()

# ─────────────────────────────────────────────
# STEP 6: Compile
# ─────────────────────────────────────────────
print("\n[5/7] Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("  Optimizer: Adam | Loss: Categorical Cross-Entropy | Metric: Accuracy")

# ─────────────────────────────────────────────
# STEP 7: Train
# ─────────────────────────────────────────────
print("\n[6/7] Training model (this may take a few minutes)...")
early_stop = EarlyStopping(monitor='val_accuracy', patience=10,
                           restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('saved_model/best_model.keras',
                             monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(
    x_train, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
print("  Training complete!")

# ─────────────────────────────────────────────
# STEP 8: Plot Training History
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train Accuracy',      color='royalblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='tomato',    linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss',      color='royalblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='tomato',    linewidth=2)
axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Training History', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/training_history.png")

# ─────────────────────────────────────────────
# STEP 9: Evaluate
# ─────────────────────────────────────────────
print("\n[7/7] Evaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print("=" * 40)
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print("=" * 40)

# ─────────────────────────────────────────────
# STEP 10: Confusion Matrix & Classification Report
# ─────────────────────────────────────────────
y_pred         = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_test.flatten()

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/confusion_matrix.png")

print('\nClassification Report:')
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# ─────────────────────────────────────────────
# STEP 11: Sample Predictions
# ─────────────────────────────────────────────
import random
indices = random.sample(range(len(x_test)), 10)
plt.figure(figsize=(15, 4))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx])
    true_label = class_names[y_test[idx][0]]
    pred_label = class_names[y_pred_classes[idx]]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'T: {true_label}\nP: {pred_label}', color=color, fontsize=8)
    plt.axis('off')
plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/predictions.png")

print("\n✓ All outputs saved. Model saved to saved_model/best_model.keras")
print("✓ Project complete!")
