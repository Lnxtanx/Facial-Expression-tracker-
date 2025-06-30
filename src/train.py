import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import load_data
from model import build_emotion_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard,
    CSVLogger, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# Config
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = (48, 48)
NUM_CLASSES = 7

# Load data
print("📦 Loading dataset...")
train_generator, test_generator = load_data(
    train_dir='data/train',
    test_dir='data/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

print(f"✅ Samples - Train: {train_generator.samples}, Test: {test_generator.samples}")
print(f"🧠 Classes found: {train_generator.class_indices}")

# Build model
print("🔧 Building model...")
model = build_emotion_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_path = 'models/emotion_model.keras'  # Updated from .h5 to .keras
log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    CSVLogger("training_log.csv"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)
]

# Train
print("🚀 Starting training...")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model (safety)
model.save(checkpoint_path)
print(f"\n✅ Final model saved to: {checkpoint_path}")

# Plot history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Loss over Epochs')

    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.show()

plot_history(history)
