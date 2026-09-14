import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ensure GPU is used if available
print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

# Paths to dataset directories
train_dir = "C:/Users/wcnna/PycharmProjects/pythonProjectai/archive/train"  # Path to your training directory
test_dir = "C:/Users/wcnna/PycharmProjects/pythonProjectai/archive/validation"  # Path to your testing directory

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to keep labels in the same order as predictions
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data,
    epochs=50,
    validation_data=test_data,
    callbacks=[early_stopping]
)

# Save the model
model.save('face_recognition_model.h5')

# Plot Training and Validation Results
def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_results(history)

# Load the model for evaluation and prediction
loaded_model = load_model('face_recognition_model.h5')

# Evaluate the model on the test dataset
test_loss, test_accuracy = loaded_model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate predictions
test_labels = test_data.classes  # True labels
class_indices = test_data.class_indices  # Class label mapping
class_names = list(class_indices.keys())  # Class names

# Predict
test_predictions = loaded_model.predict(test_data)
predicted_classes = np.argmax(test_predictions, axis=1)  # Convert one-hot to labels

# Handle class name mismatch
unique_labels = np.unique(test_labels)
adjusted_class_names = [class_names[i] for i in unique_labels]  # Only include present classes

# Generate Classification Report
print("Classification Report:")
print(classification_report(test_labels, predicted_classes, target_names=adjusted_class_names, labels=unique_labels))

# Generate Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes, labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=adjusted_class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()
