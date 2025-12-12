"""
MRI Classification using Convolutional Neural Network (CNN)

This module provides a complete pipeline for MRI image classification including:
- Data loading with augmentation
- CNN model definition
- Training with early stopping
- Evaluation metrics
- Prediction on new images
- Model saving/loading
- Visualization of training metrics

Author: AI Assistant
Date: 2025-12-12
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, Optional, List
import json
from datetime import datetime


class MRIDataLoader:
    """
    Handles loading and preprocessing of MRI image datasets.
    
    Attributes:
        data_dir (str): Root directory containing MRI images
        img_height (int): Target height for images
        img_width (int): Target width for images
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing
    """
    
    def __init__(self, data_dir: str, img_height: int = 224, img_width: int = 224,
                 batch_size: int = 32, validation_split: float = 0.2,
                 test_split: float = 0.1):
        """
        Initialize the MRI data loader.
        
        Args:
            data_dir: Path to directory containing MRI images
            img_height: Target height for resizing images
            img_width: Target width for resizing images
            batch_size: Number of images per batch
            validation_split: Proportion of data for validation
            test_split: Proportion of data for testing
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        
    def create_data_generators(self, augment: bool = True) -> Tuple:
        """
        Create data generators for training, validation, and testing.
        
        Args:
            augment: Whether to apply data augmentation to training data
            
        Returns:
            Tuple containing (train_generator, val_generator, test_generator, class_names)
        """
        # Data augmentation for training set
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=self.validation_split + self.test_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=self.validation_split + self.test_split
            )
        
        # No augmentation for validation and test sets
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split + self.test_split
        )
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation data generator
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # For test set, we need to split the validation set further
        # This is a simplified approach - in production, you'd want separate directories
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        class_names = list(train_generator.class_indices.keys())
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Test samples: {test_generator.samples}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {class_names}")
        
        return train_generator, val_generator, test_generator, class_names


class MRICNNModel:
    """
    Defines and manages the CNN model for MRI classification.
    
    Attributes:
        img_height (int): Input image height
        img_width (int): Input image width
        num_classes (int): Number of output classes
        model: Keras model instance
    """
    
    def __init__(self, img_height: int = 224, img_width: int = 224, 
                 num_classes: int = 2):
        """
        Initialize the CNN model architecture.
        
        Args:
            img_height: Input image height
            img_width: Input image width
            num_classes: Number of classification categories
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self) -> models.Model:
        """
        Build a CNN model with convolutional, pooling, dropout, 
        batch normalization, and fully connected layers.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        
        print("\nModel Architecture:")
        print("=" * 80)
        model.summary()
        print("=" * 80)
        
        return model
    
    def get_model(self) -> models.Model:
        """
        Get the current model instance.
        
        Returns:
            Keras model instance
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return self.model


class MRITrainer:
    """
    Handles training and evaluation of the MRI classification model.
    
    Attributes:
        model: Keras model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator
        history: Training history object
    """
    
    def __init__(self, model: models.Model, train_generator, 
                 val_generator, test_generator):
        """
        Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            train_generator: Training data generator
            val_generator: Validation data generator
            test_generator: Test data generator
        """
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.history = None
        
    def train(self, epochs: int = 50, model_save_path: str = 'models/mri_cnn_model.h5',
              patience: int = 10) -> dict:
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            epochs: Maximum number of training epochs
            model_save_path: Path to save the best model
            patience: Number of epochs to wait before early stopping
            
        Returns:
            Training history dictionary
        """
        # Create directory for saving models if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save the best model
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\nStarting training...")
        print("=" * 80)
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("=" * 80)
        print("Training completed!")
        
        return self.history.history
    
    def evaluate(self, generator=None) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            generator: Data generator for evaluation (uses test_generator if None)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if generator is None:
            generator = self.test_generator
            
        print("\nEvaluating model on test set...")
        print("=" * 80)
        
        # Get predictions
        generator.reset()
        predictions = self.model.predict(generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = generator.classes
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            generator, verbose=0
        )
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Classification report
        class_names = list(generator.class_indices.keys())
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                   target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        print("\nConfusion Matrix:")
        print(cm)
        
        print("=" * 80)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'predictions': predicted_classes,
            'true_labels': true_classes,
            'confusion_matrix': cm,
            'class_names': class_names
        }


class MRIVisualizer:
    """
    Handles visualization of training metrics and results.
    """
    
    @staticmethod
    def plot_training_history(history: dict, save_path: str = 'plots/training_history.png'):
        """
        Plot training and validation accuracy and loss over epochs.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training Precision', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision Over Epochs', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Precision', fontsize=12)
            axes[1, 0].legend(loc='lower right')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training Recall', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall Over Epochs', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Recall', fontsize=12)
            axes[1, 1].legend(loc='lower right')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        plt.show()
        
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                            save_path: str = 'plots/confusion_matrix.png'):
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save the plot
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
        plt.show()


class MRIPredictor:
    """
    Handles prediction on new MRI images.
    
    Attributes:
        model: Trained Keras model
        class_names: List of class names
        img_height: Input image height
        img_width: Input image width
    """
    
    def __init__(self, model_path: str, class_names: List[str],
                 img_height: int = 224, img_width: int = 224):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_path: Path to saved model file
            class_names: List of class names
            img_height: Input image height
            img_width: Input image width
        """
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_height = img_height
        self.img_width = img_width
        print(f"Model loaded from: {model_path}")
        
    def predict_image(self, image_path: str, show_image: bool = True) -> Tuple[str, np.ndarray]:
        """
        Predict the class of a single MRI image.
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image with prediction
            
        Returns:
            Tuple of (predicted_class_name, prediction_probabilities)
        """
        # Load and preprocess the image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        print(f"\nPrediction for: {image_path}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("\nAll class probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {predictions[0][i]:.4f}")
        
        # Display the image with prediction if requested
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})",
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return predicted_class, predictions[0]
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of tuples (predicted_class_name, prediction_probabilities)
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path, show_image=False)
            results.append(result)
        return results


def save_training_metadata(history: dict, eval_results: dict, 
                          class_names: List[str],
                          save_path: str = 'models/training_metadata.json'):
    """
    Save training metadata to a JSON file for future reference.
    
    Args:
        history: Training history dictionary
        eval_results: Evaluation results dictionary
        class_names: List of class names
        save_path: Path to save the metadata file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'class_names': class_names,
        'final_epoch': len(history['loss']),
        'final_train_accuracy': float(history['accuracy'][-1]),
        'final_val_accuracy': float(history['val_accuracy'][-1]),
        'final_train_loss': float(history['loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'test_accuracy': float(eval_results['test_accuracy']),
        'test_loss': float(eval_results['test_loss']),
        'test_precision': float(eval_results['test_precision']),
        'test_recall': float(eval_results['test_recall'])
    }
    
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nTraining metadata saved to: {save_path}")


def main():
    """
    Main function to demonstrate the complete MRI classification pipeline.
    
    This function orchestrates the entire workflow including:
    - Data loading and preprocessing
    - Model building
    - Training with early stopping
    - Evaluation on test set
    - Visualization of results
    - Model saving
    
    Note: Update the data_dir path to point to your MRI dataset directory.
    The directory should have subdirectories for each class, e.g.:
        data/mri_images/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                image1.jpg
                image2.jpg
                ...
    """
    print("=" * 80)
    print("MRI CLASSIFICATION USING CNN")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = 'data/mri_images'  # Update this path to your MRI dataset
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    MODEL_SAVE_PATH = 'models/mri_cnn_model.h5'
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nWarning: Data directory '{DATA_DIR}' not found!")
        print("Please create the directory and organize your MRI images as follows:")
        print(f"  {DATA_DIR}/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("      ...")
        print("    class2/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("      ...")
        print("\nCreating example directory structure...")
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Directory created: {DATA_DIR}")
        print("Please add your MRI images and run the script again.")
        return
    
    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: Loading and preparing data")
    print("=" * 80)
    data_loader = MRIDataLoader(
        data_dir=DATA_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE
    )
    
    train_gen, val_gen, test_gen, class_names = data_loader.create_data_generators(
        augment=True
    )
    
    # Step 2: Build model
    print("\n" + "=" * 80)
    print("STEP 2: Building CNN model")
    print("=" * 80)
    cnn_model = MRICNNModel(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_classes=len(class_names)
    )
    model = cnn_model.build_model()
    
    # Save model architecture diagram
    try:
        plot_model(model, to_file='plots/model_architecture.png', 
                  show_shapes=True, show_layer_names=True)
        print("\nModel architecture diagram saved to: plots/model_architecture.png")
    except Exception as e:
        print(f"\nCould not save model architecture diagram: {e}")
    
    # Step 3: Train model
    print("\n" + "=" * 80)
    print("STEP 3: Training model")
    print("=" * 80)
    trainer = MRITrainer(model, train_gen, val_gen, test_gen)
    history = trainer.train(epochs=EPOCHS, model_save_path=MODEL_SAVE_PATH)
    
    # Step 4: Evaluate model
    print("\n" + "=" * 80)
    print("STEP 4: Evaluating model")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    # Step 5: Visualize results
    print("\n" + "=" * 80)
    print("STEP 5: Visualizing results")
    print("=" * 80)
    visualizer = MRIVisualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(
        eval_results['confusion_matrix'],
        eval_results['class_names']
    )
    
    # Step 6: Save metadata
    print("\n" + "=" * 80)
    print("STEP 6: Saving training metadata")
    print("=" * 80)
    save_training_metadata(history, eval_results, class_names)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("To make predictions on new images, use the MRIPredictor class:")
    print(f"  predictor = MRIPredictor('{MODEL_SAVE_PATH}', {class_names})")
    print("  predictor.predict_image('path/to/your/mri_image.jpg')")
    

def example_prediction():
    """
    Example function demonstrating how to use the trained model for prediction.
    
    This should be called after training is complete and the model is saved.
    """
    print("=" * 80)
    print("EXAMPLE: Making predictions on new MRI images")
    print("=" * 80)
    
    # Configuration (update these paths)
    MODEL_PATH = 'models/mri_cnn_model.h5'
    CLASS_NAMES = ['class1', 'class2']  # Update with your actual class names
    TEST_IMAGE_PATH = 'path/to/test_image.jpg'  # Update with actual image path
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train the model first by running the main() function.")
        return
    
    # Create predictor
    predictor = MRIPredictor(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )
    
    # Make prediction on a single image
    if os.path.exists(TEST_IMAGE_PATH):
        predicted_class, probabilities = predictor.predict_image(TEST_IMAGE_PATH)
    else:
        print(f"\nWarning: Test image not found at {TEST_IMAGE_PATH}")
        print("Please provide a valid image path to test predictions.")


if __name__ == "__main__":
    # Run the main training pipeline
    main()
    
    # Uncomment below to run example prediction after training
    # example_prediction()
