"""
Example Usage Script for MRI Classification CNN

This script demonstrates how to use the MRI classification system
with step-by-step examples and comments.

Author: AI Assistant
Date: 2025-12-12
"""

import os
from mri_classification_cnn import (
    MRIDataLoader,
    MRICNNModel,
    MRITrainer,
    MRIVisualizer,
    MRIPredictor,
    save_training_metadata
)


def example_1_basic_training():
    """
    Example 1: Basic training pipeline with default settings.
    
    This demonstrates the simplest way to train an MRI classification model.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Training with Default Settings")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = 'data/mri_images'
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\nData directory not found: {DATA_DIR}")
        print("Please create the directory and add your MRI images.")
        print("See MRI_CNN_README.md for directory structure details.")
        return
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    data_loader = MRIDataLoader(data_dir=DATA_DIR)
    train_gen, val_gen, test_gen, class_names = data_loader.create_data_generators()
    
    # Step 2: Build model
    print("\nStep 2: Building CNN model...")
    cnn_model = MRICNNModel(num_classes=len(class_names))
    model = cnn_model.build_model()
    
    # Step 3: Train
    print("\nStep 3: Training model...")
    trainer = MRITrainer(model, train_gen, val_gen, test_gen)
    history = trainer.train(epochs=10)  # Using fewer epochs for quick demo
    
    # Step 4: Evaluate
    print("\nStep 4: Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Step 5: Visualize
    print("\nStep 5: Creating visualizations...")
    visualizer = MRIVisualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(
        eval_results['confusion_matrix'],
        eval_results['class_names']
    )
    
    print("\n" + "=" * 80)
    print("Example 1 completed!")
    print("=" * 80)


def example_2_custom_configuration():
    """
    Example 2: Training with custom configuration and hyperparameters.
    
    This shows how to customize various aspects of the training process.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Training with Custom Configuration")
    print("=" * 80)
    
    DATA_DIR = 'data/mri_images'
    
    if not os.path.exists(DATA_DIR):
        print(f"\nData directory not found: {DATA_DIR}")
        return
    
    # Custom configuration
    IMG_SIZE = 128  # Smaller image size for faster training
    BATCH_SIZE = 16  # Smaller batch size
    VAL_SPLIT = 0.15  # 15% validation data
    TEST_SPLIT = 0.15  # 15% test data
    
    print("\nCustom Configuration:")
    print(f"  - Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Validation Split: {VAL_SPLIT}")
    print(f"  - Test Split: {TEST_SPLIT}")
    
    # Load data with custom settings
    data_loader = MRIDataLoader(
        data_dir=DATA_DIR,
        img_height=IMG_SIZE,
        img_width=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    train_gen, val_gen, test_gen, class_names = data_loader.create_data_generators(
        augment=True  # Enable data augmentation
    )
    
    # Build model with custom settings
    cnn_model = MRICNNModel(
        img_height=IMG_SIZE,
        img_width=IMG_SIZE,
        num_classes=len(class_names)
    )
    model = cnn_model.build_model()
    
    # Train with custom parameters
    trainer = MRITrainer(model, train_gen, val_gen, test_gen)
    history = trainer.train(
        epochs=20,
        model_save_path='models/custom_mri_model.h5',
        patience=5  # Early stopping patience
    )
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Save metadata
    save_training_metadata(history, eval_results, class_names,
                          save_path='models/custom_training_metadata.json')
    
    print("\n" + "=" * 80)
    print("Example 2 completed!")
    print("Model saved to: models/custom_mri_model.h5")
    print("=" * 80)


def example_3_prediction_workflow():
    """
    Example 3: Using a trained model to make predictions.
    
    This demonstrates how to load a saved model and make predictions
    on new MRI images.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Prediction on New MRI Images")
    print("=" * 80)
    
    MODEL_PATH = 'models/mri_cnn_model.h5'
    TEST_IMAGE = 'path/to/test_mri.jpg'  # Update with actual path
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found: {MODEL_PATH}")
        print("Please train a model first using example_1_basic_training()")
        return
    
    # Define class names (should match training)
    class_names = ['class1', 'class2']  # Update with your actual classes
    
    # Create predictor
    print("\nLoading trained model...")
    predictor = MRIPredictor(
        model_path=MODEL_PATH,
        class_names=class_names
    )
    
    # Single image prediction
    if os.path.exists(TEST_IMAGE):
        print(f"\nPredicting class for: {TEST_IMAGE}")
        predicted_class, probabilities = predictor.predict_image(
            TEST_IMAGE,
            show_image=True
        )
        
        print(f"\nResult: {predicted_class}")
        print(f"Confidence: {probabilities[class_names.index(predicted_class)]:.2%}")
    else:
        print(f"\nTest image not found: {TEST_IMAGE}")
        print("Please provide a valid image path.")
    
    # Batch prediction example
    print("\n--- Batch Prediction Example ---")
    image_paths = [
        'image1.jpg',
        'image2.jpg',
        'image3.jpg'
    ]
    
    # Filter existing images
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if existing_images:
        print(f"\nPredicting {len(existing_images)} images...")
        results = predictor.predict_batch(existing_images)
        
        print("\nBatch Prediction Results:")
        for i, (img_path, (pred_class, probs)) in enumerate(zip(existing_images, results)):
            print(f"  {i+1}. {img_path}: {pred_class} "
                  f"(confidence: {probs[class_names.index(pred_class)]:.2%})")
    else:
        print("No test images found for batch prediction.")
    
    print("\n" + "=" * 80)
    print("Example 3 completed!")
    print("=" * 80)


def example_4_visualization_only():
    """
    Example 4: Create visualizations from existing training history.
    
    This shows how to generate plots if you have training history
    and evaluation results stored.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Visualization from Saved Results")
    print("=" * 80)
    
    # This is a mock example - in practice, you'd load actual results
    # from saved files or a completed training run
    
    # Example history dictionary (normally from training)
    history = {
        'accuracy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85],
        'val_accuracy': [0.48, 0.58, 0.68, 0.72, 0.76, 0.78, 0.79, 0.80],
        'loss': [1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3],
        'val_loss': [1.25, 1.05, 0.85, 0.65, 0.55, 0.45, 0.40, 0.38],
        'precision': [0.52, 0.62, 0.72, 0.76, 0.81, 0.83, 0.85, 0.86],
        'val_precision': [0.50, 0.60, 0.70, 0.74, 0.78, 0.80, 0.81, 0.82],
        'recall': [0.48, 0.58, 0.68, 0.73, 0.79, 0.81, 0.83, 0.84],
        'val_recall': [0.46, 0.56, 0.66, 0.70, 0.75, 0.77, 0.78, 0.79]
    }
    
    # Create visualizer
    visualizer = MRIVisualizer()
    
    # Plot training history
    print("\nCreating training history plots...")
    visualizer.plot_training_history(
        history,
        save_path='plots/example_training_history.png'
    )
    
    print("\n" + "=" * 80)
    print("Example 4 completed!")
    print("Plots saved to: plots/example_training_history.png")
    print("=" * 80)


def example_5_incremental_learning():
    """
    Example 5: Load a pre-trained model and continue training.
    
    This demonstrates transfer learning / incremental training.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Continue Training from Saved Model")
    print("=" * 80)
    
    MODEL_PATH = 'models/mri_cnn_model.h5'
    DATA_DIR = 'data/mri_images'
    
    # Check prerequisites
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found: {MODEL_PATH}")
        print("Please train a model first.")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"\nData directory not found: {DATA_DIR}")
        return
    
    # Load data
    print("\nLoading data...")
    data_loader = MRIDataLoader(data_dir=DATA_DIR)
    train_gen, val_gen, test_gen, class_names = data_loader.create_data_generators()
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model from: {MODEL_PATH}")
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    
    print("\nModel loaded successfully!")
    print("Continuing training with new data...")
    
    # Continue training
    trainer = MRITrainer(model, train_gen, val_gen, test_gen)
    history = trainer.train(
        epochs=10,  # Additional epochs
        model_save_path='models/mri_cnn_model_continued.h5',
        patience=5
    )
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    print("\n" + "=" * 80)
    print("Example 5 completed!")
    print("Updated model saved to: models/mri_cnn_model_continued.h5")
    print("=" * 80)


def print_menu():
    """Print the example menu."""
    print("\n" + "=" * 80)
    print("MRI CLASSIFICATION CNN - EXAMPLE USAGE")
    print("=" * 80)
    print("\nAvailable Examples:")
    print("  1. Basic training with default settings")
    print("  2. Training with custom configuration")
    print("  3. Prediction on new MRI images")
    print("  4. Visualization from saved results")
    print("  5. Continue training from saved model")
    print("  0. Exit")
    print("=" * 80)


def main():
    """
    Main function to run example demonstrations.
    
    This provides an interactive menu to run different examples.
    """
    examples = {
        '1': example_1_basic_training,
        '2': example_2_custom_configuration,
        '3': example_3_prediction_workflow,
        '4': example_4_visualization_only,
        '5': example_5_incremental_learning
    }
    
    while True:
        print_menu()
        choice = input("\nSelect an example to run (0-5): ").strip()
        
        if choice == '0':
            print("\nExiting. Thank you!")
            break
        elif choice in examples:
            try:
                examples[choice]()
            except Exception as e:
                print(f"\nError running example: {e}")
                print("Please check the error message and try again.")
        else:
            print("\nInvalid choice. Please select 0-5.")
    

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MRI CLASSIFICATION CNN - EXAMPLE DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis script provides interactive examples of using the MRI classification system.")
    print("Each example demonstrates different aspects of the pipeline.")
    print("\nNote: Before running examples, ensure you have:")
    print("  1. Organized your MRI images in data/mri_images/ directory")
    print("  2. Installed all required dependencies (pip install -r requirements.txt)")
    print("=" * 80)
    
    # Run interactive menu
    main()
