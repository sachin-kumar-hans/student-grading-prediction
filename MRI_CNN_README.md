# MRI Classification using Convolutional Neural Network (CNN)

## Overview

This project provides a complete, modular pipeline for MRI image classification using deep learning. The implementation uses TensorFlow/Keras to build a CNN model capable of classifying MRI images into multiple categories.

## Features

### 1. **Data Loading and Augmentation**
- Automated data loading from directory structure
- Built-in data augmentation for training robustness
- Configurable train/validation/test split
- Support for multiple image formats (JPEG, PNG, etc.)

### 2. **CNN Model Architecture**
The model includes:
- **Convolutional Layers**: 4 blocks with increasing filter sizes (32, 64, 128, 256)
- **Batch Normalization**: For stable training and faster convergence
- **Max Pooling**: For spatial dimension reduction
- **Dropout Layers**: To prevent overfitting (rates: 0.25 and 0.5)
- **Fully Connected Layers**: 2 dense layers (512 and 256 units)
- **Output Layer**: Softmax activation for multi-class classification

### 3. **Training and Evaluation**
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Model Checkpointing**: Automatically saves the best model
- **Learning Rate Reduction**: Adaptive learning rate adjustment
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Visual representation of classification performance

### 4. **Prediction on New Images**
- Easy-to-use prediction interface
- Single image prediction with confidence scores
- Batch prediction support
- Visual display of predictions

### 5. **Visualization**
- Training/validation accuracy and loss curves
- Precision and recall trends
- Confusion matrix heatmap
- Model architecture diagram

### 6. **Model Persistence**
- Save trained models for future use
- Load models for inference
- Export training metadata (JSON format)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- `tensorflow>=2.10.0`: Deep learning framework
- `keras>=2.10.0`: High-level neural networks API
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `scikit-learn`: Machine learning utilities
- `pillow>=9.0.0`: Image processing

## Dataset Structure

Organize your MRI images in the following directory structure:

```
data/mri_images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── classN/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each subdirectory represents a different class/category of MRI images.

## Usage

### Basic Usage - Complete Pipeline

Run the entire training pipeline:

```python
from mri_classification_cnn import main

# This will:
# 1. Load and preprocess data
# 2. Build the CNN model
# 3. Train with early stopping
# 4. Evaluate on test set
# 5. Generate visualizations
# 6. Save the trained model
main()
```

### Advanced Usage - Custom Configuration

#### 1. Load Data with Custom Settings

```python
from mri_classification_cnn import MRIDataLoader

data_loader = MRIDataLoader(
    data_dir='data/mri_images',
    img_height=224,
    img_width=224,
    batch_size=32,
    validation_split=0.2,
    test_split=0.1
)

train_gen, val_gen, test_gen, class_names = data_loader.create_data_generators(
    augment=True
)
```

#### 2. Build Custom CNN Model

```python
from mri_classification_cnn import MRICNNModel

cnn_model = MRICNNModel(
    img_height=224,
    img_width=224,
    num_classes=len(class_names)
)

model = cnn_model.build_model()
```

#### 3. Train the Model

```python
from mri_classification_cnn import MRITrainer

trainer = MRITrainer(model, train_gen, val_gen, test_gen)

# Train with custom parameters
history = trainer.train(
    epochs=50,
    model_save_path='models/my_mri_model.h5',
    patience=10
)
```

#### 4. Evaluate Model Performance

```python
# Evaluate on test set
eval_results = trainer.evaluate()

# Results include:
# - test_accuracy
# - test_precision
# - test_recall
# - confusion_matrix
# - classification_report
```

#### 5. Visualize Training Results

```python
from mri_classification_cnn import MRIVisualizer

visualizer = MRIVisualizer()

# Plot training history
visualizer.plot_training_history(
    history,
    save_path='plots/training_history.png'
)

# Plot confusion matrix
visualizer.plot_confusion_matrix(
    eval_results['confusion_matrix'],
    eval_results['class_names'],
    save_path='plots/confusion_matrix.png'
)
```

#### 6. Make Predictions on New Images

```python
from mri_classification_cnn import MRIPredictor

# Load trained model
predictor = MRIPredictor(
    model_path='models/mri_cnn_model.h5',
    class_names=['class1', 'class2', 'class3']
)

# Predict single image
predicted_class, probabilities = predictor.predict_image(
    'path/to/new_mri_image.jpg',
    show_image=True
)

# Batch prediction
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = predictor.predict_batch(image_paths)
```

## Model Architecture Details

### Layer Configuration

| Layer Type | Output Shape | Parameters | Description |
|------------|-------------|------------|-------------|
| Conv2D | (112, 112, 32) | 896 | First conv block (32 filters) |
| BatchNorm | (112, 112, 32) | 128 | Normalize activations |
| Conv2D | (112, 112, 32) | 9,248 | Second conv in block |
| BatchNorm | (112, 112, 32) | 128 | Normalize activations |
| MaxPool2D | (56, 56, 32) | 0 | Reduce spatial dimensions |
| Dropout | (56, 56, 32) | 0 | Dropout rate: 0.25 |
| Conv2D | (56, 56, 64) | 18,496 | Second conv block (64 filters) |
| ... | ... | ... | ... |
| Dense | (512) | 6,554,112 | Fully connected layer |
| Dense | (256) | 131,328 | Fully connected layer |
| Dense | (num_classes) | varies | Output layer |

**Total Parameters**: ~7-8 million (varies with number of classes)

### Key Design Decisions

1. **Progressive Channel Expansion**: 32 → 64 → 128 → 256 filters
   - Captures features from simple to complex
   
2. **Batch Normalization**: After every convolutional layer
   - Stabilizes training and allows higher learning rates
   
3. **Dropout Regularization**: 
   - 0.25 after convolutional blocks
   - 0.5 in fully connected layers
   - Prevents overfitting
   
4. **Adam Optimizer**: 
   - Initial learning rate: 0.001
   - Adaptive learning rate adjustment

## Training Process

### Early Stopping Strategy

The model uses early stopping with the following configuration:
- **Monitor**: Validation loss
- **Patience**: 10 epochs (configurable)
- **Restore Best Weights**: Yes

This prevents overfitting and saves training time.

### Learning Rate Scheduling

The learning rate is automatically reduced when validation loss plateaus:
- **Monitor**: Validation loss
- **Factor**: 0.5 (halves the learning rate)
- **Patience**: 5 epochs
- **Minimum LR**: 1e-7

### Data Augmentation

Training data is augmented with:
- Random rotations (up to 20 degrees)
- Width/height shifts (20%)
- Shear transformations (20%)
- Zoom (20%)
- Horizontal flipping
- Nearest-neighbor fill mode

## Output Files

After training, the following files are generated:

### Models
- `models/mri_cnn_model.h5`: Trained model (Keras format)
- `models/training_metadata.json`: Training statistics and metadata

### Plots
- `plots/training_history.png`: Accuracy and loss curves
- `plots/confusion_matrix.png`: Confusion matrix heatmap
- `plots/model_architecture.png`: Model architecture diagram (optional)

## Performance Metrics

The system tracks and reports:

1. **Accuracy**: Overall classification accuracy
2. **Precision**: Proportion of correct positive predictions
3. **Recall**: Proportion of actual positives correctly identified
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Detailed breakdown of predictions vs. actual

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
**Solution**: Reduce batch size or image dimensions

```python
data_loader = MRIDataLoader(
    data_dir='data/mri_images',
    img_height=128,  # Reduced from 224
    img_width=128,
    batch_size=16   # Reduced from 32
)
```

#### 2. Slow Training
**Solution**: Use GPU acceleration

```python
# Check GPU availability
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

#### 3. Poor Validation Accuracy
**Solutions**:
- Increase training data
- Adjust learning rate
- Modify dropout rates
- Try different augmentation strategies

#### 4. Model Not Saving
**Solution**: Ensure directory exists

```python
import os
os.makedirs('models', exist_ok=True)
```

## Extension Ideas

The modular design allows for easy extensions:

1. **Transfer Learning**: Use pre-trained models (ResNet, VGG, InceptionV3)
2. **3D CNN**: For volumetric MRI data
3. **Attention Mechanisms**: Focus on relevant image regions
4. **Ensemble Methods**: Combine multiple models
5. **Explainability**: Add Grad-CAM for visualization
6. **Web Interface**: Deploy with Streamlit or Flask

## Code Organization

The code is organized into several classes for modularity:

- **MRIDataLoader**: Handles data loading and preprocessing
- **MRICNNModel**: Defines the CNN architecture
- **MRITrainer**: Manages training and evaluation
- **MRIVisualizer**: Creates plots and visualizations
- **MRIPredictor**: Handles inference on new images

This modular structure makes it easy to:
- Swap different model architectures
- Modify data preprocessing
- Add new visualization techniques
- Extend functionality

## Best Practices

1. **Data Organization**: Keep data well-organized in class subdirectories
2. **Version Control**: Track model versions and hyperparameters
3. **Validation**: Always use separate validation and test sets
4. **Documentation**: Document any custom modifications
5. **Reproducibility**: Set random seeds for consistent results

## Contributing

To add new features:
1. Follow the existing class structure
2. Add comprehensive docstrings
3. Include usage examples
4. Update this README

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```
MRI Classification CNN - A Modular Deep Learning Pipeline
GitHub: sachin-kumar-hans/student-grading-prediction
Year: 2025
```

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This implementation is designed for binary and multi-class MRI classification tasks. Adapt the code as needed for your specific use case.
