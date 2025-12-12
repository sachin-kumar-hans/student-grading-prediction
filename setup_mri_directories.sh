#!/bin/bash
# 
# Setup script for MRI Classification CNN
# This script helps create the necessary directory structure for the MRI dataset
#
# Usage: bash setup_mri_directories.sh

echo "=========================================="
echo "MRI Classification CNN - Directory Setup"
echo "=========================================="
echo ""

# Create main directories
echo "Creating directory structure..."

# Create data directory
mkdir -p data/mri_images
echo "✓ Created: data/mri_images/"

# Create models directory for saving trained models
mkdir -p models
echo "✓ Created: models/"

# Create plots directory for visualizations
mkdir -p plots
echo "✓ Created: plots/"

# Create example class directories
# Replace these with your actual class names
echo ""
echo "Creating example class directories..."
echo "Note: Replace 'class1', 'class2' with your actual MRI class names"
echo ""

mkdir -p data/mri_images/class1
mkdir -p data/mri_images/class2

echo "✓ Created: data/mri_images/class1/"
echo "✓ Created: data/mri_images/class2/"

echo ""
echo "=========================================="
echo "Directory structure created successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Replace 'class1' and 'class2' with your actual class names"
echo "2. Add MRI images to each class directory:"
echo "   - data/mri_images/class1/image1.jpg"
echo "   - data/mri_images/class1/image2.jpg"
echo "   - data/mri_images/class2/image1.jpg"
echo "   - data/mri_images/class2/image2.jpg"
echo "   etc."
echo ""
echo "3. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Run the training script:"
echo "   python3 mri_classification_cnn.py"
echo ""
echo "Or use the interactive examples:"
echo "   python3 mri_classification_examples.py"
echo ""
echo "For more information, see:"
echo "  - MRI_CNN_README.md (comprehensive guide)"
echo "  - MRI_CNN_QUICKSTART.md (quick start guide)"
echo "  - IMPLEMENTATION_SUMMARY.md (technical details)"
echo ""
echo "=========================================="
