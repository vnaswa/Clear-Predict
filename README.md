# ClearPredict

ClearPredict is a Vision Transformer (ViT) based image classification project designed to classify images from the CIFAR-10 dataset. It leverages the power of Vision Transformers to achieve high classification accuracy on various object categories. This repository contains code to train, evaluate, and make predictions using a Vision Transformer model.

## Features

- **Vision Transformer Model**: Implements a Vision Transformer for image classification.
- **Data Augmentation**: Applies various data augmentation techniques to improve model robustness.
- **Custom Patch Extraction**: Extracts image patches and applies positional encoding for transformer input.
- **Model Checkpointing**: Saves and restores model weights for better training management.
- **Evaluation**: Provides accuracy and top-5 accuracy metrics for model evaluation.

## Installation

To run the code, you'll need to have the following Python packages installed:

- `numpy`
- `tensorflow`
- `tensorflow-addons`
- `matplotlib`

You can install the required packages using pip:

```bash
pip install numpy tensorflow tensorflow-addons matplotlib
```

## Usage

### Training the Model

1. Clone the repository:

    ```bash
    git clone https://github.com/vnaswa/Clear-Predict.git
    cd ClearPredict
    ```

2. Run the training script:

    ```bash
    python train.py
    ```

   This will start the training process, and the model will be saved to the specified checkpoint directory.

### Making Predictions

To make predictions on images:

1. Import the necessary modules and load the trained model:

    ```python
    from clear_predict import img_predict, create_vit_classifier
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the model
    vit_classifier = create_vit_classifier()
    vit_classifier.load_weights('./tmp/checkpoint')
    ```

2. Prepare your image data and use the `img_predict` function to get predictions:

    ```python
    # Load your image data
    x_test = ...  # Replace with your image data

    # Make a prediction
    index = 1
    plt.imshow(x_test[index])
    prediction = img_predict(x_test[index], vit_classifier)
    print(prediction)
    ```

## Code Overview

- **`train.py`**: Script to train and evaluate the Vision Transformer model.
- **`clear_predict.py`**: Contains the Vision Transformer model architecture, data augmentation, and image patch extraction logic.
- **`utils.py`**: Helper functions for image preprocessing and predictions.

## Data

The CIFAR-10 dataset is used for training and evaluation. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

