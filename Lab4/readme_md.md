# MLOps Lab 4 - Delivery Time Prediction

This repository contains a modularized machine learning pipeline for predicting food delivery times. The original Jupyter Notebook has been refactored into a reusable Python package (`delivery/`) and standalone command-line scripts.

## Package Modules (`delivery/`)

* **data.py**: Loads the simulated delivery times dataset (or generates it if it does not already exist)[cite: 1].
* **features.py**: Separates the dataset into input features (distance, prep time, traffic, rain) and the target delivery time, while providing an average speed calculation utility.
* **model.py**: Trains a Linear Regression model on the dataset, evaluates its mean absolute error, and saves the trained artifact to disk.
* **validate.py**: Validates incoming order parameters to ensure values are physically realistic (e.g., strictly positive distance and prep time, valid traffic categories) before allowing a prediction.
* **__init__.py**: Exposes the necessary functions from each module to bind the directory together as a unified Python package.

## Command-Line Scripts

* **train.py**: Executes the data loading, feature extraction, and model training steps to generate the saved model file.
* **predict.py**: Accepts new order metrics via the command line, validates the input, and outputs the model's estimated delivery time.

## Usage 

To train the model, run:
```bash
python train.py
## AI Disclosure
In accordance with course guidelines, I am disclosing the use of an AI assistant in the completion of this assignment. I used AI for debugging in these assignment and clearing my minor doubts. 