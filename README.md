# Derm Lens AI

This repository contains a TensorFlow model for classifying melanoma tumors into either benign or malignant types. The model is built using deep learning techniques and is designed to assist in medical diagnosis.
The code allows the users to easily train their own models by applying modifications to the way data is processed or to the model parameters. <br><br>
The provided model, <b>derm_lens.keras</b>, has been trained on approximately 10,000 images and tested on around 2,000 images, yielding an accuracy of ~85%.

## Dataset

The dataset used for training and evaluation of the model is sourced from [Kaggle](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data?select=train). It consists of images of melanoma tumors along with their corresponding labels indicating whether they are benign or malignant.

## Notebooks

- **Data Management Notebook**: This notebook contains the code for preprocessing and organizing the dataset for training the model.
- **Model Notebook**: This notebook contains the code for building, training, and evaluating the melanoma classification model.

Please refer to the notebooks for detailed documentation on the classes and functions used in the project.

## Installation
- **If you are interested only in the model**, you can download it from the resources section.
- **If you want to train your own models using this code** you can either install the Python scripts/notebooks and place in your own environment, or clone the whole repository. <br>
   It's worth noting that the project has been created in a Docker container, so using the project as it is requires <b>Docker</b>.

To clone the repo, run this command:
```bash
git clone https://github.com/AlexandruCostea/derm_lens_ai.git
```

## Credits
Dataset: Melanoma Cancer Dataset by Bhavesh Mittal on Kaggle.

## Disclaimer
 This model is intended to assist medical professionals in diagnosis and should not be solely relied upon for making medical decisions. Always consult with qualified healthcare providers for accurate diagnosis and treatment.
