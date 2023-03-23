# COMP0053_Group6 EmoPain Challenge

**This project focuses on detecting chronic pain-related protective behavior using the EmoPain Dataset through various fusion techniques and deep learning models.**

This repository contains tutorials and code for Exploratory Data Analysis (EDA), Early (Feature-Level), Middle (Model-Level), Late (Decision-Level), and Hybrid (Middle & Late) Fusion Models, as well as cross-validation methods used in our project.

## EmoPain Dataset Website Link

[EmoPain Dataset](https://wangchongyang.ai/EmoPainChallenge2020/)

## Datasets

### 1. Original EmoPain Dataset:

**Features:**

- X, Y, Z Coordinates (1-66)
- sEMG (67-70)
- Protective behavior (merged) label (73)

**Link:** [Original EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/CoordinateData)

### 2. Energy & Angle EmoPain Dataset:

**Features:**

- Angle (1-13)
- Energy (14-26)
- sEMG (27-30)
- Protective behavior (merged) label (31)

**Link:** [New EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/AngleData)

### 3. All EmoPain Dataset with features:

**Features:**

- X, Y, Z Coordinates (1-66)
- sEMG (67-70)
- Protective behavior (merged) label (73)
- Angle (79-91)
- Energy (92-104)

**Link:** [New EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/AllData)


## Tutorials

### Core Implementation

- [Early (Feature-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Early_Fusion_Pipeline.ipynb)
  - We run the Early Fusion pipeline on Stacked-Deep-LSTM-Coordinate, Stacked-Deep-LSTM-Angle, Random Forest, and AdaBoost models.

- [Middle (Model-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Middle_Fusion_Pipeline.ipynb)
  - We run the Middle Fusion pipeline on CNN-Normal-Coordinate, CNN-BANet-Coordinate, and LSTM-BANet-Coordinate models.

- [Late (Decision-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Late_Fusion_Pipeline.ipynb)
  - We run the Late Fusion pipeline on RF-SVM Ensemble model with different ensemble strategies.

- [Hybrid (Middle & Late) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Hybrid_Fusion_Pipeline.ipynb)
  - We run the Hybrid Fusion pipeline on BI-CNN-BANet-Ensemble-Coordinate and BI-CNN-BANet-Ensemble-Angle models.

### Advance Research

- [Exploratory Data Analysis (EDA) on EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/Software/EDA_EMOPain_Pipeline.ipynb)
  - We implement EDA on the EmoPain Dataset.

- [Angle & Energy Modality Transformation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Angle_Energy_Tranformation_Pipeline.ipynb)
  - We implement X, Y, Z Coordinate to Angle & Energy Modality Transformation on the EmoPain Dataset.

- [Cross Validation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/CV_Pipeline.ipynb)
  - We run example models on Leave-P-Participant-Out Cross-Validation (LPPOCV) and Time_Series_Spilt_Cross_Validation (TSSCV)  Cross-Validation.
  
- [Preliminary_Experiments in Segmentation and Downsampling](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Preliminary_Experiments_Segmentation_Pipeline.ipynb)
  - We conducted experiments to compare the performance of the model using frame-by-frame data versus segmentation data. Moreover, we compared the performance before and after downsampling.

- [SVM Perfomance Evaluation On difference Modalities trained](https://github.com/981526092/COMP0053_Group6/blob/main/Software/SVM_Performance_Pipeline.ipynb)
  - We train the SVM model with different modalities and evaluate the performance 

- [Advanced Late (Decision-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Advance_Late_Fusion_Pipeline.ipynb)
  - We run the Advanced Late Fusion pipeline on Complex Ensemble models, utilizing confusion & metric weights with different ensemble strategies.

## Code

- [Early (Feature-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/early_model.py)
  - Contains the implementation of Early Fusion models.

- [Middle (Model-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/middle_model.py)
  - Contains the implementation of Middle Fusion models.

- [Hybrid (Middle & Late) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/hybrid_model.py)
  - Contains the implementation of Hybrid Fusion models.

- [Model Pipeline](https://github.com/981526092/COMP0053_Group6/blob/main/Software/model_utils.py)
  - Provides utility functions for models.

- [Cross Validation & Metrics](https://github.com/981526092/COMP0053_Group6/blob/main/Software/evaluation_utils.py)
  - Implements Leave-P-Participant-Out Cross-Validation (LPPOCV) and Time_Series_Spilt_Cross_Validation (TSSCV) techniques and performance metrics for model evaluation.

- [Data Loader & Preprocessor](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_utils.py)
  - Handles data loading and preprocessing.

- [Data Visualization](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_visualisation.py)
  - Provides functions for visualizing the dataset and model results.

- [Data Augmentation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_augmentation.py)
  - Implements data augmentation techniques for enhancing the dataset.


## Best Model Weights
This directory store our best model weights in hdf5 format.

Link: [Best Model Weights](https://github.com/981526092/COMP0053_Group6/tree/main/Best_Model_Weights)

## Related References

[1] Aung et al. 2014. ‘Automatic recognition of fear-avoidance behavior in chronic pain physical rehabilitation’. (https://dl.acm.org/citation.cfm?id=2686916) 

[2] Aung et al. 2016. ‘The Automatic Detection of Chronic Pain-Related Expression: Requirements, Challenges and the Multimodal EmoPain Dataset’. (https://ieeexplore.ieee.org/abstract/document/7173007) 

[3] Wang et al. 2021. ‘Chronic-Pain Protective Behavior Detection with Deep Learning’. ACM HEALTH. (https://dl.acm.org/doi/abs/10.1145/3463508). 

[4] Wang et al. 2021. ‘Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data’. IMWUT. (https://dl.acm.org/doi/abs/10.1145/3449068). 

...

