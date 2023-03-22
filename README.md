# COMP0053_Group6 EmoPain Challenge

**This project task is Chronic-Pain Protective Behaviour Detection for the EmoPain Dataset.**


This repository contains the tutorials and code for Exploratory Data Analysis(EDA), Early (Feature-Level), Middle (Model-Level), Late (Decision-Level), and Hybrid (Middle & Late) Fusion Models, as well as cross-validation methods used in our project. 

## EmoPain Dataset Website Link

https://wangchongyang.ai/EmoPainChallenge2020/

## Original EmoPain Dataset:
**Features:**

X,Y,Z Coordinates / sEMG / Protective behavior (merged) label 

1-66 / 67-70 / 73

**Link:** [Original EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/CoordinateData)

## Energy & Angle EmoPain Dataset:
**Features:**

Angle / Energy / sEMG / Protective behavior (merged) label

1-13 / 14-26 / 27-30 / 31

**Link:** [New EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/AngleData)

## ALL EmoPain Dataset:

**Features:**

X,Y,Z Coordinates / sEMG / Protective behavior (merged) label / Angle / Energy

1-66 / 67-70 / 73 / 79-91 / 92-104

**Link:** [New EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/AllData)

## Tutorials

- [Exploratory Data Analysis(EDA) on EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/Software/EDA_EMOPain_Pipeline.ipynb)

We implement EDA on EmoPain Dataset.

- [Angle & Energy Modality Tranformation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Angle_Energy_Tranformation_Pipeline.ipynb)

We implement X,Y,Z Coordinates to Angle & Energy Modality Tranformation on EmoPain Dataset.

- [Early (Feature-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Early_Fusion_Pipeline.ipynb)

We run Early Fusion pipeline on Stacked-Deep-LSTM-Coordinate, Stacked-Deep-LSTM-Angle, Random Forest and Adaboost models.

- [Middle (Model-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Middle_Fusion_Pipeline.ipynb)

We run Middle Fusion pipeline on CNN-Normal-Coordinate, CNN-BANet-Coordinate, and LSTM-BANet-Coordinate models.

- [Late (Decision-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Late_Fusion_Pipeline.ipynb)

We run Late Fusion pipeline on RF-SVM Ensemble model with difference ensemble strategies.

- [Advance Late (Decision-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Advance_Late_Fusion_Pipeline.ipynb)

We run Advance Late Fusion pipeline on Complex Ensembele models with the use of confusion & Metrics weights with difference ensemble strategies.

- [Hybrid (Middle & Late) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Hybrid_Fusion_Pipeline.ipynb)

We run Hybrid Fusion pipeline on BI-CNN-BANet-Ensemble-Coordinate and BI-CNN-BANet-Ensemble-Angle models.

- [Cross Validation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/CV_Pipeline.ipynb)

We run example model on Leave-P-Out and Leave-One-Out Cross-Validation.

## Code

- [Early (Feature-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/early_model.py)
- [Middle (Model-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/middle_model.py)
- [Hybrid (Middle & Late) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/hybrid_model.py)
- [Model Pipeline](https://github.com/981526092/COMP0053_Group6/blob/main/Software/model_utils.py)
- [Cross Validation & Metrics](https://github.com/981526092/COMP0053_Group6/blob/main/Software/evaluation_utils.py)
- [Data Loader & Preprocessor](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_utils.py)
- [Data visualisation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_visualisation.py)
- [Data augmentation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_augmentation.py)

## Best Model Weights
This directory store our best model weights in hdf5 format.

Link: [Best Model Weights](https://github.com/981526092/COMP0053_Group6/tree/main/Best_Model_Weights)

## Related References

[1] Aung et al. 2014. ‘Automatic recognition of fear-avoidance behavior in chronic pain physical rehabilitation’. (https://dl.acm.org/citation.cfm?id=2686916) 

[2] Aung et al. 2016. ‘The Automatic Detection of Chronic Pain-Related Expression: Requirements, Challenges and the Multimodal EmoPain Dataset’. (https://ieeexplore.ieee.org/abstract/document/7173007) 

[3] Wang et al. 2021. ‘Chronic-Pain Protective Behavior Detection with Deep Learning’. ACM HEALTH. (https://dl.acm.org/doi/abs/10.1145/3463508). 

[4] Wang et al. 2021. ‘Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data’. IMWUT. (https://dl.acm.org/doi/abs/10.1145/3449068). 

...

