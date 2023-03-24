# COMP0053_Group6 EmoPain Challenge

**This project focuses on detecting chronic pain-related protective behavior using the EmoPain Dataset through various fusion techniques and deep learning models.**

This repository contains tutorials and code for Exploratory Data Analysis (EDA), Early (Feature-Level), Middle (Model-Level), Late (Decision-Level), and Hybrid (Middle & Late) Fusion Models, as well as cross-validation methods used in our project.

## EmoPain Dataset Website Link

[EmoPain Dataset](https://wangchongyang.ai/EmoPainChallenge2020/)

## Model Results (Sorted By F1-Score)

| Model                                 | F1 Score | Recall | Precision | Accuracy |
|---------------------------------------|----------|--------|-----------|----------|
| BI-CNN-BANet-Ensemble-Angle           | 0.76     | 0.83   | 0.72      | 0.93     |
| BI-CNN-BANet-Ensemble-Coordinate      | 0.73     | 0.83   | 0.69      | 0.92     |
| Adaboost-SVM-Ensemble                 | 0.73     | 0.72   | 0.74      | 0.94     |
| RF-SVM-Ensemble                       | 0.67     | 0.67   | 0.67      | 0.93     |
| CNN-Normal-Coordinate                 | 0.63     | 0.6    | 0.69      | 0.94     |
| Adaboost                              | 0.58     | 0.58   | 0.58      | 0.9      |
| CNN-BANet-Coordinate                  | 0.57     | 0.55   | 0.71      | 0.94     |
| Stacked Deep LSTM-Angle               | 0.58     | 0.66   | 0.57      | 0.84     |
| Adaboost                              | 0.58     | 0.58   | 0.58      | 0.9      |
| SVM(only sEMG data)                   | 0.72     | 0.72   | 0.72      | 0.94     |
| SVM(only coordinate data)             | 0.49     | 0.5    | 0.47      | 0.94     |
| SVM                                   | 0.49     | 0.5    | 0.47      | 0.94     |
| Stacked-Deep-LSTM-Coordinate          | 0.48     | 0.5    | 0.47      | 0.94     |
| LSTM-BANet-Coordinate                 | 0.48     | 0.5    | 0.47      | 0.94     |
| Random Forest(Baseline)               | 0.49     | 0.54   | 0.51      | 0.76     |


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

- [Hold Out Test on Validation Datasets of Best Models Weights ](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Best_Model_Weights_Pipeline.ipynb)
  - We show the hold out test results of our best model weight.

- [Early (Feature-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Early_Fusion_Pipeline.ipynb)
  - We run the Early Fusion pipeline on Stacked-Deep-LSTM-Coordinate, Stacked-Deep-LSTM-Angle, Random Forest, and AdaBoost models.

- [Middle (Model-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Middle_Fusion_Pipeline.ipynb)
  - We run the Middle Fusion pipeline on CNN-Normal-Coordinate, CNN-BANet-Coordinate, and LSTM-BANet-Coordinate models.

- [Late (Decision-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Late_Fusion_Pipeline.ipynb)
  - We run the Late Fusion pipeline on RF-SVM Ensemble model with different ensemble strategies.

- [Hybrid (Middle & Late) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Hybrid_Fusion_Pipeline.ipynb)
  - We run the Hybrid Fusion pipeline on BI-CNN-BANet-Ensemble-Coordinate and BI-CNN-BANet-Ensemble-Angle models.
  

### Other & Advanced Implementation

- [Exploratory Data Analysis (EDA) on EmoPain Dataset](https://github.com/981526092/COMP0053_Group6/blob/main/Software/EDA_EmoPain_Pipeline.ipynb)
  - We implement EDA on the EmoPain Dataset.

- [Angle & Energy Modality Transformation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Angle_Energy_Tranformation_Pipeline.ipynb)
  - We implement X, Y, Z Coordinate to Angle & Energy Modality Transformation on the EmoPain Dataset.

- [Cross Validation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/CV_Pipeline.ipynb)
  - We run example models on Leave-P-Participant-Out Cross-Validation (LPPOCV) and Time_Series_Spilt_Cross_Validation (TSSCV)  Cross-Validation.
  
- [Preliminary_Experiments in Segmentation and Downsampling](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Preliminary_Experiments_Pipeline.ipynb)
  - We conducted experiments to compare the performance of the model using frame-by-frame data versus segmentation data. Moreover, we compared the performance before and after downsampling.

- [SVM Perfomance Evaluation On difference Modalities trained](https://github.com/981526092/COMP0053_Group6/blob/main/Software/SVM_Performance_Pipeline.ipynb)
  - We train the SVM model with different modalities and evaluate the performance 

- [Advanced Late (Decision-Level) Fusion Models](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Advance_Late_Fusion_Pipeline.ipynb)
  - We run the Advanced Late Fusion pipeline on Complex Ensemble models, utilizing confusion & metric weights with different ensemble strategies.
  
- [Model Results Visualisation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/Result_Visualisation_Pipeline.ipynb)
  - We visualise the models results.
  
...

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

- [Data & Results Visualization](https://github.com/981526092/COMP0053_Group6/blob/main/Software/visualisation_utils.py)
  - Provides functions for visualizing the dataset and model results.

- [Data Augmentation](https://github.com/981526092/COMP0053_Group6/blob/main/Software/data_augmentation.py)
  - Implements data augmentation techniques for enhancing the dataset.


## Best Model Weights
This directory store our best model weights in hdf5 format.

Link: [Best Model Weights](https://github.com/981526092/COMP0053_Group6/tree/main/Best_Model_Weights)

## Related References

### Main References

[1] Aung et al. 2014. ‘Automatic recognition of fear-avoidance behavior in chronic pain physical rehabilitation’. (https://dl.acm.org/citation.cfm?id=2686916) 

[2] Aung et al. 2016. ‘The Automatic Detection of Chronic Pain-Related Expression: Requirements, Challenges and the Multimodal EmoPain Dataset’. (https://ieeexplore.ieee.org/abstract/document/7173007) 

[3] Wang et al. 2021. ‘Chronic-Pain Protective Behavior Detection with Deep Learning’. ACM HEALTH. (https://dl.acm.org/doi/abs/10.1145/3463508). 

[4] Wang et al. 2021. ‘Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data’. IMWUT. (https://dl.acm.org/doi/abs/10.1145/3449068). 

### Other References

[1] Ayata, Deger, Yusuf Yaslan, and Mustafa E Kamasak. 2018. ‘Emotion based music recommendation system using wearable physiological sensors’. IEEE Transactions on Consumer Electronics 64 (2): 196--203.

[2] Russell, James A. 1980. ‘A circumplex model of affect’. Journal of personality and social psychology 39 (6): 1161.

[3] Lin, Chingshun, Mingyu Liu, Weiwei Hsiung, and Jhihsiang Jhang. 2016. ‘Music emotion recognition based on two-level support vector classification’. In 2016 International conference on machine learning and cybernetics (ICMLC), 1:375--389. IEEE.

[4] Jung, Tzyy-Ping, Terrence J Sejnowski, and others. 2019. ‘Utilizing deep learning towards multi-modal bio-sensing and vision-based affective computing’. IEEE Transactions on Affective Computing 13 (1): 96--107.

[5] Puri, Colin, Leslie Olson, Ioannis Pavlidis, James Levine, and Justin Starren. 2005. ‘StressCam: non-contact measurement of users' emotional states through thermal imaging’. In CHI'05 extended abstracts on Human factors in computing systems, 1725--1728.

[6] Cicone, Antonio, and Hau-Tieng Wu. 2017. ‘How nonlinear-type time-frequency analysis can help in sensing instantaneous heart rate and instantaneous respiratory rate from photoplethysmography in a reliable way’. Frontiers in physiology 8: 701.

[7] Yu, Zitong, Xiaobai Li, and Guoying Zhao. 2021. ‘Facial-video-based physiological signal measurement: Recent advances and affective applications’. IEEE Signal Processing Magazine 38 (6): 50--58.

[8] Ekman, Paul. 1993. ‘Facial expression and emotion’. American psychologist 48 (4): 384.

[9] Baltru{\v{s}}aitis, Tadas, Peter Robinson, and Louis-Philippe Morency. 2016. ‘Openface: an open source facial behavior analysis toolkit’. In 2016 IEEE winter conference on applications of computer vision (WACV), 1--10. IEEE.

[10] Chawla, Nitesh V, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. 2002. ‘SMOTE: synthetic minority over-sampling technique’. Journal of artificial intelligence research 16: 321--357.

[11] He, Haibo, Yang Bai, Edwardo A Garcia, and Shutao Li. 2008. ‘ADASYN: Adaptive synthetic sampling approach for imbalanced learning’. In 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence), 1322--1328. IEEE.

...

