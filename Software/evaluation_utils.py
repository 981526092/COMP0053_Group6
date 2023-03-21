import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import LeavePOut, GroupKFold, LeaveOneOut, StratifiedKFold, TimeSeriesSplit
from keras.models import clone_model
from model_utils import model_pipeline
from data_utils import load_data

# Perform cross-validation and return the scores and best model
def cross_validation(model,train_participant_num,valid_participant_num,cv_selection= 'LeavePOut',p = 15,epoch = 50):

    # Choose cross-validation method based on the input parameter
    if cv_selection == 'LeavePOut':
        cv_method = LeavePOut(p=p)
    elif cv_selection == 'LeaveOneOut':
        cv_method = LeaveOneOut()
    else:
        raise ValueError("Invalid cross-validation selection. Choose from 'LeavePOut', 'LeaveOneOut'.")

    scores = []
    best_f1 = 0
    best_model = None
    X_validation, y_validation = load_data(valid_participant_num, data_set="validation")

    # Iterate through the cross-validation splits
    for train, valid in cv_method.split(train_participant_num):
        train_participants = [train_participant_num[i] for i in train]
        valid_participants = [train_participant_num[i] for i in valid]

        x_train,y_train = load_data(train_participants,data_set="train")
        x_valid,y_valid = load_data(valid_participants,data_set="train")

        # Concatenate the validation data to the training data
        x_valid = np.concatenate((X_validation,x_valid),axis=0)
        y_valid = np.concatenate((y_validation,y_valid),axis=0)

        # Use the model_pipeline function to train and evaluate the model
        y_pred, y_true, H = model_pipeline(model, x_train, y_train, x_valid, y_valid, save_model=False, print_results=False,epoch=epoch)

        score = f1_score(y_true, y_pred, average='macro')

        # Store the score
        scores.append(score)

        # Check if this is the best model so far
        if score > best_f1:
            best_f1 = score
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())

    return np.array(scores), best_model

# Compute the confusion matrix for binary classification
def confusion_matrix_binary(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm

# Compute binary classification metrics from scratch
def binary_classification_metrics_from_scratch(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = np.trace(cm) / np.sum(cm)

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    return cm, accuracy, macro_precision, macro_recall, macro_f1

# Print a classification report with various metrics
def print_classification_report(y_true, y_pred):
    cm, accuracy, macro_precision, macro_recall, macro_f1 = binary_classification_metrics_from_scratch(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print()
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Macro Precision: {macro_precision:.2f}')
    print(f'Macro Recall: {macro_recall:.2f}')
    print(f'Macro F1 Score: {macro_f1:.2f}')
