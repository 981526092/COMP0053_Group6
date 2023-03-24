import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import LeavePOut, StratifiedKFold, TimeSeriesSplit
from keras.models import clone_model
from model_utils import model_pipeline
from data_utils import load_data
from sklearn.metrics import f1_score

# Perform cross-validation and return the scores and best model
def LeavePOut_CV(model,train_participant_num,p = 15,epoch = 50):
    # Choose cross-validation method based on the input parameter
    cv_method = LeavePOut(p=p)
    scores = []
    best_f1 = 0
    best_model = None
    # Iterate through the cross-validation splits
    for train, valid in cv_method.split(train_participant_num):
        train_participants = [train_participant_num[i] for i in train]
        valid_participants = [train_participant_num[i] for i in valid]
        x_train,y_train = load_data(train_participants,data_set="train")
        x_valid,y_valid = load_data(valid_participants,data_set="train")
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

def TimeSeriesSplit_CV(model, X_train, y_train, n_splits=5, epoch=50):
    cv_method = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    best_f1 = 0
    best_model = None
    # Iterate through the cross-validation splits
    for train_idx, valid_idx in cv_method.split(X_train):
        x_train, x_valid = X_train[train_idx], X_train[valid_idx]
        y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]
        # Use the model_pipeline function to train and evaluate the model
        y_pred, y_true, H = model_pipeline(model, x_train, y_train_fold, x_valid, y_valid_fold, save_model=False, print_results=False, epoch=epoch)
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

# Compute binary classification metrics
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
