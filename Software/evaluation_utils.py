import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import LeavePOut
from keras.models import clone_model
from model_utils import model_pipeline
from data_utils import load_data
def LPSO_cross_validation(model, train_participant_num,valid_participant_num,p = 15,epoch = 50):
    loo = LeavePOut(p=p)
    scores = []
    best_f1 = 0
    best_model = None
    X_validation, y_validation = load_data(valid_participant_num, data_set="validation")

    for train, valid in loo.split(train_participant_num):
        train_participants = [train_participant_num[i] for i in train]
        valid_participants = [train_participant_num[i] for i in valid]

        x_train,y_train = load_data(train_participants,data_set="train")
        x_valid,y_valid = load_data(valid_participants,data_set="train")

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