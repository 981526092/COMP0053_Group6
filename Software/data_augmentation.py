import numpy as np
from sklearn.utils import shuffle

def jitter(X, y, std_dev):
    noise = np.random.normal(0, std_dev, X.shape)
    return X + noise, y

def crop_new(X, y, probability):
    mask = np.random.rand(*X.shape) < probability
    return X * (1 - mask), y

def augment_data(X, y):
    X_jittered_1, y_jittered_1 = jitter(X, y, 0.05)
    X_jittered_2, y_jittered_2 = jitter(X, y, 0.1)

    X_cropped_1, y_cropped_1 = crop_new(X, y, 0.05)
    X_cropped_2, y_cropped_2 = crop_new(X, y, 0.1)

    X_augmented = np.concatenate((X, X_jittered_1, X_jittered_2, X_cropped_1, X_cropped_2), axis=0)
    y_augmented = np.concatenate((y, y_jittered_1, y_jittered_2, y_cropped_1, y_cropped_2), axis=0)

    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=0)

    return X_augmented, y_augmented