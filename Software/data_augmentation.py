import numpy as np
from sklearn.utils import shuffle

# Add Gaussian noise to the input data
def jitter(X, y, std_dev):
    noise = np.random.normal(0, std_dev, X.shape)
    return X + noise, y

# Randomly set input values to zero with a given probability
def crop_new(X, y, probability):
    mask = np.random.rand(*X.shape) < probability
    return X * (1 - mask), y

# Augment the input data using jittering and cropping
def augment_data(X, y):
    # Apply jitter with 0.05 and 0.1 magnitude
    X_jittered_1, y_jittered_1 = jitter(X, y, 0.05)
    X_jittered_2, y_jittered_2 = jitter(X, y, 0.1)
    # Apply cropping with 0.05 and 0.1 magnitude
    X_cropped_1, y_cropped_1 = crop_new(X, y, 0.05)
    X_cropped_2, y_cropped_2 = crop_new(X, y, 0.1)
    # Concatenate original and augmented data
    X_augmented = np.concatenate((X, X_jittered_1, X_jittered_2, X_cropped_1, X_cropped_2), axis=0)
    y_augmented = np.concatenate((y, y_jittered_1, y_jittered_2, y_cropped_1, y_cropped_2), axis=0)
    # Shuffle data
    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=0)

    return X_augmented, y_augmented