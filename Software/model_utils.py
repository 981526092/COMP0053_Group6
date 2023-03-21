import numpy as np
from keras import losses
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras import metrics
from keras.layers import Lambda
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def ensemble_predictions(prediction1, prediction2, strategy='average', weights=None, rule=None):
    """
    Function to ensemble two predictions using various methods.

    :param prediction1: array-like object containing the first set of predictions (n x 2)
    :param prediction2: array-like object containing the second set of predictions (n x 2)
    :param strategy: string specifying the ensembling strategy, options are 'average', 'product', 'max', 'rule', default is 'average'
    :param weights: list or array-like object containing the weights for each set of predictions, default is None
    :param rule: callable function for rule-based ensembling, only used if strategy='rule', default is None
    :return: list containing the ensembled predictions
    """
    prediction1, prediction2 = np.array(prediction1), np.array(prediction2)

    if not weights:
        # If no weights are given, assume equal weights for both predictions
        weights = [0.5, 0.5]

    if prediction1.shape != prediction2.shape or len(weights) != 2:
        raise ValueError("Both predictions must have the same shape, and weights must have a length of 2.")

    if strategy == 'average':
        ensemble = prediction1 * weights[0] + prediction2 * weights[1]

    elif strategy == 'product':
        ensemble = prediction1 * prediction2

    elif strategy == 'max':
        ensemble = np.maximum(prediction1, prediction2)

    elif strategy == 'rule':
        if not rule or not callable(rule):
            raise ValueError("A callable rule function is required for rule-based ensembling.")
        ensemble = np.array([rule(pred1, pred2) for pred1, pred2 in zip(prediction1, prediction2)])

    else:
        raise ValueError("Invalid ensembling strategy. Options are 'average', 'product', 'max', or 'rule'.")

    ensemble = np.argmax(ensemble, axis=1)

    return ensemble

def crop(dimension, start, end):
    # Thanks to marc-moreaux on Github page:https://github.com/keras-team/keras/issues/890 who created this beautiful and sufficient function: )
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95

def model_pipeline(model, X_train, y_train, X_valid, y_valid, epoch=50, save_model=False,print_results=True):
    initial_learning_rate = 0.0005
    optimizer = Adam(lr=initial_learning_rate)

    model.compile(loss=losses.BinaryFocalCrossentropy(), optimizer=optimizer, metrics=[metrics.binary_accuracy])

    # Initialize the LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model with both callbacks
    H = model.fit(
        X_train,
        y_train,
        batch_size=500,
        epochs=epoch,
        validation_data=(X_valid, y_valid),
        shuffle=False,
        callbacks=[lr_scheduler,early_stopping],
    )
    if save_model:
        model.save_weights('model' + '.hdf5')
        model.load_weights('model' + '.hdf5')

    y_predraw = model.predict(X_valid)
    y_pred = np.argmax(y_predraw, axis=1)
    y_true = np.argmax(y_valid, axis=1)

    if print_results:
        # Compute and display the additional evaluation metrics
        print("Classification report:")
        print(classification_report(y_true, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    return y_pred, y_true, H

