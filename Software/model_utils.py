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
    Ensembles two sets of predictions using different strategies.
    Parameters:
    prediction1 (np.ndarray): The first set of predictions.
    prediction2 (np.ndarray): The second set of predictions.
    strategy (str): The ensembling strategy to use. Options are 'average', 'product', 'max', or 'rule'.
    weights (list): A list of weights to use for each set of predictions. If not given, defaults to equal weights.
    rule (function): A callable rule function to use for rule-based ensembling.

    Returns:
        np.ndarray: The ensembled predictions.
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

    # Crop a tensor along a given dimension between start and end indices
    # This function is used to crop the tensor inputs to a model
    # Code is from marc-moreaux on Github page: https://github.com/keras-team/keras/issues/890)
    def crop(dimension, start, end):
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

    # Plot the training history of a model
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

    # Learning rate schedule for reducing learning rate during training
    def schedule(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.95

    def model_pipeline(model, X_train, y_train, X_valid, y_valid, epoch=50, save_model=False,print_results=True):
        """
        Trains and evaluates a model on the given training and validation data, with options to save and print results.
        Parameters:
            model (keras.Model): The model to be trained and evaluated.
            X_train (numpy.ndarray): The training input data as a numpy array.
            y_train (numpy.ndarray): The training target data as a numpy array.
            X_valid (numpy.ndarray): The validation input data as a numpy array.
            y_valid (numpy.ndarray): The validation target data as a numpy array.
            epoch (int): The number of epochs to train the model for (default is 50).
            save_model (bool): A flag to indicate whether to save the trained model (default is False).
            print_results (bool): A flag to indicate whether to print the evaluation results (default is True).
        Returns:
            tuple: A tuple containing the predicted labels, true labels, and training history.
        """
        initial_learning_rate = 0.0005
        optimizer = Adam(lr=initial_learning_rate)

        model.compile(loss=losses.BinaryFocalCrossentropy(), optimizer=optimizer, metrics=[metrics.binary_accuracy])

        # Initialize the LearningRateScheduler callback
        lr_scheduler = LearningRateScheduler(schedule, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # Train the model with callbacks
        H = model.fit(
            X_train,
            y_train,
            batch_size=500,
            epochs=epoch,
            validation_data=(X_valid, y_valid),
            shuffle=False,
            callbacks=[lr_scheduler],
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

