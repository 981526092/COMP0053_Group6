import numpy as np
from keras import losses
from keras.callbacks import LearningRateScheduler
from keras import metrics
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


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
    initial_learning_rate = 0.005
    optimizer = Adam(lr=initial_learning_rate)

    model.compile(loss=losses.BinaryFocalCrossentropy(), optimizer=optimizer, metrics=[metrics.binary_accuracy])

    # Initialize the LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)

    # Train the model with both callbacks
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

