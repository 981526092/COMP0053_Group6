from keras import Sequential
from keras.layers import LSTM, Dropout, Dense

# Create a stacked LSTM model with dropout layers
def stacked_lstm(input_shape,num_classes):
    """
    Creates a stacked LSTM model with dropout layers.
    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.

    Returns:
        keras.Sequential: A sequential model containing stacked LSTM and dropout layers.
    """
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model