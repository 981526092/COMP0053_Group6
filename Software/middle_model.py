from keras import Sequential, Input, Model
from keras.layers import LSTM, Dropout, Dense, Flatten, MaxPooling1D, Activation, BatchNormalization, Conv1D, \
    Concatenate
from model_utils import crop

def cnn_normal(input_shape,num_classes):
    input_data = Input(input_shape)

    # Separating the inputs using the crop function
    input1 = crop(2, 0, 66)(input_data)
    input2 = crop(2, 66, 70)(input_data)

    # First model
    x = Conv1D(32, 31, padding='same')(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 31, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(32, 31, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 31, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    # Second model
    x1 = Conv1D(32, 31, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(32, 31, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(2)(x1)

    x1 = Conv1D(32, 31, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(32, 31, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(2)(x1)

    # Fusion
    x2 = Concatenate(axis=-1)([x, x1])
    x2 = Conv1D(64, 1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D()(x2)
    x2 = Flatten()(x2)

    out = Dense(num_classes)(x2)

    model = Model(inputs=input_data, outputs=out)
    return model
