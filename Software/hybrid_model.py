from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.layers import concatenate, Conv1D, Softmax, multiply, Permute, Flatten
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed, BatchNormalization
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Concatenate, Dense, AveragePooling1D
from keras.models import Model
from scipy.io import loadmat
from keras.utils import to_categorical
from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.optimizers import *
from keras.backend import sum
import numpy as np
from scipy.stats import mode
from keras.models import Sequential
from re import U
from model_utils import crop

# Part of the Model Code is from Chongyang Wang's Work: https://github.com/EnTimeMent/BANet
def build_bi_cnn_banet_model():
    '''
    Hyper-params list:
    Epoch = 30
    Optimizer = Adam
    Learning rate = 0.0005
    batch_size = 500
    DO NOT CHANGE THIS FUNCTION
    '''
    timestep = 180   # length of an input frame
    dimension = 66   # dimension of an input frame, 66 = 22 joints by 3 xyz coordinates, the 4 coordinates of the foot are removed.
    BodyNum = 22     # number of body segments (different sensors) to consider
    SMEGNum = 2

    #Model 1: Temporal Information encoding model for BANet (keras Model API)
    singleinput = Input(shape=(180, 3,))
    lstm_units = 30


    Conv1D1 = Conv1D(60, 60, strides = 1)(singleinput)
    Batch1 = BatchNormalization()(Conv1D1)
    Relu1 = Activation('relu')(Batch1)
    Dropout1 = Dropout(0.5)(Relu1)

    Conv1D2 = Conv1D(30, 30, strides = 1)(Dropout1)
    Batch2 = BatchNormalization()(Conv1D2)
    Relu2 = Activation('tanh')(Batch2)
    Dropout2 = Dropout(0.5)(Relu2)

    # MaxPool1 = MaxPooling1D(pool_size = 1)(Dropout3)
    # Dropout3 = Dropout(0.5)(MaxPool1)


    TemporalProcessmodel = Model(inputs=[singleinput], outputs=[Dropout2])
    # TemporalProcessmodel.summary()

    singleinput_SEMG = Input(shape=(180, 2,))

    Conv1D1_SEMG = Conv1D(60, 60, strides = 1)(singleinput_SEMG)
    Batch1_SEMG = BatchNormalization()(Conv1D1_SEMG)
    Relu1_SEMG = Activation('relu')(Batch1_SEMG)
    Dropout1_SEMG = Dropout(0.5)(Relu1_SEMG)

    Conv1D2_SEMG = Conv1D(30, 30, strides = 1)(Dropout1_SEMG)
    Batch2_SEMG = BatchNormalization()(Conv1D2_SEMG)
    Relu2_SEMG = Activation('tanh')(Batch2_SEMG)
    Dropout2_SEMG = Dropout(0.5)(Relu2_SEMG)

    # MaxPool1_SEMG = MaxPooling1D(pool_size = 1)(Dropout3_SEMG)
    # Dropout3_SEMG = Dropout(0.5)(MaxPool1_SEMG)

    TemporalProcessmodel_SEMG = Model(inputs=[singleinput_SEMG], outputs=[Dropout2_SEMG])



    # Model 2: Main Structure, starting with independent temporal information encoding and attention learning
    inputs = Input(shape=(180, 70,))      # The input data is 180 timesteps by 66 features (22 joints by 3 xyz coordinates)
    # The information each body segment provides is the coordinates of each joint

    lr = crop(2, 66, 67)(inputs)
    ll = crop(2, 67, 68)(inputs)
    SEMG_lower_back = concatenate([lr, ll], axis=-1)

    SEMGout1 = TemporalProcessmodel_SEMG(SEMG_lower_back)

    TemporalAttention_SEMG1 = Conv1D(1, 1, strides=1)(SEMGout1) # Temporal Attention Module for each body segment will starts with 1 X 1 Convolution
    TemporalAttention_SEMG1 = Softmax(axis=-2, name='TemporalAttenSEMG1')(TemporalAttention_SEMG1) # You need Keras >= 2.1.3 to call Softmax as a layer
    SEMGout1 = multiply([SEMGout1, TemporalAttention_SEMG1])
    SEMGout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(SEMGout1)
    lower_back = Permute((2, 1), input_shape=(1, lstm_units))(SEMGout1)


    ur = crop(2, 68, 69)(inputs)
    ul = crop(2, 69, 70)(inputs)
    SEMG_upper_back = concatenate([ur, ul], axis=-1)

    SEMGout2 = TemporalProcessmodel_SEMG(SEMG_upper_back)

    TemporalAttention_SEMG2 = Conv1D(1, 1, strides=1)(SEMGout2) # Temporal Attention Module for each body segment will starts with 1 X 1 Convolution
    TemporalAttention_SEMG2 = Softmax(axis=-2, name='TemporalAttenSEMG2')(TemporalAttention_SEMG2) # You need Keras >= 2.1.3 to call Softmax as a layer
    SEMGout2 = multiply([SEMGout2, TemporalAttention_SEMG2])
    SEMGout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(SEMGout2)
    upper_back = Permute((2, 1), input_shape=(1, lstm_units))(SEMGout2)

    x1 = crop(2, 0, 1)(inputs)
    y1 = crop(2, 22, 23)(inputs)
    z1 = crop(2, 44, 45)(inputs)
    B1 = concatenate([x1, y1, z1], axis=-1)

    Anglefullout1 = TemporalProcessmodel(B1)

    TemporalAttention1 = Conv1D(1, 1, strides=1)(Anglefullout1) # Temporal Attention Module for each body segment will starts with 1 X 1 Convolution
    TemporalAttention1 = Softmax(axis=-2, name='TemporalAtten1')(TemporalAttention1) # You need Keras >= 2.1.3 to call Softmax as a layer
    AngleAttout1 = multiply([Anglefullout1, TemporalAttention1])
    AngleAttout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout1)
    Blast1 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout1)

    x2 = crop(2, 1, 2)(inputs)
    y2 = crop(2, 23, 24)(inputs)
    z2 = crop(2, 45, 46)(inputs)
    B2 = concatenate([x2, y2, z2], axis=-1)
    Anglefullout2 = TemporalProcessmodel(B2)

    TemporalAttention2 = Conv1D(1, 1, strides=1)(Anglefullout2)
    TemporalAttention2 = Softmax(axis=-2, name='TemporalAtten2')(TemporalAttention2)
    AngleAttout2 = multiply([Anglefullout2, TemporalAttention2])
    AngleAttout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout2)
    Blast2 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout2)

    x3 = crop(2, 2, 3)(inputs)
    y3 = crop(2, 24, 25)(inputs)
    z3 = crop(2, 46, 47)(inputs)
    B3 = concatenate([x3, y3, z3], axis=-1)
    Anglefullout3 = TemporalProcessmodel(B3)
    TemporalAttention3 = Conv1D(1, 1, strides=1)(Anglefullout3)
    TemporalAttention3 = Softmax(axis=-2, name='TemporalAtten3')(TemporalAttention3)
    AngleAttout3 = multiply([Anglefullout3, TemporalAttention3])
    AngleAttout3 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout3)
    Blast3 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout3)

    x4 = crop(2, 3, 4)(inputs)
    y4 = crop(2, 25, 26)(inputs)
    z4 = crop(2, 47, 48)(inputs)
    B4 = concatenate([x4, y4, z4], axis=-1)
    Anglefullout4 = TemporalProcessmodel(B4)
    TemporalAttention4 = Conv1D(1, 1, strides=1)(Anglefullout4)
    TemporalAttention4 = Softmax(axis=-2, name='TemporalAtten4')(TemporalAttention4)
    AngleAttout4 = multiply([Anglefullout4, TemporalAttention4])
    AngleAttout4 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout4)
    Blast4 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout4)

    x5 = crop(2, 4, 5)(inputs)
    y5 = crop(2, 26, 27)(inputs)
    z5 = crop(2, 48, 49)(inputs)
    B5 = concatenate([x5, y5, z5], axis=-1)
    Anglefullout5 = TemporalProcessmodel(B5)
    TemporalAttention5 = Conv1D(1, 1, strides=1)(Anglefullout5)
    TemporalAttention5 = Softmax(axis=-2, name='TemporalAtten5')(TemporalAttention5)
    AngleAttout5 = multiply([Anglefullout5, TemporalAttention5])
    AngleAttout5 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout5)
    Blast5 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout5)

    x6 = crop(2, 5, 6)(inputs)
    y6 = crop(2, 27, 28)(inputs)
    z6 = crop(2, 49, 50)(inputs)
    B6 = concatenate([x6, y6, z6], axis=-1)
    Anglefullout6 = TemporalProcessmodel(B6)
    TemporalAttention6 = Conv1D(1, 1, strides=1)(Anglefullout6)
    TemporalAttention6 = Softmax(axis=-2, name='TemporalAtten6')(TemporalAttention6)
    AngleAttout6 = multiply([Anglefullout6, TemporalAttention6])
    AngleAttout6 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout6)
    Blast6 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout6)

    x7 = crop(2, 6, 7)(inputs)
    y7 = crop(2, 28, 29)(inputs)
    z7 = crop(2, 50, 51)(inputs)
    B7 = concatenate([x7, y7, z7], axis=-1)
    Anglefullout7 = TemporalProcessmodel(B7)
    TemporalAttention7 = Conv1D(1, 1, strides=1)(Anglefullout7)
    TemporalAttention7 = Softmax(axis=-2, name='TemporalAtten7')(TemporalAttention7)
    AngleAttout7 = multiply([Anglefullout7, TemporalAttention7])
    AngleAttout7 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout7)
    Blast7 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout7)

    x8 = crop(2, 7, 8)(inputs)
    y8 = crop(2, 29, 30)(inputs)
    z8 = crop(2, 51, 52)(inputs)
    B8 = concatenate([x8, y8, z8], axis=-1)
    Anglefullout8 = TemporalProcessmodel(B8)
    TemporalAttention8 = Conv1D(1, 1, strides=1)(Anglefullout8)
    TemporalAttention8 = Softmax(axis=-2, name='TemporalAtten8')(TemporalAttention8)
    AngleAttout8 = multiply([Anglefullout8, TemporalAttention8])
    AngleAttout8 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout8)
    Blast8 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout8)

    x9 = crop(2, 8, 9)(inputs)
    y9 = crop(2, 30, 31)(inputs)
    z9 = crop(2, 52, 53)(inputs)
    B9 = concatenate([x9, y9, z9], axis=-1)
    Anglefullout9 = TemporalProcessmodel(B9)
    TemporalAttention9 = Conv1D(1, 1, strides=1)(Anglefullout9)
    TemporalAttention9 = Softmax(axis=-2, name='TemporalAtten9')(TemporalAttention9)
    AngleAttout9 = multiply([Anglefullout9, TemporalAttention9])
    AngleAttout9 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout9)
    Blast9 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout9)

    x10 = crop(2, 9, 10)(inputs)
    y10 = crop(2, 31, 32)(inputs)
    z10 = crop(2, 53, 54)(inputs)
    B10 = concatenate([x10, y10, z10], axis=-1)
    Anglefullout10 = TemporalProcessmodel(B10)
    TemporalAttention10 = Conv1D(1, 1, strides=1)(Anglefullout10)
    TemporalAttention10 = Softmax(axis=-2, name='TemporalAtten10')(TemporalAttention10)
    AngleAttout10 = multiply([Anglefullout10, TemporalAttention10])
    AngleAttout10 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout10)
    Blast10 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout10)

    x11 = crop(2, 10, 11)(inputs)
    y11 = crop(2, 32, 33)(inputs)
    z11 = crop(2, 54, 55)(inputs)
    B11 = concatenate([x11, y11, z11], axis=-1)
    Anglefullout11 = TemporalProcessmodel(B11)
    TemporalAttention11 = Conv1D(1, 1, strides=1)(Anglefullout11)
    TemporalAttention11 = Softmax(axis=-2, name='TemporalAtten11')(TemporalAttention11)
    AngleAttout11 = multiply([Anglefullout11, TemporalAttention11])
    AngleAttout11 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout11)
    Blast11 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout11)

    x12 = crop(2, 11, 12)(inputs)
    y12 = crop(2, 33, 34)(inputs)
    z12 = crop(2, 55, 56)(inputs)
    B12 = concatenate([x12, y12, z12], axis=-1)
    Anglefullout12 = TemporalProcessmodel(B12)
    TemporalAttention12 = Conv1D(1, 1, strides=1)(Anglefullout12)
    TemporalAttention12 = Softmax(axis=-2, name='TemporalAtten12')(TemporalAttention12)
    AngleAttout12 = multiply([Anglefullout12, TemporalAttention12])
    AngleAttout12 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout12)
    Blast12 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout12)

    x13 = crop(2, 12, 13)(inputs)
    y13 = crop(2, 34, 35)(inputs)
    z13 = crop(2, 56, 57)(inputs)
    B13 = concatenate([x13, y13, z13], axis=-1)
    Anglefullout13 = TemporalProcessmodel(B13)
    TemporalAttention13 = Conv1D(1, 1, strides=1)(Anglefullout13)
    TemporalAttention13 = Softmax(axis=-2, name='TemporalAtten13')(TemporalAttention13)
    AngleAttout13 = multiply([Anglefullout13, TemporalAttention13])
    AngleAttout13 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout13)
    Blast13 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout13)

    x14 = crop(2, 13, 14)(inputs)
    y14 = crop(2, 35, 36)(inputs)
    z14 = crop(2, 57, 58)(inputs)
    B14 = concatenate([x14, y14, z14], axis=-1)
    Anglefullout14 = TemporalProcessmodel(B14)
    TemporalAttention14 = Conv1D(1, 1, strides=1)(Anglefullout14)
    TemporalAttention14 = Softmax(axis=-2, name='TemporalAtten14')(TemporalAttention14)
    AngleAttout14 = multiply([Anglefullout14, TemporalAttention14])
    AngleAttout14 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout14)
    Blast14 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout14)

    x15 = crop(2, 14, 15)(inputs)
    y15 = crop(2, 36, 37)(inputs)
    z15 = crop(2, 58, 59)(inputs)
    B15 = concatenate([x15, y15, z15], axis=-1)
    Anglefullout15 = TemporalProcessmodel(B15)
    TemporalAttention15 = Conv1D(1, 1, strides=1)(Anglefullout15)
    TemporalAttention15 = Softmax(axis=-2, name='TemporalAtten15')(TemporalAttention15)
    AngleAttout15 = multiply([Anglefullout15, TemporalAttention15])
    AngleAttout15 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout15)
    Blast15 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout15)

    x16 = crop(2, 15, 16)(inputs)
    y16 = crop(2, 37, 38)(inputs)
    z16 = crop(2, 59, 60)(inputs)
    B16 = concatenate([x16, y16, z16], axis=-1)
    Anglefullout16 = TemporalProcessmodel(B16)
    TemporalAttention16 = Conv1D(1, 1, strides=1)(Anglefullout16)
    TemporalAttention16 = Softmax(axis=-2, name='TemporalAtten16')(TemporalAttention16)
    AngleAttout16 = multiply([Anglefullout16, TemporalAttention16])
    AngleAttout16 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout16)
    Blast16 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout16)

    x17 = crop(2, 16, 17)(inputs)
    y17 = crop(2, 38, 39)(inputs)
    z17 = crop(2, 60, 61)(inputs)
    B17 = concatenate([x17, y17, z17], axis=-1)
    Anglefullout17 = TemporalProcessmodel(B17)
    TemporalAttention17 = Conv1D(1, 1, strides=1)(Anglefullout17)
    TemporalAttention17 = Softmax(axis=-2, name='TemporalAtten17')(TemporalAttention17)
    AngleAttout17 = multiply([Anglefullout17, TemporalAttention17])
    AngleAttout17 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout17)
    Blast17 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout17)

    x18 = crop(2, 17, 18)(inputs)
    y18 = crop(2, 39, 40)(inputs)
    z18 = crop(2, 61, 62)(inputs)
    B18 = concatenate([x18, y18, z18], axis=-1)
    Anglefullout18 = TemporalProcessmodel(B18)
    TemporalAttention18 = Conv1D(1, 1, strides=1)(Anglefullout18)
    TemporalAttention18 = Softmax(axis=-2, name='TemporalAtten18')(TemporalAttention18)
    AngleAttout18 = multiply([Anglefullout18, TemporalAttention18])
    AngleAttout18 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout18)
    Blast18 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout18)

    x19 = crop(2, 18, 19)(inputs)
    y19 = crop(2, 40, 41)(inputs)
    z19 = crop(2, 62, 63)(inputs)
    B19 = concatenate([x19, y19, z19], axis=-1)
    Anglefullout19 = TemporalProcessmodel(B19)
    TemporalAttention19 = Conv1D(1, 1, strides=1)(Anglefullout19)
    TemporalAttention19 = Softmax(axis=-2, name='TemporalAtten19')(TemporalAttention19)
    AngleAttout19 = multiply([Anglefullout19, TemporalAttention19])
    AngleAttout19 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout19)
    Blast19 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout19)

    x20 = crop(2, 19, 20)(inputs)
    y20 = crop(2, 41, 42)(inputs)
    z20 = crop(2, 63, 64)(inputs)
    B20 = concatenate([x20, y20, z20], axis=-1)
    Anglefullout20 = TemporalProcessmodel(B20)
    TemporalAttention20 = Conv1D(1, 1, strides=1)(Anglefullout20)
    TemporalAttention20 = Softmax(axis=-2, name='TemporalAtten20')(TemporalAttention20)
    AngleAttout20 = multiply([Anglefullout20, TemporalAttention20])
    AngleAttout20 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout20)
    Blast20 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout20)

    x21 = crop(2, 20, 21)(inputs)
    y21 = crop(2, 42, 43)(inputs)
    z21 = crop(2, 64, 65)(inputs)
    B21 = concatenate([x21, y21, z21], axis=-1)
    Anglefullout21 = TemporalProcessmodel(B21)
    TemporalAttention21 = Conv1D(1, 1, strides=1)(Anglefullout21)
    TemporalAttention21 = Softmax(axis=-2, name='TemporalAtten21')(TemporalAttention21)
    AngleAttout21 = multiply([Anglefullout21, TemporalAttention21])
    AngleAttout21 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout21)
    Blast21 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout21)

    x22 = crop(2, 21, 22)(inputs)
    y22 = crop(2, 43, 44)(inputs)
    z22 = crop(2, 65, 66)(inputs)
    B22 = concatenate([x22, y22, z22], axis=-1)
    Anglefullout22 = TemporalProcessmodel(B22)
    TemporalAttention22 = Conv1D(1, 1, strides=1)(Anglefullout22)
    TemporalAttention22 = Softmax(axis=-2, name='TemporalAtten22')(TemporalAttention22)
    AngleAttout22 = multiply([Anglefullout22, TemporalAttention22])
    AngleAttout22 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout22)
    Blast22 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout22)

    # Model 3: Feature Concatenation for Bodily Attention Learning
    # The size of the output from each body segment is k X 1, while k is the number of LSTM hidden units
    # In early experiments, we found that it is better to keep the dimension k instead of merging them into one

    DATA = concatenate([Blast1, Blast2, Blast3, Blast4, Blast5, Blast6, Blast7, Blast8,
                        Blast9, Blast10, Blast11, Blast12, Blast13, Blast14, Blast15, Blast16,
                        Blast17, Blast18, Blast19, Blast20, Blast21, Blast22
                        ], axis=2)
    
    DATA_SEMG = concatenate([lower_back, upper_back], axis=2)

    # Bodily Attention Module
    a = Dense(BodyNum, activation='tanh')(DATA)
    a = Dense(BodyNum, activation='softmax', name='bodyattention')(a)
    attentionresult = multiply([DATA, a])
    attentionresult = Flatten()(attentionresult)

    output1 = Dense(2, activation='softmax',name='suboutput1')(attentionresult)

    # SEMG Attention Module
    a_SEMG = Dense(SMEGNum, activation='tanh')(DATA_SEMG)
    a_SEMG = Dense(SMEGNum, activation='softmax', name='semgattention')(a_SEMG)
    attentionresult_SEMG = multiply([DATA_SEMG, a_SEMG])
    attentionresult_SEMG = Flatten()(attentionresult_SEMG)

    output2 = Dense(2, activation='softmax',name='suboutput2')(attentionresult_SEMG)

    # ensemble
    output = (output1 + output2)/2
    # output = 0.2*output1 + 0.8*output2


    model = Model(inputs=[inputs], outputs=[output])
    # model.summary()

    return model

# Part of the Model Code is from Chongyang Wang's Work: https://github.com/EnTimeMent/BANet
def build_bi_cnn_banet_angle_model(first_cnn=64, second_cnn=32, lstm_units=32):

    BodyNum = 13

    #Model 1: Temporal Information encoding model
    # For angles
    singleinput = Input(shape=(180, 2,))
    lstm_units = lstm_units

    Conv1D1 = Conv1D(first_cnn, first_cnn, strides = 1)(singleinput)
    Batch1 = BatchNormalization()(Conv1D1)
    Relu1 = Activation('relu')(Batch1)
    Dropout1 = Dropout(0.5)(Relu1)

    Conv1D2 = Conv1D(second_cnn, second_cnn, strides = 1)(Dropout1)
    Batch2 = BatchNormalization()(Conv1D2)
    Relu2 = Activation('relu')(Batch2)
    Dropout2 = Dropout(0.5)(Relu2)

    TemporalProcessmodel = Model(inputs=[singleinput], outputs=[Dropout2])

    # for Semg
    singleinput_Semg = Input(shape=(180, 2,))

    Conv1D1_Semg = Conv1D(first_cnn, first_cnn, strides = 1)(singleinput_Semg)
    Batch1_Semg = BatchNormalization()(Conv1D1_Semg)
    Relu1_Semg = Activation('relu')(Batch1_Semg)
    Dropout1_Semg = Dropout(0.5)(Relu1_Semg)

    Conv1D2_Semg = Conv1D(second_cnn, second_cnn, strides = 1)(Dropout1_Semg)
    Batch2_Semg = BatchNormalization()(Conv1D2_Semg)
    Relu2_Semg = Activation('tanh')(Batch2_Semg)
    Dropout2_Semg = Dropout(0.5)(Relu2_Semg)

    TemporalProcessmodel_Semg = Model(inputs=[singleinput_Semg], outputs=[Dropout2_Semg])

    # Model 2: Main Structure, starting with independent temporal information encoding and attention learning
    inputs = Input(shape=(180, 30,))        # The input data is 180 timesteps by 30 features (13 angles + 13 energies + 4 sEMG)
                                            # The information each body segment included is the angle and energy
    
    Angle1 = crop(2, 0, 1)(inputs)
    Acc1 = crop(2, 13, 14)(inputs)
    B1 = concatenate([Angle1, Acc1], axis=-1)
    Anglefullout1 = TemporalProcessmodel(B1)
    TemporalAttention1 = Conv1D(1, 1, strides=1)(Anglefullout1) # Temporal Attention Module for each body segment will starts with 1 X 1 Convolution
    TemporalAttention1 = Softmax(axis=-2, name='TemporalAtten1')(TemporalAttention1) # You need Keras >= 2.1.3 to call Softmax as a layer
    AngleAttout1 = multiply([Anglefullout1, TemporalAttention1])
    AngleAttout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout1)
    Blast1 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout1)

    Angle2 = crop(2, 1, 2)(inputs)
    Acc2 = crop(2, 14, 15)(inputs)
    B2 = concatenate([Angle2, Acc2], axis=-1)
    Anglefullout2 = TemporalProcessmodel(B2)
    TemporalAttention2 = Conv1D(1, 1, strides=1)(Anglefullout2)
    TemporalAttention2 = Softmax(axis=-2, name='TemporalAtten2')(TemporalAttention2)
    AngleAttout2 = multiply([Anglefullout2, TemporalAttention2])
    AngleAttout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout2)
    Blast2 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout2)

    Angle3 = crop(2, 2, 3)(inputs)
    Acc3 = crop(2, 15, 16)(inputs)
    B3 = concatenate([Angle3, Acc3], axis=-1)
    Anglefullout3 = TemporalProcessmodel(B3)
    TemporalAttention3 = Conv1D(1, 1, strides=1)(Anglefullout3)
    TemporalAttention3 = Softmax(axis=-2, name='TemporalAtten3')(TemporalAttention3)
    AngleAttout3 = multiply([Anglefullout3, TemporalAttention3])
    AngleAttout3 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout3)
    Blast3 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout3)

    Angle4 = crop(2, 3, 4)(inputs)
    Acc4 = crop(2, 16, 17)(inputs)
    B4 = concatenate([Angle4, Acc4], axis=-1)
    Anglefullout4 = TemporalProcessmodel(B4)
    TemporalAttention4 = Conv1D(1, 1, strides=1)(Anglefullout4)
    TemporalAttention4 = Softmax(axis=-2, name='TemporalAtten4')(TemporalAttention4)
    AngleAttout4 = multiply([Anglefullout4, TemporalAttention4])
    AngleAttout4 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout4)
    Blast4 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout4)

    Angle5 = crop(2, 4, 5)(inputs)
    Acc5 = crop(2, 17, 18)(inputs)
    B5 = concatenate([Angle5, Acc5], axis=-1)
    Anglefullout5 = TemporalProcessmodel(B5)
    TemporalAttention5 = Conv1D(1, 1, strides=1)(Anglefullout5)
    TemporalAttention5 = Softmax(axis=-2, name='TemporalAtten5')(TemporalAttention5)
    AngleAttout5 = multiply([Anglefullout5, TemporalAttention5])
    AngleAttout5 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout5)
    Blast5 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout5)

    Angle6 = crop(2, 5, 6)(inputs)
    Acc6 = crop(2, 18, 19)(inputs)
    B6 = concatenate([Angle6, Acc6], axis=-1)
    Anglefullout6 = TemporalProcessmodel(B6)
    TemporalAttention6 = Conv1D(1, 1, strides=1)(Anglefullout6)
    TemporalAttention6 = Softmax(axis=-2, name='TemporalAtten6')(TemporalAttention6)
    AngleAttout6 = multiply([Anglefullout6, TemporalAttention6])
    AngleAttout6 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout6)
    Blast6 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout6)

    Angle7 = crop(2, 6, 7)(inputs)
    Acc7 = crop(2, 19, 20)(inputs)
    B7 = concatenate([Angle7, Acc7], axis=-1)
    Anglefullout7 = TemporalProcessmodel(B7)
    TemporalAttention7 = Conv1D(1, 1, strides=1)(Anglefullout7)
    TemporalAttention7 = Softmax(axis=-2, name='TemporalAtten7')(TemporalAttention7)
    AngleAttout7 = multiply([Anglefullout7, TemporalAttention7])
    AngleAttout7 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout7)
    Blast7 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout7)

    Angle8 = crop(2, 7, 8)(inputs)
    Acc8 = crop(2, 20, 21)(inputs)
    B8 = concatenate([Angle8, Acc8], axis=-1)
    Anglefullout8 = TemporalProcessmodel(B8)
    TemporalAttention8 = Conv1D(1, 1, strides=1)(Anglefullout8)
    TemporalAttention8 = Softmax(axis=-2, name='TemporalAtten8')(TemporalAttention8)
    AngleAttout8 = multiply([Anglefullout8, TemporalAttention8])
    AngleAttout8 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout8)
    Blast8 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout8)

    Angle9 = crop(2, 8, 9)(inputs)
    Acc9 = crop(2, 21, 22)(inputs)
    B9 = concatenate([Angle9, Acc9], axis=-1)
    Anglefullout9 = TemporalProcessmodel(B9)
    TemporalAttention9 = Conv1D(1, 1, strides=1)(Anglefullout9)
    TemporalAttention9 = Softmax(axis=-2, name='TemporalAtten9')(TemporalAttention9)
    AngleAttout9 = multiply([Anglefullout9, TemporalAttention9])
    AngleAttout9 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout9)
    Blast9 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout9)

    Angle10 = crop(2, 9, 10)(inputs)
    Acc10 = crop(2, 22, 23)(inputs)
    B10 = concatenate([Angle10, Acc10], axis=-1)
    Anglefullout10 = TemporalProcessmodel(B10)
    TemporalAttention10 = Conv1D(1, 1, strides=1)(Anglefullout10)
    TemporalAttention10 = Softmax(axis=-2, name='TemporalAtten10')(TemporalAttention10)
    AngleAttout10 = multiply([Anglefullout10, TemporalAttention10])
    AngleAttout10 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout10)
    Blast10 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout10)

    Angle11 = crop(2, 10, 11)(inputs)
    Acc11 = crop(2, 23, 24)(inputs)
    B11 = concatenate([Angle11, Acc11], axis=-1)
    Anglefullout11 = TemporalProcessmodel(B11)
    TemporalAttention11 = Conv1D(1, 1, strides=1)(Anglefullout11)
    TemporalAttention11 = Softmax(axis=-2, name='TemporalAtten11')(TemporalAttention11)
    AngleAttout11 = multiply([Anglefullout11, TemporalAttention11])
    AngleAttout11 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout11)
    Blast11 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout11)

    Angle12 = crop(2, 11, 12)(inputs)
    Acc12 = crop(2, 24, 25)(inputs)
    B12 = concatenate([Angle12, Acc12], axis=-1)
    Anglefullout12 = TemporalProcessmodel(B12)
    TemporalAttention12 = Conv1D(1, 1, strides=1)(Anglefullout12)
    TemporalAttention12 = Softmax(axis=-2, name='TemporalAtten12')(TemporalAttention12)
    AngleAttout12 = multiply([Anglefullout12, TemporalAttention12])
    AngleAttout12 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout12)
    Blast12 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout12)

    Angle13 = crop(2, 12, 13)(inputs)
    Acc13 = crop(2, 25, 26)(inputs)
    B13 = concatenate([Angle13, Acc13], axis=-1)
    Anglefullout13 = TemporalProcessmodel(B13)
    TemporalAttention13 = Conv1D(1, 1, strides=1)(Anglefullout13)
    TemporalAttention13 = Softmax(axis=-2, name='TemporalAtten13')(TemporalAttention13)
    AngleAttout13 = multiply([Anglefullout13, TemporalAttention13])
    AngleAttout13 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout13)
    Blast13 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout13)

    # Semg
    SemgBodyNum = 2     # number of body segments (different sensors) to consider
    # lower back
    Semg1 = crop(2, 26, 27)(inputs)
    Semg2 = crop(2, 27, 28)(inputs)
    B14 = concatenate([Semg1, Semg2], axis=-1)
    Semgfullout1 = TemporalProcessmodel_Semg(B14)
    TemporalAttention14 = Conv1D(1, 1, strides=1)(Semgfullout1)
    TemporalAttention14 = Softmax(axis=-2, name='TemporalAtten14')(TemporalAttention14)
    SemgAttout1 = multiply([Semgfullout1, TemporalAttention14])
    SemgAttout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(SemgAttout1)
    Blast14 = Permute((2, 1), input_shape=(1, lstm_units))(SemgAttout1)

    # upper back
    Semg3 = crop(2, 28, 29)(inputs)
    Semg4 = crop(2, 29, 30)(inputs)
    B15 = concatenate([Semg3, Semg4], axis=-1)
    Semgfullout2 = TemporalProcessmodel_Semg(B15)
    TemporalAttention15 = Conv1D(1, 1, strides=1)(Semgfullout2)
    TemporalAttention15 = Softmax(axis=-2, name='TemporalAtten15')(TemporalAttention15)
    SemgAttout2 = multiply([Semgfullout2, TemporalAttention15])
    SemgAttout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(SemgAttout2)
    Blast15 = Permute((2, 1), input_shape=(1, lstm_units))(SemgAttout2)

    # Model 3: Feature Concatenation for Bodily Attention Learning

    DATA = concatenate([Blast1, Blast2, Blast3, Blast4, Blast5, Blast6, Blast7, Blast8,
                        Blast9, Blast10, Blast11, Blast12, Blast13
                        ], axis=2)

    a = Dense(BodyNum, activation='tanh')(DATA)
    a = Dense(BodyNum, activation='softmax', name='bodyattention')(a)
    attentionresult = multiply([DATA, a])
    attentionresult = Flatten()(attentionresult)
    output1 = Dense(2, activation='softmax')(attentionresult)

    # Semg

    DATA_Semg = concatenate([Blast14, Blast15], axis=2)

    # Bodily Attention Module
    a2 = Dense(SemgBodyNum, activation='tanh')(DATA_Semg)
    a2 = Dense(SemgBodyNum, activation='softmax', name='bodyattention2')(a2)
    attentionresult2 = multiply([DATA_Semg, a2])
    attentionresult2 = Flatten()(attentionresult2)
    output2 = Dense(2, activation='softmax')(attentionresult2)

    # late fusion
    output = output1*0.4 + output2*0.6
    model = Model(inputs=inputs, outputs=output)
    return model
