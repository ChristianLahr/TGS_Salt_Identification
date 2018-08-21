from keras.models import Model, load_model
from keras.layers import Concatenate
from keras.layers import Conv2D, Input, ZeroPadding2D
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose
from keras.layers import concatenate, Reshape, Cropping2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm, tnrange
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
import pandas as pd
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras.losses import binary_crossentropy

SEED = 12345
BATCH_SIZE = 8
EPOCHS = 5
PARTIENCE = 4
width = 128
heigth = 128
im_chan = 1
fold_count = 1
max_n = 500
max_n_test = 100

MODEL_DIR =       r'models/U-Net/pretrained_encoder/'
MODEL_NAME =      'model.h5'
TRAIN_DIR =       r'assets/train/'
TEST_DIR =        r'assets/test/'
LOG_DIR =         MODEL_DIR + 'logs/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


class Architectures():

    def unet_pretrainedEncoder(self):
        input_img = Input((101, 101, 3), name='img')
        resized = ZeroPadding2D(padding=((99,99), (99,99)))(input_img) #pad tp shape=(None, 299,299,3))
        # use better padding method here!!!

        resNet = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=resized, input_shape=None, pooling=None)
        resNet.trainable = False
        layers = resNet.layers
        for l in layers:
            l.trainable = False

        resNetDim16_anzapfen_1 = layers[605].output
        resNetDim16_anzapfen_2 = layers[606].output
        resNetDim16_anzapfen_3 = layers[607].output
        resNetDim16_anzapfen = concatenate([resNetDim16_anzapfen_1, resNetDim16_anzapfen_2, resNetDim16_anzapfen_3])
        resNetDim16_fully = Conv2D(128, (1, 1), activation='relu') (resNetDim16_anzapfen)
        resNetDim16_fully = Cropping2D(cropping=((0,1), (0,1)))(resNetDim16_fully)

        resNetDim32_anzapfen = layers[267].output
        resNetDim32_fully = Conv2D(64, (1, 1), activation='relu') (resNetDim32_anzapfen)
        resNetDim32_fully = Cropping2D(cropping=((1,2), (1,2)))(resNetDim32_fully)

        resNetDim64_anzapfen = layers[17].output
        resNetDim64_fully = Conv2D(64, (1, 1), activation='relu') (resNetDim64_anzapfen)
        resNetDim64_fully = Cropping2D(cropping=((3,4), (3,4)))(resNetDim64_fully)

        resNetDim128_anzapfen = layers[10].output
        resNetDim128_fully = Conv2D(32, (1, 1), activation='relu') (resNetDim128_anzapfen)
        resNetDim128_fully = Cropping2D(cropping=((9,10), (9,10)))(resNetDim128_fully)

        resNetDim8_anzapfen = resNet.output
        resNetDim8_fully = Conv2D(256, (1, 1), activation='relu') (resNetDim8_anzapfen)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (resNetDim8_fully)
        u6 = concatenate([u6, resNetDim16_fully])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(32, (1, 1), activation='relu', padding='same') (c6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, resNetDim32_fully])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(16, (1, 1), activation='relu', padding='same') (c7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, resNetDim64_fully])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(8, (1, 1), activation='relu', padding='same') (c8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, resNetDim128_fully], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped, resNet.layers

    def unet_128_betterDecoder(self):
        # smallest grid: 8x8
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='c1') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(32, (1, 1), activation='relu', padding='same') (c6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(16, (1, 1), activation='relu', padding='same') (c7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(8, (1, 1), activation='relu', padding='same') (c8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        model = Model(inputs=input_img, outputs=c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (model.output)
        u9 = concatenate([u9, model.get_layer('c1').output], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

def build_model():
    architectures = Architectures()
    inp, out, layers = architectures.unet_pretrainedEncoder()
    model = Model(inputs=inp, outputs=out)
    #inp1, inp2, out = architectures.unet_depth(n_features)
    #model = Model(inputs=[inp1, inp2], outputs=[out])
    model.summary()
    model.compile(optimizer='adam',loss=binary_crossentropy)
    return model, layers

def build_model2():
    architectures = Architectures()
    inp, out = architectures.unet_128_betterDecoder()
    model = Model(inputs=inp, outputs=out)
    #inp1, inp2, out = architectures.unet_depth(n_features)
    #model = Model(inputs=[inp1, inp2], outputs=[out])
    model.summary()
    model.compile(optimizer='adam',loss=binary_crossentropy)
    return model


#model, layers = build_model()
model2 = build_model2()

#model_path = './' + MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
#history = model.fit_generator(image_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

#history = model.fit(X_train, Y_train,
#                    batch_size=BATCH_SIZE,
#                    epochs=EPOCHS,
#                    verbose = 1)


for i, l in enumerate(layers):
    if l.name == 'activation_409':
        print(i)