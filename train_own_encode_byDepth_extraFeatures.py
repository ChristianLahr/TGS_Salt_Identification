"""
# 19.08.2018

train an decoder with depth information
loss:     19102
val_loss: 22683

"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, tnrange
tqdm.pandas()

from keras.models import Model, load_model
from keras.layers import Conv2D, Input, ZeroPadding2D, concatenate, Concatenate, Flatten, Dense
from keras.layers import MaxPooling2D, Conv2DTranspose, Reshape, RepeatVector, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.losses import mean_squared_error

from sklearn.model_selection import train_test_split

SEED = 12345
BATCH_SIZE =8
EPOCHS = 30
width = 128
heigth = 128
PREPROCESS = True

kaggle_kernal = True

if kaggle_kernal:
    MODEL_DIR =       r'models/'
    MODEL_NAME =      'model.h5'
    INPUT_DIR =       r'../input/'
    TF_LOG_DIR =      r'tf_logs/'
    TF_LOG_DIR_RUN =  TF_LOG_DIR + 'run1/'
    CONSOLE_LOG_DIR = r'console_logs/'
else:
    MODEL_DIR =       r'models/U-Net/pretrained_encoder/'
    MODEL_NAME =      'model.h5'
    INPUT_DIR =       r'assets/'
    TF_LOG_DIR =      MODEL_DIR + 'logs/'
    CONSOLE_LOG_DIR = MODEL_DIR + 'console_logs/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)
if not os.path.exists(CONSOLE_LOG_DIR):
    os.mkdir(CONSOLE_LOG_DIR)

def load_test_and_train(ids_test, ids_train, path, im_height, im_width):
    X_test_pd = pd.DataFrame()
    X_train_pd = pd.DataFrame()
    X_test_images = np.zeros( (len(ids_test), 101, 101, 1), dtype=np.uint8)
    X_train_images = np.zeros( (len(ids_train), 101, 101, 1), dtype=np.uint8)
    if kaggle_kernal:
        Y = pd.read_csv(path + 'depths.csv')
    else:
        Y = pd.read_csv(path + 'train/depths.csv')

    for n, id_ in tqdm(enumerate(ids_test), total=len(ids_test)):
        img = load_img(path + 'test/images/' + id_)
        x = img_to_array(img)[:,:,1]
        x = np.expand_dims(x,-1)
        X_test_images[n] = x
    X_test_pd['id'] = ids_test
    X_test_pd['id'] = X_test_pd['id'].progress_apply(lambda x: x[:-4])
    X_test_pd['images'] = list(X_test_images)

    for n, id_ in tqdm(enumerate(ids_train), total=len(ids_train)):
        img = load_img(path + 'train//images/' + id_)
        x = img_to_array(img)[:,:,1]
        x = np.expand_dims(x,-1)
        X_train_images[n] = x
    X_train_pd['id'] = ids_train
    X_train_pd['id'] = X_train_pd['id'].progress_apply(lambda x: x[:-4])
    X_train_pd['images'] = list(X_train_images)

    X_train = pd.merge(X_train_pd, Y, how='left', on='id')
    X_test = pd.merge(X_test_pd, Y, how='left', on='id')

    XY = pd.concat([X_train, X_test])

    return XY

print('load the data incl. depth')
train_ids = next(os.walk(INPUT_DIR + 'train/images'))[2]
test_ids = next(os.walk(INPUT_DIR + 'test/images'))[2]

sys.stdout.flush()
XY = load_test_and_train(test_ids, train_ids, INPUT_DIR, width, heigth)
X = np.array(list(XY['images']))
Y = XY['z'].values

def preprocess_image(x):
    x = x/255
    x = x-0.5
    x = x*2
    return x

if PREPROCESS:
    X = preprocess_image(X)

class Architectures():

    def unet_128_betterDecoder_with_Depth(self):
        # smallest grid: 8x8
        n_features = 1
        input_img = Input((101, 101, 1), name='img')
        input_features = Input((n_features, ), name='feat')

        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='c1') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='c2') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='c3') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='c4') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        f_repeat = RepeatVector(8*8)(input_features)
        f_conv = Reshape((8, 8, n_features))(f_repeat)
        p4_feat = concatenate([p4, f_conv], -1)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='c5') (c5)

        fully = Conv2D(1, (1, 1), activation='relu') (c5)
        flat = Flatten() (fully)
        depth = Dense(1, activation=None) (flat)

        return input_img, input_features, depth

class LossEvaluation(Callback):
    def __init__(self, interval=1):
        super(Callback, self).__init__()
        self.interval = interval

    def on_train_begin(self, logs={}):
        self.scores = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.model.validation_data[0], verbose=0)
            score = mean_squared_error(self.model.validation_data[1], y_pred)
            #print("mean_squared_error - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))
            self.scores.append(score)

def build_model():
    architectures = Architectures()
    inp_img, inp_feat, out = architectures.unet_128_betterDecoder_with_Depth()
    model = Model(inputs=[inp_img, inp_feat], outputs=out)
    model.summary()
    model.compile(optimizer=Adam(lr=0.001),loss=mean_squared_error, metrics=[mean_squared_error])
    return model

print('build and train the model')
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=12345)

# normalize depth
def normalize_train_valid(feature_train, feture_valid):
    mean = feature_train.mean(axis=0, keepdims=True)
    std = feature_train.std(axis=0, keepdims=True)
    print('mean:', mean, 'std:', std)
    feature_train = feature_train - mean
    feature_train = feature_train / std
    feture_valid = feture_valid - mean
    feture_valid = feture_valid / std
    return feature_train, feture_valid
# normalize with train mean and train std
Y_train_normaized, Y_valid_normalized = normalize_train_valid(Y_train, Y_valid)

model = build_model()

CONSOLE_LOG_path = CONSOLE_LOG_DIR + 'console_output.csv'

checkpointer = ModelCheckpoint(MODEL_DIR + MODEL_NAME, monitor = "val_loss", save_best_only = True, verbose = 2)
earlystopper = EarlyStopping(patience=5, verbose=1)
tensorBoard = TensorBoard(log_dir=TF_LOG_DIR_RUN)
ra_val = LossEvaluation(interval=1)
rop = ReduceLROnPlateau(patience=2, factor=0.1, min_lr=0, verbose=1)
CSVLog = CSVLogger(CONSOLE_LOG_path, separator=',', append=False)

history = model.fit({'img': X_train, 'feat': Y_train_normaized}, Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[checkpointer, earlystopper, rop, tensorBoard], #, ra_val, CSVLog],
                    validation_data=({'img': X_valid, 'feat': Y_valid_normalized}, Y_valid),
                    verbose = 1)
print('loss:', history.history["loss"][-1], 'val_loss:', history.history["val_loss"][-1])
#print(ra_val.scores)

"""
17600/17600 [==============================] - 38s 2ms/step - loss: 24995.2848 - val_loss: 24862.8620
Epoch 00010: val_loss improved from 25027.96325 to 24862.86199, saving model to models/model.h5
Epoch 11/20
"""