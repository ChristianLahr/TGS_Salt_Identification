"""
# 18.08.2018
# Result kaggle: 0.708
# Result Eval:

unet_pretrainedEncoder
compiletime for 1 of 5 folds: 2:40 h

"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, tnrange
tqdm.pandas()

from keras.models import Model, load_model
from keras.layers import Conv2D, Input, ZeroPadding2D, Concatenate, concatenate, Reshape
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose, Cropping2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import Callback
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

SEED = 12345
BATCH_SIZE = 8
EPOCHS = 30
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
fold_count = 5
fold_count_calculate = 1
max_n = 1000000
max_n_test = 1000000

kaggle_kernal = True

if kaggle_kernal:
    MODEL_DIR =       r'models/'
    MODEL_NAME =      'model.h5'
    TRAIN_DIR =       r'../input/tgs-salt-identification-challenge/train/'
    TEST_DIR =        r'../input/tgs-salt-identification-challenge/test/'
    LOG_DIR =         r'logs/'
    TF_LOG_DIR_RUN =  LOG_DIR + 'tf_run1/'
else:
    MODEL_DIR =       r'models/U-Net/pretrained_encoder/'
    MODEL_NAME =      'model1.h5'
    TRAIN_DIR =       r'assets/train/'
    TEST_DIR =        r'assets/test/'
    LOG_DIR =         MODEL_DIR + 'logs/'
    TF_LOG_DIR_RUN =  LOG_DIR + 'tf_run1/'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.exists(TF_LOG_DIR_RUN):
    os.mkdir(TF_LOG_DIR_RUN)
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


def load_data(ids_, path, im_height, im_width, im_chan, max_n, train=True):
    X_ = np.zeros((min(len(ids_), max_n), 101, 101, im_chan), dtype=np.uint8)
    Y_ = np.zeros((min(len(ids_), max_n), 101, 101, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(ids_), total=min(len(ids_), max_n)):
        if n > max_n -1:
            break
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        x = np.expand_dims(x,-1)
        X_[n] = x
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
            mask = np.expand_dims(mask,-1)
            Y_[n] = mask
    if train:
        return X_, Y_
    else:
        return X_

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def custom_loss(y_true, y_pred):
    loss1=binary_crossentropy(y_true,y_pred)
    loss2=mean_iou(y_true,y_pred)
    a1 = 1
    a2 = 0
    return a1*loss1 + a2*K.log(loss2)

class Architectures():

    def unet_128_InceptionResNetV2_encoder(self, trainable=0):
        input_img = Input((101, 101, 3), name='img')
        resized = ZeroPadding2D(padding=((99,99), (99,99)))(input_img) #pad tp shape=(None, 299,299,3))
        # use better padding method here!!!

        if kaggle_kernal:
            resNet = InceptionResNetV2(include_top=False, weights=None, input_tensor=resized, input_shape=None, pooling=None)
            resNet.load_weights(r'../input/inception_resnet_v2_wights.h5/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
        else:
            resNet = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=resized, input_shape=None, pooling=None)

        layers = resNet.layers
        if not trainable:
            resNet.trainable = False
            for l in layers:
                l.trainable = False

        resNetDim8_anzapfen = resNet.output
        resNetDim8_fully = Conv2D(256, (1, 1), activation='relu') (resNetDim8_anzapfen)

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

        return input_img, resNetDim8_fully, resNetDim16_fully, resNetDim32_fully, resNetDim64_fully, resNetDim128_fully

    def unet_128_decoder(self, c5, c4, c3, c2, c1):

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

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return cropped

class LossEvaluation(Callback):
    def __init__(self, interval=1):
        super(Callback, self).__init__()
        self.interval = interval

    def on_train_begin(self, logs={}):
        self.scores = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.model.validation_data[0], verbose=0)
            score = binary_crossentropy(self.model.validation_data[1], y_pred)
            self.scores.append(score)

train_ids = next(os.walk(TRAIN_DIR + "images"))[2]
test_ids = next(os.walk(TEST_DIR + "images"))[2]

# define and train the model
def build_model():
    architectures = Architectures()
    inp, c5, c4, c3, c2, c1 = architectures.unet_128_InceptionResNetV2_encoder(trainable=1)
    out = architectures.unet_128_decoder(c5, c4, c3, c2, c1)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    model.compile(optimizer=Adam(),loss=binary_crossentropy)
    return model

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
X, Y = load_data(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n)
X_test = load_data(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False)

#resnet needs three channels
X = np.stack((X[:,:,:,0],)*3, -1)
X_test = np.stack((X_test[:,:,:,0],)*3, -1)

# kfold training
fold_size = len(X) // fold_count
print("X:", X.shape)
for fold_id in range(0, min(fold_count, fold_count_calculate)):
    print('train fold', fold_id+1)
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X)

    print(fold_start, fold_end)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    model = build_model()

    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    log_path = TF_LOG_DIR_RUN + 'fold_' + str(fold_id) + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path_console_output = log_path+'console_output.csv'

    checkpointer = ModelCheckpoint(model_path, monitor = "val_loss", save_best_only = True, verbose = 2)
    earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
    tensorBoard = TensorBoard(log_dir=log_path)
    ra_val = LossEvaluation(interval=1)
    rop = ReduceLROnPlateau(patience=2, factor=0.1, min_lr=0)

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpointer, earlystopper, rop, tensorBoard],
                        validation_data=(X_valid,Y_valid),
                        verbose = 0)

    #print(ra_val.scores)

print('load trained models and combine')
list_of_preds = []
list_of_y = []
for fold_id in range(0, min(fold_count, fold_count_calculate)):
    print('load model', fold_id, 'of', fold_count, 'from', model_path)
    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
    print('predict')
    preds = model.predict(X_test, verbose = 0)
    list_of_preds.append(preds)

test_predicts = np.zeros(list_of_preds[0].shape)
for fold_predict in list_of_preds:
    test_predicts += fold_predict
test_predicts /= len(list_of_preds)

# Threshold predictions
preds_test_t = (test_predicts > 0.5).astype(np.uint8)

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_t[i])) for i,fn in tqdm(enumerate(test_ids[:min(len(test_ids), max_n_test)]))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(MODEL_DIR + 'submission.csv')