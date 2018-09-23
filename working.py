"""
# 15.07.2018
# Result kaggle: 0.722
# Result Eval:
unet_128_betterDecoder_dropout
/255
own augmentation (flip and rotate and choice)

losses: loss1 + loss2 + +loss4 + loss5 ; loss5 = focal from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65226

"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, tnrange
tqdm.pandas()

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Conv2D, Input, ZeroPadding2D, Concatenate, concatenate, Reshape
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import Callback
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras import backend as K
from keras.backend.tensorflow_backend import _to_tensor

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

kaggle_kernal = True

SEED = 12345
BATCH_SIZE = 8
EPOCHS = 50
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
fold_count = 10
fold_count_calculate = 10
max_n = 1000000
max_n_test = 1000000

if kaggle_kernal:
    MODEL_DIR =       r'models/'
    MODEL_NAME =      'model.h5'
    TRAIN_DIR =       r'../input/train/'
    TEST_DIR =        r'../input/test/'
    LOG_DIR =         r'logs/'
    TF_LOG_DIR_RUN =  LOG_DIR + 'tf_run1/'
else:
    MODEL_DIR =       r'models/U-Net/unet_128//03_kaggle/'
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

### funtions from other skripts
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
    def dice_coef(y_true, y_pred, smooth=1.0):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)
    def dice_coef_loss_border(y_true, y_pred):
        return (1 - dice_coef_border(y_true, y_pred)) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)
    def bce_dice_loss_border(y_true, y_pred):
        return bce_border(y_true, y_pred) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)
    def dice_coef_border(y_true, y_pred):
        border = get_border_mask((21, 21), y_true)
        border = K.flatten(border)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_true_f = K.tf.gather(y_true_f, K.tf.where(border > 0.5))
        y_pred_f = K.tf.gather(y_pred_f, K.tf.where(border > 0.5))
        return dice_coef(y_true_f, y_pred_f)
    def bce_border(y_true, y_pred):
        border = get_border_mask((21, 21), y_true)
        border = K.flatten(border)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_true_f = K.tf.gather(y_true_f, K.tf.where(border > 0.5))
        y_pred_f = K.tf.gather(y_pred_f, K.tf.where(border > 0.5))
        return binary_crossentropy(y_true_f, y_pred_f)
    def get_border_mask(pool_size, y_true):
        negative = 1 - y_true
        positive = y_true
        positive = K.pool2d(positive, pool_size=pool_size, padding="same")
        negative = K.pool2d(negative, pool_size=pool_size, padding="same")
        border = positive * negative
        return border
    def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
        target_tensor = y_true
        prediction_tensor = y_pred
        _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
        prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
        prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))

        if bootstrap_type == 'soft':
            bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.sigmoid(prediction_tensor)
        else:
            bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(
                K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)
        return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(
            labels=bootstrap_target_tensor, logits=prediction_tensor))
    def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
        return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice

    def focal_loss_fixed(y_true, y_pred, gamma = 2., alpha = 0.75):
        y_pred = K.clip(y_pred, 1e-6, 1 - 1e-6)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), K.ones_like(y_pred) * K.constant(alpha), K.ones_like(y_pred) * K.constant(1. - alpha))
        loss = K.mean(-1. * alpha_t * (1. - p_t)**gamma * K.log(p_t))
        return tf.exp(loss)

    loss1=binary_crossentropy(y_true,y_pred)
    loss2=dice_coef_loss_border(y_true,y_pred)
    loss3=bce_dice_loss_border(y_true,y_pred)
    loss4=dice_coef_loss_bce(y_true,y_pred, dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)
    loss5=focal_loss_fixed(y_true, y_pred, gamma = 2., alpha = 0.75)

    return loss5 + loss1 + loss2 + loss4  # loss3 has a bug with dimensions

class Architectures():

    def unet_128_betterDecoder_dropout(self):
        # smallest grid: 8x8
        DROP_FRAC = 0.5
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(DROP_FRAC)(p1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(DROP_FRAC)(p2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(DROP_FRAC)(p3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(DROP_FRAC)(p4)

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

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
        p9 = Dropout(DROP_FRAC)(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

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
            #print("mean_squared_error - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))
            self.scores.append(score)

train_ids = next(os.walk(TRAIN_DIR + "images"))[2]
test_ids = next(os.walk(TEST_DIR + "images"))[2]

# define and train the model
def build_model():
    architectures = Architectures()
    inp, out = architectures.unet_128_betterDecoder_dropout()
    model = Model(inputs=inp, outputs=out)
    #model.summary()
    model.compile(optimizer=Adam(),loss=custom_loss, metrics=["accuracy", mean_iou])
    return model

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
X, Y = load_data(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n)
X_test = load_data(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False)

# normalize the image values to [0, 1] (dont devide again when loading the images into agmentation!!!!)
X = X / 255
X_test = X_test / 255

# kfold training
fold_size = len(X) // fold_count
print("X:", X.shape)
for fold_id in range(0, min(fold_count, fold_count_calculate)):
    K.clear_session()
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

    print('image augmentation of train data (n = n*8)')
    def flip_and_transpose_augmentation(X):
        X_t = [np.transpose(x, axes=[1,0,2]) for x in X]
        X_tf = [np.fliplr(x) for x in X_t]
        X_tft = [np.transpose(x, axes=[1,0,2]) for x in X_tf]
        X_tftf = [np.fliplr(x) for x in X_tft]
        X_tftft = [np.transpose(x, axes=[1,0,2]) for x in X_tftf]
        X_tftftf = [np.fliplr(x) for x in X_tftft]
        X = np.append(X, [np.fliplr(x) for x in X], axis=0)
        X = np.append(X, X_t, axis=0)
        X = np.append(X, X_tf, axis=0)
        X = np.append(X, X_tft, axis=0)
        X = np.append(X, X_tftf, axis=0)
        X = np.append(X, X_tftft, axis=0)
        X = np.append(X, X_tftftf, axis=0)
        return X
    X_train_aug = flip_and_transpose_augmentation(X_train)
    Y_train_aug = flip_and_transpose_augmentation(Y_train)
    print('X:', X_train_aug.shape)
    print('Y:', Y_train_aug.shape)

    # just train with half of the augmentated images in this fold
    choice_ids = np.random.choice(len(X_train_aug), int(len(X_train_aug)/2))
    X_train = X_train_aug[choice_ids,:,:,:]
    Y_train = Y_train_aug[choice_ids,:,:,:]
    del X_train_aug, Y_train_aug


    model = build_model()

    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    log_path = TF_LOG_DIR_RUN + 'fold_' + str(fold_id) + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    checkpointer = ModelCheckpoint(model_path, monitor = "val_loss", save_best_only = True, verbose = 2)
    earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
    tensorBoard = TensorBoard(log_dir=log_path)
    ra_val = LossEvaluation(interval=1)
    rop = ReduceLROnPlateau(patience=2, factor=0.1, min_lr=0, verbose=1)

    #history = model.fit_generator(image_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    #steps_per_epoch=3*len(X_train)//BATCH_SIZE,
    #validation_steps=3*len(X_valid)//BATCH_SIZE,

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpointer, earlystopper, tensorBoard, rop],
                        validation_data=(X_valid, Y_valid),
                        verbose = 0)
    del X_train, Y_train
    print(history.history['mean_iou'])


print('load trained models and combine')
list_of_preds = []
list_of_preds_X = []
for fold_id in range(0, min(fold_count, fold_count_calculate)):
    K.clear_session()
    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    print('load model', fold_id+1, 'of', fold_count, 'from', model_path)
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss, 'mean_iou': mean_iou})
    print('predict')
    preds = model.predict(X_test, verbose = 0)
    preds_X = model.predict(X, verbose = 0)
    list_of_preds.append(preds)
    list_of_preds_X.append(preds_X)
    os.remove(model_path)
    print('prediction test max:', preds.max())
    print('prediction train max:', preds_X.max())

preds_test_t = np.zeros(list_of_preds[0].shape)
for fold_predict in list_of_preds:
    preds_test_t += fold_predict
preds_test_t /= len(list_of_preds)
del list_of_preds
print('average prediction test max:', preds_test_t.max())

preds_train_t = np.zeros(list_of_preds_X[0].shape)
for fold_predict_X in list_of_preds_X:
    preds_train_t += fold_predict_X
preds_train_t /= len(list_of_preds_X)
del list_of_preds_X
print('average prediction train max:', preds_train_t.max())

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

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_t[i]).astype(np.uint8)) for i,fn in tqdm(enumerate(test_ids[:min(len(test_ids), max_n_test)]))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
SUBMISSION_PATH = 'submission.csv'
if os.path.exists(SUBMISSION_PATH):
    os.remove(SUBMISSION_PATH)
sub.to_csv(SUBMISSION_PATH)

### Auswertungen
number_of_salt_images = np.array([1 for rle in sub['rle_mask'].values if len(rle)>0]).sum()
print('number_of_salt_images TEST:', number_of_salt_images, "of", len(sub), str(number_of_salt_images / len(sub)) + '%')

number_of_salt_images_X = np.array([1 for pred in preds_train_t if np.round(pred).astype(np.uint8).sum()>0]).sum()
print('number_of_salt_images TRAIN:', number_of_salt_images_X, "of", len(preds_train_t), str(number_of_salt_images_X / len(preds_train_t)) + '%')

number_of_salt_images_Y = np.array([1 for mask in Y if mask.sum()>0]).sum()
print('number_of_salt_images TRAIN MASKS:', number_of_salt_images_Y, "of", len(Y), str(number_of_salt_images_Y / len(Y)) + '%')

