"""
# 18.08.2018
# Result kaggle: 0.722
# Result Eval:
unet_128_betterDecoder
epochs 20
old load and resize
no /255

"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, tnrange
tqdm.pandas()

from keras.models import Model, load_model
from keras.layers import Conv2D, Input, ZeroPadding2D, Concatenate, concatenate, Reshape
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose, Cropping2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import Callback
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

kaggle_kernal = True

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

if kaggle_kernal:
    MODEL_DIR =       r'models'
    MODEL_NAME =      'model.h5'
    TRAIN_DIR =       r'../input/train/'
    TEST_DIR =        r'../input/test/'
    LOG_DIR =         r'logs_'
else:
    MODEL_DIR =       r'models/U-Net/unet_128//03_kaggle/'
    MODEL_NAME =      'model1.h5'
    TRAIN_DIR =       r'assets/train/'
    TEST_DIR =        r'assets/test/'
    LOG_DIR =         MODEL_DIR + 'logs/'
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

### funtions from other skripts
def load_and_resize(ids_, path, im_height, im_width, im_chan, max_n, train=True, resize_=True):
    if resize_:
        X_ = np.zeros((min(len(ids_), max_n), im_height, im_width, im_chan), dtype=np.uint8)
        Y_ = np.zeros((min(len(ids_), max_n), im_height, im_width, 1), dtype=np.bool)
    else:
        X_ = np.zeros((min(len(ids_), max_n), 101, 101, im_chan), dtype=np.uint8)
        Y_ = np.zeros((min(len(ids_), max_n), 101, 101, 1), dtype=np.bool)
    sizes_ = []
    for n, id_ in tqdm(enumerate(ids_), total=min(len(ids_), max_n)):
        if n > max_n -1:
            break
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        if resize_:
            x = resize(x, (im_height, im_width, 1), mode='constant', preserve_range=True)
        else:
            x = np.expand_dims(x,-1)
        X_[n] = x
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
            if resize_:
                mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
            else:
                mask = np.expand_dims(mask,-1)
            Y_[n] = mask
        else:
            sizes_.append([x.shape[0], x.shape[1]])
    if train:
        return X_, Y_
    else:
        return X_, sizes_

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

    def unet_128_betterDecoder(self):
        # smallest grid: 8x8
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
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

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

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
    inp, out = architectures.unet_128_betterDecoder()
    model = Model(inputs=inp, outputs=out)
    #model.summary()
    model.compile(optimizer=Adam(lr=0.001),loss=binary_crossentropy)
    return model

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
X, Y = load_and_resize(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n, resize_=False)
X_test, _ = load_and_resize(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False, resize_=False)

# normalize the image values to [0, 1] (dont devide again when loading the images into agmentation!!!!)
#X = X / 255

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
    log_path = LOG_DIR + 'fold_' + str(fold_id) + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path_console_output = log_path+'console_output.csv'

    checkpointer = ModelCheckpoint(model_path, monitor = "val_loss", save_best_only = True, verbose = 2)
    earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
    tensorBoard = TensorBoard(log_dir=log_path)
    ra_val = LossEvaluation(interval=1)
    rop = ReduceLROnPlateau(patience=2, factor=0.1, min_lr=0)
    #CSVLog = CSVLogger(log_path_console_output, separator=',', append=False)

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpointer, earlystopper, rop],#, tensorBoard, ra_val, rop, CSVLog],
                        validation_data=(X_valid,Y_valid),
                        verbose = 0)

    #print(ra_val.scores)


print('load trained models and combine')
list_of_preds = []
list_of_preds_X = []
for fold_id in range(0, min(fold_count, fold_count_calculate)):
    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    print('load model', fold_id+1, 'of', fold_count, 'from', model_path)
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
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
print('average prediction test max:', preds_test_t.max())

preds_train_t = np.zeros(list_of_preds_X[0].shape)
for fold_predict_X in list_of_preds_X:
    preds_train_t += fold_predict_X
preds_train_t /= len(list_of_preds_X)
print('average prediction train max:', preds_train_t.max())

# Threshold predictions
# preds_test_t = (preds_test_t > 0.5).astype(np.uint8)

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
