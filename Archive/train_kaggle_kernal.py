"""
author: Christian Lahr
"""
SEED = 12345
BATCH_SIZE = 8
EPOCHS = 20
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
fold_count = 5
max_n = 1000000
max_n_test = 1000000

MODEL_DIR =       r'unet_128/'
MODEL_NAME =      'model1.h5'
TRAIN_DIR =       r'../input/train/'
TEST_DIR =        r'../input/test/'
LOG_DIR =         MODEL_DIR + 'logs/'

from keras.models import Model, load_model
from keras.layers import Concatenate
from keras.layers import Conv2D, Input, ZeroPadding2D
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose
from keras.layers import concatenate, Reshape, Cropping2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
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

#from load import load_and_resize
#from metrics_losses import custom_loss, mean_iou, LossEvaluation
#from architectures import Architectures

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

    def unet_128(self):
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
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

# validation function
class LossEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            score2 = mean_iou(self.y_val, y_pred)
            score3 = binary_crossentropy(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))
            print("\n MEAN_IOU - epoch: {:d} - score: {:.6f}".format(epoch + 1, score2))
            print("\n BINARY_CROSSENTROPY - epoch: {:d} - score: {:.6f}".format(epoch + 1, score3))

### \n funtions from other skripts




""" TODOs
+ loss an der Kante erhöhen
+ weitere Architekturen: Encoder ersetzen durch vortrainiertes Modell
+ weitere Architekturen: Extra features: depth, "geographische Farbe", 
+ Activation Sigmoid vs Softmax
+ Test unet bis 4x4 & bis 2x2 & bis 1x1

"""

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

train_ids = next(os.walk(TRAIN_DIR + "images"))[2]
test_ids = next(os.walk(TEST_DIR + "images"))[2]

# image augmentation
data_gen_args = dict(rescale=1./255,
                     vertical_flip=True,
                     horizontal_flip=True,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     shear_range=0.2,
                     rotation_range=90)

image_datagen = ImageDataGenerator(**data_gen_args)

# define and train the model
def build_model():
    architectures = Architectures()
    inp, out = architectures.unet_128()
    model = Model(inputs=inp, outputs=out)
    #inp1, inp2, out = architectures.unet_depth(n_features)
    #model = Model(inputs=[inp1, inp2], outputs=[out])
    #model.summary()
    model.compile(optimizer='adam',loss=custom_loss) # Achtung Gewichtung der losses in custom_loss anpassen!
    return model

earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
tensorBoard = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False)

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
X, Y = load_and_resize(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n, resize_=False)

# kfold training
fold_size = len(X) // fold_count
print("X", X.shape)
for fold_id in range(0, fold_count):
    print('train fold', fold_id+1)
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    model = build_model()
    #ra_val = LossEvaluation(validation_data=(X_valid, Y_valid), interval=1)
    #print("output loss evaluation:", ra_val)
    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    checkpointer = ModelCheckpoint(model_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
    history = model.fit_generator(image_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  epochs=EPOCHS,
                                  callbacks=[checkpointer, earlystopper], #, tensorBoard],
                                  validation_data=(X_valid,Y_valid),
                                  verbose = 1)

sys.stdout.flush()
X_test, sizes_test = load_and_resize(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False, resize_=False)
list_of_preds = []
list_of_y = []
print('load trained models and combine')
for fold_id in range(0, fold_count):
    print('load model', fold_id, 'of', fold_count)

    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
    preds = model.predict(X_test, verbose = 1)
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
if not os.path.exists(r"submissions/"):
    os.mkdir(r"submissions/")
sub.to_csv(r'submissions/submission.csv')


