from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
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

SEED = 12345
BATCH_SIZE = 8
EPOCHS = 1
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
TRAIN_IMAGE_DIR = r'assets/train/images/'
TRAIN_MASK_DIR =  r'assets/train/masks/'
MODEL_DIR =       r'models/U-Net/model1/'
MODEL_NAME =      'model1.h5'
TEST_IMAGE_DIR =  r'assets/test/images/'
TRAIN_DIR =       r'assets/train/'
TEST_DIR =        r'assets/test/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

train_ids = next(os.walk(TRAIN_DIR + "images"))[2]
test_ids = next(os.walk(TEST_DIR + "images"))[2]

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
def load_and_resize(ids_, path, im_height, im_width, im_chan, max_n, train=True):
    X_ = np.zeros((min(len(ids_), max_n), im_height, im_width, im_chan), dtype=np.uint8)
    Y_ = np.zeros((min(len(ids_), max_n), im_height, im_width, 1), dtype=np.bool)
    sizes_ = []
    for n, id_ in tqdm(enumerate(ids_), total=min(len(ids_), max_n)):
        if n > max_n -1:
            break
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        x = resize(x, (im_height, im_width, 1), mode='constant', preserve_range=True)
        X_[n] = x
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
            Y_[n] = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
        else:
            sizes_.append([x.shape[0], x.shape[1]])
    if train:
        return X_, Y_
    else:
        return X_, sizes_

X_train, Y_train = load_and_resize(train_ids, TRAIN_DIR, heigth, width, im_chan, 100)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, random_state=SEED, test_size = 0.1)


data_gen_args = dict(rescale=1./255,
                     vertical_flip=True,
                     horizontal_flip=True,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     shear_range=0.2,
                     rotation_range=90)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

def conv_block(num_layers,inp,units,kernel):
    x = inp
    for l in range(num_layers):
        x = Conv2D(units, kernel_size=kernel, padding='SAME',activation='relu')(x)
    return x

inp = Input(shape=(128,128,1))
cnn1 = conv_block(4,inp,32,3)
cnn2 = conv_block(4,inp,24,5)
cnn3 = conv_block(4,inp,16,7)
concat = Concatenate()([cnn1,cnn2,cnn3])
d1 = Conv2D(16,1, activation='relu')(concat)
out = Conv2D(1,1, activation='sigmoid')(d1)

model = Model(inputs = inp, outputs = out)
model.summary()

# Define IoU metric
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
    a2 = 1
    return a1*loss1 + a2*K.log(loss2)

model.compile(optimizer='adam',loss=custom_loss)

early_stop = EarlyStopping(patience=5)
check_point = ModelCheckpoint('model.hdf5',save_best_only=True)

earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
checkpointer = ModelCheckpoint(MODEL_DIR + MODEL_NAME, verbose=1, save_best_only=True)

model.fit_generator(image_datagen.flow(X_train,
                    Y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(X_train) / 32,
                    epochs=EPOCHS,
                    callbacks=[earlystopper, checkpointer],
                    validation_data=(X_valid,Y_valid))

sys.stdout.flush()
max_n_test = 640
X_test, sizes_test = load_and_resize(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False)

# Predict on train, val and test
model = load_model(MODEL_DIR + 'model1.h5', custom_objects={'custom_loss': custom_loss})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks # resize back to original size
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

print(preds_test_upsampled[0].shape)

"""
test_fns = os.listdir(TEST_IMAGE_DIR)
X_test = [np.array(cv2.imread(TEST_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(test_fns)]
X_test = np.array(X_test)/255
X_test = np.expand_dims(X_test,axis=3)

pred = model.predict(X_test, verbose = True)
"""

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

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm(enumerate(test_ids[:max_n_test]))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')


