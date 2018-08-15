import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

### Parameters
im_width = 128
im_height = 128
im_chan = 1

path_train = r'assets/train/'
path_test = r'assets/test/'
path_model = r'models/U-Net/model1/'
if not os.path.exists(path_model):
    os.mkdir(path_model)

batch_size = 8
epochs = 30
patience = 5
### Parameters

train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]


# plot some data
ids = ['1f1cc6b3a4.png', '5b7c160d0d.png', '6c40978ddf.png', '7dfdf6eeb8.png', '7e5a6e5013.png']
idxs = np.random.randint(0, len(train_ids), size=5)
ids = np.array(train_ids)[idxs]
ids

plt.figure(figsize=(20, 10))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img('assets/train/images/' + img_name)
    img_mask = load_img('assets/train/masks/' + img_name )

    plt.subplot(1, 2 * (1 + len(ids)), q * 2 - 1)
    plt.imshow(img)
    plt.subplot(1, 2 * (1 + len(ids)), q * 2)
    plt.imshow(img_mask)
plt.show()

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
def load_and_resize(ids_, path, im_height, im_width, im_chan, train=True):
    X_ = np.zeros((len(ids_), im_height, im_width, im_chan), dtype=np.uint8)
    Y_ = np.zeros((len(ids_), im_height, im_width, 1), dtype=np.bool)
    sizes_ = []
    for n, id_ in tqdm(enumerate(ids_), total=len(ids_)):
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        X_[n] = x
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
            Y_[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
        else:
            sizes_.append([x.shape[0], x.shape[1]])
    if train:
        return X_, Y_
    else:
        return X_, sizes_

X_train, Y_train = load_and_resize(train_ids, path_train, im_height, im_width, im_chan)

print('Done!')


# Check if training data looks all right
ix = random.randint(0, len(train_ids))
plt.subplot(1, 2, 1)
plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
plt.show()
plt.subplot(1, 2, 2)
tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()


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


# Build U-Net model
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
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

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

# Train
earlystopper = EarlyStopping(patience=patience, verbose=1)
checkpointer = ModelCheckpoint(path_model + 'model1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs,
                    callbacks=[earlystopper, checkpointer])

# Get and resize test images
print('Getting and resizing test images ... ')
sys.stdout.flush()
X_test, sizes_test = load_and_resize(test_ids, path_test, im_height, im_width, im_chan, train=False)

print('Done!')

# Predict on train, val and test
model = load_model(path_model + 'model1.h5', custom_objects={'mean_iou': mean_iou})
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

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
plt.show()
tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()
tmp = np.squeeze(preds_train_t[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()

#Prepare Submission
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

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm(enumerate(test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(path_model + 'submission.csv')