import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

TRAIN_IMAGE_DIR = r'assets/train/images/'
TRAIN_MASK_DIR = r'assets/train/masks/'
TEST_IMAGE_DIR = r'assets/test/images/'
MODEL_DIR = r'models/U-Net/model1/'

train_fns = os.listdir(TRAIN_IMAGE_DIR)

X = [np.array(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_fns)]
X = np.array(X)/255
X = np.expand_dims(X,axis=3)

y = [np.array(cv2.imread(TRAIN_MASK_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_fns)]
y = np.array(y)/255
y = np.expand_dims(y,axis=3)

X_train, X_valid, y_train, y_valid = train_test_split(X,y, random_state=23, test_size = 0.2)

from keras.models import Model
from keras.layers import Conv2D, Input, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
def conv_block(num_layers,inp,units,kernel):
    x = inp
    for l in range(num_layers):
        x = Conv2D(units, kernel_size=kernel, padding='SAME',activation='relu')(x)
    return x


inp = Input(shape=(101,101,1))
cnn1 = conv_block(4,inp,32,3)
cnn2 = conv_block(4,inp,24,5)
cnn3 = conv_block(4,inp,16,7)
concat = Concatenate()([cnn1,cnn2,cnn3])
d1 = Conv2D(16,1, activation='relu')(concat)
out = Conv2D(1,1, activation='sigmoid')(d1)

model = Model(inputs = inp, outputs = out)
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy')

early_stop = EarlyStopping(patience=5)
check_point = ModelCheckpoint('model.hdf5',save_best_only=True)
model.fit(X_train,y_train, epochs=50, validation_data=(X_valid,y_valid), callbacks=[early_stop,check_point],batch_size=128)









test_fns = os.listdir(TEST_IMAGE_DIR)
X_test = [np.array(cv2.imread(TEST_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(test_fns)]
X_test = np.array(X_test)/255
X_test = np.expand_dims(X_test,axis=3)

pred = model.predict(X_test, verbose = True)


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

pred_dict = {fn[:-4]:RLenc(np.round(pred[i,:,:,0])) for i,fn in tqdm(enumerate(test_fns))}


import pandas as pd



sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')




