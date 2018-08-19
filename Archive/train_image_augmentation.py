from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from tqdm import tqdm, tnrange
import os
import numpy as np
from skimage.transform import resize
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

from Archive.load import load_and_resize
from Archive.metrics_losses import custom_loss, mean_iou
from architectures import Architectures

SEED = 12345
BATCH_SIZE = 8
EPOCHS = 1
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
max_n = 100
max_n_test = 640
MODEL_DIR =       r'models/U-Net/model1/'
MODEL_NAME =      'model1.h5'
TRAIN_DIR =       r'assets/train/'
TEST_DIR =        r'assets/test/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

train_ids = next(os.walk(TRAIN_DIR + "images"))[2]
test_ids = next(os.walk(TEST_DIR + "images"))[2]

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

X_train, Y_train = load_and_resize(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, random_state=SEED, test_size = 0.1)

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
mask_datagen = ImageDataGenerator(**data_gen_args)

# define and train the model
architectures = Architectures()
inp, out = architectures.unet()
model = Model(inputs = inp, outputs = out)
model.summary()

model.compile(optimizer='adam',loss=custom_loss)

earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
checkpointer = ModelCheckpoint(MODEL_DIR + MODEL_NAME, monitor = "val_loss", mode = "min", verbose=1, save_best_only=True)

model.fit_generator(image_datagen.flow(X_train,
                    Y_train,
                    batch_size=BATCH_SIZE),
                    steps_per_epoch=len(X_train) / 32,
                    epochs=EPOCHS,
                    callbacks=[earlystopper, checkpointer],
                    validation_data=(X_valid,Y_valid))

sys.stdout.flush()
X_test, sizes_test = load_and_resize(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False)

# Predict on train, val and test
model = load_model(MODEL_DIR + 'model1.h5', custom_objects={'custom_loss': custom_loss})
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
#preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
#preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
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

#X_train.shape
#X_valid.shape
#Y_valid.shape
#preds_val_t.shape
print("mean_iou in validation set:", mean_iou(X_valid,Y_valid))
print("binary_crossentropy in validation set:", binary_crossentropy(X_valid,Y_valid))

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
if not os.path.exists(r"submissions/augmentation/"):
    os.mkdir(r"submissions/augmentation/")
sub.to_csv(r'submissions/augmentation/submission.csv')


