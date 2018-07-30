from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Concatenate
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

from load import load_and_resize
from metrics_losses import custom_loss, mean_iou
from architectures import Architectures
from keras.losses import binary_crossentropy
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

SEED = 12345
BATCH_SIZE = 8
EPOCHS = 1
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
fold_count = 10
max_n = 100
max_n_test = 640
MODEL_DIR =       r'models/U-Net/model1/'
MODEL_NAME =      'model1.h5'
TRAIN_DIR =       r'assets/train/'
TEST_DIR =        r'assets/test/'
LOG_DIR =         r'models/U-Net/model1/logs/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

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

# valifation function
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

# define and train the model
def build_model():
    architectures = Architectures()
    inp, out = architectures.unet()
    model = Model(inputs=inp, outputs=out)
    #inp1, inp2, out = architectures.unet_depth(n_features)
    #model = Model(inputs=[inp1, inp2], outputs=[out])
    model.summary()
    model.compile(optimizer='adam',loss=custom_loss)
    return model

earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)
checkpointer = ModelCheckpoint(MODEL_DIR + MODEL_NAME, verbose=1, save_best_only=True)
tensorBoard = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False)

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
X, Y = load_and_resize(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n, resize_=False)

# kfold training
fold_size = len(X) // fold_count
for fold_id in range(0, fold_count):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    model = build_model()#rnn_units = 64, de_units = 64, lr = 1e-3)
    ra_val = LossEvaluation(validation_data=(X_valid, Y_valid), interval=1)
    print("output loss evaluation:", ra_val)
    model_path = MODEL_DIR + MODEL_NAME[:-3] + fold_id + '.h5'
    check_point = ModelCheckpoint(model_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
    history = model.fit_generator(image_datagen.flow(X_train, Y_train,
                        batch_size=BATCH_SIZE),
                        steps_per_epoch=len(X_train) / BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[earlystopper, checkpointer, tensorBoard],
                        validation_data=(X_valid,Y_valid),
                        verbose = 1)

sys.stdout.flush()
X_test, sizes_test = load_and_resize(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False, resize_=False)
list_of_preds = []
list_of_vals = []
list_of_y = []
fold_size = len(X) // fold_count
for fold_id in range(0, fold_count):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    model_path = MODEL_DIR + MODEL_NAME[:-3] + fold_id + '.h5'
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
    preds = model.predict(X_test, verbose = 1)
    list_of_preds.append(preds)
    vals = model.predict(X_valid, verbose = 1)
    list_of_vals.append(vals)
    list_of_y.append(Y_valid)

test_predicts = np.zeros(list_of_preds[0].shape)
for fold_predict in list_of_preds:
    test_predicts += fold_predict
test_predicts /= len(list_of_preds)

val_predicts = np.zeros(list_of_vals[0].shape)
for fold_vals in list_of_vals:
    val_predicts += fold_vals
val_predicts /= len(list_of_vals)

# Threshold predictions
preds_val_t = (val_predicts > 0.5).astype(np.uint8)
preds_test_t = (test_predicts > 0.5).astype(np.uint8)

# Create list of upsampled test masks # resize back to original size
preds_test_upsampled = []
for i in tnrange(len(test_predicts)):
    preds_test_upsampled.append(resize(np.squeeze(test_predicts[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))
# achtung noch falsches resize ausbauen --> besser in model/keras einbauen

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


