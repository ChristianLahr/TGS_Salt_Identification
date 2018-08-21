from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tqdm import tqdm
import os
import numpy as np
import sys
import pandas as pd

from Archive.load import load_and_resize
from Archive.metrics_losses import mean_iou
from architectures import Architectures

""" TODOs
+ loss an der Kante erhöhen
+ weitere Architekturen: Encoder ersetzen durch vortrainiertes Modell
+ weitere Architekturen: Extra features: depth, "geographische Farbe", 
+ ist uint8 der richtige datentyp für die submission???
"""
SEED = 12345
BATCH_SIZE = 8
EPOCHS = 2
PARTIENCE = 5
width = 128
heigth = 128
im_chan = 1
fold_count = 2
max_n = 100
max_n_test = 200
MODEL_DIR =       r'models/U-Net/unet_128/03_kaggle/'
MODEL_NAME =      'model1.h5'
TRAIN_DIR =       r'assets/train/'
TEST_DIR =        r'assets/test/'
LOG_DIR =         MODEL_DIR + 'logs/'

""" TODOs
+ loss an der Kante erhöhen
+ weitere Architekturen: Encoder ersetzen durch vortrainiertes Modell
+ weitere Architekturen: Extra features: depth, "geographische Farbe", 
+ Activation Sigmoid vs Softmax
+ Test unet bis 4x4 & bis 2x2 & bis 1x1

"""

train_ids = next(os.walk(TRAIN_DIR + "images"))[2]
test_ids = next(os.walk(TEST_DIR + "images"))[2]

# image augmentation
#data_gen_args = dict(rescale=1./255,
#                     vertical_flip=True,
#                     horizontal_flip=True,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     zoom_range=0.2,
#                     shear_range=0.2,
#                     rotation_range=90)

#image_datagen = ImageDataGenerator(**data_gen_args)

# define and train the model
def build_model():
    architectures = Architectures()
    inp, out = architectures.unet_128()
    model = Model(inputs=inp, outputs=out)
    #inp1, inp2, out = architectures.unet_depth(n_features)
    #model = Model(inputs=[inp1, inp2], outputs=[out])
    #model.summary()
    model.compile(optimizer='adam',loss=mean_iou)
    return model

earlystopper = EarlyStopping(patience=PARTIENCE, verbose=1)

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
X, Y = load_and_resize(train_ids, TRAIN_DIR, heigth, width, im_chan, max_n, resize_=False)
X_test, _ = load_and_resize(test_ids, TEST_DIR, heigth, width, im_chan, max_n_test, train=False, resize_=False)

# kfold training
list_of_preds = []
list_of_y = []
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
    #print("Validation loss:", ra_val)
    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
    checkpointer = ModelCheckpoint(model_path, monitor = "val_loss", save_best_only = True, verbose = 1)
    log_path = LOG_DIR + 'fold_' + str(fold_id) + '/'
    tensorBoard = TensorBoard(log_dir=log_path, histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False)
    #history = model.fit_generator(image_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    history = model.fit(X_train, Y_train,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=[checkpointer, earlystopper, tensorBoard],
                          validation_data=(X_valid,Y_valid),
                          verbose = 1)
    print('predict test with this fold model')
    preds = model.predict(X_test, verbose = 1)
    list_of_preds.append(preds)

#print('load trained models and combine')
#list_of_preds = []
#list_of_y = []
#for fold_id in range(0, fold_count):
#    print('load model', fold_id, 'of', fold_count)
#    model_path = MODEL_DIR + MODEL_NAME[:-3] + str(fold_id) + '.h5'
#    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
#    preds = model.predict(X_test, verbose = 1)
#    list_of_preds.append(preds)

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


