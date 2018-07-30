from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tqdm import tqdm

def load_and_resize(ids_, path, im_height, im_width, im_chan, max_n, train=True, resize_=True):
    X_ = np.zeros((min(len(ids_), max_n), im_height, im_width, im_chan), dtype=np.uint8)
    Y_ = np.zeros((min(len(ids_), max_n), im_height, im_width, 1), dtype=np.bool)
    sizes_ = []
    for n, id_ in tqdm(enumerate(ids_), total=min(len(ids_), max_n)):
        if n > max_n -1:
            break
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,1]
        if resize_:
            x = resize(x, (im_height, im_width, 1), mode='constant', preserve_range=True)
        X_[n] = x
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
            if resize_:
                mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
            Y_[n] = mask
        else:
            sizes_.append([x.shape[0], x.shape[1]])
    if train:
        return X_, Y_
    else:
        return X_, sizes_
