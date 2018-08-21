from skimage.transform import resize, rescale
from keras.preprocessing.image import img_to_array, load_img

#look at images:
img = load_img(r'assets/test/images/7eab4d8284.png')
x = img_to_array(img)[:,:,1]
x2 = resize(x, (202, 202, 1), mode='symmetric', preserve_range=True)
#x2 = rescale(x / 255, 2)
#x2 = x2*255
#x2 = x2.astype(np.int32)
import cv2
cv2.imwrite(r'/Users/Chris/PycharmProjects/TGS_Salt_Identification/models/U-Net/unet_128/00_tests/test.png', x)
cv2.imwrite(r'/Users/Chris/PycharmProjects/TGS_Salt_Identification/models/U-Net/unet_128/00_tests/test2.png', x2)

img1 = load_img(r'/Users/Chris/PycharmProjects/TGS_Salt_Identification/models/U-Net/unet_128/00_tests/test.png')
img2 = load_img(r'/Users/Chris/PycharmProjects/TGS_Salt_Identification/models/U-Net/unet_128/00_tests/test2.png')
#img1.show()
#img2.show()
x1 = img_to_array(img1)[:,:,1]
x2 = img_to_array(img2)[:,:,1]





###################
import numpy as np
img = load_img(r'assets/test/images/7eab4d8284.png')
img.show()
x = img_to_array(img)[:,:,1]
x = np.transpose(x)
cv2.imwrite(r'/Users/Chris/PycharmProjects/TGS_Salt_Identification/models/U-Net/unet_128/00_tests/test.png', x)
img = load_img(r'/Users/Chris/PycharmProjects/TGS_Salt_Identification/models/U-Net/unet_128/00_tests/test.png')
img.show()


x

x = np.fliplr(x)

x = np.transpose(x)

x = np.transpose(x)
x = np.fliplr(x)

x = np.transpose(x)
x = np.fliplr(x)
x = np.transpose(x)

x = np.transpose(x)
x = np.fliplr(x)
x = np.transpose(x)
x = np.fliplr(x)

x = np.transpose(x)
x = np.fliplr(x)
x = np.transpose(x)
x = np.fliplr(x)
x = np.transpose(x)

x = np.transpose(x)
x = np.fliplr(x)
x = np.transpose(x)
x = np.fliplr(x)
x = np.transpose(x)
x = np.fliplr(x)


X = np.append(X, [np.fliplr(x) for x in X], axis=0)
X_t = [np.transpose(x) for x in X]
X_tf = [np.fliplr(x) for x in X_t]
X_tft = [np.transpose(x) for x in X_tf]
X_tftf = [np.fliplr(x) for x in X_tft]
X_tftft = [np.transpose(x) for x in X_tftf]
X_tftftf = [np.fliplr(x) for x in X_tftft]
X = np.append(X, X_t, axis=0)
X = np.append(X, X_tf, axis=0)
X = np.append(X, X_tft, axis=0)
X = np.append(X, X_tftf, axis=0)
X = np.append(X, X_tftft, axis=0)
X = np.append(X, X_tftftf, axis=0)

