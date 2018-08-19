from keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy
from sklearn.metrics import roc_auc_score
import numpy as np
from keras.callbacks import Callback

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