from keras.layers import Conv2D, Input, Concatenate, ZeroPadding2D
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose
from keras.layers import concatenate, Reshape, Cropping2D

class Architectures():
    def unet_flat(self):

        def conv_block(num_layers,inp,units,kernel):
            x = inp
            for l in range(num_layers):
                x = Conv2D(units, kernel_size=kernel, padding='SAME',activation='relu')(x)
            return x

        inp = Input(shape=(101,101,1))
        resized = ZeroPadding2D(padding=((9,8), (9,8)))(inp) #pad tp shape=(None, 128,128,1))
        cnn1 = conv_block(4,resized,32,3)
        cnn2 = conv_block(4,resized,24,5)
        cnn3 = conv_block(4,resized,16,7)
        concat = Concatenate()([cnn1,cnn2,cnn3])
        d1 = Conv2D(16,1, activation='relu')(concat)
        out = Conv2D(1,1, activation='sigmoid')(d1)

        return inp, out

    def unet(self):
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((9,8), (9,8)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
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
        cropped = Cropping2D(cropping=((9,8), (9,8)))(outputs)

        return input_img, cropped

    def unet_depth(self, n_features):
        input_img = Input((101, 101, 1), name='img')
        input_features = Input((n_features, ), name='feat')

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
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

        # Join features information in the depthest layer
        f_repeat = RepeatVector(8*8)(input_features)
        f_conv = Reshape((8, 8, n_features))(f_repeat)
        p4_feat = concatenate([p4, f_conv], -1)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
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

        return input_img, input_features, outputs