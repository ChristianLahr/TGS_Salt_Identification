from keras.layers import Conv2D, Input, Concatenate, ZeroPadding2D
from keras.layers import MaxPooling2D, RepeatVector, Conv2DTranspose
from keras.layers import concatenate, Reshape, Cropping2D

class Architectures():
    def cnns(self):

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

    def unet_128(self):
        # smallest grid: 8x8
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

    def unet_128_betterDecoder(self):
        # smallest grid: 8x8
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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
        c6 = Conv2D(32, (1, 1), activation='relu', padding='same') (c6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(16, (1, 1), activation='relu', padding='same') (c7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(8, (1, 1), activation='relu', padding='same') (c8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

    def unet_256(self):
        # smallest grid: 4x4
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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
        p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

        c6 = Conv2D(256, (3, 3), activation='relu', padding='same') (p5)
        c6 = Conv2D(256, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c5])
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c4])
        c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c3])
        c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (c9)

        u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
        u10 = concatenate([u10, c2])
        c10 = Conv2D(16, (3, 3), activation='relu', padding='same') (u10)
        c10 = Conv2D(16, (3, 3), activation='relu', padding='same') (c10)

        u11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c10)
        u11 = concatenate([u11, c1], axis=3)
        c11 = Conv2D(8, (3, 3), activation='relu', padding='same') (u11)
        c11 = Conv2D(8, (3, 3), activation='relu', padding='same') (c11)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

    def unet_512(self):
        # smallest grid: 2x2
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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
        p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

        c55 = Conv2D(256, (3, 3), activation='relu', padding='same') (p5)
        c55 = Conv2D(256, (3, 3), activation='relu', padding='same') (c55)
        p55 = MaxPooling2D(pool_size=(2, 2)) (c55)

        c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (p55)
        c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (c6)

        u77 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
        u77 = concatenate([u77, c55])
        c77 = Conv2D(256, (3, 3), activation='relu', padding='same') (u77)
        c77 = Conv2D(256, (3, 3), activation='relu', padding='same') (c77)

        u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c77)
        u7 = concatenate([u7, c5])
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c4])
        c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c3])
        c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (c9)

        u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
        u10 = concatenate([u10, c2])
        c10 = Conv2D(16, (3, 3), activation='relu', padding='same') (u10)
        c10 = Conv2D(16, (3, 3), activation='relu', padding='same') (c10)

        u11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c10)
        u11 = concatenate([u11, c1], axis=3)
        c11 = Conv2D(8, (3, 3), activation='relu', padding='same') (u11)
        c11 = Conv2D(8, (3, 3), activation='relu', padding='same') (c11)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

    def unet_64(self):
        # smallest grid: 16x16
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
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

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped

    def unet_32(self):
        # smallest grid: 32x32
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14, 13), (14, 13)))(input_img) #pad tp shape=(None, 128,128,1))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c3)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14, 13), (14, 13)))(outputs)

        return input_img, cropped

    def unet_128_extra_features(self, n_features, im_chan):
        input_img = Input((101, 101, im_chan), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))
        input_features = Input((n_features, ), name='feat')

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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
        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, input_features, cropped

    def wnet(self):
        # smallest grid: 8x8
        input_img = Input((101, 101, 1), name='img')
        resized = ZeroPadding2D(padding=((14,13), (14,13)))(input_img) #pad tp shape=(None, 128,128,1))

        #down
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (resized)
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

        #up 32
        u6_up32 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c3)
        u6_up32 = concatenate([u6_up32, c2])
        c6_up32 = Conv2D(16, (3, 3), activation='relu', padding='same') (u6_up32)
        c6_up32 = Conv2D(16, (3, 3), activation='relu', padding='same') (c6_up32)

        u7_up32 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c6_up32)
        u7_up32 = concatenate([u7_up32, c1], axis=3)
        c7_up32 = Conv2D(8, (3, 3), activation='relu', padding='same') (u7_up32)
        c7_up32 = Conv2D(8, (3, 3), activation='relu', padding='same') (c7_up32)

        outputs_up32 = Conv2D(1, (1, 1), activation='sigmoid') (c7_up32)

        #up 64
        u6_up64 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
        u6_up64 = concatenate([u6_up64, c3])
        c6_up64 = Conv2D(32, (3, 3), activation='relu', padding='same') (u6_up64)
        c6_up64 = Conv2D(32, (3, 3), activation='relu', padding='same') (c6_up64)

        u7_up64 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c6_up64)
        u7_up64 = concatenate([u7_up64, c2])
        c7_up64 = Conv2D(16, (3, 3), activation='relu', padding='same') (u7_up64)
        c7_up64 = Conv2D(16, (3, 3), activation='relu', padding='same') (c7_up64)

        u8_up64 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c7_up64)
        u8_up64 = concatenate([u8_up64, c1], axis=3)
        c8_up64 = Conv2D(8, (3, 3), activation='relu', padding='same') (u8_up64)
        c8_up64 = Conv2D(8, (3, 3), activation='relu', padding='same') (c8_up64)

        outputs_up64 = Conv2D(1, (1, 1), activation='sigmoid') (c8_up64)

        #up 128
        u6_up128 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
        u6_up128 = concatenate([u6_up128, c4])
        c6_up128 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6_up128)
        c6_up128 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6_up128)

        u7_up128 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_up128)
        u7_up128 = concatenate([u7_up128, c3])
        c7_up128 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7_up128)
        c7_up128 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7_up128)

        u8_up128 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7_up128)
        u8_up128 = concatenate([u8_up128, c2])
        c8_up128 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8_up128)
        c8_up128 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8_up128)

        u9_up128 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8_up128)
        u9_up128 = concatenate([u9_up128, c1], axis=3)
        c9_up128 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9_up128)
        c9_up128 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9_up128)

        outputs_up128 = Conv2D(1, (1, 1), activation='sigmoid') (c9_up128)

        outputs_first_u = concatenate([outputs_up32, outputs_up64, outputs_up128], axis=3)

        c1_second_u = Conv2D(8, (3, 3), activation='relu', padding='same') (outputs_first_u)
        c1_second_u = Conv2D(8, (3, 3), activation='relu', padding='same') (c1_second_u)
        p1_second_u = MaxPooling2D((2, 2)) (c1_second_u)

        c2_second_u = Conv2D(16, (3, 3), activation='relu', padding='same') (p1_second_u)
        c2_second_u = Conv2D(16, (3, 3), activation='relu', padding='same') (c2_second_u)
        p2_second_u = MaxPooling2D((2, 2)) (c2_second_u)

        c3_second_u = Conv2D(32, (3, 3), activation='relu', padding='same') (p2_second_u)
        c3_second_u = Conv2D(32, (3, 3), activation='relu', padding='same') (c3_second_u)
        p3_second_u = MaxPooling2D((2, 2)) (c3_second_u)

        c4_second_u = Conv2D(64, (3, 3), activation='relu', padding='same') (p3_second_u)
        c4_second_u = Conv2D(64, (3, 3), activation='relu', padding='same') (c4_second_u)
        p4_second_u = MaxPooling2D((2, 2)) (c4_second_u)

        c5_second_u = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_second_u)
        c5_second_u = Conv2D(128, (3, 3), activation='relu', padding='same') (c5_second_u)

        u6_second_u = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5_second_u)
        u6_second_u = concatenate([u6_second_u, c4_second_u])
        c6_second_u = Conv2D(64, (3, 3), activation='relu', padding='same') (u6_second_u)
        c6_second_u = Conv2D(32, (1, 1), activation='relu', padding='same') (c6_second_u)
        c6_second_u = Conv2D(64, (3, 3), activation='relu', padding='same') (c6_second_u)

        u7_second_u = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_second_u)
        u7_second_u = concatenate([u7_second_u, c3_second_u])
        c7_second_u = Conv2D(32, (3, 3), activation='relu', padding='same') (u7_second_u)
        c7_second_u = Conv2D(16, (1, 1), activation='relu', padding='same') (c7_second_u)
        c7_second_u = Conv2D(32, (3, 3), activation='relu', padding='same') (c7_second_u)

        u8_second_u = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7_second_u)
        u8_second_u = concatenate([u8_second_u, c2_second_u])
        c8_second_u = Conv2D(16, (3, 3), activation='relu', padding='same') (u8_second_u)
        c8_second_u = Conv2D(8, (1, 1), activation='relu', padding='same') (c8_second_u)
        c8_second_u = Conv2D(16, (3, 3), activation='relu', padding='same') (c8_second_u)

        u9_second_u = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8_second_u)
        u9_second_u = concatenate([u9_second_u, c1_second_u], axis=3)
        c9_second_u = Conv2D(8, (3, 3), activation='relu', padding='same') (u9_second_u)
        c9_second_u = Conv2D(4, (1, 1), activation='relu', padding='same') (c9_second_u)
        c9_second_u = Conv2D(8, (3, 3), activation='relu', padding='same') (c9_second_u)

        outputs_second_u = Conv2D(1, (1, 1), activation='sigmoid') (c9_second_u)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs_second_u)

        return input_img, cropped

    def unet_pretrainedEncoder(self):
        input_img = Input((101, 101, 3), name='img')
        resized = ZeroPadding2D(padding=((99,99), (99,99)))(input_img) #pad tp shape=(None, 299,299,3))
        # use better padding method here!!!

        resNet = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=resized, input_shape=None, pooling=None)
        resNet.trainable = False
        layers = resNet.layers
        for l in layers:
            l.trainable = False

        resNetDim16_anzapfen_1 = layers[605].output
        resNetDim16_anzapfen_2 = layers[606].output
        resNetDim16_anzapfen_3 = layers[607].output
        resNetDim16_anzapfen = concatenate([resNetDim16_anzapfen_1, resNetDim16_anzapfen_2, resNetDim16_anzapfen_3])
        resNetDim16_fully = Conv2D(128, (1, 1), activation='relu') (resNetDim16_anzapfen)
        resNetDim16_fully = Cropping2D(cropping=((0,1), (0,1)))(resNetDim16_fully)

        resNetDim32_anzapfen = layers[267].output
        resNetDim32_fully = Conv2D(64, (1, 1), activation='relu') (resNetDim32_anzapfen)
        resNetDim32_fully = Cropping2D(cropping=((1,2), (1,2)))(resNetDim32_fully)

        resNetDim64_anzapfen = layers[17].output
        resNetDim64_fully = Conv2D(64, (1, 1), activation='relu') (resNetDim64_anzapfen)
        resNetDim64_fully = Cropping2D(cropping=((3,4), (3,4)))(resNetDim64_fully)

        resNetDim128_anzapfen = layers[10].output
        resNetDim128_fully = Conv2D(32, (1, 1), activation='relu') (resNetDim128_anzapfen)
        resNetDim128_fully = Cropping2D(cropping=((9,10), (9,10)))(resNetDim128_fully)

        resNetDim8_anzapfen = resNet.output
        resNetDim8_fully = Conv2D(256, (1, 1), activation='relu') (resNetDim8_anzapfen)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (resNetDim8_fully)
        u6 = concatenate([u6, resNetDim16_fully])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(32, (1, 1), activation='relu', padding='same') (c6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, resNetDim32_fully])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(16, (1, 1), activation='relu', padding='same') (c7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, resNetDim64_fully])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(8, (1, 1), activation='relu', padding='same') (c8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, resNetDim128_fully], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(4, (1, 1), activation='relu', padding='same') (c9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        cropped = Cropping2D(cropping=((14,13), (14,13)))(outputs)

        return input_img, cropped, resNet.layers


