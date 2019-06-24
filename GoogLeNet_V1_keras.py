from keras.layers import (Input, Conv2D, Dense, MaxPooling2D, Dropout,
                          AveragePooling2D, GlobalAveragePooling2D, Flatten)
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K



class GoogLeNet_V1_22():
    def __init__(self, H_size=224, W_size=224, channels=3, nclasses=1000, include_top=False):
        self.H_size = H_size
        self.W_size = W_size
        self.channels = channels
        self.nclasses = nclasses
        self.include_top = include_top
        return


    def __InceptionV1(self, layers, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                      filters_maxpool_proj):
        conv1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding="same",
                         activation="relu", kernel_regularizer=l2(0.01))(layers)

        conv3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding="same",
                                activation="relu", kernel_regularizer=l2(0.01))(layers)
        conv3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding="same",
                         activation="relu", kernel_regularizer=l2(0.01))(conv3x3_reduce)

        conv5x5_reduce = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding="same",
                                activation="relu", kernel_regularizer=l2(0.01))(layers)
        conv5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), padding="same",
                         activation="relu", kernel_regularizer=l2(0.01))(conv5x5_reduce)

        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(layers)
        maxpool_proj = Conv2D(filters=filters_maxpool_proj, kernel_size=(1, 1), padding="same",
                              activation="relu", kernel_regularizer=l2(0.01))(maxpool)

        inception_output = concatenate([conv1x1, conv3x3, conv5x5, maxpool_proj], axis=3)

        return inception_output


    def __GoogLeNetV1(self):
        input = Input(shape=(self.H_size, self.W_size, self.channels))

        conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same",
                              activation="relu", kernel_regularizer=l2(0.01))(input)
        maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1_7x7_s2)

        conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1, 1), padding="same",
                                  activation="relu", kernel_regularizer=l2(0.01))(maxpool1_3x3_s2)
        conv2_3x3 = Conv2D(filters=192, kernel_size=(3, 3), padding="same",
                           activation="relu", kernel_regularizer=l2(0.01))(conv2_3x3_reduce)
        maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv2_3x3)

        _inception_3a = self.__InceptionV1(layers=maxpool2_3x3_s2, filters_1x1=64,
                                           filters_3x3_reduce=96, filters_3x3=128,
                                           filters_5x5_reduce=16, filters_5x5=32,
                                           filters_maxpool_proj=32)

        _inception_3b = self.__InceptionV1(layers=_inception_3a, filters_1x1=128,
                                           filters_3x3_reduce=128, filters_3x3=192,
                                           filters_5x5_reduce=32, filters_5x5=96,
                                           filters_maxpool_proj=64)

        maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(_inception_3b)

        _inception_4a = self.__InceptionV1(layers=maxpool3_3x3_s2, filters_1x1=192,
                                           filters_3x3_reduce=96, filters_3x3=208,
                                           filters_5x5_reduce=16, filters_5x5=48,
                                           filters_maxpool_proj=64)
        _inception_4b = self.__InceptionV1(layers=_inception_4a, filters_1x1=160,
                                           filters_3x3_reduce=112, filters_3x3=224,
                                           filters_5x5_reduce=24, filters_5x5=64,
                                           filters_maxpool_proj=64)
        _inception_4c = self.__InceptionV1(layers=_inception_4b, filters_1x1=128,
                                           filters_3x3_reduce=128, filters_3x3=256,
                                           filters_5x5_reduce=24, filters_5x5=64,
                                           filters_maxpool_proj=64)
        _inception_4d = self.__InceptionV1(layers=_inception_4c, filters_1x1=112,
                                           filters_3x3_reduce=144, filters_3x3=288,
                                           filters_5x5_reduce=32, filters_5x5=64,
                                           filters_maxpool_proj=64)
        _inception_4e = self.__InceptionV1(layers=_inception_4d, filters_1x1=256,
                                           filters_3x3_reduce=160, filters_3x3=320,
                                           filters_5x5_reduce=32, filters_5x5=128,
                                           filters_maxpool_proj=128)

        maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(_inception_4e)

        _inception_5a = self.__InceptionV1(layers=maxpool4_3x3_s2, filters_1x1=256,
                                      filters_3x3_reduce=160, filters_3x3=320,
                                      filters_5x5_reduce=32, filters_5x5=128,
                                      filters_maxpool_proj=128)
        _inception_5b = self.__InceptionV1(layers=_inception_5a, filters_1x1=384,
                                           filters_3x3_reduce=192, filters_3x3=384,
                                           filters_5x5_reduce=48, filters_5x5=128,
                                           filters_maxpool_proj=128)

        avgpool1_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding="same")(_inception_5b)
        dropout1_4 = Dropout(rate=0.4)(avgpool1_7x7_s1)

        if self.include_top:
            # Classification block
            globavgpool_1 = GlobalAveragePooling2D(name="global_avg_pool")(dropout1_4)    # Flatten()
            linear1 = Dense(units=self.nclasses, activation="softmax", kernel_regularizer=l2(0.01), name="predictions")(globavgpool_1)
        else:
            linear1 = Dense(units=self.nclasses, activation="softmax", kernel_regularizer=l2(0.01))(dropout1_4)

        out = linear1
        model = Model(inputs=input, outputs=out, name="Inception_V1")
        return model


# def main(H_size=224, W_size=224, channels=3, nclasses=1000):
#     NetFrame = GoogLeNet_V1_22(H_size=H_size, W_size=W_size, channels=channels, nclasses=nclasses)
#     model = NetFrame._GoogLeNet_V1_22__GoogLeNetV1()
#     return model




if __name__ == '__main__':
    googlenet_frame_v1 = GoogLeNet_V1_22(H_size=224, W_size=224, channels=3, nclasses=1000, include_top=False)
    model_GoogLeNetV1 = googlenet_frame_v1._GoogLeNet_V1_22__GoogLeNetV1()
    print(model_GoogLeNetV1)

    # model_GoogLeNetV1 = main(H_size=224, W_size=224, channels=3, nclasses=1000)
    print(model_GoogLeNetV1.summary())


    # if not include_top:
    #     x = model_GoogLeNetV1.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(500, activation="relu")(x)
    #     predictions = Dense(200, activation="softmax")(x)
    #     model = Model(inputs=model_GoogLeNetV1.input, outputs=predictions)
    #     model.compile(optimizer="rmsprop", loss="categorical_crossentropy")


    del model_GoogLeNetV1





























