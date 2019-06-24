from keras.layers import (Input, Conv2D, Dense, MaxPooling2D, Dropout, BatchNormalization,
                          AveragePooling2D, GlobalAveragePooling2D, Flatten, ReLU)
from keras.layers.merge import concatenate
from keras.layers import Lambda
from keras.models import Model



def _Base_BN(layers, filter_num, ks_val, strides_val):
    conv_nxn = Conv2D(filters=filter_num, kernel_size=ks_val, strides=strides_val, padding="same")(layers)
    conv_nxn = BatchNormalization()(conv_nxn)
    conv_nxn = ReLU()(conv_nxn)
    return conv_nxn


def _InceptionV2(in_layers, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce,
                  filters_5x5, filters_maxpool_proj, strid_pma, name, concat_axis=3):

    # strid_pma = strides_pass or strides_max, or strides_avg

    if filters_1x1 > 0:
        conv1x1 = _Base_BN(layers=in_layers, filter_num=filters_1x1, ks_val=1, strides_val=strid_pma)

    #### conv 3x3 ####
    conv3x3_reduce = _Base_BN(layers=in_layers, filter_num=filters_3x3_reduce, ks_val=1, strides_val=strid_pma)
    conv3x3 = _Base_BN(layers=conv3x3_reduce, filter_num=filters_3x3, ks_val=3, strides_val=1)

    #### conv 5x5 ####
    conv5x5_reduce = _Base_BN(layers=in_layers, filter_num=filters_5x5_reduce, ks_val=1, strides_val=strid_pma)
    conv5x5 = _Base_BN(layers=conv5x5_reduce, filter_num=filters_5x5, ks_val=3, strides_val=1)
    conv5x5 = _Base_BN(layers=conv5x5, filter_num=filters_5x5, ks_val=3, strides_val=1)

    if name == "passthrough":
        pool1 = MaxPooling2D(pool_size=3, strides=strid_pma, padding="same")(in_layers)
        out = concatenate([conv3x3, conv5x5, pool1], concat_axis)

    elif name == "maxpool":
        pool1 = MaxPooling2D(pool_size=3, strides=strid_pma, padding="same")(in_layers)
        pool_proj = _Base_BN(layers=pool1, filter_num=filters_maxpool_proj, ks_val=1, strides_val=1)
        out = concatenate([conv1x1, conv3x3, conv5x5, pool_proj], concat_axis)

    else:
        pool1 = AveragePooling2D(pool_size=3, strides=strid_pma, padding="same")(in_layers)
        pool_proj = _Base_BN(layers=pool1, filter_num=filters_maxpool_proj, ks_val=1, strides_val=1)
        out = concatenate([conv1x1, conv3x3, conv5x5, pool_proj], concat_axis)

    return out


def _GoogLeNet_V2(in_shape, n_classes, include_top=True):
    # include_top:  Classification block
    input = Input(shape=in_shape, name="inputs")   # in_shape=(299, 299, 3)

    ### first conv ###
    conv1_7x7_s2 = _Base_BN(layers=input, filter_num=64, ks_val=7, strides_val=2)
    pool1_3x3_s2 = MaxPooling2D(pool_size=3, strides=2, padding="same")(conv1_7x7_s2)

    ### second conv ###
    conv2_3x3_reduce = _Base_BN(layers=pool1_3x3_s2, filter_num=64, ks_val=1, strides_val=1)
    conv2_3x3 = _Base_BN(layers=conv2_3x3_reduce, filter_num=192, ks_val=3, strides_val=1)
    pool2_3x3_s2 = MaxPooling2D(pool_size=3, strides=2, padding="same")(conv2_3x3)

    _inception_3a = _InceptionV2(in_layers=pool2_3x3_s2, filters_1x1=64,
                                 filters_3x3_reduce=64, filters_3x3=64,
                                 filters_5x5_reduce=64, filters_5x5=96,
                                 filters_maxpool_proj=32, strid_pma=1,
                                 name="avg", concat_axis=3)
    _inception_3b = _InceptionV2(_inception_3a, 64, 64, 96, 64, 96, 64, strid_pma=1, name="avg", concat_axis=3)
    _inception_3c = _InceptionV2(_inception_3b, 0, 128, 160, 64, 96, 0, strid_pma=1, name="passthrough", concat_axis=3)

    _inception_4a = _InceptionV2(_inception_3c, 224, 64, 96, 96, 128, 128, strid_pma=2, name="avg", concat_axis=3)
    _inception_4b = _InceptionV2(_inception_4a, 192, 96, 128, 96, 128, 128, strid_pma=1, name="avg", concat_axis=3)
    _inception_4c = _InceptionV2(_inception_4b, 160, 128, 160, 128, 160, 96, strid_pma=1, name="avg", concat_axis=3)
    _inception_4d = _InceptionV2(_inception_4c, 96, 128, 192, 160, 192, 96, strid_pma=1, name="avg", concat_axis=3)
    _inception_4e = _InceptionV2(_inception_4d, 0, 128, 192, 192, 256, 0, strid_pma=1, name="passthrough", concat_axis=3)

    _inception_5a = _InceptionV2(_inception_4e, 352, 192, 320, 160, 224, 128, strid_pma=2, name="avg", concat_axis=3)
    _inception_5b = _InceptionV2(_inception_5a, 352, 192, 320, 192, 224, 128, strid_pma=1, name="maxpool", concat_axis=3)

    _avgpool = AveragePooling2D(pool_size=7, strides=1, padding="valid")(_inception_5b)
    _flatten = Flatten()(_avgpool)

    ###  out , 主分类器###
    main_out = Dense(n_classes, activation="softmax")(_flatten)

    ###  aux_out-1 , Auxiliary classifier-1 ###
    aux1_pool_1 = AveragePooling2D(pool_size=5, strides=3, padding="valid")(_inception_4a)
    aux1_conv_1 = _Base_BN(aux1_pool_1, filter_num=128, ks_val=1, strides_val=1)
    aux1_fc_1 = Dense(units=1024, activation="relu")(aux1_conv_1)
    aux1_drop_1 = Dropout(0.3)(aux1_fc_1)
    if include_top:
        aux1_globavg_1 = GlobalAveragePooling2D()(aux1_drop_1)
        aux_out_1 = Dense(n_classes, activation="softmax")(aux1_globavg_1)
    else:
        aux_out_1 = Dense(n_classes, activation="softmax")(aux1_drop_1)

    ###  aux_out-2 , Auxiliary classifier-2 ###
    aux2_pool_1 = AveragePooling2D(pool_size=5, strides=3, padding="valid")(_inception_4d)
    aux2_conv_1 = _Base_BN(aux2_pool_1, filter_num=128, ks_val=1, strides_val=1)
    aux2_fc_1 = Dense(units=1024, activation="relu")(aux2_conv_1)
    aux2_drop_1 = Dropout(0.3)(aux2_fc_1)
    if include_top:
        aux2_globavg_1 = GlobalAveragePooling2D()(aux2_drop_1)
        aux_out_2 = Dense(n_classes, activation="softmax")(aux2_globavg_1)
    else:
        aux_out_2 = Dense(n_classes, activation="softmax")(aux2_drop_1)

    # ### Method-1: Output three results simultaneously ###
    # model = Model(inputs=[input], outputs=[main_out, aux_out_1, aux_out_2])

    ### Method-1: Merge the outputs ###
    def _Add_tensor(IN_out):
        out = IN_out[0] + 0.3 * IN_out[1] + 0.3 * IN_out[2]
        return out
    out = Lambda(_Add_tensor)([main_out, aux_out_1, aux_out_2])
    model = Model(inputs=input, outputs=out)

    return model





if __name__ == '__main__':
    model = _GoogLeNet_V2(in_shape=(224, 224, 3), n_classes=10, include_top=True)
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    # model.fit({"inputs": X}, {"out": Y})
    print(model.summary())
    del model

    print("="*30)




