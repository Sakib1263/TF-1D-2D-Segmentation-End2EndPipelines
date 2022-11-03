# Import Necessary Libraries
import tensorflow as tf
import numpy as np
import pandas as pd


def Conv_Block(inputs, model_width, kernel, bn=True, activation_fun='relu'):
    # 2D Convolutional Block
    x = tf.keras.layers.Conv2D(model_width, kernel, padding='same')(inputs)
    if bn == True:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_fun)(x)

    return x


def trans_conv2D(inputs, model_width, bn=True, activation_fun='relu'):
    # 2D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv2DTranspose(model_width, (2, 2), strides=(2, 2), padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    if bn == True:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_fun)(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the Keras Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = tf.keras.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs, size=(2, 2), interpolation_mode='bilinear'):
    # 2D UpSampling Block
    up = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation_mode)(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    shape = inputs.shape
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1] * shape[2])(latent)
    latent = tf.keras.layers.Reshape((shape[1], shape[2], model_width))(latent)

    return latent


def dense_block(x, num_filters, kernel_size, num_layers):
    x = Conv_Block(x, num_filters, kernel_size)
    for _ in range(num_layers):
        cb = Conv_Block(x, num_filters, kernel_size)
        x = tf.keras.layers.add([x, cb])

    return x


def Attention_Block(skip_connection, gating_signal, num_filters, multiplier):
    # Attention Block
    conv1x1_1 = tf.keras.layers.Conv2D(num_filters * multiplier, (1, 1), strides=(2, 2))(skip_connection)
    conv1x1_1 = tf.keras.layers.BatchNormalization()(conv1x1_1)
    conv1x1_2 = tf.keras.layers.Conv2D(num_filters * multiplier, (1, 1), strides=(1, 1))(gating_signal)
    conv1x1_2 = tf.keras.layers.BatchNormalization()(conv1x1_2)
    conv1_2 = tf.keras.layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = tf.keras.layers.Activation('relu')(conv1_2)
    conv1_2 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1))(conv1_2)
    conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = tf.keras.layers.Activation('sigmoid')(conv1_2)
    resampler1 = upConv_Block(conv1_2)
    resampler2 = trans_conv2D(conv1_2, 1)
    resampler = tf.keras.layers.add([resampler1, resampler2])
    out = skip_connection * resampler

    return out


def MultiResBlock(inputs, model_width, kernel, alpha):
    # MultiRes Block
    # U {int} -- Number of filters in a corrsponding UNet stage
    # inp {keras layer} -- input layer

    w = alpha * model_width

    shortcut = inputs
    shortcut = Conv_Block(shortcut, int(w * 0.167) + int(w * 0.333) + int(w * 0.5), (1, 1))

    conv3x3 = Conv_Block(inputs, int(w * 0.167), kernel)
    conv5x5 = Conv_Block(conv3x3, int(w * 0.333), kernel)
    conv7x7 = Conv_Block(conv5x5, int(w * 0.5), kernel)

    out = tf.keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Add()([shortcut, out])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization()(out)

    return out


def ResPath(inputs, model_depth, model_width, kernel):
    # ResPath
    # filters {int} -- [description]
    # length {int} -- length of ResPath
    # inp {keras layer} -- input layer

    shortcut = inputs
    shortcut = Conv_Block(shortcut, model_width, (1, 1))

    out = Conv_Block(inputs, model_width, kernel)
    out = tf.keras.layers.Add()([shortcut, out])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization()(out)

    for _ in range(1, model_depth):
        shortcut = out
        shortcut = Conv_Block(shortcut, model_width, (1, 1))

        out = Conv_Block(out, model_width, kernel)
        out = tf.keras.layers.Add()([shortcut, out])
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization()(out)

    return out


def UNet(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv):
    # UNet based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    deconv = convs_list[-1]
    for j in range(0, model_depth):
        skip_connection = convs_list[model_depth - j - 1]
        if A_G == 1:
            skip_connection = Attention_Block(convs_list[model_depth - j - 1], deconv, model_width, 2 ** (model_depth - j - 1))
        if D_S == 1:
            # For Deep Supervision
            level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - j}')(deconv)
            levels.append(level)
        if is_transconv:
            deconv = trans_conv2D(deconv, model_width * (2 ** (model_depth - j - 1)))
        elif not is_transconv:
            deconv = upConv_Block(deconv)
        SC_Shape = skip_connection.shape
        DCNV_Shape = deconv.shape
        if LSTM == 1:
            x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** (model_depth - j - 1))))(skip_connection)
            x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** (model_depth - j - 1))))(deconv)
            merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
            deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (model_depth - j - 2))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                go_backwards=True, kernel_initializer='he_normal')(merge)
        elif LSTM == 0:
            deconv = Concat_Block(deconv, skip_connection)
        deconv = Conv_Block(deconv, model_width * (2 ** (model_depth - j - 1)), (3, 3))
        # deconv = Conv_Block(deconv, model_width * (2 ** (model_depth - j - 1)), (3, 3))

    return deconv, levels


def UNetE(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv):
    # Ensembled UNet based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    if D_S == 1:
        level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth}')(convs_list[0])
        levels.append(level)

    deconv = []
    deconvs = {}

    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if i == 1:
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(convs_list[j + 1], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(convs_list[j + 1])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
            elif i > 1:
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                levels.append(level)

    deconv = deconvs["deconv%s%s" % (0, model_depth)]

    return deconv, levels


def UNetP(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv):
    # UNet+ based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    if D_S == 1:
        level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth}')(convs_list[0])
        levels.append(level)

    deconv = []
    deconvs = {}

    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if i == 1:
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(convs_list[j + 1], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(convs_list[j + 1])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
            elif i > 1:
                skip_connection = deconvs["deconv%s%s" % (j, (i - 1))]
                if A_G == 1:
                    skip_connection = Attention_Block(deconvs["deconv%s%s" % (j, (i - 1))], deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                levels.append(level)

    deconv = deconvs["deconv%s%s" % (0, model_depth)]

    return deconv, levels


def UNetPP(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv):
    # UNet++ based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    if D_S == 1:
        level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth}')(convs_list[0])
        levels.append(level)

    deconv = []
    deconvs = {}

    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if i == 1:
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(convs_list[j + 1], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(convs_list[j + 1])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
            elif i > 1:
                deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                if A_G == 1:
                    deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                for k in range(2, i):
                    deconv_temp = deconvs["deconv%s%s" % (j, k)]
                    if A_G == 1:
                        deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                    deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                    merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                levels.append(level)

    deconv = deconvs["deconv%s%s" % (0, model_depth)]

    return deconv, levels


def UNet3P(skip_connections, model_width, model_depth, D_S):
    # UNet3+ based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    deconv = convs_list[-1]
    deconvs = {}

    for j in range(0, model_depth):
        skip_connections_all = convs_list[model_depth - j - 1]
        skip_connections_all = Conv_Block(skip_connections_all, model_width * (2 ** 0), (3, 3))
        for k in range(0, model_depth - j - 1):
            skip_connection = convs_list[k]
            skip_connection = tf.keras.layers.MaxPooling2D(pool_size=((2 ** ((model_depth - j) - k - 1)), (2 ** ((model_depth - j) - k - 1))))(skip_connection)
            skip_connection = Conv_Block(skip_connection, model_width * (2 ** 0), (3, 3))
            skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
        deconv_tot = Conv_Block(deconv, model_width * (2 ** 0), (3, 3))
        deconv_tot = upConv_Block(deconv_tot, size=(2 ** 1, 2 ** 1), interpolation_mode='bilinear')
        deconv_tot = tf.keras.layers.Activation('sigmoid')(deconv_tot)
        deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
        if j > 0:
            for m in range(0, j):
                deconv = Conv_Block(deconvs["deconv%s" % m], model_width * (2 ** 0), (3, 3))
                deconv = upConv_Block(deconv, size=((2 ** (j - m)), (2 ** (j - m))), interpolation_mode='bilinear')
                deconv = tf.keras.layers.Activation('sigmoid')(deconv)
                deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
        deconv = Conv_Block(deconv_tot, model_width * (model_depth + 1), (3, 3))
        deconvs["deconv%s" % j] = deconv
        if D_S == 1:
            # For Deep Supervision
            level = tf.keras.layers.Conv2D(1, (1, 1), (2, 2), name=f'level{model_depth - j}')(deconv)
            levels.append(level)

    return deconv, levels


def UNet4P(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv):
    # UNet4+ based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    if D_S == 1:
        level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth}')(convs_list[0])
        levels.append(level)

    deconv = []
    deconvs = {}
    deconvs_skip = {}

    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if i == 1:
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(convs_list[j + 1], model_width * 2 ** j)
                elif not is_transconv:
                    deconv = upConv_Block(convs_list[j + 1])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
                if (i + j) == (model_depth):
                    deconvs_skip["deconv_skip%s" % i] = deconv
            elif i > 1:
                deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                if A_G == 1:
                    deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                for k in range(2, i):
                    deconv_temp = deconvs["deconv%s%s" % (j, k)]
                    if A_G == 1:
                        deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                    deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                    merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                if (i + j) == (model_depth) and (j != (model_depth - 1)):
                    for m in range(1, i - 1):
                        temp_deconv = upConv_Block(deconvs_skip["deconv_skip%s" % m], size=(2 ** (i - m)))
                        temp_deconv = tf.keras.layers.Activation('sigmoid')(temp_deconv)
                        deconv = tf.keras.layers.concatenate([deconv, temp_deconv], axis=-1)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
                if (i + j) == (model_depth):
                    deconvs_skip["deconv_skip%s" % i] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                levels.append(level)

    deconv = deconvs["deconv%s%s" % (0, model_depth)]

    return deconv, levels


def MultiResUNet(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv, kernel, alpha=1.0):
    # MultiResUNet based Decoder
    levels = []
    mresblocks_list = list(skip_connections.values())
    deconv = mresblocks_list[-1]
    for j in range(0, model_depth):
        skip_connection = mresblocks_list[model_depth - j - 1]
        if A_G == 1:
            skip_connection = Attention_Block(mresblocks_list[model_depth - j - 1], deconv, model_width, 2 ** (model_depth - j - 1))
        if D_S == 1:
            level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - j}')(deconv)
            levels.append(level)
        if is_transconv:
            deconv = trans_conv2D(deconv, model_width * (2 ** (model_depth - j - 1)))
        elif not is_transconv:
            deconv = upConv_Block(deconv)
        if LSTM == 1:
            x1 = tf.keras.layers.Reshape(
                target_shape=(1, np.int32(length / (2 ** (model_depth - j - 1))), np.int32(width / (2 ** (model_depth - j - 1))), np.int32(model_width * (2 ** (model_depth - j - 1)))))(
                skip_connection)
            x2 = tf.keras.layers.Reshape(
                target_shape=(1, np.int32(length / (2 ** (model_depth - j - 1))), np.int32(width / (2 ** (model_depth - j - 1))), np.int32(model_width * (2 ** (model_depth - j - 1)))))(deconv)
            merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
            deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (model_depth - j - 2))), kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                                                kernel_initializer='he_normal')(merge)
        elif LSTM == 0:
            deconv = Concat_Block(deconv, skip_connection)
        deconv = MultiResBlock(deconv, model_width * (2 ** (model_depth - j - 1)), kernel, alpha)

    return deconv, levels


def MultiResUNet3P(skip_connections, model_width, model_depth, D_S, kernel, alpha=1.0):
    # MultiResUNet3+ based Decoder
    levels = []
    mresblocks_list = list(skip_connections.values())
    deconv = mresblocks_list[-1]
    deconvs = {}

    for j in range(0, model_depth):
        skip_connections_all = mresblocks_list[model_depth - j - 1]
        skip_connections_all = MultiResBlock(skip_connections_all, model_width*(2 ** 0), kernel, alpha)
        for k in range(0, model_depth - j - 1):
            skip_connection = mresblocks_list[k]
            skip_connection = tf.keras.layers.MaxPooling2D(pool_size=(2 ** ((model_depth - j) - k - 1), 2 ** ((model_depth - j) - k - 1)))(skip_connection)
            skip_connection = MultiResBlock(skip_connection, model_width*(2 ** 0), kernel, alpha)
            skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
        deconv_tot = MultiResBlock(deconv, model_width*(2 ** 0), kernel, alpha)
        deconv_tot = upConv_Block(deconv_tot, size=(2 ** 1, 2 ** 1))
        deconv_tot = tf.keras.layers.Activation('sigmoid')(deconv_tot)
        deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
        if j > 0:
            for m in range(0, j):
                deconv = ResPath(deconvs["deconv%s" % m], j, model_width*(2 ** 0), kernel)
                deconv = upConv_Block(deconv, size=(2 ** (j - m), 2 ** (j - m)))
                deconv = tf.keras.layers.Activation('sigmoid')(deconv)
                deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
        deconv = MultiResBlock(deconv_tot, model_width*model_depth, kernel, alpha)
        deconvs["deconv%s" % j] = deconv
        if D_S == 1:
            # For Deep Supervision
            level = tf.keras.layers.Conv2D(1, (1, 1), (2, 2), name=f'level{model_depth - j}')(deconv)
            levels.append(level)

    return deconv, levels


def AHNet(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv, kernel):
    # AHNet 2DCNN Segmentation Model Decoder
    levels = []
    convs_list = list(skip_connections.values())
    if D_S == 1:
        level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth}')(convs_list[0])
        levels.append(level)

    deconv = []
    deconvs = {}
    deconvs_skip = {}

    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if i == 1:
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(convs_list[j + 1], model_width * 2 ** j)
                elif not is_transconv:
                    deconv = upConv_Block(convs_list[j + 1])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, skip_connection)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
                if (i + j) == (model_depth):
                    deconvs_skip["deconv_skip%s" % i] = deconv
            elif i > 1:
                deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                if A_G == 1:
                    deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                for k in range(2, i):
                    deconv_temp = deconvs["deconv%s%s" % (j, k)]
                    if A_G == 1:
                        deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                    deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                skip_connection = convs_list[j]
                if A_G == 1:
                    skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width, 2 ** j)
                if is_transconv:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], model_width * (2 ** j))
                elif not is_transconv:
                    deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                SC_Shape = skip_connection.shape
                DCNV_Shape = deconv.shape
                if LSTM == 1:
                    x1 = tf.keras.layers.Reshape(target_shape=(1, SC_Shape[1], SC_Shape[2], model_width * (2 ** j)))(skip_connection)
                    x2 = tf.keras.layers.Reshape(target_shape=(1, DCNV_Shape[1], DCNV_Shape[2], model_width * (2 ** j)))(deconv)
                    deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                    merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                    deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (j - 1))), kernel_size=(3, 3), padding='same', return_sequences=False,
                                                        go_backwards=True, kernel_initializer='he_normal')(merge)
                elif LSTM == 0:
                    deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                if (i + j) == (model_depth) and (j != (model_depth - 1)):
                    for m in range(1, i - 1):
                        temp_deconv = ResPath(deconvs_skip["deconv_skip%s" % m], j, model_width * (2 ** 0), kernel)
                        temp_deconv = upConv_Block(temp_deconv, size=(2 ** (i - m), 2 ** (i - m)))
                        temp_deconv = tf.keras.layers.Activation('sigmoid')(temp_deconv)
                        deconv = tf.keras.layers.concatenate([deconv, temp_deconv], axis=-1)
                deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                # deconv = Conv_Block(deconv, model_width * (2 ** j), (3, 3))
                deconvs["deconv%s%s" % (j, i)] = deconv
                if (i + j) == (model_depth):
                    deconvs_skip["deconv_skip%s" % i] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = tf.keras.layers.Conv2D(1, (1, 1), name=f'level{model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                levels.append(level)

    deconv = deconvs["deconv%s%s" % (0, model_depth)]

    return deconv, levels


class UNetWithPretrainedEncoder:
    def __init__(self, decoder_name, length, width, model_width, model_depth, problem_type='Regression', num_channels=3, output_nums=1, ds=0, ae=0, ag=0, lstm=0,
                 dense_loop=1, feature_number=1024, is_transconv=True, alpha=1.0, final_activation="linear", train_mode='pretrained_encoder', is_base_model_trainable=False):
        # decoder_name: Name of the decoder model e.g., UNet, UNet++, etc.
        # length: Input Image Length (x-dim)
        # width: Input Image Width (y-dim) [Normally same as the x-dim i.e., Square shape]
        # model_width: Width of the Input Layer of the Model
        # problem_type: Classification (Binary or Multiclass) or Regression
        # num_channels: Number of Channels in the model
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ag: Checks where Attention Guided is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # lstm: Checks where Bidirectional LSTM is active or not, either 0 or 1 [Default value set as 0]
        # dense_loop: Number of Dense Block in the most bottom layers (1 and 3 are defaults for the UNet's latent layer)
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer
        # alpha: This Parameter is only for MultiResUNet, default value is 1
        # train_mode: Training Mode for the Network [pretrained_encoder: use pretrained weights (e.g., ImageNet), from_scratch: Start training from scratch]
        # is_base_model_trainable: (TRUE: Fine Tuning mode, FALSE: Freeze Mode)
        self.decoder_name = decoder_name
        self.length = length
        self.width = width
        self.model_depth = model_depth
        self.model_width = model_width
        self.problem_type = problem_type
        self.num_channels = num_channels
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        self.dense_loop = dense_loop
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        self.final_activation = final_activation
        self.train_mode = train_mode
        self.is_base_model_trainable = is_base_model_trainable
        self.alpha = alpha
        #
        if self.train_mode == 'pretrained_encoder':
            if (self.model_depth > 5) or (self.model_depth < 1):
                raise ValueError('The depth of a TF-ImageNet Pretrained model can only be discretely varied from 1 to 5')
        else:
            if self.model_depth < 1:
                raise ValueError('The depth of the model cannot be less than 1')

    def ResNet50(self):
        # UNet Variants with ResNet50 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "ResNet50" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # iw*ih*ch
            # conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # iw*ih*mw
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # iw*ih*mw
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1_relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # (w/2)*(h/2)*(mw*2)
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # (w/2)*(h/2)*(mw*2)
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # (w/2)*(h/2)*(mw*2)
                for k in range(1, 2):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # (w/2)*(h/2)*(mw*2)
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # (w/2)*(h/2)*(mw*2)
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_out').output  # (w/4)*(h/4)*256
            conv2 = Base_Model.get_layer('conv2_block2_out').output  # (w/4)*(h/4)*256
            conv3 = Base_Model.get_layer('conv2_block3_out').output  # (w/4)*(h/4)*256
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # (w/4)*(h/4)*(mw*4)
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # (w/4)*(h/4)*(mw*4)
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # (w/4)*(h/4)*(mw*4)
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # (w/4)*(h/4)*(mw*4)
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # (w/4)*(h/4)*(mw*4)
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_out').output  # (w/8)*(h/8)*512
            conv2 = Base_Model.get_layer('conv3_block2_out').output  # (w/8)*(h/8)*512
            conv3 = Base_Model.get_layer('conv3_block3_out').output  # (w/8)*(h/8)*512
            conv4 = Base_Model.get_layer('conv3_block4_out').output  # (w/8)*(h/8)*512
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # (w/8)*(h/8)*(mw*8)
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # (w/8)*(h/8)*(mw*8)
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # (w/8)*(h/8)*(mw*8)
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # (w/8)*(h/8)*(mw*8)
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # (w/8)*(h/8)*(mw*8)
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_out').output  # (w/16)*(h/16)*1024
            conv2 = Base_Model.get_layer('conv4_block2_out').output  # (w/16)*(h/16)*1024
            conv3 = Base_Model.get_layer('conv4_block3_out').output  # (w/16)*(h/16)*1024
            conv4 = Base_Model.get_layer('conv4_block4_out').output  # (w/16)*(h/16)*1024
            conv5 = Base_Model.get_layer('conv4_block5_out').output  # (w/16)*(h/16)*1024
            conv6 = Base_Model.get_layer('conv4_block6_out').output  # (w/16)*(h/16)*1024
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # (w/16)*(h/16)*(mw*16)
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # (w/16)*(h/16)*(mw*16)
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # (w/16)*(h/16)*(mw*16)
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # (w/16)*(h/16)*(mw*16)
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # (w/16)*(h/16)*(mw*16)
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_out').output  # (w/32)*(h/32)*2048
                conv2 = Base_Model.get_layer('conv5_block2_out').output  # (w/32)*(h/32)*2048
                conv3 = Base_Model.get_layer('conv5_block3_out').output  # (w/32)*(h/32)*2048
                #
                encoder_level_6 = tf.keras.layers.add([conv1, conv2, conv3])
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs

            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # (w/32)*(h/32)*(mw*32)
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.dense_loop)  # (w/32)*(h/32)*(mw*32)
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def ResNet50V2(self):
        # UNet Variants with ResNet50V2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "ResNet50V2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('conv1_conv').output  # 112*112*64
            conv = tf.keras.layers.Activation('relu')(conv1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_preact_relu').output  # 56*56*64
            conv2 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*64
            conv3 = Base_Model.get_layer('conv2_block1_2_relu').output  # 56*56*64
            conv4 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*64
            conv5 = Base_Model.get_layer('conv2_block2_2_relu').output  # 56*56*64
            conv6 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*64
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv7 = Base_Model.get_layer('conv2_block2_preact_relu').output  # 56*56*256
            conv8 = Base_Model.get_layer('conv2_block3_preact_relu').output  # 56*56*256
            conv1_2 = tf.keras.layers.add([conv7, conv8])
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(conv, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv2_block3_2_relu').output  # 28*28*64
            #
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block1_2_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block2_2_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block3_2_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8])
            #
            conv9 = Base_Model.get_layer('conv3_block2_preact_relu').output  # 28*28*512
            conv10 = Base_Model.get_layer('conv3_block3_preact_relu').output  # 28*28*512
            conv11 = Base_Model.get_layer('conv3_block4_preact_relu').output  # 28*28*512
            conv1_2 = tf.keras.layers.add([conv9, conv10, conv11])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv3_block4_2_relu').output  # 14*14*128
            #
            conv2 = Base_Model.get_layer('conv4_block1_preact_relu').output  # 14*14*512
            #
            conv3 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*256
            conv4 = Base_Model.get_layer('conv4_block1_2_relu').output  # 14*14*256
            conv5 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*256
            conv6 = Base_Model.get_layer('conv4_block2_2_relu').output  # 14*14*256
            conv7 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*256
            conv8 = Base_Model.get_layer('conv4_block3_2_relu').output  # 14*14*256
            conv9 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*256
            conv10 = Base_Model.get_layer('conv4_block4_2_relu').output  # 14*14*256
            conv11 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*256
            conv12 = Base_Model.get_layer('conv4_block5_2_relu').output  # 14*14*256
            conv13 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*256
            conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('conv4_block2_preact_relu').output  # 14*14*1024
            conv15 = Base_Model.get_layer('conv4_block3_preact_relu').output  # 14*14*1024
            conv16 = Base_Model.get_layer('conv4_block4_preact_relu').output  # 14*14*1024
            conv17 = Base_Model.get_layer('conv4_block5_preact_relu').output  # 14*14*1024
            conv18 = Base_Model.get_layer('conv4_block6_preact_relu').output  # 14*14*1024
            conv1_2 = tf.keras.layers.add([conv14, conv15, conv16, conv17, conv18])
            #
            conv = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv4_block6_2_relu').output  # 7*7*256
                #
                conv2 = Base_Model.get_layer('conv5_block1_preact_relu').output  # 7*7*1024
                #
                conv3 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*512
                conv4 = Base_Model.get_layer('conv5_block1_2_relu').output  # 7*7*512
                conv5 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*512
                conv6 = Base_Model.get_layer('conv5_block2_2_relu').output  # 7*7*512
                conv7 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*512
                conv8 = Base_Model.get_layer('conv5_block3_2_relu').output  # 7*7*512
                conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8])
                #
                conv9 = Base_Model.get_layer('conv5_block2_preact_relu').output  # 7*7*2048
                conv10 = Base_Model.get_layer('conv5_block3_preact_relu').output  # 7*7*2048
                conv11 = Base_Model.get_layer('post_relu').output  # 7*7*2048
                conv1_2 = tf.keras.layers.add([conv9, conv10, conv11])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def ResNet101(self):
        # UNet Variants with ResNet101 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "ResNet101" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1_relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_out').output  # 56*56*256
            conv2 = Base_Model.get_layer('conv2_block2_out').output  # 56*56*256
            conv3 = Base_Model.get_layer('conv2_block3_out').output  # 56*56*256
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_out').output  # 28*28*512
            conv2 = Base_Model.get_layer('conv3_block2_out').output  # 28*28*512
            conv3 = Base_Model.get_layer('conv3_block3_out').output  # 28*28*512
            conv4 = Base_Model.get_layer('conv3_block4_out').output  # 28*28*512
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_out').output  # 14*14*1024
            conv2 = Base_Model.get_layer('conv4_block2_out').output  # 14*14*1024
            conv3 = Base_Model.get_layer('conv4_block3_out').output  # 14*14*1024
            conv4 = Base_Model.get_layer('conv4_block4_out').output  # 14*14*1024
            conv5 = Base_Model.get_layer('conv4_block5_out').output  # 14*14*1024
            conv6 = Base_Model.get_layer('conv4_block6_out').output  # 14*14*1024
            conv7 = Base_Model.get_layer('conv4_block7_out').output  # 14*14*1024
            conv8 = Base_Model.get_layer('conv4_block8_out').output  # 14*14*1024
            conv9 = Base_Model.get_layer('conv4_block9_out').output  # 14*14*1024
            conv10 = Base_Model.get_layer('conv4_block10_out').output  # 14*14*1024
            conv11 = Base_Model.get_layer('conv4_block11_out').output  # 14*14*1024
            conv12 = Base_Model.get_layer('conv4_block12_out').output  # 14*14*1024
            conv13 = Base_Model.get_layer('conv4_block13_out').output  # 14*14*1024
            conv14 = Base_Model.get_layer('conv4_block14_out').output  # 14*14*1024
            conv15 = Base_Model.get_layer('conv4_block15_out').output  # 14*14*1024
            conv16 = Base_Model.get_layer('conv4_block16_out').output  # 14*14*1024
            conv17 = Base_Model.get_layer('conv4_block17_out').output  # 14*14*1024
            conv18 = Base_Model.get_layer('conv4_block18_out').output  # 14*14*1024
            conv19 = Base_Model.get_layer('conv4_block19_out').output  # 14*14*1024
            conv20 = Base_Model.get_layer('conv4_block20_out').output  # 14*14*1024
            conv21 = Base_Model.get_layer('conv4_block21_out').output  # 14*14*1024
            conv22 = Base_Model.get_layer('conv4_block22_out').output  # 14*14*1024
            conv23 = Base_Model.get_layer('conv4_block23_out').output  # 14*14*1024
            #
            conv = tf.keras.layers.add(
                [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                 conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23]
            )
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_out').output  # 7*7*2048
                conv2 = Base_Model.get_layer('conv5_block2_out').output  # 7*7*2048
                conv3 = Base_Model.get_layer('conv5_block3_out').output  # 7*7*2048
                #
                encoder_level_6 = tf.keras.layers.add([conv1, conv2, conv3])
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def ResNet101V2(self):
        # UNet Variants with ResNet101V2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "ResNet101V2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('conv1_conv').output  # 112*112*64
            conv = tf.keras.layers.Activation('relu')(conv1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_preact_relu').output  # 56*56*64
            conv2 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*64
            conv3 = Base_Model.get_layer('conv2_block1_2_relu').output  # 56*56*64
            conv4 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*64
            conv5 = Base_Model.get_layer('conv2_block2_2_relu').output  # 56*56*64
            conv6 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*64
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv7 = Base_Model.get_layer('conv2_block2_preact_relu').output  # 56*56*256
            conv8 = Base_Model.get_layer('conv2_block3_preact_relu').output  # 56*56*256
            conv1_2 = tf.keras.layers.add([conv7, conv8])
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv2_block3_2_relu').output  # 28*28*64
            #
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block1_2_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block2_2_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block3_2_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8])
            #
            conv9 = Base_Model.get_layer('conv3_block2_preact_relu').output  # 28*28*512
            conv10 = Base_Model.get_layer('conv3_block3_preact_relu').output  # 28*28*512
            conv11 = Base_Model.get_layer('conv3_block4_preact_relu').output  # 28*28*512
            conv1_2 = tf.keras.layers.add([conv9, conv10, conv11])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv3_block4_2_relu').output  # 14*14*128
            #
            conv2 = Base_Model.get_layer('conv4_block1_preact_relu').output  # 14*14*512
            #
            conv3 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*256
            conv4 = Base_Model.get_layer('conv4_block1_2_relu').output  # 14*14*256
            conv5 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*256
            conv6 = Base_Model.get_layer('conv4_block2_2_relu').output  # 14*14*256
            conv7 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*256
            conv8 = Base_Model.get_layer('conv4_block3_2_relu').output  # 14*14*256
            conv9 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*256
            conv10 = Base_Model.get_layer('conv4_block4_2_relu').output  # 14*14*256
            conv11 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*256
            conv12 = Base_Model.get_layer('conv4_block5_2_relu').output  # 14*14*256
            conv13 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*256
            conv14 = Base_Model.get_layer('conv4_block6_2_relu').output  # 14*14*256
            conv15 = Base_Model.get_layer('conv4_block7_1_relu').output  # 14*14*256
            conv16 = Base_Model.get_layer('conv4_block7_2_relu').output  # 14*14*256
            conv17 = Base_Model.get_layer('conv4_block8_1_relu').output  # 14*14*256
            conv18 = Base_Model.get_layer('conv4_block8_2_relu').output  # 14*14*256
            conv19 = Base_Model.get_layer('conv4_block9_1_relu').output  # 14*14*256
            conv20 = Base_Model.get_layer('conv4_block9_2_relu').output  # 14*14*256
            conv21 = Base_Model.get_layer('conv4_block10_1_relu').output  # 14*14*256
            conv22 = Base_Model.get_layer('conv4_block10_2_relu').output  # 14*14*256
            conv23 = Base_Model.get_layer('conv4_block11_1_relu').output  # 14*14*256
            conv24 = Base_Model.get_layer('conv4_block11_2_relu').output  # 14*14*256
            conv25 = Base_Model.get_layer('conv4_block12_1_relu').output  # 14*14*256
            conv26 = Base_Model.get_layer('conv4_block12_2_relu').output  # 14*14*256
            conv27 = Base_Model.get_layer('conv4_block13_1_relu').output  # 14*14*256
            conv28 = Base_Model.get_layer('conv4_block13_2_relu').output  # 14*14*256
            conv29 = Base_Model.get_layer('conv4_block14_1_relu').output  # 14*14*256
            conv30 = Base_Model.get_layer('conv4_block14_2_relu').output  # 14*14*256
            conv31 = Base_Model.get_layer('conv4_block15_1_relu').output  # 14*14*256
            conv32 = Base_Model.get_layer('conv4_block15_2_relu').output  # 14*14*256
            conv33 = Base_Model.get_layer('conv4_block16_1_relu').output  # 14*14*256
            conv34 = Base_Model.get_layer('conv4_block16_2_relu').output  # 14*14*256
            conv35 = Base_Model.get_layer('conv4_block17_1_relu').output  # 14*14*256
            conv36 = Base_Model.get_layer('conv4_block17_2_relu').output  # 14*14*256
            conv37 = Base_Model.get_layer('conv4_block18_1_relu').output  # 14*14*256
            conv38 = Base_Model.get_layer('conv4_block18_2_relu').output  # 14*14*256
            conv39 = Base_Model.get_layer('conv4_block19_1_relu').output  # 14*14*256
            conv40 = Base_Model.get_layer('conv4_block19_2_relu').output  # 14*14*256
            conv41 = Base_Model.get_layer('conv4_block20_1_relu').output  # 14*14*256
            conv42 = Base_Model.get_layer('conv4_block20_2_relu').output  # 14*14*256
            conv43 = Base_Model.get_layer('conv4_block21_1_relu').output  # 14*14*256
            conv44 = Base_Model.get_layer('conv4_block21_2_relu').output  # 14*14*256
            conv45 = Base_Model.get_layer('conv4_block22_1_relu').output  # 14*14*256
            conv46 = Base_Model.get_layer('conv4_block22_2_relu').output  # 14*14*256
            conv47 = Base_Model.get_layer('conv4_block23_1_relu').output  # 14*14*256
            conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13,
                                           conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                                           conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35,
                                           conv36, conv37, conv38, conv39, conv40, conv41, conv42, conv43, conv44, conv45, conv46, conv47
                                           ])
            #
            conv48 = Base_Model.get_layer('conv4_block2_preact_relu').output  # 14*14*1024
            conv49 = Base_Model.get_layer('conv4_block3_preact_relu').output  # 14*14*1024
            conv50 = Base_Model.get_layer('conv4_block4_preact_relu').output  # 14*14*1024
            conv51 = Base_Model.get_layer('conv4_block5_preact_relu').output  # 14*14*1024
            conv52 = Base_Model.get_layer('conv4_block6_preact_relu').output  # 14*14*1024
            conv53 = Base_Model.get_layer('conv4_block7_preact_relu').output  # 14*14*1024
            conv54 = Base_Model.get_layer('conv4_block8_preact_relu').output  # 14*14*1024
            conv55 = Base_Model.get_layer('conv4_block9_preact_relu').output  # 14*14*1024
            conv56 = Base_Model.get_layer('conv4_block10_preact_relu').output  # 14*14*1024
            conv57 = Base_Model.get_layer('conv4_block11_preact_relu').output  # 14*14*1024
            conv58 = Base_Model.get_layer('conv4_block12_preact_relu').output  # 14*14*1024
            conv59 = Base_Model.get_layer('conv4_block13_preact_relu').output  # 14*14*1024
            conv60 = Base_Model.get_layer('conv4_block14_preact_relu').output  # 14*14*1024
            conv61 = Base_Model.get_layer('conv4_block15_preact_relu').output  # 14*14*1024
            conv62 = Base_Model.get_layer('conv4_block16_preact_relu').output  # 14*14*1024
            conv63 = Base_Model.get_layer('conv4_block17_preact_relu').output  # 14*14*1024
            conv64 = Base_Model.get_layer('conv4_block18_preact_relu').output  # 14*14*1024
            conv65 = Base_Model.get_layer('conv4_block19_preact_relu').output  # 14*14*1024
            conv66 = Base_Model.get_layer('conv4_block20_preact_relu').output  # 14*14*1024
            conv67 = Base_Model.get_layer('conv4_block21_preact_relu').output  # 14*14*1024
            conv68 = Base_Model.get_layer('conv4_block22_preact_relu').output  # 14*14*1024
            conv69 = Base_Model.get_layer('conv4_block23_preact_relu').output  # 14*14*1024
            conv1_2 = tf.keras.layers.add([conv48, conv49, conv50, conv51, conv52, conv53, conv54, conv55, conv56, conv57, conv58,
                                           conv59, conv60, conv61, conv62, conv63, conv64, conv65, conv66, conv67, conv68, conv69
                                           ])
            #
            conv = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv4_block23_2_relu').output  # 7*7*256
                #
                conv2 = Base_Model.get_layer('conv5_block1_preact_relu').output  # 7*7*1024
                #
                conv3 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*512
                conv4 = Base_Model.get_layer('conv5_block1_2_relu').output  # 7*7*512
                conv5 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*512
                conv6 = Base_Model.get_layer('conv5_block2_2_relu').output  # 7*7*512
                conv7 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*512
                conv8 = Base_Model.get_layer('conv5_block3_2_relu').output  # 7*7*512
                conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8])
                #
                conv9 = Base_Model.get_layer('conv5_block2_preact_relu').output  # 7*7*2048
                conv10 = Base_Model.get_layer('conv5_block3_preact_relu').output  # 7*7*2048
                conv11 = Base_Model.get_layer('post_relu').output  # 7*7*2048
                conv1_2 = tf.keras.layers.add([conv9, conv10, conv11])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def ResNet152(self):
        # UNet Variants with ResNet152 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "ResNet152" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1_relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_out').output  # 56*56*256
            conv2 = Base_Model.get_layer('conv2_block2_out').output  # 56*56*256
            conv3 = Base_Model.get_layer('conv2_block3_out').output  # 56*56*256
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_out').output  # 28*28*512
            conv2 = Base_Model.get_layer('conv3_block2_out').output  # 28*28*512
            conv3 = Base_Model.get_layer('conv3_block3_out').output  # 28*28*512
            conv4 = Base_Model.get_layer('conv3_block4_out').output  # 28*28*512
            conv5 = Base_Model.get_layer('conv3_block5_out').output  # 28*28*512
            conv6 = Base_Model.get_layer('conv3_block6_out').output  # 28*28*512
            conv7 = Base_Model.get_layer('conv3_block7_out').output  # 28*28*512
            conv8 = Base_Model.get_layer('conv3_block8_out').output  # 28*28*512
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_out').output  # 14*14*1024
            conv2 = Base_Model.get_layer('conv4_block2_out').output  # 14*14*1024
            conv3 = Base_Model.get_layer('conv4_block3_out').output  # 14*14*1024
            conv4 = Base_Model.get_layer('conv4_block4_out').output  # 14*14*1024
            conv5 = Base_Model.get_layer('conv4_block5_out').output  # 14*14*1024
            conv6 = Base_Model.get_layer('conv4_block6_out').output  # 14*14*1024
            conv7 = Base_Model.get_layer('conv4_block7_out').output  # 14*14*1024
            conv8 = Base_Model.get_layer('conv4_block8_out').output  # 14*14*1024
            conv9 = Base_Model.get_layer('conv4_block9_out').output  # 14*14*1024
            conv10 = Base_Model.get_layer('conv4_block10_out').output  # 14*14*1024
            conv11 = Base_Model.get_layer('conv4_block11_out').output  # 14*14*1024
            conv12 = Base_Model.get_layer('conv4_block12_out').output  # 14*14*1024
            conv13 = Base_Model.get_layer('conv4_block13_out').output  # 14*14*1024
            conv14 = Base_Model.get_layer('conv4_block14_out').output  # 14*14*1024
            conv15 = Base_Model.get_layer('conv4_block15_out').output  # 14*14*1024
            conv16 = Base_Model.get_layer('conv4_block16_out').output  # 14*14*1024
            conv17 = Base_Model.get_layer('conv4_block17_out').output  # 14*14*1024
            conv18 = Base_Model.get_layer('conv4_block18_out').output  # 14*14*1024
            conv19 = Base_Model.get_layer('conv4_block19_out').output  # 14*14*1024
            conv20 = Base_Model.get_layer('conv4_block20_out').output  # 14*14*1024
            conv21 = Base_Model.get_layer('conv4_block21_out').output  # 14*14*1024
            conv22 = Base_Model.get_layer('conv4_block22_out').output  # 14*14*1024
            conv23 = Base_Model.get_layer('conv4_block23_out').output  # 14*14*1024
            conv24 = Base_Model.get_layer('conv4_block24_out').output  # 14*14*1024
            conv25 = Base_Model.get_layer('conv4_block25_out').output  # 14*14*1024
            conv26 = Base_Model.get_layer('conv4_block26_out').output  # 14*14*1024
            conv27 = Base_Model.get_layer('conv4_block27_out').output  # 14*14*1024
            conv28 = Base_Model.get_layer('conv4_block28_out').output  # 14*14*1024
            conv29 = Base_Model.get_layer('conv4_block29_out').output  # 14*14*1024
            conv30 = Base_Model.get_layer('conv4_block30_out').output  # 14*14*1024
            conv31 = Base_Model.get_layer('conv4_block31_out').output  # 14*14*1024
            conv32 = Base_Model.get_layer('conv4_block32_out').output  # 14*14*1024
            conv33 = Base_Model.get_layer('conv4_block33_out').output  # 14*14*1024
            conv34 = Base_Model.get_layer('conv4_block34_out').output  # 14*14*1024
            conv35 = Base_Model.get_layer('conv4_block35_out').output  # 14*14*1024
            conv36 = Base_Model.get_layer('conv4_block36_out').output  # 14*14*1024
            #
            conv = tf.keras.layers.add(
                [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                 conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23,
                 conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35, conv36])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_out').output  # 7*7*2048
                conv2 = Base_Model.get_layer('conv5_block2_out').output  # 7*7*2048
                conv3 = Base_Model.get_layer('conv5_block3_out').output  # 7*7*2048
                #
                encoder_level_6 = tf.keras.layers.add([conv1, conv2, conv3])
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def ResNet152V2(self):
        # UNet Variants with ResNet152V2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "ResNet152V2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('conv1_conv').output  # 112*112*64
            conv = tf.keras.layers.Activation('relu')(conv1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_preact_relu').output  # 56*56*64
            conv2 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*64
            conv3 = Base_Model.get_layer('conv2_block1_2_relu').output  # 56*56*64
            conv4 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*64
            conv5 = Base_Model.get_layer('conv2_block2_2_relu').output  # 56*56*64
            conv6 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*64
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv7 = Base_Model.get_layer('conv2_block2_preact_relu').output  # 56*56*256
            conv8 = Base_Model.get_layer('conv2_block3_preact_relu').output  # 56*56*256
            conv1_2 = tf.keras.layers.add([conv7, conv8])
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv2_block3_2_relu').output  # 28*28*64
            #
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block1_2_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block2_2_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block3_2_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv9 = Base_Model.get_layer('conv3_block4_2_relu').output  # 28*28*128
            conv10 = Base_Model.get_layer('conv3_block5_1_relu').output  # 28*28*128
            conv11 = Base_Model.get_layer('conv3_block5_2_relu').output  # 28*28*128
            conv12 = Base_Model.get_layer('conv3_block6_1_relu').output  # 28*28*128
            conv13 = Base_Model.get_layer('conv3_block6_2_relu').output  # 28*28*128
            conv14 = Base_Model.get_layer('conv3_block7_1_relu').output  # 28*28*128
            conv15 = Base_Model.get_layer('conv3_block7_2_relu').output  # 28*28*128
            conv16 = Base_Model.get_layer('conv3_block8_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9,
                                           conv10, conv11, conv12, conv13, conv14, conv15, conv16])
            #
            conv9 = Base_Model.get_layer('conv3_block2_preact_relu').output  # 28*28*512
            conv10 = Base_Model.get_layer('conv3_block3_preact_relu').output  # 28*28*512
            conv11 = Base_Model.get_layer('conv3_block4_preact_relu').output  # 28*28*512
            conv12 = Base_Model.get_layer('conv3_block5_preact_relu').output  # 28*28*512
            conv13 = Base_Model.get_layer('conv3_block6_preact_relu').output  # 28*28*512
            conv14 = Base_Model.get_layer('conv3_block7_preact_relu').output  # 28*28*512
            conv15 = Base_Model.get_layer('conv3_block8_preact_relu').output  # 28*28*512
            conv1_2 = tf.keras.layers.add([conv9, conv10, conv11, conv12, conv13, conv14, conv15])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv3_block8_2_relu').output  # 14*14*128
            #
            conv2 = Base_Model.get_layer('conv4_block1_preact_relu').output  # 14*14*512
            #
            conv3 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*256
            conv4 = Base_Model.get_layer('conv4_block1_2_relu').output  # 14*14*256
            conv5 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*256
            conv6 = Base_Model.get_layer('conv4_block2_2_relu').output  # 14*14*256
            conv7 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*256
            conv8 = Base_Model.get_layer('conv4_block3_2_relu').output  # 14*14*256
            conv9 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*256
            conv10 = Base_Model.get_layer('conv4_block4_2_relu').output  # 14*14*256
            conv11 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*256
            conv12 = Base_Model.get_layer('conv4_block5_2_relu').output  # 14*14*256
            conv13 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*256
            conv14 = Base_Model.get_layer('conv4_block6_2_relu').output  # 14*14*256
            conv15 = Base_Model.get_layer('conv4_block7_1_relu').output  # 14*14*256
            conv16 = Base_Model.get_layer('conv4_block7_2_relu').output  # 14*14*256
            conv17 = Base_Model.get_layer('conv4_block8_1_relu').output  # 14*14*256
            conv18 = Base_Model.get_layer('conv4_block8_2_relu').output  # 14*14*256
            conv19 = Base_Model.get_layer('conv4_block9_1_relu').output  # 14*14*256
            conv20 = Base_Model.get_layer('conv4_block9_2_relu').output  # 14*14*256
            conv21 = Base_Model.get_layer('conv4_block10_1_relu').output  # 14*14*256
            conv22 = Base_Model.get_layer('conv4_block10_2_relu').output  # 14*14*256
            conv23 = Base_Model.get_layer('conv4_block11_1_relu').output  # 14*14*256
            conv24 = Base_Model.get_layer('conv4_block11_2_relu').output  # 14*14*256
            conv25 = Base_Model.get_layer('conv4_block12_1_relu').output  # 14*14*256
            conv26 = Base_Model.get_layer('conv4_block12_2_relu').output  # 14*14*256
            conv27 = Base_Model.get_layer('conv4_block13_1_relu').output  # 14*14*256
            conv28 = Base_Model.get_layer('conv4_block13_2_relu').output  # 14*14*256
            conv29 = Base_Model.get_layer('conv4_block14_1_relu').output  # 14*14*256
            conv30 = Base_Model.get_layer('conv4_block14_2_relu').output  # 14*14*256
            conv31 = Base_Model.get_layer('conv4_block15_1_relu').output  # 14*14*256
            conv32 = Base_Model.get_layer('conv4_block15_2_relu').output  # 14*14*256
            conv33 = Base_Model.get_layer('conv4_block16_1_relu').output  # 14*14*256
            conv34 = Base_Model.get_layer('conv4_block16_2_relu').output  # 14*14*256
            conv35 = Base_Model.get_layer('conv4_block17_1_relu').output  # 14*14*256
            conv36 = Base_Model.get_layer('conv4_block17_2_relu').output  # 14*14*256
            conv37 = Base_Model.get_layer('conv4_block18_1_relu').output  # 14*14*256
            conv38 = Base_Model.get_layer('conv4_block18_2_relu').output  # 14*14*256
            conv39 = Base_Model.get_layer('conv4_block19_1_relu').output  # 14*14*256
            conv40 = Base_Model.get_layer('conv4_block19_2_relu').output  # 14*14*256
            conv41 = Base_Model.get_layer('conv4_block20_1_relu').output  # 14*14*256
            conv42 = Base_Model.get_layer('conv4_block20_2_relu').output  # 14*14*256
            conv43 = Base_Model.get_layer('conv4_block21_1_relu').output  # 14*14*256
            conv44 = Base_Model.get_layer('conv4_block21_2_relu').output  # 14*14*256
            conv45 = Base_Model.get_layer('conv4_block22_1_relu').output  # 14*14*256
            conv46 = Base_Model.get_layer('conv4_block22_2_relu').output  # 14*14*256
            conv47 = Base_Model.get_layer('conv4_block23_1_relu').output  # 14*14*256
            conv48 = Base_Model.get_layer('conv4_block23_2_relu').output  # 14*14*256
            conv49 = Base_Model.get_layer('conv4_block24_1_relu').output  # 14*14*256
            conv50 = Base_Model.get_layer('conv4_block24_2_relu').output  # 14*14*256
            conv51 = Base_Model.get_layer('conv4_block25_1_relu').output  # 14*14*256
            conv52 = Base_Model.get_layer('conv4_block25_2_relu').output  # 14*14*256
            conv53 = Base_Model.get_layer('conv4_block26_1_relu').output  # 14*14*256
            conv54 = Base_Model.get_layer('conv4_block26_2_relu').output  # 14*14*256
            conv55 = Base_Model.get_layer('conv4_block27_1_relu').output  # 14*14*256
            conv56 = Base_Model.get_layer('conv4_block27_2_relu').output  # 14*14*256
            conv57 = Base_Model.get_layer('conv4_block28_1_relu').output  # 14*14*256
            conv58 = Base_Model.get_layer('conv4_block28_2_relu').output  # 14*14*256
            conv59 = Base_Model.get_layer('conv4_block29_1_relu').output  # 14*14*256
            conv60 = Base_Model.get_layer('conv4_block29_2_relu').output  # 14*14*256
            conv61 = Base_Model.get_layer('conv4_block30_1_relu').output  # 14*14*256
            conv62 = Base_Model.get_layer('conv4_block30_2_relu').output  # 14*14*256
            conv63 = Base_Model.get_layer('conv4_block31_1_relu').output  # 14*14*256
            conv64 = Base_Model.get_layer('conv4_block31_2_relu').output  # 14*14*256
            conv65 = Base_Model.get_layer('conv4_block32_1_relu').output  # 14*14*256
            conv66 = Base_Model.get_layer('conv4_block32_2_relu').output  # 14*14*256
            conv67 = Base_Model.get_layer('conv4_block33_1_relu').output  # 14*14*256
            conv68 = Base_Model.get_layer('conv4_block33_2_relu').output  # 14*14*256
            conv69 = Base_Model.get_layer('conv4_block34_1_relu').output  # 14*14*256
            conv70 = Base_Model.get_layer('conv4_block34_2_relu').output  # 14*14*256
            conv71 = Base_Model.get_layer('conv4_block35_1_relu').output  # 14*14*256
            conv72 = Base_Model.get_layer('conv4_block35_2_relu').output  # 14*14*256
            conv73 = Base_Model.get_layer('conv4_block36_1_relu').output  # 14*14*256
            conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13,
                                           conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                                           conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35,
                                           conv36, conv37, conv38, conv39, conv40, conv41, conv42, conv43, conv44, conv45, conv46,
                                           conv47, conv48, conv49, conv50, conv51, conv52, conv53, conv54, conv55, conv56, conv57,
                                           conv58, conv59, conv60, conv61, conv62, conv63, conv64, conv65, conv66, conv67, conv68,
                                           conv69, conv70, conv71, conv72, conv73
                                           ])
            #
            conv74 = Base_Model.get_layer('conv4_block2_preact_relu').output  # 14*14*1024
            conv75 = Base_Model.get_layer('conv4_block3_preact_relu').output  # 14*14*1024
            conv76 = Base_Model.get_layer('conv4_block4_preact_relu').output  # 14*14*1024
            conv77 = Base_Model.get_layer('conv4_block5_preact_relu').output  # 14*14*1024
            conv78 = Base_Model.get_layer('conv4_block6_preact_relu').output  # 14*14*1024
            conv79 = Base_Model.get_layer('conv4_block7_preact_relu').output  # 14*14*1024
            conv80 = Base_Model.get_layer('conv4_block8_preact_relu').output  # 14*14*1024
            conv81 = Base_Model.get_layer('conv4_block9_preact_relu').output  # 14*14*1024
            conv82 = Base_Model.get_layer('conv4_block10_preact_relu').output  # 14*14*1024
            conv83 = Base_Model.get_layer('conv4_block11_preact_relu').output  # 14*14*1024
            conv84 = Base_Model.get_layer('conv4_block12_preact_relu').output  # 14*14*1024
            conv85 = Base_Model.get_layer('conv4_block13_preact_relu').output  # 14*14*1024
            conv86 = Base_Model.get_layer('conv4_block14_preact_relu').output  # 14*14*1024
            conv87 = Base_Model.get_layer('conv4_block15_preact_relu').output  # 14*14*1024
            conv88 = Base_Model.get_layer('conv4_block16_preact_relu').output  # 14*14*1024
            conv89 = Base_Model.get_layer('conv4_block17_preact_relu').output  # 14*14*1024
            conv90 = Base_Model.get_layer('conv4_block18_preact_relu').output  # 14*14*1024
            conv91 = Base_Model.get_layer('conv4_block19_preact_relu').output  # 14*14*1024
            conv92 = Base_Model.get_layer('conv4_block20_preact_relu').output  # 14*14*1024
            conv93 = Base_Model.get_layer('conv4_block21_preact_relu').output  # 14*14*1024
            conv94 = Base_Model.get_layer('conv4_block22_preact_relu').output  # 14*14*1024
            conv95 = Base_Model.get_layer('conv4_block23_preact_relu').output  # 14*14*1024
            conv96 = Base_Model.get_layer('conv4_block24_preact_relu').output  # 14*14*1024
            conv97 = Base_Model.get_layer('conv4_block25_preact_relu').output  # 14*14*1024
            conv98 = Base_Model.get_layer('conv4_block26_preact_relu').output  # 14*14*1024
            conv99 = Base_Model.get_layer('conv4_block27_preact_relu').output  # 14*14*1024
            conv100 = Base_Model.get_layer('conv4_block28_preact_relu').output  # 14*14*1024
            conv101 = Base_Model.get_layer('conv4_block29_preact_relu').output  # 14*14*1024
            conv102 = Base_Model.get_layer('conv4_block30_preact_relu').output  # 14*14*1024
            conv103 = Base_Model.get_layer('conv4_block31_preact_relu').output  # 14*14*1024
            conv104 = Base_Model.get_layer('conv4_block32_preact_relu').output  # 14*14*1024
            conv105 = Base_Model.get_layer('conv4_block33_preact_relu').output  # 14*14*1024
            conv106 = Base_Model.get_layer('conv4_block34_preact_relu').output  # 14*14*1024
            conv107 = Base_Model.get_layer('conv4_block35_preact_relu').output  # 14*14*1024
            conv108 = Base_Model.get_layer('conv4_block36_preact_relu').output  # 14*14*1024
            conv1_2 = tf.keras.layers.add([conv74, conv75, conv76, conv77, conv78, conv79, conv80, conv81, conv82, conv83, conv84,
                                           conv85, conv86, conv87, conv88, conv89, conv90, conv91, conv92, conv93, conv94, conv95,
                                           conv96, conv97, conv98, conv99, conv100, conv101, conv102, conv103, conv104, conv105, conv106, conv107, conv108
                                           ])
            #
            conv = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv4_block36_2_relu').output  # 7*7*256
                #
                conv2 = Base_Model.get_layer('conv5_block1_preact_relu').output  # 7*7*1024
                #
                conv3 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*512
                conv4 = Base_Model.get_layer('conv5_block1_2_relu').output  # 7*7*512
                conv5 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*512
                conv6 = Base_Model.get_layer('conv5_block2_2_relu').output  # 7*7*512
                conv7 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*512
                conv8 = Base_Model.get_layer('conv5_block3_2_relu').output  # 7*7*512
                conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8])
                #
                conv9 = Base_Model.get_layer('conv5_block2_preact_relu').output  # 7*7*2048
                conv10 = Base_Model.get_layer('conv5_block3_preact_relu').output  # 7*7*2048
                conv11 = Base_Model.get_layer('post_relu').output  # 7*7*2048
                conv1_2 = tf.keras.layers.add([conv9, conv10, conv11])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def VGG16(self):
        # UNet Variants with VGG16 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "VGG16" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block2_conv1').output  # 112*112*128
            conv2 = Base_Model.get_layer('block2_conv2').output  # 112*112*128
            conv = tf.keras.layers.add([conv1, conv2])  # 112*112*128
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block3_conv1').output  # 56*56*256
            conv2 = Base_Model.get_layer('block3_conv2').output  # 56*56*256
            conv3 = Base_Model.get_layer('block3_conv3').output  # 56*56*256
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block4_conv1').output  # 28*28*512
            conv2 = Base_Model.get_layer('block4_conv2').output  # 28*28*512
            conv3 = Base_Model.get_layer('block4_conv3').output  # 28*28*512
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block5_conv1').output  # 14*14*1024
            conv2 = Base_Model.get_layer('block5_conv2').output  # 14*14*1024
            conv3 = Base_Model.get_layer('block5_conv3').output  # 14*14*1024
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                encoder_level_6 = Base_Model.get_layer('block5_pool').output  # 7*7*512
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def VGG19(self):
        # UNet Variants with VGG19 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "VGG19" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block2_conv1').output  # 112*112*128
            conv2 = Base_Model.get_layer('block2_conv2').output  # 112*112*128
            conv = tf.keras.layers.add([conv1, conv2])  # 112*112*128
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block3_conv1').output  # 56*56*256
            conv2 = Base_Model.get_layer('block3_conv2').output  # 56*56*256
            conv3 = Base_Model.get_layer('block3_conv3').output  # 56*56*256
            conv4 = Base_Model.get_layer('block3_conv4').output  # 56*56*256
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block4_conv1').output  # 28*28*512
            conv2 = Base_Model.get_layer('block4_conv2').output  # 28*28*512
            conv3 = Base_Model.get_layer('block4_conv3').output  # 28*28*512
            conv4 = Base_Model.get_layer('block4_conv4').output  # 28*28*512
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block5_conv1').output  # 14*14*512
            conv2 = Base_Model.get_layer('block5_conv2').output  # 14*14*512
            conv3 = Base_Model.get_layer('block5_conv3').output  # 14*14*512
            conv4 = Base_Model.get_layer('block5_conv4').output  # 14*14*512
            #
            conv = tf.keras.layers.add([conv1, conv2, conv3, conv4])
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                encoder_level_6 = Base_Model.get_layer('block5_pool').output  # 7*7*512
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def DenseNet121(self):
        # UNet Variants with DenseNet121 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "DenseNet121" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels), name='input')

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1/relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*128
            conv2 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*128
            conv3 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*128
            conv4 = Base_Model.get_layer('conv2_block4_1_relu').output  # 56*56*128
            conv5 = Base_Model.get_layer('conv2_block5_1_relu').output  # 56*56*128
            conv6 = Base_Model.get_layer('conv2_block6_1_relu').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv10 = Base_Model.get_layer('pool2_relu').output  # 56*56*256
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv10], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_0_relu').output  # 28*28*128
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block5_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block6_1_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block7_1_relu').output  # 28*28*128
            conv9 = Base_Model.get_layer('conv3_block8_1_relu').output  # 28*28*128
            conv10 = Base_Model.get_layer('conv3_block9_1_relu').output  # 28*28*128
            conv11 = Base_Model.get_layer('conv3_block10_1_relu').output  # 28*28*128
            conv12 = Base_Model.get_layer('conv3_block11_1_relu').output  # 28*28*128
            conv13 = Base_Model.get_layer('conv3_block12_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('pool3_relu').output  # 28*28*512
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv14], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*128
            conv2 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*128
            conv3 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*128
            conv4 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*128
            conv5 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*128
            conv6 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*128
            conv7 = Base_Model.get_layer('conv4_block7_1_relu').output  # 14*14*128
            conv8 = Base_Model.get_layer('conv4_block8_1_relu').output  # 14*14*128
            conv9 = Base_Model.get_layer('conv4_block9_1_relu').output  # 14*14*128
            conv10 = Base_Model.get_layer('conv4_block10_1_relu').output  # 14*14*128
            conv11 = Base_Model.get_layer('conv4_block11_1_relu').output  # 14*14*128
            conv12 = Base_Model.get_layer('conv4_block12_1_relu').output  # 14*14*128
            conv13 = Base_Model.get_layer('conv4_block13_1_relu').output  # 14*14*128
            conv14 = Base_Model.get_layer('conv4_block14_1_relu').output  # 14*14*128
            conv15 = Base_Model.get_layer('conv4_block15_1_relu').output  # 14*14*128
            conv16 = Base_Model.get_layer('conv4_block16_1_relu').output  # 14*14*128
            conv17 = Base_Model.get_layer('conv4_block17_1_relu').output  # 14*14*128
            conv18 = Base_Model.get_layer('conv4_block18_1_relu').output  # 14*14*128
            conv19 = Base_Model.get_layer('conv4_block19_1_relu').output  # 14*14*128
            conv20 = Base_Model.get_layer('conv4_block20_1_relu').output  # 14*14*128
            conv21 = Base_Model.get_layer('conv4_block21_1_relu').output  # 14*14*128
            conv22 = Base_Model.get_layer('conv4_block22_1_relu').output  # 14*14*128
            conv23 = Base_Model.get_layer('conv4_block23_1_relu').output  # 14*14*128
            conv24 = Base_Model.get_layer('conv4_block24_1_relu').output  # 14*14*128
            #
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                                           conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24])
            #
            conv25 = Base_Model.get_layer('pool4_relu').output  # 14*14*512
            conv = tf.keras.layers.concatenate([conv1_1, conv25], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*128
                conv2 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*128
                conv3 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*128
                conv4 = Base_Model.get_layer('conv5_block4_1_relu').output  # 7*7*128
                conv5 = Base_Model.get_layer('conv5_block5_1_relu').output  # 7*7*128
                conv6 = Base_Model.get_layer('conv5_block6_1_relu').output  # 7*7*128
                conv7 = Base_Model.get_layer('conv5_block7_1_relu').output  # 7*7*128
                conv8 = Base_Model.get_layer('conv5_block8_1_relu').output  # 7*7*128
                conv9 = Base_Model.get_layer('conv5_block9_1_relu').output  # 7*7*128
                conv10 = Base_Model.get_layer('conv5_block10_1_relu').output  # 7*7*128
                conv11 = Base_Model.get_layer('conv5_block11_1_relu').output  # 7*7*128
                conv12 = Base_Model.get_layer('conv5_block12_1_relu').output  # 7*7*128
                conv13 = Base_Model.get_layer('conv5_block13_1_relu').output  # 7*7*128
                conv14 = Base_Model.get_layer('conv5_block14_1_relu').output  # 7*7*128
                conv15 = Base_Model.get_layer('conv5_block15_1_relu').output  # 7*7*128
                conv16 = Base_Model.get_layer('conv5_block16_1_relu').output  # 7*7*128
                conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
                                               conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16])
                #
                conv17 = Base_Model.get_layer('relu').output  # 7*7*1024
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1_1, conv17], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def DenseNet169(self):
        # UNet Variants with DenseNet169 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "DenseNet169" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1/relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*128
            conv2 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*128
            conv3 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*128
            conv4 = Base_Model.get_layer('conv2_block4_1_relu').output  # 56*56*128
            conv5 = Base_Model.get_layer('conv2_block5_1_relu').output  # 56*56*128
            conv6 = Base_Model.get_layer('conv2_block6_1_relu').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv10 = Base_Model.get_layer('pool2_relu').output  # 56*56*256
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv10], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_0_relu').output  # 28*28*128
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block5_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block6_1_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block7_1_relu').output  # 28*28*128
            conv9 = Base_Model.get_layer('conv3_block8_1_relu').output  # 28*28*128
            conv10 = Base_Model.get_layer('conv3_block9_1_relu').output  # 28*28*128
            conv11 = Base_Model.get_layer('conv3_block10_1_relu').output  # 28*28*128
            conv12 = Base_Model.get_layer('conv3_block11_1_relu').output  # 28*28*128
            conv13 = Base_Model.get_layer('conv3_block12_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('pool3_relu').output  # 28*28*512
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv14], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*128
            conv2 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*128
            conv3 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*128
            conv4 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*128
            conv5 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*128
            conv6 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*128
            conv7 = Base_Model.get_layer('conv4_block7_1_relu').output  # 14*14*128
            conv8 = Base_Model.get_layer('conv4_block8_1_relu').output  # 14*14*128
            conv9 = Base_Model.get_layer('conv4_block9_1_relu').output  # 14*14*128
            conv10 = Base_Model.get_layer('conv4_block10_1_relu').output  # 14*14*128
            conv11 = Base_Model.get_layer('conv4_block11_1_relu').output  # 14*14*128
            conv12 = Base_Model.get_layer('conv4_block12_1_relu').output  # 14*14*128
            conv13 = Base_Model.get_layer('conv4_block13_1_relu').output  # 14*14*128
            conv14 = Base_Model.get_layer('conv4_block14_1_relu').output  # 14*14*128
            conv15 = Base_Model.get_layer('conv4_block15_1_relu').output  # 14*14*128
            conv16 = Base_Model.get_layer('conv4_block16_1_relu').output  # 14*14*128
            conv17 = Base_Model.get_layer('conv4_block17_1_relu').output  # 14*14*128
            conv18 = Base_Model.get_layer('conv4_block18_1_relu').output  # 14*14*128
            conv19 = Base_Model.get_layer('conv4_block19_1_relu').output  # 14*14*128
            conv20 = Base_Model.get_layer('conv4_block20_1_relu').output  # 14*14*128
            conv21 = Base_Model.get_layer('conv4_block21_1_relu').output  # 14*14*128
            conv22 = Base_Model.get_layer('conv4_block22_1_relu').output  # 14*14*128
            conv23 = Base_Model.get_layer('conv4_block23_1_relu').output  # 14*14*128
            conv24 = Base_Model.get_layer('conv4_block24_1_relu').output  # 14*14*128
            conv25 = Base_Model.get_layer('conv4_block25_1_relu').output  # 14*14*128
            conv26 = Base_Model.get_layer('conv4_block26_1_relu').output  # 14*14*128
            conv27 = Base_Model.get_layer('conv4_block27_1_relu').output  # 14*14*128
            conv28 = Base_Model.get_layer('conv4_block28_1_relu').output  # 14*14*128
            conv29 = Base_Model.get_layer('conv4_block29_1_relu').output  # 14*14*128
            conv30 = Base_Model.get_layer('conv4_block30_1_relu').output  # 14*14*128
            conv31 = Base_Model.get_layer('conv4_block31_1_relu').output  # 14*14*128
            conv32 = Base_Model.get_layer('conv4_block32_1_relu').output  # 14*14*128
            #
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                                           conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                                           conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32])
            #
            conv25 = Base_Model.get_layer('pool4_relu').output  # 14*14*512
            conv = tf.keras.layers.concatenate([conv1_1, conv25], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*128
                conv2 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*128
                conv3 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*128
                conv4 = Base_Model.get_layer('conv5_block4_1_relu').output  # 7*7*128
                conv5 = Base_Model.get_layer('conv5_block5_1_relu').output  # 7*7*128
                conv6 = Base_Model.get_layer('conv5_block6_1_relu').output  # 7*7*128
                conv7 = Base_Model.get_layer('conv5_block7_1_relu').output  # 7*7*128
                conv8 = Base_Model.get_layer('conv5_block8_1_relu').output  # 7*7*128
                conv9 = Base_Model.get_layer('conv5_block9_1_relu').output  # 7*7*128
                conv10 = Base_Model.get_layer('conv5_block10_1_relu').output  # 7*7*128
                conv11 = Base_Model.get_layer('conv5_block11_1_relu').output  # 7*7*128
                conv12 = Base_Model.get_layer('conv5_block12_1_relu').output  # 7*7*128
                conv13 = Base_Model.get_layer('conv5_block13_1_relu').output  # 7*7*128
                conv14 = Base_Model.get_layer('conv5_block14_1_relu').output  # 7*7*128
                conv15 = Base_Model.get_layer('conv5_block15_1_relu').output  # 7*7*128
                conv16 = Base_Model.get_layer('conv5_block16_1_relu').output  # 7*7*128
                conv17 = Base_Model.get_layer('conv5_block17_1_relu').output  # 7*7*128
                conv18 = Base_Model.get_layer('conv5_block18_1_relu').output  # 7*7*128
                conv19 = Base_Model.get_layer('conv5_block19_1_relu').output  # 7*7*128
                conv20 = Base_Model.get_layer('conv5_block20_1_relu').output  # 7*7*128
                conv21 = Base_Model.get_layer('conv5_block21_1_relu').output  # 7*7*128
                conv22 = Base_Model.get_layer('conv5_block22_1_relu').output  # 7*7*128
                conv23 = Base_Model.get_layer('conv5_block23_1_relu').output  # 7*7*128
                conv24 = Base_Model.get_layer('conv5_block24_1_relu').output  # 7*7*128
                conv25 = Base_Model.get_layer('conv5_block25_1_relu').output  # 7*7*128
                conv26 = Base_Model.get_layer('conv5_block26_1_relu').output  # 7*7*128
                conv27 = Base_Model.get_layer('conv5_block27_1_relu').output  # 7*7*128
                conv28 = Base_Model.get_layer('conv5_block28_1_relu').output  # 7*7*128
                conv29 = Base_Model.get_layer('conv5_block29_1_relu').output  # 7*7*128
                conv30 = Base_Model.get_layer('conv5_block30_1_relu').output  # 7*7*128
                conv31 = Base_Model.get_layer('conv5_block31_1_relu').output  # 7*7*128
                conv32 = Base_Model.get_layer('conv5_block32_1_relu').output  # 7*7*128
                conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16,
                                               conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32])
                #
                conv33 = Base_Model.get_layer('relu').output  # 7*7*1024
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1_1, conv33], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def DenseNet201(self):
        # UNet Variants with DenseNet121 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "DenseNet201" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels), name='input')

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1/relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*128
            conv2 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*128
            conv3 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*128
            conv4 = Base_Model.get_layer('conv2_block4_1_relu').output  # 56*56*128
            conv5 = Base_Model.get_layer('conv2_block5_1_relu').output  # 56*56*128
            conv6 = Base_Model.get_layer('conv2_block6_1_relu').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv10 = Base_Model.get_layer('pool2_relu').output  # 56*56*256
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv10], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_0_relu').output  # 28*28*128
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block5_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block6_1_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block7_1_relu').output  # 28*28*128
            conv9 = Base_Model.get_layer('conv3_block8_1_relu').output  # 28*28*128
            conv10 = Base_Model.get_layer('conv3_block9_1_relu').output  # 28*28*128
            conv11 = Base_Model.get_layer('conv3_block10_1_relu').output  # 28*28*128
            conv12 = Base_Model.get_layer('conv3_block11_1_relu').output  # 28*28*128
            conv13 = Base_Model.get_layer('conv3_block12_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('pool3_relu').output  # 28*28*512
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv14], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*128
            conv2 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*128
            conv3 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*128
            conv4 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*128
            conv5 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*128
            conv6 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*128
            conv7 = Base_Model.get_layer('conv4_block7_1_relu').output  # 14*14*128
            conv8 = Base_Model.get_layer('conv4_block8_1_relu').output  # 14*14*128
            conv9 = Base_Model.get_layer('conv4_block9_1_relu').output  # 14*14*128
            conv10 = Base_Model.get_layer('conv4_block10_1_relu').output  # 14*14*128
            conv11 = Base_Model.get_layer('conv4_block11_1_relu').output  # 14*14*128
            conv12 = Base_Model.get_layer('conv4_block12_1_relu').output  # 14*14*128
            conv13 = Base_Model.get_layer('conv4_block13_1_relu').output  # 14*14*128
            conv14 = Base_Model.get_layer('conv4_block14_1_relu').output  # 14*14*128
            conv15 = Base_Model.get_layer('conv4_block15_1_relu').output  # 14*14*128
            conv16 = Base_Model.get_layer('conv4_block16_1_relu').output  # 14*14*128
            conv17 = Base_Model.get_layer('conv4_block17_1_relu').output  # 14*14*128
            conv18 = Base_Model.get_layer('conv4_block18_1_relu').output  # 14*14*128
            conv19 = Base_Model.get_layer('conv4_block19_1_relu').output  # 14*14*128
            conv20 = Base_Model.get_layer('conv4_block20_1_relu').output  # 14*14*128
            conv21 = Base_Model.get_layer('conv4_block21_1_relu').output  # 14*14*128
            conv22 = Base_Model.get_layer('conv4_block22_1_relu').output  # 14*14*128
            conv23 = Base_Model.get_layer('conv4_block23_1_relu').output  # 14*14*128
            conv24 = Base_Model.get_layer('conv4_block24_1_relu').output  # 14*14*128
            conv25 = Base_Model.get_layer('conv4_block25_1_relu').output  # 14*14*128
            conv26 = Base_Model.get_layer('conv4_block26_1_relu').output  # 14*14*128
            conv27 = Base_Model.get_layer('conv4_block27_1_relu').output  # 14*14*128
            conv28 = Base_Model.get_layer('conv4_block28_1_relu').output  # 14*14*128
            conv29 = Base_Model.get_layer('conv4_block29_1_relu').output  # 14*14*128
            conv30 = Base_Model.get_layer('conv4_block30_1_relu').output  # 14*14*128
            conv31 = Base_Model.get_layer('conv4_block31_1_relu').output  # 14*14*128
            conv32 = Base_Model.get_layer('conv4_block32_1_relu').output  # 14*14*128
            conv33 = Base_Model.get_layer('conv4_block33_1_relu').output  # 14*14*128
            conv34 = Base_Model.get_layer('conv4_block34_1_relu').output  # 14*14*128
            conv35 = Base_Model.get_layer('conv4_block35_1_relu').output  # 14*14*128
            conv36 = Base_Model.get_layer('conv4_block36_1_relu').output  # 14*14*128
            conv37 = Base_Model.get_layer('conv4_block37_1_relu').output  # 14*14*128
            conv38 = Base_Model.get_layer('conv4_block38_1_relu').output  # 14*14*128
            conv39 = Base_Model.get_layer('conv4_block39_1_relu').output  # 14*14*128
            conv40 = Base_Model.get_layer('conv4_block40_1_relu').output  # 14*14*128
            conv41 = Base_Model.get_layer('conv4_block41_1_relu').output  # 14*14*128
            conv42 = Base_Model.get_layer('conv4_block42_1_relu').output  # 14*14*128
            conv43 = Base_Model.get_layer('conv4_block43_1_relu').output  # 14*14*128
            conv44 = Base_Model.get_layer('conv4_block44_1_relu').output  # 14*14*128
            conv45 = Base_Model.get_layer('conv4_block45_1_relu').output  # 14*14*128
            conv46 = Base_Model.get_layer('conv4_block46_1_relu').output  # 14*14*128
            conv47 = Base_Model.get_layer('conv4_block47_1_relu').output  # 14*14*128
            conv48 = Base_Model.get_layer('conv4_block48_1_relu').output  # 14*14*128
            #
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                                           conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                                           conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35, conv36,
                                           conv37, conv38, conv39, conv40, conv41, conv42, conv43, conv44, conv45, conv46, conv47, conv48])
            #
            conv25 = Base_Model.get_layer('pool4_relu').output  # 14*14*512
            conv = tf.keras.layers.concatenate([conv1_1, conv25], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*128
                conv2 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*128
                conv3 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*128
                conv4 = Base_Model.get_layer('conv5_block4_1_relu').output  # 7*7*128
                conv5 = Base_Model.get_layer('conv5_block5_1_relu').output  # 7*7*128
                conv6 = Base_Model.get_layer('conv5_block6_1_relu').output  # 7*7*128
                conv7 = Base_Model.get_layer('conv5_block7_1_relu').output  # 7*7*128
                conv8 = Base_Model.get_layer('conv5_block8_1_relu').output  # 7*7*128
                conv9 = Base_Model.get_layer('conv5_block9_1_relu').output  # 7*7*128
                conv10 = Base_Model.get_layer('conv5_block10_1_relu').output  # 7*7*128
                conv11 = Base_Model.get_layer('conv5_block11_1_relu').output  # 7*7*128
                conv12 = Base_Model.get_layer('conv5_block12_1_relu').output  # 7*7*128
                conv13 = Base_Model.get_layer('conv5_block13_1_relu').output  # 7*7*128
                conv14 = Base_Model.get_layer('conv5_block14_1_relu').output  # 7*7*128
                conv15 = Base_Model.get_layer('conv5_block15_1_relu').output  # 7*7*128
                conv16 = Base_Model.get_layer('conv5_block16_1_relu').output  # 7*7*128
                conv17 = Base_Model.get_layer('conv5_block17_1_relu').output  # 7*7*128
                conv18 = Base_Model.get_layer('conv5_block18_1_relu').output  # 7*7*128
                conv19 = Base_Model.get_layer('conv5_block19_1_relu').output  # 7*7*128
                conv20 = Base_Model.get_layer('conv5_block20_1_relu').output  # 7*7*128
                conv21 = Base_Model.get_layer('conv5_block21_1_relu').output  # 7*7*128
                conv22 = Base_Model.get_layer('conv5_block22_1_relu').output  # 7*7*128
                conv23 = Base_Model.get_layer('conv5_block23_1_relu').output  # 7*7*128
                conv24 = Base_Model.get_layer('conv5_block24_1_relu').output  # 7*7*128
                conv25 = Base_Model.get_layer('conv5_block25_1_relu').output  # 7*7*128
                conv26 = Base_Model.get_layer('conv5_block26_1_relu').output  # 7*7*128
                conv27 = Base_Model.get_layer('conv5_block27_1_relu').output  # 7*7*128
                conv28 = Base_Model.get_layer('conv5_block28_1_relu').output  # 7*7*128
                conv29 = Base_Model.get_layer('conv5_block29_1_relu').output  # 7*7*128
                conv30 = Base_Model.get_layer('conv5_block30_1_relu').output  # 7*7*128
                conv31 = Base_Model.get_layer('conv5_block31_1_relu').output  # 7*7*128
                conv32 = Base_Model.get_layer('conv5_block32_1_relu').output  # 7*7*128
                conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16,
                                               conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32])
                #
                conv33 = Base_Model.get_layer('relu').output  # 7*7*1024
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1_1, conv33], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv

        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)

        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def MobileNet(self):
        # UNet Variants with MobileNet ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "MobileNet" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('conv1_relu').output  # 112*112*32
            conv2 = Base_Model.get_layer('conv_dw_1_relu').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('conv_pw_1_relu').output  # 112*112*64
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv_dw_2_relu').output  # 56*56*64
            #
            conv2 = Base_Model.get_layer('conv_pw_2_relu').output  # 56*56*128
            conv3 = Base_Model.get_layer('conv_dw_3_relu').output  # 56*56*128
            conv4 = Base_Model.get_layer('conv_pw_3_relu').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv_dw_4_relu').output  # 28*28*128
            #
            conv2 = Base_Model.get_layer('conv_pw_4_relu').output  # 28*28*256
            conv3 = Base_Model.get_layer('conv_dw_5_relu').output  # 28*28*256
            conv4 = Base_Model.get_layer('conv_pw_5_relu').output  # 28*28*256
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('conv_dw_6_relu').output  # 14*14*256
            #
            conv2 = Base_Model.get_layer('conv_pw_6_relu').output  # 14*14*512
            conv3 = Base_Model.get_layer('conv_dw_7_relu').output  # 14*14*512
            conv4 = Base_Model.get_layer('conv_pw_7_relu').output  # 14*14*512
            conv5 = Base_Model.get_layer('conv_dw_8_relu').output  # 14*14*512
            conv6 = Base_Model.get_layer('conv_pw_8_relu').output  # 14*14*512
            conv7 = Base_Model.get_layer('conv_dw_9_relu').output  # 14*14*512
            conv8 = Base_Model.get_layer('conv_pw_9_relu').output  # 14*14*512
            conv9 = Base_Model.get_layer('conv_dw_10_relu').output  # 14*14*512
            conv10 = Base_Model.get_layer('conv_pw_10_relu').output  # 14*14*512
            conv11 = Base_Model.get_layer('conv_dw_11_relu').output  # 14*14*512
            conv12 = Base_Model.get_layer('conv_pw_11_relu').output  # 14*14*512
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv_dw_12_relu').output  # 7*7*512
                #
                conv2 = Base_Model.get_layer('conv_pw_12_relu').output  # 7*7*1024
                conv3 = Base_Model.get_layer('conv_dw_13_relu').output  # 7*7*1024
                conv4 = Base_Model.get_layer('conv_pw_13_relu').output  # 7*7*1024
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def MobileNetV2(self):
        # UNet Variants with MobileNetV2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "MobileNetV2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('Conv1_relu').output  # 112*112*32
            conv2 = Base_Model.get_layer('expanded_conv_depthwise_relu').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('block_1_expand_relu').output  # 112*112*96
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block_1_depthwise_relu').output  # 56*56*96
            #
            conv2 = Base_Model.get_layer('block_2_expand_relu').output  # 56*56*144
            conv3 = Base_Model.get_layer('block_2_depthwise_relu').output  # 56*56*144
            conv4 = Base_Model.get_layer('block_3_expand_relu').output  # 56*56*144
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block_3_depthwise_relu').output  # 28*28*144
            #
            conv2 = Base_Model.get_layer('block_4_expand_relu').output  # 28*28*192
            conv3 = Base_Model.get_layer('block_4_depthwise_relu').output  # 28*28*192
            conv4 = Base_Model.get_layer('block_5_expand_relu').output  # 28*28*192
            conv5 = Base_Model.get_layer('block_5_depthwise_relu').output  # 28*28*192
            conv6 = Base_Model.get_layer('block_6_expand_relu').output  # 28*28*192
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('block_6_depthwise_relu').output  # 14*14*192
            #
            conv2 = Base_Model.get_layer('block_7_expand_relu').output  # 14*14*384
            conv3 = Base_Model.get_layer('block_7_depthwise_relu').output  # 14*14*384
            conv4 = Base_Model.get_layer('block_8_expand_relu').output  # 14*14*384
            conv5 = Base_Model.get_layer('block_8_depthwise_relu').output  # 14*14*384
            conv6 = Base_Model.get_layer('block_9_expand_relu').output  # 14*14*384
            conv7 = Base_Model.get_layer('block_9_depthwise_relu').output  # 14*14*384
            conv8 = Base_Model.get_layer('block_10_expand_relu').output  # 14*14*384
            conv9 = Base_Model.get_layer('block_10_depthwise_relu').output  # 14*14*384
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9])
            #
            conv10 = Base_Model.get_layer('block_11_expand_relu').output  # 14*14*576
            conv11 = Base_Model.get_layer('block_11_depthwise_relu').output  # 14*14*576
            conv12 = Base_Model.get_layer('block_12_expand_relu').output  # 14*14*576
            conv13 = Base_Model.get_layer('block_12_depthwise_relu').output  # 14*14*576
            conv14 = Base_Model.get_layer('block_13_expand_relu').output  # 14*14*576
            conv1_2 = tf.keras.layers.add([conv10, conv11, conv12, conv13, conv14])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block_13_depthwise_relu').output  # 7*7*576
                #
                conv2 = Base_Model.get_layer('block_14_expand_relu').output  # 14*14*960
                conv3 = Base_Model.get_layer('block_14_depthwise_relu').output  # 14*14*960
                conv4 = Base_Model.get_layer('block_15_expand_relu').output  # 14*14*960
                conv5 = Base_Model.get_layer('block_15_depthwise_relu').output  # 14*14*960
                conv6 = Base_Model.get_layer('block_16_expand_relu').output  # 14*14*960
                conv7 = Base_Model.get_layer('block_16_depthwise_relu').output  # 14*14*960
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def MobileNetV3Small(self):
        # UNet Variants with MobileNetV3_Small ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "MobileNetV3Small" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.MobileNetV3Small(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('re_lu').output  # 112*112*16
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('re_lu_1').output  # 56*56*16
            conv2 = Base_Model.get_layer('re_lu_3').output  # 56*56*72
            #
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('re_lu_4').output  # 28*28*72
            #
            conv2 = Base_Model.get_layer('re_lu_5').output  # 28*28*88
            conv3 = Base_Model.get_layer('re_lu_6').output  # 28*28*88
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv4 = Base_Model.get_layer('re_lu_7').output  # 28*28*96
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv4], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('re_lu_8').output  # 14*14*96
            #
            conv2 = Base_Model.get_layer('re_lu_10').output  # 14*14*240
            conv3 = Base_Model.get_layer('re_lu_11').output  # 14*14*240
            conv4 = Base_Model.get_layer('re_lu_13').output  # 14*14*240
            conv5 = Base_Model.get_layer('re_lu_14').output  # 14*14*240
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5])
            #
            conv6 = Base_Model.get_layer('re_lu_16').output  # 14*14*120
            conv7 = Base_Model.get_layer('re_lu_17').output  # 14*14*120
            conv1_2 = tf.keras.layers.add([conv6, conv7])
            #
            conv8 = Base_Model.get_layer('re_lu_19').output  # 14*14*144
            conv9 = Base_Model.get_layer('re_lu_20').output  # 14*14*144
            conv1_3 = tf.keras.layers.add([conv8, conv9])
            #
            conv10 = Base_Model.get_layer('re_lu_22').output  # 14*14*288
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3, conv10], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('re_lu_23').output  # 7*7*288
                #
                conv2 = Base_Model.get_layer('re_lu_25').output  # 7*7*576
                conv3 = Base_Model.get_layer('re_lu_26').output  # 7*7*576
                conv4 = Base_Model.get_layer('re_lu_28').output  # 7*7*576
                conv5 = Base_Model.get_layer('re_lu_29').output  # 7*7*576
                conv6 = Base_Model.get_layer('re_lu_31').output  # 7*7*576
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def MobileNetV3Large(self):
        # UNet Variants with MobileNetV3_Large ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "MobileNetV3Large" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('re_lu').output  # 112*112*16
            conv2 = Base_Model.get_layer('re_lu_1').output  # 112*112*16
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            conv3 = Base_Model.get_layer('re_lu_2').output  # 112*112*64
            conv = tf.keras.layers.concatenate([conv1_1, conv3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('re_lu_3').output  # 56*56*64
            #
            conv2 = Base_Model.get_layer('re_lu_4').output  # 56*56*72
            conv3 = Base_Model.get_layer('re_lu_5').output  # 56*56*72
            conv4 = Base_Model.get_layer('re_lu_6').output  # 56*56*72
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('re_lu_7').output  # 28*28*72
            #
            conv2 = Base_Model.get_layer('re_lu_9').output  # 28*28*120
            conv3 = Base_Model.get_layer('re_lu_10').output  # 28*28*120
            conv4 = Base_Model.get_layer('re_lu_12').output  # 28*28*120
            conv5 = Base_Model.get_layer('re_lu_13').output  # 28*28*120
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5])
            #
            conv6 = Base_Model.get_layer('re_lu_15').output  # 28*28*240
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv6], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('re_lu_16').output  # 14*14*240
            #
            conv2 = Base_Model.get_layer('re_lu_17').output  # 14*14*200
            conv3 = Base_Model.get_layer('re_lu_18').output  # 14*14*200
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv4 = Base_Model.get_layer('re_lu_19').output  # 14*14*184
            conv5 = Base_Model.get_layer('re_lu_20').output  # 14*14*184
            conv6 = Base_Model.get_layer('re_lu_21').output  # 14*14*184
            conv7 = Base_Model.get_layer('re_lu_22').output  # 14*14*184
            conv1_2 = tf.keras.layers.add([conv4, conv5, conv6, conv7])
            #
            conv8 = Base_Model.get_layer('re_lu_23').output  # 14*14*480
            conv9 = Base_Model.get_layer('re_lu_24').output  # 14*14*480
            conv1_3 = tf.keras.layers.add([conv8, conv9])
            #
            conv10 = Base_Model.get_layer('re_lu_26').output  # 14*14*672
            conv11 = Base_Model.get_layer('re_lu_27').output  # 14*14*672
            conv12 = Base_Model.get_layer('re_lu_29').output  # 14*14*672
            conv1_4 = tf.keras.layers.add([conv10, conv11, conv12])

            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3, conv1_4], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('re_lu_30').output  # 7*7*672
                #
                conv2 = Base_Model.get_layer('re_lu_32').output  # 7*7*960
                conv3 = Base_Model.get_layer('re_lu_33').output  # 7*7*960
                conv4 = Base_Model.get_layer('re_lu_35').output  # 7*7*960
                conv5 = Base_Model.get_layer('re_lu_36').output  # 7*7*960
                conv6 = Base_Model.get_layer('re_lu_38').output  # 7*7*960
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def InceptionV3(self):
        # UNet Variants with Inception V3 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "InceptionV3" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('activation').output  # 111*111*64
            conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            conv1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 112*112*64
            conv2 = Base_Model.get_layer('activation_1').output  # 109*109*32
            conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)
            conv2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('activation_2').output  # 109*109*64
            conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
            conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)
            conv3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv3)  # 112*112*64
            #
            conv = tf.keras.layers.concatenate([conv3, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('activation_3').output  # 54*54*80
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 56*56*80
            conv2 = Base_Model.get_layer('activation_4').output  # 52*52*192
            conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)
            conv2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)  # 56*56*192
            #
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('activation_11').output  # 25*25*32
            conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)
            conv1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)  # 28*28*32
            #
            conv2 = Base_Model.get_layer('activation_5').output  # 25*25*64
            conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)
            conv2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)  # 28*28*64
            conv3 = Base_Model.get_layer('activation_7').output  # 25*25*64
            conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
            conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)
            conv3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv3)  # 28*28*64
            conv4 = Base_Model.get_layer('activation_8').output  # 25*25*64
            conv4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
            conv4 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv4)
            conv4 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)  # 28*28*64
            conv5 = Base_Model.get_layer('activation_12').output  # 25*25*64
            conv5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)
            conv5 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv5)
            conv5 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)  # 28*28*64
            conv6 = Base_Model.get_layer('activation_14').output  # 25*25*64
            conv6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)
            conv6 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv6)
            conv6 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)  # 28*28*64
            conv7 = Base_Model.get_layer('activation_15').output  # 25*25*64
            conv7 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv7)
            conv7 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv7)
            conv7 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7)  # 28*28*64
            conv8 = Base_Model.get_layer('activation_18').output  # 25*25*64
            conv8 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv8)
            conv8 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv8)
            conv8 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8)  # 28*28*64
            conv9 = Base_Model.get_layer('activation_19').output  # 25*25*64
            conv9 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv9)
            conv9 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv9)
            conv9 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv9)  # 28*28*64
            conv10 = Base_Model.get_layer('activation_21').output  # 25*25*64
            conv10 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv10)
            conv10 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv10)
            conv10 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv10)  # 28*28*64
            conv11 = Base_Model.get_layer('activation_22').output  # 25*25*64
            conv11 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv11)
            conv11 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv11)
            conv11 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv11)  # 28*28*64
            conv12 = Base_Model.get_layer('activation_25').output  # 25*25*64
            conv12 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv12)
            conv12 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv12)
            conv12 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv12)  # 28*28*64
            conv13 = Base_Model.get_layer('activation_27').output  # 25*25*64
            conv13 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv13)
            conv13 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv13)
            conv13 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv13)  # 28*28*64
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('activation_6').output  # 25*25*48
            conv14 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv14)
            conv14 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv14)
            conv14 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv14)  # 28*28*48
            conv15 = Base_Model.get_layer('activation_13').output  # 25*25*48
            conv15 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv15)
            conv15 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv15)
            conv15 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv15)  # 28*28*48
            conv16 = Base_Model.get_layer('activation_20').output  # 25*25*48
            conv16 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv16)
            conv16 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv16)
            conv16 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv16)  # 28*28*48
            conv1_2 = tf.keras.layers.add([conv14, conv15, conv16])
            #
            conv17 = Base_Model.get_layer('activation_9').output  # 25*25*96
            conv17 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv17)
            conv17 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv17)
            conv17 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv17)  # 28*28*96
            conv18 = Base_Model.get_layer('activation_10').output  # 25*25*96
            conv18 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv18)
            conv18 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv18)
            conv18 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv18)  # 28*28*96
            conv19 = Base_Model.get_layer('activation_16').output  # 25*25*96
            conv19 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv19)
            conv19 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv19)
            conv19 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv19)  # 28*28*96
            conv20 = Base_Model.get_layer('activation_17').output  # 25*25*96
            conv20 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv20)
            conv20 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv20)
            conv20 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv20)  # 28*28*96
            conv21 = Base_Model.get_layer('activation_23').output  # 25*25*96
            conv21 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv21)
            conv21 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv21)
            conv21 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv21)  # 28*28*96
            conv22 = Base_Model.get_layer('activation_24').output  # 25*25*96
            conv22 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv22)
            conv22 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv22)
            conv22 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv22)  # 28*28*96
            conv23 = Base_Model.get_layer('activation_28').output  # 25*25*96
            conv23 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv23)
            conv23 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv23)
            conv23 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv23)  # 28*28*96
            conv1_3 = tf.keras.layers.add([conv17, conv18, conv19, conv20, conv21, conv22, conv23])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('activation_26').output  # 12*12*384
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 14*14*384
            #
            conv2 = Base_Model.get_layer('activation_29').output  # 12*12*96
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)  # 14*14*96
            #
            conv3 = Base_Model.get_layer('activation_31').output  # 12*12*128
            conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)  # 14*14*128
            conv4 = Base_Model.get_layer('activation_32').output  # 12*12*128
            conv4 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv4)  # 14*14*128
            conv5 = Base_Model.get_layer('activation_34').output  # 12*12*128
            conv5 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv5)  # 14*14*128
            conv6 = Base_Model.get_layer('activation_35').output  # 12*12*128
            conv6 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv6)  # 14*14*128
            conv7 = Base_Model.get_layer('activation_36').output  # 12*12*128
            conv7 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv7)  # 14*14*128
            conv8 = Base_Model.get_layer('activation_37').output  # 12*12*128
            conv8 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv8)  # 14*14*128
            conv1_1 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8])
            #
            conv9 = Base_Model.get_layer('activation_30').output  # 12*12*192
            conv9 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv9)  # 14*14*192
            conv10 = Base_Model.get_layer('activation_33').output  # 12*12*192
            conv10 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv10)  # 14*14*192
            conv11 = Base_Model.get_layer('activation_38').output  # 12*12*192
            conv11 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv11)  # 14*14*192
            conv12 = Base_Model.get_layer('activation_39').output  # 12*12*192
            conv12 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv12)  # 14*14*192
            conv13 = Base_Model.get_layer('activation_40').output  # 12*12*192
            conv13 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv13)  # 14*14*192
            conv14 = Base_Model.get_layer('activation_43').output  # 12*12*192
            conv14 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv14)  # 14*14*192
            conv15 = Base_Model.get_layer('activation_48').output  # 12*12*192
            conv15 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv15)  # 14*14*192
            conv16 = Base_Model.get_layer('activation_49').output  # 12*12*192
            conv16 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv16)  # 14*14*192
            conv17 = Base_Model.get_layer('activation_50').output  # 12*12*192
            conv17 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv17)  # 14*14*192
            conv18 = Base_Model.get_layer('activation_53').output  # 12*12*192
            conv18 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv18)  # 14*14*192
            conv19 = Base_Model.get_layer('activation_58').output  # 12*12*192
            conv19 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv19)  # 14*14*192
            conv20 = Base_Model.get_layer('activation_59').output  # 12*12*192
            conv20 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv20)  # 14*14*192
            conv21 = Base_Model.get_layer('activation_60').output  # 12*12*192
            conv21 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv21)  # 14*14*192
            conv22 = Base_Model.get_layer('activation_63').output  # 12*12*192
            conv22 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv22)  # 14*14*192
            conv23 = Base_Model.get_layer('activation_68').output  # 12*12*192
            conv23 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv23)  # 14*14*192
            conv24 = Base_Model.get_layer('activation_69').output  # 12*12*192
            conv24 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv24)  # 14*14*192
            conv25 = Base_Model.get_layer('activation_70').output  # 12*12*192
            conv25 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv25)  # 14*14*192
            conv26 = Base_Model.get_layer('activation_72').output  # 12*12*192
            conv26 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv26)  # 14*14*192
            conv27 = Base_Model.get_layer('activation_73').output  # 12*12*192
            conv27 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv27)  # 14*14*192
            conv28 = Base_Model.get_layer('activation_74').output  # 12*12*192
            conv28 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv28)  # 14*14*192
            conv29 = Base_Model.get_layer('activation_61').output  # 12*12*192
            conv29 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv29)  # 14*14*192
            conv30 = Base_Model.get_layer('activation_62').output  # 12*12*192
            conv30 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv30)  # 14*14*192
            conv31 = Base_Model.get_layer('activation_64').output  # 12*12*192
            conv31 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv31)  # 14*14*192
            conv32 = Base_Model.get_layer('activation_65').output  # 12*12*192
            conv32 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv32)  # 14*14*192
            conv33 = Base_Model.get_layer('activation_66').output  # 12*12*192
            conv33 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv33)  # 14*14*192
            conv34 = Base_Model.get_layer('activation_67').output  # 12*12*192
            conv34 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv34)  # 14*14*192
            conv1_2 = tf.keras.layers.add([conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18,
                                           conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28,
                                           conv29, conv30, conv31, conv32, conv33, conv34])
            #
            conv35 = Base_Model.get_layer('activation_41').output  # 12*12*160
            conv35 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv35)  # 14*14*160
            conv36 = Base_Model.get_layer('activation_42').output  # 12*12*160
            conv36 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv36)  # 14*14*160
            conv37 = Base_Model.get_layer('activation_44').output  # 12*12*160
            conv37 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv37)  # 14*14*160
            conv38 = Base_Model.get_layer('activation_45').output  # 12*12*160
            conv38 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv38)  # 14*14*160
            conv39 = Base_Model.get_layer('activation_46').output  # 12*12*160
            conv39 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv39)  # 14*14*160
            conv40 = Base_Model.get_layer('activation_47').output  # 12*12*160
            conv40 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv40)  # 14*14*160
            conv41 = Base_Model.get_layer('activation_51').output  # 12*12*160
            conv41 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv41)  # 14*14*160
            conv42 = Base_Model.get_layer('activation_52').output  # 12*12*160
            conv42 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv42)  # 14*14*160
            conv43 = Base_Model.get_layer('activation_54').output  # 12*12*160
            conv43 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv43)  # 14*14*160
            conv44 = Base_Model.get_layer('activation_55').output  # 12*12*160
            conv44 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv44)  # 14*14*160
            conv45 = Base_Model.get_layer('activation_56').output  # 12*12*160
            conv45 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv45)  # 14*14*160
            conv46 = Base_Model.get_layer('activation_57').output  # 12*12*160
            conv46 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv46)  # 14*14*160
            conv1_3 = tf.keras.layers.add([conv35, conv36, conv37, conv38, conv39, conv40, conv41, conv42, conv43, conv44,
                                           conv45, conv46])
            #
            conv = tf.keras.layers.concatenate([conv1, conv2, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('activation_71').output  # 5*5*320
                conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 7*7*320
                conv2 = Base_Model.get_layer('activation_76').output  # 5*5*320
                conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)  # 7*7*320
                conv3 = Base_Model.get_layer('activation_85').output  # 5*5*320
                conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)  # 7*7*320
                conv1_1 = tf.keras.layers.add([conv1, conv2, conv3])
                #
                conv4 = Base_Model.get_layer('activation_75').output  # 5*5*192
                conv4 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv4)  # 7*7*192
                conv5 = Base_Model.get_layer('activation_84').output  # 5*5*192
                conv5 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv5)  # 7*7*192
                conv6 = Base_Model.get_layer('activation_93').output  # 5*5*192
                conv6 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv6)  # 7*7*192
                conv1_2 = tf.keras.layers.add([conv4, conv5, conv6])
                #
                conv7 = Base_Model.get_layer('activation_77').output  # 5*5*384
                conv7 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv7)  # 7*7*384
                conv8 = Base_Model.get_layer('activation_78').output  # 5*5*384
                conv8 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv8)  # 7*7*384
                conv9 = Base_Model.get_layer('activation_79').output  # 5*5*384
                conv9 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv9)  # 7*7*384
                conv10 = Base_Model.get_layer('activation_81').output  # 5*5*384
                conv10 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv10)  # 7*7*384
                conv11 = Base_Model.get_layer('activation_82').output  # 5*5*384
                conv11 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv11)  # 7*7*384
                conv12 = Base_Model.get_layer('activation_83').output  # 5*5*384
                conv12 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv12)  # 7*7*384
                conv13 = Base_Model.get_layer('activation_86').output  # 5*5*384
                conv13 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv13)  # 7*7*384
                conv14 = Base_Model.get_layer('activation_87').output  # 5*5*384
                conv14 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv14)  # 7*7*384
                conv15 = Base_Model.get_layer('activation_88').output  # 5*5*384
                conv15 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv15)  # 7*7*384
                conv16 = Base_Model.get_layer('activation_90').output  # 5*5*384
                conv16 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv16)  # 7*7*384
                conv17 = Base_Model.get_layer('activation_91').output  # 5*5*384
                conv17 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv17)  # 7*7*384
                conv18 = Base_Model.get_layer('activation_92').output  # 5*5*384
                conv18 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv18)  # 7*7*384
                conv1_3 = tf.keras.layers.add([conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18])
                #
                conv19 = Base_Model.get_layer('activation_80').output  # 5*5*448
                conv19 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv19)  # 7*7*448
                conv20 = Base_Model.get_layer('activation_89').output  # 5*5*448
                conv20 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv20)  # 7*7*448
                conv1_4 = tf.keras.layers.add([conv19, conv20])
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1_1, conv1_2, conv1_3, conv1_4], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def InceptionResNetV2(self):
        # UNet Variants with InceptionResNetV2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "InceptionResNetV2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('activation').output  # 111*111*64
            conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            conv1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 112*112*64
            conv2 = Base_Model.get_layer('activation_1').output  # 109*109*32
            conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)
            conv2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('activation_2').output  # 109*109*64
            conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
            conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)
            conv3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv3)  # 112*112*64
            #
            conv = tf.keras.layers.concatenate([conv3, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('activation_3').output  # 54*54*80
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 56*56*80
            conv2 = Base_Model.get_layer('activation_4').output  # 52*52*192
            conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)
            conv2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)  # 56*56*192
            #
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('activation_12').output  # 25*25*32
            conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)
            conv1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)  # 28*28*32
            conv2 = Base_Model.get_layer('activation_13').output  # 25*25*32
            conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)
            conv2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv2)  # 28*28*32
            conv3 = Base_Model.get_layer('activation_14').output  # 25*25*32
            conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
            conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)
            conv3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv3)  # 28*28*32
            conv4 = Base_Model.get_layer('activation_15').output  # 25*25*32
            conv4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
            conv4 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv4)
            conv4 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)  # 28*28*32
            conv5 = Base_Model.get_layer('activation_18').output  # 25*25*32
            conv5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)
            conv5 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv5)
            conv5 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)  # 28*28*32
            conv6 = Base_Model.get_layer('activation_19').output  # 25*25*32
            conv6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)
            conv6 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv6)
            conv6 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)  # 28*28*32
            conv7 = Base_Model.get_layer('activation_20').output  # 25*25*32
            conv7 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv7)
            conv7 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv7)
            conv7 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7)  # 28*28*32
            conv8 = Base_Model.get_layer('activation_21').output  # 25*25*32
            conv8 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv8)
            conv8 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv8)
            conv8 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8)  # 28*28*32
            conv9 = Base_Model.get_layer('activation_24').output  # 25*25*32
            conv9 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv9)
            conv9 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv9)
            conv9 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv9)  # 28*28*32
            conv10 = Base_Model.get_layer('activation_25').output  # 25*25*32
            conv10 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv10)
            conv10 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv10)
            conv10 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv10)  # 28*28*32
            conv11 = Base_Model.get_layer('activation_26').output  # 25*25*32
            conv11 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv11)
            conv11 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv11)
            conv11 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv11)  # 28*28*32
            conv12 = Base_Model.get_layer('activation_27').output  # 25*25*32
            conv12 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv12)
            conv12 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv12)
            conv12 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv12)  # 28*28*32
            conv13 = Base_Model.get_layer('activation_30').output  # 25*25*32
            conv13 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv13)
            conv13 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv13)
            conv13 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv13)  # 28*28*32
            conv14 = Base_Model.get_layer('activation_31').output  # 25*25*32
            conv14 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv14)
            conv14 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv14)
            conv14 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv14)  # 28*28*32
            conv15 = Base_Model.get_layer('activation_32').output  # 25*25*32
            conv15 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv15)
            conv15 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv15)
            conv15 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv15)  # 28*28*32
            conv16 = Base_Model.get_layer('activation_33').output  # 25*25*32
            conv16 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv16)
            conv16 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv16)
            conv16 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv16)  # 28*28*32
            conv17 = Base_Model.get_layer('activation_36').output  # 25*25*32
            conv17 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv17)
            conv17 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv17)
            conv17 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv17)  # 28*28*32
            conv18 = Base_Model.get_layer('activation_37').output  # 25*25*32
            conv18 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv18)
            conv18 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv18)
            conv18 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv18)  # 28*28*32
            conv19 = Base_Model.get_layer('activation_38').output  # 25*25*32
            conv19 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv19)
            conv19 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv19)
            conv19 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv19)  # 28*28*32
            conv20 = Base_Model.get_layer('activation_39').output  # 25*25*32
            conv20 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv20)
            conv20 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv20)
            conv20 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv20)  # 28*28*32
            conv21 = Base_Model.get_layer('activation_42').output  # 25*25*32
            conv21 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv21)
            conv21 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv21)
            conv21 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv21)  # 28*28*32
            conv22 = Base_Model.get_layer('activation_43').output  # 25*25*32
            conv22 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv22)
            conv22 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv22)
            conv22 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv22)  # 28*28*32
            conv23 = Base_Model.get_layer('activation_44').output  # 25*25*32
            conv23 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv23)
            conv23 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv23)
            conv23 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv23)  # 28*28*32
            conv24 = Base_Model.get_layer('activation_45').output  # 25*25*32
            conv24 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv24)
            conv24 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv24)
            conv24 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv24)  # 28*28*32
            conv25 = Base_Model.get_layer('activation_48').output  # 25*25*32
            conv25 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv25)
            conv25 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv25)
            conv25 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv25)  # 28*28*32
            conv26 = Base_Model.get_layer('activation_49').output  # 25*25*32
            conv26 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv26)
            conv26 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv26)
            conv26 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv26)  # 28*28*32
            conv27 = Base_Model.get_layer('activation_50').output  # 25*25*32
            conv27 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv27)
            conv27 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv27)
            conv27 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv27)  # 28*28*32
            conv28 = Base_Model.get_layer('activation_51').output  # 25*25*32
            conv28 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv28)
            conv28 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv28)
            conv28 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv28)  # 28*28*32
            conv29 = Base_Model.get_layer('activation_54').output  # 25*25*32
            conv29 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv29)
            conv29 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv29)
            conv29 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv29)  # 28*28*32
            conv30 = Base_Model.get_layer('activation_55').output  # 25*25*32
            conv30 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv30)
            conv30 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv30)
            conv30 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv30)  # 28*28*32
            conv31 = Base_Model.get_layer('activation_56').output  # 25*25*32
            conv31 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv31)
            conv31 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv31)
            conv31 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv31)  # 28*28*32
            conv32 = Base_Model.get_layer('activation_57').output  # 25*25*32
            conv32 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv32)
            conv32 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv32)
            conv32 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv32)  # 28*28*32
            conv33 = Base_Model.get_layer('activation_60').output  # 25*25*32
            conv33 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv33)
            conv33 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv33)
            conv33 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv33)  # 28*28*32
            conv34 = Base_Model.get_layer('activation_61').output  # 25*25*32
            conv34 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv34)
            conv34 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv34)
            conv34 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv34)  # 28*28*32
            conv35 = Base_Model.get_layer('activation_62').output  # 25*25*32
            conv35 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv35)
            conv35 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv35)
            conv35 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv35)  # 28*28*32
            conv36 = Base_Model.get_layer('activation_63').output  # 25*25*32
            conv36 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv36)
            conv36 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv36)
            conv36 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv36)  # 28*28*32
            conv37 = Base_Model.get_layer('activation_66').output  # 25*25*32
            conv37 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv37)
            conv37 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv37)
            conv37 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv37)  # 28*28*32
            conv38 = Base_Model.get_layer('activation_67').output  # 25*25*32
            conv38 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv38)
            conv38 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv38)
            conv38 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv38)  # 28*28*32
            conv39 = Base_Model.get_layer('activation_68').output  # 25*25*32
            conv39 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv39)
            conv39 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv39)
            conv39 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv39)  # 28*28*32
            conv40 = Base_Model.get_layer('activation_69').output  # 25*25*32
            conv40 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv40)
            conv40 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv40)
            conv40 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv40)  # 28*28*32
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10,
                                           conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20,
                                           conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30,
                                           conv31, conv32, conv32, conv33, conv34, conv35, conv36, conv37, conv38, conv39,
                                           conv40])
            #
            conv41 = Base_Model.get_layer('activation_6').output  # 25*25*48
            conv41 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv41)
            conv41 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv41)
            conv41 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv41)  # 28*28*48
            conv42 = Base_Model.get_layer('activation_16').output  # 25*25*48
            conv42 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv42)
            conv42 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv42)
            conv42 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv42)  # 28*28*48
            conv43 = Base_Model.get_layer('activation_22').output  # 25*25*48
            conv43 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv43)
            conv43 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv43)
            conv43 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv43)  # 28*28*48
            conv44 = Base_Model.get_layer('activation_28').output  # 25*25*48
            conv44 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv44)
            conv44 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv44)
            conv44 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv44)  # 28*28*48
            conv45 = Base_Model.get_layer('activation_34').output  # 25*25*48
            conv45 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv45)
            conv45 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv45)
            conv45 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv45)  # 28*28*48
            conv46 = Base_Model.get_layer('activation_40').output  # 25*25*48
            conv46 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv46)
            conv46 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv46)
            conv46 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv46)  # 28*28*48
            conv47 = Base_Model.get_layer('activation_46').output  # 25*25*48
            conv47 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv47)
            conv47 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv47)
            conv47 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv47)  # 28*28*48
            conv48 = Base_Model.get_layer('activation_52').output  # 25*25*48
            conv48 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv48)
            conv48 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv48)
            conv48 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv48)  # 28*28*48
            conv49 = Base_Model.get_layer('activation_58').output  # 25*25*48
            conv49 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv49)
            conv49 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv49)
            conv49 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv49)  # 28*28*48
            conv50 = Base_Model.get_layer('activation_64').output  # 25*25*48
            conv50 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv50)
            conv50 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv50)
            conv50 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv50)  # 28*28*48
            conv51 = Base_Model.get_layer('activation_70').output  # 25*25*48
            conv51 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv51)
            conv51 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv51)
            conv51 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv51)  # 28*28*48
            conv1_2 = tf.keras.layers.add([conv41, conv42, conv43, conv44, conv45, conv46, conv47, conv48, conv49, conv50, conv51])
            #
            conv52 = Base_Model.get_layer('activation_7').output  # 25*25*64
            conv52 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv52)
            conv52 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv52)
            conv52 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv52)  # 28*28*64
            conv53 = Base_Model.get_layer('activation_8').output  # 25*25*64
            conv53 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv53)
            conv53 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv53)
            conv53 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv53)  # 28*28*64
            conv54 = Base_Model.get_layer('activation_17').output  # 25*25*64
            conv54 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv54)
            conv54 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv54)
            conv54 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv54)  # 28*28*64
            conv55 = Base_Model.get_layer('activation_11').output  # 25*25*64
            conv55 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv55)
            conv55 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv55)
            conv55 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv55)  # 28*28*64
            conv56 = Base_Model.get_layer('activation_23').output  # 25*25*64
            conv56 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv56)
            conv56 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv56)
            conv56 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv56)  # 28*28*64
            conv57 = Base_Model.get_layer('activation_29').output  # 25*25*64
            conv57 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv57)
            conv57 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv57)
            conv57 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv57)  # 28*28*64
            conv58 = Base_Model.get_layer('activation_35').output  # 25*25*64
            conv58 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv58)
            conv58 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv58)
            conv58 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv58)  # 28*28*64
            conv59 = Base_Model.get_layer('activation_41').output  # 25*25*64
            conv59 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv59)
            conv59 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv59)
            conv59 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv59)  # 28*28*64
            conv60 = Base_Model.get_layer('activation_47').output  # 25*25*64
            conv60 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv60)
            conv60 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv60)
            conv60 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv60)  # 28*28*64
            conv61 = Base_Model.get_layer('activation_53').output  # 25*25*64
            conv61 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv61)
            conv61 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv61)
            conv61 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv61)  # 28*28*64
            conv62 = Base_Model.get_layer('activation_59').output  # 25*25*64
            conv62 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv62)
            conv62 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv62)
            conv62 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv62)  # 28*28*64
            conv63 = Base_Model.get_layer('activation_65').output  # 25*25*64
            conv63 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv63)
            conv63 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv63)
            conv63 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv63)  # 28*28*64
            conv64 = Base_Model.get_layer('activation_71').output  # 25*25*64
            conv64 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv64)
            conv64 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv64)
            conv64 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv64)  # 28*28*64
            conv1_3 = tf.keras.layers.add([conv52, conv53, conv54, conv55, conv56, conv57, conv58, conv59, conv60, conv61, conv62, conv63, conv64])
            #
            conv65 = Base_Model.get_layer('activation_5').output  # 25*25*96
            conv65 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv65)
            conv65 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv65)
            conv65 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv65)  # 28*28*96
            conv66 = Base_Model.get_layer('activation_9').output  # 25*25*96
            conv66 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv66)
            conv66 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv66)
            conv66 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv66)  # 28*28*96
            conv67 = Base_Model.get_layer('activation_10').output  # 25*25*96
            conv67 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv67)
            conv67 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv67)
            conv67 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv67)  # 28*28*96
            conv1_4 = tf.keras.layers.add([conv65, conv66, conv67])
            #
            conv68 = Base_Model.get_layer('activation_73').output  # 25*25*256
            conv68 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv68)
            conv68 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv68)
            conv68 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv68)  # 28*28*256
            conv69 = Base_Model.get_layer('activation_74').output  # 25*25*256
            conv69 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv69)
            conv69 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv69)
            conv69 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv69)  # 28*28*256
            conv1_5 = tf.keras.layers.add([conv68, conv69])
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2, conv1_3, conv1_4, conv1_5], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('activation_72').output  # 12*12*384
            conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 14*14*384
            conv2 = Base_Model.get_layer('activation_75').output  # 12*12*384
            conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)  # 14*14*384
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('activation_77').output  # 12*12*128
            conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)  # 14*14*128
            conv4 = Base_Model.get_layer('activation_81').output  # 12*12*128
            conv4 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv4)  # 14*14*128
            conv5 = Base_Model.get_layer('activation_85').output  # 12*12*128
            conv5 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv5)  # 14*14*128
            conv6 = Base_Model.get_layer('activation_89').output  # 12*12*128
            conv6 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv6)  # 14*14*128
            conv7 = Base_Model.get_layer('activation_93').output  # 12*12*128
            conv7 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv7)  # 14*14*128
            conv8 = Base_Model.get_layer('activation_97').output  # 12*12*128
            conv8 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv8)  # 14*14*128
            conv9 = Base_Model.get_layer('activation_101').output  # 12*12*128
            conv9 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv9)  # 14*14*128
            conv10 = Base_Model.get_layer('activation_105').output  # 12*12*128
            conv10 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv10)  # 14*14*128
            conv11 = Base_Model.get_layer('activation_109').output  # 12*12*128
            conv11 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv11)  # 14*14*128
            conv12 = Base_Model.get_layer('activation_113').output  # 12*12*128
            conv12 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv12)  # 14*14*128
            conv13 = Base_Model.get_layer('activation_117').output  # 12*12*128
            conv13 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv13)  # 14*14*128
            conv14 = Base_Model.get_layer('activation_121').output  # 12*12*128
            conv14 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv14)  # 14*14*128
            conv15 = Base_Model.get_layer('activation_125').output  # 12*12*128
            conv15 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv15)  # 14*14*128
            conv16 = Base_Model.get_layer('activation_129').output  # 12*12*128
            conv16 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv16)  # 14*14*128
            conv17 = Base_Model.get_layer('activation_133').output  # 12*12*128
            conv17 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv17)  # 14*14*128
            conv18 = Base_Model.get_layer('activation_137').output  # 12*12*128
            conv18 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv18)  # 14*14*128
            conv19 = Base_Model.get_layer('activation_141').output  # 12*12*128
            conv19 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv19)  # 14*14*128
            conv20 = Base_Model.get_layer('activation_145').output  # 12*12*128
            conv20 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv20)  # 14*14*128
            conv21 = Base_Model.get_layer('activation_149').output  # 12*12*128
            conv21 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv21)  # 14*14*128
            conv22 = Base_Model.get_layer('activation_153').output  # 12*12*128
            conv22 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv22)  # 14*14*128
            conv1_2 = tf.keras.layers.add([conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                                           conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22])
            #
            conv23 = Base_Model.get_layer('activation_78').output  # 12*12*160
            conv23 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv23)  # 14*14*160
            conv24 = Base_Model.get_layer('activation_82').output  # 12*12*160
            conv24 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv24)  # 14*14*160
            conv25 = Base_Model.get_layer('activation_86').output  # 12*12*160
            conv25 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv25)  # 14*14*160
            conv26 = Base_Model.get_layer('activation_90').output  # 12*12*160
            conv26 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv26)  # 14*14*160
            conv27 = Base_Model.get_layer('activation_94').output  # 12*12*160
            conv27 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv27)  # 14*14*160
            conv28 = Base_Model.get_layer('activation_98').output  # 12*12*160
            conv28 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv28)  # 14*14*160
            conv29 = Base_Model.get_layer('activation_102').output  # 12*12*160
            conv29 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv29)  # 14*14*160
            conv30 = Base_Model.get_layer('activation_106').output  # 12*12*160
            conv30 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv30)  # 14*14*160
            conv31 = Base_Model.get_layer('activation_110').output  # 12*12*160
            conv31 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv31)  # 14*14*160
            conv32 = Base_Model.get_layer('activation_114').output  # 12*12*160
            conv32 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv32)  # 14*14*160
            conv33 = Base_Model.get_layer('activation_118').output  # 12*12*160
            conv33 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv33)  # 14*14*160
            conv34 = Base_Model.get_layer('activation_122').output  # 12*12*160
            conv34 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv34)  # 14*14*160
            conv35 = Base_Model.get_layer('activation_126').output  # 12*12*160
            conv35 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv35)  # 14*14*160
            conv36 = Base_Model.get_layer('activation_130').output  # 12*12*160
            conv36 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv36)  # 14*14*160
            conv37 = Base_Model.get_layer('activation_134').output  # 12*12*160
            conv37 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv37)  # 14*14*160
            conv38 = Base_Model.get_layer('activation_138').output  # 12*12*160
            conv38 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv38)  # 14*14*160
            conv39 = Base_Model.get_layer('activation_142').output  # 12*12*160
            conv39 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv39)  # 14*14*160
            conv40 = Base_Model.get_layer('activation_146').output  # 12*12*160
            conv40 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv40)  # 14*14*160
            conv41 = Base_Model.get_layer('activation_150').output  # 12*12*160
            conv41 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv41)  # 14*14*160
            conv42 = Base_Model.get_layer('activation_154').output  # 12*12*160
            conv42 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv42)  # 14*14*160
            conv1_3 = tf.keras.layers.add([conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32,
                                           conv33, conv34, conv35, conv36, conv37, conv38, conv39, conv40, conv41, conv42])
            #
            conv43 = Base_Model.get_layer('activation_76').output  # 12*12*192
            conv43 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv43)  # 14*14*192
            conv44 = Base_Model.get_layer('activation_79').output  # 12*12*192
            conv44 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv44)  # 14*14*192
            conv45 = Base_Model.get_layer('activation_80').output  # 12*12*192
            conv45 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv45)  # 14*14*192
            conv46 = Base_Model.get_layer('activation_83').output  # 12*12*192
            conv46 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv46)  # 14*14*192
            conv47 = Base_Model.get_layer('activation_84').output  # 12*12*192
            conv47 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv47)  # 14*14*192
            conv48 = Base_Model.get_layer('activation_87').output  # 12*12*192
            conv48 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv48)  # 14*14*192
            conv49 = Base_Model.get_layer('activation_88').output  # 12*12*192
            conv49 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv49)  # 14*14*192
            conv50 = Base_Model.get_layer('activation_91').output  # 12*12*192
            conv50 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv50)  # 14*14*192
            conv51 = Base_Model.get_layer('activation_92').output  # 12*12*192
            conv51 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv51)  # 14*14*192
            conv52 = Base_Model.get_layer('activation_95').output  # 12*12*192
            conv52 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv52)  # 14*14*192
            conv53 = Base_Model.get_layer('activation_96').output  # 12*12*192
            conv53 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv53)  # 14*14*192
            conv54 = Base_Model.get_layer('activation_99').output  # 12*12*192
            conv54 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv54)  # 14*14*192
            conv55 = Base_Model.get_layer('activation_100').output  # 12*12*192
            conv55 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv55)  # 14*14*192
            conv56 = Base_Model.get_layer('activation_103').output  # 12*12*192
            conv56 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv56)  # 14*14*192
            conv57 = Base_Model.get_layer('activation_104').output  # 12*12*192
            conv57 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv57)  # 14*14*192
            conv58 = Base_Model.get_layer('activation_107').output  # 12*12*192
            conv58 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv58)  # 14*14*192
            conv59 = Base_Model.get_layer('activation_108').output  # 12*12*192
            conv59 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv59)  # 14*14*192
            conv60 = Base_Model.get_layer('activation_111').output  # 12*12*192
            conv60 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv60)  # 14*14*192
            conv61 = Base_Model.get_layer('activation_112').output  # 12*12*192
            conv61 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv61)  # 14*14*192
            conv62 = Base_Model.get_layer('activation_115').output  # 12*12*192
            conv62 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv62)  # 14*14*192
            conv63 = Base_Model.get_layer('activation_116').output  # 12*12*192
            conv63 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv63)  # 14*14*192
            conv64 = Base_Model.get_layer('activation_119').output  # 12*12*192
            conv64 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv64)  # 14*14*192
            conv65 = Base_Model.get_layer('activation_120').output  # 12*12*192
            conv65 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv65)  # 14*14*192
            conv66 = Base_Model.get_layer('activation_123').output  # 12*12*192
            conv66 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv66)  # 14*14*192
            conv67 = Base_Model.get_layer('activation_124').output  # 12*12*192
            conv67 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv67)  # 14*14*192
            conv68 = Base_Model.get_layer('activation_127').output  # 12*12*192
            conv68 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv68)  # 14*14*192
            conv69 = Base_Model.get_layer('activation_128').output  # 12*12*192
            conv69 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv69)  # 14*14*192
            conv70 = Base_Model.get_layer('activation_131').output  # 12*12*192
            conv70 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv70)  # 14*14*192
            conv71 = Base_Model.get_layer('activation_132').output  # 12*12*192
            conv71 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv71)  # 14*14*192
            conv72 = Base_Model.get_layer('activation_135').output  # 12*12*192
            conv72 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv72)  # 14*14*192
            conv73 = Base_Model.get_layer('activation_136').output  # 12*12*192
            conv73 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv73)  # 14*14*192
            conv74 = Base_Model.get_layer('activation_139').output  # 12*12*192
            conv74 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv74)  # 14*14*192
            conv75 = Base_Model.get_layer('activation_140').output  # 12*12*192
            conv75 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv75)  # 14*14*192
            conv76 = Base_Model.get_layer('activation_143').output  # 12*12*192
            conv76 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv76)  # 14*14*192
            conv77 = Base_Model.get_layer('activation_144').output  # 12*12*192
            conv77 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv77)  # 14*14*192
            conv78 = Base_Model.get_layer('activation_147').output  # 12*12*192
            conv78 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv78)  # 14*14*192
            conv79 = Base_Model.get_layer('activation_148').output  # 12*12*192
            conv79 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv79)  # 14*14*192
            conv80 = Base_Model.get_layer('activation_151').output  # 12*12*192
            conv80 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv80)  # 14*14*192
            conv81 = Base_Model.get_layer('activation_152').output  # 12*12*192
            conv81 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv81)  # 14*14*192
            conv82 = Base_Model.get_layer('activation_155').output  # 12*12*192
            conv82 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv82)  # 14*14*192
            conv1_4 = tf.keras.layers.add([conv43, conv44, conv45, conv46, conv47, conv48, conv49, conv50, conv51, conv52,
                                           conv53, conv54, conv55, conv56, conv57, conv58, conv59, conv60, conv61, conv62,
                                           conv63, conv64, conv65, conv66, conv67, conv68, conv69, conv70, conv71, conv72,
                                           conv73, conv74, conv75, conv76, conv77, conv78, conv79, conv80, conv81, conv82
                                           ])
            #
            conv83 = Base_Model.get_layer('activation_156').output  # 12*12*256
            conv83 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv83)  # 14*14*256
            conv84 = Base_Model.get_layer('activation_158').output  # 12*12*256
            conv84 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv84)  # 14*14*256
            conv85 = Base_Model.get_layer('activation_160').output  # 12*12*256
            conv85 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv85)  # 14*14*256
            conv1_5 = tf.keras.layers.add([conv83, conv84, conv85])
            #
            conv86 = Base_Model.get_layer('activation_161').output  # 12*12*288
            conv86 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv86)  # 14*14*288
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2, conv1_3, conv1_4, conv1_5, conv86], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('activation_163').output  # 5*5*192
                conv1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv1)  # 7*7*192
                conv2 = Base_Model.get_layer('activation_164').output  # 5*5*192
                conv2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv2)  # 7*7*192
                conv3 = Base_Model.get_layer('activation_167').output  # 5*5*192
                conv3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv3)  # 7*7*192
                conv4 = Base_Model.get_layer('activation_168').output  # 5*5*192
                conv4 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv4)  # 7*7*192
                conv5 = Base_Model.get_layer('activation_171').output  # 5*5*192
                conv5 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv5)  # 7*7*192
                conv6 = Base_Model.get_layer('activation_172').output  # 5*5*192
                conv6 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv6)  # 7*7*192
                conv7 = Base_Model.get_layer('activation_175').output  # 5*5*192
                conv7 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv7)  # 7*7*192
                conv8 = Base_Model.get_layer('activation_176').output  # 5*5*192
                conv8 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv8)  # 7*7*192
                conv9 = Base_Model.get_layer('activation_179').output  # 5*5*192
                conv9 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv9)  # 7*7*192
                conv10 = Base_Model.get_layer('activation_180').output  # 5*5*192
                conv10 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv10)  # 7*7*192
                conv11 = Base_Model.get_layer('activation_183').output  # 5*5*192
                conv11 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv11)  # 7*7*192
                conv12 = Base_Model.get_layer('activation_184').output  # 5*5*192
                conv12 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv12)  # 7*7*192
                conv13 = Base_Model.get_layer('activation_187').output  # 5*5*192
                conv13 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv13)  # 7*7*192
                conv14 = Base_Model.get_layer('activation_188').output  # 5*5*192
                conv14 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv14)  # 7*7*192
                conv15 = Base_Model.get_layer('activation_191').output  # 5*5*192
                conv15 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv15)  # 7*7*192
                conv16 = Base_Model.get_layer('activation_192').output  # 5*5*192
                conv16 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv16)  # 7*7*192
                conv17 = Base_Model.get_layer('activation_195').output  # 5*5*192
                conv17 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv17)  # 7*7*192
                conv18 = Base_Model.get_layer('activation_196').output  # 5*5*192
                conv18 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv18)  # 7*7*192
                conv19 = Base_Model.get_layer('activation_199').output  # 5*5*192
                conv19 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv19)  # 7*7*192
                conv20 = Base_Model.get_layer('activation_200').output  # 5*5*192
                conv20 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv20)  # 7*7*192
                conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9,
                                               conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18,
                                               conv19, conv20])
                #
                conv21 = Base_Model.get_layer('activation_165').output  # 5*5*224
                conv21 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv21)  # 7*7*224
                conv22 = Base_Model.get_layer('activation_169').output  # 5*5*224
                conv22 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv22)  # 7*7*224
                conv23 = Base_Model.get_layer('activation_173').output  # 5*5*224
                conv23 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv23)  # 7*7*224
                conv24 = Base_Model.get_layer('activation_177').output  # 5*5*224
                conv24 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv24)  # 7*7*224
                conv25 = Base_Model.get_layer('activation_181').output  # 5*5*224
                conv25 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv25)  # 7*7*224
                conv26 = Base_Model.get_layer('activation_185').output  # 5*5*224
                conv26 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv26)  # 7*7*224
                conv27 = Base_Model.get_layer('activation_189').output  # 5*5*224
                conv27 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv27)  # 7*7*224
                conv28 = Base_Model.get_layer('activation_193').output  # 5*5*224
                conv28 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv28)  # 7*7*224
                conv29 = Base_Model.get_layer('activation_197').output  # 5*5*224
                conv29 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv29)  # 7*7*224
                conv30 = Base_Model.get_layer('activation_201').output  # 5*5*224
                conv30 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv30)  # 7*7*224
                conv1_2 = tf.keras.layers.add([conv21, conv22, conv23, conv24, conv25,
                                               conv26, conv27, conv28, conv29, conv30])
                #
                conv31 = Base_Model.get_layer('activation_166').output  # 5*5*256
                conv31 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv31)  # 7*7*256
                conv32 = Base_Model.get_layer('activation_170').output  # 5*5*256
                conv32 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv32)  # 7*7*256
                conv33 = Base_Model.get_layer('activation_174').output  # 5*5*256
                conv33 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv33)  # 7*7*256
                conv34 = Base_Model.get_layer('activation_178').output  # 5*5*256
                conv34 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv34)  # 7*7*256
                conv35 = Base_Model.get_layer('activation_182').output  # 5*5*256
                conv35 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv35)  # 7*7*256
                conv36 = Base_Model.get_layer('activation_186').output  # 5*5*256
                conv36 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv36)  # 7*7*256
                conv37 = Base_Model.get_layer('activation_190').output  # 5*5*256
                conv37 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv37)  # 7*7*256
                conv38 = Base_Model.get_layer('activation_194').output  # 5*5*256
                conv38 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv38)  # 7*7*256
                conv39 = Base_Model.get_layer('activation_198').output  # 5*5*256
                conv39 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv39)  # 7*7*256
                conv40 = Base_Model.get_layer('activation_202').output  # 5*5*256
                conv40 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv40)  # 7*7*256
                conv1_3 = tf.keras.layers.add([conv31, conv32, conv33, conv34, conv35,
                                               conv36, conv37, conv38, conv39, conv40])
                #
                conv41 = Base_Model.get_layer('activation_159').output  # 5*5*288
                conv41 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv41)  # 7*7*288
                #
                conv42 = Base_Model.get_layer('activation_162').output  # 5*5*320
                conv42 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv42)  # 7*7*320
                #
                conv43 = Base_Model.get_layer('activation_157').output  # 5*5*384
                conv43 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv43)  # 7*7*384
                #
                conv44 = Base_Model.get_layer('conv_7b_ac').output  # 5*5*1536
                conv44 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv44)  # 7*7*1536
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1_1, conv1_2, conv1_3, conv41, conv42, conv43, conv44], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB0(self):
        # UNet Variants with EfficientNetB0 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB0" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels), name='input')

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*96
            conv2 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*96
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*144
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*144
            conv4 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*144
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*144
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*240
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*240
            conv4 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*240
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*240
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*480
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*480
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*480
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*480
            conv6 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*480
            conv7 = Base_Model.get_layer('block5a_activation').output  # 14*14*480
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7])
            #
            conv8 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*672
            conv9 = Base_Model.get_layer('block5b_activation').output  # 14*14*672
            conv10 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*672
            conv11 = Base_Model.get_layer('block5c_activation').output  # 14*14*672
            conv12 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*672
            conv1_2 = tf.keras.layers.add([conv8, conv9, conv10, conv11, conv12])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*672
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1152
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1152
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1152
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1152
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1152
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1152
                conv8 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1152
                conv9 = Base_Model.get_layer('block7a_activation').output  # 7*7*1152
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9])
                #
                conv10 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv10], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="output")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB1(self):
        # UNet Variants with EfficientNetB1 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB1" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*16
            #
            conv2 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            conv3 = Base_Model.get_layer('block1a_activation').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv4 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*96
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv4], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*96
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*144
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*144
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*144
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*144
            conv6 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*144
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*144
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*240
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*240
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*240
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*240
            conv6 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*240
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*240
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*480
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*480
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*480
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*480
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*480
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*480
            conv8 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*480
            conv9 = Base_Model.get_layer('block5a_activation').output  # 14*14*480
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9])
            #
            conv10 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*672
            conv11 = Base_Model.get_layer('block5b_activation').output  # 14*14*672
            conv12 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*672
            conv13 = Base_Model.get_layer('block5c_activation').output  # 14*14*672
            conv14 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*672
            conv15 = Base_Model.get_layer('block5d_activation').output  # 14*14*672
            conv16 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*672
            conv1_2 = tf.keras.layers.add([conv10, conv11, conv12, conv13, conv14, conv15, conv16])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*672
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1152
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1152
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1152
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1152
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1152
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1152
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1152
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1152
                conv10 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1152
                conv11 = Base_Model.get_layer('block7a_activation').output  # 7*7*1152
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11])
                #
                conv12 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*1920
                conv13 = Base_Model.get_layer('block7b_activation').output  # 7*7*1920
                conv1_2 = tf.keras.layers.add([conv12, conv13])
                #
                conv14 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv14], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB2(self):
        # UNet Variants with EfficientNetB2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*16
            #
            conv2 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            conv3 = Base_Model.get_layer('block1a_activation').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv4 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*96
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv4], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*96
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*144
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*144
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*144
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*144
            conv6 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*144
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*144
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*288
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*288
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*288
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*288
            conv6 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*288
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*288
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*528
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*528
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*528
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*528
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*528
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*528
            conv8 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*528
            conv9 = Base_Model.get_layer('block5a_activation').output  # 14*14*528
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9])
            #
            conv10 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*720
            conv11 = Base_Model.get_layer('block5b_activation').output  # 14*14*720
            conv12 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*720
            conv13 = Base_Model.get_layer('block5c_activation').output  # 14*14*720
            conv14 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*720
            conv15 = Base_Model.get_layer('block5d_activation').output  # 14*14*720
            conv16 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*720
            conv1_2 = tf.keras.layers.add([conv10, conv11, conv12, conv13, conv14, conv15, conv16])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*720
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1248
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1248
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1248
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1248
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1248
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1248
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1248
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1248
                conv10 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1248
                conv11 = Base_Model.get_layer('block7a_activation').output  # 7*7*1248
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11])
                #
                conv12 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*2112
                conv13 = Base_Model.get_layer('block7b_activation').output  # 7*7*2112
                conv1_2 = tf.keras.layers.add([conv12, conv13])
                #
                conv14 = Base_Model.get_layer('top_activation').output  # 7*7*1408
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv14], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB3(self):
        # UNet Variants with EfficientNetB3 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB3" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*24
            #
            conv2 = Base_Model.get_layer('stem_activation').output  # 112*112*40
            conv3 = Base_Model.get_layer('block1a_activation').output  # 112*112*40
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv4 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*14
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv4], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*144
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*192
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*192
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*192
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*192
            conv6 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*192
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*192
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*288
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*288
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*288
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*288
            conv6 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*288
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*288
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*576
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*576
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*576
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*576
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*576
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*576
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*576
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*576
            conv10 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*576
            conv11 = Base_Model.get_layer('block5a_activation').output  # 14*14*576
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11])
            #
            conv12 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*816
            conv13 = Base_Model.get_layer('block5b_activation').output  # 14*14*816
            conv14 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*816
            conv15 = Base_Model.get_layer('block5c_activation').output  # 14*14*816
            conv16 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*816
            conv17 = Base_Model.get_layer('block5d_activation').output  # 14*14*816
            conv18 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*816
            conv19 = Base_Model.get_layer('block5e_activation').output  # 14*14*816
            conv20 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*816
            conv1_2 = tf.keras.layers.add([conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*720
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1392
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1392
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1392
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1392
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1392
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1392
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1392
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1392
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1392
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1392
                conv12 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1392
                conv13 = Base_Model.get_layer('block7a_activation').output  # 7*7*1392
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
                #
                conv14 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*2304
                conv15 = Base_Model.get_layer('block7b_activation').output  # 7*7*2304
                conv1_2 = tf.keras.layers.add([conv14, conv15])
                #
                conv16 = Base_Model.get_layer('top_activation').output  # 7*7*1536
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv16], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB4(self):
        # UNet Variants with EfficientNetB4 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB4" + "_" + str(self.decoder_name)
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*24
            #
            conv2 = Base_Model.get_layer('stem_activation').output  # 112*112*48
            conv3 = Base_Model.get_layer('block1a_activation').output  # 112*112*48
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv4 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*14
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv4], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*144
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*192
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*192
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*192
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*192
            conv6 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*192
            conv7 = Base_Model.get_layer('block2d_activation').output  # 56*56*192
            conv8 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*192
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*192
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*336
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*336
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*336
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*336
            conv6 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*336
            conv7 = Base_Model.get_layer('block3d_activation').output  # 28*28*336
            conv8 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*336
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*336
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*672
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*672
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*672
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*672
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*672
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*672
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*672
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*672
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*672
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*672
            conv12 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*672
            conv13 = Base_Model.get_layer('block5a_activation').output  # 14*14*672
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*960
            conv15 = Base_Model.get_layer('block5b_activation').output  # 14*14*960
            conv16 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*960
            conv17 = Base_Model.get_layer('block5c_activation').output  # 14*14*960
            conv18 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*960
            conv19 = Base_Model.get_layer('block5d_activation').output  # 14*14*960
            conv20 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*960
            conv21 = Base_Model.get_layer('block5e_activation').output  # 14*14*960
            conv22 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*960
            conv23 = Base_Model.get_layer('block5f_activation').output  # 14*14*960
            conv24 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*960
            conv1_2 = tf.keras.layers.add([conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*960
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1632
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1632
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1632
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1632
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1632
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1632
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1632
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1632
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1632
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1632
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1632
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1632
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1632
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1632
                conv16 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1632
                conv17 = Base_Model.get_layer('block7a_activation').output  # 7*7*1632
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17])
                #
                conv18 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*2688
                conv19 = Base_Model.get_layer('block7b_activation').output  # 7*7*2688
                conv1_2 = tf.keras.layers.add([conv18, conv19])
                #
                conv20 = Base_Model.get_layer('top_activation').output  # 7*7*1792
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv20], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB5(self):
        # UNet Variants with EfficientNetB5 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB5" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*24
            conv2 = Base_Model.get_layer('block1c_activation').output  # 112*112*24
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('stem_activation').output  # 112*112*48
            conv4 = Base_Model.get_layer('block1a_activation').output  # 112*112*48
            conv1_2 = tf.keras.layers.add([conv3, conv4])
            #
            conv5 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*14
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2, conv5], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*144
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*240
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*240
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*240
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*240
            conv6 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*240
            conv7 = Base_Model.get_layer('block2d_activation').output  # 56*56*240
            conv8 = Base_Model.get_layer('block2e_expand_activation').output  # 56*56*240
            conv9 = Base_Model.get_layer('block2e_activation').output  # 56*56*240
            conv10 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*240
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*240
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*384
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*384
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*384
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*384
            conv6 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*384
            conv7 = Base_Model.get_layer('block3d_activation').output  # 28*28*384
            conv8 = Base_Model.get_layer('block3e_expand_activation').output  # 28*28*384
            conv9 = Base_Model.get_layer('block3e_activation').output  # 28*28*384
            conv10 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*384
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*384
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*768
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*768
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*768
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*768
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*768
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*768
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*768
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*768
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*768
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*768
            conv12 = Base_Model.get_layer('block4g_expand_activation').output  # 14*14*768
            conv13 = Base_Model.get_layer('block4g_activation').output  # 14*14*768
            conv14 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*768
            conv15 = Base_Model.get_layer('block5a_activation').output  # 14*14*768
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15])
            #
            conv16 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*1056
            conv17 = Base_Model.get_layer('block5b_activation').output  # 14*14*1056
            conv18 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*1056
            conv19 = Base_Model.get_layer('block5c_activation').output  # 14*14*1056
            conv20 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*1056
            conv21 = Base_Model.get_layer('block5d_activation').output  # 14*14*1056
            conv22 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*1056
            conv23 = Base_Model.get_layer('block5e_activation').output  # 14*14*1056
            conv24 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*1056
            conv25 = Base_Model.get_layer('block5f_activation').output  # 14*14*1056
            conv26 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*1056
            conv27 = Base_Model.get_layer('block5g_activation').output  # 14*14*1056
            conv28 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*1056
            conv1_2 = tf.keras.layers.add([conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*1056
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1824
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1824
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1824
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1824
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1824
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1824
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1824
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1824
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1824
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1824
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1824
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1824
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1824
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1824
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*1824
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*1824
                conv18 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1824
                conv19 = Base_Model.get_layer('block7a_activation').output  # 7*7*1824
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19])
                #
                conv20 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*3072
                conv21 = Base_Model.get_layer('block7b_activation').output  # 7*7*3072
                conv22 = Base_Model.get_layer('block7c_expand_activation').output  # 7*7*3072
                conv23 = Base_Model.get_layer('block7c_activation').output  # 7*7*3072
                conv1_2 = tf.keras.layers.add([conv20, conv21, conv22, conv23])
                #
                conv24 = Base_Model.get_layer('top_activation').output  # 7*7*2048
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv24], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB6(self):
        # UNet Variants with EfficientNetB6 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB6" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*32
            conv2 = Base_Model.get_layer('block1c_activation').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv1, conv2])
            #
            conv3 = Base_Model.get_layer('stem_activation').output  # 112*112*56
            conv4 = Base_Model.get_layer('block1a_activation').output  # 112*112*56
            conv1_2 = tf.keras.layers.add([conv3, conv4])
            #
            conv5 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*19
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2, conv5], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*144
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*240
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*240
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*240
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*240
            conv6 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*240
            conv7 = Base_Model.get_layer('block2d_activation').output  # 56*56*240
            conv8 = Base_Model.get_layer('block2e_expand_activation').output  # 56*56*240
            conv9 = Base_Model.get_layer('block2e_activation').output  # 56*56*240
            conv10 = Base_Model.get_layer('block2f_expand_activation').output  # 56*56*240
            conv11 = Base_Model.get_layer('block2f_activation').output  # 56*56*240
            conv12 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*240
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*240
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*432
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*432
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*432
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*432
            conv6 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*432
            conv7 = Base_Model.get_layer('block3d_activation').output  # 28*28*432
            conv8 = Base_Model.get_layer('block3e_expand_activation').output  # 28*28*432
            conv9 = Base_Model.get_layer('block3e_activation').output  # 28*28*432
            conv10 = Base_Model.get_layer('block3f_expand_activation').output  # 28*28*432
            conv11 = Base_Model.get_layer('block3f_activation').output  # 28*28*432
            conv12 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*432
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*432
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*864
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*864
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*864
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*864
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*864
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*864
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*864
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*864
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*864
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*864
            conv12 = Base_Model.get_layer('block4g_expand_activation').output  # 14*14*864
            conv13 = Base_Model.get_layer('block4g_activation').output  # 14*14*864
            conv14 = Base_Model.get_layer('block4h_expand_activation').output  # 14*14*864
            conv15 = Base_Model.get_layer('block4h_activation').output  # 14*14*864
            conv16 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*864
            conv17 = Base_Model.get_layer('block5a_activation').output  # 14*14*864
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17])
            #
            conv18 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*1200
            conv19 = Base_Model.get_layer('block5b_activation').output  # 14*14*1200
            conv20 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*1200
            conv21 = Base_Model.get_layer('block5c_activation').output  # 14*14*1200
            conv22 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*1200
            conv23 = Base_Model.get_layer('block5d_activation').output  # 14*14*1200
            conv24 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*1200
            conv25 = Base_Model.get_layer('block5e_activation').output  # 14*14*1200
            conv26 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*1200
            conv27 = Base_Model.get_layer('block5f_activation').output  # 14*14*1200
            conv28 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*1200
            conv29 = Base_Model.get_layer('block5g_activation').output  # 14*14*1200
            conv30 = Base_Model.get_layer('block5h_expand_activation').output  # 14*14*1200
            conv31 = Base_Model.get_layer('block5h_activation').output  # 14*14*1200
            conv32 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*1200
            conv1_2 = tf.keras.layers.add([conv18, conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*1200
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*2064
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*2064
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*2064
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*2064
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*2064
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*2064
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*2064
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*2064
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*2064
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*2064
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*2064
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*2064
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*2064
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*2064
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*2064
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*2064
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*2064
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*2064
                conv20 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*2064
                conv21 = Base_Model.get_layer('block6k_activation').output  # 7*7*2064
                conv22 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*2064
                conv23 = Base_Model.get_layer('block7a_activation').output  # 7*7*2064
                conv1_1 = tf.keras.layers.add(
                    [conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23])
                #
                conv24 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*3456
                conv25 = Base_Model.get_layer('block7b_activation').output  # 7*7*3456
                conv26 = Base_Model.get_layer('block7c_expand_activation').output  # 7*7*3456
                conv27 = Base_Model.get_layer('block7c_activation').output  # 7*7*3456
                conv1_2 = tf.keras.layers.add([conv24, conv25, conv26, conv27])
                #
                conv28 = Base_Model.get_layer('top_activation').output  # 7*7*2304
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv28], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetB7(self):
        # UNet Variants with EfficientNetB7 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetB7" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('block1b_activation').output  # 112*112*32
            conv2 = Base_Model.get_layer('block1c_activation').output  # 112*112*32
            conv3 = Base_Model.get_layer('block1d_activation').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3])
            #
            conv4 = Base_Model.get_layer('stem_activation').output  # 112*112*64
            conv5 = Base_Model.get_layer('block1a_activation').output  # 112*112*64
            conv1_2 = tf.keras.layers.add([conv4, conv5])
            #
            conv6 = Base_Model.get_layer('block2a_expand_activation').output  # 112*112*19
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2, conv6], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_activation').output  # 56*56*192
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*288
            conv3 = Base_Model.get_layer('block2b_activation').output  # 56*56*288
            conv4 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*288
            conv5 = Base_Model.get_layer('block2c_activation').output  # 56*56*288
            conv6 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*288
            conv7 = Base_Model.get_layer('block2d_activation').output  # 56*56*288
            conv8 = Base_Model.get_layer('block2e_expand_activation').output  # 56*56*288
            conv9 = Base_Model.get_layer('block2e_activation').output  # 56*56*288
            conv10 = Base_Model.get_layer('block2f_expand_activation').output  # 56*56*288
            conv11 = Base_Model.get_layer('block2f_activation').output  # 56*56*288
            conv12 = Base_Model.get_layer('block2g_expand_activation').output  # 56*56*288
            conv13 = Base_Model.get_layer('block2g_activation').output  # 56*56*288
            conv14 = Base_Model.get_layer('block3a_expand_activation').output  # 56*56*288
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_activation').output  # 28*28*288
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*480
            conv3 = Base_Model.get_layer('block3b_activation').output  # 28*28*480
            conv4 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*480
            conv5 = Base_Model.get_layer('block3c_activation').output  # 28*28*480
            conv6 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*480
            conv7 = Base_Model.get_layer('block3d_activation').output  # 28*28*480
            conv8 = Base_Model.get_layer('block3e_expand_activation').output  # 28*28*480
            conv9 = Base_Model.get_layer('block3e_activation').output  # 28*28*480
            conv10 = Base_Model.get_layer('block3f_expand_activation').output  # 28*28*480
            conv11 = Base_Model.get_layer('block3f_activation').output  # 28*28*480
            conv12 = Base_Model.get_layer('block3g_expand_activation').output  # 28*28*480
            conv13 = Base_Model.get_layer('block3g_activation').output  # 28*28*480
            conv14 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*480
            #
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*480
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*960
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*960
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*960
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*960
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*960
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*960
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*960
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*960
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*960
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*960
            conv12 = Base_Model.get_layer('block4g_expand_activation').output  # 14*14*960
            conv13 = Base_Model.get_layer('block4g_activation').output  # 14*14*960
            conv14 = Base_Model.get_layer('block4h_expand_activation').output  # 14*14*960
            conv15 = Base_Model.get_layer('block4h_activation').output  # 14*14*960
            conv16 = Base_Model.get_layer('block4i_expand_activation').output  # 14*14*960
            conv17 = Base_Model.get_layer('block4i_activation').output  # 14*14*960
            conv18 = Base_Model.get_layer('block4j_expand_activation').output  # 14*14*960
            conv19 = Base_Model.get_layer('block4j_activation').output  # 14*14*960
            conv20 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*960
            conv21 = Base_Model.get_layer('block5a_activation').output  # 14*14*960
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21])
            #
            conv22 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*1344
            conv23 = Base_Model.get_layer('block5b_activation').output  # 14*14*1344
            conv24 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*1344
            conv25 = Base_Model.get_layer('block5c_activation').output  # 14*14*1344
            conv26 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*1344
            conv27 = Base_Model.get_layer('block5d_activation').output  # 14*14*1344
            conv28 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*1344
            conv29 = Base_Model.get_layer('block5e_activation').output  # 14*14*1344
            conv30 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*1344
            conv31 = Base_Model.get_layer('block5f_activation').output  # 14*14*1344
            conv32 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*1344
            conv33 = Base_Model.get_layer('block5g_activation').output  # 14*14*1344
            conv34 = Base_Model.get_layer('block5h_expand_activation').output  # 14*14*1344
            conv35 = Base_Model.get_layer('block5h_activation').output  # 14*14*1344
            conv36 = Base_Model.get_layer('block5i_expand_activation').output  # 14*14*1344
            conv37 = Base_Model.get_layer('block5i_activation').output  # 14*14*1344
            conv38 = Base_Model.get_layer('block5j_expand_activation').output  # 14*14*1344
            conv39 = Base_Model.get_layer('block5j_activation').output  # 14*14*1344
            conv40 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*1344
            conv1_2 = tf.keras.layers.add([conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35, conv36, conv37, conv38, conv39, conv40])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*1344
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*2304
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*2304
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*2304
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*2304
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*2304
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*2304
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*2304
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*2304
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*2304
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*2304
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*2304
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*2304
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*2304
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*2304
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*2304
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*2304
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*2304
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*2304
                conv20 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*2304
                conv21 = Base_Model.get_layer('block6k_activation').output  # 7*7*2304
                conv22 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*2304
                conv23 = Base_Model.get_layer('block6k_activation').output  # 7*7*2304
                conv24 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*2304
                conv25 = Base_Model.get_layer('block6k_activation').output  # 7*7*2304
                conv26 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*2304
                conv27 = Base_Model.get_layer('block7a_activation').output  # 7*7*2064
                conv1_1 = tf.keras.layers.add(
                    [conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                     conv25, conv26, conv27])
                #
                conv28 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*3840
                conv29 = Base_Model.get_layer('block7b_activation').output  # 7*7*3840
                conv30 = Base_Model.get_layer('block7c_expand_activation').output  # 7*7*3840
                conv31 = Base_Model.get_layer('block7c_activation').output  # 7*7*3840
                conv32 = Base_Model.get_layer('block7d_expand_activation').output  # 7*7*3840
                conv33 = Base_Model.get_layer('block7d_activation').output  # 7*7*3840
                conv1_2 = tf.keras.layers.add([conv28, conv29, conv30, conv31, conv32, conv33])
                #
                conv34 = Base_Model.get_layer('top_activation').output  # 7*7*2560
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv34], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2B0(self):
        # UNet Variants with EfficientNetV2B0 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2B0" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*16
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*64
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*128
            #
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*128
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*192
            conv3 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*192
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*192
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*384
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*384
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*384
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*384
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5])
            #
            conv6 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*576
            conv7 = Base_Model.get_layer('block5a_activation').output  # 14*14*576
            conv1_2 = tf.keras.layers.add([conv6, conv7])
            #
            conv8 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*672
            conv9 = Base_Model.get_layer('block5b_activation').output  # 14*14*672
            conv10 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*672
            conv11 = Base_Model.get_layer('block5c_activation').output  # 14*14*672
            conv12 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*672
            conv13 = Base_Model.get_layer('block5d_activation').output  # 14*14*672
            conv14 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*672
            conv15 = Base_Model.get_layer('block5e_activation').output  # 14*14*672
            conv16 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*672
            conv1_3 = tf.keras.layers.add([conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*672
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1152
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1152
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1152
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1152
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1152
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1152
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1152
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1152
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1152
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1152
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1152
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1152
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1152
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1152
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15])
                #
                conv16 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv16], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2B1(self):
        # UNet Variants with EfficientNetV2B1 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2B1" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            #
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*16
            conv3 = Base_Model.get_layer('block1b_project_activation').output  # 112*112*16
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*64
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*128
            conv3 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*128
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*192
            conv3 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*192
            conv4 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*192
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*192
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*384
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*384
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*384
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*384
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*384
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*384
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7])
            #
            conv8 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*576
            conv9 = Base_Model.get_layer('block5a_activation').output  # 14*14*576
            conv1_2 = tf.keras.layers.add([conv8, conv9])
            #
            conv10 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*672
            conv11 = Base_Model.get_layer('block5b_activation').output  # 14*14*672
            conv12 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*672
            conv13 = Base_Model.get_layer('block5c_activation').output  # 14*14*672
            conv14 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*672
            conv15 = Base_Model.get_layer('block5d_activation').output  # 14*14*672
            conv16 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*672
            conv17 = Base_Model.get_layer('block5e_activation').output  # 14*14*672
            conv18 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*672
            conv19 = Base_Model.get_layer('block5f_activation').output  # 14*14*672
            conv20 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*672
            conv1_3 = tf.keras.layers.add([conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*672
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1152
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1152
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1152
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1152
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1152
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1152
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1152
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1152
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1152
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1152
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1152
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1152
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1152
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1152
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*1152
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*1152
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17])
                #
                conv18 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv18], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2B2(self):
        # UNet Variants with EfficientNetV2B2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2B2" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            #
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*16
            conv3 = Base_Model.get_layer('block1b_project_activation').output  # 112*112*16
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*64
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*128
            conv3 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*128
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*224
            conv3 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*224
            conv4 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*224
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*224
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*416
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*416
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*416
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*416
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*416
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*416
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7])
            #
            conv8 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*624
            conv9 = Base_Model.get_layer('block5a_activation').output  # 14*14*624
            conv1_2 = tf.keras.layers.add([conv8, conv9])
            #
            conv10 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*720
            conv11 = Base_Model.get_layer('block5b_activation').output  # 14*14*720
            conv12 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*720
            conv13 = Base_Model.get_layer('block5c_activation').output  # 14*14*720
            conv14 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*720
            conv15 = Base_Model.get_layer('block5d_activation').output  # 14*14*720
            conv16 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*720
            conv17 = Base_Model.get_layer('block5e_activation').output  # 14*14*720
            conv18 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*720
            conv19 = Base_Model.get_layer('block5f_activation').output  # 14*14*720
            conv20 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*720
            conv1_3 = tf.keras.layers.add([conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*720
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1248
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1248
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1248
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1248
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1248
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1248
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1248
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1248
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1248
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1248
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1248
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1248
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1248
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1248
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*1248
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*1248
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*1248
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*1248
                conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19])
                #
                conv20 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv20], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2B3(self):
        # UNet Variants with EfficientNetV2B3 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2B3" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*40
            #
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*16
            conv3 = Base_Model.get_layer('block1b_project_activation').output  # 112*112*16
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*64
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*160
            conv3 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*160
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*160
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*224
            conv3 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*224
            conv4 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*224
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*224
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*448
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*448
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*448
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*448
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*448
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*448
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*448
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*448
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9])
            #
            conv10 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*672
            conv11 = Base_Model.get_layer('block5a_activation').output  # 14*14*672
            conv1_2 = tf.keras.layers.add([conv10, conv11])
            #
            conv12 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*816
            conv13 = Base_Model.get_layer('block5b_activation').output  # 14*14*816
            conv14 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*816
            conv15 = Base_Model.get_layer('block5c_activation').output  # 14*14*816
            conv16 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*816
            conv17 = Base_Model.get_layer('block5d_activation').output  # 14*14*816
            conv18 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*816
            conv19 = Base_Model.get_layer('block5e_activation').output  # 14*14*816
            conv20 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*816
            conv21 = Base_Model.get_layer('block5f_activation').output  # 14*14*816
            conv22 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*816
            conv23 = Base_Model.get_layer('block5g_activation').output  # 14*14*816
            conv24 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*816
            conv1_3 = tf.keras.layers.add([conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*816
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1392
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1392
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1392
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1392
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1392
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1392
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1392
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1392
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1392
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1392
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1392
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1392
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1392
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1392
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*1392
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*1392
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*1392
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*1392
                conv20 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*1392
                conv21 = Base_Model.get_layer('block6k_activation').output  # 7*7*1392
                conv22 = Base_Model.get_layer('block6l_expand_activation').output  # 7*7*1392
                conv23 = Base_Model.get_layer('block6l_activation').output  # 7*7*1392
                conv1_1 = tf.keras.layers.add(
                    [conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23])
                #
                conv20 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv20], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2S(self):
        # UNet Variants with EfficientNetV2S (Small) ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2S" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*24
            #
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*24
            conv3 = Base_Model.get_layer('block1b_project_activation').output  # 112*112*24
            conv1_1 = tf.keras.layers.add([conv2, conv3])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*96
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*192
            conv3 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*192
            conv4 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*192
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*192
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*256
            conv3 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*256
            conv4 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*256
            conv5 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*256
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*256
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*512
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*512
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*512
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*512
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*512
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*512
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*512
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*512
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*512
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*512
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11])
            #
            conv12 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*768
            conv13 = Base_Model.get_layer('block5a_activation').output  # 14*14*768
            conv1_2 = tf.keras.layers.add([conv12, conv13])
            #
            conv14 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*960
            conv15 = Base_Model.get_layer('block5b_activation').output  # 14*14*960
            conv16 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*960
            conv17 = Base_Model.get_layer('block5c_activation').output  # 14*14*960
            conv18 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*960
            conv19 = Base_Model.get_layer('block5d_activation').output  # 14*14*960
            conv20 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*960
            conv21 = Base_Model.get_layer('block5e_activation').output  # 14*14*960
            conv22 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*960
            conv23 = Base_Model.get_layer('block5f_activation').output  # 14*14*960
            conv24 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*960
            conv25 = Base_Model.get_layer('block5g_activation').output  # 14*14*960
            conv26 = Base_Model.get_layer('block5h_expand_activation').output  # 14*14*960
            conv27 = Base_Model.get_layer('block5h_activation').output  # 14*14*960
            conv28 = Base_Model.get_layer('block5i_expand_activation').output  # 14*14*960
            conv29 = Base_Model.get_layer('block5i_activation').output  # 14*14*960
            conv30 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*960
            conv1_3 = tf.keras.layers.add([conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21,
                                           conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*960
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1536
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1536
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1536
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1536
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1536
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1536
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1536
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1536
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1536
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1536
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1536
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1536
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1536
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1536
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*1536
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*1536
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*1536
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*1536
                conv20 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*1536
                conv21 = Base_Model.get_layer('block6k_activation').output  # 7*7*1536
                conv22 = Base_Model.get_layer('block6l_expand_activation').output  # 7*7*1536
                conv23 = Base_Model.get_layer('block6l_activation').output  # 7*7*1536
                conv24 = Base_Model.get_layer('block6m_expand_activation').output  # 7*7*1536
                conv25 = Base_Model.get_layer('block6m_activation').output  # 7*7*1536
                conv26 = Base_Model.get_layer('block6n_expand_activation').output  # 7*7*1536
                conv27 = Base_Model.get_layer('block6n_activation').output  # 7*7*1536
                conv28 = Base_Model.get_layer('block6o_expand_activation').output  # 7*7*1536
                conv29 = Base_Model.get_layer('block6o_activation').output  # 7*7*1536
                conv1_1 = tf.keras.layers.add(
                    [conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15,
                     conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29])
                #
                conv20 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv20], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2M(self):
        # UNet Variants with EfficientNetV2M (Medium) ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2M" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*24
            #
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*24
            conv3 = Base_Model.get_layer('block1b_project_activation').output  # 112*112*24
            conv4 = Base_Model.get_layer('block1c_project_activation').output  # 112*112*24
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*96
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*192
            conv3 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*192
            conv4 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*192
            conv5 = Base_Model.get_layer('block2e_expand_activation').output  # 56*56*192
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*192
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*320
            conv3 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*320
            conv4 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*320
            conv5 = Base_Model.get_layer('block3e_expand_activation').output  # 28*28*320
            conv6 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*320
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*320
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*640
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*640
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*640
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*640
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*640
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*640
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*640
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*640
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*640
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*640
            conv12 = Base_Model.get_layer('block4g_expand_activation').output  # 14*14*640
            conv13 = Base_Model.get_layer('block4g_activation').output  # 14*14*640
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*960
            conv15 = Base_Model.get_layer('block5a_activation').output  # 14*14*960
            conv1_2 = tf.keras.layers.add([conv14, conv15])
            #
            conv16 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*1056
            conv17 = Base_Model.get_layer('block5b_activation').output  # 14*14*1056
            conv18 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*1056
            conv19 = Base_Model.get_layer('block5c_activation').output  # 14*14*1056
            conv20 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*1056
            conv21 = Base_Model.get_layer('block5d_activation').output  # 14*14*1056
            conv22 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*1056
            conv23 = Base_Model.get_layer('block5e_activation').output  # 14*14*1056
            conv24 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*1056
            conv25 = Base_Model.get_layer('block5f_activation').output  # 14*14*1056
            conv26 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*1056
            conv27 = Base_Model.get_layer('block5g_activation').output  # 14*14*1056
            conv28 = Base_Model.get_layer('block5h_expand_activation').output  # 14*14*1056
            conv29 = Base_Model.get_layer('block5h_activation').output  # 14*14*1056
            conv30 = Base_Model.get_layer('block5i_expand_activation').output  # 14*14*1056
            conv31 = Base_Model.get_layer('block5i_activation').output  # 14*14*1056
            conv32 = Base_Model.get_layer('block5j_expand_activation').output  # 14*14*1056
            conv33 = Base_Model.get_layer('block5j_activation').output  # 14*14*1056
            conv34 = Base_Model.get_layer('block5k_expand_activation').output  # 14*14*1056
            conv35 = Base_Model.get_layer('block5k_activation').output  # 14*14*1056
            conv36 = Base_Model.get_layer('block5l_expand_activation').output  # 14*14*1056
            conv37 = Base_Model.get_layer('block5l_activation').output  # 14*14*1056
            conv38 = Base_Model.get_layer('block5m_expand_activation').output  # 14*14*1056
            conv39 = Base_Model.get_layer('block5m_activation').output  # 14*14*1056
            conv40 = Base_Model.get_layer('block5n_expand_activation').output  # 14*14*1056
            conv41 = Base_Model.get_layer('block5n_activation').output  # 14*14*1056
            conv42 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*1056
            conv1_3 = tf.keras.layers.add([conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                                           conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33,
                                           conv34, conv35, conv36, conv37, conv38, conv39, conv40, conv41, conv42])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*1056
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*1824
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*1824
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*1824
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*1824
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*1824
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*1824
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*1824
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*1824
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*1824
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*1824
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*1824
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*1824
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*1824
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*1824
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*1824
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*1824
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*1824
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*1824
                conv20 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*1824
                conv21 = Base_Model.get_layer('block6k_activation').output  # 7*7*1824
                conv22 = Base_Model.get_layer('block6l_expand_activation').output  # 7*7*1824
                conv23 = Base_Model.get_layer('block6l_activation').output  # 7*7*1824
                conv24 = Base_Model.get_layer('block6m_expand_activation').output  # 7*7*1824
                conv25 = Base_Model.get_layer('block6m_activation').output  # 7*7*1824
                conv26 = Base_Model.get_layer('block6n_expand_activation').output  # 7*7*1824
                conv27 = Base_Model.get_layer('block6n_activation').output  # 7*7*1824
                conv28 = Base_Model.get_layer('block6o_expand_activation').output  # 7*7*1824
                conv29 = Base_Model.get_layer('block6o_activation').output  # 7*7*1824
                conv30 = Base_Model.get_layer('block6p_expand_activation').output  # 7*7*1824
                conv31 = Base_Model.get_layer('block6p_activation').output  # 7*7*1824
                conv32 = Base_Model.get_layer('block6q_expand_activation').output  # 7*7*1824
                conv33 = Base_Model.get_layer('block6q_activation').output  # 7*7*1824
                conv34 = Base_Model.get_layer('block6r_expand_activation').output  # 7*7*1824
                conv35 = Base_Model.get_layer('block6r_activation').output  # 7*7*1824
                conv36 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*1824
                conv37 = Base_Model.get_layer('block7a_activation').output  # 7*7*1824
                conv1_1 = tf.keras.layers.add(
                    [conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13,
                     conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                     conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35, conv36, conv37])
                #
                conv38 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*3072
                conv39 = Base_Model.get_layer('block7b_activation').output  # 7*7*3072
                conv40 = Base_Model.get_layer('block7c_expand_activation').output  # 7*7*3072
                conv41 = Base_Model.get_layer('block7c_activation').output  # 7*7*3072
                conv42 = Base_Model.get_layer('block7d_expand_activation').output  # 7*7*3072
                conv43 = Base_Model.get_layer('block7d_activation').output  # 7*7*3072
                conv44 = Base_Model.get_layer('block7e_expand_activation').output  # 7*7*3072
                conv45 = Base_Model.get_layer('block7e_activation').output  # 7*7*3072
                conv1_2 = tf.keras.layers.add([conv38, conv39, conv40, conv41, conv42, conv43, conv44, conv45])
                #
                conv20 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv20], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def EfficientNetV2L(self):
        # UNet Variants with EfficientNetV2L (Large) ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "EfficientNetV2L" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv1 = Base_Model.get_layer('stem_activation').output  # 112*112*32
            #
            conv2 = Base_Model.get_layer('block1a_project_activation').output  # 112*112*32
            conv3 = Base_Model.get_layer('block1b_project_activation').output  # 112*112*32
            conv4 = Base_Model.get_layer('block1c_project_activation').output  # 112*112*32
            conv5 = Base_Model.get_layer('block1d_project_activation').output  # 112*112*32
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('block2a_expand_activation').output  # 56*56*128
            #
            conv2 = Base_Model.get_layer('block2b_expand_activation').output  # 56*56*256
            conv3 = Base_Model.get_layer('block2c_expand_activation').output  # 56*56*256
            conv4 = Base_Model.get_layer('block2d_expand_activation').output  # 56*56*256
            conv5 = Base_Model.get_layer('block2e_expand_activation').output  # 56*56*256
            conv6 = Base_Model.get_layer('block2f_expand_activation').output  # 56*56*256
            conv7 = Base_Model.get_layer('block2g_expand_activation').output  # 56*56*256
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('block3a_expand_activation').output  # 28*28*192
            #
            conv2 = Base_Model.get_layer('block3b_expand_activation').output  # 28*28*384
            conv3 = Base_Model.get_layer('block3c_expand_activation').output  # 28*28*384
            conv4 = Base_Model.get_layer('block3d_expand_activation').output  # 28*28*384
            conv5 = Base_Model.get_layer('block3e_expand_activation').output  # 28*28*384
            conv6 = Base_Model.get_layer('block3f_expand_activation').output  # 28*28*384
            conv7 = Base_Model.get_layer('block3g_expand_activation').output  # 28*28*384
            conv8 = Base_Model.get_layer('block4a_expand_activation').output  # 28*28*384
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('block4a_activation').output  # 14*14*384
            #
            conv2 = Base_Model.get_layer('block4b_expand_activation').output  # 14*14*768
            conv3 = Base_Model.get_layer('block4b_activation').output  # 14*14*768
            conv4 = Base_Model.get_layer('block4c_expand_activation').output  # 14*14*768
            conv5 = Base_Model.get_layer('block4c_activation').output  # 14*14*768
            conv6 = Base_Model.get_layer('block4d_expand_activation').output  # 14*14*768
            conv7 = Base_Model.get_layer('block4d_activation').output  # 14*14*768
            conv8 = Base_Model.get_layer('block4e_expand_activation').output  # 14*14*768
            conv9 = Base_Model.get_layer('block4e_activation').output  # 14*14*768
            conv10 = Base_Model.get_layer('block4f_expand_activation').output  # 14*14*768
            conv11 = Base_Model.get_layer('block4f_activation').output  # 14*14*768
            conv12 = Base_Model.get_layer('block4g_expand_activation').output  # 14*14*768
            conv13 = Base_Model.get_layer('block4g_activation').output  # 14*14*768
            conv14 = Base_Model.get_layer('block4h_expand_activation').output  # 14*14*768
            conv15 = Base_Model.get_layer('block4h_activation').output  # 14*14*768
            conv16 = Base_Model.get_layer('block4i_expand_activation').output  # 14*14*768
            conv17 = Base_Model.get_layer('block4i_activation').output  # 14*14*768
            conv18 = Base_Model.get_layer('block4j_expand_activation').output  # 14*14*768
            conv19 = Base_Model.get_layer('block4j_activation').output  # 14*14*768
            conv1_1 = tf.keras.layers.add([conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10,
                                           conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19])
            #
            conv14 = Base_Model.get_layer('block5a_expand_activation').output  # 14*14*1152
            conv15 = Base_Model.get_layer('block5a_activation').output  # 14*14*1152
            conv1_2 = tf.keras.layers.add([conv14, conv15])
            #
            conv16 = Base_Model.get_layer('block5b_expand_activation').output  # 14*14*1344
            conv17 = Base_Model.get_layer('block5b_activation').output  # 14*14*1344
            conv18 = Base_Model.get_layer('block5c_expand_activation').output  # 14*14*1344
            conv19 = Base_Model.get_layer('block5c_activation').output  # 14*14*1344
            conv20 = Base_Model.get_layer('block5d_expand_activation').output  # 14*14*1344
            conv21 = Base_Model.get_layer('block5d_activation').output  # 14*14*1344
            conv22 = Base_Model.get_layer('block5e_expand_activation').output  # 14*14*1344
            conv23 = Base_Model.get_layer('block5e_activation').output  # 14*14*1344
            conv24 = Base_Model.get_layer('block5f_expand_activation').output  # 14*14*1344
            conv25 = Base_Model.get_layer('block5f_activation').output  # 14*14*1344
            conv26 = Base_Model.get_layer('block5g_expand_activation').output  # 14*14*1344
            conv27 = Base_Model.get_layer('block5g_activation').output  # 14*14*1344
            conv28 = Base_Model.get_layer('block5h_expand_activation').output  # 14*14*1344
            conv29 = Base_Model.get_layer('block5h_activation').output  # 14*14*1344
            conv30 = Base_Model.get_layer('block5i_expand_activation').output  # 14*14*1344
            conv31 = Base_Model.get_layer('block5i_activation').output  # 14*14*1344
            conv32 = Base_Model.get_layer('block5j_expand_activation').output  # 14*14*1344
            conv33 = Base_Model.get_layer('block5j_activation').output  # 14*14*1344
            conv34 = Base_Model.get_layer('block5k_expand_activation').output  # 14*14*1344
            conv35 = Base_Model.get_layer('block5k_activation').output  # 14*14*1344
            conv36 = Base_Model.get_layer('block5l_expand_activation').output  # 14*14*1344
            conv37 = Base_Model.get_layer('block5l_activation').output  # 14*14*1344
            conv38 = Base_Model.get_layer('block5m_expand_activation').output  # 14*14*1344
            conv39 = Base_Model.get_layer('block5m_activation').output  # 14*14*1344
            conv40 = Base_Model.get_layer('block5n_expand_activation').output  # 14*14*1344
            conv41 = Base_Model.get_layer('block5n_activation').output  # 14*14*1344
            conv42 = Base_Model.get_layer('block5o_expand_activation').output  # 14*14*1344
            conv43 = Base_Model.get_layer('block5o_activation').output  # 14*14*1344
            conv44 = Base_Model.get_layer('block5p_expand_activation').output  # 14*14*1344
            conv45 = Base_Model.get_layer('block5p_activation').output  # 14*14*1344
            conv46 = Base_Model.get_layer('block5q_expand_activation').output  # 14*14*1344
            conv47 = Base_Model.get_layer('block5q_activation').output  # 14*14*1344
            conv48 = Base_Model.get_layer('block5r_expand_activation').output  # 14*14*1344
            conv49 = Base_Model.get_layer('block5r_activation').output  # 14*14*1344
            conv50 = Base_Model.get_layer('block5s_expand_activation').output  # 14*14*1344
            conv51 = Base_Model.get_layer('block5s_activation').output  # 14*14*1344
            conv52 = Base_Model.get_layer('block6a_expand_activation').output  # 14*14*1344
            conv1_3 = tf.keras.layers.add([conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                                           conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33,
                                           conv34, conv35, conv36, conv37, conv38, conv39, conv40, conv41, conv42,
                                           conv43, conv44, conv45, conv46, conv47, conv48, conv49, conv50, conv51, conv52])
            #
            conv = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv1_3], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('block6a_activation').output  # 7*7*1344
                #
                conv2 = Base_Model.get_layer('block6b_expand_activation').output  # 7*7*2304
                conv3 = Base_Model.get_layer('block6b_activation').output  # 7*7*2304
                conv4 = Base_Model.get_layer('block6c_expand_activation').output  # 7*7*2304
                conv5 = Base_Model.get_layer('block6c_activation').output  # 7*7*2304
                conv6 = Base_Model.get_layer('block6d_expand_activation').output  # 7*7*2304
                conv7 = Base_Model.get_layer('block6d_activation').output  # 7*7*2304
                conv8 = Base_Model.get_layer('block6e_expand_activation').output  # 7*7*2304
                conv9 = Base_Model.get_layer('block6e_activation').output  # 7*7*2304
                conv10 = Base_Model.get_layer('block6f_expand_activation').output  # 7*7*2304
                conv11 = Base_Model.get_layer('block6f_activation').output  # 7*7*2304
                conv12 = Base_Model.get_layer('block6g_expand_activation').output  # 7*7*2304
                conv13 = Base_Model.get_layer('block6g_activation').output  # 7*7*2304
                conv14 = Base_Model.get_layer('block6h_expand_activation').output  # 7*7*2304
                conv15 = Base_Model.get_layer('block6h_activation').output  # 7*7*2304
                conv16 = Base_Model.get_layer('block6i_expand_activation').output  # 7*7*2304
                conv17 = Base_Model.get_layer('block6i_activation').output  # 7*7*2304
                conv18 = Base_Model.get_layer('block6j_expand_activation').output  # 7*7*2304
                conv19 = Base_Model.get_layer('block6j_activation').output  # 7*7*2304
                conv20 = Base_Model.get_layer('block6k_expand_activation').output  # 7*7*2304
                conv21 = Base_Model.get_layer('block6k_activation').output  # 7*7*2304
                conv22 = Base_Model.get_layer('block6l_expand_activation').output  # 7*7*2304
                conv23 = Base_Model.get_layer('block6l_activation').output  # 7*7*2304
                conv24 = Base_Model.get_layer('block6m_expand_activation').output  # 7*7*2304
                conv25 = Base_Model.get_layer('block6m_activation').output  # 7*7*2304
                conv26 = Base_Model.get_layer('block6n_expand_activation').output  # 7*7*2304
                conv27 = Base_Model.get_layer('block6n_activation').output  # 7*7*2304
                conv28 = Base_Model.get_layer('block6o_expand_activation').output  # 7*7*2304
                conv29 = Base_Model.get_layer('block6o_activation').output  # 7*7*2304
                conv30 = Base_Model.get_layer('block6p_expand_activation').output  # 7*7*2304
                conv31 = Base_Model.get_layer('block6p_activation').output  # 7*7*2304
                conv32 = Base_Model.get_layer('block6q_expand_activation').output  # 7*7*2304
                conv33 = Base_Model.get_layer('block6q_activation').output  # 7*7*2304
                conv34 = Base_Model.get_layer('block6r_expand_activation').output  # 7*7*2304
                conv35 = Base_Model.get_layer('block6r_activation').output  # 7*7*2304
                conv36 = Base_Model.get_layer('block6s_expand_activation').output  # 7*7*2304
                conv37 = Base_Model.get_layer('block6s_activation').output  # 7*7*2304
                conv38 = Base_Model.get_layer('block6t_expand_activation').output  # 7*7*2304
                conv39 = Base_Model.get_layer('block6t_activation').output  # 7*7*2304
                conv40 = Base_Model.get_layer('block6u_expand_activation').output  # 7*7*2304
                conv41 = Base_Model.get_layer('block6u_activation').output  # 7*7*2304
                conv42 = Base_Model.get_layer('block6v_expand_activation').output  # 7*7*2304
                conv43 = Base_Model.get_layer('block6v_activation').output  # 7*7*2304
                conv44 = Base_Model.get_layer('block6w_expand_activation').output  # 7*7*2304
                conv45 = Base_Model.get_layer('block6w_activation').output  # 7*7*2304
                conv46 = Base_Model.get_layer('block6x_expand_activation').output  # 7*7*2304
                conv47 = Base_Model.get_layer('block6x_activation').output  # 7*7*2304
                conv48 = Base_Model.get_layer('block6y_expand_activation').output  # 7*7*2304
                conv49 = Base_Model.get_layer('block6y_activation').output  # 7*7*2304
                conv50 = Base_Model.get_layer('block7a_expand_activation').output  # 7*7*2304
                conv51 = Base_Model.get_layer('block7a_activation').output  # 7*7*2304
                conv1_1 = tf.keras.layers.add(
                    [conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13,
                     conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24,
                     conv25, conv26, conv27, conv28, conv29, conv30, conv31, conv32, conv33, conv34, conv35,
                     conv36, conv37, conv38, conv39, conv40, conv41, conv42, conv43, conv44, conv45, conv46,
                     conv47, conv48, conv49, conv50, conv51])
                #
                conv52 = Base_Model.get_layer('block7b_expand_activation').output  # 7*7*3840
                conv53 = Base_Model.get_layer('block7b_activation').output  # 7*7*3840
                conv54 = Base_Model.get_layer('block7c_expand_activation').output  # 7*7*3840
                conv55 = Base_Model.get_layer('block7c_activation').output  # 7*7*3840
                conv56 = Base_Model.get_layer('block7d_expand_activation').output  # 7*7*3840
                conv57 = Base_Model.get_layer('block7d_activation').output  # 7*7*3840
                conv58 = Base_Model.get_layer('block7e_expand_activation').output  # 7*7*3840
                conv59 = Base_Model.get_layer('block7e_activation').output  # 7*7*3840
                conv60 = Base_Model.get_layer('block7f_expand_activation').output  # 7*7*3840
                conv61 = Base_Model.get_layer('block7f_activation').output  # 7*7*3840
                conv62 = Base_Model.get_layer('block7g_expand_activation').output  # 7*7*3840
                conv63 = Base_Model.get_layer('block7g_activation').output  # 7*7*3840
                conv1_2 = tf.keras.layers.add([conv52, conv53, conv54, conv55, conv56, conv57,
                                               conv58, conv59, conv60, conv61, conv62, conv63])
                #
                conv20 = Base_Model.get_layer('top_activation').output  # 7*7*1280
                encoder_level_6 = tf.keras.layers.concatenate([conv1, conv1_1, conv1_2, conv20], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model

    def CheXNet(self):
        # UNet Variants with DenseNet121 CheXNet Trained Encoder from their GitHub Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        conv = []
        model_name = "DenseNet121(CheXNet)" + "_" + str(self.decoder_name)

        '''Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))

        '''Encoder'''
        if self.train_mode == 'pretrained_encoder':
            Chexnet_Weights = "CheXNet_TF_Weights.h5"
            Base_Model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            predictions = tf.keras.layers.Dense(14, activation='softmax', name='predictions')(Base_Model.output)  # CheXNet originally trained on 14 Classes
            Base_Model = tf.keras.Model(inputs=Base_Model.input, outputs=predictions)
            Base_Model.load_weights(Chexnet_Weights)

            '''Encoder Level 1'''
            layers = Base_Model.layers
            conv = layers[0].output  # 224*224*3
            conv = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3), activation_fun='sigmoid')
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_1 = MultiResBlock(conv, self.model_width * (2 ** 0), (3, 3), self.alpha)
                convs["conv1"] = ResPath(encoder_level_1, self.model_depth, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
            else:
                encoder_level_1 = Conv_Block(conv, self.model_width * (2 ** 0), (3, 3))  # 224*224*32
                convs["conv1"] = encoder_level_1
            '''Encoder Level 2'''
            conv = Base_Model.get_layer('conv1/relu').output  # 112*112*64
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_2 = MultiResBlock(conv, self.model_width * (2 ** 1), (3, 3), self.alpha)  # 112*112*64
                convs["conv2"] = ResPath(encoder_level_2, self.model_depth - 1, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3), activation_fun='sigmoid')  # 112*112*64
                for k in range(1, 2):
                    conv_temp = convs["conv%s" % k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            else:
                encoder_level_2 = Conv_Block(conv, self.model_width * (2 ** 1), (3, 3))  # 112*112*64
                convs["conv2"] = encoder_level_2
            '''Encoder Level 3'''
            conv1 = Base_Model.get_layer('conv2_block1_1_relu').output  # 56*56*128
            conv2 = Base_Model.get_layer('conv2_block2_1_relu').output  # 56*56*128
            conv3 = Base_Model.get_layer('conv2_block3_1_relu').output  # 56*56*128
            conv4 = Base_Model.get_layer('conv2_block4_1_relu').output  # 56*56*128
            conv5 = Base_Model.get_layer('conv2_block5_1_relu').output  # 56*56*128
            conv6 = Base_Model.get_layer('conv2_block6_1_relu').output  # 56*56*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6])
            #
            conv10 = Base_Model.get_layer('pool2_relu').output  # 56*56*256
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv10], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_3 = MultiResBlock(conv, self.model_width * (2 ** 2), (3, 3), self.alpha)  # 56*56*256
                convs["conv3"] = ResPath(encoder_level_3, self.model_depth - 2, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3), activation_fun='sigmoid')  # 56*56*256
                for k in range(1, 3):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            else:
                encoder_level_3 = Conv_Block(conv, self.model_width * (2 ** 2), (3, 3))  # 56*56*256
                convs["conv3"] = encoder_level_3
            '''Encoder Level 4'''
            conv1 = Base_Model.get_layer('conv3_block1_0_relu').output  # 28*28*128
            conv2 = Base_Model.get_layer('conv3_block1_1_relu').output  # 28*28*128
            conv3 = Base_Model.get_layer('conv3_block2_1_relu').output  # 28*28*128
            conv4 = Base_Model.get_layer('conv3_block3_1_relu').output  # 28*28*128
            conv5 = Base_Model.get_layer('conv3_block4_1_relu').output  # 28*28*128
            conv6 = Base_Model.get_layer('conv3_block5_1_relu').output  # 28*28*128
            conv7 = Base_Model.get_layer('conv3_block6_1_relu').output  # 28*28*128
            conv8 = Base_Model.get_layer('conv3_block7_1_relu').output  # 28*28*128
            conv9 = Base_Model.get_layer('conv3_block8_1_relu').output  # 28*28*128
            conv10 = Base_Model.get_layer('conv3_block9_1_relu').output  # 28*28*128
            conv11 = Base_Model.get_layer('conv3_block10_1_relu').output  # 28*28*128
            conv12 = Base_Model.get_layer('conv3_block11_1_relu').output  # 28*28*128
            conv13 = Base_Model.get_layer('conv3_block12_1_relu').output  # 28*28*128
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13])
            #
            conv14 = Base_Model.get_layer('pool3_relu').output  # 28*28*512
            #
            conv = tf.keras.layers.concatenate([conv1_1, conv14], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_4 = MultiResBlock(conv, self.model_width * (2 ** 3), (3, 3), self.alpha)  # 28*28*512
                convs["conv4"] = ResPath(encoder_level_4, self.model_depth - 3, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 4):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            else:
                encoder_level_4 = Conv_Block(conv, self.model_width * (2 ** 3), (3, 3))  # 28*28*512
                convs["conv4"] = encoder_level_4
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block1_1_relu').output  # 14*14*128
            conv2 = Base_Model.get_layer('conv4_block2_1_relu').output  # 14*14*128
            conv3 = Base_Model.get_layer('conv4_block3_1_relu').output  # 14*14*128
            conv4 = Base_Model.get_layer('conv4_block4_1_relu').output  # 14*14*128
            conv5 = Base_Model.get_layer('conv4_block5_1_relu').output  # 14*14*128
            conv6 = Base_Model.get_layer('conv4_block6_1_relu').output  # 14*14*128
            conv7 = Base_Model.get_layer('conv4_block7_1_relu').output  # 14*14*128
            conv8 = Base_Model.get_layer('conv4_block8_1_relu').output  # 14*14*128
            conv9 = Base_Model.get_layer('conv4_block9_1_relu').output  # 14*14*128
            conv10 = Base_Model.get_layer('conv4_block10_1_relu').output  # 14*14*128
            conv11 = Base_Model.get_layer('conv4_block11_1_relu').output  # 14*14*128
            conv12 = Base_Model.get_layer('conv4_block12_1_relu').output  # 14*14*128
            conv13 = Base_Model.get_layer('conv4_block13_1_relu').output  # 14*14*128
            conv14 = Base_Model.get_layer('conv4_block14_1_relu').output  # 14*14*128
            conv15 = Base_Model.get_layer('conv4_block15_1_relu').output  # 14*14*128
            conv16 = Base_Model.get_layer('conv4_block16_1_relu').output  # 14*14*128
            conv17 = Base_Model.get_layer('conv4_block17_1_relu').output  # 14*14*128
            conv18 = Base_Model.get_layer('conv4_block18_1_relu').output  # 14*14*128
            conv19 = Base_Model.get_layer('conv4_block19_1_relu').output  # 14*14*128
            conv20 = Base_Model.get_layer('conv4_block20_1_relu').output  # 14*14*128
            conv21 = Base_Model.get_layer('conv4_block21_1_relu').output  # 14*14*128
            conv22 = Base_Model.get_layer('conv4_block22_1_relu').output  # 14*14*128
            conv23 = Base_Model.get_layer('conv4_block23_1_relu').output  # 14*14*128
            conv24 = Base_Model.get_layer('conv4_block24_1_relu').output  # 14*14*128
            #
            conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
                                           conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24])
            #
            conv25 = Base_Model.get_layer('pool4_relu').output  # 14*14*512
            conv = tf.keras.layers.concatenate([conv1_1, conv25], axis=-1)
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                encoder_level_5 = MultiResBlock(conv, self.model_width * (2 ** 4), (3, 3), self.alpha)  # 14*14*1024
                convs["conv5"] = ResPath(encoder_level_5, self.model_depth - 4, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                conv = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3), activation_fun='sigmoid')  # 28*28*512
                for k in range(1, 5):
                    conv_temp = convs["conv%s" %k]
                    if self.decoder_name == 'AHNet':
                        conv_temp = ResPath(conv_temp, self.model_depth-k, self.model_width*(2 ** 0), (3, 3))
                    conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
                    conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
                    conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 28*28*512
                convs["conv5"] = encoder_level_5
            else:
                encoder_level_5 = Conv_Block(conv, self.model_width * (2 ** 4), (3, 3))  # 14*14*1024
                convs["conv5"] = encoder_level_5

            bottom = []
            if self.model_depth == 1:
                bottom = encoder_level_2
            elif self.model_depth == 2:
                bottom = encoder_level_3
            elif self.model_depth == 3:
                bottom = encoder_level_4
            elif self.model_depth == 4:
                bottom = encoder_level_5
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv5_block1_1_relu').output  # 7*7*128
                conv2 = Base_Model.get_layer('conv5_block2_1_relu').output  # 7*7*128
                conv3 = Base_Model.get_layer('conv5_block3_1_relu').output  # 7*7*128
                conv4 = Base_Model.get_layer('conv5_block4_1_relu').output  # 7*7*128
                conv5 = Base_Model.get_layer('conv5_block5_1_relu').output  # 7*7*128
                conv6 = Base_Model.get_layer('conv5_block6_1_relu').output  # 7*7*128
                conv7 = Base_Model.get_layer('conv5_block7_1_relu').output  # 7*7*128
                conv8 = Base_Model.get_layer('conv5_block8_1_relu').output  # 7*7*128
                conv9 = Base_Model.get_layer('conv5_block9_1_relu').output  # 7*7*128
                conv10 = Base_Model.get_layer('conv5_block10_1_relu').output  # 7*7*128
                conv11 = Base_Model.get_layer('conv5_block11_1_relu').output  # 7*7*128
                conv12 = Base_Model.get_layer('conv5_block12_1_relu').output  # 7*7*128
                conv13 = Base_Model.get_layer('conv5_block13_1_relu').output  # 7*7*128
                conv14 = Base_Model.get_layer('conv5_block14_1_relu').output  # 7*7*128
                conv15 = Base_Model.get_layer('conv5_block15_1_relu').output  # 7*7*128
                conv16 = Base_Model.get_layer('conv5_block16_1_relu').output  # 7*7*128
                conv1_1 = tf.keras.layers.add([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
                                               conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16])
                #
                conv17 = Base_Model.get_layer('relu').output  # 7*7*1024
                #
                encoder_level_6 = tf.keras.layers.concatenate([conv1_1, conv17], axis=-1)
                bottom = encoder_level_6

            conv = bottom

        elif self.train_mode == 'from_scratch':
            pool = inputs
            if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
                for i in range(1, (self.model_depth + 1)):
                    conv = MultiResBlock(pool, self.model_width * (2 ** (i - 1)), (3, 3), self.alpha)
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = ResPath(conv, (self.model_depth - i + 1), self.model_width * (2 ** (i - 1)), (3, 3))
            elif (self.decoder_name == 'UNet4P') or (self.decoder_name == 'UNet4PV2') or (self.decoder_name == 'AHNet'):
                for i in range(1, (self.model_depth + 2)):
                    if i > 1:
                        for k in range(1, i):
                            conv = convs["conv%s" %k]
                            if self.decoder_name == 'AHNet':
                                conv = ResPath(conv, self.model_depth - k, self.model_width * (2 ** 0), (3, 3))
                            conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                            conv = tf.keras.layers.Activation('sigmoid')(conv)
                            pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    convs["conv%s" % i] = conv
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            else:
                for i in range(1, (self.model_depth + 2)):
                    conv = Conv_Block(pool, self.model_width * (2 ** (i - 1)), (3, 3))
                    conv = Conv_Block(conv, self.model_width * (2 ** (i - 1)), (3, 3))
                    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                    convs["conv%s" % i] = conv
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")

        '''Latent Layers'''
        if (self.decoder_name == 'MultiResUNet') or (self.decoder_name == 'MultiResUNet3P'):
            conv = MultiResBlock(conv, self.model_width * (2 ** self.model_depth), (3, 3), self.alpha)  # 7*7*2048
        else:
            conv = dense_block(conv, self.model_width * (2 ** self.model_depth), (3, 3), 6 - self.model_depth)  # 7*7*2048
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))
            # conv = Conv_Block(conv, self.model_width * (2 ** self.model_depth), (3, 3))

        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv

        '''Decoder'''
        deconv = []
        levels = []
        if self.decoder_name == 'UNet':
            deconv, levels = UNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetE':
            deconv, levels = UNetE(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetP':
            deconv, levels = UNetP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'UNetPP':
            deconv, levels = UNetPP(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif (self.decoder_name == 'UNet3P') or (self.decoder_name == 'UNet4PV2'):
            deconv, levels = UNet3P(convs, self.model_width, self.model_depth, self.D_S)
        elif self.decoder_name == 'UNet4P':
            deconv, levels = UNet4P(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv)
        elif self.decoder_name == 'MultiResUNet':
            deconv, levels = MultiResUNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3), self.alpha)
        elif self.decoder_name == 'MultiResUNet3P':
            deconv, levels = MultiResUNet3P(convs, self.model_width, self.model_depth, self.D_S, (3, 3), self.alpha)
        elif self.decoder_name == 'AHNet':
            deconv, levels = AHNet(convs, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, (3, 3))

        '''Output'''
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)

        return model


if __name__ == '__main__':
    # Configurations
    length = 224  # Length of each Image
    width = 224  # Width of each Image
    decoder_name = 'UNet'  # Decoder Architecture (UNet, UNetPP, etc.)
    model_width = 16  # Width of the Initial Layer, subsequent layers start from here
    model_depth = 5  # Depth or Number of Layers in the Model (Maximum 5, Minimum 1)
    D_S = 0  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 0  # Turn on for Guided Attention
    LSTM = 0  # Turn on for LSTM
    num_dense_loop = 2  # Number of Dense Blocks in the BottleNeck Layer
    problem_type = 'Regression'  # Regression or Classification Segmentation Tasks
    output_nums = 1  # Number of Classes for Classification Problems, always '1' for Regression Problems
    is_transconv = True  # True: Transposed Convolution, False: UpSampling
    train_mode = 'pretrained_encoder'  # Training Modes: 'pretrained_encoder' or 'from_scratch'
    base_model_trainable = False  # Whether Base Model is trainable or not. True: Fine Tuning Mode, False: Freeze or Inference only Mode
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    #
    Model = UNetWithPretrainedEncoder(decoder_name, length, width, model_width, model_depth, problem_type=problem_type, output_nums=output_nums,
                                      ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv, train_mode=train_mode,
                                      is_base_model_trainable=base_model_trainable).EfficientNetB0()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
