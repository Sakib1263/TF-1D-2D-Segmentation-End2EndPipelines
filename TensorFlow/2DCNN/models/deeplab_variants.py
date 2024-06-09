# Import Necessary Libraries
import tensorflow as tf
import numpy as np
from models.onn_layers import Oper2D, Oper2DTranspose


def Conv_Block(inputs, model_width, kernel, padding='same', bn=True, activation_fun='ReLU', kernel_initialization_method='he_uniform'):
    # 2D Convolutional Block
    x = tf.keras.layers.Conv2D(model_width, kernel, padding=padding, kernel_initializer=kernel_initialization_method)(inputs)
    if bn == True:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation_fun is not None:
        x = tf.keras.layers.Activation(activation_fun)(x)
    return x


def trans_conv2D(inputs, model_width, kernel=(4, 4), bn=False, strides=(2, 2), padding='same', activation_fun='LeakyReLU'):
    # 2D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv2DTranspose(model_width, kernel, strides=strides, padding=padding)(inputs)  # Stride = 2, Kernel Size = 2
    if bn == True:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation_fun is not None:
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


def operational_dense_block(x, num_filters, kernel_size, num_layers, q):
    x = Oper2D(num_filters, kernel_size, q=q)(x)
    for _ in range(num_layers):
        cb = Oper2D(num_filters, kernel_size, q=q)(x)
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
    deconv = convs_list[-1]
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


def KSSNet(skip_connections, model_width, model_depth, D_S, A_G, LSTM, is_transconv, kernel, alpha=1.0):
    # KSSNet based Decoder
    levels = []
    deconvs = {}
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
        for m in range(0, j+1):
            if m == 0:
                temp_deconv = mresblocks_list[-1]
            else:
                temp_deconv = deconvs["deconv%s" % m]
            temp_deconv = upConv_Block(temp_deconv, size=(2 ** (j - m + 1), 2 ** (j - m + 1)))
            temp_deconv = tf.keras.layers.Activation('sigmoid')(temp_deconv)
            deconv = tf.keras.layers.concatenate([deconv, temp_deconv], axis=-1)
        deconv = MultiResBlock(deconv, model_width * (2 ** (model_depth - j - 1)), kernel, alpha)
        deconvs["deconv%s" % (j+1)] = deconv
    return deconv, levels


def SelfUNet(skip_connections, model_width, model_depth, D_S, is_transconv, q):
    # UNet based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    deconv = convs_list[-1]
    for j in range(0, model_depth):
        skip_connection = convs_list[model_depth - j - 1]
        if D_S == 1:
            # For Deep Supervision
            level = Oper2D(1, (1, 1), q=q)(deconv)
            levels.append(level)
        if is_transconv:
            deconv = Oper2DTranspose(model_width * (2 ** (model_depth - j - 1)), (4, 4), strides=(2, 2), padding='same', activation='tanh', q=q)(deconv)
        elif not is_transconv:
            deconv = upConv_Block(deconv)
        deconv = Concat_Block(deconv, skip_connection)
        deconv = Oper2D(model_width * (2 ** (model_depth - j - 1)), (3, 3), q=q)(deconv)
        # deconv = Oper2D(model_width * (2 ** (model_depth - j - 1)), (3, 3), q=q)(deconv)
        deconv = tf.keras.layers.BatchNormalization(name=f'bn_layer_{j}')(deconv)
        deconv = tf.keras.layers.Activation('tanh', name=f'activ_func_{j}')(deconv)
    return deconv, levels


def SelfUNetPP(skip_connections, model_width, model_depth, D_S, is_transconv, q):
    # UNet++ based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    if D_S == 1:
        level = Oper2D(1, (1, 1), q=q)(convs_list[0])
        levels.append(level)
    deconv = []
    deconvs = {}
    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if i == 1:
                skip_connection = convs_list[j]
                if is_transconv:
                    deconv = Oper2DTranspose(model_width * (2 ** j), (4, 4), strides=(2, 2), padding='same', activation='tanh', q=q)(convs_list[j + 1])
                elif not is_transconv:
                    deconv = upConv_Block(convs_list[j + 1])
                deconv = Concat_Block(deconv, skip_connection)
                deconv = Oper2D(model_width * (2 ** j), (3, 3), q=q)(deconv)
                # deconv = Oper2D(model_width * (2 ** j), (3, 3), q=q)(deconv)
                deconv = tf.keras.layers.BatchNormalization(name=f'bn_layer_{i}_{j}')(deconv)
                deconv = tf.keras.layers.Activation('tanh', name=f'activ_func_{i}_{j}')(deconv)
                deconvs["deconv%s%s" % (j, i)] = deconv
            elif i > 1:
                deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                for k in range(2, i):
                    deconv_temp = deconvs["deconv%s%s" % (j, k)]
                    deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                skip_connection = convs_list[j]
                if is_transconv:
                    deconv = Oper2DTranspose(model_width * (2 ** j), (4, 4), strides=(2, 2), padding='same', activation='tanh', q=q)(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                elif not is_transconv:
                    deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                deconv = Oper2D(model_width * (2 ** j), (3, 3), q=q)(deconv)
                # deconv = Oper2D(model_width * (2 ** j), (3, 3), q=q)(deconv)
                deconv = tf.keras.layers.BatchNormalization(name=f'bn_layer_{i}_{j}')(deconv)
                deconv = tf.keras.layers.Activation('tanh', name=f'activ_func_{i}_{j}')(deconv)
                deconvs["deconv%s%s" % (j, i)] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = Oper2D(1, (1, 1), q=q)(deconvs["deconv%s%s" % (j, i)])
                levels.append(level)
    deconv = deconvs["deconv%s%s" % (0, model_depth)]
    return deconv, levels


def SelfUNet3P(skip_connections, model_width, model_depth, D_S, q):
    # UNet3+ based Decoder
    levels = []
    convs_list = list(skip_connections.values())
    deconv = convs_list[-1]
    deconvs = {}
    for j in range(0, model_depth):
        skip_connections_all = convs_list[model_depth - j - 1]
        skip_connections_all = Oper2D(model_width * (2 ** 0), (3, 3), q=q)(skip_connections_all)
        skip_connections_all = tf.keras.layers.BatchNormalization(name=f'bn_layer_{j}')(skip_connections_all)
        skip_connections_all = tf.keras.layers.Activation('tanh', name=f'activ_func_{j}')(skip_connections_all)
        for k in range(0, model_depth - j - 1):
            skip_connection = convs_list[k]
            skip_connection = tf.keras.layers.MaxPooling2D(pool_size=((2 ** ((model_depth - j) - k - 1)), (2 ** ((model_depth - j) - k - 1))))(skip_connection)
            skip_connection = Oper2D(model_width * (2 ** 0), (3, 3), q=q)(skip_connection)
            skip_connection = tf.keras.layers.BatchNormalization(name=f'bn_layer_{j}_{k}')(skip_connection)
            skip_connection = tf.keras.layers.Activation('tanh', name=f'activ_func_{j}_{k}')(skip_connection)
            skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
        deconv_tot = Oper2D(model_width * (2 ** 0), (3, 3), q=q)(deconv)
        deconv_tot = upConv_Block(deconv_tot, size=(2 ** 1, 2 ** 1), interpolation_mode='bilinear')
        deconv_tot = tf.keras.layers.Activation('tanh')(deconv_tot)
        deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
        if j > 0:
            for m in range(0, j):
                deconv = Oper2D(model_width * (2 ** 0), (3, 3), q=q)(deconvs["deconv%s" % m])
                deconv = upConv_Block(deconv, size=((2 ** (j - m)), (2 ** (j - m))), interpolation_mode='bilinear')
                deconv = tf.keras.layers.Activation('tanh')(deconv)
                deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
        deconv = Oper2D(model_width * (model_depth + 1), (3, 3), q=q)(deconv_tot)
        deconvs["deconv%s" % j] = deconv
        if D_S == 1:
            # For Deep Supervision
            level = Oper2D(1, (1, 1), strides=(2, 2), q=q)(deconv)
            levels.append(level)
    return deconv, levels


def encoder_block_scratch(inputs, decoder_name, model_width, model_depth, alpha, q):
    convs = {}
    pool = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P'):
        for i in range(1, (model_depth + 2)):
            conv = MultiResBlock(pool, model_width * (2 ** (i - 1)), (3, 3), alpha)
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            convs["conv%s" % i] = ResPath(conv, (model_depth - i + 1), model_width * (2 ** (i - 1)), (3, 3))
    elif decoder_name == 'KSSNet':
        for i in range(1, (model_depth + 2)):
            if i > 1:
                for k in range(1, i):
                    conv = convs["conv%s" %k]
                    conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                    conv = tf.keras.layers.Activation('sigmoid')(conv)
                    pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
            conv = MultiResBlock(pool, model_width * (2 ** (i - 1)), (3, 3), alpha)
            convs["conv%s" % i] = ResPath(conv, (model_depth - i + 1), model_width * (2 ** (i - 1)), (3, 3))
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    elif (decoder_name == 'UNet4P') or (decoder_name == 'UNet4PV2') or (decoder_name == 'AHNet'):
        for i in range(1, (model_depth + 2)):
            if i > 1:
                for k in range(1, i):
                    conv = convs["conv%s" %k]
                    if decoder_name == 'AHNet':
                        conv = ResPath(conv, model_depth - k, model_width * (2 ** 0), (3, 3))
                    conv = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (i - k)), (2 ** (i - k))))(conv)
                    conv = tf.keras.layers.Activation('sigmoid')(conv)
                    pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
            conv = Conv_Block(pool, model_width * (2 ** (i - 1)), (3, 3))
            convs["conv%s" % i] = conv
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    elif (decoder_name[0:4] == 'Self'):
        for i in range(1, (model_depth + 2)):
            conv = Oper2D(model_width * (2 ** (i - 1)), (3, 3), q=q)(pool)
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            convs["conv%s" % i] = conv
    else:
        for i in range(1, (model_depth + 2)):
            conv = Conv_Block(pool, model_width * (2 ** (i - 1)), (3, 3))
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            convs["conv%s" % i] = conv
    return convs, conv


def encoder_block_pretrained_level_1(inputs, decoder_name, model_width, model_depth, alpha, q):
    conv = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P') or (decoder_name == 'KSSNet'):
        conv = MultiResBlock(conv, model_width * (2 ** 0), (3, 3), alpha)
        conv = ResPath(conv, model_depth, model_width * (2 ** 0), (3, 3))
    elif (decoder_name[0:4] == 'Self'):
        conv = Oper2D(model_width * (2 ** 0), (3, 3), q=q)(conv)
    else:
        conv = Conv_Block(conv, model_width * (2 ** 0), (3, 3), bn=False, activation_fun=None)
    return conv


def encoder_block_pretrained_level_2(inputs, decoder_name, model_width, model_depth, alpha, q):
    convs = {}
    conv = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P'):
        conv = MultiResBlock(conv, model_width * (2 ** 1), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 1, model_width * (2 ** 1), (3, 3))
    elif (decoder_name == 'KSSNet'):
        conv = Conv_Block(conv, model_width * (2 ** 1), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 2):
            conv_temp = convs["conv%s" %k]
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = MultiResBlock(conv, model_width * (2 ** 1), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 1, model_width * (2 ** 1), (3, 3))
    elif (decoder_name == 'UNet4P') or (decoder_name == 'UNet4PV2') or (decoder_name == 'AHNet'):
        conv = Conv_Block(conv, model_width * (2 ** 1), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 2):
            conv_temp = convs["conv%s" %k]
            if decoder_name == 'AHNet':
                conv_temp = ResPath(conv_temp, model_depth-k, model_width*(2 ** 1), (3, 3))
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (2 - k)), (2 ** (2 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = Conv_Block(conv, model_width * (2 ** 1), (3, 3))
    elif (decoder_name[0:4] == 'Self'):
        conv = Oper2D(model_width * (2 ** 1), (1, 1), q=q)(conv)
    else:
        conv = Conv_Block(conv, model_width * (2 ** 1), (1, 1), bn=False, activation_fun=None)
    return conv


def encoder_block_pretrained_level_3(inputs, decoder_name, model_width, model_depth, alpha, q):
    convs = {}
    conv = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P'):
        conv = MultiResBlock(conv, model_width * (2 ** 2), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 2, model_width * (2 ** 2), (3, 3))
    elif (decoder_name == 'KSSNet'):
        conv = Conv_Block(conv, model_width * (2 ** 2), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 3):
            conv_temp = convs["conv%s" %k]
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = MultiResBlock(conv, model_width * (2 ** 2), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 1, model_width * (2 ** 2), (3, 3))
    elif (decoder_name == 'UNet4P') or (decoder_name == 'UNet4PV2') or (decoder_name == 'AHNet'):
        conv = Conv_Block(conv, model_width * (2 ** 2), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 3):
            conv_temp = convs["conv%s" %k]
            if decoder_name == 'AHNet':
                conv_temp = ResPath(conv_temp, model_depth-k, model_width*(2 ** 2), (3, 3))
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (3 - k)), (2 ** (3 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = Conv_Block(conv, model_width * (2 ** 2), (3, 3))
    elif (decoder_name[0:4] == 'Self'):
        conv = Oper2D(model_width * (2 ** 2), (1, 1), q=q)(conv)
    else:
        conv = Conv_Block(conv, model_width * (2 ** 2), (1, 1), bn=False, activation_fun=None)
    return conv


def encoder_block_pretrained_level_4(inputs, decoder_name, model_width, model_depth, alpha, q):
    convs = {}
    conv = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P'):
        conv = MultiResBlock(conv, model_width * (2 ** 3), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 3, model_width * (2 ** 3), (3, 3))
    elif (decoder_name == 'KSSNet'):
        conv = Conv_Block(conv, model_width * (2 ** 3), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 4):
            conv_temp = convs["conv%s" %k]
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = MultiResBlock(conv, model_width * (2 ** 3), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 1, model_width * (2 ** 3), (3, 3))
    elif (decoder_name == 'UNet4P') or (decoder_name == 'UNet4PV2') or (decoder_name == 'AHNet'):
        conv = Conv_Block(conv, model_width * (2 ** 3), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 4):
            conv_temp = convs["conv%s" %k]
            if decoder_name == 'AHNet':
                conv_temp = ResPath(conv_temp, model_depth-k, model_width*(2 ** 3), (3, 3))
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (4 - k)), (2 ** (4 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = Conv_Block(conv, model_width * (2 ** 3), (3, 3))
    elif (decoder_name[0:4] == 'Self'):
        conv = Oper2D(model_width * (2 ** 3), (1, 1), q=q)(conv)
    else:
        conv = Conv_Block(conv, model_width * (2 ** 3), (1, 1), bn=False, activation_fun=None)
    return conv


def encoder_block_pretrained_level_5(inputs, decoder_name, model_width, model_depth, alpha, q):
    convs = {}
    conv = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P'):
        conv = MultiResBlock(conv, model_width * (2 ** 4), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 4, model_width * (2 ** 4), (3, 3))
    elif (decoder_name == 'KSSNet'):
        conv = Conv_Block(conv, model_width * (2 ** 4), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 5):
            conv_temp = convs["conv%s" %k]
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = MultiResBlock(conv, model_width * (2 ** 4), (3, 3), alpha)
        conv = ResPath(conv, model_depth - 1, model_width * (2 ** 4), (3, 3))
    elif (decoder_name == 'UNet4P') or (decoder_name == 'UNet4PV2') or (decoder_name == 'AHNet'):
        conv = Conv_Block(conv, model_width * (2 ** 4), (1, 1), bn=False, activation_fun=None)
        for k in range(1, 5):
            conv_temp = convs["conv%s" %k]
            if decoder_name == 'AHNet':
                conv_temp = ResPath(conv_temp, model_depth-k, model_width*(2 ** 4), (3, 3))
            conv_temp = tf.keras.layers.MaxPooling2D(pool_size=((2 ** (5 - k)), (2 ** (5 - k))))(conv_temp)
            conv_temp = tf.keras.layers.Activation('sigmoid')(conv_temp)
            conv = tf.keras.layers.concatenate([conv, conv_temp], axis=-1)
        conv = Conv_Block(conv, model_width * (2 ** 4), (3, 3))
    elif (decoder_name[0:4] == 'Self'):
        conv = Oper2D(model_width * (2 ** 4), (1, 1), q=q)(conv)
    else:
        conv = Conv_Block(conv, model_width * (2 ** 4), (1, 1), bn=False, activation_fun=None)
    return conv


def decoder_block(convs, decoder_name, model_width, model_depth, D_S, A_G, LSTM, is_transconv, alpha, q):
    if decoder_name == 'UNet':
        deconv, levels = UNet(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv)
    elif decoder_name == 'UNetE':
        deconv, levels = UNetE(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv)
    elif decoder_name == 'UNetP':
        deconv, levels = UNetP(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv)
    elif decoder_name == 'UNetPP':
        deconv, levels = UNetPP(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv)
    elif (decoder_name == 'UNet3P') or (decoder_name == 'UNet4PV2'):
        deconv, levels = UNet3P(convs, model_width, model_depth, D_S)
    elif decoder_name == 'UNet4P':
        deconv, levels = UNet4P(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv)
    elif decoder_name == 'MultiResUNet':
        deconv, levels = MultiResUNet(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv, (3, 3), alpha)
    elif decoder_name == 'MultiResUNet3P':
        deconv, levels = MultiResUNet3P(convs, model_width, model_depth, D_S, (3, 3), alpha)
    elif decoder_name == 'KSSNet':
        deconv, levels = KSSNet(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv, (3, 3), alpha)
    elif decoder_name == 'AHNet':
        deconv, levels = AHNet(convs, model_width, model_depth, D_S, A_G, LSTM, is_transconv, (3, 3))
    elif decoder_name == 'SelfUNet':
        deconv, levels = SelfUNet(convs, model_width, model_depth, D_S, is_transconv, q)
    elif decoder_name == 'SelfUNetPP':
        deconv, levels = SelfUNetPP(convs, model_width, model_depth, D_S, is_transconv, q)
    elif decoder_name == 'SelfUNet3P':
        deconv, levels = SelfUNet3P(convs, model_width, model_depth, D_S, q)
    return deconv, levels


def latent_layer(inputs, decoder_name, model_width, model_depth, alpha, q, dense_loop):
    conv = inputs
    if (decoder_name == 'MultiResUNet') or (decoder_name == 'MultiResUNet3P') or (decoder_name == 'KSSNet'):
        conv = MultiResBlock(conv, model_width * (2 ** model_depth), (3, 3), alpha)
    elif (decoder_name[0:4] == 'Self'):
        conv = operational_dense_block(conv, model_width * (2 ** model_depth), (3, 3), dense_loop, q)
    else:
        conv = dense_block(conv, model_width * (2 ** model_depth), (3, 3), dense_loop)
    return conv


class unet_model_builder:
    def __init__(self, 
                 decoder_name, 
                 length, 
                 width, 
                 model_width, 
                 model_depth, 
                 num_channels=3, 
                 output_nums=1, 
                 ds=0, 
                 ae=0, 
                 ag=0, 
                 lstm=0, 
                 dense_loop=1, 
                 feature_number=1024, 
                 is_transconv=True, 
                 alpha=1.0, 
                 q=3, 
                 final_activation="sigmoid", 
                 train_mode='pretrained_encoder', 
                 is_base_model_trainable=False
                 ):
        # decoder_name: Name of the decoder model e.g., UNet, UNet++, etc.
        # length: Input Image Length (x-dim)
        # width: Input Image Width (y-dim) [Normally same as the x-dim i.e., Square shape]
        # model_width: Width of the Input Layer of the Model
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
        # q: q-order for ONNs
        self.decoder_name = decoder_name
        self.length = length
        self.width = width
        self.model_depth = model_depth
        self.model_width = model_width
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
        self.q = q
        #
        if self.train_mode == 'pretrained_encoder':
            if (self.model_depth > 5) or (self.model_depth < 1):
                raise ValueError('The depth of a TF-ImageNet Pretrained model can only be discretely varied from 1 to 5')
        elif self.train_mode == 'from_scratch':
            if self.model_depth < 1:
                raise ValueError('The depth of the model cannot be less than 1')
        else:
            raise ValueError('The Train Mode can only be: "pretrained_encoder" or "from_scratch"')
        
    def ResNet50(self):
        # UNet Variants with ResNet50 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "ResNet50" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1_relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('conv2_block3_out').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('conv3_block4_out').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('conv4_block6_out').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('conv5_block3_out').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
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
        model_name = "ResNet50V2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1_conv').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1_1 = Base_Model.get_layer('conv2_block3_1_relu').output
            conv1_2 = Base_Model.get_layer('conv2_block3_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1_1 = Base_Model.get_layer('conv3_block4_1_relu').output
            conv1_2 = Base_Model.get_layer('conv3_block4_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv1_1 = Base_Model.get_layer('conv4_block6_1_relu').output
            conv1_2 = Base_Model.get_layer('conv4_block6_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1_1, conv1_2], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('post_relu').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels, name=model_name)
        return model

    def ResNet101(self):
        # UNet Variants with ResNet101 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        model_name = "ResNet101" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1_relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('conv2_block3_out').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('conv3_block4_out').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('conv4_block23_out').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('conv5_block3_out').output
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
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
        model_name = "ResNet101V2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1_conv').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1 = Base_Model.get_layer('conv2_block3_1_relu').output
            conv2 = Base_Model.get_layer('conv2_block3_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1 = Base_Model.get_layer('conv2_block3_2_relu').output
            conv2 = Base_Model.get_layer('conv3_block4_1_relu').output
            conv3 = Base_Model.get_layer('conv3_block4_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2, conv3], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv1 = Base_Model.get_layer('conv3_block4_2_relu').output
            conv2 = Base_Model.get_layer('conv4_block1_preact_relu').output
            conv3 = Base_Model.get_layer('conv4_block23_1_relu').output
            conv4 = Base_Model.get_layer('conv4_block23_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv4_block23_2_relu').output
                conv2 = Base_Model.get_layer('conv5_block1_preact_relu').output
                conv3 = Base_Model.get_layer('conv5_block3_2_relu').output
                conv4 = Base_Model.get_layer('post_relu').output
                bottom = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4], axis=-1)  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
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
        model_name = "ResNet152" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1_relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('conv2_block3_out').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('conv3_block8_out').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('conv4_block36_out').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('conv5_block1_out').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
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
        model_name = "ResNet152V2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1_conv').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1 = Base_Model.get_layer('conv2_block3_1_relu').output
            conv2 = Base_Model.get_layer('conv2_block3_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1 = Base_Model.get_layer('conv2_block3_2_relu').output
            conv2 = Base_Model.get_layer('conv3_block8_1_relu').output
            conv3 = Base_Model.get_layer('conv3_block8_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2, conv3], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv1 = Base_Model.get_layer('conv3_block8_2_relu').output
            conv2 = Base_Model.get_layer('conv4_block1_preact_relu').output
            conv3 = Base_Model.get_layer('conv4_block36_1_relu').output
            conv4 = Base_Model.get_layer('conv4_block36_preact_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv4_block36_2_relu').output
                conv2 = Base_Model.get_layer('conv5_block1_preact_relu').output
                conv3 = Base_Model.get_layer('conv5_block3_2_relu').output
                conv4 = Base_Model.get_layer('post_relu').output
                bottom = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4], axis=-1)  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
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
        model_name = "VGG16" + "_" + str(self.decoder_name)
        # Input'''
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2_conv2').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3_conv3').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4_conv3').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block5_conv3').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('block5_pool').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
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
        model_name = "VGG19" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2_conv2').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3_conv4').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4_conv4').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block5_conv4').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('block5_pool').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def DenseNet121(self):
        # UNet Variants with DenseNet121 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "DenseNet121" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels), name='input')
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1/relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1 = Base_Model.get_layer('conv2_block6_1_relu').output
            conv2 = Base_Model.get_layer('pool2_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1 = Base_Model.get_layer('conv3_block12_1_relu').output
            conv2 = Base_Model.get_layer('pool3_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv1 = Base_Model.get_layer('conv4_block24_1_relu').output
            conv2 = Base_Model.get_layer('pool4_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('relu').output
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def DenseNet169(self):
        # UNet Variants with DenseNet169 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "DenseNet169" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1/relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1 = Base_Model.get_layer('conv2_block6_1_relu').output
            conv2 = Base_Model.get_layer('pool2_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1 = Base_Model.get_layer('conv3_block12_1_relu').output
            conv2 = Base_Model.get_layer('pool3_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            '''Encoder Level 5'''
            conv1 = Base_Model.get_layer('conv4_block32_1_relu').output
            conv2 = Base_Model.get_layer('pool4_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('relu').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def DenseNet201(self):
        # UNet Variants with DenseNet201 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "DenseNet201" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels), name='input')
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1/relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1 = Base_Model.get_layer('conv2_block6_1_relu').output
            conv2 = Base_Model.get_layer('pool2_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1 = Base_Model.get_layer('conv3_block12_1_relu').output
            conv2 = Base_Model.get_layer('pool3_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv1 = Base_Model.get_layer('conv4_block48_1_relu').output
            conv2 = Base_Model.get_layer('pool4_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Bottleneck
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                bottom = Base_Model.get_layer('relu').output  # (w/32)*(h/32)*(k*32)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def MobileNet(self):
        # UNet Variants with MobileNet ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "MobileNet" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv1 = Base_Model.get_layer('conv_dw_1_relu').output
            conv2 = Base_Model.get_layer('conv_pw_1_relu').output  
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv1 = Base_Model.get_layer('conv_dw_2_relu').output
            conv2 = Base_Model.get_layer('conv_pw_3_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv1 = Base_Model.get_layer('conv_dw_4_relu').output
            conv2 = Base_Model.get_layer('conv_pw_5_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            '''Encoder Level5'''
            conv1 = Base_Model.get_layer('conv_dw_6_relu').output
            conv2 = Base_Model.get_layer('conv_pw_11_relu').output
            conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv1 = Base_Model.get_layer('conv_dw_12_relu').output
                conv2 = Base_Model.get_layer('conv_pw_13_relu').output
                conv = tf.keras.layers.concatenate([conv1, conv2], axis=-1)  # (w/32)*(h/32)*(k*32)
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def MobileNetV2(self):
        # UNet Variants with MobileNetV2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "MobileNetV2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block_1_expand_relu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block_3_expand_relu').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block_6_expand_relu').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level5
            conv = Base_Model.get_layer('block_13_expand_relu').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('block_16_depthwise_relu').output  # (w/32)*(h/32)*(k*32)
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def MobileNetV3Small(self):
        # UNet Variants with MobileNetV3_Small ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "MobileNetV3Small" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.MobileNetV3Small(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('re_lu').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('re_lu_3').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('re_lu_7').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level5
            conv = Base_Model.get_layer('re_lu_22').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('re_lu_31').output  # (w/32)*(h/32)*(k*32)
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def MobileNetV3Large(self):
        # UNet Variants with MobileNetV3_Large ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "MobileNetV3Large" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('re_lu_2').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('re_lu_6').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('re_lu_15').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level5
            conv = Base_Model.get_layer('re_lu_29').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('re_lu_38').output  # (w/32)*(h/32)*(k*32)
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def InceptionV3(self):
        # UNet Variants with Inception V3 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "InceptionV3" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('activation_2').output
            conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('activation_4').output
            conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('activation_28').output
            conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level5
            conv = Base_Model.get_layer('activation_67').output
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('activation_92').output  # (w/32)*(h/32)*(k*32)
                conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def InceptionResNetV2(self):
        # UNet Variants with InceptionResNetV2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "InceptionResNetV2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('activation_2').output
            conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('activation_4').output
            conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('activation_74').output
            conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('activation_161').output
            conv = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('conv_7b_ac').output  # (w/32)*(h/32)*(k*32)
                bottom = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(conv)
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB0(self):
        # UNet Variants with EfficientNetB0 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB0" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels), name='input')
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output  # (w)*(h)*(k)
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output  # (w/2)*(h/2)*(k*2)
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output  # (w/4)*(h/4)*(k*4)
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output  # (w/8)*(h/8)*(k*8)
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output  # (w/16)*(h/16)*(k*16)
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output  # (w/32)*(h/32)*(k*32)
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB1(self):
        # UNet Variants with EfficientNetB1 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB1" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            '''Encoder Level 3'''
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            '''Encoder Level 4'''
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            '''Encoder Level 5'''
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB2(self):
        # UNet Variants with EfficientNetB2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB3(self):
        # UNet Variants with EfficientNetB3 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB3" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB4(self):
        # UNet Variants with EfficientNetB4 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB4" + "_" + str(self.decoder_name)
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB5(self):
        # UNet Variants with EfficientNetB5 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB5" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB6(self):
        # UNet Variants with EfficientNetB6 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB6" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetB7(self):
        # UNet Variants with EfficientNetB7 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetB7" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block2a_expand_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block3a_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2B0(self):
        # UNet Variants with EfficientNetV2B0 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2B0" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1a_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2b_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv3 = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2B1(self):
        # UNet Variants with EfficientNetV2B1 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2B1" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1b_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2c_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2B2(self):
        # UNet Variants with EfficientNetV2B2 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2B2" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1b_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2c_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2B3(self):
        # UNet Variants with EfficientNetV2B3 ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2B3" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1b_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2c_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2S(self):
        # UNet Variants with EfficientNetV2S (Small) ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2S" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1b_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2d_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2M(self):
        # UNet Variants with EfficientNetV2M (Medium) ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2M" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1c_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2e_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def EfficientNetV2L(self):
        # UNet Variants with EfficientNetV2L (Large) ImageNet Trained Encoder from TF Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "EfficientNetV2L" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Base_Model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False, weights='imagenet', input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('block1d_project_activation').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('block2g_expand_activation').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('block4a_expand_activation').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('block6a_expand_activation').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('top_activation').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model

    def CheXNet(self):
        # UNet Variants with DenseNet121 based CheXNet Trained Encoder from their GitHub Repository
        if self.length == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")
        convs = {}
        model_name = "DenseNet121(CheXNet)" + "_" + str(self.decoder_name)
        # Input
        if self.train_mode == 'pretrained_encoder':
            self.num_channels = 3
        inputs = tf.keras.Input((self.length, self.width, self.num_channels))
        # Encoder
        if self.train_mode == 'pretrained_encoder':
            Chexnet_Weights = "CheXNet_TF_Weights.h5"
            Base_Model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=inputs)
            Base_Model.trainable = self.is_base_model_trainable  # True: Fully Trainable (Except non-trainable Params); False: Trainable only in the Top Dense Layer
            predictions = tf.keras.layers.Dense(14, activation='softmax', name='predictions')(Base_Model.output)  # CheXNet originally trained on 14 Classes
            Base_Model = tf.keras.Model(inputs=Base_Model.input, outputs=predictions)
            Base_Model.load_weights(Chexnet_Weights)
            # Encoder Level 1
            layers = Base_Model.layers
            conv = layers[0].output
            convs["conv1"] = encoder_block_pretrained_level_1(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 2
            conv = Base_Model.get_layer('conv1/relu').output
            convs["conv2"] = encoder_block_pretrained_level_2(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 3
            conv = Base_Model.get_layer('pool2_relu').output
            convs["conv3"] = encoder_block_pretrained_level_3(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 4
            conv = Base_Model.get_layer('pool3_relu').output
            convs["conv4"] = encoder_block_pretrained_level_4(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            # Encoder Level 5
            conv = Base_Model.get_layer('pool4_relu').output
            convs["conv5"] = encoder_block_pretrained_level_5(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
            bottom = []
            if self.model_depth == 1:
                bottom = convs["conv2"]
            elif self.model_depth == 2:
                bottom = convs["conv3"]
            elif self.model_depth == 3:
                bottom = convs["conv4"]
            elif self.model_depth == 4:
                bottom = convs["conv5"]
            elif self.model_depth == 5:
                conv = Base_Model.get_layer('relu').output
                bottom = conv
            conv = bottom
        # Encoder from Scratch
        elif self.train_mode == 'from_scratch':
            pool = inputs
            convs, conv = encoder_block_scratch(pool, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q)
        else:
            raise ValueError("Please Select a Correct Training Mode (Check Spelling First)!")
        # Latent Layers
        conv = latent_layer(conv, self.decoder_name, self.model_width, self.model_depth, self.alpha, self.q, self.dense_loop)
        # Feature Extractor for the AutoEncoder Mode
        if self.A_E == 1:
            conv = Feature_Extraction_Block(conv, self.model_width * (2 ** self.model_depth), self.feature_number)
        # Decoder
        convs = dict(list(convs.items())[:self.model_depth])
        convs["conv%s" % (self.model_depth + 1)] = conv
        deconv = []
        levels = []
        deconv, levels = decoder_block(convs, self.decoder_name, self.model_width, self.model_depth, self.D_S, self.A_G, self.LSTM, self.is_transconv, self.alpha, self.q)
        # Output
        outputs = tf.keras.layers.Conv2D(self.output_nums, (1, 1), activation=self.final_activation, name="out")(deconv)
        if (self.decoder_name[0:4] == 'Self'):
            outputs = Oper2D(self.output_nums, (1, 1), activation=self.final_activation, q=self.q)(deconv)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)
        # Output with Deep Supervision
        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)
        return model


if __name__ == '__main__':
    # Configurations
    length = 224  # Length of each Image
    width = 224  # Width of each Image
    decoder_name = 'SelfUNetPP'  # Decoder Architecture (UNet, UNetPP, SelfUNet, etc.)
    model_width = 16  # Width of the Initial Layer, subsequent layers start from here
    model_depth = 5  # Depth or Number of Layers in the Model (Maximum 5, Minimum 1)
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 1  # Turn on for Guided Attention
    LSTM = 1  # Turn on for LSTM
    num_dense_loop = 2  # Number of Dense Blocks in the BottleNeck Layer
    output_nums = 1  # Number of Classes for Classification Problems, always '1' for Regression Problems
    is_transconv = True  # True: Transposed Convolution, False: UpSampling
    train_mode = 'pretrained_encoder'  # Training Modes: 'pretrained_encoder' or 'from_scratch'
    base_model_trainable = False  # Whether Base Model is trainable or not. True: Fine Tuning Mode, False: Freeze or Inference only Mode
    feature_number = 1024  # Number of Features to be Extracted
    q = 3  # q-order for Self-ONNs
    Model = unet_model_builder(decoder_name, 
                               length, 
                               width, 
                               model_width, 
                               model_depth, 
                               output_nums=output_nums,
                               ds=D_S, 
                               ae=A_E, 
                               ag=A_G, 
                               lstm=LSTM, 
                               dense_loop=num_dense_loop, 
                               q=q, 
                               is_transconv=is_transconv, 
                               train_mode=train_mode,
                               is_base_model_trainable=base_model_trainable).CheXNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
