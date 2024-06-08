# Author: Sakib Mahmud
# Source:
# MIT License


# Import Necessary Libraries
import numpy as np
import tensorflow as tf


def Conv_Block(inputs, model_width, kernel, multiplier, use_batchnorm=True):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width*multiplier, kernel, padding='same')(inputs)
    if use_batchnorm == True:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv1D(inputs, model_width, multiplier, kernel_size, level):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, kernel_size, strides=level, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = tf.keras.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs, size=2):
    # 1D UpSampling Block
    up = tf.keras.layers.UpSampling1D(size=size)(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    shape = inputs.shape
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1])(latent)
    latent = tf.keras.layers.Reshape((shape[1], model_width))(latent)

    return latent


def mix_pool_layer(pool_size):
    def apply(x):
        return tf.keras.layers.Conv1D(int(x.shape[-1]), 1)(
            tf.keras.layers.Add()([tf.keras.layers.MaxPooling1D(3, strides=pool_size, padding='same')(x),
                                   tf.keras.layers.AveragePooling1D(3, strides=pool_size, padding='same')(x)]))
    return apply


def MSP_Unit(inputs, level, conv_filters, multiplier, pooling_type='mix', use_batchnorm=True):
    # MSP Unit
    pool_size = up_size = level
    x = []
    if pooling_type == 'avg':
        x = tf.keras.layers.AveragePooling1D(3, strides=pool_size, padding='same')(inputs)
    elif pooling_type == 'max':
        x = tf.keras.layers.MaxPooling1D(3, strides=pool_size, padding='same')(inputs)
    elif pooling_type == 'mix':
        x = mix_pool_layer(pool_size)(inputs)
    x = Conv_Block(x, conv_filters, 1, multiplier, use_batchnorm=True)
    # x1 = trans_conv1D(x, conv_filters, multiplier, 1, 1)
    x1 = trans_conv1D(x, conv_filters, multiplier, 4, up_size)
    x2 = tf.keras.layers.UpSampling1D(size=up_size)(x)
    x = tf.keras.layers.concatenate([x1, x2], axis=-1)
    x = Conv_Block(x, conv_filters, 1, 1, use_batchnorm=False)

    return x


def MRP_Block(inputs, conv_filters, multiplier, pooling_type='mix', cardinality=5, use_batchnorm=True):
    # MRP Block
    x = []
    if cardinality == 0:
        x = inputs
    else:
        for ii in range(0, cardinality):
            x_temp = MSP_Unit(inputs, 2 ** ii, conv_filters, multiplier, pooling_type=pooling_type, use_batchnorm=False)
            if ii == 0:
                x = tf.keras.layers.concatenate([inputs, x_temp], axis=-1)
            else:
                x = tf.keras.layers.concatenate([x, x_temp], axis=-1)
    # out = Conv_Block(x, conv_filters, 3, multiplier, use_batchnorm=False)
    x3 = Conv_Block(x, conv_filters, 3, multiplier, use_batchnorm=False)
    x5 = Conv_Block(x, conv_filters, 5, multiplier, use_batchnorm=False)
    x7 = Conv_Block(x, conv_filters, 7, multiplier, use_batchnorm=False)
    out = Conv_Block(tf.keras.layers.concatenate([x3, x5, x7], axis=-1), conv_filters, 1, multiplier, use_batchnorm=True)
    return out


def Attention_Block(skip_connection, gating_signal, num_filters, multiplier):
    # Attention Block
    conv1x1_1 = tf.keras.layers.Conv1D(num_filters*multiplier, 1, strides=2)(skip_connection)
    conv1x1_2 = tf.keras.layers.Conv1D(num_filters*multiplier, 1, strides=1)(gating_signal)
    conv1_2 = tf.keras.layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = tf.keras.layers.Activation('relu')(conv1_2)
    conv1_2 = tf.keras.layers.Conv1D(1, 1, strides=1)(conv1_2)
    conv1_2 = tf.keras.layers.Activation('sigmoid')(conv1_2)
    resampler = upConv_Block(conv1_2)
    out = skip_connection*resampler

    return out


class MLMRSNet:
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_nums=1, ds=0, ae=0, cardinality=5, pooling_type='avg', feature_number=1024, is_transconv=True):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # cardinality: Number of MSP layers or units per MRP Block
        # pooling type: 'average', 'max' or 'mix'
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: Transposed Convolution or Interpolation based UpSampling
        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.cardinality = cardinality
        self.pooling_type = pooling_type
        self.feature_number = feature_number
        self.is_transconv = is_transconv

    def MLMRSNet(self):
        """MLMRSNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = MRP_Block(pool, self.model_width, 2 ** (i - 1), pooling_type=self.pooling_type, cardinality=self.cardinality, use_batchnorm=True)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(pool, self.model_width, 2 ** self.model_depth, pooling_type=self.pooling_type, cardinality=self.cardinality, use_batchnorm=True)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            layer_num = self.model_depth - j
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{layer_num}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = Concat_Block(trans_conv1D(deconv, self.model_width, 2 ** (layer_num - 1), 1, 2), convs_list[layer_num - 1])
            elif not self.is_transconv:
                deconv = Concat_Block(upConv_Block(deconv), convs_list[layer_num - 1])
            deconv = MRP_Block(deconv, self.model_width, 2 ** (layer_num - 1), pooling_type=self.pooling_type, cardinality=self.cardinality, use_batchnorm=True)

        # Output
        outputs = []
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model
    
    def MLMRSNet_V2(self):
        """LDNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(0, self.model_depth):
            if i > 0:
                for k in range(1, i):
                    conv = convs["conv%s" % (k-1)]
                    conv = tf.keras.layers.MaxPooling1D(pool_size=(2**(i-k)))(conv)
                    pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
            conv = MRP_Block(pool, self.model_width, 2 ** i, pooling_type=self.pooling_type, cardinality=self.model_depth-i+1, use_batchnorm=True)
            convs["conv%s" % (i-1)] = conv
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
        
        
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(pool, self.model_width, 2 ** self.model_depth, pooling_type=self.pooling_type, cardinality=1, use_batchnorm=True)
        convs["conv%s" % self.model_depth] = conv
        
        # Decoding
        deconv = conv
        deconvs = {}
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connections_all = convs_list[self.model_depth - j - 1]
            # skip_connections_all = MRP_Block(skip_connections_all, self.model_width, 2 ** 0, pooling_type=self.pooling_type, cardinality=j+1, use_batchnorm=True)
            for k in range(0, self.model_depth - j - 1):
                skip_connection = convs_list[k]
                skip_connection = tf.keras.layers.MaxPooling1D(pool_size=(2**((self.model_depth-j)-k-1)))(skip_connection)
                # skip_connection = MRP_Block(skip_connection, self.model_width, 2 ** 0, pooling_type=self.pooling_type, cardinality=j+1, use_batchnorm=True)
                skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
            # deconv_tot = MRP_Block(deconv, self.model_width, 2 ** 0, pooling_type=self.pooling_type, cardinality=j+1, use_batchnorm=True)
            deconv_tot = upConv_Block(deconv, size=2 ** 1)
            deconv_tot = tf.keras.layers.Activation('sigmoid')(deconv_tot)
            deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
            if j > 0:
                for m in range(0, j):
                    # deconv = MRP_Block(deconvs["deconv%s" % m], self.model_width, 2 ** 0, pooling_type=self.pooling_type, cardinality=j+1, use_batchnorm=True)
                    deconv = deconvs["deconv%s" % m]
                    deconv = upConv_Block(deconv, size=(2 ** (j-m)))
                    deconv = tf.keras.layers.Activation('sigmoid')(deconv)
                    deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
            deconv = MRP_Block(deconv_tot, self.model_width, self.model_depth + 1, pooling_type=self.pooling_type, cardinality=j+1, use_batchnorm=True)
            deconvs["deconv%s" % j] = deconv
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, 2, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)

        # Output
        outputs = []
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model
    
    def LDNet(self):
        # Variable LDNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = MRP_Block(pool, self.model_width, 2 ** (i - 1), pooling_type=self.pooling_type, cardinality=self.model_depth-i+1, use_batchnorm=True)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(pool, self.model_width, 2 ** (i - 1), pooling_type=self.pooling_type, cardinality=0, use_batchnorm=True)
        convs["conv%s" % (self.model_depth + 1)] = conv

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}
        deconvs_skip = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if i == 1:
                    skip_connection = convs_list[j]
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j, 2, 2)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                    if (i+j) == (self.model_depth):
                        deconvs_skip["deconv_skip%s" % i] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.is_transconv:
                        deconv = trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j, 2, 2)
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                    if (i + j) == (self.model_depth) and (j != (self.model_depth - 1)):
                        for m in range(1, i-1):
                            temp_deconv = upConv_Block(deconvs_skip["deconv_skip%s" % m], size=(2 ** (i-m)))
                            deconv = tf.keras.layers.concatenate([deconv, temp_deconv], axis=-1)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                    if (i+j) == (self.model_depth):
                        deconvs_skip["deconv_skip%s" % i] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        outputs = []
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model


if __name__ == '__main__':
    # Configurations
    signal_length = 21600  # Length of each Segment
    model_name = 'LDNet'  # UNet or UNetPP
    model_depth = 5  # Number of Level in the CNN Model
    model_width = 32  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 3  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    problem_type = 'Regression'
    cardinality = 5
    pooling_type = 'mix'
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    # Build, Compile and Print Summary
    Model = MLMRSNet(signal_length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
                  ds=D_S, ae=A_E, cardinality=cardinality, pooling_type=pooling_type).LDNet()
    Model.compile(loss= 'mean_absolute_error', optimizer= 'adam', metrics= ['mean_squared_error'])
    # Model.summary()
