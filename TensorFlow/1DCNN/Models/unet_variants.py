# Import Necessary Libraries
import numpy as np
import tensorflow as tf


# Operational Layers.
class Oper1D(tf.keras.Model):
  def __init__(self, filters, kernel_size, padding='same', strides=1, activation=None, q=1):
    super(Oper1D, self).__init__()
    self.activation = activation
    self.q = q
    self.all_layers = []
    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, activation=None))

  @tf.function
  def call(self, input_tensor, training=False):
    x = self.all_layers[0](input_tensor)  # First convolutional layer.
    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i+1))
    if self.activation is not None:
      # return eval('tf.nn.' + self.activation + '(x)')
      x = tf.keras.layers.Activation(self.activation)(x)
      return x
    else:
      return x


# Transposed Operational Layers.
class Oper1DTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides=1, padding='same', activation=None, q=1):
    super(Oper1DTranspose, self).__init__()
    self.activation = activation
    self.q = q
    self.all_layers = []
    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv1DTranspose(filters, kernel_size, strides=strides, padding=padding, activation=None))
  @tf.function
  def call(self, input_tensor, training=False):
    x = self.all_layers[0](input_tensor)  # First convolutional layer.
    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i+1))
    if self.activation is not None:
      # return eval('tf.nn.' + self.activation + '(x)')
      x = tf.keras.layers.Activation(self.activation)(x)
      return x
    else:
      return x


def Conv_Block(inputs, model_width, kernel, multiplier, use_batchnorm=True):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width*multiplier, kernel, padding='same')(inputs)
    if use_batchnorm == True:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Recurrent_Conv_Block(inputs, model_width, kernel_size, multiplier, t):
    # 1D Recurrent Convolutional Block for RUNet variants
    x = inputs
    for _ in range(t):
        x_temp = Conv_Block(x, model_width, kernel_size, multiplier)
        x = tf.keras.layers.concatenate([x_temp, inputs], axis=-1)
    
    x = Conv_Block(x, model_width, kernel_size, multiplier)
    
    return x


def Self_Recurrent_Conv_Block(inputs, model_width, kernel_size, multiplier, t, q):
    # 1D Recurrent Convolutional Block for RUNet variants
    x = inputs
    for i in range(t):
        x_temp = Oper1D(model_width * multiplier,  kernel_size, q=q)(x)
        x = tf.keras.layers.concatenate([x_temp, inputs], axis=-1)
    
    x = Conv_Block(x, model_width, kernel_size, multiplier)
    
    return x


def Conv_Block_DW(inputs, model_width, kernel, multiplier, strides=1):
    # Depthwise Convolution.
    x = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel, strides=strides, padding="same")(inputs)
    x = tf.keras.activations.gelu(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, inputs])

    # Pointwise Convolution.
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel_size=1)(x)
    x = tf.keras.activations.gelu(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def trans_conv1D(inputs, model_width, multiplier):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, 2, strides=2, padding='same')(inputs)  # Stride = 2, Kernel Size = 2
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


def dense_block(x, num_filters, num_layers, bottleneck=True):
    for i in range(num_layers):
        cb = Conv_Block(x, num_filters, bottleneck=bottleneck)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)

    return x


def mix_pool_layer(pool_size):
    def apply(x):
        return tf.keras.layers.Conv1D(int(x.shape[-1]), 1)(
            tf.keras.layers.Add()([tf.keras.layers.MaxPooling1D(3, strides=pool_size, padding='same')(x),
                                   tf.keras.layers.AveragePooling1D(3, strides=pool_size, padding='same')(x)]))
    return apply


def Attention_Block(skip_connection, gating_signal, num_filters, multiplier):
    # Attention Block
    conv1x1_1 = tf.keras.layers.Conv1D(num_filters*multiplier, 1, strides=2)(skip_connection)
    conv1x1_1 = tf.keras.layers.BatchNormalization()(conv1x1_1)
    conv1x1_2 = tf.keras.layers.Conv1D(num_filters*multiplier, 1, strides=1)(gating_signal)
    conv1x1_2 = tf.keras.layers.BatchNormalization()(conv1x1_2)
    conv1_2 = tf.keras.layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = tf.keras.layers.Activation('relu')(conv1_2)
    conv1_2 = tf.keras.layers.Conv1D(1, 1, strides=1)(conv1_2)
    conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = tf.keras.layers.Activation('sigmoid')(conv1_2)
    resampler1 = upConv_Block(conv1_2)
    resampler2 = trans_conv1D(conv1_2, 1, 1)
    resampler = tf.keras.layers.add([resampler1, resampler2])
    out = skip_connection * resampler

    return out


def MultiResBlock(inputs, model_width, kernel, multiplier, alpha):
    # MultiRes Block
    # U {int} -- Number of filters in a corrsponding UNet stage
    # inp {keras layer} -- input layer

    w = alpha * model_width

    shortcut = inputs
    shortcut = Conv_Block(shortcut, int(w * 0.167) + int(w * 0.333) + int(w * 0.5), 1, multiplier)

    conv3x3 = Conv_Block(inputs, int(w * 0.167), kernel, multiplier)
    conv5x5 = Conv_Block(conv3x3, int(w * 0.333), kernel, multiplier)
    conv7x7 = Conv_Block(conv5x5, int(w * 0.5), kernel, multiplier)

    out = tf.keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Add()([shortcut, out])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization()(out)

    return out


def ResPath(inputs, model_depth, model_width, kernel, multiplier):
    # ResPath
    # filters {int} -- [description]
    # length {int} -- length of ResPath
    # inp {keras layer} -- input layer

    shortcut = inputs
    shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

    out = Conv_Block(inputs, model_width, kernel, multiplier)
    out = tf.keras.layers.Add()([shortcut, out])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization()(out)

    for _ in range(1, model_depth):
        shortcut = out
        shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

        out = Conv_Block(out, model_width, kernel, multiplier)
        out = tf.keras.layers.Add()([shortcut, out])
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization()(out)

    return out


class UNet:
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_nums=1, ds=1, ae=0, ag=0, lstm=0, alpha=1, t=2, feature_number=1024, is_transconv=True, q=3):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # ag: Checks where Attention Guided is active or not, either 0 or 1 [Default value set as 0]
        # lstm: Checks where Bidirectional LSTM is active or not, either 0 or 1 [Default value set as 0]
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer
        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        self.alpha = alpha
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        self.t = t
        self.q = q

    def UNet(self):
        # Variable UNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            if self.LSTM == 1:
                x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(skip_connection)
                x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(deconv)
                merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            elif self.LSTM == 0:
                deconv = Concat_Block(deconv, skip_connection)
            deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))

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

    def UNetE(self):
        # Variable Ensemble UNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)

        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        # Decoding
        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], conv, self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(conv, self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(conv)
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
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

    def UNetP(self):
        # Variable UNet+ Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        # Decoding
        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], conv, self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(conv, self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(conv)
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    skip_connection = deconvs["deconv%s%s" % (j, (i - 1))]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(deconvs["deconv%s%s" % (j, (i - 1))], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
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

    def UNetPP(self):
        """Variable UNet++ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        convs["conv%s" % (self.model_depth + 1)] = conv

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if i == 1:
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    if self.A_G == 1:
                        deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        if self.A_G == 1:
                            deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                        merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
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

    def UNet3P(self):
        # Variable UNet3+ Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        # Decoding
        deconv = conv
        deconvs = {}
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connections_all = convs_list[self.model_depth - j - 1]
            skip_connections_all = Conv_Block(skip_connections_all, self.model_width, self.kernel_size, 2 ** 0)
            for k in range(0, self.model_depth - j - 1):
                skip_connection = convs_list[k]
                skip_connection = tf.keras.layers.MaxPooling1D(pool_size=(2**((self.model_depth-j)-k-1)))(skip_connection)
                skip_connection = Conv_Block(skip_connection, self.model_width, self.kernel_size, 2 ** 0)
                skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
            deconv_tot = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** 0)
            deconv_tot = upConv_Block(deconv_tot, size=2 ** 1)
            deconv_tot = tf.keras.layers.Activation('sigmoid')(deconv_tot)
            deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
            if j > 0:
                for m in range(0, j):
                    deconv = Conv_Block(deconvs["deconv%s" % m], self.model_width, self.kernel_size, 2 ** 0)
                    deconv = upConv_Block(deconv, size=(2 ** (j-m)))
                    deconv = tf.keras.layers.Activation('sigmoid')(deconv)
                    deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
            deconv = Conv_Block(deconv_tot, self.model_width, self.kernel_size, self.model_depth + 1)
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

    def UNet4P(self):
        # Variable UNet3+ Model Design
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
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** i)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** i)
            convs["conv%s" % (i-1)] = conv
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        convs["conv%s" %(self.model_depth - 1)] = conv

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
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True,
                                                            kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                    if (i+j) == (self.model_depth):
                        deconvs_skip["deconv_skip%s" % i] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    if self.A_G == 1:
                        deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        if self.A_G == 1:
                            deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                        merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True,
                                                            kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
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

    def MultiResUNet(self):
        """Variable MultiResUNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        mresblocks = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.alpha)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(mresblock)
            mresblocks["mres%s" % i] = ResPath(mresblock, (self.model_depth - i + 1), self.model_width, self.kernel_size, 2 ** (i - 1))

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)

        mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)

        # Decoding
        deconv = mresblock
        mresblocks_list = list(mresblocks.values())

        for j in range(0, self.model_depth):
            skip_connection = mresblocks_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(mresblocks_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            if self.LSTM == 1:
                x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(skip_connection)
                x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(deconv)
                merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            elif self.LSTM == 0:
                deconv = Concat_Block(deconv, skip_connection)
            deconv = MultiResBlock(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha)

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

    def MultiResUNet3P(self):
        """Variable MultiResUNet3+ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        mresblocks = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 2)):
            if i > 1:
                for k in range(1, i):
                    conv = mresblocks["mres%s" %k]
                    conv = tf.keras.layers.MaxPooling1D(pool_size=((2 ** (i - k))))(conv)
                    pool = tf.keras.layers.Activation('sigmoid')(conv)
                    pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
                    # pool = Conv_Block(pool, self.model_width, 1, 2 ** (i - k), use_batchnorm=False)
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.alpha)
            mresblocks["mres%s" %i] = ResPath(mresblock, self.model_depth - i + 1, self.model_width, self.kernel_size, 2 ** i)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(mresblock)

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)

        # Decoding
        deconvs = {}
        convs_list = list(mresblocks.values())
        deconv = convs_list[-1]
        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            if self.LSTM == 1:
                x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (model_depth - j - 1)))))(skip_connection)
                x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (model_depth - j - 1)))))(deconv)
                merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            elif self.LSTM == 0:
                deconv = Concat_Block(deconv, skip_connection)
            for m in range(0, j+1):
                if m == 0:
                    temp_deconv = convs_list[-1]
                else:
                    temp_deconv = deconvs["deconv%s" % m]
                temp_deconv = upConv_Block(temp_deconv, size=(2 ** (j-m+1)))
                temp_deconv = tf.keras.layers.Activation('sigmoid')(temp_deconv)
                deconv = tf.keras.layers.concatenate([deconv, temp_deconv], axis=-1)
                # deconv = Conv_Block(deconv, self.model_width, 1, 2 ** (j-m+1), use_batchnorm=False)
            deconv = MultiResBlock(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha)
            deconvs["deconv%s" %(j+1)] = deconv
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
    
    def RUNet(self):
        # Variable Recurrent UNet (RUNet) Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            conv = Recurrent_Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)
        conv = Recurrent_Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            if self.LSTM == 1:
                x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(skip_connection)
                x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(deconv)
                merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            elif self.LSTM == 0:
                deconv = Concat_Block(deconv, skip_connection)
            deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.t)
            deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.t)
            

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
    
    def R2UNet(self):
        # Variable Recurrent Residual UNet (RUNet) Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv_raw = Conv_Block(pool, self.model_width, 1, 2 ** (i - 1))
            conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            conv = Recurrent_Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            conv = tf.keras.layers.Add()([conv_raw, conv])
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv_raw = Conv_Block(pool, self.model_width, 1, 2 ** self.model_depth)
        conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)
        conv = Recurrent_Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)
        conv = tf.keras.layers.Add()([conv_raw, conv])

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            if self.LSTM == 1:
                x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(skip_connection)
                x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(deconv)
                merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            elif self.LSTM == 0:
                deconv = Concat_Block(deconv, skip_connection)
            deconv_raw = Conv_Block(deconv, self.model_width, 1, 2 ** (self.model_depth - j - 1))
            deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.t)
            deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.t)
            deconv = tf.keras.layers.Add()([deconv_raw, deconv])
            

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
    
    def R2UNetPP(self):
        """Variable R2UNet++ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv_raw = Conv_Block(pool, self.model_width, 1, 2 ** (i - 1))
            conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            conv = tf.keras.layers.Add()([conv_raw, conv])
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv_raw = Conv_Block(pool, self.model_width, 1, 2 ** self.model_depth)
        conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)
        conv = tf.keras.layers.Add()([conv_raw, conv])
        convs["conv%s" % (self.model_depth + 1)] = conv

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if i == 1:
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv_raw = Conv_Block(deconv, self.model_width, 1, 2 ** j)
                    deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j, self.t)
                    deconv = tf.keras.layers.Add()([deconv_raw, deconv])
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    if self.A_G == 1:
                        deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        if self.A_G == 1:
                            deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                        merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                    deconv_raw = Conv_Block(deconv, self.model_width, 1, 2 ** j)
                    deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j, self.t)
                    deconv = tf.keras.layers.Add()([deconv_raw, deconv])
                    deconvs["deconv%s%s" % (j, i)] = deconv
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
    
    def R2UNet3P(self):
        # Variable UNet3+ Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv_raw = Conv_Block(pool, self.model_width, 1, 2 ** (i - 1))
            conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            conv = Recurrent_Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1), self.t)
            conv = tf.keras.layers.Add()([conv_raw, conv])
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv_raw = Conv_Block(pool, self.model_width, 1, 2 ** self.model_depth)
        conv = Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)
        conv = Recurrent_Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t)
        conv = tf.keras.layers.Add()([conv_raw, conv])

        # Decoding
        deconv = conv
        deconvs = {}
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connections_all = convs_list[self.model_depth - j - 1]
            skip_connections_all = Conv_Block(skip_connections_all, self.model_width, self.kernel_size, 2 ** 0)
            for k in range(0, self.model_depth - j - 1):
                skip_connection = convs_list[k]
                skip_connection = tf.keras.layers.MaxPooling1D(pool_size=(2**((self.model_depth-j)-k-1)))(skip_connection)
                skip_connection_raw = Conv_Block(skip_connection, self.model_width, 1, 2 ** 0)
                skip_connection = Recurrent_Conv_Block(skip_connection, self.model_width, self.kernel_size, 2 ** 0, self.t)
                skip_connection = Recurrent_Conv_Block(skip_connection, self.model_width, self.kernel_size, 2 ** 0, self.t)
                skip_connection = tf.keras.layers.Add()([skip_connection_raw, skip_connection])
                skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
            deconv_raw = Conv_Block(deconv, self.model_width, 1, 2 ** 0)
            deconv_tot = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** 0, self.t)
            deconv_tot = Recurrent_Conv_Block(deconv_tot, self.model_width, self.kernel_size, 2 ** 0, self.t)
            deconv_tot = tf.keras.layers.Add()([deconv_raw, deconv_tot])
            deconv_tot = upConv_Block(deconv_tot, size=2 ** 1)
            deconv_tot = tf.keras.layers.Activation('sigmoid')(deconv_tot)
            deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
            if j > 0:
                for m in range(0, j):
                    deconv_raw = Conv_Block(deconvs["deconv%s" % m], self.model_width, 1, 2 ** 0)
                    deconv = Recurrent_Conv_Block(deconvs["deconv%s" % m], self.model_width, self.kernel_size, 2 ** 0, self.t)
                    deconv = Recurrent_Conv_Block(deconvs["deconv%s" % m], self.model_width, self.kernel_size, 2 ** 0, self.t)
                    deconv = tf.keras.layers.Add()([deconv_raw, deconv])
                    deconv = upConv_Block(deconv, size=(2 ** (j-m)))
                    deconv = tf.keras.layers.Activation('sigmoid')(deconv)
                    deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
            deconv_raw = Conv_Block(deconv_tot, self.model_width, 1, self.model_depth + 1)
            deconv = Recurrent_Conv_Block(deconv_tot, self.model_width, self.kernel_size, self.model_depth + 1, self.t)
            deconv = Recurrent_Conv_Block(deconv, self.model_width, self.kernel_size, self.model_depth + 1, self.t)
            deconv = tf.keras.layers.Add()([deconv_raw, deconv])
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
    
    def SelfR2UNetPP(self):
        """Variable Self-UNet++ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Self_Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.t, q=self.q)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Self_Recurrent_Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.t, q=1)
        convs["conv%s" % (self.model_depth + 1)] = conv

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if i == 1:
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        # deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j)
                        deconv = Oper1DTranspose(self.model_width * (2 ** j), 4, strides=2, padding='same', activation='tanh', q=self.q)(convs_list[j + 1])
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Oper1D(self.model_width * (2 ** j),  self.kernel_size, q=self.q)(deconv)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    if self.A_G == 1:
                        deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        if self.A_G == 1:
                            deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = Oper1DTranspose(self.model_width * (2 ** j), 4, strides=2, padding='same', activation='tanh', q=self.q)(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                        merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                    deconv = Oper1D(self.model_width * (2 ** j),  self.kernel_size, q=self.q)(deconv)
                    deconvs["deconv%s%s" % (j, i)] = deconv
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
    
    def SelfUNetPP(self):
        """Variable Self-ONN based UNet++ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Oper1D(self.model_width * (2 ** (i - 1)),  self.kernel_size, q=self.q)(pool)
            conv = Oper1D(self.model_width * (2 ** (i - 1)),  self.kernel_size, q=self.q)(conv)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Oper1D(self.model_width * (2 ** self.model_depth),  self.kernel_size, q=self.q)(pool)
        conv = Oper1D(self.model_width * (2 ** self.model_depth),  self.kernel_size, q=self.q)(conv)
        convs["conv%s" % (self.model_depth + 1)] = conv

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if i == 1:
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], convs_list[j + 1], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = Oper1DTranspose(self.model_width * (2 ** j), 4, strides=2, padding='same', activation='tanh', q=self.q)(convs_list[j + 1])
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, skip_connection)
                    deconv = Oper1D(self.model_width * (2 ** j),  self.kernel_size, q=self.q)(deconv)
                    deconv = Oper1D(self.model_width * (2 ** j),  self.kernel_size, q=self.q)(deconv)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    if self.A_G == 1:
                        deconv_tot = Attention_Block(deconv_tot, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        if self.A_G == 1:
                            deconv_temp = Attention_Block(deconv_temp, deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.A_G == 1:
                        skip_connection = Attention_Block(convs_list[j], deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    if self.is_transconv:
                        deconv = Oper1DTranspose(self.model_width * (2 ** j), 4, strides=2, padding='same', activation='tanh', q=self.q)(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    if self.LSTM == 1:
                        x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(skip_connection)
                        x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** j), np.int32(self.model_width * (2 ** j))))(deconv)
                        deconv_tot = tf.expand_dims(deconv_tot, axis=1)
                        merge = tf.keras.layers.concatenate([x1, x2, deconv_tot], axis=-1)
                        deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (j - 1))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
                    elif self.LSTM == 0:
                        deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                    deconv = Oper1D(self.model_width * (2 ** j),  self.kernel_size, q=self.q)(deconv)
                    deconv = Oper1D(self.model_width * (2 ** j),  self.kernel_size, q=self.q)(deconv)
                    deconvs["deconv%s%s" % (j, i)] = deconv
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
    
    def SelfUNet3P(self):
        # Variable Self-ONN-based UNet3+ Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Oper1D(self.model_width * (2 ** (i - 1)),  self.kernel_size, q=self.q)(pool)
            conv = Oper1D(self.model_width * (2 ** (i - 1)),  self.kernel_size, q=self.q)(conv)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Oper1D(self.model_width * (2 ** self.model_depth),  self.kernel_size, q=self.q)(pool)
        conv = Oper1D(self.model_width * (2 ** self.model_depth),  self.kernel_size, q=self.q)(conv)

        # Decoding
        deconv = conv
        deconvs = {}
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connections_all = convs_list[self.model_depth - j - 1]
            skip_connections_all = Oper1D(self.model_width * (2 ** 0),  self.kernel_size, q=self.q)(skip_connections_all)
            for k in range(0, self.model_depth - j - 1):
                skip_connection = convs_list[k]
                skip_connection = tf.keras.layers.MaxPooling1D(pool_size=(2**((self.model_depth-j)-k-1)))(skip_connection)
                skip_connection = Oper1D(self.model_width * (2 ** 0),  self.kernel_size, q=self.q)(skip_connection)
                skip_connections_all = tf.keras.layers.concatenate([skip_connections_all, skip_connection], axis=-1)
            deconv_tot = Oper1D(self.model_width * (2 ** 0),  self.kernel_size, q=self.q)(deconv)
            deconv_tot = upConv_Block(deconv_tot, size=2 ** 1)
            deconv_tot = tf.keras.layers.Activation('sigmoid')(deconv_tot)
            deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
            if j > 0:
                for m in range(0, j):
                    deconv = Oper1D(self.model_width * (2 ** 0),  self.kernel_size, q=self.q)(deconvs["deconv%s" % m])
                    deconv = upConv_Block(deconv, size=(2 ** (j-m)))
                    deconv = tf.keras.layers.Activation('sigmoid')(deconv)
                    deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
            deconv = Oper1D(self.model_width * (self.model_depth + 1),  self.kernel_size, q=self.q)(deconv_tot)
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


if __name__ == '__main__':
    # Configurations
    length = 21600  # Length of each Segment
    model_name = 'MultiResUNet3P'  # UNet or UNetPP
    model_depth = 5  # Number of Level in the CNN Model
    model_width = 32  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 3  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 0  # Turn on for Guided Attention
    LSTM = 0  # Turn on for LSTM, Implemented for UNet and MultiResUNet only
    problem_type = 'Regression'
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    is_transconv = True # True: Transposed Convolution, False: UpSampling
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    '''Only required for MultiResUNet'''
    alpha = 1  # Model Width Expansion Parameter, for MultiResUNet only
    t = 2
    q = 3
    #
    Model = UNet(length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
                 ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, alpha=alpha, t=t, is_transconv=is_transconv, q=q).UNet4P()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    # Model.summary()
