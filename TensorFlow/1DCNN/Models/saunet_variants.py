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


class DropBlock1D(tf.keras.layers.Layer):
    def __init__(self, block_size, keep_prob, sync_channels=False, **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock1D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels

    def get_config(self):
        config = {'block_size': self.block_size, 'keep_prob': self.keep_prob, 'sync_channels': self.sync_channels}
        base_config = super(DropBlock1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, feature_dim):
        """Get the number of activation units to drop"""
        feature_dim = tf.cast(feature_dim, tf.keras.backend.floatx())
        block_size = tf.constant(self.block_size, dtype=tf.keras.backend.floatx())
        return ((1.0 - self.keep_prob) / block_size) * (feature_dim / (feature_dim - block_size + 1.0))

    def _compute_valid_seed_region(self, seq_length):
        positions = tf.range(seq_length)
        half_block_size = self.block_size // 2
        valid_seed_region = tf.keras.backend.switch(
            tf.keras.backend.all(
                tf.keras.backend.stack(
                    [
                        positions >= half_block_size,
                        positions < seq_length - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            tf.ones((seq_length,), tf.float32),
            tf.zeros((seq_length,), tf.float32),
        )
        return tf.expand_dims(tf.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        seq_length = shape[1]
        mask = tf.keras.backend.random_bernoulli(shape, p=self._get_gamma(seq_length))
        mask *= self._compute_valid_seed_region(seq_length)
        mask = tf.keras.layers.MaxPool1D(pool_size=self.block_size, padding='same', strides=1)(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):
        def dropped_inputs():
            outputs = inputs
            shape = tf.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (tf.cast(tf.math.reduce_prod(shape), dtype=tf.keras.backend.floatx()) / tf.math.reduce_sum(mask))
            return outputs
        return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=training)


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


def Conv_Block(inputs, model_width, kernel, multiplier):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Conv_Block_Regulated(inputs, model_width, kernel, multiplier, block_size, keep_prob):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding='same')(inputs)
    x = DropBlock1D(block_size=block_size, keep_prob=keep_prob)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

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
        cb = conv_block(x, num_filters, bottleneck=bottleneck)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)

    return x


def spatial_attention(input_feature):
    kernel_size = 7
    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = tf.keras.layers.Concatenate(axis=2)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1
    return tf.math.multiply(input_feature, cbam_feature)


def MultiResBlock_Regulated(inputs, model_width, kernel, multiplier, alpha, block_size, keep_prob):
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
    out = DropBlock1D(block_size=block_size, keep_prob=keep_prob)(out)
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


class SAUNet:
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, output_nums=1, ds=1, ae=0, alpha=1, 
                 feature_number=1024, block_size=7, keep_prob=0.9, is_transconv=True, q=3):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer
        # q: q-order for ONNs
        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.alpha = alpha
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.q = q

    def SAUNet(self):
        # Variable Spatial Attention (SA) UNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0 or self.output_nums < 1:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block_Regulated(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.block_size, self.keep_prob)
            conv = Conv_Block_Regulated(conv, self.model_width, self.kernel_size, 2 ** (i - 1), self.block_size, self.keep_prob)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Conv_Block_Regulated(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.block_size, self.keep_prob)
        conv = spatial_attention(conv)
        conv = Conv_Block_Regulated(conv, self.model_width, self.kernel_size, 2 ** self.model_depth, self.block_size, self.keep_prob)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            deconv = Concat_Block(deconv, skip_connection)
            deconv = Conv_Block_Regulated(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.block_size, self.keep_prob)
            deconv = Conv_Block_Regulated(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.block_size, self.keep_prob)

        # Output
        outputs = []
        if self.output_nums == 1:
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        elif self.output_nums > 1:
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model
    
    def SAMultiResUNet(self):
        """Variable Spatial Attention (SA) Multi-Residual (MultiRes) UNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0 or self.output_nums < 1:
            raise ValueError("Please Check the Values of the Input Parameters!")

        mresblocks = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            mresblock = MultiResBlock_Regulated(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.alpha, self.block_size, self.keep_prob)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(mresblock)
            mresblocks["mres%s" % i] = ResPath(mresblock, (self.model_depth - i + 1), self.model_width, self.kernel_size, 2 ** (i - 1))

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)

        mresblock = MultiResBlock_Regulated(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha, self.block_size, self.keep_prob)
        mresblock = spatial_attention(mresblock)
        mresblock = MultiResBlock_Regulated(mresblock, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha, self.block_size, self.keep_prob)

        # Decoding
        deconv = mresblock
        mresblocks_list = list(mresblocks.values())

        for j in range(0, self.model_depth):
            skip_connection = mresblocks_list[self.model_depth - j - 1]
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            deconv = Concat_Block(deconv, skip_connection)
            deconv = MultiResBlock_Regulated(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha, self.block_size, self.keep_prob)

        # Output
        outputs = []
        if self.output_nums == 1:
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        elif self.output_nums > 1:
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model
    
    def SelfSAUNet(self):
        # Variable Self-ONN based Spatial Attention (SA) UNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0 or self.output_nums < 1:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Oper1D(self.model_width * (2 ** (i - 1)),  self.kernel_size, q=self.q)(pool)
            conv = DropBlock1D(block_size=self.block_size, keep_prob=self.keep_prob)(conv)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.Activation('tanh')(conv)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Oper1D(self.model_width * (2 ** self.model_depth),  self.kernel_size, q=self.q)(pool)
        conv = DropBlock1D(block_size=self.block_size, keep_prob=self.keep_prob)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('tanh')(conv)
        conv = spatial_attention(conv)
        conv = Oper1D(self.model_width * (2 ** self.model_depth),  self.kernel_size, q=self.q)(conv)
        conv = DropBlock1D(block_size=self.block_size, keep_prob=self.keep_prob)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('tanh')(conv)
        
        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = Oper1DTranspose(self.model_width * (2 ** (self.model_depth - j - 1)), 4, strides=2, padding='same', activation='tanh', q=self.q)(deconv)
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            deconv = Concat_Block(deconv, skip_connection)
            deconv = Oper1D(self.model_width * (2 ** (self.model_depth - j - 1)),  self.kernel_size, q=self.q)(deconv)
            deconv = DropBlock1D(block_size=self.block_size, keep_prob=self.keep_prob)(deconv)
            deconv = tf.keras.layers.BatchNormalization()(deconv)
            deconv = tf.keras.layers.Activation('tanh')(deconv)

        # Output
        outputs = []
        if self.output_nums == 1:
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        elif self.output_nums > 1:
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model


if __name__ == '__main__':
    # Configurations
    length = 1024  # Length of each Segment
    model_name = 'SelfSAUNet'  # UNet or UNetPP
    model_depth = 5  # Number of Level in the CNN Model
    model_width = 32  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 1  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    is_transconv = True # True: Transposed Convolution, False: UpSampling
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    '''Only required for MultiResUNet'''
    alpha = 1  # Model Width Expansion Parameter, for MultiResUNet only
    q = 3
    block_size = 9
    keep_prob = 0.9
    #
    Model = SAUNet(length, model_depth, num_channel, model_width, kernel_size, output_nums=output_nums, ds=D_S, alpha=alpha, is_transconv=is_transconv, block_size=block_size, keep_prob= keep_prob, q=q).SelfSAUNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
