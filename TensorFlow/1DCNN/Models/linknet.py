# Author: Sakib Mahmud
# Source: https://github.com/Sakib1263/LinkNet-Segmentation-AutoEncoder-1D-2D-Tensorflow-Keras
# MIT License


# Import Necessary Libraries
import tensorflow as tf


def Conv_Block(inputs, model_width, kernel, multiplier):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv1D(inputs, model_width, multiplier):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, 2, strides=2, padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Add_Block(input1, *argv):
    # Add Block from the TensorFlow.KERAS Library
    addblk = input1
    for arg in range(0, len(argv)):
        addblk = tf.keras.layers.Add()([addblk, argv[arg]])

    return addblk


def Feature_Extraction_Block(inputs, model_width, Dim2, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * Dim2)(latent)
    latent = tf.keras.layers.Reshape((Dim2, model_width))(latent)

    return latent


def MultiResBlock(inputs, model_width, kernel, multiplier, alpha):
    # MultiRes Block
    # U {int} -- Number of filters in a corrsponding LinkNet stage
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

    for i in range(1, model_depth):
        shortcut = out
        shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

        out = Conv_Block(out, model_width, kernel, multiplier)
        out = tf.keras.layers.Add()([shortcut, out])
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization()(out)

    return out


class LinkNet:
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_nums=1, ds=0, ae=0, alpha=1, feature_number=1024):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # alpha: This Parameter is only for MultiResLinkNet, default value is 1
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode

        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.alpha = alpha
        self.feature_number = feature_number

    def LinkNet(self):
        # Variable LinkNet Model Design
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
            pool = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            deconv = Add_Block(trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1)), convs_list[self.model_depth - j - 1])
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

    def LinkNetE(self):
        # Variable Ensemble LinkNet Model Design
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
            pool = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)

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
                    deconv = Add_Block(convs_list[j], trans_conv1D(conv, self.model_width, 2 ** j))
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = Add_Block(convs_list[j], trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j))
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv = Add_Block(convs_list[j], trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j))
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

    def LinkNetP(self):
        # Variable LinkNet+ Model Design
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
            pool = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
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
                    deconv = Add_Block(convs_list[j], trans_conv1D(conv, self.model_width, 2 ** j))
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = Add_Block(convs_list[j], trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j))
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv = Add_Block(deconvs["deconv%s%s" % (j, (i - 1))], trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j))
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

    def LinkNetPP(self):
        """Variable LinkNet++ Model Design"""
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
            pool = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
        conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = Add_Block(convs_list[j], trans_conv1D(conv, self.model_width, 2 ** j))
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = Add_Block(convs_list[j], trans_conv1D(convs_list[j + 1], self.model_width, 2 ** j))
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        deconv_tot = Add_Block(deconv_tot, deconv_temp)
                    deconv = Add_Block(convs_list[j], deconv_tot, trans_conv1D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j))
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

    def MultiResLinkNet(self):
        # 1D MultiResLinkNet with an option for Deep Supervision and/or being used as an AutoEncoder
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
            pool = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)

        mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)

        # Decoding
        deconv = mresblock
        mresblocks_list = list(mresblocks.values())

        for j in range(0, self.model_depth):
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            deconv = Add_Block(trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1)), mresblocks_list[self.model_depth - j - 1])
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


if __name__ == '__main__':
    # Libraries
    import keras
    # Configurations
    signal_length = 1024  # Length of each Segment
    model_name = 'LinkNet'  # LinkNet or LinkNetPP
    model_depth = 5  # Number of Level in the CNN Model
    model_width = 64  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 4  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    problem_type = 'Regression'
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    '''Only required for MultiResLinkNet'''
    alpha = 1  # Model Width Expansion Parameter, for MultiResLinkNet only
    Total_Subjects = 23  # Total 23 Subjects in this case
    #
    Model = LinkNet(signal_length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums, ds=D_S, ae=A_E, alpha=alpha).LinkNetPP()
    Model.compile(loss= 'mean_absolute_error', optimizer= 'adam', metrics= ['mean_squared_error','accuracy'])
    Model.summary()