'''Author: Sakib Mahmud'''
'''Source: https://github.com/Sakib1263/UNet-2D-Segmentation-AutoEncoder-Model-Builder-KERAS/blob/main/Codes/UNet_2DCNN.py'''
'''MIT Free License'''

# Import Necessary Libraries
from keras.models import Model
from keras.layers import Input, Reshape, Flatten, Dense, Add, Concatenate, BatchNormalization, Activation
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose


def Conv_Block(inputs, model_width, kernel, multiplier):
    # 1D Convolutional Block
    conv = Conv2D(model_width * multiplier, kernel, padding='same')(inputs)
    batch_norm = BatchNormalization()(conv)
    activate = Activation('relu')(batch_norm)

    return activate


def trans_conv2D(inputs, model_width, multiplier):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    transposed_conv = Conv2DTranspose(model_width * multiplier, (2,2), strides=(2,2), padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    batch_norm = BatchNormalization()(transposed_conv)
    activate = Activation('relu')(batch_norm)

    return activate


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = Concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs):
    # 1D UpSampling Block
    up = UpSampling2D(size=(2, 2))(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, Dim2, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    latent = Flatten()(inputs)
    latent = Dense(feature_number, name='features')(latent)
    latent = Dense(model_width * Dim2)(latent)
    latent = Reshape((Dim2, model_width))(latent)

    return latent


def MultiResBlock(inputs, model_width, kernel, multiplier, alpha):
    ''' MultiRes Block'''
    # U {int} -- Number of filters in a corrsponding UNet stage
    # inp {keras layer} -- input layer

    w = alpha * model_width

    shortcut = inputs
    shortcut = Conv_Block(shortcut, int(w * 0.167) + int(w * 0.333) + int(w * 0.5), 1, multiplier)

    conv3x3 = Conv_Block(inputs, int(w * 0.167), kernel, multiplier)
    conv5x5 = Conv_Block(conv3x3, int(w * 0.333), kernel, multiplier)
    conv7x7 = Conv_Block(conv5x5, int(w * 0.5), kernel, multiplier)

    out = Concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = BatchNormalization()(out)
    out = Add()([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    return out


def ResPath(inputs, model_depth, model_width, kernel, multiplier):

    ''' ResPath '''
    # filters {int} -- [description]
    # length {int} -- length of ResPath
    # inp {keras layer} -- input layer

    shortcut = inputs
    shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

    out = Conv_Block(inputs, model_width, kernel, multiplier)
    out = Add()([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    for i in range(1, model_depth):
        shortcut = out
        shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

        out = Conv_Block(out, model_width, kernel, multiplier)
        out = Add()([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

    return out


'''Version 2 (v2) of all Models use Transposed Convolution instead of UpSampling'''
class UNet:
    def __init__(self, length, width, model_depth, num_channel, model_width, kernel_size,
                 problem_type='Regression', output_nums=1, ds=0, ae=0, *argv):
        # length: Input Image Length (x-dim)
        # width: Input Image Width (y-dim) [Normally same as the x-dim i.e., Square shape]
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder, only useful in the A_E Mode
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # alpha: This Parameter is only for MultiResUNet, default value is 1
        self.length = length
        self.width = width
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        if len(argv) == 0 and ae == 1:
            print("ERROR: Please Check the Input Parameters! Autoencoder mode was selected but arguments were not provided!")
        elif len(argv) == 2 and ae == 0:
            print("ERROR: Please Check the Input Parameters! Autoencoder mode was not selected but extra arguments were provided!")
        elif len(argv) == 1 and ae == 1:
            self.feature_number = argv[0]
        elif len(argv) == 1 and ae == 0:
            self.alpha = argv[0]  # Alpha parameter, only for MultiResUNet
        elif len(argv) == 2 and ae == 1:
            self.feature_number = argv[0]
            self.alpha = argv[1]
        elif len(argv) > 2:
            print("ERROR: Please Check the Input Parameters! More than 2 optional arguments are not expected!")

    def UNet(self):
        """Variable UNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=2)(conv)
        convs["conv%s" % i] = conv

        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if (self.A_E == 0) and (self.D_S == 0):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif (self.A_E == 0) and (self.D_S == 1):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
            level0 = Conv2D(1, (1, 1), name=f'level{self.model_depth}')(conv)
            levels.append(level0)
        elif (self.A_E == 1) and (self.D_S == 0):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif (self.A_E == 1) and (self.D_S == 1):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
            level0 = Conv2D(1, (1, 1), name=f'level{self.model_depth}')(conv)
            levels.append(level0)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        deconv = Conv_Block(Concat_Block(upConv_Block(conv), convs_list[self.model_depth - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - 1))
        deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - 1))

        for j in range(1, self.model_depth):
            if self.D_S == 0:
                deconv = Conv_Block(Concat_Block(upConv_Block(deconv), convs_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
                deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            elif self.D_S == 1:
                level = Conv2D(1, (1, 1), name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
                deconv = Conv_Block(Concat_Block(upConv_Block(deconv), convs_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
                deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            else:
                print("ERROR: Please Check the Values of the Input Parameters!")

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNet_v2(self):
        """Variable UNet Model Design - Version 2"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv

        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if (self.A_E == 0) and (self.D_S == 0):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif (self.A_E == 0) and (self.D_S == 1):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
            level0 = Conv2D(1, (1, 1), name=f'level{self.model_depth}')(conv)
            levels.append(level0)
        elif (self.A_E == 1) and (self.D_S == 0):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif (self.A_E == 1) and (self.D_S == 1):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
            level0 = Conv2D(1, (1, 1), name=f'level{self.model_depth}')(conv)
            levels.append(level0)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        deconv = trans_conv2D(conv, self.model_width, 2 ** (self.model_depth - 1))
        deconv = Conv_Block(Concat_Block(deconv, convs_list[self.model_depth - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - 1))
        deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - 1))

        for j in range(1, self.model_depth):
            if self.D_S == 0:
                deconv = trans_conv2D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
                deconv = Conv_Block(Concat_Block(deconv, convs_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
                deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            elif self.D_S == 1:
                level = Conv2D(1, (1, 1), name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
                deconv = trans_conv2D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
                deconv = Conv_Block(Concat_Block(deconv, convs_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
                deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            else:
                print("ERROR: Please Check the Values of the Input Parameters!")

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNetE(self):
        """Variable Ensemble UNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv
        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 0:
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = Conv2D(1, (1, 1), name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconvs = {}
        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(conv)), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(convs_list[j + 1])), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = Conv2D(1, (1, 1), name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNetE_v2(self):
        """Variable Ensemble UNet Model Design - Version 2"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv
        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 0:
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = Conv2D(1, (1, 1), name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconvs = {}
        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = trans_conv2D(conv, self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = trans_conv2D(convs_list[j + 1], self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = Conv2D(1, (1, 1), name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNetP(self):
        """Variable UNet+ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv
        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 0:
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconvs = {}
        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(conv)), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(convs_list[j + 1])), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv = Conv_Block(Concat_Block(deconvs["deconv%s%s" % (j, (i - 1))], upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = Conv2D(1, (1,1), name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNetP_v2(self):
        """Variable UNet+ Model Design - Version 2"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv
        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 0:
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconvs = {}
        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = trans_conv2D(conv, self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = trans_conv2D(convs_list[j + 1], self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(deconvs["deconv%s%s" % (j, (i - 1))], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = Conv2D(1, (1,1), name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNetPP(self):
        """Variable UNet++ Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv
        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 0:
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconvs = {}
        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(conv)), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = Conv_Block(Concat_Block(convs_list[j], upConv_Block(convs_list[j + 1])), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv_tot, upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = Conv2D(1, (1,1), name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def UNetPP_v2(self):
        """Variable UNet++ Model Design - Version 2"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, self.kernel_size, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** 0)
        pool = MaxPooling2D(pool_size=(2,2))(conv)
        convs["conv%s" % i] = conv
        for i in range(2, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 0:
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        elif self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            conv = Conv_Block(latent, self.model_width, self.kernel_size, 2 ** self.model_depth)
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(convs_list[0])
            levels.append(level)

        deconvs = {}
        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if (i == 1) and (j == (self.model_depth - 1)):
                    deconv = trans_conv2D(conv, self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif (i == 1) and (j < (self.model_depth - 1)):
                    deconv = trans_conv2D(convs_list[j + 1], self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    deconv = trans_conv2D(deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2 ** j)
                    deconv = Conv_Block(Concat_Block(convs_list[j], deconv_tot, deconv), self.model_width, self.kernel_size, 2 ** j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = Conv2D(1, (1,1), name=f'level{self.model_depth - i}')(deconvs["deconv%s%s" % (j, i)])
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def MultiResUNet(self):
        ''' 1D MultiResUNet with an option for Deep Supervision and/or being used as an AutoEncoder '''
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        mresblocks = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        mresblock = MultiResBlock(inputs, self.model_width, self.kernel_size, 2 ** 0, self.alpha)
        pool = MaxPooling2D(pool_size=(2,2))(mresblock)
        mresblocks["mres%s" % i] = ResPath(mresblock, self.model_depth, self.model_width, self.kernel_size, 2 ** 0)

        for i in range(2, (self.model_depth + 1)):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.alpha)
            pool = MaxPooling2D(pool_size=(2,2))(mresblock)
            mresblocks["mres%s" % i] = ResPath(mresblock, (self.model_depth- i + 1), self.model_width, self.kernel_size, 2 ** (i - 1))

        # Collect Latent Features or Embeddings from AutoEncoders
        if (self.A_E == 0) and (self.D_S == 0):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
        elif (self.A_E == 0) and (self.D_S == 1):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(mresblock)
            levels.append(level)
        elif (self.A_E == 1) and (self.D_S == 0):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            mresblock = MultiResBlock(latent, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
        elif (self.A_E == 1) and (self.D_S == 1):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            mresblock = MultiResBlock(latent, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(mresblock)
            levels.append(level)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding

        mresblocks_list = list(mresblocks.values())
        deconv = MultiResBlock(Concat_Block(upConv_Block(mresblock), mresblocks_list[self.model_depth - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - 1), self.alpha)

        for j in range(1, self.model_depth):
            if self.D_S == 0:
                deconv = MultiResBlock(Concat_Block(upConv_Block(deconv), mresblocks_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha)
            elif self.D_S == 1:
                level = Conv2D(1, (1,1), name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
                deconv = MultiResBlock(Concat_Block(upConv_Block(deconv), mresblocks_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha)
            else:
                print("ERROR: Please Check the Values of the Input Parameters!")

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model


    def MultiResUNet_v2(self):
        ''' 1D MultiResUNet with an option for Deep Supervision and/or being used as an AutoEncoder - Version 2'''
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        mresblocks = {}
        levels = []
        i = 1

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        mresblock = MultiResBlock(inputs, self.model_width, self.kernel_size, 2 ** 0, self.alpha)
        pool = MaxPooling2D(pool_size=(2,2))(mresblock)
        mresblocks["mres%s" % i] = ResPath(mresblock, self.model_depth, self.model_width, self.kernel_size, 2 ** 0)

        for i in range(2, (self.model_depth + 1)):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** (i - 1), self.alpha)
            pool = MaxPooling2D(pool_size=(2,2))(mresblock)
            mresblocks["mres%s" % i] = ResPath(mresblock, (self.model_depth- i + 1), self.model_width, self.kernel_size, 2 ** (i - 1))

        # Collect Latent Features or Embeddings from AutoEncoders
        if (self.A_E == 0) and (self.D_S == 0):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
        elif (self.A_E == 0) and (self.D_S == 1):
            mresblock = MultiResBlock(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(mresblock)
            levels.append(level)
        elif (self.A_E == 1) and (self.D_S == 0):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            mresblock = MultiResBlock(latent, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
        elif (self.A_E == 1) and (self.D_S == 1):
            latent = Feature_Extraction_Block(pool, self.model_width, int(self.length / (2 ** self.model_depth)), self.feature_number)
            mresblock = MultiResBlock(latent, self.model_width, self.kernel_size, 2 ** self.model_depth, self.alpha)
            level = Conv2D(1, (1,1), name=f'level{self.model_depth}')(mresblock)
            levels.append(level)
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

        # Decoding

        mresblocks_list = list(mresblocks.values())
        deconv = MultiResBlock(Concat_Block(trans_conv2D(mresblock, self.model_width, 2 ** (self.model_depth - 1)), mresblocks_list[self.model_depth - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - 1), self.alpha)

        for j in range(1, self.model_depth):
            if self.D_S == 0:
                deconv = MultiResBlock(Concat_Block(trans_conv2D(deconv, self.model_width, 2 ** (self.model_depth - j - 1)), mresblocks_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha)
            elif self.D_S == 1:
                level = Conv2D(1, (1,1), name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
                deconv = MultiResBlock(Concat_Block(trans_conv2D(deconv, self.model_width, 2 ** (self.model_depth - j - 1)), mresblocks_list[self.model_depth - j - 1]), self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1), self.alpha)
            else:
                print("ERROR: Please Check the Values of the Input Parameters!")

        # Output
        if self.problem_type == 'Classification':
            outputs = Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model
