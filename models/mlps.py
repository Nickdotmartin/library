import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dense, Dropout, GaussianNoise
from tensorflow.keras.initializers import RandomUniform

# from keras import backend as K


class mlp:
    @staticmethod
    def build(features, classes, n_layers=1, units_per_layer=200,
              act_func='relu', y_1hot=True, weight_init='GlorotUniform',
              use_bias=True, dropout=0.0, batch_norm=True, output_act='softmax'):


        if weight_init == 'taxo':
            weight_init = RandomUniform(minval=-0.01, maxval=0.01, seed=None)

        model = Sequential(name=f"h{n_layers}_u{units_per_layer}")

        l_input_width = units_per_layer

        for layer in range(n_layers):
            if layer == 0:  # first layer
                l_input_width = features
                batch_norm = False
                dropout = False

            if batch_norm is True:
                model.add(BatchNormalization(name=f'bn_{layer}'))

            if dropout > 0:
                model.add(Dropout(rate=dropout, name=f'dropout_{layer}'))

            '''
            # as first layer in a sequential model:
            # model = Sequential()
            # model.add(Dense(32, input_shape=(16,)))
            # now the model will take as input arrays of shape (*, 16)
            # and output arrays of shape (*, 32)
            # after the first layer, you don't need to specify
            # the size of the input anymore:
            # model.add(Dense(32))
            '''

            model.add(Dense(units=units_per_layer,
                            input_shape=(l_input_width, ),
                            use_bias=use_bias,
                            kernel_initializer=weight_init,
                            activation=act_func,
                            name=f"hid{layer}",))

        if y_1hot:
            model.add(Dense(classes, name='output', activation='softmax',
                            use_bias=use_bias,
                            kernel_initializer=weight_init,
                            ))
        else:
            # dist output classifier
            model.add(Dense(classes, name='output', activation=output_act,
                            use_bias=use_bias,
                            kernel_initializer=weight_init,
                            ))

        return model




class fc1:
    @staticmethod
    def build(classes, units_per_layer, batch_norm=True, dropout=True):
        model = Sequential()

        model.add(Dense(units_per_layer, activation='relu', name='fc_1'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, name='output'))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model



class fc2:
    @staticmethod
    def build(classes, units_per_layer, batch_norm=True, dropout=True):
        model = Sequential()

        model.add(Dense(units_per_layer, activation='relu', name='fc_1'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        model.add(Dense(units_per_layer, activation='relu', name='fc_2'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, name='output'))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


class fc4:
    @staticmethod
    def build(classes, units_per_layer, batch_norm=True, dropout=True):
        model = Sequential()

        model.add(Dense(units_per_layer, activation='relu', name='fc_1'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        model.add(Dense(units_per_layer, activation='relu', name='fc_2'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        model.add(Dense(units_per_layer, activation='relu', name='fc_3'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        model.add(Dense(units_per_layer, activation='relu', name='fc_4'))
        if batch_norm is True:
            model.add(BatchNormalization())
        if dropout is True:
            model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, name='output'))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
























