import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Input, TimeDistributed, Masking
from tensorflow.keras import Model
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.initializers import he_normal, RandomUniform

# from keras import backend as K

"""
To use the model call it somethig like this...from 
/home/nm13850/Documents/PhD/Python/learning_new_functions/CNN_sim_script/conv_march_2019/conv_tutorial3/train_vgg.py

"""

"""
Bowers2014 replication
inputs = 31
hidden = 200
outputs = 30/300
Simultaneous recall of all items
"""

class Bowers14rnn:
    @staticmethod
    def build(features, classes, timesteps, batch_size, n_layers=1, units_per_layer=200,
              serial_recall=True, act_func='sigmoid', y_1hot=False, dropout=0.0,
              weight_init='GlorotUniform', unroll=False):
        """
        :param features: input shape, which is n_letters (30) + 1, for end_of_seq_cue.
        :param classes: Vocab size (either 30 or 300)
        :param timesteps: or seq_len.  Use 1, 3, 5, 7
        :param units_per_layer: 200
        :param act_func: Jeff Used Sigmoids, typically SimpleRNN uses Tanh
        :param y_1hot: If output is 1hot/softmax
        :param dropout: Not used

        :return:
        """
        model = Sequential(name="Bowers14rnn")

        model.add(SimpleRNN(units=units_per_layer,

                            # input_shape=(timesteps, features),


                            batch_input_shape=(batch_size, timesteps, features),

                            return_sequences=serial_recall,  # this stops it from giving an output after each item


                            # it allows the model to learn from activations at all timesteps not just the final one.
                            # this also allows truncated backprop thru time.
                            # stateful=True,
                            activation=act_func, dropout=dropout, name="hid0",

                            unroll=unroll))

        if y_1hot:
            model.add(Dense(classes, name='output', activation='softmax'))
        else:
            # dist output classifier
            model.add(Dense(classes, name='output', activation='sigmoid'))

        return model


class Bowers_14_Elman:
    @staticmethod
    def build(features, classes, timesteps, batch_size, n_layers=1, units_per_layer=200,
              serial_recall=True, act_func='sigmoid', y_1hot=False, dropout=0.0,
              masking=False,
              weight_init='GlorotUniform', unroll=False, stateful=False):
        """

        I'm not sure how just adding a time distributed layer makes this an elman network, but that is implied here:
        https://ahstat.github.io/RNN-Keras-understanding-computations/
        https://github.com/mesnilgr/is13/blob/master/examples/elman-keras.py
        https://github.com/keras-team/keras/issues/11403


        :param features: input shape, which is n_letters (30) + 1, for end_of_seq_cue.
        :param classes: Vocab size (either 30 or 300)
        :param timesteps: or seq_len.  Use 1, 3, 5, 7
        :param units_per_layer: 200
        :param act_func: Jeff Used Sigmoids, typically SimpleRNN uses Tanh
        :param y_1hot: If output is 1hot/softmax
        :param dropout: Not used

        :return:
        """
        model = Sequential(name="Bowers_14_Elman")

        # layer_seqs = True
        # l_input_width = units_per_layer

        if weight_init == 'LENS':
            weight_init = RandomUniform(minval=-1.0, maxval=1.0, seed=None)


        # if masking:
        #     model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        #
        # for layer in range(n_layers):
        #     if layer == 0:  # first layer
        #         l_input_width = features
        #
        #     if layer == n_layers - 1:  # last layer
        #         layer_seqs = serial_recall
        #
        #     # # add concatenate function here for context and input being added together.
        #     # # features + units_per_layer
        #
        #     model.add(SimpleRNN(units=units_per_layer,
        #
        #                         kernel_initializer=weight_init,
        #
        #                         # chenged batch_input_shape to input_shape on 07122019 trying to
        #                         # get the model to run with variable inpjut length.
        #                         # Also tried with both"""
        #                         # input_shape=(timesteps, l_input_width),
        #                         batch_input_shape=(batch_size, timesteps, l_input_width),
        #
        #                         return_sequences=layer_seqs,  # this stops it from giving an output after each item
        #
        #                         # # stateful allows the model to learn from all timesteps
        #                         # # not just previous one.  also allows truncated backprop thru time.
        #                         stateful=stateful,
        #
        #                         activation=act_func, dropout=dropout, name=f"hid{layer}",
        #
        #                         unroll=unroll))

        # # somehow the the output from hid layer needs to go to output and context.
        # # weights from hidden to context should be fixed and 1, and single connections not fc

        # model.add(TimeDistributed(Dense(units=units_per_layer, name='context',
        #                                 activation=act_func,
        #                                 # # no need for input shape if used after first layer
        #                                 ),
        #           input_shape=(batch_size, timesteps, units_per_layer)))

        model.add(SimpleRNN(batch_input_shape=(batch_size, timesteps, features),
                            input_shape=(timesteps, features),
                            return_sequences=False, unroll=True,
                            units=units_per_layer))
        model.add(TimeDistributed(Dense(activation='sigmoid', units=units_per_layer,
                                        input_shape=(None, 1, 200))))

        '''i give up'''

        # # somehow the output of the context layer needs to go back into the hidden layer.
        # # These weights are learnable and fully connected.

        # if y_1hot:
        #     model.add(Dense(classes, name='output', activation='softmax'))
        # else:
        # dist output classifier
        model.add(Dense(classes, name='output', activation='sigmoid'))

        return model


class Bowers_14_Elman2:
    @staticmethod
    def build(features, classes, timesteps, batch_size, n_layers=1, units_per_layer=200,
              serial_recall=True, act_func='sigmoid', y_1hot=False, dropout=0.0,
              masking=False,
              weight_init='GlorotUniform', unroll=False, stateful=False):
        """

        I'm not sure how just adding a time distributed layer makes this an elman network, but that is implied here:
        https://ahstat.github.io/RNN-Keras-understanding-computations/
        https://github.com/mesnilgr/is13/blob/master/examples/elman-keras.py
        https://github.com/keras-team/keras/issues/11403.

        However, I am going to try to build a version myself.


        :param features: input shape, which is n_letters (30) + 1, for end_of_seq_cue.
        :param classes: Vocab size (either 30 or 300)
        :param timesteps: or seq_len.  Use 1, 3, 5, 7
        :param units_per_layer: 200
        :param act_func: Jeff Used Sigmoids, typically SimpleRNN uses Tanh
        :param y_1hot: If output is 1hot/softmax
        :param dropout: Not used

        :return:
        """
        # model = Sequential(name="Bowers_14_Elman")

        if weight_init == 'LENS':
            weight_init = RandomUniform(minval=-1.0, maxval=1.0, seed=None)

        data_input = Input(shape=(batch_size, timesteps, features), name='data_input')

        input_cntxt_width = features + units_per_layer

        hidden_input = Input(shape=(timesteps, input_cntxt_width), name="con_input")

        # # add concatenate function here for context and input being added together.
        # # features + units_per_layer

        hidden = SimpleRNN(units=units_per_layer,

                           kernel_initializer=weight_init,

                           # chenged batch_input_shape to input_shape on 07122019 trying to
                           # get the model to run with variable inpjut length.
                           # Also tried with both"""
                           input_shape=(timesteps, ),
                           batch_input_shape=(batch_size, timesteps, l_input_width),

                           return_sequences=serial_recall,  # this stops it from giving an output after each item

                           # # stateful allows the model to learn from all timesteps
                           # # not just previous one.  also allows truncated backprop thru time.
                           stateful=stateful,

                           activation=act_func, dropout=dropout, name=f"hid{layer}",

                           unroll=unroll)

        # # somehow the the output from hid layer needs to go to output and context.
        # # weights from hidden to context should be fixed and 1, and single connections not fc

        model.add(TimeDistributed(Dense(units=units_per_layer, name='context',
                                        activation=act_func,
                                        # # no need for input shape if used after first layer
                                        )))

        # # somehow the output of the context layer needs to go back into the hidden layer.
        # # These weights are learnable and fully connected.

        if y_1hot:
            model.add(Dense(classes, name='output', activation='softmax'))
        else:
            # dist output classifier
            model.add(Dense(classes, name='output', activation='sigmoid'))

        return model

class SimpleRNNn:
    @staticmethod
    def build(features, classes, timesteps, batch_size, n_layers=1, units_per_layer=200,
              serial_recall=True, act_func='tanh', y_1hot=False, dropout=0.0,
              masking=False,
              weight_init='GlorotUniform', unroll=False, stateful=False):
        """
        :param features: input shape, which is n_letters (30) + 1, for end_of_seq_cue.
        :param classes: Vocab size (either 30 or 300)
        :param timesteps: or seq_len.  Use 1, 3, 5, 7
        :param batch_size: number of sequences inputed in a batch
        :param n_layers: number of hidden layers
        :param units_per_layer: 200
        :param serial_recall: predict series or single item
        :param act_func: Jeff Used Sigmoids, typically SimpleRNN uses Tanh
        :param y_1hot: If output is 1hot/softmax
        :param dropout: Not used
        :param masking: Whether to use a masking layer for variable seq len
        :param weight_init: weight initialization
        :param unroll: uses more memory but trains faster
        :param stateful: stateful RNN does not reset states after each sequence.
            I need to set this to True in order to set the state to a given value at the start of
            a sequences as in Bowers 14.

        :return: model
        """
        model = Sequential(name="SimpleRNNn")

        if weight_init == 'LENS':
            weight_init = RandomUniform(minval=-1.0, maxval=1.0, seed=None)

        layer_seqs = True
        l_input_width = units_per_layer

        if masking:
            model.add(Masking(mask_value=0., input_shape=(timesteps, features)))

        for layer in range(n_layers):
            if layer == 0:  # first layer
                l_input_width = features

            if layer == n_layers-1:  # last layer
                layer_seqs = serial_recall

            model.add(SimpleRNN(units=units_per_layer,

                                kernel_initializer=weight_init,

                                # chenged batch_input_shape to input_shape on 07122019 trying to
                                # get the model to run with variable inpjut length.
                                # Also tried with both"""
                                input_shape=(timesteps, l_input_width),
                                batch_input_shape=(batch_size, timesteps, l_input_width),

                                return_sequences=layer_seqs,  # False stops it from giving an output after each item

                                # # stateful allows the model to learn from all timesteps
                                # # not just previous one.  also allows truncated backprop thru time.
                                stateful=stateful,

                                activation=act_func, dropout=dropout, name=f"hid{layer}",

                                unroll=unroll))

        if y_1hot:
            model.add(Dense(classes, name='output', activation='softmax'))
        else:
            # dist output classifier
            model.add(Dense(classes, name='output', activation='sigmoid', kernel_initializer=weight_init,))

        return model


class GRUn:
    @staticmethod
    def build(features, classes, timesteps, batch_size, n_layers=1, units_per_layer=200,
              serial_recall=True, act_func='tanh', y_1hot=False, dropout=0.0,
              masking=False,
              weight_init='glorot_uniform', unroll=False, stateful=False):
        """
        :param features: input shape, which is n_letters (30) + 1, for end_of_seq_cue.
        :param classes: Vocab size (either 30 or 300)
        :param timesteps: or seq_len.  Use 1, 3, 5, 7
        :param batch_size: number of sequences inputed in a batch
        :param n_layers: number of hidden layers
        :param units_per_layer: 200
        :param serial_recall: predict series or single item
        :param act_func: Jeff Used Sigmoids, typically SimpleRNN uses Tanh
        :param y_1hot: If output is 1hot/softmax
        :param dropout: Not used
        :param masking: Whether to use a masking layer for variable seq len
        :param weight_init: weight initialization
        :param unroll: uses more memory but trains faster
        :param stateful: stateful RNN does not reset states after each sequence.
            I need to set this to True in order to set the state to a given value at the start of
            a sequences as in Bowers 14.

        :return: model
        """
        model = Sequential(layers=n_layers, name="GRUn")

        layer_seqs = True
        l_input_width = units_per_layer

        if masking:
            model.add(Masking(mask_value=0., input_shape=(timesteps, features)))

        for layer in range(n_layers):
            if layer == 0:  # first layer
                l_input_width = features

            if layer == n_layers - 1:  # last layer
                layer_seqs = serial_recall

            model.add(GRU(units=units_per_layer,
                          batch_input_shape=(batch_size, timesteps, l_input_width),
                          kernel_initializer=weight_init,

                          return_sequences=layer_seqs,  # this stops it from giving an output after each item
                          # # stateful allows the model to learn from all timesteps
                          # # not just previous one.  also allows truncated backprop thru time.
                          # stateful=True,
                          activation=act_func, dropout=dropout, name=f"hid{layer}",

                          unroll=unroll))

        if y_1hot:
            model.add(Dense(classes, name='output', activation='softmax'))
        else:
            # dist output classifier
            model.add(Dense(classes, name='output', activation='sigmoid'))

        return model




class LSTMn:
    @staticmethod
    def build(features, classes, timesteps, batch_size, n_layers=1, units_per_layer=200,
              serial_recall=True, act_func='tanh', y_1hot=False, dropout=0.0,
              masking=False,
              weight_init='glorot_uniform', unroll=False, stateful=False):
        """
        :param features: input shape, which is n_letters (30) + 1, for end_of_seq_cue.
        :param classes: Vocab size (either 30 or 300)
        :param timesteps: or seq_len.  Use 1, 3, 5, 7
        :param batch_size: number of sequences inputed in a batch
        :param n_layers: number of hidden layers
        :param units_per_layer: 200
        :param serial_recall: predict series or single item
        :param act_func: Jeff Used Sigmoids, typically SimpleRNN uses Tanh
        :param y_1hot: If output is 1hot/softmax
        :param dropout: Not used
        :param masking: Whether to use a masking layer for variable seq len
        :param weight_init: weight initialization
        :param unroll: uses more memory but trains faster
        :param stateful: stateful RNN does not reset states after each sequence.
            I need to set this to True in order to set the state to a given value at the start of
            a sequences as in Bowers 14.

        :return: model
        """
        model = Sequential(name="LSTMn")
        # model = Sequential(layers=n_layers, name="LSTMn")
        # model = tf.keras.models.Sequential(layers=n_layers, name="LSTMn")

        if masking:
            model.add(Masking(mask_value=0., input_shape=(timesteps, features)))

        layer_seqs = True
        l_input_width = units_per_layer

        for layer in range(n_layers):
            if layer == 0:  # first layer
                l_input_width = features

            if layer == n_layers - 1:  # last layer
                layer_seqs = serial_recall

            model.add(LSTM(units=units_per_layer,

                           # input_shape=(timesteps, l_input_width),

                           batch_input_shape=(batch_size, timesteps, l_input_width),

                           kernel_initializer=weight_init,

                           return_sequences=layer_seqs,  # this stops it from giving an output after each item
                           # # stateful allows the model to learn from all timesteps
                           # # not just previous one.  also allows truncated backprop thru time.
                           stateful=stateful,
                           activation=act_func, dropout=dropout, name=f"hid{layer}",

                           unroll=unroll))

        if y_1hot:
            model.add(Dense(classes, name='output', activation='softmax'))
        else:
            # dist output classifier
            model.add(Dense(classes, name='output', activation='sigmoid'))

        return model




class Seq2Seq:
    @staticmethod
    def build(x_size, n_classes, n_units=128):
        """
        https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
        # returns train, inference_encoder and inference_decoder models
        # n_units is the same in encoder and decoder models

        :param x_size:
        :param n_classes:
        :param n_units:
        :return:
        """

        # define training encoder
        encoder_inputs = Input(shape=(None, x_size))
        encoder = LSTM(n_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        # define training decoder
        decoder_inputs = Input(shape=(None, n_classes))
        decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(n_classes, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = Model(encoder_inputs, encoder_states)

        # define inference decoder
        decoder_state_input_h = Input(shape=(n_units,))
        decoder_state_input_c = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

