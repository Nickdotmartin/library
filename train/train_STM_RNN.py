import csv
import datetime
import json
import os
# import git
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export

# import six
# from six.moves import zip # used in tf.keras optimizers github so probably not needed here
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.data import load_x_data, load_y_data, find_path_to_dir, switch_home_dirs
from tools.network import get_model_dict, get_scores
from tools.RNN_STM import generate_STM_RNN_seqs, get_label_seqs, get_test_scores, free_rec_acc
from models.rnns import Bowers14rnn, SimpleRNNn, GRUn, LSTMn, Seq2Seq


'''
Get this script working, then turn it into a callable function

backprop through time*****************************************

generator for data: 
- from list of y-label seqs
- for each seq
- generate X data from vocab_dict
- generate Y data 

various datasets (vocab, seq len, rpt)

models (rnn, GRU, LSTM, seq2seq)

'''

# def mean_IoU(y_true, y_pred):
#     mean_IoU_acc = free_rec_acc(y_true=K.eval(y_true), y_pred=K.eval(y_pred), get_prop_corr=False)
#     return mean_IoU_acc
#
# def prop_corr(y_true, y_pred):
#     print(f"eagerly? {tf.executing_eagerly()}")
#     print(f"numpy: {y_pred.numpy()}")
#
#     prop_corr_acc = free_rec_acc(y_true=K.eval(y_true), y_pred=K.eval(y_pred), get_prop_corr=True)
#     return K.variable(prop_corr_acc)
#
#
# tf.enable_eager_execution()

# # custom optimizer from LENS
class dougsMomentum(optimizer_v2.OptimizerV2):
    """Default LENS optimizer
        http://tedlab.mit.edu/~dr/Lens/Commands/dougsMomentum.html

        based on
        https://medium.com/@mlguy/adding-custom-loss-and-optimizer-in-keras-e255764e1b7d

        adapted code from
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizers.py


        Has a momentum term similar to SGD, but previous update clipped to 1.
        instead of:
              v(t+1) = momentum * v(t) - learning_rate * gradient
        uses:
              v(t+1) = momentum * K.clip(v(t), min_value=None, max_value=1.0) - learning_rate * gradient

        then updates with
          theta(t+1) = theta(t) + v(t+1)

        I've had some problems with it not recognizing self.lr, so going to use self.learn_rate
    """

    def __init__(self,
                   learning_rate=0.01,
                   momentum=0.0,
                   nesterov=False,
                   name="dougsMomentum",
                   **kwargs):
        """        # # initialize dougsMomentum optimizer
        """
        super(dougsMomentum, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(dougsMomentum, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
                        self._get_hyper("momentum", var_dtype))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # print(f"orig var: {var}")
        K.set_value(var, K.clip(var, min_value=-1.0, max_value=1.0))
        # print(f"set var: {var}")

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))




        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return training_ops.resource_apply_keras_momentum(
                var.handle,
                momentum_var.handle,
                coefficients["lr_t"],
                grad,
                coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov)
        else:
            return training_ops.resource_apply_gradient_descent(
              var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                                   **kwargs):
        if self._momentum:
            return super(dougsMomentum, self)._resource_apply_sparse_duplicate_indices(
            grad, var, indices, **kwargs)
        else:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                              or self._fallback_apply_state(var_device, var_dtype))

        return resource_variable_ops.resource_scatter_add(
            var.handle, indices, -grad * coefficients["lr_t"])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        momentum_var = self.get_slot(var, "momentum")
        return training_ops.resource_sparse_apply_keras_momentum(
            var.handle,
            momentum_var.handle,
            coefficients["lr_t"],
            grad,
            indices,
            coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(dougsMomentum, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config




def train_model(exp_name,
                data_dict_path,
                model_path,
                cond_name=None,
                cond=None, run=None,
                hid_layers=1,
                units_per_layer=200,
                act_func='sigmoid',
                serial_recall=False,
                generator=True,
                x_data_type='dist_letter_X',
                end_seq_cue=True,
                max_epochs=100, use_optimizer='adam',
                loss_target=0.01, min_loss_change=0.0001, batch_size=32,
                augmentation=True, grey_image=False,
                use_batch_norm=True, use_dropout=0.0,
                use_val_data=True,
                timesteps=1,
                weight_init='GlorotUniform',
                lr=0.001,
                unroll=False,
                LENS_states=True,
                exp_root='/home/nm13850/Documents/PhD/python_v2/experiments/',
                verbose=False,
                test_run=False
                ):

    """
    script to train a recurrent neural network on a Short-term memory task.


    1. get details - dset_name, model, other key variables
    2. load datasets
    3. compile model
    4. fit/evaluate model
    5. make plots
    6. output:
        training plots
        model accuracy etc.

    this model will stop training if
    a. loss does not improve for [patience] epochs by [min_loss_change]
    b. accuracy reaches 100%
    c. when max_epochs is reached.


    :param exp_name: name of this experiemnt or set of model/dataset pairs
    :param data_dict_path: path to data dict
    :param model_path: dir for model
    :param cond_name: name of this condition
    :param cond: number for this condition
    :param run: number for this run (e.g., multiple runs of same conditin from different initializations)
    :param hid_layers: number of hidden layers
    :param units_per_layer: number of units in each layer
    :param act_func: activation function to use
    :param serial_recall: if True, output a sequence, else use dist output and no repeats in data.
    :param generator: If true, will generate training data, else load data as usual.
    :param x_data_type: input coding: local words (1hot), local letters (3hot), dist letters (9hot)
    :param end_seq_cue: Add input unit to cue recall
    :param max_epochs: Stop training after this many epochs
    :param use_optimizer: Optimizer to use
    :param loss_target: stop training when this target is reached
    :param min_loss_change: stop training of loss does not improve by this much
    :param batch_size: number of items loaded in at once
    :param augmentation: whether data aug is used (for images)
    :param grey_image: whether the images are grey (if false, they are colour)
    :param use_batch_norm: use batch normalization
    :param use_dropout: use dropout
    :param use_val_data: use validation set (either separate set, or train/val split)
    :param timesteps: if RNN length of sequence
    :param weight_init: change the initializatation of the weights
    :param lr: set the learning rate for the optimizer
    :param unroll:  Whether to unroll the model.
    :param exp_root: root directory for saving experiments

    :param verbose: if 0, not verbose; if 1 - print basics; if 2, print all

    :return: training_info csv
    :return: sim_dict with dataset info, model info and training info

    """

    print("\n\n\nTraining a new model\n********************")

    dset_dir, data_dict_name = os.path.split(data_dict_path)
    dset_dir, dset_name = os.path.split(dset_dir)
    model_dir, model_name = os.path.split(model_path)

    print(f"dset_dir: {dset_dir}\ndset_name: {dset_name}")
    print(f"model_dir: {model_dir}\nmodel_name: {model_name}")

    # Output files
    if not cond_name:
        # output_filename = f"{exp_name}_{model_name}_{dset_name}"
        output_filename = f"{model_name}_{dset_name}"

    else:
        # output_filename = f"{exp_name}_{cond_name}"
        output_filename = cond_name

    print(f"\noutput_filename: {output_filename}")


    # # get info from dict
    if os.path.isfile(data_dict_path):
        data_dict = load_dict(data_dict_path)
    elif os.path.isfile(os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/', data_dict_path)):
        # work computer
        data_dict_path = os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/', data_dict_path)
        data_dict = load_dict(data_dict_path)
    elif os.path.isfile(os.path.join('/Users/nickmartin/Documents/PhD/python_v2/datasets', data_dict_path)):
        # laptop
        data_dict_path = os.path.join('/Users/nickmartin/Documents/PhD/python_v2/datasets', data_dict_path)
        data_dict = load_dict(data_dict_path)
    else:
        raise FileNotFoundError(data_dict_path)

    if verbose:
        # # print basic details
        print("\n**** STUDY DETAILS ****")
        print(f"output_filename: {output_filename}\ndset_name: {dset_name}\nmodel: {model_name}\n"
              f"max_epochs: {max_epochs}\nuse_optimizer: {use_optimizer}\n"
              f"lr: {lr}\n"
              f"loss_target: {loss_target}\nmin_loss_change: {min_loss_change}\n"
              f"batch_norm: {use_batch_norm}\nval_data: {use_val_data}\naugemntation: {augmentation}\n")
        focussed_dict_print(data_dict, 'data_dict')

    n_cats = data_dict["n_cats"]
    x_size = data_dict['X_size']
    if end_seq_cue:
        x_size = x_size + 1


    if not generator:
        x_load = np.load('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep/'
                         'test_dist_letter_X_10batches_4seqs_3ts_31feat.npy')
        # print(f"x_load: {np.shape(x_load)}")
        y_load = np.load('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep/'
                         'test_freerecall_Y_10batches_4seqs_3ts.npy')
        # print(f"y_load: {np.shape(y_load)}")
        labels_load = np.load('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep/'
                              'test_freerecall_Y_labels_10batches_4seqs_3ts.npy')
        # print(f"labels_load: {np.shape(labels_load)}")
        x_train = x_load
        y_train = y_load

        n_items = np.shape(x_train)[0]
    else:
        # # if generator is true
        x_data_path = 'RNN_STM_tools/generate_STM_RNN_seqs'
        y_data_path = 'RNN_STM_tools/generate_STM_RNN_seqs'
        n_items = 'unknown'


    # # save path
    exp_cond_path = os.path.join(exp_root, exp_name, output_filename)

    train_folder = 'training'

    exp_cond_path = os.path.join(exp_cond_path, train_folder)

    if not os.path.exists(exp_cond_path):
        os.makedirs(exp_cond_path)
    os.chdir(exp_cond_path)
    print(f"\nsaving to exp_cond_path: {exp_cond_path}")


    # # The Model
    if 'rnn' in model_dir:
        print("loading a recurrent model")
        augmentation = False

        stateful = False
        if LENS_states != False:
            stateful = True

        models_dict = {'Bowers14rnn': Bowers14rnn,
                       'SimpleRNNn': SimpleRNNn,
                       'GRUn': GRUn,
                       'LSTMn': LSTMn,
                       'Seq2Seq': Seq2Seq}

        model = models_dict[model_name].build(features=x_size, classes=n_cats, timesteps=timesteps,
                                              batch_size=batch_size, n_layers=hid_layers,
                                              serial_recall=serial_recall,
                                              units_per_layer=units_per_layer, act_func=act_func,
                                              y_1hot=serial_recall,
                                              dropout=use_dropout, weight_init=weight_init,
                                              unroll=unroll,
                                              stateful=stateful)
    else:
        print("model_dir not recognised")


    # loss
    loss_func = 'binary_crossentropy'
    if serial_recall:
        loss_func = 'categorical_crossentropy'


    # optimizer
    sgd = SGD(lr=lr, momentum=.9)  # decay=sgd_lr / max_epochs)
    this_optimizer = sgd

    if use_optimizer == 'SGD_Nesterov':
        this_optimizer = SGD(lr=lr, momentum=.1, nesterov=True)  # decay=sgd_lr / max_epochs)

    elif use_optimizer == 'SGD_mom_clip':
        this_optimizer = SGD(lr=lr, momentum=.9, clipnorm=1.)  # decay=sgd_lr / max_epochs)

    elif use_optimizer == 'dougs':
        this_optimizer = dougsMomentum(lr=lr, momentum=.9)
        # this_optimizer = dougsMomentum()

    elif use_optimizer == 'adam':
        this_optimizer = Adam(lr=lr, amsgrad=True)

    elif use_optimizer == 'RMSprop':
        this_optimizer = RMSprop(lr=lr)

    elif use_optimizer == 'Adagrad':
        this_optimizer = Adagrad()

    elif use_optimizer == 'Adadelta':
        this_optimizer = Adadelta()

    elif use_optimizer == 'Adamax':
        this_optimizer = Adamax(lr=lr)

    elif use_optimizer == 'Nadam':
        this_optimizer = Nadam()



    # # metrics
    main_metric = 'binary_accuracy'
    if not serial_recall:
        main_metric = 'categorical_accuracy'
        # mean_IoU = free_rec_acc()
        # prop_corr_acc = prop_corr(y_true, y_pred)
        # iou_acc = mean_IoU(y_true, y_pred)

        # # compile model
        model.compile(loss=loss_func, optimizer=this_optimizer,
                      metrics=[main_metric])

    else:
        # # compile model
        model.compile(loss=loss_func, optimizer=this_optimizer,
                      metrics=[main_metric])

    optimizer_details = model.optimizer.get_config()
    # print_nested_round_floats(model_details)
    focussed_dict_print(optimizer_details, 'optimizer_details')

    # # check states
    # model.layers[0](initial_state=)
    # print(f"model states\n{states}")

    # for state in states:
    #     # if tf.__version__[0] == '2':
    #     #     print(state.numpy())
    #     # else:
    #     print(K.get_value(state))


    # # get model dict
    model_info = get_model_dict(model)  # , verbose=True)
    # print("\nmodel_info:")
    print_nested_round_floats(model_info, 'model_info')
    tf.compat.v1.keras.utils.plot_model(model, to_file=f'{model_name}_diag.png', show_shapes=True)

    # # call backs and training parameters
    checkpoint_path = f'{output_filename}_model.hdf5'

    checkpoint_mon = 'loss'
    if use_val_data:
        checkpoint_mon = 'val_loss'

    # checkpointing.  Save model and weights with best val loss.
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor=checkpoint_mon, verbose=1,
                                                      save_best_only=True, save_weights_only=False, mode='auto',
                                                      load_weights_on_restart=True)

    # patience_for_loss_change: wait this long to see if loss improves
    # patience_for_loss_change = int(max_epochs / 50)

    # todo: change this back in neccesary
    patience_for_loss_change = int(max_epochs / 10)

    if patience_for_loss_change < 5:
        patience_for_loss_change = 5

    # early_stop_plateau - if there is no imporovement
    early_stop_plateau = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=min_loss_change,
                                                          patience=patience_for_loss_change,
                                                          verbose=1, mode='min')

    val_early_stop_plateau = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_loss_change,
                                                              patience=patience_for_loss_change, verbose=verbose,
                                                              mode='min')

    date_n_time = int(datetime.datetime.now().strftime("%Y%m%d%H%M"))
    tensorboard_path = os.path.join(exp_cond_path, 'tb', str(date_n_time))

    tensorboard = TensorBoard(log_dir=tensorboard_path)

    print('\n\nto access tensorboard, in terminal use\n'
          f'tensorboard --logdir={tensorboard_path}'
          '\nthen click link''')

    callbacks_list = [early_stop_plateau, checkpointer, tensorboard]
    val_callbacks_list = [val_early_stop_plateau, checkpointer, tensorboard]

    ############################
    # # train model
    print("\n**** TRAINING ****")
    if augmentation:
        # # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                 shear_range=0.1,  # set range for random shear (tilt image)
                                 zoom_range=0.1,  # set range for random zoom
                                 # horizontal_flip=True,
                                 fill_mode="nearest")

        if use_val_data:
            fit_model = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                            validation_data=(x_val, y_val),
                                            # steps_per_epoch=len(x_train) // batch_size,
                                            epochs=max_epochs, verbose=1, callbacks=val_callbacks_list)
        else:
            fit_model = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                            # steps_per_epoch=len(x_train) // batch_size,
                                            epochs=max_epochs, verbose=1, callbacks=callbacks_list)

    else:
        if use_val_data:
            fit_model = model.fit(x_train, y_train,
                                  validation_data=(x_val, y_val),
                                  epochs=max_epochs, batch_size=batch_size, verbose=1, callbacks=val_callbacks_list)
        elif not generator:
            print("Using loaded data")
            print(f"x: {np.shape(x_train)}, y: {np.shape(y_train)}")

            fit_model = model.fit(x_train, y_train,
                                  epochs=max_epochs, batch_size=batch_size, verbose=1, callbacks=callbacks_list)
        else:
            # # use generator
            print("Using data generator")

            generate_data = generate_STM_RNN_seqs(data_dict=data_dict,
                                                  seq_len=timesteps,
                                                  batch_size=batch_size,
                                                  serial_recall=serial_recall,
                                                  x_data_type=x_data_type,
                                                  end_seq_cue=end_seq_cue
                                                  )

            fit_model = model.fit_generator(generate_data,
                                            steps_per_epoch=100,
                                            epochs=max_epochs,
                                            callbacks=callbacks_list,
                                            shuffle=False)

    #########################################################
    print("\n**** TRAINING COMPLETE ****")

    print(f"\nModel name: {checkpoint_path}")

    # # plot the training loss and accuracy
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(fit_model.history[main_metric])
    if use_val_data:
        ax1.plot(fit_model.history['val_acc'])
    ax1.set_title(f'{main_metric} (top); loss (bottom)')
    ax1.set_ylabel(f'{main_metric}')
    ax1.set_xlabel('epoch')
    ax2.plot(fit_model.history['loss'])
    if use_val_data:
        ax2.plot(fit_model.history['val_loss'])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    fig.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(output_filename) + '_training.png')
    plt.close()


    # # get best epoch number
    if use_val_data:
        # print(fit_model.history['val_loss'])
        trained_for = int(fit_model.history['val_loss'].index(min(fit_model.history['val_loss'])))
        end_val_loss = float(fit_model.history['val_loss'][trained_for])
        end_val_acc = float(fit_model.history['val_acc'][trained_for])
    else:
        # print(fit_model.history['loss'])
        trained_for = int(fit_model.history['loss'].index(min(fit_model.history['loss'])))
        end_val_loss = np.nan
        end_val_acc = np.nan

    end_loss = float(fit_model.history['loss'][trained_for])
    end_acc = float(fit_model.history[main_metric][trained_for])
    print(f'\nTraining Info\nbest loss after {trained_for} epochs\n'
          f'end loss: {end_loss}\nend acc: {end_acc}\n')
    if use_val_data:
        print(f'end val loss: {end_val_loss}\nend val acc: {end_val_acc}')


    # # # # PART 3 get_scores() # # #
    """accuracy can be two things
    1. What proportion of all sequences are entirely correct
    2. what is the average proportion of each sequence that is correct
    e.g., might get none of the sequences correct but on average get 50% of each sequence correct
    """
    # if "test_label_seqs" in list(data_dict.keys()):
    #     if type(data_dict["test_label_seqs"]) is "numpy.ndarray":
    #         test_label_seqs = data_dict["test_label_seqs"]
    #     elif type(data_dict["test_label_seqs"]) is 'str':
    #         # if data_dict["test_label_seqs"][-3:] == 'csv':
    #             # load csv
    #         if data_dict["test_label_seqs"][-3:] == 'npy':
    #             test_label_seqs = np.load(data_dict["test_label_seqs"])
    # else:

    # #     # # get labels for 100 sequences
    # test_label_seqs = get_label_seqs(n_labels=n_cats, seq_len=timesteps,
    #                                  serial_recall=serial_recall, n_seqs=10*batch_size)

    # # load test label seqs
    data_path = data_dict['data_path']
    test_filename = f'seq{timesteps}_v{n_cats}_960_test_seq_labels.npy'
    test_seq_path = os.path.join(data_path, test_filename)

    if not os.path.isfile(test_seq_path):
        if os.path.isfile(switch_home_dirs(test_seq_path)):
            test_seq_path = switch_home_dirs(test_seq_path)
    # if os.path.isfile(test_seq_path):
    test_label_seqs = np.load(test_seq_path)


    # # call get test accracy(serial_recall,
    scores_dict = get_test_scores(model=model, data_dict=data_dict, test_label_seqs=test_label_seqs,
                                  serial_recall=serial_recall,
                                  end_seq_cue=end_seq_cue,
                                  batch_size=batch_size,
                                  verbose=verbose)

    mean_IoU = scores_dict['mean_IoU']
    prop_seq_corr = scores_dict['prop_seq_corr']


    trained_date = int(datetime.datetime.now().strftime("%y%m%d"))
    trained_time = int(datetime.datetime.now().strftime("%H%M"))
    model_info['overview'] = {'model_type': model_dir, 'model_name': model_name,
                              "trained_model": checkpoint_path,
                              "hid_layers": hid_layers,
                              "units_per_layer": units_per_layer,
                              'act_func': act_func,
                              "serial_recall": serial_recall,
                              "generator": generator,
                              "x_data_type": x_data_type,
                              "end_seq_cue": end_seq_cue,
                              "use_val_data": use_val_data,
                              "optimizer": use_optimizer,
                              "loss_func": loss_func,
                              "use_batch_norm": use_batch_norm,
                              "batch_size": batch_size,
                              "augmentation": augmentation,
                              "grey_image": grey_image,
                              "use_dropout": use_dropout,
                              "loss_target": loss_target,
                              "min_loss_change": min_loss_change,
                              "max_epochs": max_epochs,
                              'timesteps': timesteps}


    # repo = "git.Repo('/home/nm13850/Documents/PhD/code/library')"

    sim_dict_name = f"{output_filename}_sim_dict.txt"

    sim_dict_path = os.path.join(exp_cond_path, sim_dict_name)

    # # simulation_info_dict
    sim_dict = {"topic_info": {"output_filename": output_filename, "cond": cond, "run": run,
                               "data_dict_path": data_dict_path, "model_path": model_path,
                               "exp_cond_path": exp_cond_path,
                               'exp_name': exp_name, 'cond_name': cond_name},
                "data_info": data_dict,
                "model_info": model_info,
                "training_info": {"trained_for": trained_for,
                                  "loss": end_loss, "acc": end_acc, 'use_val_data': use_val_data,
                                  "end_val_acc": end_val_acc, "end_val_loss": end_val_loss,
                                  'scores': scores_dict,
                                  "trained_date": trained_date, "trained_time": trained_time,
                                  'x_data_path': x_data_path, 'y_data_path': y_data_path,
                                  'sim_dict_path': sim_dict_path,
                                  'tensorboard_path': tensorboard_path,
                                  # 'commit': repo.head.object.hexsha,
                                  }
                }


    if not use_val_data:
        sim_dict['training_info']['end_val_acc'] = 'NaN'
        sim_dict['training_info']['end_val_loss'] = 'NaN'

    with open(sim_dict_name, 'w') as fp:
        json.dump(sim_dict, fp, indent=4, separators=(',', ':'))


    """converts lists of units per layer [32, 64, 128] to str "32-64-128".
    Convert these strings back to lists of ints with:
    back_to_ints = [int(i) for i in str_upl.split(sep='-')]
    """
    str_upl = "-".join(map(str, model_info['layers']['hid_layers']['hid_totals']['UPL']))
    str_fpl = "-".join(map(str, model_info['layers']['hid_layers']['hid_totals']['FPL']))



    # record training info comparrisons
    training_info = [output_filename, cond, run,
                     dset_name, x_size, n_cats, timesteps, n_items,
                     model_dir, model_name,
                     model_info['layers']['totals']['hid_layers'],
                     str_upl,
                     model_info['layers']['hid_layers']['hid_totals']['analysable'],
                     x_data_type,
                     act_func,
                     serial_recall,
                     use_optimizer, use_batch_norm, use_dropout, batch_size, augmentation, grey_image,
                     use_val_data, loss_target, min_loss_change,
                     max_epochs, trained_for, end_acc, end_loss, end_val_acc, end_val_loss,
                     checkpoint_path, trained_date, trained_time, mean_IoU, prop_seq_corr,

                     ]


    # exp_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # os.chdir(exp_path)

    # # save sel summary in exp folder not condition folder
    exp_path = find_path_to_dir(long_path=exp_cond_path, target_dir=exp_name)
    os.chdir(exp_path)

    print(f"save_summaries: {exp_path}")


    # check if training_info.csv exists
    if not os.path.isfile(f"{exp_name}_training_summary.csv"):

        headers = ["file", "cond", "run",
                   "dataset", "x_size", "n_cats", 'timesteps', "n_items",
                   "model_type", "model",
                   'hid_layers',
                   "UPL",
                   "analysable",
                   "x_data_type",
                   "act_func",
                   "serial_recall",
                   "optimizer", "batch_norm", "dropout", "batch_size", "aug", "grey_image",
                   "val_data", "loss_target", "min_loss_change",
                   "max_epochs", "trained_for", "end_acc", "end_loss", "end_val_acc", "end_val_loss",
                   "model_file", "date", "time", "mean_IoU", "prop_seq_corr"]

        training_overview = open(f"{exp_name}_training_summary.csv", 'w')
        mywriter = csv.writer(training_overview)
        mywriter.writerow(headers)
    else:
        training_overview = open(f"{exp_name}_training_summary.csv", 'a')
        mywriter = csv.writer(training_overview)

    mywriter.writerow(training_info)
    training_overview.close()

    if verbose:
        focussed_dict_print(sim_dict, 'sim_dict')

    print('\n\nto access tensorboard, in terminal use\n'
          f'tensorboard --logdir={tensorboard_path}'
          '\nthen click link')

    print("\ntrain_model() finished")


    return sim_dict
