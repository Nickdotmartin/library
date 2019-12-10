import csv
import datetime
import json
import os
# import git
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
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

# import keras.backend as K

# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

def nick_acc(y_true, y_pred, serial_recall=True, output_type='classes'):
    """
    :param y_true:
    :param y_pred:
    :param serial_recall: if True, Y array is a list of vectors
                        If False, y-array is a single vector  e.g., activate all words simultaneously
    :param output_type: default 'classes': output units represent class labels
                        'letters' output units correspond to letters

    :return: nick_acc (float)
    """
    hid0_input = tf.compat.v1.placeholder(tf.float32, shape=(32,None,31), name="hid0_input")
    print(f"\nhid0_input: {hid0_input}")
    output_target = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None), name="output_target")
    print(f"\noutput_target: {output_target}")
    tf_session = K.get_session()

    y_1hot = False

    # if output_type=='classes':
    #
    #     if serial_recall:
    #         # multiple 1hot CLASS vectors
    #         y_1hot = True
    #
    #     elif not serial_recall:
    #         # single 3hot CLASS vectors
    #
    # elif output_type=='letters':
    #
    #     # # must be serial recall
    #     # multiple 3hot LETTER vectors

    seq_corr = []

    # to convert a tensor to a numpy array use tf.keras.backend.eval(x)
    true_shape = tf.keras.backend.int_shape(y_true)
    print(f"\ntrue_shape int_shape: {true_shape}")

    pred_shape = tf.keras.backend.int_shape(y_pred)
    print(f"\npred_shape: {pred_shape}")

    batch_size, timesteps, out_units = tf.keras.backend.int_shape(y_pred)
    print(f"\nbatch_size: {batch_size}\ntimesteps: {timesteps}\nout_units: {out_units}")

    # check_shape = y_pred.eval(session=tf_session)
    # print(f"\ncheck_shape: {check_shape}")

    # pred_eval = tf.keras.backend.eval(y_pred)
    # print(f"\npred_eval: {pred_eval}")

    ###
    # decoded_softmax = tf.keras.backend.ctc_decode(y_pred,
    #                                               input_length=(out_units, ),
    #                                               greedy=True,
    #                                               beam_width=100,
    #                                               top_paths=1)
    # print(f"\ndecoded_softmax: {decoded_softmax}")
    ###

    # guess = tf.compat.v1.placeholder(tf.float32, true_shape)
    # truth = tf.compat.v1.placeholder(tf.float32, true_shape)
    # print(f"\nguess: {guess}")
    # print(f"\ntruth: {truth}")


    # with tf.compat.v1.Session() as sess:
    #
    #     feed_dict = {"guess": y_pred, "guess": y_true}
    #
    #     pred_dict = {"pred": y_pred}
    #     pred_eval_dict = tf.keras.backend.eval(pred_dict)
    #     pred_eval = pred_eval_dict['pred']
    #     print(f"\npred_eval: {pred_eval}")
    #     true_dict = {"true": y_true}
    #     true_eval = tf.keras.backend.eval(true_dict)
    #     print(f"\ntrue_eval: {true_eval}")

        # classification = sess.run(y, feed_dict)
        # print(classification)

    # might need to iterate over time
    for i in range(batch_size):
        # for t in range(timesteps):
        this_pred = K.variable(y_pred[i])  # [t]
        this_true = y_true[i]  # [t]
        print(f"\nthis_pred: {this_pred} {this_pred.shape}")
        print(f"this_true: {this_true}")

    flat_preds = K.flatten(y_pred)
    print(f"\nK.int_shape(flat_preds): {K.int_shape(flat_preds)}")

    flat_true = K.flatten(y_true)
    print(f"\nK.int_shape(flat_true): {K.int_shape(flat_true)}")

    # v = K.variable(flat_true)
    # print(K.eval(v))
    # print(K.eval(K.sqrt(v)))


    # pred_lengths = tf.keras.backend.int_shape(y_pred)
    # print(f"\npred_lengths int_shape: {pred_lengths}")
    # pred_length = pred_lengths[0]
    # print(f"\npred_length: {pred_length}")

    if serial_recall and output_type is 'classes':
        # decoded_softmax = tf.keras.backend.ctc_decode(y_pred,
        #                                               pred_length=pred_length[0],
        #                                               greedy=True,
        #                                               beam_width=100,
        #                                               top_paths=1)
        # print(f"\ndecoded_softmax: {decoded_softmax}")
        # if this decode doesn't work I could use
        # max_val = tf.keras.backend.max(this_pred)
        # print(f"\nmax_val: {max_val}")
        # with sess.as_default():
        # print(f"\nK.eval(max_val): {K.eval(max_val)}")


        # then make a new variable of zeros
        # prepared_pred = tf.keras.backend.zeros_like(this_pred, name='all_zeros')
        # pred_list = [0.0] * out_units
        # prepared_pred = K.eval(all_zeros)

        # then convert the zero in the correct location to a 1
        # pred_list[max_val] = 1

        # pred_array = np.array(pred_list)

        # prepared_pred = K.variable(value=pred_array, dtype='float64', name='prepared_pred')
        # todo: this is wrong dor decoding softmax, use max val somehow
        y_pred_binary = K.round(flat_preds)
        prepared_pred = y_pred_binary

        # print(f"\nprepared_pred: {prepared_pred}")

        # print(f"\nprepared_pred.eval(): {prepared_pred.eval(session=tf_session)}")


    else:
        # multiple active items

        # make vector of same length where each item is .5
        # point5_array = np.array([.5] * pred_length)
        # point5_vector = K.variable(value=point5_array, dtype='float64', name='point5_vector')
        #
        # prepared_pred = tf.keras.backend.greater_equal(this_pred, point5_vector)
        y_pred_binary = tf.where(flat_preds >= 0.5, 1., 0.)
        prepared_pred = y_pred_binary
    print(f"\nprepared_pred: {prepared_pred}")

    # compare
    correct_item = tf.keras.backend.equal(prepared_pred, flat_true)
    print(f"\ncorrect_item: {correct_item}")

    cast_input = K.cast(correct_item, dtype='int32')
    print(f"\ncast_input: {cast_input}")
    seq_corr.append(cast_input)


    # if correct_item == 'Tensor("metrics/nick_acc/Equal:0", shape=(?,), dtype=bool)':
    #     print("yup")
    # if correct_item == K.variable(value=False,
    #                               dtype=bool,
    #                               name=None,
    #                               constraint=None):
    #     print('match')
    #
    # print_this = tf.keras.backend.print_tensor(
    #     correct_item,
    #     message='here tis'
    # )
    #
    # print(print_this)
    # # summed_items == K.sum(correct_item
    # if cast_input <= .5:
    #     print('True')
    #     seq_corr.append(1)
    # else:
    #     print('False')
    #     seq_corr.append(0)
    # seq_corr.append(correct_item)

    print(f"\nseq_corr: {seq_corr}")

    n_correct = K.sum(seq_corr)
    print(f"\nn_correct: {n_correct}")

    total_items = K.shape(seq_corr)
    print(f"\ntotal_items: {total_items}")

    acc_prop = n_correct/total_items
    print(f"\nacc_prop: {acc_prop}")

    return acc_prop

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy', mean_pred])


def train_model(exp_name,
                data_dict_path,
                model_path,
                cond_name=None,
                cond=None, run=None,
                hid_layers=1,
                units_per_layer=200,
                act_func='sigmoid',
                serial_recall=False,
                y_1hot=True,
                output_units='n_cats',
                generator=True,
                x_data_type='dist_letter_X',
                end_seq_cue=False,
                max_epochs=100, use_optimizer='adam',
                loss_target=0.01, min_loss_change=0.0001, batch_size=32,
                augmentation=True, grey_image=False,
                use_batch_norm=True, use_dropout=0.0,
                use_val_data=True,
                timesteps=1,
                train_cycles=False,
                weight_init='GlorotUniform',
                lr=0.001,
                unroll=False,
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
    :param y_1hot: if True, output is 1hot, if False, output is multi_label.
    :param output_units: default='n_cats', e.g., number of categories.
                        If 'x_size', will be same as number of input feats.
                        Can also accept int.
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
    :param train_cycles: if False, all lists lengths = timesteps.
                        If True, train on varying length, [1, 2, 3,...timesteps].
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
    # K.clear_session()

    if train_cycles:
        train_ts = None
    else:
        train_ts = timesteps

    output_type = 'classes'
    if type(output_units) is int:
        n_output_units = output_units
        if n_output_units == x_size:
            output_type = 'letters'
        elif n_output_units == n_cats:
            output_type = 'classes'
        else:
            raise ValueError(f"n_output_units does not match x_size or n_cats\n"
                             f"need to specifiy output_type as words or letters")

    elif type(output_units) is str:
        if output_units.lower() == 'n_cats':
            n_output_units = n_cats
        elif output_units.lower() == 'x_size':
            n_output_units = x_size
            output_type = 'letters'
        else:
            raise ValueError(f'output_units should be specified as an int, '
                             f'or with a string "n_cats" or "x_size"\noutput_units: {output_units}')

    if 'rnn' in model_dir:
        print("loading a recurrent model")
        augmentation = False

        models_dict = {'Bowers14rnn': Bowers14rnn,
                       'SimpleRNNn': SimpleRNNn,
                       'GRUn': GRUn,
                       'LSTMn': LSTMn,
                       'Seq2Seq': Seq2Seq}

        model = models_dict[model_name].build(features=x_size, classes=n_output_units, timesteps=train_ts,
                                              batch_size=batch_size, n_layers=hid_layers,
                                              serial_recall=serial_recall,
                                              units_per_layer=units_per_layer, act_func=act_func,
                                              y_1hot=y_1hot,
                                              dropout=use_dropout,
                                              masking=train_cycles,
                                              weight_init=weight_init,
                                              unroll=unroll)
    else:
        print("model_dir not recognised")


    # # loss
    loss_func = 'categorical_crossentropy'
    if not y_1hot:
        loss_func = 'binary_crossentropy'


    # optimizer
    sgd = SGD(lr=lr, momentum=.9)  # decay=sgd_lr / max_epochs)
    this_optimizer = sgd

    if use_optimizer == 'SGD_no_momentum':
        this_optimizer = SGD(lr=lr, momentum=0.0, nesterov=False)  # decay=sgd_lr / max_epochs)
    elif use_optimizer == 'SGD_Nesterov':
        this_optimizer = SGD(lr=lr, momentum=.1, nesterov=True)  # decay=sgd_lr / max_epochs)
    elif use_optimizer == 'adam':
        this_optimizer = Adam(lr=lr, amsgrad=False)
    elif use_optimizer == 'adam_amsgrad':
        # simulations run prior to 05122019 did not have this option, and may have use amsgrad under the name 'adam'
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
    if y_1hot:
        main_metric = 'categorical_accuracy'
        # mean_IoU = free_rec_acc()
        # prop_corr_acc = prop_corr(y_true, y_pred)
        # iou_acc = mean_IoU(y_true, y_pred)

        # compile model
        # model.compile(loss=loss_func, optimizer=this_optimizer,
        #               metrics=[main_metric, nick_acc])

    # todo: come back and try metric again once I have got the generator working.
    #   and once I have written the analysis script for current get scores.
    model.compile(loss=loss_func, optimizer=this_optimizer,
                  metrics=[main_metric])  # , nick_acc])

    optimizer_details = model.optimizer.get_config()
    # print_nested_round_floats(model_details)
    focussed_dict_print(optimizer_details, 'optimizer_details')


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
    patience_for_loss_change = int(max_epochs / 50)


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
                                                  output_type=output_type,
                                                  x_data_type=x_data_type,
                                                  end_seq_cue=end_seq_cue,
                                                  train_cycles=train_cycles,
                                                  # verbose=verbose
                                                  )

            fit_model = model.fit_generator(generate_data,
                                            steps_per_epoch=100,
                                            epochs=max_epochs,
                                            callbacks=callbacks_list,
                                            shuffle=False)

    ########################################################
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
            # todo: separate serial_recall and y_1hot.  how to decide if repetitions are allowed?
    # #     # # get labels for 100 sequences
    # test_label_seqs = get_label_seqs(n_labels=n_cats, seq_len=timesteps,
    #                                  repetitions=False, n_seqs=10*batch_size)

    # # load test label seqs
    data_path = data_dict['data_path']
    if train_cycles:
        timesteps = 7
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
                                  x_data_type=x_data_type,
                                  output_type=output_type,
                                  end_seq_cue=end_seq_cue,
                                  batch_size=batch_size,
                                  verbose=verbose)

    focussed_dict_print(scores_dict, 'scores_dict')
    # mean_IoU = scores_dict['mean_IoU']
    # prop_seq_corr = scores_dict['prop_seq_corr']
    #
    #    #        # todo: separate serial_recall and y_1hot
    # trained_date = int(datetime.datetime.now().strftime("%y%m%d"))
    # trained_time = int(datetime.datetime.now().strftime("%H%M"))
    # model_info['overview'] = {'model_type': model_dir, 'model_name': model_name,
    #                           "trained_model": checkpoint_path,
    #                           "hid_layers": hid_layers,
    #                           "units_per_layer": units_per_layer,
    #                           'act_func': act_func,
    #                           "serial_recall": serial_recall,
    #                           "generator": generator,
    #                           "x_data_type": x_data_type,
    #                           "end_seq_cue": end_seq_cue,
    #                           "use_val_data": use_val_data,
    #                           "optimizer": use_optimizer,
    #                           "loss_func": loss_func,
    #                           "use_batch_norm": use_batch_norm,
    #                           "batch_size": batch_size,
    #                           "augmentation": augmentation,
    #                           "grey_image": grey_image,
    #                           "use_dropout": use_dropout,
    #                           "loss_target": loss_target,
    #                           "min_loss_change": min_loss_change,
    #                           "max_epochs": max_epochs,
    #                           'timesteps': timesteps}
    #
    #
    # # repo = "git.Repo('/home/nm13850/Documents/PhD/code/library')"
    #
    # sim_dict_name = f"{output_filename}_sim_dict.txt"
    #
    # sim_dict_path = os.path.join(exp_cond_path, sim_dict_name)
    #    #        # todo: separate serial_recall and y_1hot
    # # # simulation_info_dict
    # sim_dict = {"topic_info": {"output_filename": output_filename, "cond": cond, "run": run,
    #                            "data_dict_path": data_dict_path, "model_path": model_path,
    #                            "exp_cond_path": exp_cond_path,
    #                            'exp_name': exp_name, 'cond_name': cond_name},
    #             "data_info": data_dict,
    #             "model_info": model_info,
    #             "training_info": {"trained_for": trained_for,
    #                               "loss": end_loss, "acc": end_acc, 'use_val_data': use_val_data,
    #                               "end_val_acc": end_val_acc, "end_val_loss": end_val_loss,
    #                               'scores': scores_dict,
    #                               "trained_date": trained_date, "trained_time": trained_time,
    #                               'x_data_path': x_data_path, 'y_data_path': y_data_path,
    #                               'sim_dict_path': sim_dict_path,
    #                               'tensorboard_path': tensorboard_path,
    #                               # 'commit': repo.head.object.hexsha,
    #                               }
    #             }
    #
    #
    # if not use_val_data:
    #     sim_dict['training_info']['end_val_acc'] = 'NaN'
    #     sim_dict['training_info']['end_val_loss'] = 'NaN'
    #
    # with open(sim_dict_name, 'w') as fp:
    #     json.dump(sim_dict, fp, indent=4, separators=(',', ':'))
    #
    #
    # """converts lists of units per layer [32, 64, 128] to str "32-64-128".
    # Convert these strings back to lists of ints with:
    # back_to_ints = [int(i) for i in str_upl.split(sep='-')]
    # """
    # str_upl = "-".join(map(str, model_info['layers']['hid_layers']['hid_totals']['UPL']))
    # str_fpl = "-".join(map(str, model_info['layers']['hid_layers']['hid_totals']['FPL']))
    #
    #
    #    #        # todo: separate serial_recall and y_1hot

    # # record training info comparrisons
    # training_info = [output_filename, cond, run,
    #                  dset_name, x_size, n_cats, timesteps, n_items,
    #                  model_dir, model_name,
    #                  model_info['layers']['totals']['hid_layers'],
    #                  str_upl,
    #                  model_info['layers']['hid_layers']['hid_totals']['analysable'],
    #                  x_data_type,
    #                  act_func,
    #                  serial_recall,
    #                  use_optimizer, use_batch_norm, use_dropout, batch_size, augmentation, grey_image,
    #                  use_val_data, loss_target, min_loss_change,
    #                  max_epochs, trained_for, end_acc, end_loss, end_val_acc, end_val_loss,
    #                  checkpoint_path, trained_date, trained_time, mean_IoU, prop_seq_corr,
    #
    #                  ]
    #
    #
    # # exp_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # # os.chdir(exp_path)
    #
    # # # save sel summary in exp folder not condition folder
    # exp_path = find_path_to_dir(long_path=exp_cond_path, target_dir=exp_name)
    # os.chdir(exp_path)
    #
    # print(f"save_summaries: {exp_path}")
    #
    #     #        # todo: separate serial_recall and y_1hot
    # # check if training_info.csv exists
    # if not os.path.isfile(f"{exp_name}_training_summary.csv"):
    #
    #     headers = ["file", "cond", "run",
    #                "dataset", "x_size", "n_cats", 'timesteps', "n_items",
    #                "model_type", "model",
    #                'hid_layers',
    #                "UPL",
    #                "analysable",
    #                "x_data_type",
    #                "act_func",
    #                "serial_recall",
    #                "optimizer", "batch_norm", "dropout", "batch_size", "aug", "grey_image",
    #                "val_data", "loss_target", "min_loss_change",
    #                "max_epochs", "trained_for", "end_acc", "end_loss", "end_val_acc", "end_val_loss",
    #                "model_file", "date", "time", "mean_IoU", "prop_seq_corr"]
    #
    #     training_overview = open(f"{exp_name}_training_summary.csv", 'w')
    #     mywriter = csv.writer(training_overview)
    #     mywriter.writerow(headers)
    # else:
    #     training_overview = open(f"{exp_name}_training_summary.csv", 'a')
    #     mywriter = csv.writer(training_overview)
    #
    # mywriter.writerow(training_info)
    # training_overview.close()
    #
    # if verbose:
    #     focussed_dict_print(sim_dict, 'sim_dict')
    #
    # print('\n\nto access tensorboard, in terminal use\n'
    #       f'tensorboard --logdir={tensorboard_path}'
    #       '\nthen click link')

    print("\ntrain_model() finished")

    sim_dict = {'empty': 'dict', 'nothing': 'here'}

    return sim_dict
