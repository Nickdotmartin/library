import csv
import datetime
import os.path
import json
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import time
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, focussed_dict_print, print_nested_round_floats
from nick_data_tools import load_x_data, load_y_data
from nick_network_tools import get_model_dict, get_scores

sys.path.append('/home/nm13850/Documents/PhD/python_v2/models')
from mlps import fc1, fc2, fc4
from cnns import con6_pool3_fc1, con2_pool2_fc1, con4_pool2_fc1, con4_pool2_fc1_reluconv, con4_pool2_fc1_noise_layer
from rnns import lstm_1, lstm_2, lstm_4


'''following coding tips session with Ben'''

def train_model(exp_name,
                data_dict_path,
                model_path,
                cond_name=None,
                cond=None, run=None,
                max_epochs=100, use_optimizer='adam',
                loss_target=0.01, min_loss_change=0.001, batch_size=32,
                augmentation=True, grey_image=False,
                use_batch_norm=True, use_dropout=True,
                use_val_data=True,
                timesteps=1,
                exp_root='/home/nm13850/Documents/PhD/python_v2/experiments/',
                verbose=False):

    """
    script to train a neural network on a task


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
    :param exp_root: root directory for saving experiments

    :param verbose: if 0, not verbose; if 1 - print basics; if 2, print all

    :return: training_info csv
    :return: sim_dict with dataset info, model info and training info

    """

    # todo: change 'format' to f-strings (22 of them!)
    # todo: set line length
    # todo: have levels of verbosity, e.g., if verbose > 1:

    dset_dir, data_dict_name = os.path.split(data_dict_path)
    dset_dir, dset_name = os.path.split(dset_dir)
    model_dir, model_name = os.path.split(model_path)

    # print("dset_dir: {}\ndset_name: {}".format(dset_dir, dset_name))
    print(f"dset_dir: {dset_dir}\ndset_name: {dset_name}")
    print(f"model_dir: {model_dir}\nmodel_name: {model_name}")

    # Output files
    if not cond_name:
        output_filename = "{}_{}_{}".format(exp_name, model_name, dset_name)
    else:
        # output_filename = "{}_{}_{}_{}".format(exp_name, model_name, dset_name, cond_name)
        # todo: check this incase it needs changing back
        output_filename = "{}_{}".format(exp_name, cond_name)

    print("\noutput_filename: {}".format(output_filename))


    # # get info from dict
    if os.path.isfile(data_dict_path):
        data_dict = load_dict(data_dict_path)
    elif os.path.isfile(os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/', data_dict_path)):
        data_dict_path = os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/', data_dict_path)
        data_dict = load_dict(data_dict_path)
    else:
        raise FileNotFoundError(data_dict_path)
    # if type(data_dict) not dict():

    if verbose:
        # # print basic details
        print("\n**** STUDY DETAILS ****")
        print("output_filename: {}\ndset_name: {}\nmodel: {}".format(
            output_filename, dset_name, model_name))
        print("max_epochs: {}\nuse_optimizer: {}".format(max_epochs, use_optimizer))
        # print("cond: {}\nrun: {}".format(cond, run))
        print("loss_target: {}\nmin_loss_change: {}".format(loss_target, min_loss_change))
        print("batch_norm: {}\nval_data: {}\naugemntation: {}".format(use_batch_norm, use_val_data, augmentation))
        print("\n**** data_dict: ****")
        focussed_dict_print(data_dict)



    # # check for training data
    if 'train_set' in data_dict:
        x_data_path = os.path.join(data_dict['data_path'], data_dict['train_set']['X_data'])
        y_data_path = os.path.join(data_dict['data_path'], data_dict['train_set']['Y_labels'])
    else:
        x_data_path = os.path.join(data_dict['data_path'], data_dict['X_data'])
        y_data_path = os.path.join(data_dict['data_path'], data_dict['Y_labels'])

    x_data = load_x_data(x_data_path)
    y_df, y_label_list = load_y_data(y_data_path)

    # y_dict = load_y_data(y_data_path)
    #
    # y_df = y_dict['y_df']
    # y_label_list = y_dict['y_label_list']
    n_cats = data_dict["n_cats"]
    y_data = to_categorical(y_label_list, num_classes=n_cats)

    # # val data
    if use_val_data:
        print("\n**** Loading validation data ****")
        if "val_set" in data_dict:
            # x_val = load_data(data_dict, 'x', test_train_val='val_set')
            # y_val_dict = load_data(data_dict, 'y', test_train_val='val_set')
            # y_val_df = y_val_dict['y_df']
            # y_val_label_list = y_val_dict['y_label_list']

            x_val = load_x_data(os.path.join(data_dict['data_path'], data_dict['val_set']['X_data']))
            y_val_df, y_val_label_list = load_y_data(os.path.join(data_dict['data_path'],
                                                                  data_dict['val_set']['Y_labels']))
            y_val = to_categorical(y_val_label_list, num_classes=n_cats)
        else:
            print("validation data not found - performing split")
            x_train, x_val, y_train_label_list, y_val_label_list = train_test_split(x_data, y_label_list, test_size=0.2,
                                                                                    random_state=1)
            print("y_train_label_list: {}.  Count: {}\ny_val_label_list: {}.  count {}".format(
                np.shape(y_train_label_list), np.unique(y_train_label_list, return_counts=True)[1],
                np.shape(y_val_label_list), np.unique(y_val_label_list, return_counts=True)[1]))
            y_train = to_categorical(y_train_label_list, num_classes=n_cats)
            y_val = to_categorical(y_val_label_list, num_classes=n_cats)
    else:
        x_train = x_data
        y_train = y_data

    n_items = len(y_train)

    # fix random seed for reproducability during development - not for simulations
    seed = 7
    np.random.seed(seed)

    # # data preprocessing
    x_size = data_dict["X_size"]

    if len(np.shape(x_train)) == 4:
        image_dim = data_dict['image_dim']
        n_items, width, height, channels = np.shape(x_train)
    else:
        # # this is just for MNIST
        if model_dir == 'cnn':
            width, height = data_dict['image_dim']
            x_train = x_train.reshape(x_train.shape[0], width, height, 1)
            print("\nRESHAPING x_train to: {}".format(np.shape(x_train)))
            if use_val_data:
                x_val = x_val.reshape(x_val.shape[0], width, height, 1)

    """# # # proprocessing here - put data in range 0-1
    # if len(np.shape(x_train)) == 2:
    #     scaler = MinMaxScaler()
    #     x_train = scaler.fit_transform(x_train)
    #     if use_val_data is True:
    #         x_val = scaler.fit_transform(x_val)"""


    # # The Model
    if model_dir in ['mlp', 'mlps']:
        print("\nloading an mlp model")
        augmentation = False
        # todo: add units_per_layer for mlp - or fix number of units in model
        units_per_layer = 32
        # models[model_name].build(...)
        if model_name == 'fc4':
            build_model = fc4.build(classes=n_cats, units_per_layer=units_per_layer,
                                    batch_norm=use_batch_norm, dropout=use_dropout)
        if model_name == 'fc2':
            build_model = fc2.build(classes=n_cats, units_per_layer=units_per_layer,
                                    batch_norm=use_batch_norm, dropout=use_dropout)
        if model_name == 'fc1':
            build_model = fc1.build(classes=n_cats, units_per_layer=units_per_layer,
                                    batch_norm=use_batch_norm, dropout=use_dropout)

    elif model_dir in ['cnn', 'cnns']:
        print("loading a cnn model")
        units_per_layer = None  # todo: add way to get UPL for complex models
        width, height = data_dict['image_dim']
        depth = 3
        if grey_image:
            depth = 1

        if model_name == 'con6_pool3_fc1':
            build_model = con6_pool3_fc1.build(width=width, height=height, depth=depth, classes=n_cats,
                                               batch_norm=use_batch_norm, dropout=use_dropout)
        elif model_name == 'con4_pool2_fc1':
            build_model = con4_pool2_fc1.build(width=width, height=height, depth=depth, classes=n_cats,
                                               batch_norm=use_batch_norm, dropout=use_dropout)
        elif model_name == 'con2_pool2_fc1':
            build_model = con2_pool2_fc1.build(width=width, height=height, depth=depth, classes=n_cats,
                                               batch_norm=use_batch_norm, dropout=use_dropout)
        elif model_name == 'con4_pool2_fc1_reluconv':
            build_model = con4_pool2_fc1_reluconv.build(width=width, height=height, depth=depth, classes=n_cats,
                                                        batch_norm=use_batch_norm, dropout=use_dropout)
        elif model_name == 'con4_pool2_fc1_noise_layer':
            build_model = con4_pool2_fc1_noise_layer.build(width=width, height=height, depth=depth, classes=n_cats,
                                                           batch_norm=use_batch_norm, dropout=use_dropout)
        else:
            raise TypeError("Model name not recognised")


    elif model_dir in ['rnn', 'rnns']:
        print("loading a recurrent model")
        augmentation = False

        units_per_layer = 32  # todo: add way to get UPL for complex models
        features = data_dict['X_size']

        if model_name == 'lstm_4':
            build_model = lstm_4.build(features=features, classes=n_cats, timesteps=timesteps,
                                       units_per_layer=units_per_layer, batch_norm=use_batch_norm, dropout=use_dropout)
        if model_name == 'lstm_2':
            build_model = lstm_2.build(features=features, classes=n_cats, timesteps=timesteps,
                                       units_per_layer=units_per_layer, batch_norm=use_batch_norm, dropout=use_dropout)
        if model_name == 'lstm_1':
            build_model = lstm_1.build(features=features, classes=n_cats, timesteps=timesteps,
                                       units_per_layer=units_per_layer, batch_norm=use_batch_norm, dropout=use_dropout)
    else:
        print("model_dir not recognised")

    model = build_model

    # loss
    loss_func = 'categorical_crossentropy'
    if n_cats == 2:
        loss_func = 'binary_crossentropy'

    # optimizer
    SGD_LR = 0.01  # initialize learning rate
    sgd = SGD(lr=SGD_LR, decay=SGD_LR / max_epochs)
    this_optimizer = sgd
    if use_optimizer == 'adam':
        this_optimizer = Adam(lr=0.001)
    elif use_optimizer == 'rmsprop':
        this_optimizer = RMSprop(lr=0.0001, decay=1e-6)


    # # compile model
    model.compile(loss=loss_func, optimizer=this_optimizer, metrics=['accuracy'])


    # # get model dict
    model_info = get_model_dict(model)  # , verbose=True)
    print("\nmodel_info:")
    print_nested_round_floats(model_info)
    plot_model(model, to_file='model_diag.png', show_shapes=True)

    # print(model.summary())


    # # training parameters
    # patience_for_loss_change: wait this long to see if loss improves
    patience_for_loss_change = int(max_epochs / 50)

    checkpoint_path = 'model_{"epoch":02d}-{"loss":.2f}.hdf5'
    checkpoint_mon = 'loss'
    if use_val_data:
        checkpoint_path = 'model_{"epoch":02d}-{"val_loss":.2f}.hdf5'
        checkpoint_mon = 'val_loss'

    # checkpointing.  Save model and weights with best val loss.
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor=checkpoint_mon, verbose=0,
                                   save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1,
    #                                save_best_only=True)


    #  Only save model with best (val) loss.

    # early_stop_plateau - if there is no imporovement
    early_stop_plateau = EarlyStopping(monitor='loss', min_delta=min_loss_change,
                                       patience=patience_for_loss_change,
                                       verbose=1, mode='min')

    val_early_stop_plateau = EarlyStopping(monitor='val_loss', min_delta=min_loss_change,
                                           patience=patience_for_loss_change, verbose=verbose, mode='min')

    tensorboard = TensorBoard(log_dir=f'logs/{time()}')
    '''to access tensorboard, in terminal use 
    tensorboard --logdir=logs/
    then click link'''

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
        else:
            fit_model = model.fit(x_train, y_train,
                                  epochs=max_epochs, batch_size=batch_size, verbose=1, callbacks=callbacks_list)

    ############################################
    """Once training is done"""
    ############################################
    print("\n**** TRAINING COMPLETE ****")
    root = os.path.here()
    exp_cond_path = os.path.join(exp_root, exp_name, output_filename)
    if not os.path.exists(exp_cond_path):
        os.makedirs(exp_cond_path)
    save_here = os.chdir(exp_cond_path)
    print("saving to: {}".format(exp_cond_path))


    # # plot the training loss and accuracy
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(fit_model.history['acc'])
    if use_val_data:
        ax1.plot(fit_model.history['val_acc'])
    ax1.set_title('model accuracy (top); loss (bottom)')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax2.plot(fit_model.history['loss'])
    if use_val_data:
        ax2.plot(fit_model.history['val_loss'])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    fig.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(output_filename) + '_training')

    # # Training info
    trained_for = len(fit_model.history['loss'])
    end_accuracy = np.around(fit_model.history['acc'][-1], decimals=3)
    end_loss = np.around(fit_model.history['loss'][-1], decimals=3)
    print("\nTraining Info\nepochs: {}\nacc: {}\nloss: {}".format(trained_for, end_accuracy, end_loss))

    if use_val_data:
        end_val_acc = np.around(fit_model.history['val_acc'][-1], decimals=3)
        end_val_loss = np.around(fit_model.history['val_loss'][-1], decimals=3)
        print("val_acc: {}\nval_loss: {}".format(end_val_acc, end_val_loss))
    else:
        end_val_acc = end_val_loss = np.nan  # -np.inf

    # # # PART 3 get_scores() # # #
    # # these three lines are to re-shape MNIST
    if len(np.shape(x_data)) != 4:
        if model_dir == 'cnn':
            x_data = x_data.reshape(x_data.shape[0], width, height, 1)

    predicted_outputs = model.predict(x_data)  # use x_data NOT x_train to fit shape of y_df
    item_correct_df, scores_dict, incorrect_items = get_scores(predicted_outputs, y_df, output_filename,
                                                               verbose=True, save_all_csvs=True)

    if verbose:
        print("\n****Scores_dict****")
        focussed_dict_print(scores_dict)

    # save the model (and weights) after training
    # trained_model_file = "{}_model.h5".format(output_filename)
    # model.save(trained_model_file)

    trained_date = int(datetime.datetime.now().strftime("%y%m%d"))
    trained_time = int(datetime.datetime.now().strftime("%H%M"))
    model_info['overview'] = {'model_type': model_dir, 'model_name': model_name, "trained_model": checkpoint_path,
                              "units_per_layer": units_per_layer, "optimizer": use_optimizer,
                              "use_batch_norm": use_batch_norm, "batch_size": batch_size, "augmentation": augmentation,
                              "grey_image": grey_image, "use_dropout": use_dropout, "loss_target": loss_target,
                              "min_loss_change": min_loss_change, "max_epochs": max_epochs, 'timesteps': timesteps}


    # # simulation_info_dict
    sim_dict = {"topic_info": {"output_filename": output_filename, "cond": cond, "run": run,
                               "data_dict_path": data_dict_path, "model_path": model_path,
                               "exp_cond_path": exp_cond_path,
                               'exp_name': exp_name, 'cond_name': cond_name},
                "data_info": data_dict,
                "model_info": model_info,
                'scores': scores_dict,
                "training_info": {"trained_for": trained_for,
                                  "loss": end_loss, "acc": end_accuracy, 'use_val_data': use_val_data,
                                  "end_val_acc": end_val_acc, "end_val_loss": end_val_loss,
                                  "trained_date": trained_date, "trained_time": trained_time,
                                  'x_data_path': x_data_path, 'y_data_path': y_data_path}
                }

    sim_dict_name = "{}_sim_dict".format(output_filename)

    with open(sim_dict_name + '.txt', 'w') as fp:
        json.dump(sim_dict, fp, indent=4)

    """converts lists of units per layer [32, 64, 128] to str "32-64-128".
    Convert these strings back to lists of ints with:
    back_to_ints = [int(i) for i in str_UPL.split(sep='-')]
    """
    str_UPL = "-".join(map(str, model_info['layers']['hid_layers']['hid_totals']['UPL']))
    str_FPL = "-".join(map(str, model_info['layers']['hid_layers']['hid_totals']['FPL']))

    # record training info comparrisons
    training_info = [output_filename, cond, run,
                     dset_name, x_size, n_cats, timesteps, n_items,
                     model_dir, model_name,
                     model_info['layers']['totals']['all_layers'],
                     model_info['layers']['totals']['hid_layers'],

                     model_info['layers']['hid_layers']['hid_totals']['act_layers'],
                     model_info['layers']['hid_layers']['hid_totals']['dense_layers'],
                     str_UPL,
                     model_info['layers']['hid_layers']['hid_totals']['conv_layers'],
                     str_FPL,
                     model_info['layers']['hid_layers']['hid_totals']['analysable'],
                     use_optimizer, use_batch_norm, use_dropout, batch_size, augmentation, grey_image,
                     use_val_data, loss_target, min_loss_change,
                     max_epochs, trained_for, end_accuracy, end_loss, end_val_acc, end_val_loss,
                     checkpoint_path, trained_date, trained_time,
                     ]


    exp_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_summaries = os.chdir(exp_path)
    print("save_summaries: {}".format(exp_path))

    # check if training_info.csv exists
    if not os.path.isfile("{}_training_summary.csv".format(exp_name)):
        # # todo: use pandas dataframe and save with nick_to_csv

        headers = ["file", "cond", "run",
                   "dataset", "x_size", "n_cats", 'timesteps', "n_items",
                   "model_type", "model", "all_layers", 'hid_layers',
                   "act_layers",
                   "dense_layers", "UPL", "conv_layers", "FPL", "analysable",
                   "optimizer", "batch_norm", "dropout", "batch_size", "aug", "grey_image",
                   "val_data", "loss_target", "min_loss_change",
                   "max_epochs", "trained_for", "end_acc", "end_loss", "end_val_acc", "end_val_loss",
                   "model_file", "date", "time"]

        training_overview = open("{}_training_summary.csv".format(exp_name), 'w')
        mywriter = csv.writer(training_overview)
        mywriter.writerow(headers)
    else:
        # todo: change to nick_read_csv then edit with pandas
        training_overview = open("{}_training_summary.csv".format(exp_name), 'a')
        mywriter = csv.writer(training_overview)

    mywriter.writerow(training_info)
    training_overview.close()

    if verbose:
        focussed_dict_print(sim_dict)

    print("\nff_conv_colour_sim finished")

    return training_info, sim_dict


######################

# # small mlp
# training_info, sim_dict = train_model(exp_name='script_test',
#                                       data_dict_path='other_classification/iris/orig/iris',
#                                       model_path='mlps/fc2',
#                                       max_epochs=2,
#                                       verbose=True,
#                                       cond_name='check_train_png',
#                                       use_batch_norm=False, use_dropout=False,
#                                       )
#
#
# # # # small cnn
# training_info, sim_dict = train_model(exp_name='script_test',
#                                       data_dict_path='digits/MNIST_June18/orig_data/mnist',
#                                       model_path='cnn/con2_pool2_fc1',
#                                       max_epochs=2,
#                                       grey_image=True,
#                                       verbose=True)
#
# # # medium cnn
# training_info, sim_dict = train_model(exp_name='script_test',
#                                       data_dict_path='objects/CIFAR_10/CIFAR_10_2019',
#                                       model_path='cnn/con6_pool3_fc1',
#                                       max_epochs=2,
#                                       verbose=True)
######################
