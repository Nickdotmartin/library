import csv
import datetime
import os.path
import json
import git

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.data import load_x_data, load_y_data, switch_home_dirs
from tools.network import get_model_dict, get_scores

from models.cnns import con6_pool3_fc1, con2_pool2_fc1, con4_pool2_fc1, \
    con4_pool2_fc1_reluconv, con4_pool2_fc1_noise_layer
from models.rnns import Bowers14rnn, SimpleRNNn, GRUn, LSTMn, Seq2Seq
from models.mlps import mlp, fc1, fc2, fc4


'''following coding tips session with Ben'''
# todo: have levels of verbosity, e.g., if verbose > 1:


print("tf: ", tf.version.VERSION)
print("keras: ", tf.keras.__version__)


def train_model(exp_name,
                data_dict_path,
                model_path,
                cond_name=None,
                cond=None, run=None,
                max_epochs=100, use_optimizer='adam',
                loss_target=0.01, min_loss_change=0.0001,
                batch_size=32,
                lr=0.001,
                n_layers=1, units_per_layer=200,
			    act_func='relu', use_bias=True,
                y_1hot=True, output_act='softmax',
                weight_init='GlorotUniform',
                augmentation=True, grey_image=False,
                use_batch_norm=False, use_dropout=0.0,
                use_val_data=True,
                timesteps=1,
                exp_root='/home/nm13850/Documents/PhD/python_v2/experiments/',
                verbose=False,
                test_run=False,
                ):

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


    dset_dir, data_dict_name = os.path.split(data_dict_path)
    dset_dir, dset_name = os.path.split(dset_dir)
    model_dir, model_name = os.path.split(model_path)

    print(f"dset_dir: {dset_dir}\ndset_name: {dset_name}")
    print(f"model_dir: {model_dir}\nmodel_name: {model_name}")

    # Output files
    if not cond_name:
        output_filename = f"{exp_name}_{model_name}_{dset_name}"
    else:
        output_filename = f"{exp_name}_{cond_name}"

    print(f"\noutput_filename: {output_filename}")

    print(data_dict_path)
    # # get info from dict
    if os.path.isfile(data_dict_path):
        data_dict = load_dict(data_dict_path)
    elif os.path.isfile(os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/', data_dict_path)):
        data_dict_path = os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/', data_dict_path)
        data_dict = load_dict(data_dict_path)
    elif os.path.isfile(switch_home_dirs(data_dict_path)):
        data_dict_path = switch_home_dirs(data_dict_path)
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



    # # check for training data
    if 'train_set' in data_dict:
        if os.path.isfile(os.path.join(data_dict['data_path'], data_dict['train_set']['X_data'])):
            x_data_path = os.path.join(data_dict['data_path'], data_dict['train_set']['X_data'])
            y_data_path = os.path.join(data_dict['data_path'], data_dict['train_set']['Y_labels'])
        elif os.path.isfile(switch_home_dirs(os.path.join(data_dict['data_path'],
                                                          data_dict['train_set']['X_data']))):
            x_data_path = switch_home_dirs(os.path.join(data_dict['data_path'],
                                                        data_dict['train_set']['X_data']))
            y_data_path = switch_home_dirs(os.path.join(data_dict['data_path'],
                                                        data_dict['train_set']['Y_labels']))
        else:
            raise FileNotFoundError(f"training data not found\n"
                                    f"{os.path.join(data_dict['data_path'], data_dict['train_set']['X_data'])}")
    else:
        # # if no training set
        if os.path.isfile(os.path.join(data_dict['data_path'], data_dict['X_data'])):
            x_data_path = os.path.join(data_dict['data_path'], data_dict['X_data'])
            y_data_path = os.path.join(data_dict['data_path'], data_dict['Y_labels'])
        else:
            data_path = switch_home_dirs(data_dict['data_path'])
            if os.path.isfile(os.path.join(data_path, data_dict['X_data'])):
                x_data_path = os.path.join(data_path, data_dict['X_data'])
                y_data_path = os.path.join(data_path, data_dict['Y_labels'])
                data_dict['data_path'] = data_path
            else:
                raise FileNotFoundError(f'cant find x data at\n'
                                        f'{os.path.join(data_path, data_dict["X_data"])}')


    x_data = load_x_data(x_data_path)
    y_df, y_label_list = load_y_data(y_data_path)

    n_cats = data_dict["n_cats"]
    y_data = to_categorical(y_label_list, num_classes=n_cats)

    # # val data
    if use_val_data:
        print("\n**** Loading validation data ****")
        if "val_set" in data_dict:
            x_val = load_x_data(os.path.join(data_dict['data_path'], data_dict['val_set']['X_data']))
            y_val_df, y_val_label_list = load_y_data(os.path.join(data_dict['data_path'],
                                                                  data_dict['val_set']['Y_labels']))
            y_val = to_categorical(y_val_label_list, num_classes=n_cats)
        else:
            print("validation data not found - performing split")
            x_train, x_val, y_train_label_list, y_val_label_list = train_test_split(x_data, y_label_list, test_size=0.2,
                                                                                    random_state=1)
            print(f"y_train_label_list: {np.shape(y_train_label_list)}.  "
                  f"Count: {np.unique(y_train_label_list, return_counts=True)[1]}\n"
                  f"y_val_label_list: {np.shape(y_val_label_list)}.  "
                  f"count {np.unique(y_val_label_list, return_counts=True)[1]}")
            y_train = to_categorical(y_train_label_list, num_classes=n_cats)
            y_val = to_categorical(y_val_label_list, num_classes=n_cats)
    else:
        x_train = x_data
        y_train = y_data

    n_items = len(y_train)

    # fix random seed for reproducability during development - not for simulations
    if test_run:
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
            print(f"\nRESHAPING x_train to: {np.shape(x_train)}")
            if use_val_data:
                x_val = x_val.reshape(x_val.shape[0], width, height, 1)

    """# # # proprocessing here - put data in range 0-1
    # if len(np.shape(x_train)) == 2:
    #     scaler = MinMaxScaler()
    #     x_train = scaler.fit_transform(x_train)
    #     if use_val_data is True:
    #         x_val = scaler.fit_transform(x_val)"""

    # # save path
    exp_cond_path = os.path.join(exp_root, exp_name, output_filename)
    if test_run:
        exp_cond_path = os.path.join(exp_cond_path, 'test')

    if not os.path.exists(exp_cond_path):
        os.makedirs(exp_cond_path)
    os.chdir(exp_cond_path)
    print(f"\nsaving to: {exp_cond_path}")


    # # The Model
    if model_dir in ['mlp', 'mlps']:
        print("\nloading an mlp model")
        augmentation = False
        model_dict = {'mlp': mlp,
                      'fc1': fc1,
                      'fc2': fc2,
                      'fc4': fc4}

        model = model_dict[model_name].build(features=x_size, classes=n_cats,
                                             n_layers=n_layers,
                                             units_per_layer=units_per_layer,
                                             act_func=act_func,
                                             use_bias=use_bias,
                                             y_1hot=y_1hot,
                                             output_act=output_act,
                                             batch_size=batch_size,
                                             weight_init=weight_init,
                                             batch_norm=use_batch_norm,
                                             dropout=use_dropout)
        # augmentation = False
        # units_per_layer = 32
        # # models[model_name].build(...)
        # elif model_name == 'fc4':
        #     build_model = fc4.build(classes=n_cats, units_per_layer=units_per_layer,
        #                             batch_norm=use_batch_norm, dropout=use_dropout)
        # elif model_name == 'fc2':
        #     build_model = fc2.build(classes=n_cats, units_per_layer=units_per_layer,
        #                             batch_norm=use_batch_norm, dropout=use_dropout)
        # elif model_name == 'fc1':
        #     build_model = fc1.build(classes=n_cats, units_per_layer=units_per_layer,
        #                             batch_norm=use_batch_norm, dropout=use_dropout)

    elif model_dir in ['cnn', 'cnns']:
        print("loading a cnn model")

        model_dict = {'con6_pool3_fc1': con6_pool3_fc1,
                      'con4_pool2_fc1': con4_pool2_fc1,
                      'con2_pool2_fc1': con2_pool2_fc1,
                      'con4_pool2_fc1_reluconv': con4_pool2_fc1_reluconv,
                      'con4_pool2_fc1_noise_layer': con4_pool2_fc1_noise_layer}

        units_per_layer = None
        width, height = data_dict['image_dim']
        depth = 3
        if grey_image:
            depth = 1

        model = model_dict[model_name].build(width=width, height=height, depth=depth, classes=n_cats,
                                             batch_norm=use_batch_norm, dropout=use_dropout)


    elif 'rnn' in model_dir:
        print("loading a recurrent model")
        augmentation = False
        model_dict = {'Bowers14rnn': Bowers14rnn,
                      'SimpleRNNn': SimpleRNNn,
                      'GRUn': GRUn,
                      'LSTMn': LSTMn,
                      'Seq2Seq': Seq2Seq}


        model = model_dict[model_name].build(features=x_size, classes=n_cats, timesteps=timesteps,
                                             batch_size=batch_size, n_layers=n_layers,
                                             serial_recall=serial_recall,
                                             units_per_layer=units_per_layer, act_func=act_func,
                                             y_1hot=serial_recall,
                                             dropout=use_dropout)
    else:
        print("model_dir not recognised")

    # model = build_model

    # loss
    loss_func = 'categorical_crossentropy'
    if n_cats == 2:
        loss_func = 'binary_crossentropy'

    # optimizer
    if use_optimizer in ['sgd', 'SGD']:
        this_optimizer = SGD(lr=lr)
    elif use_optimizer in ['sgd_decay', 'SGD_decay']:
        this_optimizer = SGD(lr=lr, decay=lr / max_epochs)
    elif use_optimizer == 'adam':
        this_optimizer = Adam(lr=lr)
    elif use_optimizer == 'rmsprop':
        this_optimizer = RMSprop(lr=lr, decay=1e-6)
    else:
        raise ValueError(f'use_optimizer not recognized: {use_optimizer}')


    # # compile model
    model.compile(loss=loss_func, optimizer=this_optimizer, metrics=['accuracy'])


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

    # # early stop acc
    # # should stop when acc reaches 1.0 (e.g., will not carry on training)
    early_stop_acc = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=.1,
                                                      patience=1, baseline=1.0)

    val_early_stop_acc = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=.1,
                                                          patience=1, baseline=1.0)

    date_n_time = int(datetime.datetime.now().strftime("%Y%m%d%H%M"))
    tensorboard_path = os.path.join(exp_cond_path, 'tb', str(date_n_time))

    tensorboard = TensorBoard(log_dir=tensorboard_path,
                              # histogram_freq=1,
                              # batch_size=batch_size,
                              # write_graph=True,
                              # # write_grads=False,
                              # # write_images=False,
                              # # embeddings_freq=0,
                              # # embeddings_layer_names=None,
                              # # embeddings_metadata=None,
                              # # embeddings_data=None,
                              # update_freq='epoch',
                              # profile_batch=2
                              )

    print('\n\nto access tensorboard, in terminal use\n'
          f'tensorboard --logdir={tensorboard_path}'
          '\nthen click link''')

    callbacks_list = [early_stop_plateau, early_stop_acc, checkpointer, tensorboard]
    val_callbacks_list = [val_early_stop_plateau, val_early_stop_acc, checkpointer, tensorboard]

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

    print("\n**** TRAINING COMPLETE ****")
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
    plt.savefig(str(output_filename) + '_training.png')


    # # Training info
    print(f"Model name: {checkpoint_path}")

    # # get best epoch
    if use_val_data:
        print(fit_model.history['val_loss'])
        trained_for = int(fit_model.history['val_loss'].index(min(fit_model.history['val_loss'])))
        end_val_loss = float(fit_model.history['val_loss'][trained_for])
        end_val_acc = float(fit_model.history['val_acc'][trained_for])
    else:
        print(fit_model.history['loss'])
        trained_for = int(fit_model.history['loss'].index(min(fit_model.history['loss'])))
        end_val_loss = np.nan
        end_val_acc = np.nan

    end_loss = float(fit_model.history['loss'][trained_for])
    end_acc = float(fit_model.history['acc'][trained_for])
    print(f'\nTraining Info\nbest loss after {trained_for} epochs\n'
          f'end loss: {end_loss}\nend acc: {end_acc}\n'
          f'end val loss: {end_val_loss}\nend val acc: {end_val_acc}')


    # # # PART 3 get_scores() # # #
    # # these three lines are to re-shape MNIST
    if len(np.shape(x_data)) != 4:
        if model_dir == 'cnn':
            x_data = x_data.reshape(x_data.shape[0], width, height, 1)

    predicted_outputs = model.predict(x_data)  # use x_data NOT x_train to fit shape of y_df
    item_correct_df, scores_dict, incorrect_items = get_scores(predicted_outputs, y_df, output_filename,
                                                               verbose=True, save_all_csvs=True)

    if verbose:
        focussed_dict_print(scores_dict, 'Scores_dict')


    trained_date = int(datetime.datetime.now().strftime("%y%m%d"))
    trained_time = int(datetime.datetime.now().strftime("%H%M"))
    model_info['overview'] = {'model_type': model_dir,
                              'model_name': model_name,
                              "trained_model": checkpoint_path,
                              "n_layers": n_layers,
                              "units_per_layer": units_per_layer,
                              "act_func": act_func,
                              "optimizer": use_optimizer,
                              "use_bias": use_bias,
                              "weight_init": weight_init,

                              "y_1hot": y_1hot, "output_act": output_act,
                              "lr": lr, "max_epochs": max_epochs,

                              "batch_size": batch_size,
                              "use_batch_norm": use_batch_norm,
                              "use_dropout": use_dropout,

                              "use_val_data": use_val_data,

                              "augmentation": augmentation,
                              "grey_image": grey_image,
                              "loss_target": loss_target,
                              "min_loss_change": min_loss_change,
                              'timesteps': timesteps
                              }


    git_repository = '/home/nm13850/Documents/PhD/code/library'
    if os.path.isdir('/Users/nickmartin/Documents/PhD/code/library'):
        git_repository = '/Users/nickmartin/Documents/PhD/code/library'

    repo = git.Repo(git_repository)

    sim_dict_name = f"{output_filename}_sim_dict.txt"

    # # simulation_info_dict
    sim_dict = {"topic_info": {"output_filename": output_filename, "cond": cond, "run": run,
                               "data_dict_path": data_dict_path, "model_path": model_path,
                               "exp_cond_path": exp_cond_path,
                               'exp_name': exp_name, 'cond_name': cond_name},
                "data_info": data_dict,
                "model_info": model_info,
                'scores': scores_dict,
                "training_info": {"sim_dict_name": sim_dict_name,
                                  "trained_for": trained_for,
                                  "loss": end_loss, "acc": end_acc, 'use_val_data': use_val_data,
                                  "end_val_acc": end_val_acc, "end_val_loss": end_val_loss,
                                  "trained_date": trained_date, "trained_time": trained_time,
                                  'x_data_path': x_data_path, 'y_data_path': y_data_path,
                                  'tensorboard_path': tensorboard_path,
                                  'commit': repo.head.object.hexsha,
                                  }
                }


    focussed_dict_print(sim_dict, 'sim_dict')

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
                     model_info['layers']['totals']['all_layers'],
                     model_info['layers']['totals']['hid_layers'],

                     model_info['layers']['hid_layers']['hid_totals']['act_layers'],
                     model_info['layers']['hid_layers']['hid_totals']['dense_layers'],
                     str_upl,
                     model_info['layers']['hid_layers']['hid_totals']['conv_layers'],
                     str_fpl,
                     model_info['layers']['hid_layers']['hid_totals']['analysable'],
                     use_optimizer, use_batch_norm, use_dropout, batch_size, augmentation, grey_image,
                     use_val_data, loss_target, min_loss_change,
                     max_epochs, trained_for, end_acc, end_loss, end_val_acc, end_val_loss,
                     checkpoint_path, trained_date, trained_time,
                     ]


    exp_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    os.chdir(exp_path)
    print(f"save_summaries: {exp_path}")

    # check if training_info.csv exists
    if not os.path.isfile(f"{exp_name}_training_summary.csv"):

        headers = ["file", "cond", "run",
                   "dataset", "x_size", "n_cats", 'timesteps', "n_items",
                   "model_type", "model", "all_layers", 'hid_layers',
                   "act_layers",
                   "dense_layers", "UPL", "conv_layers", "FPL", "analysable",
                   "optimizer", "batch_norm", "dropout", "batch_size", "aug", "grey_image",
                   "val_data", "loss_target", "min_loss_change",
                   "max_epochs", "trained_for", "end_acc", "end_loss", "end_val_acc", "end_val_loss",
                   "model_file", "date", "time"]

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

    print("\nff_sim finished")

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
