import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import csv
import pickle
import os.path
import datetime
import copy
import sys
sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, focussed_dict_print
from nick_data_tools import load_x_data, load_y_data

import keras
from keras.datasets import cifar10
######################
# functions
#####################
# todo: only save summary docs as csv.  all other output should be numpy, pickle or excel.
#  or for csv use this https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv
'''pretty similar to script from previous day - but with hashed-out code deleted'''

def ff_conv_colour_sim(topic_name, dataset,
                       cond='date', run='time',
                       hid_layers=4, hid_units=12, act_func='relu',
                       max_epochs=50, use_optimizer='sgd',
                       loss_target=0.01, min_loss_change=0.001,
                       use_batch_norm=False, use_val_data=False,
                       verbose=False):

    """ff_conv_colour_sim
    loss_target: stop training when this target is reached
    min_loss_change:# stop training of loss does not improve by this much


    1. get details
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
    """
    sim_date = int(datetime.datetime.now().strftime("%y%m%d"))
    sim_time = int(datetime.datetime.now().strftime("%H%M"))
    if cond is 'date':
        cond = sim_date
    if run is 'time':
        run = sim_time

    # # get info from dict
    if type(dataset) is dict:
        data_dict = dataset
    elif type(dataset) is str:
        data_dict = load_dict(dataset)
    else:
        print("ERROR - expected dict or str(dict_name) as dataset")

    if verbose is True:
        # # print basic details
        print("ff_conv_colour_sim set to run with these settings...")
        print("topic_name: {}\ndataset: {}\ncond: {}\nrun: {}\nmax_epochs: {}\nhid_layers: {}\n"
              "hid_units: {}\nact_func: {}\nuse_optimizer: {}".format(topic_name, dataset, cond, run, max_epochs,
                                                                      hid_layers, hid_units, act_func,
                                                                      use_optimizer))
        print("loss_target: {}\nmin_loss_change: {}".format(loss_target, min_loss_change))
        print("batch_norm: {}\nval_data: {}".format(use_batch_norm, use_val_data))
        focussed_dict_print(data_dict)


    # # # load datasets
    # x_data = load_data(data_dict, 'x')
    #
    # # todo:  add proprocessing here - put data in range 0-1.  PER COLUMN/FEATURE?
    #
    # y_dict = load_data(data_dict, 'y')
    # y_df = y_dict['y_df']
    # y_label_list = y_dict['y_label_list']

    # # check for training data
    if 'train_set' in data_dict:
        x_data_path = os.path.join(data_dict['data_path'], data_dict['train_set']['X_data'])
        y_data_path = os.path.join(data_dict['data_path'], data_dict['train_set']['Y_labels'])
    else:
        x_data_path = os.path.join(data_dict['data_path'], data_dict['X_data'])
        y_data_path = os.path.join(data_dict['data_path'], data_dict['Y_labels'])

    x_data = load_x_data(x_data_path)
    y_df, y_label_list = load_y_data(y_data_path)

    n_cats = data_dict["n_cats"]
    y_data = to_categorical(y_label_list, num_classes=n_cats)

    # # val data
    if use_val_data is True:
        print("\nLoading validation data")
        if "val_set" in data_dict.keys():
            # val_x_data = load_data(data_dict, 'x', test_train_val='val_set')
            # val_y_dict = load_data(data_dict, 'y', test_train_val='val_set')
            # val_y_df = val_y_dict['y_df']
            # val_y_label_list = val_y_dict['y_label_list']

            val_x_data = load_x_data(os.path.join(data_dict['data_path'], data_dict['val_set']['X_data']))
            val_y_df, val_y_label_list = load_y_data(os.path.join(data_dict['data_path'], data_dict['val_set']['Y_labels']))
            val_y_data = to_categorical(val_y_label_list, num_classes=n_cats)
        else:
            print("validation data not found - performing split")
            x_data, val_x_data, y_label_list, val_y_label_list = train_test_split(x_data, y_label_list, test_size=0.2,
                                                                                   random_state=1)
            print("y_label_list: {}.  Count: {}\nval_y_label_list: {}.  count {}".format(
                np.shape(y_label_list), np.unique(y_label_list, return_counts=True)[1],
                np.shape(val_y_label_list), np.unique(val_y_label_list, return_counts=True)[1]))
            y_data = to_categorical(y_label_list, num_classes=n_cats)
            val_y_data = to_categorical(val_y_label_list, num_classes=n_cats)


    # Output files
    output_filename = "{}_{}r{}".format(topic_name, cond, run)
    print("Output file: " + output_filename)

    # fix random seed for reproducability during development - not for simulations
    seed = 7
    np.random.seed(seed)

    x_size = data_dict["X_size"]
    image_dim = data_dict['image_dim']
    n_items, width, height, channels = np.shape(x_data)

    units_per_layer = [32, 64, 64, 128, 128, 128, 512]  # int(hid_units / hid_layers)

    # # # 3. compile model
    batch_size = 32
    num_classes = 10
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'



    #     """check out https://github.com/abhijeet3922/Object-recognition-CIFAR-10/blob/master/cifar10_90.py"""
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself

    # todo: make model objects for each mode (e.g., smallVGG.py)
    # todo: make version with all convs (conv1 not fc layers).
    #  https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11
    # todo:  make version with global average pooling instead of max pool
    # todo: model info is saving the wrong things? - make sure I have my model details, not just the config

    # load model from his script
    import sys
    sys.path.insert(0, '/home/nm13850/Documents/PhD/Python/learning_new_functions/CNN_sim_script/conv_march_2019/'
                       'conv_tutorial3/')
    from pyimagesearch.smallvggnet import SmallVGGNet

    trainX = x_data
    trainY = y_data
    testX = val_x_data
    testY = val_y_data

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    # initialize our VGG-like Convolutional Neural Network
    model_type = 'conv'
    model_name = 'SmallVGGNet'
    model = SmallVGGNet.build(width=32, height=32, depth=3,
                              classes=n_cats)

    # initialize our initial learning rate, # of epochs to train for,
    # and batch size
    INIT_LR = 0.01  # initialize learning rate
    EPOCHS = max_epochs
    BS = batch_size  # batchsize

    # initialize the model and optimizer (you'll want to use
    # binary_crossentropy for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1)
                                # , target_names=list(range(n_cats))
                                ))

    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (SmallVGGNet)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("training loaded smallVGG.png")

    # save the model and label binarizer to disk
    # print("[INFO] serializing network and label binarizer...")
    # model.save("conv_tut3_load_smallVGG.h5")


    """from my owld script"""
    trained_for = len(H.history['loss'])
    end_accuracy = np.around(H.history['acc'][-1], decimals=3)
    end_loss = np.around(H.history['loss'][-1], decimals=3)

    stopped_because = "max_epochs"
    # TODO: callbacks?
    # if H.history['acc'][-1] > .999:
    #     stopped_because = "max_acc"
    # else:
    #     stopped_because = "plateaued"

    end_val_acc = -999
    end_val_loss = -999
    if 'val_set' in data_dict:
        if use_val_data is True:
            end_val_acc = np.around(H.history['val_acc'][-1], decimals=3)
            end_val_loss = np.around(H.history['val_loss'][-1], decimals=3)


    # save the model (and weights) after training
    # todo: link this back to something useful (output filename
    model_trained = "{}_model.h5".format("conv_tut3_load_smallVGG")
    model.save(model_trained)


    # # simulation_info_dict
    model_from = 'https://www.pyimagesearch.com/2018/09/10/' \
                 'keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/'


    sim_dict = {"topic_info": {"topic_name": topic_name, "cond": cond, "run": run, "output_filename": output_filename},
                "data_info": data_dict,
                "model_info": {'model_type': model_type, 'model_name': model_name,
                               "hid_layers": hid_layers, "hid_units": hid_units, "units_per_layer": units_per_layer,
                               "act_func": act_func, "optimizer": use_optimizer, "use_batch_norm": use_batch_norm,
                               "model_trained": model_trained,
                               'model_from': model_from},
                "training_info": {"max_epochs": max_epochs, "trained_for": trained_for, "use_val_data": use_val_data,
                                  "stopped_because": stopped_because, "loss": end_loss, "acc": end_accuracy,
                                  "end_val_acc": end_val_acc, "end_val_loss": end_val_loss,
                                  "sim_date": sim_date, "sim_time": sim_time}
                }

    sim_dict_name = "{}_sim_dict.pickle".format(output_filename)

    pickle_out = open(sim_dict_name, "wb")
    pickle.dump(sim_dict, pickle_out)
    pickle_out.close()

    # record training info comparrisons
    training_info = [cond, run, max_epochs, hid_layers, hid_units, units_per_layer, act_func,
                     dataset, x_size, n_cats, n_items, int(n_items/n_cats), use_batch_norm,  use_val_data,
                     use_optimizer, trained_for, stopped_because, end_accuracy, end_loss, end_val_acc, end_val_loss]

    # check if training_info.csv exists
    if not os.path.isfile("{}_training_summary.csv".format(topic_name)):
        training_overview = open("{}_training_summary.csv".format(topic_name), 'w')
        mywriter = csv.writer(training_overview)
        headers = ["cond", "run", "max_epochs", "hid_layers", "hid_units", "units_per_layer", "act_func",
                   "dataset", "x_size", "n_cats", "n_items", "items_per_cat", "use_batch_norm",  "use_val_data",
                   "use_optimizer", "trained_for", "stopped_because", "accuracy", "loss", "end_val_acc",
                   "end_val_loss"]
        mywriter.writerow(headers)
    else:
        training_overview = open("{}_training_summary.csv".format(topic_name), 'a')
        mywriter = csv.writer(training_overview)

    mywriter.writerow(training_info)
    training_overview.close()

    if verbose is True:
        focussed_dict_print(sim_dict)

    print("\nff_conv_colour_sim finished")

    return training_info, sim_dict


######################
# training_info, sim_dict = ff_conv_colour_sim(topic_name='CNN_test', dataset='CIFAR_10_2019_load_dict',
#                                              max_epochs=10,
#                                              hid_layers={'conv': 6, 'fc': 1}, hid_units=1056,
#                                              act_func='relu',
#                                              verbose=True, use_val_data=True)
######################