import csv
import pickle
import os.path
import datetime
import copy
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model

sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, focussed_dict_print, print_nested_round_floats
from nick_data_tools import load_x_data, load_y_data, get_dset_path
from nick_network_tools import get_scores




########################
def kernel_to_2d(layer_activation_4d, reduce_type='max', verbose=False):
    """
    To perform selectivity analysis 'per unit', 4d layer need to be reduced to 2d.
    where shape is (items, width, height, depth/n_kernels) 
    convert to (items, n_kernels)
    
    :param layer_activation_4d: the GHA of a filter/kernel (conv/pool) layer with 4d (e.g., shape: (1, 2, 3, 4))
    :param reduce_type: the method for simplifying the kernel e.g. max, mean etc
    :param verbose: whether to print intermediate steps to screen

    :return: 2d hid acts - 1 float per kernel per item
    """
    print('\n**** kernel_to_2d GHA() ****')

    # TODO  CAN THIS BE SPED UP WITH NUMPY

    items, width, height, kernels = np.shape(layer_activation_4d)

    if verbose:
        print("\t{} kernels, shape ({}, {})".format(kernels, width, height))

    # # to save all item averages per conv filter make: layer_mean_acts
    layer_mean_acts = np.empty((items, 0))

    # # loop through conv filters
    for kernel in range(kernels):
        this_kernel = layer_activation_4d[:, :, :, kernel]

        # # to save averages per item as computed
        kernel_means = []
        for item in range(items):
            kernel_acts = this_kernel[item]

            if reduce_type is 'mean':
                kernel_mean = np.mean(kernel_acts)

            else:  # use max
                kernel_mean = np.amax(kernel_acts)

            kernel_means.append(kernel_mean)

        # # append column to layer means
        layer_mean_acts = np.column_stack((layer_mean_acts, kernel_means))

        if verbose:
            print("\t{}. layer_mean_acts: {} {}".format(kernel, np.shape(layer_mean_acts), type(layer_mean_acts)))

    return layer_mean_acts

######################


def ff_gha(exp_cond_path,
           get_classes=("Conv2D", "Dense", "Activation"),
           gha_incorrect=True,
           use_dataset='train_set',
           save_2d_layers=True,
           save_4d_layers=False,
           exp_root='/home/nm13850/Documents/PhD/python_v2/experiments/',
           verbose=False,
           test_run=False
           ):

    """
    gets activations from hidden units.

    1. load simulation dict (with data info) (*_load_dict.pickle)
        sim_dict can be fed in from sim script, or loaded separately
    2. load model - get structure and details
    3. run dataset through once, recording accuracy per item/class
    4. run on 2nd model to get hid acts

    :param exp_cond_path: path of the experiment and condition folders (exp_folder/cond_folder)
    :param get_classes: which types of layer are we interested in?
    :param gha_incorrect: GHA for ALL items (True) or just correct items (False)
    :param use_dataset: GHA for train/test data
    :param save_2d_layers: get 1 value per kernel for conv/pool layers
    :param save_4d_layers: keep original shape of conv/pool layers (for other analysis maybe?)
    :param exp_root: root to save experiments
    :param verbose:
    :param test_run: Set test = True to just do one unit per layer


    :return: dict with hid acts per layer.  saved as dict so different shaped arrays don't matter too much
    """

    print('**** ff_gha GHA() ****')

    # # sort path to data
    print("exp_cond_path: {}".format(exp_cond_path))

    full_exp_cond_path = os.path.join(exp_root, exp_cond_path)
    if not os.path.exists(full_exp_cond_path):
        print("ERROR - path for this experiment not found")
    set_path = os.chdir(full_exp_cond_path)
    print("set_path to full_exp_cond_path: {}".format(full_exp_cond_path))

    exp_dir, cond_dir = os.path.split(exp_cond_path)
    sim_dict = cond_dir + '_sim_dict'
    print("sim_dict: {}".format(sim_dict))


    # # # PART 1 # # #
    # # load details from dict
    if type(sim_dict) is str:
        sim_dict = load_dict(sim_dict)

    # if verbose is True:
    #     print("\n**** Sim Dictionary ****")
    focussed_dict_print(sim_dict, 'sim_dict')

    # # # load datasets
    # # get location of data to load from get dset path

    # data_path = get_dset_path(cond_dir)
    # if data_path is not None:
    #     print("data_path: {}".format(data_path))
    #     sim_dict['topic_info']['dataset_path'] = data_path
    #     sim_dict['topic_info']['exp_cond_path'] = exp_cond_path
    # else:
    #     if 'dataset_path' in sim_dict['topic_info'].keys():
    #         data_path = sim_dict['topic_info']['dataset_path']
    #     else:
    #         raise TypeError("CAN@T FIND DATA")
    #         sys.exit()
    #
    # # # don't use 'load_from_datasets' its slow,
    # # x_data = load_from_datasets(data_path, 'x', use_dataset)
    #
    # # # # load datasets
    # # # get location of data to load from get dset path
    # if 'data_path' in sim_dict['data_info']:
    #     data_path = sim_dict['data_info']['data_path']
    # elif 'dataset_path' in sim_dict['topic_info']:
    #     dataset_path = sim_dict['topic_info']['dataset_path']
    #     data_path, dset_name = os.path.split(dataset_path)
    # else:
    #     data_path = input("\n\n~*~*~*~*~*~*~*~enter path to dir containing data:")
    #
    # if 'dataset' not in data_path:
    #     dataset_root = '/home/nm13850/Documents/PhD/python_v2/datasets/'
    #     data_path = os.path.join(dataset_root, data_path)
    #
    # if 'X_data' in sim_dict['data_info']:
    #     x_filename = sim_dict['data_info']['X_data']
    #     y_filename = sim_dict['data_info']['Y_labels']
    #
    # elif use_dataset in sim_dict['data_info']:
    #     x_filename = sim_dict['data_info'][use_dataset]['X_data']
    #     y_filename = sim_dict['data_info'][use_dataset]['Y_labels']
    # else:
    #     print("use_dataset: ", use_dataset)
    #     print("\nsim_dict{...")
    #     focussed_dict_print(sim_dict['data_info'])
    #     dict_for_data_names = input("\n\n~*~*~*~*~*~*~*~enter sim dict path for data names: ")
    #     # x_filename = input("\n\n~*~*~*~*~*~*~*~enter x_filename: ")
    #     x_filename = dict_for_data_names['X_data']
    #     y_filename = dict_for_data_names['Y_labels']
    #
    # x_data_path = os.path.join(data_path, x_filename)
    # print("data_path: {}\nx_data_path:{}\nx_filename:{}".format(data_path, x_data_path, x_filename))
    # x_data = load_x_data(x_data_path)
    #
    # # y_dict = load_from_datasets(data_path, 'y', use_dataset)
    # y_data_path = os.path.join(data_path, y_filename)
    #
    # y_df, y_label_list = load_y_data(y_data_path)
    # # y_df = y_dict['y_df']

    # # check for training data
    if use_dataset in sim_dict['data_info']:
        x_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info'][use_dataset]['X_data'])
        y_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info'][use_dataset]['Y_labels'])
        print("\nloading {}\nx_data_path: {}\ny_data_path: {}".format(use_dataset, x_data_path, y_data_path))
    elif use_dataset == 'train_set':
        x_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info']['X_data'])
        y_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info']['Y_labels'])
        print("\nloading {} (only dset available):".format(use_dataset))
    else:
        print("\nERROR! requested dataset ({}) not found in dict:".format(use_dataset))
        focussed_dict_print(sim_dict['data_info'])

    x_data = load_x_data(x_data_path)
    y_df, y_label_list = load_y_data(y_data_path)

    # y_dict = load_y_data(y_data_path)
    #
    # y_df = y_dict['y_df']
    # y_label_list = y_dict['y_label_list']
    n_cats = sim_dict['data_info']["n_cats"]


    # # data preprocessing
    # # if network is cnn but data is 2d (e.g., MNIST)
    if len(np.shape(x_data)) != 4:
        if sim_dict['model_info']['overview']['model_type'] == 'cnn':
            width, height = sim_dict['data_info']['image_dim']
            x_data = x_data.reshape(x_data.shape[0], width, height, 1)
            print("\nRESHAPING x_data to: {}".format(np.shape(x_data)))


    # Output files
    output_filename = sim_dict["topic_info"]["output_filename"]
    print("\nOutput file: " + output_filename)


    # # # # PART 2 # # #
    print("\n**** THE MODEL ****")
    model_name = sim_dict['model_info']['overview']['trained_model']
    model_path = os.path.join(full_exp_cond_path, model_name)
    loaded_model = load_model(model_path)
    model_details = loaded_model.get_config()
    print_nested_round_floats(model_details)

    n_layers = len(model_details['layers'])
    model_dict = dict()

    # # turn off "trainable" and get useful info
    for layer in range(n_layers):
        # set to not train
        model_details['layers'][layer]['config']['trainable'] = 'False'

        if verbose:
            print("Model layer {}: {}".format(layer, model_details['layers'][layer]))

        # # get useful info
        layer_dict = {'layer': layer,
                      'name': model_details['layers'][layer]['config']['name'],
                      'class': model_details['layers'][layer]['class_name']}

        if 'units' in model_details['layers'][layer]['config']:
            layer_dict['units'] = model_details['layers'][layer]['config']['units']
        if 'activation' in model_details['layers'][layer]['config']:
            layer_dict['act_func'] = model_details['layers'][layer]['config']['activation']
        if 'filters' in model_details['layers'][layer]['config']:
            layer_dict['filters'] = model_details['layers'][layer]['config']['filters']
        if 'kernel_size' in model_details['layers'][layer]['config']:
            layer_dict['size'] = model_details['layers'][layer]['config']['kernel_size'][0]
        if 'pool_size' in model_details['layers'][layer]['config']:
            layer_dict['size'] = model_details['layers'][layer]['config']['pool_size'][0]
        if 'strides' in model_details['layers'][layer]['config']:
            layer_dict['strides'] = model_details['layers'][layer]['config']['strides'][0]
        if 'rate' in model_details['layers'][layer]['config']:
            layer_dict["rate"] = model_details['layers'][layer]['config']['rate']

        # # set and save layer details
        model_dict[layer] = layer_dict

    # # my model summary
    model_df = pd.DataFrame.from_dict(data=model_dict, orient='index',
                                      columns=['layer', 'name', 'class', 'act_func',
                                               'units', 'filters', 'size', 'strides', 'rate'], )

    # # just classes of layers specified in get_classes
    key_layers_df = model_df.loc[model_df['class'].isin(get_classes)]
    key_layers_df.reset_index(inplace=True)
    del key_layers_df['index']
    key_layers_df.index.name = 'index'
    key_layers_df = key_layers_df.drop(columns=['size', 'strides', 'rate'])

    # # add column ('n_units_filts')to say how many things needs gha per layer (number of units or filters)
    # # add zeros to rows with no units or filters
    key_layers_df.loc[:, 'n_units_filts'] = key_layers_df.units.fillna(0) + key_layers_df.filters.fillna(0)

    print("\nkey_layers_df:\n{}".format(key_layers_df))


    # # # add n_cats to output layer
    # # key_layers_df.loc[key_layers_df['class'] == 'Activation', 'n_units_filts'] = n_cats
    # # todo: if I just want to add the number of units to the final output layer I can just count the number of
    # #  activation layers and only apply this to the last one.
    # act_indices = key_layers_df.index[key_layers_df['class'] == 'Activation'].tolist()
    # prev_indices = [i-1 for i in act_indices]  # get index of activation layers
    # # # get n_unit_filts values for activation layers-1
    # prev_filt_values = pd.Series(key_layers_df['n_units_filts'], index=prev_indices)
    # # # update activation n_units_fils values to be same as precceding layer
    # for index, (prev_row, new_value) in enumerate(prev_filt_values.iteritems()):
    #     act_row = act_indices[index]
    #     key_layers_df.at[act_row, 'n_units_filts'] = new_value

    key_layers_df.loc[:, "n_units_filts"] = key_layers_df["n_units_filts"].astype(int)

    # # get to total number of units or filters in key layers of the network
    key_n_units_fils = sum(key_layers_df['n_units_filts'])

    print("\nkey_layers_df:\n{}".format(key_layers_df.head()))
    # print("\nkey_layers_df:\n{}".format(key_layers_df))
    print("key_n_units_fils: ", key_n_units_fils)

    '''i currently get output layer, make sure I keep this in to make sure I can do class correlation'''

    # # # set dir to save gha stuff # # #
    hid_act_items = 'all'
    if gha_incorrect == False:
        hid_act_items = 'correct'

    gha_folder = '{}_{}_gha'.format(hid_act_items, use_dataset)

    # todo: add if test run is true, save as test run - check this works
    if test_run:
        gha_folder = os.path.join(gha_folder, 'test')
    gha_path = os.path.join(full_exp_cond_path, gha_folder)

    if not os.path.exists(gha_path):
        os.makedirs(gha_path)
    save_hid_acts = os.chdir(gha_path)
    print("saving hid_acts to: {}".format(gha_path))

    # # # PART 3 get_scores() # # #
    predicted_outputs = loaded_model.predict(x_data)

    item_correct_df, scores_dict, incorrect_items = get_scores(predicted_outputs, y_df, output_filename,
                                                               save_all_csvs=True)

    if verbose:
        print("\n****Scores_dict****")
        focussed_dict_print(scores_dict)


    # # # PART 4 # # #
    print("\n**** REMOVE INCORRECT FROM X DATA ****")
    mask = np.ones(len(x_data), dtype=bool)
    mask[incorrect_items] = False
    x_correct = copy.copy(x_data[mask])

    gha_items = x_correct
    if gha_incorrect:  # If I want ALL items including those classified incorrectly
        gha_items = x_data
    print("gha_items: (incorrect items={}) {}".format(gha_incorrect, np.shape(gha_items)))





    # # PART 5
    print("\n**** Get Hidden unit activations ****")
    hid_act_2d_dict = dict()        # # to use to get 2d hid acts (e.g., means from 4d layers)
    hid_act_any_d_dict = dict()     # # to use to get all hid acts (e.g., both 2d and 4d layers)

    # # loop through key layers df
    gha_key_layers = []
    for index, row in key_layers_df.iterrows():
        if test_run:
            if index > 3:
                continue

        layer_number, layer_name, layer_class = row['layer'], row['name'], row['class']
        print("{}. name {} class {}".format(layer_number, layer_name, layer_class))

        if layer_class not in get_classes:  # skip layers/classes not in list
            continue
        else:
            print('getting layer')
            converted_to_2d = False  # set to True if 4d acts have been converted to 2d
            model = loaded_model
            layer_name = layer_name
            gha_key_layers.append(layer_name)

            # model to record hid acts
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            intermediate_output = intermediate_layer_model.predict(gha_items, verbose=1)
            layer_acts_shape = np.shape(intermediate_output)

            if save_2d_layers:
                if len(layer_acts_shape) == 2:
                    acts_2d = intermediate_output

                elif len(layer_acts_shape) == 4:  # # call mean_act_conv
                    acts_2d = kernel_to_2d(intermediate_output, verbose=True)
                    layer_acts_shape = np.shape(acts_2d)
                    converted_to_2d = True

                else:
                    print("\n\n\n\nSHAPE ERROR - UNEXPECTED DIMENSIONS\n\n\n\n")
                    acts_2d = 'SHAPE_ERROR'
                    layer_acts_shape = 'NONE'

                hid_act_2d_dict[index] = {'layer_name': layer_name, 'layer_class': layer_class,
                                          "layer_shape": layer_acts_shape, '2d_acts': acts_2d}

                if converted_to_2d == True:
                    hid_act_2d_dict[index]['converted_to_2d'] = True

                print("\nlayer{}. hid_act_2d_dict: {}\n".format(index, layer_acts_shape))

                # # save distplot for sanity check
                sns.distplot(np.ravel(acts_2d))
                plt.title(str(layer_name))
                plt.savefig("{}_{}_layer_act_distplot.png".format(output_filename, layer_name))
                plt.close()


    print("\n**** saving info to summary page and dictionary ****")

    hid_act_filenames = {'2d': None, 'any_d': None}
    if save_2d_layers:
        dict_2d_save_name = '{}_hid_act_2d.pickle'.format(output_filename)
        with open(dict_2d_save_name, "wb") as pkl:  # 'wb' mean 'w'rite the file in 'b'inary mode
            pickle.dump(hid_act_2d_dict, pkl)
        # np.save(dict_2d_save_name, hid_act_2d_dict)
        hid_act_filenames['2d'] = dict_2d_save_name

    if save_4d_layers:
        dict_4dsave_name = '{}_hid_act_any_d.pickle'.format(output_filename)
        with open(dict_4dsave_name, "wb") as pkl:  # 'wb' mean 'w'rite the file in 'b'inary mode
            pickle.dump(hid_act_any_d_dict, pkl)
        # np.save(dict_4dsave_name, hid_act_any_d_dict)
        hid_act_filenames['any_d'] = dict_4dsave_name


    cond = sim_dict["topic_info"]["cond"]
    run = sim_dict["topic_info"]["run"]

    upl_list = sim_dict["model_info"]['layers']['hid_layers']["hid_totals"]['UPL'][:-1]
    hid_units = sum(upl_list)
    # # todo: hid_units: how should I now express this?  Maybe a single value adding all
    # #  kernels and units excluding last layer?

    trained_for = sim_dict["training_info"]["trained_for"]
    end_accuracy = sim_dict["training_info"]["acc"]
    dataset = sim_dict["data_info"]["dataset"]
    gha_date = int(datetime.datetime.now().strftime("%y%m%d"))
    gha_time = int(datetime.datetime.now().strftime("%H%M"))

    gha_acc = scores_dict['gha_acc']
    n_cats_correct = scores_dict['n_cats_correct']

    # # GHA_info_dict
    gha_dict_name = "{}_GHA_dict.pickle".format(output_filename)
    gha_dict_path = os.path.join(gha_path, gha_dict_name)


    gha_dict = {"topic_info": sim_dict['topic_info'],
                "data_info": sim_dict['data_info'],
                "model_info": sim_dict['model_info'],
                "training_info": sim_dict['training_info'],
                "GHA_info": {"use_dataset": use_dataset,
                             'x_data_path': x_data_path,
                             'y_data_path': y_data_path,
                             'gha_path': gha_path,
                             'gha_dict_path': gha_dict_path,
                             "gha_incorrect": gha_incorrect,
                             "hid_act_files": hid_act_filenames,
                             'gha_key_layers': gha_key_layers,
                             'key_n_units_fils': key_n_units_fils,
                             "gha_date": gha_date, "gha_time": gha_time,
                             "scores_dict": scores_dict,
                             "model_dict": model_dict
                             }
                }

    # pickle_out = open(gha_dict_name, "wb")
    # pickle.dump(gha_dict, pickle_out)
    # pickle_out.close()

    with open(gha_dict_name, "wb") as pickle_out:
        pickle.dump(gha_dict, pickle_out)


    if verbose:
        print("\n*** gha_dict ***")
        focussed_dict_print(gha_dict, ['GHA_info', "scores_dict"])

    # make a list of dict names to do sel on
    if not os.path.isfile("{}_dict_list_for_sel.csv".format(output_filename)):
        dict_list = open("{}_dict_list_for_sel.csv".format(output_filename), 'w')
        mywriter = csv.writer(dict_list)
    else:
        dict_list = open("{}_dict_list_for_sel.csv".format(output_filename), 'a')
        mywriter = csv.writer(dict_list)

    mywriter.writerow([gha_dict_name[:-7]])
    dict_list.close()

    print("\nadded to list for selectivity analysis: {}".format(gha_dict_name[:-7]))

    gha_info = [cond, run, output_filename, n_layers, hid_units, dataset, use_dataset,
                gha_incorrect, n_cats, trained_for, end_accuracy, gha_acc, n_cats_correct]


    # # check if gha_summary.csv exists
    # # save summary file in exp folder (grandparent dir to gha folder: exp/cond/gha)
    # to move up to parent just use '..' rather than '../..'
    exp_name = exp_dir.strip('/')
    move_to_exp_folder = os.chdir('../..')
    exp_path = os.getcwd()

    if not os.path.isfile(exp_name + "_GHA_summary.csv"):
        gha_summary = open(exp_name + "_GHA_summary.csv", 'w')
        mywriter = csv.writer(gha_summary)
        summary_headers = ["cond", "run", 'filename', "n_layers", "hid_units", "dataset", "GHA_on",
                           'incorrect', "n_cats", "trained_for", "train_acc", "gha_acc", 'n_cats_correct']

        mywriter.writerow(summary_headers)
        print("creating summary csv at: {}".format(exp_path))

    else:
        gha_summary = open(exp_name + "_GHA_summary.csv", 'a')
        mywriter = csv.writer(gha_summary)
        print("appending to summary csv at: {}".format(exp_path))

    mywriter.writerow(gha_info)
    gha_summary.close()

    print("\nend of ff_gha")

    return gha_info, gha_dict


###############################
# # # GHA on its own
# gha_info, sim_dict = ff_gha(exp_cond_path='train_script_check/train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug',
#                             gha_incorrect=True, use_dataset='train_set',
#                             verbose=True,
#                             test_run=True)
