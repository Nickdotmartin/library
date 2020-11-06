import csv
import pickle
import os.path
import datetime
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.data import load_x_data, load_y_data, get_dset_path
from tools.network import get_scores


# todo: Since I take the max single value from each kernel,
#  I might as well just record from pooling layers,
#  just to save on the number of values I am throwing away/the size of the file?

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

    items, width, height, kernels = np.shape(layer_activation_4d)

    if verbose:
        print(f"\t{kernels} kernels, shape ({width}, {height})")

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
            print(f"\t{kernel}. layer_mean_acts: {np.shape(layer_mean_acts)} {type(layer_mean_acts)}")

    return layer_mean_acts


######################


def ff_gha(sim_dict_path,
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

    :param sim_dict_path: path to the dictionary for this experiment condition
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

    # # # PART 1 # # #
    # # load details from dict
    if type(sim_dict_path) is dict:
        sim_dict = sim_dict_path
        full_exp_cond_path = sim_dict['topic_info']['exp_cond_path']

    elif os.path.isfile(sim_dict_path):
        print(f"sim_dict_path: {sim_dict_path}")
        sim_dict = load_dict(sim_dict_path)
        full_exp_cond_path, sim_dict_name = os.path.split(sim_dict_path)

    elif os.path.isfile(os.path.join(exp_root, sim_dict_path)):
        sim_dict_path = os.path.join(exp_root, sim_dict_path)
        print(f"sim_dict_path: {sim_dict_path}")
        sim_dict = load_dict(sim_dict_path)
        full_exp_cond_path, sim_dict_name = os.path.split(sim_dict_path)
    else:
        raise FileNotFoundError(sim_dict_path)

    os.chdir(full_exp_cond_path)
    print(f"set_path to full_exp_cond_path: {full_exp_cond_path}")

    # exp_dir, _ = os.path.split(exp_cond_path)

    focussed_dict_print(sim_dict, 'sim_dict')

    # # # load datasets

    # # check for training data
    if use_dataset in sim_dict['data_info']:
        x_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info'][use_dataset]['X_data'])
        y_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info'][use_dataset]['Y_labels'])
        print(f"\nloading {use_dataset}\nx_data_path: {x_data_path}\ny_data_path: {y_data_path}")
    elif use_dataset == 'train_set':
        x_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info']['X_data'])
        y_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info']['Y_labels'])
        print(f"\nloading {use_dataset} (only dset available):")
    else:
        print(f"\nERROR! requested dataset ({use_dataset}) not found in dict:")
        focussed_dict_print(sim_dict['data_info'], "sim_dict['data_info']")
        if 'X_data' in sim_dict['data_info']:
            print(f"\nloading only dset available:")
            x_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info']['X_data'])
            y_data_path = os.path.join(sim_dict['data_info']['data_path'], sim_dict['data_info']['Y_labels'])

    x_data = load_x_data(x_data_path)
    y_df, y_label_list = load_y_data(y_data_path)

    n_cats = sim_dict['data_info']["n_cats"]

    # # data preprocessing
    # # if network is cnn but data is 2d (e.g., MNIST)
    # # old version
    # if len(np.shape(x_data)) != 4:
    #     if sim_dict['model_info']['overview']['model_type'] == 'cnn':
    #         width, height = sim_dict['data_info']['image_dim']
    #         x_data = x_data.reshape(x_data.shape[0], width, height, 1)
    #         print(f"\nRESHAPING x_data to: {np.shape(x_data)}")

    # new version
    print(f"\ninput shape: {np.shape(x_data)}")
    if len(np.shape(x_data)) == 4:
        image_dim = sim_dict['image_dim']
        n_items, width, height, channels = np.shape(x_data)
    else:
        # # this is just for MNIST
        if sim_dict['model_info']['overview']['model_type'] in ['cnn', 'cnns']:
            print("reshaping mnist for cnn")
            width, height = sim_dict['image_dim']
            x_data = x_data.reshape(x_data.shape[0], width, height, 1)
            print(f"\nRESHAPING x_data to: {np.shape(x_data)}")

        if sim_dict['model_info']['overview']['model_type'] == 'mlps':
            if len(np.shape(x_data)) > 2:
                print(f"reshaping image data from {len(np.shape(x_data))}d to 2d for mlp")
                x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1] * x_data.shape[2]))

                print(f"\nNEW input shape: {np.shape(x_data)}")
                x_size = np.shape(x_data)[1]
                print(f"NEW x_size: {x_size}")


    # Output files
    output_filename = sim_dict["topic_info"]["output_filename"]
    print(f"\nOutput file: {output_filename}")

    # # # # PART 2 # # #
    print("\n**** THE MODEL ****")
    model_name = sim_dict['model_info']['overview']['trained_model']
    model_path = os.path.join(full_exp_cond_path, model_name)
    loaded_model = load_model(model_path)
    model_details = loaded_model.get_config()
    print_nested_round_floats(model_details, 'model_details')

    n_layers = len(model_details['layers'])
    model_dict = dict()

    # # turn off "trainable" and get useful info
    for layer in range(n_layers):
        # set to not train
        model_details['layers'][layer]['config']['trainable'] = 'False'

        if verbose:
            print(f"Model layer {layer}: {model_details['layers'][layer]}")

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

    print(f"\nkey_layers_df:\n{key_layers_df}")

    key_layers_df.loc[:, "n_units_filts"] = key_layers_df["n_units_filts"].astype(int)

    # # get to total number of units or filters in key layers of the network
    key_n_units_fils = sum(key_layers_df['n_units_filts'])

    print(f"\nkey_layers_df:\n{key_layers_df.head()}")
    # print("\nkey_layers_df:\n{}".format(key_layers_df))
    print(f"key_n_units_fils: {key_n_units_fils}")

    '''i currently get output layer, make sure I keep this in to make sure I can do class correlation'''

    # # # set dir to save gha stuff # # #
    hid_act_items = 'all'
    if not gha_incorrect:
        hid_act_items = 'correct'

    gha_folder = f'{hid_act_items}_{use_dataset}_gha'

    if test_run:
        gha_folder = os.path.join(gha_folder, 'test')
    gha_path = os.path.join(full_exp_cond_path, gha_folder)

    if not os.path.exists(gha_path):
        os.makedirs(gha_path)
    os.chdir(gha_path)
    print(f"saving hid_acts to: {gha_path}")

    # # # PART 3 get_scores() # # #
    predicted_outputs = loaded_model.predict(x_data)

    item_correct_df, scores_dict, incorrect_items = get_scores(predicted_outputs, y_df, output_filename,
                                                               save_all_csvs=True, verbose=True)

    if verbose:
        focussed_dict_print(scores_dict, 'Scores_dict')

    # # # PART 4 # # #
    print("\n**** REMOVE INCORRECT FROM X DATA ****")
    mask = np.ones(len(x_data), dtype=bool)
    mask[incorrect_items] = False
    x_correct = copy.copy(x_data[mask])

    gha_items = x_correct
    if gha_incorrect:  # If I want ALL items including those classified incorrectly
        gha_items = x_data
    print(f"gha_items: (incorrect items={gha_incorrect}) {np.shape(gha_items)}")

    # # PART 5
    print("\n**** Get Hidden unit activations ****")
    hid_act_2d_dict = dict()  # # to use to get 2d hid acts (e.g., means from 4d layers)
    hid_act_any_d_dict = dict()  # # to use to get all hid acts (e.g., both 2d and 4d layers)

    # # loop through key layers df
    gha_key_layers = []
    for index, row in key_layers_df.iterrows():
        if test_run:
            if index > 3:
                continue

        layer_number, layer_name, layer_class = row['layer'], row['name'], row['class']
        print(f"{layer_number}. name {layer_name} class {layer_class}")

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

                if converted_to_2d:
                    hid_act_2d_dict[index]['converted_to_2d'] = True

                print(f"\nlayer{index}. hid_act_2d_dict: {layer_acts_shape}\n")

                # # save distplot for sanity check
                sns.distplot(np.ravel(acts_2d))
                plt.title(str(layer_name))
                # plt.savefig(f"layer_act_dist/{output_filename}_{layer_name}_layer_act_distplot.png")
                plt.savefig(f"{output_filename}_{layer_name}_layer_act_distplot.png")

                plt.close()

    print("\n**** saving info to summary page and dictionary ****")

    hid_act_filenames = {'2d': None, 'any_d': None}
    if save_2d_layers:
        dict_2d_save_name = f'{output_filename}_hid_act_2d.pickle'
        with open(dict_2d_save_name, "wb") as pkl:  # 'wb' mean 'w'rite the file in 'b'inary mode
            pickle.dump(hid_act_2d_dict, pkl)
        # np.save(dict_2d_save_name, hid_act_2d_dict)
        hid_act_filenames['2d'] = dict_2d_save_name

    if save_4d_layers:
        dict_4dsave_name = f'{output_filename}_hid_act_any_d.pickle'
        with open(dict_4dsave_name, "wb") as pkl:  # 'wb' mean 'w'rite the file in 'b'inary mode
            pickle.dump(hid_act_any_d_dict, pkl)
        # np.save(dict_4dsave_name, hid_act_any_d_dict)
        hid_act_filenames['any_d'] = dict_4dsave_name

    cond = sim_dict["topic_info"]["cond"]
    run = sim_dict["topic_info"]["run"]

    hid_units = sim_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable']

    trained_for = sim_dict["training_info"]["trained_for"]
    end_accuracy = sim_dict["training_info"]["acc"]
    dataset = sim_dict["data_info"]["dataset"]
    gha_date = int(datetime.datetime.now().strftime("%y%m%d"))
    gha_time = int(datetime.datetime.now().strftime("%H%M"))

    gha_acc = scores_dict['gha_acc']
    n_cats_correct = scores_dict['n_cats_correct']

    # # GHA_info_dict
    gha_dict_name = f"{output_filename}_GHA_dict.pickle"
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
        focussed_dict_print(gha_dict, 'gha_dict', ['GHA_info', "scores_dict"])

    # make a list of dict names to do sel on
    if not os.path.isfile(f"{output_filename}_dict_list_for_sel.csv"):
        dict_list = open(f"{output_filename}_dict_list_for_sel.csv", 'w')
        mywriter = csv.writer(dict_list)
    else:
        dict_list = open(f"{output_filename}_dict_list_for_sel.csv", 'a')
        mywriter = csv.writer(dict_list)

    mywriter.writerow([gha_dict_name[:-7]])
    dict_list.close()

    print(f"\nadded to list for selectivity analysis: {gha_dict_name[:-7]}")

    # # # spare variables to make anaysis easier
    # if 'chanProp' in output_filename:
    #     var_one = 'chanProp'
    # elif 'chanDist' in output_filename:
    #     var_one = 'chanDist'
    # elif 'cont' in output_filename:
    #     var_one = 'cont'
    # elif 'bin' in output_filename:
    #     var_one = 'bin'
    # else:
    #     raise ValueError("dset_type not found (v1)")
    #
    # if 'pro_sm' in output_filename:
    #     var_two = 'pro_sm'
    # elif 'pro_med' in output_filename:
    #     var_two = 'pro_med'
    # # elif 'LB' in output_filename:
    # #     var_two = 'LB'
    # else:
    #     raise ValueError("between not found (v2)")
    #
    # if 'v1' in output_filename:
    #     var_three = 'v1'
    # elif 'v2' in output_filename:
    #     var_three = 'v2'
    # elif 'v3' in output_filename:
    #     var_three = 'v3'
    # else:
    #     raise ValueError("within not found (v3)")
    #
    # var_four = var_two + var_three
    #
    # if 'ReLu' in output_filename:
    #     var_five = 'relu'
    # elif 'relu' in output_filename:
    #     var_five = 'relu'
    # elif 'sigm' in output_filename:
    #     var_five = 'sigm'
    # else:
    #     raise ValueError("act_func not found (v4)")
    #
    # if '10' in output_filename:
    #     var_six = 10
    # elif '25' in output_filename:
    #     var_six = 25
    # elif '50' in output_filename:
    #     var_six = 50
    # elif '100' in output_filename:
    #     var_six = 100
    # elif '500' in output_filename:
    #     var_six = 500
    # else:
    #     raise ValueError("hid_units not found in output_filename (var6)")

    # print(f"\n{output_filename}: {var_one} {var_two} {var_three} {var_four} {var_five} {var_six}")
    

    gha_info = [cond, run, output_filename, n_layers, hid_units, dataset, use_dataset,
                gha_incorrect, n_cats, trained_for, end_accuracy, gha_acc, n_cats_correct, 
                # var_one, var_two, var_three, var_four, var_five, var_six
                ]

    # # check if gha_summary.csv exists
    # # save summary file in exp folder (grandparent dir to gha folder: exp/cond/gha)
    # to move up to parent just use '..' rather than '../..'

    # exp_name = exp_dir.strip('/')
    exp_name = sim_dict['topic_info']['exp_name']

    os.chdir('../..')
    exp_path = os.getcwd()

    if not os.path.isfile(exp_name + "_GHA_summary.csv"):
        gha_summary = open(exp_name + "_GHA_summary.csv", 'w')
        mywriter = csv.writer(gha_summary)
        summary_headers = ["cond", "run", 'filename', "n_layers", "hid_units", "dataset", "GHA_on",
                           'incorrect', "n_cats", "trained_for", "train_acc", "gha_acc", 'n_cats_correct', 
                           # 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
                           ]

        mywriter.writerow(summary_headers)
        print(f"creating summary csv at: {exp_path}")

    else:
        gha_summary = open(exp_name + "_GHA_summary.csv", 'a')
        mywriter = csv.writer(gha_summary)
        print(f"appending to summary csv at: {exp_path}")

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
