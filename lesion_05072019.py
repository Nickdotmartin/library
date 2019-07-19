import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import copy
import datetime
import os
import sys

sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, focussed_dict_print, print_nested_round_floats
from nick_data_tools import load_x_data, load_y_data, load_hid_acts, nick_to_csv, nick_read_csv
from nick_network_tools import get_scores, VGG_get_scores



def lesion_study(gha_dict_path,
                 get_classes=("Conv2D", "Dense", "Activation"),
                 verbose=False,
                 test_run=False):
    """
    :param exp_cond_gha_path: path to GHA dict - ideally should work for be GHA, sel or sim
    :param gha_dict_name: name of dict to be loaded
    :param get_classes: which types of layer are we interested in?

    :param verbose: will print less if false, otherwise will print eveything
    :param test_run: just run a few units for a test, print lots of output

    :return: lesion_dict:   lesion_path: path to dir where everything is saved,
                            loaded_dict: name of lesion dict, 
                            x_data_path, y_data_path: paths to data used, 
                            key_layer_classes: classes of layers lesioned,
                            key_lesion_layers_list: layer names of lesioned layers (excludes output)
                            total_units_filts: total number of lesionable units
                            lesion_highlights: dict with biggest total and class increase and decrease per layer and 
                                                for the whole model
                            lesion_means_dict: 'mean total change' and 'mean max class drop' per layer
                            
                             

    lesion study
    1. load dict from study (should run with sim, GHA or sel dict)
    2. from dict get x, y, num of items, IPC etc
    3. load original model and weights (model from, num hid units, num hid outputs)
    4. run get scores on ALL items - record total acc, class acc, item sucess
    5. loop through:
        lesion unit (inputs, bias, outputs)
        test on ALL items - record total acc, class acc, item success (pass/fail)
    6. output:
        overall acc change per unit
        class acc change per unit
        item success per unit (pass, l_fail, l_pass, fail) l_fail if pass on full network, fail when lesioned
    """

    print('**** lesion_21052019 lesion_study() ****')


    # # # chdir to this folder
    full_exp_cond_gha_path, gha_dict_name = os.path.split(gha_dict_path)
    training_dir, _ = os.path.split(full_exp_cond_gha_path)
    if not os.path.exists(full_exp_cond_gha_path):
        print("ERROR - path for this experiment not found")
    os.chdir(full_exp_cond_gha_path)
    print("set_path to full_exp_cond_gha_path: {}".format(full_exp_cond_gha_path))


    # # # PART 1 # # #
    # # load details from dict
    if type(gha_dict_path) is str:
        gha_dict = load_dict(gha_dict_path)

    print("\n**** gha_dict ****")
    focussed_dict_print(gha_dict)

    # # # load datasets
    # todo:  if gha_info records x and y path, load data from gha info (not data_info)
    use_dataset = gha_dict['GHA_info']['use_dataset']

    # # check for training data
    if use_dataset in gha_dict['data_info']:
        x_data_path = os.path.join(gha_dict['data_info']['data_path'], gha_dict['data_info'][use_dataset]['X_data'])
        y_data_path = os.path.join(gha_dict['data_info']['data_path'], gha_dict['data_info'][use_dataset]['Y_labels'])
        n_items = gha_dict["data_info"][use_dataset]["n_items"]
        items_per_cat = gha_dict["data_info"][use_dataset]["items_per_cat"]

    else:
        x_data_path = os.path.join(gha_dict['data_info']['data_path'], gha_dict['data_info']['X_data'])
        y_data_path = os.path.join(gha_dict['data_info']['data_path'], gha_dict['data_info']['Y_labels'])
        n_items = gha_dict["data_info"]["n_items"]
        items_per_cat = gha_dict["data_info"]["items_per_cat"]

    n_cats = gha_dict['data_info']["n_cats"]
    if type(items_per_cat) is int:
        items_per_cat = dict(zip(list(range(n_cats)), [items_per_cat] * n_cats))

    if gha_dict['GHA_info']['gha_incorrect'] == 'False':
        # # only gha for correct items
        n_items = gha_dict['GHA_info']['scores_dict']['n_correct']
        items_per_cat = gha_dict['GHA_info']['scores_dict']['corr_per_cat_dict']

    x_data = load_x_data(x_data_path)
    y_df, y_label_list = load_y_data(y_data_path)

    if verbose is True:
        print("y_df: {}\n{}".format(y_df.shape, y_df.head()))
        print("y_df dtypes: {}".format(y_df.dtypes))
        print("y_label_list:\n{}".format(y_label_list[:10]))

    # # data preprocessing
    # # if network is cnn but data is 2d (e.g., MNIST)
    if len(np.shape(x_data)) != 4:
        if gha_dict['model_info']['overview']['model_type'] == 'cnn':
            width, height = gha_dict['data_info']['image_dim']
            x_data = x_data.reshape(x_data.shape[0], width, height, 1)
            print("\nRESHAPING x_data to: {}".format(np.shape(x_data)))

    output_filename = gha_dict["topic_info"]["output_filename"]
    print("\nOutput file: " + output_filename)

    # # set up dicts to save stuff
    count_per_cat_dict = dict()  # count_per_cat_dict is for storing n_items_correct for the lesion study
    prop_change_dict = dict()  # prop_change dict - to compare with Zhou_2018
    item_change_dict = dict()
    lesion_highlights_dict = dict()  # biggest total and per class change
    lesion_highlights_dict["highlights"] = {"total_increase": ("None", 0),
                                            "total_decrease": ("None", 0),
                                            "class_increase": ("None", 0),
                                            "class_decrease": ("None", 0)}

    # # # # PART 2 # # #
    print("\n**** load original trained MODEL ****")
    model_architecture_name = gha_dict['model_info']['overview']['model_name']
    trained_model_name = gha_dict['model_info']['overview']['trained_model']

    optimizer = gha_dict['model_info']['overview']['optimizer']

    print("model_architecture_name: {}".format(model_architecture_name))
    if model_architecture_name == 'VGG16':
        original_model = VGG16(weights='imagenet')
        x_data = preprocess_input(x_data)  # preprocess the inputs loaded as RGB to BGR
    else:
        model_path = os.path.join(training_dir, trained_model_name)
        original_model = load_model(model_path)

    if verbose is True:
        print("original_model.summary: {}".format(original_model.summary()))

    model_details = original_model.get_config()
    print_nested_round_floats(model_details)

    n_layers = len(model_details['layers'])
    model_dict = dict()

    weights_layer_counter = 0

    # # turn off "trainable" and get useful info
    for layer in range(n_layers):
        # set to not train
        model_details['layers'][layer]['config']['trainable'] = False

        if verbose is True:
            print("Model layer {}: {}".format(layer, model_details['layers'][layer]))

        # # get useful info
        layer_dict = {'layer': layer,
                      'name': model_details['layers'][layer]['config']['name'],
                      'class': model_details['layers'][layer]['class_name']}

        if 'units' in model_details['layers'][layer]['config']:
            layer_dict['units'] = int(model_details['layers'][layer]['config']['units'])
        if 'activation' in model_details['layers'][layer]['config']:
            layer_dict['act_func'] = model_details['layers'][layer]['config']['activation']
        if 'filters' in model_details['layers'][layer]['config']:
            layer_dict['filters'] = int(model_details['layers'][layer]['config']['filters'])
        if 'kernel_size' in model_details['layers'][layer]['config']:
            layer_dict['size'] = model_details['layers'][layer]['config']['kernel_size'][0]
        if 'pool_size' in model_details['layers'][layer]['config']:
            layer_dict['size'] = model_details['layers'][layer]['config']['pool_size'][0]
        if 'strides' in model_details['layers'][layer]['config']:
            layer_dict['strides'] = model_details['layers'][layer]['config']['strides'][0]
        if 'rate' in model_details['layers'][layer]['config']:
            layer_dict["rate"] = model_details['layers'][layer]['config']['rate']

        # # record which layuers of the weights matric apply to this layer
        if layer_dict['class'] in ["Conv2D", 'Dense']:
            layer_dict["weights_layer"] = [weights_layer_counter, weights_layer_counter + 1]
            weights_layer_counter += 2  # weights and biases
        elif layer_dict['class'] is 'BatchNormalization':  # BN weights: [gamma, beta, mean, std]
            layer_dict["weights_layer"] = [weights_layer_counter, weights_layer_counter + 1,
                                           weights_layer_counter + 2, weights_layer_counter + 3]
            weights_layer_counter += 4
        elif layer_dict['class'] in ["Dropout", 'Activation', 'MaxPooling2D', "Flatten"]:
            layer_dict["weights_layer"] = []

        # # set and save layer details
        model_dict[layer] = layer_dict

    # # my model summary
    model_df = pd.DataFrame.from_dict(data=model_dict, orient='index',
                                      columns=['layer', 'name', 'class', 'act_func',
                                               'units', 'filters', 'size', 'strides', 'rate', 'weights_layer'])

    key_layers_df = model_df.loc[model_df['class'].isin(get_classes)]
    key_layers_df = key_layers_df.drop(columns=['size', 'strides', 'rate'])

    # # just classes of layers specified in get_classes
    if 'VGG16_GHA_layer_dict' in gha_dict['model_info']:
        get_layers_dict = gha_dict['model_info']['VGG16_GHA_layer_dict']
        print('get_layers_dict: {}'.format(get_layers_dict))
        get_layer_names = []
        for k, v in get_layers_dict.items():
            get_layer_names.append(v['name'])
        key_layers_df = model_df.loc[model_df['name'].isin(get_layer_names)]


    # # add column ('n_units_filts')to say how many mthings needs lesioning per layer (number of units or filters)
    # # add zeros to rows with no units or filters
    key_layers_df.loc[:, 'n_units_filts'] = key_layers_df.units.fillna(0) + key_layers_df.filters.fillna(0)

    key_lesion_layers_list = key_layers_df['name'].to_list()

    # # remove unnecesary items from key layers list
    if 'output' in key_lesion_layers_list:
        output_idx = key_lesion_layers_list.index('output')
        drop_layer_names = key_lesion_layers_list[output_idx:]
        key_lesion_layers_list = key_lesion_layers_list[:output_idx]
        key_layers_df = key_layers_df.loc[~key_layers_df['name'].isin(drop_layer_names)]

        # key_lesion_layers_list.remove('output')
    if 'Output' in key_lesion_layers_list:
        output_idx = key_lesion_layers_list.index('Output')
        drop_layer_names = key_lesion_layers_list[output_idx:]
        key_lesion_layers_list = key_lesion_layers_list[:output_idx]
        key_layers_df = key_layers_df.loc[~key_layers_df['name'].isin(drop_layer_names)]

        # key_lesion_layers_list.remove('Output')

    key_layers_df.reset_index(inplace=True)

    total_units_filts = key_layers_df['n_units_filts'].sum()

    if verbose is True:
        print("\nmodel_df:\n{}".format(model_df))
        print("\n{} key_lesion_layers_list: {}".format(len(key_lesion_layers_list), key_lesion_layers_list))

    print("\nkey_layers_df:\n{}".format(key_layers_df))

    # get original values for weights
    print("\n**** load trained weights ****")

    full_weights = original_model.get_weights()
    n_weight_arrays = np.shape(full_weights)[0]
    print("n_weight_arrays: {}".format(n_weight_arrays))

    if test_run is True:
        if verbose is True:
            print("full_weights: {}".format(np.shape(full_weights)))
            for w_array in range(n_weight_arrays):
                print("\nfull_weights{}: {}\n{}".format(w_array, np.shape(full_weights[w_array]),
                                                        full_weights[w_array]))

    original_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print("\nLoaded '{}' model with original weights: {}".format(model_architecture_name, trained_model_name))

    """# # save this in case I come back ina few months and need a refersher.
    # no all network layers have associated weights (only learnable ones)
    # some layers have multiple layers of weights associated.
    # conv2d layers have 2 sets of arrays [connection weights, biases] 
    # dense layers have 2 sets of arrays [connection weights, biases]
    # batchnorm has 4 sets of weights
    # print("Figuring out layers")
    # print("weights shape {}".format(trained_model_name))
    # for index, layer in enumerate(original_model.layers):
    #     g = layer.get_config()
    #     h = layer.get_weights()
    #     if h:
    #         q = 'has shape'
    #         s = len(h)
    #     if not h:
    #         q = 'not h'
    #         s = (0, )
    #     print("\n{}. {}  {}\n{}\n".format(index, s, g, h))"""

    # # 4. run get scores on ALL items - record total acc, class acc, item sucess

    # # count_per_cat_dict is for storing n_items_correct for the lesion study
    count_per_cat_dict['dataset'] = items_per_cat
    # count_per_cat_dict['dataset'] = items_per_cat
    count_per_cat_dict['dataset']['total'] = n_items
    print("\ncount_per_cat_dict: {}".format(count_per_cat_dict))

    print("\n**** Get original model scores ****")
    predicted_outputs = original_model.predict(x_data)

    if model_architecture_name == 'VGG16':
        item_correct_df, scores_dict, incorrect_items = VGG_get_scores(predicted_outputs, y_df, output_filename,
                                                                       save_all_csvs=True)
    else:
        item_correct_df, scores_dict, incorrect_items = get_scores(predicted_outputs, y_df, output_filename,
                                                                   save_all_csvs=False, return_flat_conf=True)

    if verbose is True:
        print("\n****Scores_dict****")
        focussed_dict_print(scores_dict)

    # # # get scores per class for full model
    full_model_CPC = scores_dict['corr_per_cat_dict']
    full_model_CPC['total'] = scores_dict['n_correct']
    count_per_cat_dict['full_model'] = full_model_CPC

    # # use these (unlesioned) pages as the basis for LAYER pages
    # flat_conf_MASTER = scores_dict.loc[:, 'flat_conf']
    if model_architecture_name != 'VGG16':
        flat_conf_MASTER = scores_dict['flat_conf']

    item_correct_MASTER = copy.copy(item_correct_df)

    lesion_means_dict = dict()

    # # # set dir to save lesion stuff stuff # # #
    lesion_path = os.path.join(os.getcwd(), 'lesion')

    if test_run is True:
        lesion_path = os.path.join(lesion_path, 'test')

    if not os.path.exists(lesion_path):
        os.makedirs(lesion_path)

    os.chdir(lesion_path)
    print("saving lesion data to: {}".format(lesion_path))

    # # # PART 5 # # #
    # # loop through key layers df
    # #     lesion unit (inputs, bias, outputs)
    # #     test on ALL items - record total acc, class acc, item success (pass/fail)
    # print("\n'BEFORE - full_weights'{} {}\n{}\n\n".format(np.shape(full_weights), type(full_weights), full_weights))

    print("\n\n\n**** loop through key layers df ****")
    for index, row in key_layers_df.iterrows():

        if test_run is True:
            if index > 3:
                continue

        layer_number, layer_name, layer_class, n_units_filts = \
            row['layer'], row['name'], row['class'], row['n_units_filts']
        print("\n{}. name {}, class {}, n_units_filts {}".format(layer_number, layer_name, layer_class, n_units_filts))

        if layer_class not in get_classes:  # no longer using this - skip class types not in list
            # if layer_name not in get_layer_list:  # skip layers/classes not in list
            print("skip this")
            continue

        # # make places to save layer details
        count_per_cat_dict[layer_name] = dict()
        item_correct_LAYER = copy.copy(item_correct_MASTER)
        prop_change_dict[layer_name] = dict()

        lesion_means_dict[layer_name] = dict()

        layer_total_change_list = []
        layer_max_drop_list = []

        if model_architecture_name != 'VGG16':
            flat_conf_LAYER = copy.copy(flat_conf_MASTER)





        weights_n_biases = row['weights_layer']
        print("weights_n_biases: ", weights_n_biases)

        if not weights_n_biases:  # if empty list []
            print("skip this")
            continue

        weights_layer = weights_n_biases[0]
        biases_layer = weights_n_biases[1]

        for unit in range(int(n_units_filts)):

            if test_run is True:
                if unit > 3:
                    continue

            layer_and_unit = "{}.{}".format(layer_name, unit)
            print("\n\n**** lesioning layer {}. ({}) {} of {}****".format(layer_number, layer_class, layer_and_unit,
                                                                          n_units_filts))
            # # load original weights each time
            edit_full_weights = copy.deepcopy(full_weights)

            # print("type(edit_full_weights): {}".format(type(edit_full_weights)))
            # print("np.shape(edit_full_weights): {}".format(np.shape(edit_full_weights)))
            # print("np.shape(edit_full_weights[0]): {}".format(np.shape(edit_full_weights[0])))
            # print("888888888888\n\t88888888888888888\n\t\t8888888888\n\t\t\t\t888888")

            # # change input to hid unit
            if layer_class is 'Conv2D':
                edit_full_weights[weights_layer][:, :, :, unit] = 0.0
                '''eg: for 20 conv filters of shape 5 x 5, layer shape (5, 5, 1, 20)'''
            else:
                edit_full_weights[weights_layer][:, unit] = 0.0
            # # change unit bias(index with layer*2 + 1)
            edit_full_weights[biases_layer][unit] = 0.0

            if test_run is True:
                print("\n'AFTER l{}h{}".format(layer, unit))
                for array in range(n_weight_arrays):
                    print(edit_full_weights[array])

            original_model.set_weights(edit_full_weights)
            original_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

            # # # get scores

            predicted_outputs = original_model.predict(x_data)

            if model_architecture_name == 'VGG16':
                item_correct_df, scores_dict, incorrect_items = VGG_get_scores(predicted_outputs, y_df, output_filename,
                                                                               save_all_csvs=True)
            else:
                item_correct_df, scores_dict, incorrect_items = get_scores(predicted_outputs, y_df, output_filename,
                                                                           save_all_csvs=False, return_flat_conf=True)

            if verbose is True:
                print("\n****Scores_dict****")
                focussed_dict_print(scores_dict)

            # # # get scores per class for this layer
            corr_per_cat_dict = scores_dict['corr_per_cat_dict']
            corr_per_cat_dict['total'] = scores_dict['n_correct']
            count_per_cat_dict[layer_name][unit] = corr_per_cat_dict

            item_correct_LAYER[layer_and_unit] = item_correct_df['full_model']
            # item_correct_LAYER.to_csv("{}_{}_item_correct.csv".format(output_filename, layer_name), index=False)
            nick_to_csv(item_correct_LAYER, "{}_{}_item_correct.csv".format(output_filename, layer_name))

            if model_architecture_name != 'VGG16':
                flat_conf_LAYER[layer_and_unit] = scores_dict['flat_conf']['full_model']
                # flat_conf_LAYER.to_csv("{}_{}_flat_conf.csv".format(output_filename, layer_name))
                nick_to_csv(flat_conf_LAYER, "{}_{}_flat_conf.csv".format(output_filename, layer_name))

            # # make item change per laye df
            """# # four possible states
                full model      after_lesion    code
            1.  1 (correct)     0 (wrong)       -1
            2.  0 (wrong)       0 (wrong)       0
            3.  1 (correct)     1 (correct)     1
            4.  0 (wrong)       1 (correct)     2

            """
            # # get column names/unit ids
            all_column_list = list(item_correct_LAYER)
            column_list = all_column_list[3:]
            # print(len(column_list), column_list)

            # # get columns to be used in new df
            full_model = item_correct_LAYER['full_model'].to_list()
            item_id = item_correct_LAYER['item'].to_list()
            classes = item_correct_LAYER['class'].to_list()
            # print("full model: {}\n{}".format(len(full_model), full_model))

            # # set up dict to make new df
            item_change_dict[layer_name] = dict()
            item_change_dict[layer_name]['item'] = item_id
            item_change_dict[layer_name]['class'] = classes
            item_change_dict[layer_name]['full_model'] = full_model

            # # loop through old item_correct_LAYER
            for idx in column_list:
                item_change_list = []
                for item in item_correct_LAYER[idx].iteritems():
                    index, lesnd_acc = item
                    unlesnd_acc = full_model[index]

                    # # four possible states
                    if unlesnd_acc == 1 and lesnd_acc == 0:  # lesioning causes failure
                        item_change = -1
                    elif unlesnd_acc == 0 and lesnd_acc == 0:  # no effect of lesioning, still incorrect
                        item_change = 0
                    elif unlesnd_acc == 1 and lesnd_acc == 1:  # no effect of lesioning, still correct
                        item_change = 1
                    elif unlesnd_acc == 0 and lesnd_acc == 1:  # lesioning causes failed item to pass
                        item_change = 2
                    else:
                        item_change = 'ERROR'
                    # print("{}. unlesnd_acc: {} lesnd_acc: {} item_change: {}".format(index, unlesnd_acc,
                    #                                                                  lesnd_acc, item_change))
                    item_change_list.append(item_change)
                item_change_dict[layer_name][idx] = item_change_list

            # # # convert item_change_dict to df
            # item_change_df = pd.DataFrame.from_dict(item_change_dict[layer_name])
            # print("\n\nitem_change_df:\n{}".format(item_change_df.head()))

            # # # get class_change scores for this layer
            print("\nget class_change scores:")
            # proportion change = (after_lesion/unlesioned) - 1
            unit_prop_change_dict = dict()
            for (fk, fv), (k2, v2) in zip(full_model_CPC.items(), corr_per_cat_dict.items()):
                if fv == 0:
                    prop_change = -1
                else:
                    prop_change = (v2 / fv) - 1
                unit_prop_change_dict[fk] = prop_change
                # print(fk, 'v2: ', v2, '/ fv: ', fv, '= pc: ', prop_change)

            prop_change_dict[layer_name][unit] = unit_prop_change_dict

            # # get layer means
            layer_total_change_list.append(unit_prop_change_dict['total'])
            layer_max_drop_list.append(min(list(unit_prop_change_dict.values())[:-1]))

        lesion_means_dict[layer_name]['mean_total'] = np.mean(layer_total_change_list)
        lesion_means_dict[layer_name]['mean_max_drop'] = np.mean(layer_max_drop_list)



        # # save layer info
        print("\n**** save layer info for {} ****".format(layer_name))

        count_per_cat_df = pd.DataFrame.from_dict(count_per_cat_dict[layer_name])
        count_per_cat_df.to_csv("{}_{}_count_per_cat.csv".format(output_filename, layer_name))
        print("why won't count_per_cat_save with nick_to_csv.  "
              "I get error for line 42 of data_tools:      df2.index = df2.index + 1"
              "saying it needs to be a string?")
        print(count_per_cat_df.head())
        # nick_to_csv(count_per_cat_df, "{}_{}_count_per_cat.csv".format(output_filename, layer_name))

        prop_change_df = pd.DataFrame.from_dict(prop_change_dict[layer_name])
        prop_change_df.to_csv("{}_{}_prop_change.csv".format(output_filename, layer_name))
        # nick_to_csv(prop_change_df, "{}_{}_prop_change.csv".format(output_filename, layer_name))

        # # convert item_change_dict to df
        item_change_df = pd.DataFrame.from_dict(item_change_dict[layer_name])
        item_change_df.to_csv("{}_{}_item_change.csv".format(output_filename, layer_name))
        # nick_to_csv(item_change_df, "{}_{}_item_change.csv".format(output_filename, layer_name))

        print("\n\nitem_change_df:\n{}".format(item_change_df.head()))

        # # for each layer
        layer_highlights_dict = dict()

        # cols = list(prop_change_df)
        # 1. 3 units with biggest total increase
        total_series = prop_change_df.loc['total', :]
        total_biggest_units = total_series.sort_values(ascending=False).head(3).index.to_list()
        total_biggest_vals = total_series.sort_values(ascending=False).head(3).to_list()

        print("\n\ntotal_biggest_units: {}".format(total_biggest_units))
        print("\n\ntotal_biggest_vals: {}".format(total_biggest_vals))

        total_increase_dict = {total_biggest_units[i]: total_biggest_vals[i]
                               for i in range(sum(1 for x in total_biggest_vals if x > 0.0))}
        layer_highlights_dict["total_increase"] = total_increase_dict

        print("max tot increase: ", lesion_highlights_dict["highlights"]["total_increase"][1])
        if lesion_highlights_dict["highlights"]["total_increase"][1] < total_biggest_vals[0]:
            lesion_highlights_dict["highlights"]["total_increase"] = (layer_and_unit, total_biggest_vals[0])

        # # 2. 3 units with biggest total decrease
        # total_smallest = total_series.nsmallest(n=3, columns=cols)
        total_smallest_units = total_series.sort_values().head(3).index.to_list()
        total_smallest_vals = list(total_series.sort_values().head(3))

        print("\n\ntotal_smallest_units: {}".format(total_smallest_units))
        print("\n\ntotal_smallest_vals: {}".format(total_smallest_vals))
        total_decrease_dict = {total_smallest_units[i]: total_smallest_vals[i]
                               for i in range(sum(1 for x in total_biggest_vals if x < 0.0))}
        layer_highlights_dict["total_decrease"] = total_decrease_dict

        print("max tot decrease: ", lesion_highlights_dict["highlights"]["total_decrease"][1])
        if lesion_highlights_dict["highlights"]["total_decrease"][1] > total_smallest_vals[0]:
            lesion_highlights_dict["highlights"]["total_decrease"] = (layer_and_unit, total_smallest_vals[0])

        get_class_highlights = prop_change_df.drop('total')

        # biggest class increase
        top3_vals = sorted(set(get_class_highlights.to_numpy().ravel()), reverse=True)[:3]
        top3_tup = []
        for val in top3_vals:
            units = get_class_highlights.columns[get_class_highlights.isin([val]).any()].to_list()
            for idx in units:
                top3_tup.append((idx, val))

        class_increase_dict = {top3_tup[i][0]: top3_tup[i][1]
                               for i in range(sum(1 for x in top3_vals if x > 0.0))}
        layer_highlights_dict["class_increase"] = class_increase_dict

        print("max class_increasee: ", lesion_highlights_dict["highlights"]["class_increase"][1])
        if lesion_highlights_dict["highlights"]["class_increase"][1] < top3_vals[0]:
            lesion_highlights_dict["highlights"]["class_increase"] = (layer_and_unit, top3_vals[0])

        # biggest class decrease
        bottom3_vals = sorted(set(get_class_highlights.to_numpy().ravel()))[:3]
        bottom3_tup = []
        for val in bottom3_vals:
            units = get_class_highlights.columns[get_class_highlights.isin([val]).any()].to_list()
            for idx in units:
                bottom3_tup.append((idx, val))

        class_decrease_dict = {bottom3_tup[i][0]: bottom3_tup[i][1]
                               for i in range(sum(1 for x in bottom3_vals if x < 0.0))}
        layer_highlights_dict["class_decrease"] = class_decrease_dict

        print("max class_decrease: ", lesion_highlights_dict["highlights"]["class_decrease"][1])
        if lesion_highlights_dict["highlights"]["class_decrease"][1] > bottom3_vals[0]:
            lesion_highlights_dict["highlights"]["class_decrease"] = (layer_and_unit, bottom3_vals[0])

        lesion_highlights_dict[layer_name] = layer_highlights_dict

    # # 6. output:
    print("\n**** make output files and save ****")
    date = int(datetime.datetime.now().strftime("%y%m%d"))
    time = int(datetime.datetime.now().strftime("%H%M"))

    lesion_summary_dict = gha_dict
    lesion_info = {"lesion_path": lesion_path,
                   "loaded_dict": gha_dict_name,
                   "x_data_path": x_data_path,
                   "y_data_path": y_data_path,
                   "key_layer_classes": get_classes,
                   "key_lesion_layers_list": key_lesion_layers_list,
                   'total_units_filts': total_units_filts,
                   "sel_date": date,
                   "sel_time": time,
                   'lesion_highlights': lesion_highlights_dict,
                   'lesion_means_dict': lesion_means_dict}

    lesion_summary_dict["lesion_info"] = lesion_info

    print("Saving dict to: {}".format(lesion_path))
    lesion_dict_name = "{}/{}_lesion_dict.pickle".format(lesion_path, output_filename)
    pickle_out = open(lesion_dict_name, "wb")
    pickle.dump(gha_dict, pickle_out)
    pickle_out.close()

    print("\nlesion_summary_dict:")
    # focussed_dict_print(lesion_summary_dict)
    print_nested_round_floats(lesion_summary_dict)

    # # lesion summary page
    """ls_headers = ['date', 'time', 'cond_name', 'data', 'dset', 'model',
                     'total_increase', 'total_decrease', 'class_increase', 'class_decrease']"""

    lesion_summary_path = '/home/nm13850/Documents/PhD/python_v2/experiments/lesioning/lesion_summary.csv'
    # lesion_summary = pd.read_csv(lesion_summary_path, index_col=0)
    lesion_summary = nick_read_csv(lesion_summary_path)
    # lesion_summary.set_index(0)
    ls_cols = list(lesion_summary)  # 14 columns

    ls_info = [date, time, output_filename, gha_dict['data_info']['dataset'], gha_dict['GHA_info']['use_dataset'],
               gha_dict['topic_info']['model_path'],
               lesion_highlights_dict['highlights']["total_increase"][0],
               lesion_highlights_dict['highlights']["total_increase"][1],
               lesion_highlights_dict['highlights']["total_decrease"][0],
               lesion_highlights_dict['highlights']["total_decrease"][1],
               lesion_highlights_dict['highlights']["class_increase"][0],
               lesion_highlights_dict['highlights']["class_increase"][1],
               lesion_highlights_dict['highlights']["class_decrease"][0],
               lesion_highlights_dict['highlights']["class_decrease"][1]]

    print("len ls_info: {}\nls_cols: {}".format(len(ls_info), len(ls_cols)))
    for_df = dict(zip(ls_cols, ls_info))
    lesion_summary = lesion_summary.append(for_df, ignore_index=True)

    print(lesion_summary.head())

    if test_run is False:
        lesion_summary.to_csv(lesion_summary_path)

    end = "\nscript_finished\n\n\n\n\n\n"
    print(end)

    return lesion_summary_dict


# # low acc iris test
# print("\n\n\nRUNNING FROM BOTTOM OF GHA SCRIPT\n\n\nTEST RUNS ONLY")
# # # lesion
# 
# lesion_dict = lesion_study(gha_dict_path='/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/'
#                                          'CIFAR10_models_c4p2_adam_bn/all_test_set_gha/'
#                                          'CIFAR10_models_c4p2_adam_bn_GHA_dict.pickle',
#                            # verbose=True,
#                            test_run=True
#                            )

# lesion_output = lesion_study(exp_cond_gha_path="train_script_check/train_script_check_fc2_iris_None/"
#                                                "correct_train_set_gha/",
#                              gha_dict_name="train_script_check_fc2_iris_None_GHA_dict.pickle",
#                              # x_data_path='/home/nm13850/Documents/PhD/python_v2/datasets/other_classification/iris/'
#                              #             'orig/iris_X.csv',
#                              # y_data_path='/home/nm13850/Documents/PhD/python_v2/datasets/other_classification/iris/'
#                              #              'orig/iris_labels.csv',
#                              verbose=True,
#                              test_run=True
#                              )
# # lesion_output = lesion_study(exp_cond_gha_path='train_script_check/'
# #                                                                 'train_script_check_con2_pool2_fc1_mnist_None/'
# #                                                                 'correct_test_set_gha',
# #                                         gha_dict_name="train_script_check_con2_pool2_fc1_mnist_None_GHA_dict.pickle",
# #
# #                                         verbose=True)
#
#
# lesion_cifar = lesion_study(exp_cond_gha_path='train_script_check/train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/'
#                                               'all_test_set_gha',
#                             gha_dict_name="train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug_GHA_dict.pickle",
#                             verbose=True,
#                             # test_run=True
#                             )

# lesion_mnist = lesion_study(exp_cond_gha_path='train_script_check/train_script_check_con2_pool2_fc1_mnist_None/'
#                                               'correct_test_set_gha',
#                              gha_dict_name="train_script_check_con2_pool2_fc1_mnist_None_GHA_dict.pickle",
#                             test_run=True)

