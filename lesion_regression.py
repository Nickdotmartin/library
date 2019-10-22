import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
import pickle
import os
import sys
sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, focussed_dict_print, print_nested_round_floats
from nick_data_tools import nick_read_csv, nick_to_csv

#todo: change it so it gets sel scores from lesion layers (e.g., conv) or so it gets sel from following ReLu layers

def lesion_sel_regression(lesion_dict_path, sel_dict_path, use_relu=False, verbose=False):
    """
    Script uses lesion dict and sel dict and has a simple logistic regression model to see how well various selectivity
    measures predict max class fail

    :param lesion_dict_path: path to dict from output of lesioning
    :param sel_dict_path: path to dict of output from selectivity
    :param use_relu: if False, only uses sel scores for lesioned layers (conv, dense), if true, only uses sel scores
        for ReLu layers, as found with link_layers_dict
    :return: lesion_regression_dict
    """

    print("**** running lesion_sel_regression() ****")

    lesion_dict = load_dict(lesion_dict_path)
    focussed_dict_print(lesion_dict, "lesion dict")

    n_cats = lesion_dict['data_info']['n_cats']
    output_filename = lesion_dict['topic_info']['output_filename']

    lesion_info = lesion_dict['lesion_info']
    lesion_path = lesion_info['lesion_path']

    # # get key layers list
    lesion_highlighs = lesion_info["lesion_highlights"]
    # focussed_dict_print(lesion_highlighs)
    key_lesion_layers_list = list(lesion_highlighs.keys())
    # # remove unnecesary items from key layers list
    # # remove unnecesary items from key layers list
    if 'highlights' in key_lesion_layers_list:
        key_lesion_layers_list.remove('highlights')
    if 'output' in key_lesion_layers_list:
        key_lesion_layers_list.remove('output')
    if 'Output' in key_lesion_layers_list:
        key_lesion_layers_list.remove('Output')
    print("\nkey_lesion_layers_list\n{}".format(key_lesion_layers_list))



    # # sel_dict
    sel_dict = load_dict(sel_dict_path)
    if key_lesion_layers_list[0] in sel_dict['sel_info']:
        print('\n found old sel dict layout')
        old_sel_dict=True
        sel_info = sel_dict['sel_info']
        short_sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0]['sel'].keys())
        csb_list = list(sel_info[key_lesion_layers_list[0]][0]['class_sel_basics'].keys())
        sel_measures_list = short_sel_measures_list + csb_list
    else:
        print('\n found NEW sel dict layout')
        old_sel_dict=False
        sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
        sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0].keys())


    # todo: fix these dodgy measures or find the problem
    '''get rid of measures that I've had trouble with'''

    if 'nz_count' in sel_measures_list:
        sel_measures_list.remove('nz_count')
    if 'max_info_f1' in sel_measures_list:
        sel_measures_list.remove('max_info_f1')

    if use_relu is True:
        # # get key_gha_sel_layers_list
        key_gha_sel_layers_list = list(sel_info.keys())
        # # remove unnecesary items from key layers list
        if 'sel_analysis_info' in key_gha_sel_layers_list:
            key_gha_sel_layers_list.remove('sel_analysis_info')
        if 'output' in key_gha_sel_layers_list:
            output_idx = key_gha_sel_layers_list.index('output')
            key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]
        if 'Output' in key_gha_sel_layers_list:
            output_idx = key_gha_sel_layers_list.index('Output')
            key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]

        # # put together lists of 1. sel_gha_layers, 2. key_lesion_layers_list.
        n_activation_layers = sum("activation" in layers for layers in key_gha_sel_layers_list)
        n_lesion_layers = len(key_lesion_layers_list)

        if n_activation_layers == n_lesion_layers:
            n_layers = n_activation_layers
            activation_layers = [layers for layers in key_gha_sel_layers_list if "activation" in layers]
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(activation_layers)))
        else:
            raise TypeError('should be same number of activation layers and lesioned layers')

        if verbose is True:
            focussed_dict_print(link_layers_dict, 'link_layers_dict')


    print("\nsel_measures_list\n{}".format(sel_measures_list))

    regression_measures_dict = dict()  # key=measure, val=score

    """loop through sel_measures_list
    get long array of all layers concatenated and list all max lesion drops.
    do regression model on this whole thing at once,
    save then repeat for next sel measure"""

    print("\nlooping through network to get selectivity scores as big array")
    for sel_measure in sel_measures_list:
        this_sel_measure = sel_measure
        print("\n{}".format(this_sel_measure))

        """loop through all layers"""
        all_layer_sel_array = []  # append each sel_layer to this to give array (each unit, classes)

        all_layer_class_drops = []  # # extend each lesion_layer to this to give list of all units in all layers

        for lesion_layer in key_lesion_layers_list:

            sel_layer = lesion_layer
            if use_relu:
                sel_layer = link_layers_dict[lesion_layer]

            if verbose == True:
                print("\tsel_layer: {}\tlesion_layer: {}".format(sel_layer, lesion_layer))

            sel_layer_info = sel_info[sel_layer]

            '''get sel_layer sel values'''
            # get array of sel values for regression model
            layer_sel_array = []
            for k, v in sel_layer_info.items():
                unit = k
                sel_measure = this_sel_measure

                if old_sel_dict:
                    if sel_measure in v['sel']:
                        sel_items = v['sel'][sel_measure]
                    elif sel_measure in v['class_sel_basics']:
                        sel_items = v['class_sel_basics'][sel_measure]
                else:
                    sel_items = v[sel_measure]

                # # just check it is just classes in there
                if 'total' in sel_items.keys():
                    del sel_items['total']
                if 'perplexity' in sel_items.keys():
                    del sel_items['perplexity']

                if len(list(sel_items.keys())) != n_cats:
                    print("\nERROR, {} hasn't got enough classes".format(sel_measure))
                    # print("error found", sel_items)
                    for i in range(n_cats):
                        if i not in sel_items:
                            sel_items[i] = 0.0
                    ordered_dict = dict()
                    for j in range(n_cats):
                        ordered_dict[j] = sel_items[j]

                    sel_items = dict()
                    sel_items = ordered_dict
                    print('sel_items should be sorted now', sel_items)


                sel_values = list(sel_items.values())

                layer_sel_array.append(sel_values)
                # print(unit, sel_values)

            # print("\nlayer_sel_array\n{}".format(layer_sel_array))
            # print("\t\tlayer_sel_array: {}".format(np.shape(layer_sel_array)))

            all_layer_sel_array = all_layer_sel_array + layer_sel_array
            # print("\t\tall_layer_sel_array: {}".format(np.shape(all_layer_sel_array)))

            # # lesion stuff
            # # conv2d_6
            lesion_per_unit_path = '{}/{}_{}_prop_change.csv'.format(lesion_path, output_filename, lesion_layer)

            lesion_per_unit = pd.read_csv(lesion_per_unit_path, index_col=0)
            # lesion_per_unit = nick_read_csv(lesion_per_unit_path)
            # lesion_per_unit.set_index(0)

            lesion_cols = list(lesion_per_unit)
            # print("\nlesion_per_unit")
            # print(lesion_cols)
            # print(lesion_per_unit.head())



            '''get max class drop per lesion_layer'''
            # # loop through lesion units (df columns) to find min class drop
            lesion_min_dict = dict()
            max_class_drop_list = []
            for index, l_unit in enumerate(lesion_cols):
                # print(lesion_per_unit[l_unit])
                min_class_val = lesion_per_unit[l_unit].min()
                min_class = lesion_per_unit[l_unit].idxmin()
                max_class_drop_list.append(int(min_class))
                # print("{}: class: {}  {}".format(index, min_class, min_class_val))
                lesion_min_dict[index] = {'unit': index, "l_min_class": min_class, 'l_min_drop': min_class_val}



            # # check for missing values (prob dead relus) when using different layers
            sel_units, classes = np.shape(layer_sel_array)
            les_units = len(max_class_drop_list)
            if use_relu:
                if sel_units != les_units:
                    print("\n\number of units is wrong")
                    print("sel_units: {}\nles_units: {}".format(sel_units, les_units))

                    if len(max_class_drop_list) > sel_units:
                        available_sel_units = list(sel_layer_info.keys())
                        masked_class_drops = [max_class_drop_list[i] for i in available_sel_units]
                        max_class_drop_list = masked_class_drops

            # # if there are any NaNs:
            if np.any(np.isnan(layer_sel_array)):
                print("TRUE nan")
                # layer_sel_array[np.isnan(layer_sel_array)] = 0
                layer_sel_array = np.nan_to_num(layer_sel_array, copy=False)

                if np.any(np.isnan(layer_sel_array)):
                    print("TRUE still nan")
            if np.all(np.isfinite(layer_sel_array)):
                print("TRUE inf")

            # layer_sel_array = np.array(layer_sel_array)  # [np.isnan(layer_sel_array)] = 0.0
            # layer_sel_array = np.nan_to_num(layer_sel_array, copy=False)
            # layer_sel_array[np.isneginf(layer_sel_array)] = 0


            all_layer_class_drops.extend(max_class_drop_list)
            # print("\t\tall_layer_class_drops: {}".format(np.shape(all_layer_class_drops)))

            # focussed_dict_print(lesion_min_dict)
            # print("\nmax_class_drop_list\n{}".format(max_class_drop_list))
            if verbose == True:
                print("\t\tlayer_sel: {}, all_layers_sel: {}; all_layers_class_drops: "
                      "{}".format(np.shape(layer_sel_array), np.shape(all_layer_sel_array),
                                  np.shape(all_layer_class_drops)))



        # # sel lesion model
        X = all_layer_sel_array  # sel measure
        print("\nX: {}".format(np.shape(X)))


        y = all_layer_class_drops  # max class drop list
        print("y: {}".format(np.shape(y)))

        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(X, y)

        # todo: add mode info here, predict_log_proba,

        # print("\nsimple logistic regression")
        # # regression sanity checks
        # print("\nX[0]\n{}\nX[1]\n{}".format(X[0], X[1]))
        # print("\ny[0]: {}\ty[1]: {}".format(y[0], y[1]))
        # print("\nclf.predict(X[:2])\n{}".format(clf.predict(X[:2])))
        # print("\nclf.predict_proba(X[:2])\n{}".format(clf.predict_proba(X[:2])))

        regression_score = clf.score(X, y)

        print("{} regression_score: {}".format(this_sel_measure, regression_score))

        regression_measures_dict[this_sel_measure] = regression_score

    print("\n\nfinal regression_measures_dict values")
    print_nested_round_floats(regression_measures_dict)

    print("\n\nSaving lesion_regression_dict")
    save_path, _ = os.path.split(lesion_dict_path)
    print("Saving dict to: {}".format(save_path))

    lesion_regression_dict = lesion_dict
    lesion_regression_dict['regression_info'] = regression_measures_dict
    print_nested_round_floats(lesion_regression_dict, 'lesion_regression_dict')

    if use_relu:
        output_filename = output_filename + '_onlyReLu'
    print("output_filename: ", output_filename)

    les_reg_dict_name = "{}/{}_les_reg_dict.pickle".format(save_path, output_filename)
    pickle_out = open(les_reg_dict_name, "wb")
    pickle.dump(lesion_regression_dict, pickle_out)
    pickle_out.close()

    les_reg_df = pd.DataFrame(data=regression_measures_dict, index=[0])
    nick_to_csv(les_reg_df, "{}/{}_les_reg_dict_nm.csv".format(save_path, output_filename))
    les_reg_df.to_csv("{}/{}_les_reg_dict_pd.csv".format(save_path, output_filename))

    return lesion_regression_dict
#
#
# # #########################
# # # old tes hid act experiments
# # lesion_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/train_script_check/' \
# #                    'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/all_test_set_gha/lesion/' \
# #                    'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug_lesion_dict.pickle'
# # sel_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/train_script_check/' \
# #                 'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/all_test_set_gha/correct_sel/' \
# #                 'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug_sel_dict.pickle'
# #
# #
# # # # new cifar experiments
# # lesion_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/CIFAR10_models_c4p2_adam_bn/' \
# #                    'all_test_set_gha/lesion/CIFAR10_models_c4p2_adam_bn_lesion_dict.pickle'
# # sel_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/CIFAR10_models_c4p2_adam_bn/' \
# #                 'all_test_set_gha/correct_sel/CIFAR10_models_c4p2_adam_bn_sel_dict.pickle'
# #
# # lesion_sel_regression_dict = lesion_sel_regression(lesion_dict_path=lesion_dict_path, sel_dict_path=sel_dict_path,
# #                                                    use_relu=True,
# #                                                    verbose=True)
# #


####################################################################################

def item_act_fail_regression(sel_dict_path, lesion_dict_path, plot_type='classes',
                             sel_measures=['Zhou_prec', 'CCMAs', 'max_informed'],
                             top_layers='all',
                             use_relu=False,
                             verbose=False, test_run=False,
                             ):
    """
    predict item fail from item activation

    logistic regression, ROC, t-test

    load hid_acts
    Load sel_per_unit
    Load lesion fail_per unit

    for each layer:
        for each unit:
            for each item (correct in full model):
                get normed act and pass-fail
                does normed act predict pass/fail

                get class, class sel,
                does normed act, class, class sel predict item fail.

     I only have lesion data for [conv2d, dense] layers
     I have GHA and sel data from  [conv2d, activation, max_pooling2d, dense] layers

     so for each lesioned layer [conv2d, dense] I will use the following activation layer to take GHA and sel data from.

     Join these into groups using the activation number as the layer numbers.
     e.g., layer 1 (first conv layer) = conv2d_1 & activation_1.  layer 7 (first fc layer) = dense1 & activation 7)

    :param sel_dict_path:  path to selectivity dict
    :param lesion_dict_path: path to lesion dict
    :param plot_type: all classes or OneVsAll.  if n_cats > 10, should automatically revert to oneVsAll.
    :param sel_measures: measure to use when choosing which class should be the COI.  Either the best performing sel
            measures (c_informed, c_ROC) or max class drop from lesioning.
            if sel_measures='all' it will use all selectivity measures available.
    :param top_layers: if int, it will just do the top n mayers (excluding output).  If not int, will do all layers.

    :param :


    :return: print and save plots
    """

    print("\n**** running item_act_fail_regression()****")

    # # lesion dict
    lesion_dict = load_dict(lesion_dict_path)
    focussed_dict_print(lesion_dict, 'lesion_dict')

    # # get key_conv_layers_list
    lesion_info = lesion_dict['lesion_info']
    lesion_path = lesion_info['lesion_path']
    lesion_highlighs = lesion_info["lesion_highlights"]
    key_conv_layers_list = list(lesion_highlighs.keys())
    # # remove unnecesary items from key layers list
    if 'highlights' in key_conv_layers_list:
        key_conv_layers_list.remove('highlights')
    if 'output' in key_conv_layers_list:
        key_conv_layers_list.remove('output')
    if 'Output' in key_conv_layers_list:
        key_conv_layers_list.remove('Output')

    # # sel_dict
    sel_dict = load_dict(sel_dict_path)
    if key_conv_layers_list[0] in sel_dict['sel_info']:
        print('\n found old sel dict layout')
        key_relu_layers_list = list(sel_dict['sel_info'].keys())
        old_sel_dict = True

        sel_info = sel_dict['sel_info']
        short_sel_measures_list = list(sel_info[key_conv_layers_list[0]][0]['sel'].keys())
        csb_list = list(sel_info[key_conv_layers_list[0]][0]['class_sel_basics'].keys())
        sel_measures_list = short_sel_measures_list + csb_list

    else:
        print('\n found NEW sel dict layout')
        old_sel_dict = False
        sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
        sel_measures_list = list(sel_info[key_conv_layers_list[0]][0].keys())
        key_relu_layers_list = list(sel_info.keys())
        # print(sel_info.keys())


    n_conv_layers = len(key_conv_layers_list)
    n_layers = n_conv_layers

    # # get key_relu_layers_list
    if use_relu:
        # # remove unnecesary items from key layers list
        if 'sel_analysis_info' in key_relu_layers_list:
            key_relu_layers_list.remove('sel_analysis_info')
        if 'output' in key_relu_layers_list:
            output_idx = key_relu_layers_list.index('output')
            key_relu_layers_list = key_relu_layers_list[:output_idx]
        if 'Output' in key_relu_layers_list:
            output_idx = key_relu_layers_list.index('Output')
            key_relu_layers_list = key_relu_layers_list[:output_idx]

        # # put together lists of 1. sel_relu_layers, 2. key_conv_layers_list.
        n_activation_layers = sum("activation" in layers for layers in key_relu_layers_list)
        n_conv_layers = len(key_conv_layers_list)

        if n_activation_layers == n_conv_layers:
            activation_layers = [layers for layers in key_relu_layers_list if "activation" in layers]
            link_layers_dict = dict(zip(reversed(key_conv_layers_list), reversed(activation_layers)))
        else:
            print("n_activation_layers: {}\n{}".format(n_activation_layers, key_relu_layers_list))
            print("n_conv_layers: {}\n{}".format(n_conv_layers, key_conv_layers_list))

            raise TypeError('should be same number of activation layers and lesioned layers')

        if verbose is True:
            focussed_dict_print(link_layers_dict, 'link_layers_dict')

    # # # get info
    exp_cond_path = sel_dict['topic_info']['exp_cond_path']
    output_filename = sel_dict['topic_info']['output_filename']

    # # load data
    # # check for training data
    use_dataset = sel_dict['GHA_info']['use_dataset']

    n_cats = sel_dict['data_info']["n_cats"]

    if use_dataset in sel_dict['data_info']:
        n_items = sel_dict["data_info"][use_dataset]["n_items"]
        items_per_cat = sel_dict["data_info"][use_dataset]["items_per_cat"]
    else:
        n_items = sel_dict["data_info"]["n_items"]
        items_per_cat = sel_dict["data_info"]["items_per_cat"]
    if type(items_per_cat) is int:
        items_per_cat = dict(zip(list(range(n_cats)), [items_per_cat] * n_cats))

    if plot_type != 'OneVsAll':
        if n_cats > 20:
            plot_type = 'OneVsAll'
            print("\n\n\nWARNING!  There are lots of classes, it might make a messy plot"
                  "Switching to OneVsAll\n")

    if sel_dict['GHA_info']['gha_incorrect'] == 'False':
        # # only gha for correct items
        n_items = sel_dict['GHA_info']['scores_dict']['n_correct']
        items_per_cat = sel_dict['GHA_info']['scores_dict']['corr_per_cat_dict']

    # # load hid acts dict called hid_acts.pickle
    """
    Hid_acts dict has numbers as the keys for each layer.
    Some layers (will be missing) as acts only recorded from some layers (e.g., [17, 19, 20, 22, 25, 26, 29, 30])
    hid_acts_dict.keys(): dict_keys([0, 1, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 19, 20, 22, 25, 26, 29, 30])
    hid_acts_dict[0].keys(): dict_keys(['layer_name', 'layer_class', 'layer_shape', '2d_acts', 'converted_to_2d'])
    In each layer there is ['layer_name', 'layer_class', 'layer_shape', '2d_acts']
    For 4d layers (conv, pool) there is also, key, value 'converted_to_2d': True
    """

    # # check if I have saved the location to this file
    hid_acts_pickle_name = sel_dict["GHA_info"]["hid_act_files"]['2d']
    if 'gha_path' in sel_dict['GHA_info']:
        gha_path = sel_dict['GHA_info']['gha_path']
        hid_acts_path = os.path.join(gha_path, hid_acts_pickle_name)
    else:
        hid_act_items = 'all'
        if sel_dict['GHA_info']['gha_incorrect'] == False:
            hid_act_items = 'correct'

        gha_folder = '{}_{}_gha'.format(hid_act_items, use_dataset)
        hid_acts_path = os.path.join(exp_cond_path, gha_folder, hid_acts_pickle_name)
    with open(hid_acts_path, 'rb') as pkl:
        hid_acts_dict = pickle.load(pkl)
    print("\nopened hid_acts.pickle")
    print(hid_acts_dict.keys())


    # # dict to get the hid_acts_dict key for each layer based on its name
    get_hid_acts_number_dict = dict()
    for key, value in hid_acts_dict.items():
        hid_acts_dict_layer = value['layer_name']
        hid_acts_layer_number = key
        get_hid_acts_number_dict[hid_acts_dict_layer] = hid_acts_layer_number


    if sel_measures == 'all':
        sel_measures = sel_measures_list

    # # new master page [layer, unit, item, class, hid_act, normed_act, item_change, prec, CCMA, info)
    # item_act_fail_reg_array = []
    master_headers = ['layer', 'unit', 'item', 'class', 'hid_acts', 'normed_acts', 'l_fail'] + sel_measures

    MASTER_df = pd.DataFrame(data=None, columns=master_headers)
    print(MASTER_df.head())

    # # where to save files
    save_plots_name = 'item_act_fail_reg'
    if plot_type is "OneVsAll":
        save_plots_name = 'item_act_fail_reg/{}'.format(sel_measures)
    # if use_relu:
    #     save_plots_name = 'item_act_fail_reg_ReLu'
    save_plots_path = os.path.join(lesion_path, save_plots_name)
    if test_run is True:
        save_plots_path = os.path.join(save_plots_path, 'test')
    if not os.path.exists(save_plots_path):
        os.makedirs(save_plots_path)
    os.chdir(save_plots_path)
    print("\ncurrent wd: {}".format(os.getcwd()))

    # # check for master list
    already_got_master = False
    if use_relu:
        master_filename = "{}_item_act_fail_ReLu_MASTER.pickle".format(output_filename)
    else:
        master_filename = "{}_item_act_fail_MASTER.pickle".format(output_filename)

    if os.path.isfile(master_filename):
        MASTER_df = pickle.load(open(master_filename, "rb"))
        already_got_master = True
        print("\nAlready have the MASTER_df")
        print(MASTER_df.head())

    else:

        print("\n\n**********************"
              "\nlooping through layers"
              "\n**********************\n")

        # for layer_index, (relu_layer_name, conv_layer_name) in enumerate(link_layers_dict.items()):
        for layer_index, conv_layer_name in enumerate(reversed(key_conv_layers_list)):

            if test_run == True:
                if layer_index > 2:
                    continue

            if type(top_layers) is int:
                if top_layers < n_activation_layers:
                    if layer_index > top_layers:
                        continue

            use_layer_name = conv_layer_name
            if use_relu:
                use_layer_name = link_layers_dict[conv_layer_name]

            use_layer_number = get_hid_acts_number_dict[use_layer_name]
            hid_acts_dict_layer = hid_acts_dict[use_layer_number]

            if use_layer_name != hid_acts_dict_layer['layer_name']:
                print("conv_layer_name: {}".format(conv_layer_name))
                print("use_layer_name: {}".format(use_layer_name))
                print("use_layer_number: {}".format(use_layer_number))
                print("hid_acts_dict_layer['layer_name']: {}".format(hid_acts_dict_layer['layer_name']))
                focussed_dict_print(get_hid_acts_number_dict, 'get_hid_acts_number_dict')
                # print("get_hid_acts_number_dict: {}".format(get_hid_acts_number_dict))

                raise TypeError("use_layer_number and hid_acts_dict_layer['layer_name'] should match!"
                                "\nfrom link_layers_dict: {}"
                                "\nhid_acts_dict_layer['layer_name']: {}".format(use_layer_number,
                                                                                 hid_acts_dict_layer['layer_name']))

            hid_acts_array = hid_acts_dict_layer['2d_acts']
            hid_acts_df = pd.DataFrame(hid_acts_array, dtype=float)

            # # load item change details
            """# # four possible states
                full model      after_lesion    code
            1.  1 (correct)     0 (wrong)       -1
            2.  0 (wrong)       0 (wrong)       0
            3.  1 (correct)     1 (correct)     1
            4.  0 (wrong)       1 (correct)     2
    
            """
            item_change_name = "{}/{}_{}_item_change.csv".format(lesion_path, output_filename, conv_layer_name)
            item_change_df = pd.read_csv(item_change_name, header=0, dtype=int, index_col=0)

            # prop_change_df = pd.read_csv('{}/{}_{}_prop_change.csv'.format(lesion_path, output_filename, conv_layer_name),
            #                              header=0,
            #                              # dtype=float,
            #                              index_col=0)

            if verbose == True:
                print("\n*******************************************"
                      "\n{}. use layer {}: {} \tlesion layer: {}"
                      "\n*******************************************".format(layer_index, use_layer_number,
                                                                             use_layer_name, conv_layer_name))
                # focussed_dict_print(hid_acts_dict[layer_index])
                print("\n\thid_acts {} shape: {}".format(use_layer_name, hid_acts_df.shape))
                print(
                    "\tloaded: {}_{}_item_change.csv: {}".format(output_filename, conv_layer_name, item_change_df.shape))

            units_per_layer = len(hid_acts_df.columns)

            print("\n\n\t**** loop through units ****")
            for unit_index, unit in enumerate(hid_acts_df.columns):

                if test_run == True:
                    if unit_index > 2:
                        continue

                conv_layer_and_unit = "{}.{}".format(conv_layer_name, unit)

                print("\n\n*************\nrunning layer {} of {}: unit {} of {}\n************".format(
                    layer_index, n_layers, unit, units_per_layer))

                # # make new df with just [item, hid_acts*, class, item_change*] *for this unit
                unit_df = item_change_df[["item", "class", conv_layer_and_unit]].copy()
                # print(hid_acts_df)
                this_unit_hid_acts = hid_acts_df.loc[:, unit]

                # # check for dead relus
                if sum(np.ravel(this_unit_hid_acts)) == 0.0:
                    print("\n\n!!!!!DEAD RELU!!!!!!!!...on to the next unit\n")
                    continue

                unit_df.insert(loc=2, column='hid_acts', value=this_unit_hid_acts)

                # if normed is true:
                max_activation = max(this_unit_hid_acts)
                normed_acts = np.true_divide(this_unit_hid_acts, max_activation)
                unit_df.insert(loc=3, column='normed_acts', value=normed_acts)

                use_layer_name_col = [use_layer_name for i in this_unit_hid_acts]
                unit_df.insert(loc=0, column='layer', value=use_layer_name_col)

                unit_num_col = [unit for i in this_unit_hid_acts]
                unit_df.insert(loc=1, column='unit', value=unit_num_col)


                unit_df = unit_df.rename(index=str, columns={conv_layer_and_unit: 'item_change'})

                if verbose is True:
                    print("\n\tall items - unit_df: ", unit_df.shape)

                # # remove rows where network failed originally and after lesioning this unit - uninteresting
                old_df_length = len(unit_df)
                unit_df = unit_df.loc[unit_df['item_change'] != 0]
                if verbose is True:
                    n_fail_fail = old_df_length - len(unit_df)
                    print("\n\t{} fail-fail items removed - new shape unit_df: {}".format(n_fail_fail, unit_df.shape))


                # # remove rows where network failed originally and passed after lesioning this unit
                old_df_length = len(unit_df)
                unit_df = unit_df.loc[unit_df['item_change'] != 2]
                if verbose is True:
                    n_fail_pass = old_df_length - len(unit_df)
                    print("\n\t{} fail-pass items removed - new shape unit_df: {}".format(n_fail_pass, unit_df.shape))


                # # make item fail specific column
                l_fail = [1 if x == -1 else 0 for x in unit_df['item_change']]
                unit_df.insert(loc=6, column='l_fail', value=l_fail)
                unit_df = unit_df.drop(columns="item_change")


                print("\nunit df")
                print(unit_df.head())
                # # new master page [layer, unit, item, class, hid_act, normed_act, l_fail, prec, CCMA, info)


                # # getting best sel measures
                for measure in sel_measures:
                    # # includes if statement since some units have not score (dead relu?)
                    if old_sel_dict:
                        sel_measure_dict = sel_dict['sel_info'][use_layer_name][unit][measure]
                    else:
                        sel_measure_dict = sel_info[use_layer_name][unit][measure]

                    unit_df[measure] = unit_df['class'].map(sel_measure_dict)

                MASTER_df = MASTER_df.append(unit_df, ignore_index=True)
                print("MASTER_df: {}".format(MASTER_df.shape))

        print("\n\n********************************"
              "\nfinished looping through layers"
              "\n********************************\n")

    # # part 2 , use master to do stats
    if use_relu:
        txt = open('{}_ReLu_item_act_fail.txt'.format(output_filename), 'w')
    else:
        txt = open('{}_item_act_fail.txt'.format(output_filename), 'w')

    print("MASTER_df: {}".format(MASTER_df.shape))
    # print(list(unit_df))
    data = MASTER_df.drop(
        columns=['layer', 'unit', 'item', 'class', 'normed_acts'])  # , 'Zhou_prec', 'CCMAs', 'max_informed'], )

    txt.write("\nsel_measures: {}".format(sel_measures))
    txt.write("\nuse_relu: {}".format(use_relu))

    txt.write("\ndata shape: {}".format(data.shape))
    txt.write("\ndata columns: {}".format(list(data.columns)))

    # barplot dependent variable
    # sns.countplot(x='l_fail', data=data, palette='hls')
    # plt.show()
    # plt.close()

    # print("Check the missing values")
    # print(data.isnull().sum())

    # print("class distribution")
    # sns.countplot(y="class", data=data)
    # plt.show()
    # plt.close()

    # print("Check the independence between the independent variables")
    # sns.heatmap(data.corr())
    # plt.show()
    # plt.close()

    plt.figure(figsize=(15, 8))
    ax = sns.kdeplot(data["Zhou_prec"], color="darkturquoise", shade=True)
    sns.kdeplot(data["CCMAs"], color="lightcoral", shade=True)
    sns.kdeplot(data["max_informed"], color="orange", shade=True)

    plt.legend(['Zhou_prec', 'CCMAs', 'max_informed'])
    plt.title('Density Plot of Selectivity measures')
    ax.set(xlabel='Selectivity')
    if use_relu:
        plt.savefig("{}_ReLu_sel_dist.png".format(output_filename))
    else:
        plt.savefig("{}_sel_dist.png".format(output_filename))
    plt.close()


    plt.figure(figsize=(15, 8))
    ax = sns.kdeplot(data["hid_acts"], color="darkturquoise", shade=True)
    plt.title('Density Plot of hidden activations')
    ax.set(xlabel='Hidden Activations')
    if use_relu:
        plt.savefig("{}_ReLu_act_dist.png".format(output_filename))
    else:
        plt.savefig("{}_act_dist.png".format(output_filename))
    plt.close()
    # plt.xlim(-20, 200)
    # plt.show()

    # # compare means
    print("\nget descriptives")
    print(data.groupby('l_fail').mean())
    l_fail = data[data['l_fail'] == 1]
    l_fail_mean = l_fail['hid_acts'].mean()
    n_failed = len(l_fail)

    l_passed = data[data['l_fail'] == 0]
    l_passed_mean = l_passed['hid_acts'].mean()
    n_passed = len(l_passed)

    print("proportion failed: {}".format(data['l_fail'].mean()))
    print("n_failed: {}".format(n_failed))

    print("proportion passed: {}".format(1 - (data['l_fail'].mean())))
    print("n_passed: {}".format(n_passed))

    txt.write("\n\nDescriptives")
    txt.write("\nn_failed: {}".format(n_failed))
    txt.write("\nproportion failed: {}".format(data['l_fail'].mean()))
    txt.write("\nn_passed: {}".format(n_passed))
    txt.write("\nproportion passed: {}".format(1 - (data['l_fail'].mean())))

    print("\nt-test")
    print("l_fail_mean: {}".format(l_fail_mean))
    print("l_passed_mean: {}".format(l_passed_mean))

    t_test_t, t_test_p = stats.ttest_ind(l_fail['hid_acts'].values, l_passed['hid_acts'].values)
    print("t_test_t: {}".format(t_test_t))
    print("t_test_p: {}".format(t_test_p))

    txt.write("\n\nt-test")
    txt.write("\nl_fail_mean: {}".format(l_fail_mean))
    txt.write("\nl_passed_mean: {}".format(l_passed_mean))
    txt.write("\nt_test_t: {}, t_test_p: {}".format(t_test_t, t_test_p))

    # # ROC
    print("\ndata ROC")
    y = [1 if i == 1 else 0 for i in np.array(np.ravel(data['l_fail'].values))]
    X = np.ravel(data['hid_acts'])

    fpr, tpr, thr = roc_curve(y, X, pos_label=1)

    # fpr, tpr, thr = roc_curve(data['l_fail'].values, data['hid_acts'].values)
    roc_auc = auc(fpr, tpr)
    print("ROC_AUC: {}".format(roc_auc))
    print("ROC AUC is equal to the probability that a random positive example (failed)"
          "will be ranked above a random negative example (passed).")
    # plt.plot(fpr, tpr, label="data auc=" + str(roc_auc))
    # plt.ylabel('TPR')
    # plt.xlabel('FPR')
    # plt.legend(loc=4)
    # plt.show()
    # plt.close()
    idx = np.min(np.where(tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95

    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    # plt.show()
    if use_relu:
        plt.savefig("{}_ReLu_data_ROC.png".format(output_filename))
    else:
        plt.savefig("{}_data_ROC.png".format(output_filename))
    plt.close()


    txt.write("\n\nROC AUC is equal to the probability that a random positive example (failed)"
              "will be ranked above a random negative example (passed).")
    txt.write("\ndata ROC_AUC: {}".format(roc_auc))

    # # simple logistic regression
    data = data.drop(columns=['l_fail'])
    print("\ndata_headers: ", list(data))

    reg_cols = ['hid_acts', 'Zhou_prec', 'CCMAs', 'max_informed']
    X = data[reg_cols]

    # instantiate a logistic regression model, and fit with X and y
    class_weights = {0: n_passed, 1: n_failed}
    print("class_weights: {}".format(class_weights))
    model = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced')  # 'class_weights)
    # model.fit(X, y)

    # # Recursive feature elimination RFE
    rfe = RFE(model, 3)
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    print('Selected features: %s' % list(X.columns[rfe.support_]))
    txt.write("\n\nRecursive feature elimination RFE")
    txt.write('\nSelected features: %s' % list(X.columns[rfe.support_]))


    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=LogisticRegression(solver='lbfgs', class_weight='balanced'),
                  step=1, cv=10, scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X.columns[rfecv.support_]))

    txt.write("\nOptimal number of features: %d" % rfecv.n_features_)
    txt.write('\nSelected features: %s' % list(X.columns[rfecv.support_]))


    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()
    if use_relu:
        plt.savefig("{}_ReLu_CV_score.png".format(output_filename))
    else:
        plt.savefig("{}_CV_score.png".format(output_filename))
    plt.close()

    Selected_features = list(X.columns[rfecv.support_])
    X = data[Selected_features]

    plt.subplots(figsize=(8, 5))
    sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
    # plt.show()
    if use_relu:
        plt.savefig("{}_ReLu_feat_corr.png".format(output_filename))
    else:
        plt.savefig("{}_feat_corr.png".format(output_filename))
    plt.close()

    '''https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
    4.2.1. Model evaluation based on simple train/test split using train_test_split() function'''



    # create X (features) and y (response)
    X = data[Selected_features]
    # y = final_train['Survived']

    # use train/test split with different random_state values
    # we can change the random_state values that changes the accuracy scores
    # the scores change a lot, this is why testing scores is a high-variance estimate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # check classification scores of logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    print('Train/Test split results:')
    print(logreg.__class__.__name__ + " accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    print(logreg.__class__.__name__ + " log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    print(logreg.__class__.__name__ + " auc is %2.3f" % auc(fpr, tpr))

    txt.write('\n\nTrain/Test split results:')
    txt.write('\n' + logreg.__class__.__name__ + " accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    txt.write('\n' + logreg.__class__.__name__ + " log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    txt.write('\n' + logreg.__class__.__name__ + " auc is %2.3f" % auc(fpr, tpr))

    idx = np.min(np.where(tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95

    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    # plt.show()
    if use_relu:
        plt.savefig("{}_ReLu_model_ROC.png".format(output_filename))
    else:
        plt.savefig("{}_model_ROC.png".format(output_filename))
    plt.close()

    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +
          "and a specificity of %.3f" % (1 - fpr[idx]) +
          ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx]) * 100))

    txt.write("\n\nUsing a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +
              "and a specificity of %.3f" % (1 - fpr[idx]) +
              ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx]) * 100))

    print("\nexamine the coefficients")
    coefficients = pd.DataFrame(zip(X.columns, np.transpose(logreg.coef_)))
    print(coefficients)
    txt.write("\nexamine the coefficients\n")
    txt.write(coefficients.to_string())

    # # check the accuracy on the training set
    # y_pred = model.predict(X)
    # # print("\ny_pred: ", y_pred)
    #
    # y_pred_proba = model.predict_proba(X)  # 2d array with probs for both classes
    # # print("\ny_pred_proba: ", y_pred_proba)
    #
    # score = model.score(X, y)
    # print("\n\nNick's original model stuff")
    # txt.write("\n\nNick's original model stuff")
    #
    #
    # print("\nLogisticRegression score: ", score)

    # print("just guessing no everytime would get you {}".format(1 - (np.mean(y))))
    # print('this model does {} better than quessing chance'.format(score - (1 - (np.mean(y)))))
    #
    # print("\nexamine the coefficients")
    # coefficients = pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
    # print(coefficients)
    #
    # print("confusion matrix (diagonals are correct)")
    # conf_matrix = confusion_matrix(y, y_pred)
    # print(conf_matrix)

    # class_names = [0, 1]  # name  of classes
    # class_names = ["pass", 'fail']  # name  of classes

    # fig, ax = plt.subplots()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names)
    # plt.yticks(tick_marks, class_names)
    # # create heatmap
    # sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    # ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    # plt.title('Confusion matrix', y=1.1)
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.show()
    # plt.close()

    # accuracy = accuracy_score(y, y_pred)
    # precision = precision_score(y, y_pred)
    # recall = recall_score(y, y_pred)
    # f1 = 2 * (precision * recall) / (precision + recall)
    #
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("f1:", f1)

    # print("model ROC")
    # print("ROC AUC is equal to the probability that a random positive example (failed)"
    #       "will be ranked above a random negative example (passed).")
    # fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1], pos_label=1, )
    # model_auc = roc_auc_score(y, y_pred_proba[:, 1])
    # print("model_auc: {}".format(model_auc))

    # plt.plot(fpr, tpr, label="model auc=" + str(model_auc))
    # plt.legend(loc=4)
    # plt.show()
    # plt.close()

    # txt.write("\n\nLogistic regression with {}".format(list(X)))
    # txt.write("\nLogisticRegression score: {}".format(score))
    # txt.write("\njust guessing no everytime would get you {}".format(1 - (np.mean(y))))
    # txt.write('\nthis model does {} better than quessing chance'.format(score - (1 - (np.mean(y)))))
    # txt.write("\nexamine the coefficients\n")
    # txt.write(coefficients.to_string())
    # txt.write("\n\nconfusion matrix (diagonals are correct)")
    # txt.write("\n\tpred_0\tpred_1")
    # txt.write("\ntrue_0\t{}\t{}".format(conf_matrix[0][0], conf_matrix[0][1]))
    # txt.write("\ntrue_1\t{}\t{}".format(conf_matrix[1][0], conf_matrix[1][1]))
    #
    # txt.write("\n\naccuracy: {}".format(accuracy))
    # txt.write("\nprecision: {}".format(precision))
    # txt.write("\nrecall: {}".format(recall))
    # txt.write("\nf1: {}".format(f1))
    # txt.write("\nmodel_auc: {}".format(model_auc))

    txt.close()
    if already_got_master == False:
        if use_relu:
            MASTER_df.to_pickle("{}_item_act_fail_ReLu_MASTER.pickle".format(output_filename))
        else:
            MASTER_df.to_pickle("{}_item_act_fail_MASTER.pickle".format(output_filename))

    print("End of script")

# ###################################################################################
# print("\n!\n!\n!TEST RUN FROM BOTTOM OF SCRIPT")
#
# item_act_fail_regression(sel_dict_path='/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                        'CIFAR10_models/CIFAR10_models_c4p2_adam_bn/all_test_set_gha/correct_sel/'
#                                        'CIFAR10_models_c4p2_adam_bn_sel_dict.pickle',
#                          lesion_dict_path='/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/'
#                                           'CIFAR10_models_c4p2_adam_bn/all_test_set_gha/lesion/'
#                                           'CIFAR10_models_c4p2_adam_bn_lesion_dict.pickle',
#                          plot_type='classes',
#                          sel_measures=['Zhou_prec', 'CCMAs', 'max_informed'],
#                          use_relu=True,
#                          verbose=True,
#                          # test_run=True
#                          )


########################################################################################

def class_acc_sel_corr(lesion_dict_path, sel_dict_path,
                       use_relu=False,
                       sel_measures='all',
                       test_run=False,
                       verbose=False):
    """
    Script uses lesion dict and sel dict
    Does class sel per unit predict class drop per unit
    at the moment it only does max drop and selectivity for that class.

    for each class or for max drop class

    for each layer
        for each unit
            get class drops and sel scores
            append max drop and max sel into columns

            predict each class or max class

    :param lesion_dict_path: path to dict from output of lesioning
    :param sel_dict_path: path to dict of output from selectivity
    :param use_relu: if False, only uses sel scores for lesioned layers (conv, dense), if true, only uses sel scores
        for ReLu layers, as found with link_layers_dict
    :return: lesion_regression_dict
    """

    print("**** running class_acc_sel_corr() ****")

    lesion_dict = load_dict(lesion_dict_path)
    focussed_dict_print(lesion_dict, "lesion dict")

    n_cats = lesion_dict['data_info']['n_cats']
    output_filename = lesion_dict['topic_info']['output_filename']

    lesion_info = lesion_dict['lesion_info']
    lesion_path = lesion_info['lesion_path']

    # # get key layers list
    lesion_highlighs = lesion_info["lesion_highlights"]
    # focussed_dict_print(lesion_highlighs)
    key_lesion_layers_list = list(lesion_highlighs.keys())
    # # remove unnecesary items from key layers list
    # # remove unnecesary items from key layers list
    if 'highlights' in key_lesion_layers_list:
        key_lesion_layers_list.remove('highlights')
    if 'output' in key_lesion_layers_list:
        key_lesion_layers_list.remove('output')
    if 'Output' in key_lesion_layers_list:
        key_lesion_layers_list.remove('Output')
    print("\nkey_lesion_layers_list\n{}".format(key_lesion_layers_list))

    # # sel_dict
    sel_dict = load_dict(sel_dict_path)
    if key_lesion_layers_list[0] in sel_dict['sel_info']:
        print('\n found old sel dict layout')
        old_sel_dict = True
        sel_info = sel_dict['sel_info']
        short_sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0]['sel'].keys())
        csb_list = list(sel_info[key_lesion_layers_list[0]][0]['class_sel_basics'].keys())
        auto_sel_measures = short_sel_measures_list + csb_list
    else:
        print('\n found NEW sel dict layout')
        old_sel_dict = False
        sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
        auto_sel_measures = list(sel_info[key_lesion_layers_list[0]][0].keys())

    # todo: fix these dodgy measures or find the problem
    '''get rid of measures that I've had trouble with'''

    if 'nz_count' in auto_sel_measures:
        auto_sel_measures.remove('nz_count')
    if 'max_info_f1' in auto_sel_measures:
        auto_sel_measures.remove('max_info_f1')

    if use_relu is True:
        # # get key_gha_sel_layers_list
        key_gha_sel_layers_list = list(sel_info.keys())
        # # remove unnecesary items from key layers list
        if 'sel_analysis_info' in key_gha_sel_layers_list:
            key_gha_sel_layers_list.remove('sel_analysis_info')
        if 'output' in key_gha_sel_layers_list:
            output_idx = key_gha_sel_layers_list.index('output')
            key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]
        if 'Output' in key_gha_sel_layers_list:
            output_idx = key_gha_sel_layers_list.index('Output')
            key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]

        # # put together lists of 1. sel_gha_layers, 2. key_lesion_layers_list.
        n_activation_layers = sum("activation" in layers for layers in key_gha_sel_layers_list)
        n_lesion_layers = len(key_lesion_layers_list)

        if n_activation_layers == n_lesion_layers:
            n_layers = n_activation_layers
            activation_layers = [layers for layers in key_gha_sel_layers_list if "activation" in layers]
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(activation_layers)))
        else:
            raise TypeError('should be same number of activation layers and lesioned layers')

        if verbose is True:
            focussed_dict_print(link_layers_dict, 'link_layers_dict')

    print("\n\nSaving lesion_corr_dict")
    lesion_dir_path, _ = os.path.split(lesion_dict_path)
    sel_corr_path = os.path.join(lesion_dir_path, 'sel_les_corr')
    if not os.path.exists(sel_corr_path):
        os.makedirs(sel_corr_path)
    print("Saving stuff to: {}".format(sel_corr_path))

    if sel_measures == 'all':
        sel_measures_list = auto_sel_measures
    else:
        sel_measures_list = sel_measures

    print("\nsel_measures_list\n{}".format(sel_measures_list))

    sel_corr_dict = dict()  # key=measure, val=score
    for i in key_lesion_layers_list:
        sel_corr_dict[i] = dict()
    sel_corr_dict['all_layers'] = dict()

    """loop through sel_measures_list
    get long array of all layers concatenated and list all max lesion drops.
    do correlations on this whole thing at once,
    save then repeat for next sel measure"""

    print("\nlooping through network to get selectivity scores as big array")

    for sel_measure in sel_measures_list:

        this_sel_measure = sel_measure
        print("\n{}".format(this_sel_measure))

        """loop through all layers"""
        all_layer_sel_array = []  # append each sel_layer to this to give array (each unit, classes)

        all_layer_class_drops = []  # # extend each lesion_layer to this to give list of all units in all layers

        all_les_sel_pairs = []  # max_class_lesion_drop and sel for that class

        # measure_sel_corr_dict = dict()  # key=measure, val=score
        # sel_corr_dict[lesion_layer] = dict()

        for test_layer_count, lesion_layer in enumerate(key_lesion_layers_list):

            if test_run:
                test_layer_count = + 1
                if test_layer_count > 2:
                    continue

            sel_layer = lesion_layer
            if use_relu:
                sel_layer = link_layers_dict[lesion_layer]

            if verbose == True:
                print("\tsel_layer: {}\tlesion_layer: {}".format(sel_layer, lesion_layer))

            sel_layer_info = sel_info[sel_layer]
            
            layer_les_sel_pairs = []

            # sel_corr_dict[lesion_layer] = dict()

            # # lesion stuff
            # # conv2d_6
            lesion_per_unit_path = '{}/{}_{}_prop_change.csv'.format(lesion_path, output_filename, lesion_layer)

            lesion_per_unit = pd.read_csv(lesion_per_unit_path, index_col=0)
            # lesion_per_unit = nick_read_csv(lesion_per_unit_path)
            # lesion_per_unit.set_index(0)

            lesion_cols = list(lesion_per_unit)
            # print("\nlesion_per_unit")
            # print(lesion_cols)
            # print(lesion_per_unit.head())

            '''get max class drop per lesion_layer'''
            # # loop through lesion units (df columns) to find min class drop
            lesion_min_dict = dict()
            max_class_drop_list = []
            for index, l_unit in enumerate(lesion_cols):
                # print(lesion_per_unit[l_unit])
                min_class_val = lesion_per_unit[l_unit].min()
                min_class = lesion_per_unit[l_unit].idxmin()
                max_class_drop_list.append(int(min_class))
                # print("{}: class: {}  {}".format(index, min_class, min_class_val))
                lesion_min_dict[index] = {'unit': index, "l_min_class": min_class, 'l_min_drop': min_class_val}

            '''get sel_layer sel values'''
            # get array of sel values
            layer_sel_array = []
            for k, v in sel_layer_info.items():
                unit = k
                sel_measure = this_sel_measure

                if old_sel_dict:
                    if sel_measure in v['sel']:
                        sel_items = v['sel'][sel_measure]
                    elif sel_measure in v['class_sel_basics']:
                        sel_items = v['class_sel_basics'][sel_measure]
                else:
                    sel_items = v[sel_measure]

                # # just check it is just classes in there
                if 'total' in sel_items.keys():
                    del sel_items['total']
                if 'perplexity' in sel_items.keys():
                    del sel_items['perplexity']

                if len(list(sel_items.keys())) != n_cats:
                    print("\nERROR, {} hasn't got enough classes".format(sel_measure))
                    # print("error found", sel_items)
                    for i in range(n_cats):
                        if i not in sel_items:
                            sel_items[i] = 0.0
                    ordered_dict = dict()
                    for j in range(n_cats):
                        ordered_dict[j] = sel_items[j]

                    sel_items = dict()
                    sel_items = ordered_dict
                    print('sel_items should be sorted now', sel_items)

                sel_values = list(sel_items.values())

                layer_sel_array.append(sel_values)

                les_drop_class = int(lesion_min_dict[k]['l_min_class'])
                drop_class_sel = sel_values[les_drop_class]

                all_les_sel_pairs.append([lesion_min_dict[k]['l_min_drop'], drop_class_sel])
                layer_les_sel_pairs.append([lesion_min_dict[k]['l_min_drop'], drop_class_sel])


                # print(unit, sel_values)

            # print("\nlayer_sel_array\n{}".format(layer_sel_array))
            # print("\t\tlayer_sel_array: {}".format(np.shape(layer_sel_array)))

            all_layer_sel_array = all_layer_sel_array + layer_sel_array
            # print("\t\tall_layer_sel_array: {}".format(np.shape(all_layer_sel_array)))

            # # lesion stuff went here...

            # # check for missing values (prob dead relus) when using different layers
            sel_units, classes = np.shape(layer_sel_array)
            les_units = len(max_class_drop_list)
            if use_relu:
                if sel_units != les_units:
                    print("\n\number of units is wrong")
                    print("sel_units: {}\nles_units: {}".format(sel_units, les_units))

                    if len(max_class_drop_list) > sel_units:
                        available_sel_units = list(sel_layer_info.keys())
                        masked_class_drops = [max_class_drop_list[i] for i in available_sel_units]
                        max_class_drop_list = masked_class_drops

            # # if there are any NaNs:
            if np.any(np.isnan(layer_sel_array)):
                print("TRUE nan")
                # layer_sel_array[np.isnan(layer_sel_array)] = 0
                layer_sel_array = np.nan_to_num(layer_sel_array, copy=False)

                if np.any(np.isnan(layer_sel_array)):
                    print("TRUE still nan")
            if np.all(np.isfinite(layer_sel_array)):
                print("TRUE inf")

            # layer_sel_array = np.array(layer_sel_array)  # [np.isnan(layer_sel_array)] = 0.0
            # layer_sel_array = np.nan_to_num(layer_sel_array, copy=False)
            # layer_sel_array[np.isneginf(layer_sel_array)] = 0

            all_layer_class_drops.extend(max_class_drop_list)
            # print("\t\tall_layer_class_drops: {}".format(np.shape(all_layer_class_drops)))

            # focussed_dict_print(lesion_min_dict)
            # print("\nmax_class_drop_list\n{}".format(max_class_drop_list))
            if verbose == True:
                print("\t\tlayer_sel: {}, all_layers_sel: {}; all_layers_class_drops: "
                      "{}".format(np.shape(layer_sel_array), np.shape(all_layer_sel_array),
                                  np.shape(all_layer_class_drops)))
                
            # # layer correlations
            # # # sel lesion correlations
            print("layer_les_sel_pairs shape: ", np.shape(layer_les_sel_pairs))
            max_lesion_vals = [i[0] for i in layer_les_sel_pairs]
            sel_pair_score = [i[1] for i in layer_les_sel_pairs]
            print("max_lesion_vals shape: ", np.shape(max_lesion_vals))

            corr_coef, corr_p = stats.pearsonr(sel_pair_score, max_lesion_vals)

            print("{} corr: {}, p = {}".format(this_sel_measure, corr_coef, corr_p))

            sel_corr_dict[lesion_layer]['{}_corr_coef'.format(sel_measure)] = corr_coef
            sel_corr_dict[lesion_layer]['{}_corr_p'.format(sel_measure)] = corr_p

            sns.regplot(x=sel_pair_score, y=max_lesion_vals)
            plt.ylabel("Max class drop")
            plt.xlabel("selectivity score")
            plt.suptitle("{} class drop vs {}".format(lesion_layer, sel_measure))
            if round(corr_p, 3) == 0.000:
                plt.title("r={}, p<.001".format(round(corr_coef, 3)))
            else:
                plt.title("r={}, p={}".format(round(corr_coef, 3), round(corr_p, 3)))
            print(os.getcwd())
            if use_relu:
                plt.savefig("{}/{}_{}_{}_ReLu_corr.png".format(sel_corr_path, output_filename, lesion_layer, sel_measure))
            else:
                plt.savefig("{}/{}_{}_{}_corr.png".format(sel_corr_path, output_filename, lesion_layer, sel_measure))
            plt.close()

            # sel_corr_dict[lesion_layer] = measure_sel_corr_dict

        # # # sel lesion correlations
        print("all_les_sel_pairs shape: ", np.shape(all_les_sel_pairs))
        max_lesion_vals = [i[0] for i in all_les_sel_pairs]
        sel_pair_score = [i[1] for i in all_les_sel_pairs]
        print("max_lesion_vals shape: ", np.shape(max_lesion_vals))

        corr_coef, corr_p = stats.pearsonr(sel_pair_score, max_lesion_vals)

        print("{} corr: {}, p = {}".format(this_sel_measure, corr_coef, corr_p))

        # all_layer_sel_corr_dict = dict()
        # sel_corr_dict['all_layers'] = dict()

        sel_corr_dict['all_layers']['{}_corr_coef'.format(sel_measure)] = corr_coef
        sel_corr_dict['all_layers']['{}_corr_p'.format(sel_measure)] = corr_p

        sns.regplot(x=sel_pair_score, y=max_lesion_vals)
        plt.ylabel("Max class drop")
        plt.xlabel("selectivity score")
        plt.suptitle("class drop vs {}".format(sel_measure))
        if round(corr_p, 3) == 0.000:
            plt.title("r={}, p<.001".format(round(corr_coef, 3)))
        else:
            plt.title("r={}, p={}".format(round(corr_coef, 3), round(corr_p, 3)))
        print(os.getcwd())
        if use_relu:
            plt.savefig("{}/{}_{}_ReLu_corr.png".format(sel_corr_path, output_filename, sel_measure))
        else:
            plt.savefig("{}/{}_{}_corr.png".format(sel_corr_path, output_filename, sel_measure))
        plt.close()

        # plt.show()
        # sel_corr_dict['all_layers'] = all_layer_sel_corr_dict

    print("\n\nfinal sel_corr_dict values")
    print_nested_round_floats(sel_corr_dict)

    if use_relu:
        output_filename = output_filename + '_onlyReLu'
    print("output_filename: ", output_filename)

    lesion_corr_dict = lesion_dict
    lesion_corr_dict['corr_info'] = sel_corr_dict
    # print_nested_round_floats(lesion_corr_dict, 'lesion_corr_dict')

    les_reg_dict_name = "{}/{}_les_sel_corr_dict.pickle".format(sel_corr_path, output_filename)
    pickle_out = open(les_reg_dict_name, "wb")
    pickle.dump(lesion_corr_dict, pickle_out)
    pickle_out.close()

    # les_reg_df = pd.DataFrame(data=sel_corr_dict, index=[0])
    les_reg_df = pd.DataFrame.from_dict(data=sel_corr_dict, orient='index')

    print(les_reg_df.head())
    # nick_to_csv(les_reg_df, "{}/{}_les_sel_corr_dict_nm.csv".format(sel_corr_path, output_filename))
    les_reg_df.to_csv("{}/{}_les_sel_corr_dict_pd.csv".format(sel_corr_path, output_filename))

    return lesion_corr_dict


#
# #
# # # #########################
# # # # old tes hid act experiments
# # # lesion_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/train_script_check/' \
# # #                    'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/all_test_set_gha/lesion/' \
# # #                    'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug_lesion_dict.pickle'
# # # sel_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/train_script_check/' \
# # #                 'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/all_test_set_gha/correct_sel/' \
# # #                 'train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug_sel_dict.pickle'
# # #
# # #
# # # new cifar experiments
# lesion_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/CIFAR10_models_c4p2_adam_bn/' \
#                    'all_test_set_gha/lesion/CIFAR10_models_c4p2_adam_bn_lesion_dict.pickle'
# sel_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/CIFAR10_models_c4p2_adam_bn/' \
#                 'all_test_set_gha/correct_sel/CIFAR10_models_c4p2_adam_bn_sel_dict.pickle'
#
# class_acc_sel_corr_dict = class_acc_sel_corr(lesion_dict_path=lesion_dict_path, sel_dict_path=sel_dict_path,
#                                              use_relu=True,
#                                              sel_measures=['Zhou_prec', 'CCMAs', 'max_informed'],
#                                              verbose=True
#                                              )
#


