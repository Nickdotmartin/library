import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from itertools import product
import datetime
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, simple_dict_print, print_nested_round_floats
from nick_dict_tools import focussed_dict_print, json_key_to_int
from nick_data_tools import load_x_data, load_y_data, nick_to_csv, nick_read_csv


tools_date = int(datetime.datetime.now().strftime("%y%m%d"))
tools_time = int(datetime.datetime.now().strftime("%H%M"))


# todo: only save summary docs as csv.  all other output should be numpy, pickle or excel.
#  or for csv use this https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv


# todo: check to see if I am using boolean statemens in a list (e.g., item correct)
#  change any boolean statements (item correct: True, False) to binary (item correct: 1, 0).
#  This should make checking, summing etc, easier



############################
def get_model_dict(compiled_model, output_layers=2, verbose=False):
    """
    ASSUMES THERE ARE 2 LAYERS FOR OUTPUT (dense and activation)
    Takes a compiled model (model + data input & output shape) and returns a dict containing:
        1. the full get_config output
        2. a summary to use for analysis (just n_filters, units etc)

    :param compiled_model:
    :param model_name: str

    :return: dict: {'config': model_config, 'summary': hid_layers_summary}
    """

    model_config = compiled_model.get_config()
    hid_layers_summary = dict()

    # # get useful info
    all_layers = len(model_config['layers'])
    hid_layers = all_layers-output_layers
    hid_act_layers = 0
    hid_dense_layers = 0
    hid_conv_layers = 0
    hid_UPL = []
    hid_FPL = []

    # layers_of_interest =  ["Conv2D", "Activation", "MaxPooling2D", "Dense"]

    for layer in range(0, all_layers-2):
        if verbose is True:
            print("\nModel layer {}\n{}".format(layer, model_config['layers'][layer]))

        # # get useful info
        layer_dict = {'layer': layer,
                      'name': model_config['layers'][layer]['config']['name'],
                      'class': model_config['layers'][layer]['class_name']}

        if 'units' in model_config['layers'][layer]['config']:
            layer_dict['units'] = model_config['layers'][layer]['config']['units']
            hid_dense_layers += 1
            hid_UPL.append(model_config['layers'][layer]['config']['units'])
        if 'activation' in model_config['layers'][layer]['config']:
            layer_dict['act_func'] = model_config['layers'][layer]['config']['activation']
            hid_act_layers += 1
        if 'filters' in model_config['layers'][layer]['config']:
            layer_dict['filters'] = model_config['layers'][layer]['config']['filters']
            hid_conv_layers += 1
            hid_FPL.append(model_config['layers'][layer]['config']['filters'])
        if 'kernel_size' in model_config['layers'][layer]['config']:
            layer_dict['size'] = model_config['layers'][layer]['config']['kernel_size'][0]
        if 'pool_size' in model_config['layers'][layer]['config']:
            layer_dict['size'] = model_config['layers'][layer]['config']['pool_size'][0]
        if 'strides' in model_config['layers'][layer]['config']:
            layer_dict['strides'] = model_config['layers'][layer]['config']['strides'][0]
        if 'rate' in model_config['layers'][layer]['config']:
            layer_dict["rate"] = model_config['layers'][layer]['config']['rate']

        # # set and save layer details
        hid_layers_summary[layer] = layer_dict

    hid_layers_summary['hid_totals'] = {'act_layers': hid_act_layers, "dense_layers": hid_dense_layers,
                                    "conv_layers": hid_conv_layers, "UPL": hid_UPL, "FPL": hid_FPL,
                                    'analysable': sum([sum(hid_UPL), sum(hid_FPL)])}

    #######
    output_summary = dict()
    # # get useful info
    out_act_layers = 0
    out_dense_layers = 0
    out_conv_layers = 0
    out_UPL = []
    out_FPL = []
    for layer in range(all_layers-2, all_layers):
        if verbose is True:
            print("\nModel layer {}\n{}".format(layer, model_config['layers'][layer]))

        # # get useful info
        out_layer_dict = {'layer': layer,
                      'name': model_config['layers'][layer]['config']['name'],
                      'class': model_config['layers'][layer]['class_name']}

        if 'units' in model_config['layers'][layer]['config']:
            out_layer_dict['units'] = model_config['layers'][layer]['config']['units']
            out_dense_layers += 1
            out_UPL.append(model_config['layers'][layer]['config']['units'])
        if 'activation' in model_config['layers'][layer]['config']:
            out_layer_dict['act_func'] = model_config['layers'][layer]['config']['activation']
            out_act_layers += 1
        if 'filters' in model_config['layers'][layer]['config']:
            out_layer_dict['filters'] = model_config['layers'][layer]['config']['filters']
            out_conv_layers += 1
            out_FPL.append(model_config['layers'][layer]['config']['filters'])
        if 'kernel_size' in model_config['layers'][layer]['config']:
            out_layer_dict['size'] = model_config['layers'][layer]['config']['kernel_size'][0]
        if 'pool_size' in model_config['layers'][layer]['config']:
            out_layer_dict['size'] = model_config['layers'][layer]['config']['pool_size'][0]
        if 'strides' in model_config['layers'][layer]['config']:
            out_layer_dict['strides'] = model_config['layers'][layer]['config']['strides'][0]
        if 'rate' in model_config['layers'][layer]['config']:
            out_layer_dict["rate"] = model_config['layers'][layer]['config']['rate']

        # # set and save layer details
        output_summary[layer] = out_layer_dict

    output_summary['out_totals'] = {'act_layers': out_act_layers, "dense_layers": out_dense_layers,
                                "conv_layers": out_conv_layers, "UPL": out_UPL, "FPL": out_FPL,
                                'analysable': sum([sum(out_UPL), sum(out_FPL)])}
    #######
    layers_summary = {'totals': {"all_layers": all_layers, 'output_layers': output_layers,
                                 'hid_layers': hid_layers},
                      'hid_layers': hid_layers_summary, 'output': output_summary}

    model_info = {'config': model_config,
                  'layers': layers_summary}

    return model_info

#########################


def get_scores(predicted_outputs, y_df, output_filename, verbose=False, save_all_csvs=True,
               return_flat_conf=False):
    """
    Script will compare predicted class and true class to find whether each item was correct.

    use predicted outputs to get predicted categories

    :param predicted_outputs: from loaded_model.predict(x_data).  shape (n_items, n_cats)
    :param y_df:  y item and class
    :param output_filename:  to use when saving csvs
    :param verbose:
    :param save_all_csvs:
    :param return_flat_conf: if false, just add the name to the dict, if true, return actual flat conf matrix


    :return: item_correct_df - item number, class, correct (1 or if incorrect, 0)
    :return: scores_dict - descriptives
    :return: incorrect_items - list of item numbers that were incorrect

    """
    print("\n**** get_scores() ****")

    # todo: repair this - I've changed it for VGG

    n_items, n_cats = np.shape(predicted_outputs)

    predicted_cat = []

    # find the column (class) with the highest prediction
    for row in range(n_items):
        this_row = predicted_outputs[row]
        max_val = max(this_row)
        # max_cat = this_row.tolist().index(max_val)
        max_cat = list(this_row).index(max_val)

        predicted_cat.append(max_cat)

    # # save list of which items were incorrect
    true_cat = [int(i) for i in y_df['class'].tolist()]
    # true_cat = y_df['class'].tolist()  # was returning strings?

    # make a list of whether the preidctions were correct
    item_score = []
    incorrect_items = []
    for index, pred_item in enumerate(predicted_cat):
        if pred_item == true_cat[index]:
            item_score.append(int(1))
        else:
            item_score.append(int(0))
            incorrect_items.append(index)

    item_correct_df = y_df.copy()

    if verbose is True:
        print("item_correct_df.shape: {}".format(item_correct_df.shape))
        print("len(item_score): {}".format(len(item_score)))

    item_correct_df.insert(2, column="full_model", value=item_score)

    n_correct = item_score.count(1)

    gha_acc = np.around(n_correct/n_items, decimals=3)

    if verbose is True:
        print("items: {}\ncorrect: {}\nincorrect: {}\naccuracy: {}".format(n_items, n_correct,
                                                                           n_items - n_correct, gha_acc))

    # # get count_correct_per_class
    # print("predicted_out: {}".format(np.shape(predicted_outputs)))
    # print(predicted_outputs)
    # print("true_cat: {}".format(true_cat))
    # print("predicted_cat: {}".format(predicted_cat))
    # print("item_score: {}".format(item_score))
    # print("item_correct_df")
    # print(item_correct_df.head())



    corr_per_cat_dict = dict()
    for cat in range(n_cats):
        corr_per_cat_dict[cat] = len(item_correct_df[(item_correct_df.loc[:, 'class'] == cat) &
                                                     (item_correct_df.loc[:, 'full_model'] == 1)])
        # print("class {}".format(cat))
        # print(item_correct_df)
        # print(item_correct_df[item_correct_df['class'] == cat])
        # print(item_correct_df[item_correct_df.loc[:, 'class'] == cat])  #) &
                              # (item_correct_df.loc[:, 'full_model'] == 1)].head())


    # # # are any categories missing?
    category_fail = sum(value == 0 for value in corr_per_cat_dict.values())
    category_low = sum(value < 3 for value in corr_per_cat_dict.values())
    n_cats_correct = n_cats - category_fail

    # # report scores
    conf_matrix = confusion_matrix(y_true=true_cat, y_pred=predicted_cat)
    conf_headers = ["pred_{}".format(i) for i in range(n_cats)]
    conf_matrix_df = pd.DataFrame(data=conf_matrix, columns=conf_headers)
    conf_matrix_df.index.names = ['true_label']

    # # names for output files
    item_correct_name = "{}_item_correct.csv".format(output_filename)
    flat_conf_name = "{}_flat_conf_matrix.csv".format(output_filename)

    flat_conf_or_name = flat_conf_name
    if return_flat_conf is True:
        # make a flat conf_matrix to use with lesioning
        flat_conf = np.ravel(conf_matrix)
        flat_conf_labels = ["t{}_p{}".format(i[0], i[1]) for i in list(product(list(range(n_cats)), repeat=2))]
        flat_conf_n_labels = np.column_stack((flat_conf_labels, flat_conf))
        flat_conf_df = pd.DataFrame(data=flat_conf_n_labels, columns=['conf_matrix', 'full_model'])
        flat_conf_or_name = flat_conf_df
        if save_all_csvs is True:
            # flat_conf_df.to_csv(flat_conf_name)
            nick_to_csv(flat_conf_df, flat_conf_name)

    if verbose is True:
        print("\ncategory failures: " + str(category_fail))
        print("category_low: " + str(category_low))
        print("corr_per_cat_dict: {}".format(corr_per_cat_dict))
        print("conf_matrix_df:\n{}".format(conf_matrix_df))

    if save_all_csvs is True:
        # print(item_correct_df.head())

        # # change dtype before saving to make smaller.
        # print("\n\nidiot check")
        # print(item_correct_df.dtypes)
        if n_items < 32000:
            int_item_correct_df = item_correct_df.astype('int16')
        elif n_items > 2147483647:
            int_item_correct_df = item_correct_df  # keeps the current int64
        else:
            int_item_correct_df = item_correct_df.astype('int32')
        # int_item_correct_df.to_csv(item_correct_name, index=False)
        nick_to_csv(int_item_correct_df, item_correct_name)

    scores_dict = {"n_items": n_items, "n_correct": n_correct, "gha_acc": gha_acc,
                   "category_fail": category_fail, "category_low": category_low, "n_cats_correct": n_cats_correct,
                   "corr_per_cat_dict": corr_per_cat_dict,
                   "item_correct_name": item_correct_name,
                   "flat_conf": flat_conf_or_name,
                   "scores_date": tools_date, 'scores_time': tools_time}

    return item_correct_df, scores_dict, incorrect_items




def VGG_get_scores(predicted_outputs, y_df, output_filename, verbose=False, save_all_csvs=True):
    """
    Script will compare predicted class and true class to find whether each item was correct.

    use predicted outputs to get predicted categories

    :param predicted_outputs: from loaded_model.predict(x_data).  shape (n_items, n_cats)
    :param y_df:  y item and class
    :param output_filename:  to use when saving csvs
    :param verbose:
    :param save_all_csvs:

    :return: item_correct_df - item number, class, correct (1 or if incorrect, 0)
    :return: scores_dict - descriptives
    :return: incorrect_items - list of item numbers that were incorrect

    """
    print("\n**** get_scores() ****")

    # todo: repair this - I've changed it for VGG

    n_items, n_cats = np.shape(predicted_outputs)

    predicted_cat = []

    for row in range(n_items):
        this_row = predicted_outputs[row]
        max_val = max(this_row)
        max_cat = this_row.tolist().index(max_val)
        predicted_cat.append(max_cat)

    # # save list of which items were incorrect
    true_cat = [int(i) for i in y_df['class'].tolist()]

    item_score = []
    incorrect_items = []
    for index, pred_item in enumerate(predicted_cat):
        if pred_item == true_cat[index]:
            item_score.append(int(1))

        else:
            item_score.append(int(0))

            incorrect_items.append(index)

    item_correct_df = y_df.copy()

    print("item_correct_df.shape: {}".format(item_correct_df.shape))
    print("len(item_score): {}".format(len(item_score)))

    item_correct_df.insert(2, column="full_model", value=item_score)

    n_correct = item_score.count(1)

    gha_acc = np.around(n_correct / n_items, decimals=3)

    print("items: {}\ncorrect: {}\nincorrect: {}\naccuracy: {}".
          format(n_items, n_correct, n_items - n_correct, gha_acc))

    # # get count_correct_per_class
    corr_per_cat_dict = dict()
    for cat in range(n_cats):
        corr_per_cat_dict[cat] = len(item_correct_df[(item_correct_df['class'] == cat) &
                                                     (item_correct_df['full_model'] == 1)])

    # # # are any categories missing?
    category_fail = sum(value == 0 for value in corr_per_cat_dict.values())
    category_low = sum(value < 3 for value in corr_per_cat_dict.values())
    n_cats_correct = n_cats - category_fail

    # # report scores
    conf_matrix = confusion_matrix(y_true=true_cat, y_pred=predicted_cat)
    print("conf_matrix_shape: {}".format(np.shape(conf_matrix)))
    np.save('{}_conf_matrix.npy'.format(output_filename), conf_matrix)

    # conf_headers = ["pred_{}".format(i) for i in range(n_cats+1)]
    # conf_matrix_df = pd.DataFrame(data=conf_matrix, columns=conf_headers)
    # conf_matrix_df.index.names = ['true_label']
    # conf_matrix_df.to_hdf("{}_conf_matrix.h5".format(output_filename), key='conf_matrix_df')

    if verbose is True:
        print("\ncategory failures: " + str(category_fail))
        print("category_low: " + str(category_low))
        print("corr_per_cat_dict: {}".format(corr_per_cat_dict))
        # print("conf_matrix:\n{}".format(conf_matrix))

    # # names for output files
    item_correct_name = "{}_item_correct.csv".format(output_filename)
    flat_conf_name = "{}_flat_conf_matrix.csv".format(output_filename)

    if save_all_csvs is True:
        # print(item_correct_df.head())

        # make a flat conf_matrix to use with lesioning
        flat_conf = np.ravel(conf_matrix)
        flat_conf_labels = ["t{}_p{}".format(i[0], i[1]) for i in list(product(list(range(n_cats)), repeat=2))]
        flat_conf_df = pd.DataFrame(data=[flat_conf], columns=flat_conf_labels, index=['full_model'])

        flat_conf_df.to_csv(flat_conf_name)
        # nick_to_csv(flat_conf_df, flat_conf_name)


        # # change dtype before saving to make smaller.
        print("\n\nidiot check")
        print(item_correct_df.dtypes)
        # if n_items < 32000:
        #     int_item_correct_df = item_correct_df.astype('int16')
        # elif n_items > 2147483647:
        int_item_correct_df = item_correct_df  # keeps the current int64
        # else:
        #     int_item_correct_df = item_correct_df.astype('int32')
        int_item_correct_df.to_csv(item_correct_name, index=False)
        nick_to_csv(int_item_correct_df, item_correct_name)




        # else:
        #     item_correct_df.to_hdf(item_correct_name + '.h5', key='item_correct_df', index=False)
        # flat_conf_df.to_hdf(flat_conf_name + '.h5', key='flat_conf_df')

    scores_dict = {"n_items": n_items, "n_correct": n_correct, "gha_acc": gha_acc,
                   "category_fail": category_fail, "category_low": category_low, "n_cats_correct": n_cats_correct,
                   "corr_per_cat_dict": corr_per_cat_dict,
                   "item_correct_name": item_correct_name,
                   # "flat_conf_name": flat_conf_name,
                   "scores_date": tools_date, 'scores_time': tools_time}

    return item_correct_df, scores_dict, incorrect_items

##################################################################
# def loop_thru_units(study_dict_name, verbose=False, save_all_csvs=False):
#     """with loaded hid acts and relevant y data
#      - remove any dead units
#      loop through units
#      somehow need to be able to do something with each unit vector
#       - run selectivity (need item correct)
#       - visualise data (need item correct and hid act scores
#       - lesion unit - need weights
#     1. loop through all layers:
#         loop through all units:
#             loop through all classes:
#                 run selectivity measures : CCMA, zn_AP, pr(full), zhou_prec, top_class, informedness
#                 save dicts with values for all measures
#                 save dict with top class/value for all measures
#
#     5. output:
#         mean sel per network page
#         sel per unit page
#         sel dict with GHA dict, network_sel_dict(from network sel csv), top sel values (from per unit csv),
#          all sel values (per class per unit)
#     """
#
#     # # part 1. load dict from study (should run with sim, GHA or sel dict)
#
#     try:
#         study_dict = json.loads(open("{}.txt".format(study_dict_name)).read())
#         list(json_key_to_int(study_dict))
#
#     except FileNotFoundError:
#         try:
#             pickle_load = open(study_dict_name + ".pickle", "rb")
#             study_dict = pickle.load(pickle_load)
#         except FileNotFoundError:
#             print("dictionary not found")
#
#     if verbose is True:
#         print("\n**** {} ****".format(study_dict_name))
#         simple_dict_print(study_dict)
#
#
#     # # part 2. from dict get x, y, num of items, IPC etc
#     # # topic info
#     """for some experiments data info is in sim_dict['data_info']['data_dict']
#     for others it is just in sim_dict['data_info']"""
#     data_info_location = study_dict["data_info"]
#     if 'data_dict' in study_dict['data_info'].keys():
#         data_info_location = study_dict['data_info']['data_dict']
#
#     topic_name = study_dict["topic_info"]["topic_name"]
#     cond = study_dict["topic_info"]["cond"]
#     run = study_dict["topic_info"]["run"]
#     hid_layers = study_dict["model_info"]["hid_layers"]
#     hid_units = study_dict["model_info"]["hid_units"]
#     units_per_layer = study_dict["model_info"]["units_per_layer"]
#     n_cats = data_info_location["n_cats"]
#     model_name = study_dict["model_info"]["model_trained"]
#     n_items = study_dict["GHA_info"]["n_correct"]
#     if 'correct_per_cat' in study_dict['GHA_info'].keys():
#         items_per_cat = study_dict["GHA_info"]["correct_per_cat"]
#     else:
#         items_per_cat = study_dict["GHA_info"]["items_per_cat"]
#
#     if type(items_per_cat) is int:
#         items_per_cat = dict(zip(list(range(n_cats)), [items_per_cat] * n_cats))
#
#
#
#     # # load datasets
#     """If GHA on ALL items, load ALL items, then later remove incorrect responses"""
#     y_filename = study_dict["GHA_info"]["correct_Y_labels"]
#     hid_act_file_names = study_dict["GHA_info"]["hid_act_files"]
#
#     # # GHA on all items or just correct
#     if 'gha_incorrect' in study_dict['GHA_info'].keys():
#         gha_incorrect = study_dict['GHA_info']['gha_incorrect']
#         if gha_incorrect is True:
#             y_filename = study_dict["data_info"]["Y_labels"]
#             n_items = study_dict["data_info"]["n_items"]
#             items_per_cat = study_dict["data_info"]["items_per_cat"]
#             if type(items_per_cat) is int:
#                 items_per_cat = dict(zip(list(range(n_cats)), [items_per_cat] * n_cats))
#
#     # # load y labels
#     if y_filename[-3:] == 'csv':
#         # y_labels_items = np.loadtxt(y_filename, delimiter=',')
# #        item_and_class = pd.read_csv(y_filename, header=None, names=["item", "class"])
#         item_and_class = nick_read_csv(y_filename)

#         print("\nloaded item_and_class: {} {}".format(y_filename, item_and_class.shape))
#     elif y_filename[-3:] == 'npy':
#         y_labels_items = np.load(y_filename)
#         item_and_class = pd.DataFrame({'item': y_labels_items[:, 0], 'class': y_labels_items[:, 1]})
#         print("\nloaded item_and_class: {} {}".format(y_filename, item_and_class.shape))
#     else:
#         print("unknown y_filename file type: {}".format(y_filename[-3:]))
#
#     # print("\nloaded y_labels_items: {} {}".format(y_filename, np.shape(y_labels_items)))
#
#     # Output files
#     output_filename = study_dict["topic_info"]["output_filename"]
#     print("Output file: " + output_filename)
#
#
#
#     sel_per_unit = dict()
#
#     for layer, hidden_activation_file in enumerate(hid_act_file_names):
#
#         sel_per_unit[layer] = dict()
#
#         if hidden_activation_file[-3:] == 'npy':
#             hid_act = np.load(hidden_activation_file)
#             # # add stuff here to put it into a labelled pandad dataframe
#         elif hidden_activation_file[-3:] == 'csv':
#             # hid_act = np.genfromtxt(hidden_activation_file, delimiter=',')
# #             hid_act = pd.read_csv(hidden_activation_file)
#               hid_act = nick_read_csv(hidden_activation_file)

#         elif hidden_activation_file[-3:] == 'act':
#             try:
#                 hid_act = np.load("{}.npy".format(hidden_activation_file))
#                 # # add stuff here to put it into a labelled pandad dataframe
#             except FileNotFoundError:
#                 try:
#                     # hid_act = np.genfromtxt("{}.csv".format(hidden_activation_file), delimiter=',')
# #                     hid_act = pd.read_csv("{}.csv".format(hidden_activation_file))
#                     hid_act = nick_read_csv("{}.csv".format(hidden_activation_file))

#                 except FileNotFoundError:
#                     print("hidden_activation_file not found")
#                     break
#
#         print("loaded hidden_activation_file: {}, {}".format(hidden_activation_file, np.shape(hid_act)))
#
#         for unit in range(units_per_layer):
#             print("\nrunning layer {} unit {}".format(layer, unit))
#             this_unit_just_acts = hid_act.iloc[:, unit]
#             # 1st check for dead units
#             if this_unit_just_acts.sum() == 0:  # check for dead units, if dead, all details to 0
#                 print("dead unit found")
#                 # # all descriptives of this unit to 0 (or whatever is appropriate)
#                 # # make per unit series
#
#                 # todo: check to see if I am using boolean statemens in a list (e.g., dead unit, item correct)
#                 #  change any boolean statements (item correct: True, False) to binary (item correct: 1, 0).
#                 #  This should make checking, summing etc, easier
#
#                 sel_per_unit[layer][unit] = {"layer": layer, "unit": unit,
#                                              'dead_unit': True,
#                                              "AP_class": -999,
#                                              "AP": 0,
#                                              "PR_AUC": 0,
#                                              "ROC_AUC": 0,
#                                              "CCMA": 0,
#                                              "Zhou_prec": 0,
#                                              "Zhou_prec_thr": 0,
#                                              "informed": 0,
#                                              "informed_thr": 0,
#                                              "TCS_class": -999,
#                                              "TCS_items": 0,
#                                              "TCS_thr": 0,
#                                              "TCS_recall": 0,
#                                              "noZero_ave_prec": 0,
#                                              "noZero_pr_auc": 0,
#                                              "noZero_ROC_AUC": 0,
#                                              }
#             else:  # if not dead, do selectivity analysis
#                 print("not a dead unit, running selectivity")
#
#                 # insert act values in middle of labels (item, act, cat)
#                 this_unit_acts = item_and_class.copy()
#                 this_unit_acts.insert(1, column='activation', value=this_unit_just_acts)
#                 # print(this_unit_acts)
#
#                 # # sort by descending hid act values
#                 # # for all zero activations, cycle through classes
#
#                 this_unit_acts_df = sort_cycle_duplicates_df(df_to_sort=this_unit_acts, duplicated_value=0,
#                                                              sort_1st='activation',
#                                                              sort_2nd='class')
#
#                 # # # normalize activations
#                 just_act_values = list(this_unit_acts_df.iloc[:, 1])
#                 max_act = max(just_act_values)
#                 normed_acts = [i/max_act for i in just_act_values]
#                 # normed_acts = just_act_values / max_act
#
#                 # # insert normalized activations in here
#                 this_unit_acts_df.insert(2, column='normed', value=normed_acts)
#
#                 # print("this_unit_acts_df: {}\n{}".format(np.shape(this_unit_acts_df), this_unit_acts_df.head()))
#
#                 # # convert to pandas df
#                 this_unit_acts_df = this_unit_acts_df.astype({"item": int, "activation": float,
#                                                               "normed": float, "class": int})
#
#                 if verbose is True:
#                     print("this_unit_acts_df: {}\n{}".format(np.shape(this_unit_acts_df), this_unit_acts_df.head()))
#
#
