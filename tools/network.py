import datetime
import os
import pickle
import shelve
import h5py
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tools.data import load_y_data, nick_to_csv, nick_read_csv
from tools.dicts import load_dict, focussed_dict_print


tools_date = int(datetime.datetime.now().strftime("%y%m%d"))
tools_time = int(datetime.datetime.now().strftime("%H%M"))


############################
def get_model_dict(compiled_model, verbose=False):
    """
    Takes a compiled model (model + data input & output shape) and returns a dict containing:
        1. the full get_config output
        2. a summary to use for analysis (just n_filters, units etc)

    :param compiled_model:
    :param verbose: How much to print to screen

    :return: dict: {'config': model_config, 'summary': hid_layers_summary}
    """

    model_config = compiled_model.get_config()
    hid_layers_summary = dict()

    # # get useful info
    all_layers = len(model_config['layers'])
    hid_act_layers = 0
    hid_dense_layers = 0
    hid_conv_layers = 0
    hid_rec_layer = 0
    hid_UPL = []
    hid_FPL = []


    # # check number of output layers
    # print("checking output layers")
    # print(model_config['layers'][all_layers-1])
    if model_config['layers'][all_layers-1]['class_name'] == 'Activation':
        output_layers = 2
    elif model_config['layers'][all_layers-1]['class_name'] == 'Dense':
        output_layers = 1
    else:
        raise TypeError("Unknown output layer")

    hid_layers = all_layers-output_layers


    for layer in range(0, all_layers-output_layers):
        if verbose:
            print("\nModel layer {}\n{}".format(layer, model_config['layers'][layer]))

        # # get useful info
        layer_class = model_config['layers'][layer]['class_name']
        layer_dict = {'layer': layer,
                      'name': model_config['layers'][layer]['config']['name'],
                      'class': layer_class}

        if 'units' in model_config['layers'][layer]['config']:
            layer_dict['units'] = model_config['layers'][layer]['config']['units']
            hid_UPL.append(model_config['layers'][layer]['config']['units'])

            if layer_class == 'Dense':
                hid_dense_layers += 1
            elif layer_class in ["SimpleRNN", "GRU", "LSTM", "TimeDistributed"]:
                hid_rec_layer += 1

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
    for layer in range(all_layers-output_layers, all_layers):
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

def get_scores(predicted_outputs, y_df, output_filename,
               y_1hot=True, output_act='softmax',
               verbose=False, save_all_csvs=True,
               return_flat_conf=False):
    """
    Script will compare predicted class and true class to find whether each item was correct.

    use predicted outputs to get predicted categories

    :param predicted_outputs: from loaded_model.predict(x_data).  shape (n_items, n_cats)
    :param y_df:  y item and class
    :param output_filename:  to use when saving csvs
    :param y_1hot: (True) get most active class label.  If False, get label for all classes > .5, compare with output.
    :param verbose:
    :param save_all_csvs:
    :param return_flat_conf: if false, just add the name to the dict, if true, return actual flat conf matrix


    :return: item_correct_df - item number, class, correct (1 or if incorrect, 0)
    :return: scores_dict - descriptives
    :return: incorrect_items - list of item numbers that were incorrect

    """
    print("\n**** get_scores() ****")

    n_items, n_cats = np.shape(predicted_outputs)

    print(f'y_1hot: {y_1hot}')
    print(f'predicted_outputs:\n{predicted_outputs}')


    if y_1hot:
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

        # make a list of whether the predictions were correct
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

    else:
        # if not y_1hot
        # # get labels for classes where value is greater than .5
        all_pred_labels = []
        if output_act == 'sigmoid':
            for item in range(n_items):
                these_pred_labels = np.argwhere(predicted_outputs[item] > .5)

                # flatted predictions
                if len(np.shape(these_pred_labels)) > 1:
                    these_pred_labels = np.ravel(these_pred_labels).tolist()
                all_pred_labels.append(these_pred_labels)

        elif output_act in ['relu', 'linear']:
            print('write something to select the most active n values')



        if verbose:
            print(f"all_pred_labels: {np.shape(all_pred_labels)}\n{all_pred_labels[0]}\n")
            print(f"y_df: {y_df}")
        #
        # predicted_cat = []
        #
        # # find the column (class) with the highest prediction
        # for row in range(n_items):
        #     this_row = predicted_outputs[row]
        #     max_val = max(this_row)
        #     # max_cat = this_row.tolist().index(max_val)
        #     max_cat = list(this_row).index(max_val)
        #
        #     predicted_cat.append(max_cat)

        # # save list of which items were incorrect
        true_cat = [int(i) for i in y_df['class'].tolist()]
        # true_cat = y_df['class'].tolist()  # was returning strings?

        # make a list of whether the predictions were correct
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

        gha_acc = np.around(n_correct / n_items, decimals=3)

        if verbose is True:
            print("items: {}\ncorrect: {}\nincorrect: {}\naccuracy: {}".format(n_items, n_correct,
                                                                               n_items - n_correct, gha_acc))


    corr_per_cat_dict = dict()
    for cat in range(n_cats):
        corr_per_cat_dict[cat] = len(item_correct_df[(item_correct_df.loc[:, 'class'] == cat) &
                                                     (item_correct_df.loc[:, 'full_model'] == 1)])


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


# def get_scores(predicted_outputs, y_df, output_filename,
#                y_1hot=True,
#                verbose=False, save_all_csvs=True,
#                return_flat_conf=False):
#     """
#     Script will compare predicted class and true class to find whether each item was correct.
#
#     use predicted outputs to get predicted categories
#
#     :param predicted_outputs: from loaded_model.predict(x_data).  shape (n_items, n_cats)
#     :param y_df:  y item and class
#     :param output_filename:  to use when saving csvs
#     :param y_1hot: (True)
#     :param verbose:
#     :param save_all_csvs:
#     :param return_flat_conf: if false, just add the name to the dict, if true, return actual flat conf matrix
#
#
#     :return: item_correct_df - item number, class, correct (1 or if incorrect, 0)
#     :return: scores_dict - descriptives
#     :return: incorrect_items - list of item numbers that were incorrect
#
#     """
#     print("\n**** get_scores() ****")
#
#     n_items, n_cats = np.shape(predicted_outputs)
#
#     predicted_cat = []
#
#     # find the column (class) with the highest prediction
#     for row in range(n_items):
#         this_row = predicted_outputs[row]
#         max_val = max(this_row)
#         # max_cat = this_row.tolist().index(max_val)
#         max_cat = list(this_row).index(max_val)
#
#         predicted_cat.append(max_cat)
#
#     # # save list of which items were incorrect
#     true_cat = [int(i) for i in y_df['class'].tolist()]
#     # true_cat = y_df['class'].tolist()  # was returning strings?
#
#     # make a list of whether the predictions were correct
#     item_score = []
#     incorrect_items = []
#     for index, pred_item in enumerate(predicted_cat):
#         if pred_item == true_cat[index]:
#             item_score.append(int(1))
#         else:
#             item_score.append(int(0))
#             incorrect_items.append(index)
#
#     item_correct_df = y_df.copy()
#
#     if verbose is True:
#         print("item_correct_df.shape: {}".format(item_correct_df.shape))
#         print("len(item_score): {}".format(len(item_score)))
#
#     item_correct_df.insert(2, column="full_model", value=item_score)
#
#     n_correct = item_score.count(1)
#
#     gha_acc = np.around(n_correct/n_items, decimals=3)
#
#     if verbose is True:
#         print("items: {}\ncorrect: {}\nincorrect: {}\naccuracy: {}".format(n_items, n_correct,
#                                                                            n_items - n_correct, gha_acc))
#
#
#
#     corr_per_cat_dict = dict()
#     for cat in range(n_cats):
#         corr_per_cat_dict[cat] = len(item_correct_df[(item_correct_df.loc[:, 'class'] == cat) &
#                                                      (item_correct_df.loc[:, 'full_model'] == 1)])
#
#
#     # # # are any categories missing?
#     category_fail = sum(value == 0 for value in corr_per_cat_dict.values())
#     category_low = sum(value < 3 for value in corr_per_cat_dict.values())
#     n_cats_correct = n_cats - category_fail
#
#     # # report scores
#     conf_matrix = confusion_matrix(y_true=true_cat, y_pred=predicted_cat)
#     conf_headers = ["pred_{}".format(i) for i in range(n_cats)]
#     conf_matrix_df = pd.DataFrame(data=conf_matrix, columns=conf_headers)
#     conf_matrix_df.index.names = ['true_label']
#
#     # # names for output files
#     item_correct_name = "{}_item_correct.csv".format(output_filename)
#     flat_conf_name = "{}_flat_conf_matrix.csv".format(output_filename)
#
#     flat_conf_or_name = flat_conf_name
#     if return_flat_conf is True:
#         # make a flat conf_matrix to use with lesioning
#         flat_conf = np.ravel(conf_matrix)
#         flat_conf_labels = ["t{}_p{}".format(i[0], i[1]) for i in list(product(list(range(n_cats)), repeat=2))]
#         flat_conf_n_labels = np.column_stack((flat_conf_labels, flat_conf))
#         flat_conf_df = pd.DataFrame(data=flat_conf_n_labels, columns=['conf_matrix', 'full_model'])
#         flat_conf_or_name = flat_conf_df
#         if save_all_csvs is True:
#             # flat_conf_df.to_csv(flat_conf_name)
#             nick_to_csv(flat_conf_df, flat_conf_name)
#
#     if verbose is True:
#         print("\ncategory failures: " + str(category_fail))
#         print("category_low: " + str(category_low))
#         print("corr_per_cat_dict: {}".format(corr_per_cat_dict))
#         print("conf_matrix_df:\n{}".format(conf_matrix_df))
#
#     if save_all_csvs is True:
#         # print(item_correct_df.head())
#
#         # # change dtype before saving to make smaller.
#         # print("\n\nidiot check")
#         # print(item_correct_df.dtypes)
#         if n_items < 32000:
#             int_item_correct_df = item_correct_df.astype('int16')
#         elif n_items > 2147483647:
#             int_item_correct_df = item_correct_df  # keeps the current int64
#         else:
#             int_item_correct_df = item_correct_df.astype('int32')
#         # int_item_correct_df.to_csv(item_correct_name, index=False)
#         nick_to_csv(int_item_correct_df, item_correct_name)
#
#     scores_dict = {"n_items": n_items, "n_correct": n_correct, "gha_acc": gha_acc,
#                    "category_fail": category_fail, "category_low": category_low, "n_cats_correct": n_cats_correct,
#                    "corr_per_cat_dict": corr_per_cat_dict,
#                    "item_correct_name": item_correct_name,
#                    "flat_conf": flat_conf_or_name,
#                    "scores_date": tools_date, 'scores_time': tools_time}
#
#     return item_correct_df, scores_dict, incorrect_items




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


    if verbose is True:
        print("\ncategory failures: " + str(category_fail))
        print("category_low: " + str(category_low))
        print("corr_per_cat_dict: {}".format(corr_per_cat_dict))

    # # names for output files
    item_correct_name = "{}_item_correct.csv".format(output_filename)
    flat_conf_name = "{}_flat_conf_matrix.csv".format(output_filename)

    if save_all_csvs is True:
        # make a flat conf_matrix to use with lesioning
        if n_cats > 100:
            print("too many classes to make a flat conf matrix")
        else:
            flat_conf = np.ravel(conf_matrix)
            flat_conf_labels = ["t{}_p{}".format(i[0], i[1])
                                for i in list(product(list(range(n_cats)), repeat=2))]
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
        # nick_to_csv(int_item_correct_df, item_correct_name)


    scores_dict = {"n_items": n_items, "n_correct": n_correct, "gha_acc": gha_acc,
                   "category_fail": category_fail, "category_low": category_low, "n_cats_correct": n_cats_correct,
                   "corr_per_cat_dict": corr_per_cat_dict,
                   "item_correct_name": item_correct_name,
                   # "flat_conf_name": flat_conf_name,
                   "scores_date": tools_date, 'scores_time': tools_time}

    return item_correct_df, scores_dict, incorrect_items

##################################################################

def loop_thru_acts(gha_dict_path,
                   correct_items_only=True,
                   acts_saved_as='pickle',
                   letter_sel=False,
                   already_completed={},
                   verbose=False, test_run=False):
    """To use hidden unit activations for sel, (lesioning?) visualisation.
        1. load dict from study (GHA dict) - get variables from dict
    2. load y, sort out incorrect resonses
    3. find where to load gha from: pickle, hdf5, shelve.
        for now write assuming pickle
        load hidden activations
    4. loop through all layers:
        loop through all units:
            (loop through timesteps?)

    This is a generator-iterator, not a function, as I want it to yeild hid_acts
    and details one unit at a time.

    :param gha_dict_path: path of the gha dict
    :param correct_items_only: Whether to skip test items that that model got incorrect.
    :param letter_sel: if False, test sel for words (class-labels).
            If True, test for letters (parts) using 'local_word_X' for each word when looping through classes
    :param already_completed: None, or dict with layer_names as keys,
                            values are ether 'all' or number of last completed unit.
    :param acts_saved_as: file format used to save gha

    :param verbose: how much to print to screen
    :param test_run: if True, only do subset, e.g., 3 units from 3 layers

    :return: hid_acts_array (activation values for all items at this unit (all timesteps?))
    :return: data_details (item_numbers, class_labels, whether_correct)
    :return: unit-details (layer_name, layer_class, act_func, unit_number)
    """

    if verbose:
        print("\n**** running loop_thru_units() ****")

    # # check already completed dict
    if already_completed is not None:
        if type(already_completed) is not dict:
            TypeError("already-completed should be a dict")
        else:
            for value in already_completed.values():
                if value is not 'all':
                    if type(value) is not int:
                        ValueError("already-completed dict values should be int of last completed unit or 'all'")

    # # part 1. load dict from study (should run with sim, GHA or sel dict)
    gha_dict = load_dict(gha_dict_path)
    focussed_dict_print(gha_dict, 'gha_dict')

    # # use gha-dict_path to get exp_cond_gha_path, gha_dict_name,
    exp_cond_gha_path = gha_dict['GHA_info']['gha_path']
    # gha_dict_name = gha_dict['GHA_info']['hid_act_files']['2d']
    os.chdir(exp_cond_gha_path)
    current_wd = os.getcwd()

    # get topic_info from dict
    output_filename = gha_dict["topic_info"]["output_filename"]
    if letter_sel:
        output_filename = f"{output_filename}_lett"

    if verbose:
        print(f"\ncurrent_wd: {current_wd}")
        print(f"output_filename: {output_filename}")


    # # get model info from dict
    units_per_layer = gha_dict['model_info']["overview"]["units_per_layer"]
    n_layers = gha_dict['model_info']['overview']['hid_layers']
    model_dict = gha_dict['model_info']['config']
    if verbose:
        focussed_dict_print(model_dict, 'model_dict')



    # # check for sequences/rnn
    sequence_data = False
    y_1hot = True

    if 'timesteps' in gha_dict['model_info']['overview']:
        sequence_data = True
        timesteps = gha_dict['model_info']["overview"]["timesteps"]
        serial_recall = gha_dict['model_info']["overview"]["serial_recall"]
        y_1hot = serial_recall

    # # I can't do class correlations for letters, (as it is the equivillent of
    # having a dist output for letters
    if letter_sel:
        y_1hot = False

    # # get gha info from dict
    hid_acts_filename = gha_dict["GHA_info"]["hid_act_files"]['2d']


    '''Part 2 - load y, sort out incorrect resonses'''
    print("\n\nPart 2: loading labels")
    # # load y_labels to go with hid_acts and item_correct for sequences
    if 'seq_corr_list' in gha_dict['GHA_info']['scores_dict']:
        n_seqs = gha_dict['GHA_info']['scores_dict']['n_seqs']
        n_seq_corr = gha_dict['GHA_info']['scores_dict']['n_seq_corr']
        n_incorrect = n_seqs - n_seq_corr

        test_label_seq_name = gha_dict['GHA_info']['y_data_path']
        seqs_corr = gha_dict['GHA_info']['scores_dict']['seq_corr_list']

        test_label_seqs = np.load(f"{test_label_seq_name}labels.npy")

        if verbose:
            print(f"test_label_seqs: {np.shape(test_label_seqs)}")
            print(f"seqs_corr: {np.shape(seqs_corr)}")
            print(f"n_seq_corr: {n_seq_corr}")

        """get 1hot item vectors for 'words' and 3 hot for letters"""
        # '''Always use serial_recall True. as I want a separate 1hot vector for each item.
        # Always use x_data_type 'local_letter_X' as I want 3hot vectors'''
        # y_letters = []
        # y_words = []
        # for this_seq in test_label_seqs:
        #     get_letters, get_words = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
        #                                                        seq_line=this_seq,
        #                                                        serial_recall=True,
        #                                                        end_seq_cue=False,
        #                                                        x_data_type='local_letter_X')
        #     y_letters.append(get_letters)
        #     y_words.append(get_words)
        #
        # y_letters = np.array(y_letters)
        # y_words = np.array(y_words)
        # if verbose:
        #     print(f"\ny_letters: {type(y_letters)}  {np.shape(y_letters)}")
        #     print(f"\ny_words: {type(y_words)}  {np.shape(y_words)}")
        #     print(f"\ntest_label_seqs[0]: {test_label_seqs[0]}")
        #     if test_run:
        #         print(f"y_letters[0]:\n{y_letters[0]}")
        #         print(f"y_words[0]:\n{y_words[0]}")

        y_df_headers = [f"ts{i}" for i in range(timesteps)]
        y_scores_df = pd.DataFrame(data=test_label_seqs, columns=y_df_headers)
        y_scores_df['full_model'] = seqs_corr
        if verbose:
            print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")



    # # if not sequence data, load y_labels to go with hid_acts and item_correct for items
    elif 'item_correct_name' in gha_dict['GHA_info']['scores_dict']:
        # # load item_correct (y_data)
        item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
        # y_df = pd.read_csv(item_correct_name)
        y_scores_df = nick_read_csv(item_correct_name)



    """# # sort incorrect item data"""
    print("\n\nRemoving incorrect responses")
    # # # get values for correct/incorrect items (1/0 or True/False)
    item_correct_list = y_scores_df['full_model'].tolist()
    full_model_values = list(set(item_correct_list))

    correct_symbol = 1
    if len(full_model_values) != 2:
        TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")
    if 1 not in full_model_values:
        if True in full_model_values:
            correct_symbol = True
        else:
            TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")

    # # i need to check whether this analysis should include incorrect items
    gha_incorrect = gha_dict['GHA_info']['gha_incorrect']

    # # get item indeces for correct and incorrect items
    item_index = list(range(n_seq_corr))

    incorrect_items = []
    correct_items = []
    for index in range(len(item_correct_list)):
        if item_correct_list[index] == 0:
            incorrect_items.append(index)
        else:
            correct_items.append(index)
    if correct_items_only:
        item_index == correct_items

    else:
        '''tbh I'm not quite sure what the implications of this are, 
        just a hack to make it work for untrained model'''
        print("\n\nWARNING\nitem_index == what_shape\njust doing this for untrained model!")

        what_shape = list(range(960))
        print(f'what_shape: {np.shape(what_shape)}\n{what_shape}\n')
        item_index = what_shape
        print(f'item_index: {np.shape(item_index)}\n{item_index}\n')

    print(f"incorrect_items: {np.shape(incorrect_items)}\n{incorrect_items}")
    print(f'item_index: {np.shape(item_index)}\n{item_index}\n')
    print(f'correct_items_only: {correct_items_only}')

    if gha_incorrect:
        if correct_items_only:
            if verbose:
                print("\ngha_incorrect: True (I have incorrect responses)\n"
                      "correct_items_only: True (I only want correct responses)")
                print(f"remove {n_incorrect} incorrect from hid_acts & output using y_scores_df.")
                print("use y_correct for y_df")

            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_symbol]
            y_df = y_correct_df

            mask = np.ones(shape=len(seqs_corr), dtype=bool)
            mask[incorrect_items] = False
            test_label_seqs = test_label_seqs[mask]

        else:
            if verbose:
                print("\ngha_incorrect: True (I have incorrect responses)\n"
                      "correct_items_only: False (I want incorrect responses)")
                print("no changes needed - don't remove anything from hid_acts, output and "
                      "use y scores as y_df")

                y_df = y_scores_df
    else:
        if correct_items_only:
            if verbose:
                print("\ngha_incorrect: False (I only have correct responses)\n"
                      "correct_items_only: True (I only want correct responses)")
                print("no changes needed - don't remove anything from hid_acts or output.  "
                      "Use y_correct as y_df")
            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_symbol]
            y_df = y_correct_df
        else:
            if verbose:
                print("\ngha_incorrect: False (I only have correct responses)\n"
                      "correct_items_only: False (I want incorrect responses)")
                raise TypeError("I can not complete this as desried"
                                "change correct_items_only to True"
                                "for analysis  - don't remove anything from hid_acts, output and "
                                "use y scores as y_df")


    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")
        print(f"\ntest_label_seqs: {np.shape(test_label_seqs)}")  # \n{test_label_seqs}")



    # # Part 3 - where to load hid_acts from
    if acts_saved_as is 'pickle':
        with open(hid_acts_filename, 'rb') as pkl:
            hid_acts_dict = pickle.load(pkl)

        hid_acts_keys_list = list(hid_acts_dict.keys())

        if verbose:
            print(f"\n**** pickle opening {hid_acts_filename} ****")
            print(f"hid_acts_keys_list: {hid_acts_keys_list}")
            print(f"first layer keys: {list(hid_acts_dict[0].keys())}")
            # focussed_dict_print(hid_acts_dict, 'hid_acts_dict')

        last_layer_num = hid_acts_keys_list[-1]
        last_layer_name = hid_acts_dict[last_layer_num]['layer_name']
        
    elif acts_saved_as is 'h5':
        with h5py.File(hid_acts_filename, 'r') as hid_acts_dict:
            hid_acts_keys_list = list(hid_acts_dict.keys())

            if verbose:
                print(f"\n**** h5py opening {hid_acts_filename} ****")
                print(f"hid_acts_keys_list: {hid_acts_keys_list}")

            last_layer_num = hid_acts_keys_list[-1]
            last_layer_name = hid_acts_dict[last_layer_num]['layer_name']



    '''part 4 loop through layers and units'''
    # # loop through dict/layers
    if verbose:
        print("\n*** looping through layers ***")
    if test_run:
        layer_counter = 0

    # # loop through layer numbers in list of dict keys
    for layer_number in hid_acts_keys_list:

        # # don't run sel on output layer
        if layer_number == last_layer_num:
            if verbose:
                print(f"\nskip output layer! (layer: {layer_number})")
            continue

        if test_run is True:
            layer_counter = layer_counter + 1
            if layer_counter > 3:
                if verbose:
                    print(f"\tskip this layer!: test_run, only running subset of layers")
                continue


        '''could add something to check which layers/units have been done 
        already and start from there?'''

        # # Once I've decided to run this unit
        if acts_saved_as is 'pickle':
            with open(hid_acts_filename, 'rb') as pkl:
                hid_acts_dict = pickle.load(pkl)
                layer_dict = hid_acts_dict[layer_number]
                
        elif acts_saved_as is 'h5':
            with h5py.File(hid_acts_filename, 'r') as hid_acts_dict:
                layer_dict = hid_acts_dict[layer_number]


        layer_name = layer_dict['layer_name']

        partially_completed_layer = False
        if layer_name in already_completed:
            if already_completed[layer_name] is 'all':
                print("already completed analysis on this layer")
                continue
            else:
                partially_completed_layer = True

        if verbose:
            print(f"\nrunning layer {layer_number}: {layer_name}")
        hid_acts_array = layer_dict['hid_acts']

        if verbose:
            if sequence_data:
                print(f"np.shape(hid_acts_array) (n_seqs, timesteps, units_per_layer): "
                      f"{np.shape(hid_acts_array)}")
            else:
                print(f"np.shape(hid_acts_array) (n_items, units_per_layer): "
                      f"{np.shape(hid_acts_array)}")




        # # remove incorrect responses from np array
        if correct_items_only:
            if gha_incorrect:
                if verbose:
                    print(f"\nremoving {n_incorrect} incorrect responses from "
                          f"hid_acts_array: {np.shape(hid_acts_array)}")

                hid_acts_array = hid_acts_array[mask]
                if verbose:
                    print(f"(cleaned) np.shape(hid_acts_array) (n_seqs_corr, timesteps, units_per_layer): "
                          f"{np.shape(hid_acts_array)}"
                          f"\ntest_label_seqs: {np.shape(test_label_seqs)}")


        act_func = gha_dict['model_info']['layers']['hid_layers'][layer_number]['act_func']

        
        
        '''loop through units'''
        if verbose:
            print("\n**** loop through units ****")

        unit_counter = 0
        # for unit_index, unit in enumerate(hid_acts_df.columns):
        for unit_index in range(units_per_layer):

            if partially_completed_layer:
                if unit_index <= already_completed[layer_name]:
                    print("already run this unit")
                    continue

            if test_run is True:
                unit_counter += 1
                if unit_counter > 3:
                    continue

            print(f"\n****\nrunning layer {layer_number} of {n_layers} "
                  f"({layer_name}): unit {unit_index} of {units_per_layer}\n****")
            
            if sequence_data:
                
                one_unit_all_timesteps = hid_acts_array[:, :, unit_index]

                if np.sum(one_unit_all_timesteps) == 0:
                    dead_unit = True
                    if verbose:
                        print("dead unit")
                else:
                    if verbose:
                        print(f"\nnp.shape(one_unit_all_timesteps) (seqs, timesteps): "
                              f"{np.shape(one_unit_all_timesteps)}")
                    
                    # get hid acts for each timestep
                    for timestep in range(timesteps):
                        print("\n\tunit {} timestep {} (of {})".format(unit_index, timestep, timesteps))

                        one_unit_one_timestep = one_unit_all_timesteps[:, timestep]

                        # y_labels_one_timestep_float = combo_data[:, timestep]
                        y_labels_one_timestep_float = test_label_seqs[:, timestep]
                        y_labels_one_timestep = [int(q) for q in y_labels_one_timestep_float]


                        these_acts = one_unit_one_timestep
                        these_labels = y_labels_one_timestep

                        if verbose:
                            print(f'item_index: {np.shape(item_index)}')
                            print(f'these_acts: {np.shape(these_acts)}')
                            print(f'these_labels: {np.shape(these_labels)}')

                        # insert act values in middle of labels (item, act, cat)
                        item_act_label_array = np.vstack((item_index, these_acts, these_labels)).T

                        if verbose:
                            print(f"\t - one_unit_one_timestep shape: (n seqs) {np.shape(one_unit_one_timestep)}")
                            print(f"\t - y_labels_one_timestep shape: {np.shape(y_labels_one_timestep)}")

                            print(f"\t - item_act_label_array shape: (item_idx, hid_acts, y_label) "
                                  f"{np.shape(item_act_label_array)}")
                        # print(f"\titem_act_label_array: \n\t{item_act_label_array}")

                        loop_dict = {
                                     "sequence_data": sequence_data,
                                     "y_1hot": y_1hot,
                                     "act_func": act_func,
                                     "layer_number": layer_number, "layer_name": layer_name,
                                     "unit_index": unit_index,
                                     "timestep": timestep,
                                     'item_act_label_array': item_act_label_array,
                                     }

                        yield loop_dict

                    # return hid act, data info, unit info and timestep
            else:
                # if not sequences, just items
                this_unit_just_acts = hid_acts_array[:, unit_index]
                
                if np.sum(this_unit_just_acts) == 0:
                    dead_unit = True
                    if verbose:
                        print("dead unit")
                else:
                    if verbose:
                        print(f"\nnp.shape(this_unit_just_acts) (items, ): "
                              f"{np.shape(this_unit_just_acts)}")

                    these_acts = this_unit_just_acts
                    these_labels = item_correct_list

                    # insert act values in middle of labels (item, act, cat)
                    item_act_label_array = np.vstack((item_index, these_acts, these_labels)).T

                    if verbose:
                        print(" - item_act_label_array shape: {}".format(np.shape(item_act_label_array)))
                        print(f"item_act_label_array: (item_idx, hid_acts, y_label)\n{item_act_label_array}")

                    timestep = None

                    loop_dict = {
                        "sequence_data": sequence_data,
                        "y_1hot": y_1hot,
                        "act_func": act_func,
                        "layer_number": layer_number, "layer_name": layer_name,
                        "unit_index": unit_index,
                        "timestep": timestep,
                        'item_act_label_array': item_act_label_array,
                    }

                    yield loop_dict
