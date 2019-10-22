import datetime
import sys

from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from nick_data_tools import load_x_data, load_y_data, nick_to_csv, nick_read_csv


tools_date = int(datetime.datetime.now().strftime("%y%m%d"))
tools_time = int(datetime.datetime.now().strftime("%H%M"))


############################
def get_model_dict(compiled_model, verbose=False):
    """
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

