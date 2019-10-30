import copy
import csv
import datetime
import os
import pickle
import shelve

import h5py
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, rankdata
from sklearn.metrics import roc_curve, auc

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.RNN_STM import get_X_and_Y_data_from_seq, seq_items_per_class
from tools.data import load_y_data, nick_to_csv, nick_read_csv
from tools.network import loop_thru_acts


'''This script uses shelve instead of pickle for sel_p_unit dict.

ROC error (present in versions upto 04102019) has been fixed (results for class
zero are now being processed).


'''


def nick_roc_stuff(class_list, hid_acts, this_class, class_a_size, not_a_size,
                   act_func='relu',
                   verbose=False):
    """
    compute fpr, tpr, thr

    :param class_list: list of class labels
    :param hid_acts: list of normalized (0:1) hid_act values
    :param this_class: category of interest (which class is signal)
    :param class_a_size: number of items from this class
    :param not_a_size: number of items not in this class
    :param act_func: relu, sigmoid or tanh.
        If sigmoid: y_true = 0-1, use raw acts
        If Relu: y_true = 0-1, use normed acts?
        If tanh: y_true = -1-1, use raw acts


    :param verbose: how much to print to screen

    :return: roc_dict: fpr, tpr, thr, ROC_AUC
    """

    print("**** nick_roc_stuff() ****")

    # if class is not empty (no correct items)
    if class_a_size > 0:
        # convert class list to binary one vs all
        if act_func is 'tanh':
            binary_array = [1 if i == this_class else -1 for i in np.array(class_list)]
        else:
            binary_array = [1 if i == this_class else 0 for i in np.array(class_list)]
        hid_act_array = np.array(hid_acts)
        n_items = sum([class_a_size, not_a_size])

        # # get ROC curve
        fpr, tpr, thr = roc_curve(binary_array, hid_act_array)

        # # Use ROC dict stuff to compute all other needed vectors
        tp_count_dict = [class_a_size * i for i in tpr]
        fp_count_dict = [not_a_size * i for i in fpr]
        abv_thr_count_dict = [x + y for x, y in zip(tp_count_dict, fp_count_dict)]
        # prop_above_thr_dict = [i / n_items for i in abv_thr_count_dict]
        precision_dict = [x / y if y else 0 for x, y in zip(tp_count_dict, abv_thr_count_dict)]
        recall_dict = [i / class_a_size for i in tp_count_dict]
        recall2_dict = recall_dict[:-1]
        recall2_dict.insert(0, 0)
        recall_increase_dict = [x - y for x, y in zip(recall_dict, recall2_dict)]
        my_ave_prec_vals_dict = [x * y for x, y in zip(precision_dict, recall_increase_dict)]

        # # once we have all vectors, do necessary stats for whole range of activations
        roc_auc = auc(fpr, tpr)
        ave_prec = np.sum(my_ave_prec_vals_dict)
        pr_auc = auc(recall_dict, precision_dict)

        # # Informedness
        get_informed_dict = [a + (1 - b) - 1 for a, b in zip(tpr, fpr)]
        max_informed = max(get_informed_dict)
        max_informed_count = get_informed_dict.index(max_informed)
        max_informed_thr = thr[max_informed_count]
        if max_informed <= 0:
            max_informed = max_informed_thr = 0
        max_info_sens = tpr[max_informed_count]
        max_info_spec = 1 - fpr[max_informed_count]
        max_informed_prec = precision_dict[max_informed_count]


    else:  # if there are not items in this class
        roc_auc = ave_prec = pr_auc = 0
        max_informed = max_informed_count = max_informed_thr = 0
        max_info_sens = max_info_spec = max_informed_prec = 0

    roc_sel_dict = {'roc_auc': roc_auc,
                    'ave_prec': ave_prec,
                    'pr_auc': pr_auc,
                    'max_informed': max_informed,
                    'max_info_count': max_informed_count,
                    'max_info_thr': max_informed_thr,
                    'max_info_sens': max_info_sens,
                    'max_info_spec': max_info_spec,
                    'max_info_prec': max_informed_prec,
                    }

    return roc_sel_dict


def class_correlation(this_unit_acts, output_acts, verbose=False):
    """
    from: Revisiting the Importance of Individual Units in CNNs via Ablation
    "we can use the correlation between the activation of unit i
    and the predicted probability for class k as
    the amount of information carried by the unit."

    :param this_unit_acts: normaized activations from the unit
    :param output_acts: activations of the output
    :param verbose: activations of the output

    :return: (Pearson's correlation coefficient, 2-tailed p-value)
    """
    print("**** class_correlation() ****")

    coef, p_val = pearsonr(x=this_unit_acts, y=output_acts)
    round_p = round(p_val, 3)
    corr = {"coef": coef, 'p': round_p}

    if verbose:
        print(f"corr: {corr}")

    return corr


def class_sel_basics(this_unit_acts_df, items_per_cat, n_classes, hi_val_thr=.5,
                     act_func='relu',
                     verbose=False):
    """
    will calculate the following (count & prop, per-class and total)
    1. means: per class and total
    2. sd: per class and total
    3. non_zeros count: per-class and total)
    4. non_zeros prop: (proportion of class) per class and total
    5. above normed thr: count per-class and total
    6. above normed thr: precision per-class (e.g. proportion of items above thr per class

    :param this_unit_acts_df: dataframe containing the activations for this unit
    :param items_per_cat: Number of items in each class
                        (or correct items or all items depending on hid acts)
    :param n_classes: Number of classes
    :param hi_val_thr: threshold above which an item is considered to be 'strongly active'.
    :param act_func: relu, sigmoid or tanh.  If Relu use normed acts, else use regular.

    :param verbose: how much to print to screen

    :return: class_sel_basics_dict
    """

    print("\n**** class_sel_basics() ****")

    act_values = 'activation'
    if act_func is 'relu':
        act_values = 'normed'
    if act_func is 'sigmoid':
        hi_val_thr = .75

    if not n_classes:
        n_classes = max(list(items_per_cat.keys()))

    # # means
    means_dict = dict(this_unit_acts_df.groupby('label')[act_values].mean())
    sd_dict = dict(this_unit_acts_df.groupby('label')[act_values].std())

    # # non-zero_count
    nz_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values]
                                           > 0.0].groupby('label')[act_values].count())

    for i in range(n_classes):
        if i not in list(nz_count_dict.keys()):
            nz_count_dict[i] = 0

    # nz_perplexity = sum(1 for i in nz_count_dict.values() if i >= 0)
    non_zero_count_total = this_unit_acts_df[this_unit_acts_df[act_values] > 0][act_values].count()

    # # non-zero prop
    nz_prop_dict = {k: (0 if items_per_cat[k] == 0 else nz_count_dict[k] / items_per_cat[k])
                    for k in items_per_cat.keys() & nz_count_dict}

    # # non_zero precision
    nz_prec_dict = {k: v / non_zero_count_total for k, v in nz_count_dict.items()}

    # # hi val count
    hi_val_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values] >
                                               hi_val_thr].groupby('label')[act_values].count())
    print("check hi-val-count")
    for i in range(n_classes):
        if i not in list(hi_val_count_dict.keys()):
            hi_val_count_dict[i] = 0
            print(i, hi_val_count_dict[i])

    hi_val_total = this_unit_acts_df[this_unit_acts_df[act_values] > hi_val_thr][act_values].count()

    # # hi vals precision
    hi_val_prec_dict = {k: (0 if v == 0 else v / hi_val_total) for k, v in hi_val_count_dict.items()}

    hi_val_prop_dict = {k: (0 if items_per_cat[k] == 0 else hi_val_count_dict[k] / items_per_cat[k])
                        for k in items_per_cat.keys() & hi_val_count_dict}

    class_sel_basics_dict = {"means": means_dict, "sd": sd_dict,
                             "nz_count": nz_count_dict, "nz_prop": nz_prop_dict,
                             'nz_prec': nz_prec_dict,
                             "hi_val_count": hi_val_count_dict, 'hi_val_prop': hi_val_prop_dict,
                             "hi_val_prec": hi_val_prec_dict,
                             }

    return class_sel_basics_dict


def coi_list(class_sel_basics_dict, verbose=False):
    """
    run class sel basics first

    will choose classes worth testing to save running sel on all classes.
    Will pick the top three classes on these criteria:

    1. highest proportion of non-zero activations
    2. highest mean activation
    3. most items active above .75 of normed range

    :param class_sel_basics_dict:
    :param verbose: how much to print to screen

    :return: coi_list: a list of class labels
    """

    print("**** coi_list() ****")

    copy_dict = copy.deepcopy(class_sel_basics_dict)
    means_dict = copy_dict['means']
    del means_dict['total']
    top_mean_cats = sorted(means_dict, key=means_dict.get, reverse=True)[:3]

    nz_prop_dict = copy_dict['nz_prop']
    del nz_prop_dict['total']
    lowest_nz_cats = sorted(nz_prop_dict, key=nz_prop_dict.get, reverse=True)[:3]

    hi_val_prec_dict = copy_dict['hi_val_prec']

    top_hi_val_cats = sorted(hi_val_prec_dict, key=hi_val_prec_dict.get, reverse=True)[:3]

    c_o_i_list = list(set().union(top_hi_val_cats, lowest_nz_cats, top_mean_cats))

    if verbose is True:
        print(f"COI\ntop_mean_cats: {top_mean_cats}\nlowest_nz_cats: {lowest_nz_cats}\n"
              f"top_hi_val_cats:{top_hi_val_cats}\nc_o_i_list: {c_o_i_list}")

    return c_o_i_list


def sel_unit_max(all_sel_dict, verbose=False):
    """
    Script to take the analysis for multiple classes and
    return the most selective class and value for each selectivity measure.

    :param all_sel_dict: Dict of all selectivity values for this unit
    :param verbose: how much to print to screen

    :return: small dict with just the max class for each measure
    """

    print("\n**** sel_unit_max() ****")

    copy_sel_dict = copy.deepcopy(all_sel_dict)

    # focussed_dict_print(copy_sel_dict, 'copy_sel_dict')

    max_sel_dict = dict()

    # # loop through unit dict of sel measure vals for each class
    for measure, class_dict in copy_sel_dict.items():
        # # for each sel measure get max value and class
        measure_c_name = f"{measure}_c"
        classes = list(class_dict.keys())
        values = list(class_dict.values())
        max_val = max(values)
        max_class = classes[values.index(max_val)]
        # print(measure, measure_c_name)

        # # copy max class and value to max_class_dict
        max_sel_dict[measure] = max_val
        max_sel_dict[measure_c_name] = max_class


    max_sel_dict['max_info_count'] = copy_sel_dict['max_info_count'][max_sel_dict["max_informed_c"]]
    del max_sel_dict['max_info_count_c']
    max_sel_dict['max_info_thr'] = copy_sel_dict['max_info_thr'][max_sel_dict["max_informed_c"]]
    del max_sel_dict['max_info_thr_c']
    max_sel_dict['max_info_sens'] = copy_sel_dict['max_info_sens'][max_sel_dict["max_informed_c"]]
    del max_sel_dict['max_info_sens_c']
    max_sel_dict['max_info_spec'] = copy_sel_dict['max_info_spec'][max_sel_dict["max_informed_c"]]
    del max_sel_dict['max_info_spec_c']
    max_sel_dict['max_info_prec'] = copy_sel_dict['max_info_prec'][max_sel_dict["max_informed_c"]]
    del max_sel_dict['max_info_prec_c']
    max_sel_dict['zhou_selects'] = copy_sel_dict['zhou_selects'][max_sel_dict["zhou_prec_c"]]
    del max_sel_dict['zhou_selects_c']
    max_sel_dict['zhou_thr'] = copy_sel_dict['zhou_thr'][max_sel_dict["zhou_prec_c"]]
    del max_sel_dict['zhou_thr_c']

    # # # max corr_coef shold be the absolute max (e.g., including negative) where p < .05.
    # # get all values into df
    # coef_array = []  # [corr_coef, abs(corr_coef), p, class]
    # for coef_k, coef_v in copy_sel_dict['corr_coef'].items():
    #     abs_coef = abs(coef_v)
    #     p = copy_sel_dict['corr_p'][coef_k]
    #     coef_array.append([coef_v, abs_coef, p, coef_k])
    # coef_df = pd.DataFrame(data=coef_array, columns=['coef', 'abs', 'p', 'class'])
    #
    # # # filter and sort df
    # coef_df = coef_df.loc[coef_df['p'] < 0.05]
    #
    # if not len(coef_df):  # if there are not items with that p_value
    #     max_sel_dict['corr_coef'] = float('NaN')
    #     max_sel_dict['corr_coef_c'] = float('NaN')
    #     max_sel_dict['corr_p'] = float('NaN')
    # else:
    #     coef_df = coef_df.sort_values(by=['abs'], ascending=False).reset_index()
    #     max_sel_dict['corr_coef'] = coef_df['coef'].iloc[0]
    #     max_sel_dict['corr_coef_c'] = coef_df['class'].iloc[0]
    #     max_sel_dict['corr_p'] = coef_df['p'].iloc[0]
    #
    # del max_sel_dict['corr_p_c']

    # # round values
    for k, v in max_sel_dict.items():
        if v is 'flaot':
            max_sel_dict[k] = round(v, 3)

    # print("\n\n\n\nmax sel dict", max_sel_dict)
    # focussed_dict_print(max_sel_dict, 'max_sel_dict')

    return max_sel_dict


def new_sel_dict_layout(sel_dict, all_or_max='max'):
    """
    for 'all' (sel_per_unit_dict) there are 5 layers:
        Original dict layout: layer, unit, ts, measure, classes
        New layout: measure, layer, unit, ts, classes

    for 'max' (max_sel_p_unit_dict) there are 4 layers:
        Original dict layout: layer, unit, ts, measure
        New layout: measure, layer, unit, ts

    :param sel_dict:
    :return: new_sel_dict
    """

    new_dict = dict()
    layer_list = list(sel_dict.keys())
    unit_list = list(sel_dict[layer_list[0]].keys())
    timestep_list = list(sel_dict[layer_list[0]][0].keys())
    sel_measure_list = list(sel_dict[layer_list[0]][0]['ts0'].keys())

    if all_or_max is 'all':
        class_list = list(sel_dict[layer_list[0]][0]['ts0'][sel_measure_list[0]].keys())

    for measure in sel_measure_list:
        new_dict[measure] = dict()

        for layer in layer_list:
            new_dict[measure][layer] = dict()

            for unit in unit_list:
                new_dict[measure][layer][unit] = dict()

                for ts in timestep_list:

                    if all_or_max is 'max':
                        ts_sel_score = sel_dict[layer][unit][ts][measure]
                        new_dict[measure][layer][unit][ts] = ts_sel_score

                    elif all_or_max is 'all':
                        new_dict[measure][layer][unit][ts] = dict()

                        for label in class_list:
                            class_sel_score = sel_dict[layer][unit][ts][measure][label]
                            new_dict[measure][layer][unit][ts][label] = class_sel_score

    return new_dict

####################################################################################################

def rnn_sel(gha_dict_path, correct_items_only=True, all_classes=True,
            save_output_to='pickle',
            verbose=False, test_run=False):
    """
    Analyse hidden unit activations.
    1. get basic study details from gha_dict
    2. load y, sort out incorrect responses
        get y_labels for letter-level analysis
    3. get output-activations for class-corr (if y_1hot == True)
    4. set place to save results (per unit/timestep)
        - set a way to check if layers/units have been analysed.
            json dict.  key: layer_name, value: max_unit_completed number or 'all'
    5. loop thru with loop_gha - get loop dict
    6. from loop dict, transform data in necessary
    7. call sel functions and test by class
        - also run sel on letters if relevant
        - call plot unit if meets some criterea
    8. save sel all units, al measures, all classes
        save max_sel class per unit, all measures

    :param gha_dict_path: path of the gha dict
    :param correct_items_only: Whether selectivity considered incorrect items
    :param all_classes: Whether to test for selectivity of all classes or a subset
                        (e.g., most active classes)
    :param verbose: how much to print to screen
    :param save_output_to: file-type to use, deault piclke
    :param test_run: if True, only do subset, e.g., 3 units from 3 layers

    :return: master dict: contains 'sel_path' (e.g., dir),
                                    'all_sel_dict_name',

    :return: all_sel_dict. dict with all selectivity results in it.
             all sel values:                 [layer names][unit number][sel measure][class]
             max_sel_p_u values(and class):  [layer names][unit number]['max'][sel measure]
             layer means:                    [layer name]['means'][measure]

    """


    print("\n**** running ff_sel() ****")

    # # use gha-dict_path to get exp_cond_gha_path, gha_dict_name,
    exp_cond_gha_path, gha_dict_name = os.path.split(gha_dict_path)
    os.chdir(exp_cond_gha_path)
    current_wd = os.getcwd()

    # # part 1. load dict from study (should run with sim, GHA or sel dict)
    gha_dict = load_dict(gha_dict_path)
    focussed_dict_print(gha_dict, 'gha_dict')

    # get topic_info from dict
    output_filename = gha_dict["topic_info"]["output_filename"]


    # # where to save files
    analyse_items = 'all'
    if correct_items_only:
        analyse_items = 'correct'
    sel_folder = f'{analyse_items}_sel'
    if test_run:
        sel_folder = f'{analyse_items}_sel/test'
    sel_path = os.path.join(current_wd, sel_folder)

    if not os.path.exists(sel_path):
        os.makedirs(sel_path)


    if verbose:
        print(f"\ncurrent_wd: {current_wd}")
        print(f"output_filename: {output_filename}")
        print(f"sel_path (to save): {sel_path}")


    # # get data info from dict
    n_cats = gha_dict["data_info"]["n_cats"]

    # # get model info from dict
    model_dict = gha_dict['model_info']['config']
    if verbose:
        focussed_dict_print(model_dict, 'model_dict')
    hid_units = gha_dict['model_info']['layers']['hid_layers']['hid_totals']["analysable"]
    units_per_layer = gha_dict['model_info']["overview"]["units_per_layer"]
    n_layers = gha_dict['model_info']['overview']['hid_layers']


    # # check for sequences/rnn
    sequence_data = False
    y_1hot = True

    if 'timesteps' in list(gha_dict['model_info']['overview'].keys()):
        sequence_data = True
        timesteps = gha_dict['model_info']["overview"]["timesteps"]
        serial_recall = gha_dict['model_info']["overview"]["serial_recall"]
        y_1hot = serial_recall
        vocab_dict = load_dict(os.path.join(gha_dict['data_info']["data_path"],
                                            gha_dict['data_info']["vocab_dict"]))

    # # get gha info from dict
    hid_acts_filename = gha_dict["GHA_info"]["hid_act_files"]['2d']
    gha_incorrect = gha_dict['GHA_info']['gha_incorrect']




    '''Part 2 - load y, sort out incorrect resonses'''
    print("\n\nPart 2: loading labels")
    # # load y_labels to go with hid_acts and item_correct for sequences
    if 'seq_corr_list' in list(gha_dict['GHA_info']['scores_dict'].keys()):
        n_seqs = gha_dict['GHA_info']['scores_dict']['n_seqs']
        n_seq_corr = gha_dict['GHA_info']['scores_dict']['n_seq_corr']
        n_incorrect = n_seqs - n_seq_corr

        test_label_seq_name = gha_dict['GHA_info']['y_data_path']
        seqs_corr = gha_dict['GHA_info']['scores_dict']['seq_corr_list']

        test_label_seqs = np.load(test_label_seq_name)
        if verbose:
            print(f"test_label_seqs: {np.shape(test_label_seqs)}")
            print(f"seqs_corr: {np.shape(seqs_corr)}")
            print(f"n_seq_corr: {n_seq_corr}")


        # # get 1hot item vectors for 'words' and 3 hot for letters
        '''Always use serial_recall True. as I want a separate 1hot vector for each item.
        Always use x_data_type 'local_letter_X' as I want 3hot vectors'''
        y_letters = []
        y_words = []
        for this_seq in test_label_seqs:
            get_letters, get_words = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                               seq_line=this_seq,
                                                               serial_recall=True,
                                                               end_seq_cue=False,
                                                               x_data_type='local_letter_X')
            y_letters.append(get_letters)
            y_words.append(get_words)

        y_letters = np.array(y_letters)
        y_words = np.array(y_words)
        if verbose:
            print(f"\ny_letters: {type(y_letters)}  {np.shape(y_letters)}")
            print(f"y_words: {type(y_words)}  {np.shape(y_words)}")
            # print(f"test_label_seqs[0]: {test_label_seqs[0]}")
            # if test_run:
            #     print(f"y_letters[0]:\n{y_letters[0]}")
            #     print(f"y_words[0]:\n{y_words[0]}")

        y_df_headers = [f"ts{i}" for i in range(timesteps)]
        y_scores_df = pd.DataFrame(data=test_label_seqs, columns=y_df_headers)
        y_scores_df['full_model'] = seqs_corr
        if verbose:
            print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")


    # # if not sequence data, load y_labels to go with hid_acts and item_correct for items
    elif 'item_correct_name' in list(gha_dict['GHA_info']['scores_dict'].keys()):
        # # load item_correct (y_data)
        item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
        # y_df = pd.read_csv(item_correct_name)
        y_scores_df = nick_read_csv(item_correct_name)



    """# # sort incorrect item data"""
    print("\n\nRemoving incorrect responses")
    # # # get values for correct/incorrect items (1/0 or True/False)
    item_correct_list = y_scores_df['full_model'].tolist()
    # full_model_values = y_scores_df.full_model.unique()
    full_model_values = list(set(item_correct_list))

    correct_symbol = 1
    if len(full_model_values) != 2:
        TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")
    if 1 not in full_model_values:
        if True in full_model_values:
            correct_symbol = True
        else:
            TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")

    print(f"len(full_model_values): {len(full_model_values)}")
    print(f"correct_symbol: {correct_symbol}")

    # # i need to check whether this analysis should include incorrect items (True/False)
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
    # print(f"incorrect_items: {np.shape(incorrect_items)}\n{incorrect_items}")

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
                print("I can not complete this as desried"
                      "change correct_items_only to True"
                      "for analysis  - don't remove anything from hid_acts, output and "
                      "use y scores as y_df")
            correct_items_only = True

    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")
        print(f"\ntest_label_seqs: {np.shape(test_label_seqs)}\n{test_label_seqs}")

    n_correct, timesteps = np.shape(test_label_seqs)
    corr_test_seq_name = f"{output_filename}_{n_correct}_corr_test_label_seqs.npy"
    np.save(corr_test_seq_name, test_label_seqs)


    # # get items per class
    IPC_dict = seq_items_per_class(label_seqs=test_label_seqs, vocab_dict=vocab_dict)
    focussed_dict_print(IPC_dict, 'IPC_dict')
    corr_test_IPC_name = f"{output_filename}_{n_correct}_corr_test_IPC.pickle"
    with open(corr_test_IPC_name, "wb") as pickle_out:
        pickle.dump(IPC_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    '''get output activations'''
    # # get output activations for class-corr if y_1hot == True
    if y_1hot:
        print("getting output activations to use for class_correlation")
        acts_saved_as = 'pickle'
        if '.pickle' == hid_acts_filename[-7:]:
            with open(hid_acts_filename, 'rb') as pkl:
                hid_acts_dict = pickle.load(pkl)

            hid_acts_keys_list = list(hid_acts_dict.keys())
            print(f"hid_acts_keys_list: {hid_acts_keys_list}")

            last_layer_num = hid_acts_keys_list[-1]
            last_layer_name = hid_acts_dict[last_layer_num]['layer_name']

            # output_layer_acts = hid_acts_dict['hid_acts_2d'][last_layer_name]
            if 'hid_acts' in list(hid_acts_dict[last_layer_num].keys()):
                output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts']
            elif 'hid_acts_2d' in list(hid_acts_dict[last_layer_num].keys()):
                output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts_2d']

        elif '.h5' == hid_acts_filename[-3:]:
            acts_saved_as = 'h5'
            with h5py.File(hid_acts_filename, 'r') as hid_acts_dict:
                hid_acts_keys_list = list(hid_acts_dict.keys())
                last_layer_num = hid_acts_keys_list[-1]
                last_layer_name = hid_acts_dict[last_layer_num]['layer_name']
                # output_layer_acts = hid_acts_dict['hid_acts_2d'][last_layer_name]
                if 'hid_acts' in list(hid_acts_dict[last_layer_num].keys()):
                    output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts']
                elif 'hid_acts_2d' in list(hid_acts_dict[last_layer_num].keys()):
                    output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts_2d']

        # close hid act dict to save memory space?
        hid_acts_dict.clear()


        # # output acts need to by npy because it can be 3d (seqs, ts, classes).
        if correct_items_only:
            if gha_incorrect:
                mask = np.ones(shape=len(seqs_corr), dtype=bool)
                mask[incorrect_items] = False
                output_layer_acts = output_layer_acts[mask]
                if verbose:
                    print(f"\nremoving {n_incorrect} incorrect responses from output_layer_acts: "
                          f"{np.shape(output_layer_acts)}\n")


        # save output activations
        output_acts_name = f'{sel_path}/{output_filename}_output_acts.npy'
        np.save(output_acts_name, output_layer_acts)

        # # clear memory
        output_layer_acts = []

    '''save results
    either make a new empty place to save.
    or load previous version and get the units I have already completed'''
    already_completed = dict()
    all_sel_dict = dict()
    max_sel_dict = dict()
    dict_layout = 'old'

    if save_output_to is 'pickle':
        all_sel_dict_name = f"{sel_path}/{output_filename}_sel_per_unit.pickle"
        max_sel_dict_name = f"{sel_path}/{output_filename}_max_sel_p_unit.pickle"

        if not os.path.isfile(all_sel_dict_name):
            # save all_sel_dict here to be opened and appended to
            with open(all_sel_dict_name, "wb") as pickle_out:
                pickle.dump(all_sel_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
            with open(max_sel_dict_name, "wb") as pickle_out:
                pickle.dump(max_sel_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            all_sel_dict = load_dict(all_sel_dict_name)
            max_sel_dict = load_dict(max_sel_dict_name)


            for key, value in all_sel_dict.items():
                unit_list = list(value.keys())
                if 'means' in unit_list:
                    unit_list.remove('means')
                max_unit = max(unit_list)
                already_completed[key] = max_unit

    if save_output_to is 'shelve':
        all_sel_dict_name = f"{sel_path}/{output_filename}_sel_per_unit"
        max_sel_dict_name = f"{sel_path}/{output_filename}_max_sel_p_unit"

        if not os.path.isfile(all_sel_dict_name):
            # # for shelve we just add the keys and values directly not the actual dict
            with shelve.open(all_sel_dict_name, protocol=pickle.HIGHEST_PROTOCOL) as db:
                # # make a test item just to establish the db
                db['test_key'] = 'test_value'
        else:
            with shelve.open(all_sel_dict_name) as db:
                all_sel_dict = db['all_sel_dict']
                keys_list = list(all_sel_dict.keys())
                for key in keys_list:
                    if 'test_key' in keys_list:
                        continue
                    if 'means' in keys_list:
                        keys_list.remove('means')
                    max_unit = max(keys_list)
                    already_completed[key] = max_unit


    if test_run:
        already_completed = dict()
    print(f"\nlayers/units already_completed: {already_completed}")


    '''
    part 3   - get gha for each unit
    '''




    loop_gha = loop_thru_acts(gha_dict_path=gha_dict_path,
                              correct_items_only=correct_items_only,
                              already_completed=already_completed,
                              verbose=True,
                              test_run=True
                              )

    for index, unit_gha in enumerate(loop_gha):
        this_dict = {'roc_auc': {},
                     'ave_prec': {},
                     'pr_auc': {},
                     'max_informed': {},
                     'max_info_count': {},
                     'max_info_thr': {},
                     'max_info_sens': {},
                     'max_info_spec': {},
                     'max_info_prec': {},
                     'ccma': {},
                     'zhou_prec': {},
                     'zhou_selects': {},
                     'zhou_thr': {},
                     'corr_coef': {},
                     'corr_p': {},
                     }

        # print(f"\n\n{index}:\n{unit_gha}\n")
        sequence_data = unit_gha["sequence_data"]
        y_1hot = unit_gha["y_1hot"]
        act_func = unit_gha["act_func"]
        layer_number = unit_gha["layer_number"]
        layer_name = unit_gha["layer_name"]
        unit_index = unit_gha["unit_index"]
        timestep = unit_gha["timestep"]
        ts_name = f"ts{timestep}"
        item_act_label_array = unit_gha["item_act_label_array"]
        IPC_words = IPC_dict['word_p_class_p_ts'][ts_name]

        # #  make df
        this_unit_acts = pd.DataFrame(data=item_act_label_array, columns=['item', 'activation', 'label'])
        this_unit_acts_df = this_unit_acts.astype({'item': 'int32', 'activation': 'float', 'label': 'int32'})


        # sort by descending hid act values
        this_unit_acts_df = this_unit_acts_df.sort_values(by='activation', ascending=False)

        # # # normalize activations
        if act_func in ['relu', 'ReLu', 'Relu']:
            just_act_values = this_unit_acts_df['activation'].tolist()
            max_act = max(just_act_values)
            normed_acts = np.true_divide(just_act_values, max_act)
            this_unit_acts_df.insert(2, column='normed', value=normed_acts)

        if verbose is True:
            print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n"
                  # f"{this_unit_acts_df.head()}"
                  )



        # # run sel on word items
        # # get class_sel_basics (class_means, sd, prop > .5, prop @ 0)
        class_sel_basics_dict = class_sel_basics(this_unit_acts_df=this_unit_acts_df,
                                                 items_per_cat=IPC_words,
                                                 n_classes=n_cats,
                                                 act_func=act_func,
                                                 verbose=verbose)

        if verbose:
            focussed_dict_print(class_sel_basics_dict, 'class_sel_basics_dict')

        # # add class_sel_basics_dict to unit dict
        for csb_key, csb_value in class_sel_basics_dict.items():
            if csb_key == 'total':
                continue
            if csb_key == 'perplexity':
                continue
            this_dict[csb_key] = csb_value



        classes_of_interest = list(range(n_cats))
        if all_classes is False:
            # # I don't want to use all classes, just ones that are worth testing
            classes_of_interest = coi_list(class_sel_basics_dict, verbose=verbose)

        print('\n**** cycle through classes ****')
        for this_cat in range(len(classes_of_interest)):

            if this_cat in list(IPC_words.keys()):
                this_class_size = IPC_words[this_cat]
            else:
                this_class_size = 0
            not_a_size = n_correct - this_class_size

            if verbose is True:
                print(f"\nclass_{this_cat}: {this_class_size} items, "
                      f"not_{this_cat}: {not_a_size} items")

            # # running selectivity measures

            # # ROC_stuff includes:
            # roc_auc, ave_prec, pr_auc, nz_ave_prec, nz_pr_auc, top_class_sel, informedness

            # # only normalise activations for relu
            act_values = 'activation'
            if act_func is 'relu':
                act_values = 'normed'

            roc_stuff_dict = nick_roc_stuff(class_list=this_unit_acts_df['label'],
                                            hid_acts=this_unit_acts_df[act_values],
                                            this_class=this_cat,
                                            class_a_size=this_class_size,
                                            not_a_size=not_a_size,
                                            verbose=verbose)

            print(f"roc_stuff_dict:\n{roc_stuff_dict}")

            # # add class_sel_basics_dict to unit dict
            for roc_key, roc_value in roc_stuff_dict.items():
                this_dict[roc_key][this_cat] = roc_value

            # # CCMA
            class_a = this_unit_acts_df.loc[this_unit_acts_df['label'] == this_cat]
            class_a_mean = class_a[act_values].mean()
            not_class_a = this_unit_acts_df.loc[this_unit_acts_df['label'] != this_cat]
            not_class_a_mean = not_class_a[act_values].mean()
            ccma = (class_a_mean - not_class_a_mean) / (class_a_mean + not_class_a_mean)
            this_dict["ccma"][this_cat] = ccma

            # # zhou_prec
            zhou_cut_off = .005
            if n_correct < 20000:
                zhou_cut_off = 100 / n_correct
            if n_correct < 100:
                zhou_cut_off = 1 / n_correct
            zhou_selects = int(n_correct * zhou_cut_off)

            most_active = this_unit_acts_df.iloc[:zhou_selects]

            if 'normed' in list(this_unit_acts_df):
                zhou_thr = list(most_active["normed"])[-1]
            else:
                zhou_thr = list(most_active["activation"])[-1]

            zhou_prec = sum([1 for i in most_active['label'] if i == this_cat]) / zhou_selects
            this_dict["zhou_prec"][this_cat] = zhou_prec
            this_dict["zhou_selects"][this_cat] = zhou_selects
            this_dict["zhou_thr"][this_cat] = zhou_thr

            # class correlation
            # get output activations for class correlation
            # # can only run this on y_1hot
            if y_1hot:
                output_layer_acts = np.load(output_acts_name)
                # print(f"np.shape(output_layer_acts): {np.shape(output_layer_acts)}")
                output_acts_ts = output_layer_acts[:, timestep, :]
                # print(f"np.shape(output_acts_ts): {np.shape(output_acts_ts)}")

                class_corr = class_correlation(this_unit_acts=this_unit_acts_df[act_values],
                                               output_acts=output_acts_ts[:, this_cat],
                                               verbose=verbose)
                this_dict["corr_coef"][this_cat] = class_corr['coef']
                this_dict["corr_p"][this_cat] = class_corr['p']
            else:
                if 'corr_coef' in list(this_dict.keys()):
                    del this_dict['corr_coef']
                    del this_dict['corr_p']


        # # which class was the highest for each measure
        max_sel_p_unit_dict = sel_unit_max(this_dict, verbose=verbose)



        # # # # once sel analysis has been done for this hid_act array

        # # sort dicts to save
        # # add layer to all_sel_dict
        if layer_name not in list(all_sel_dict.keys()):
            all_sel_dict[layer_name] = dict()
            max_sel_dict[layer_name] = dict()

        # # add unit index to sel_p_unit dict
        if unit_index not in list(all_sel_dict[layer_name].keys()):
            all_sel_dict[layer_name][unit_index] = dict()
            max_sel_dict[layer_name][unit_index] = dict()

        # # if not sequences data, add this unit to all_sel_dict
        if not sequence_data:
            all_sel_dict[layer_name][unit_index] = this_dict
            max_sel_dict[layer_name][unit_index] = max_sel_p_unit_dict

        else:  # # if sequence data
            # # add timestep to max sel_p_unit dict
            if timestep not in list(all_sel_dict[layer_name][unit_index].keys()):
                all_sel_dict[layer_name][unit_index][ts_name] = dict()
                max_sel_dict[layer_name][unit_index][ts_name] = dict()

            # # add this timestep to all_sel_dict
            all_sel_dict[layer_name][unit_index][ts_name] = this_dict
            max_sel_dict[layer_name][unit_index][ts_name] = max_sel_p_unit_dict



        # # save unit/timestep to disk
        if save_output_to is 'pickle':
            with open(all_sel_dict_name, "wb") as pickle_out:
                pickle.dump(all_sel_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
            with open(max_sel_dict_name, "wb") as pickle_out:
                pickle.dump(max_sel_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        if save_output_to is 'shelve':
            with shelve.open(all_sel_dict_name, protocol=pickle.HIGHEST_PROTOCOL) as db:
                # # make a test item just to establish the db
                db['all_sel_dict'] = all_sel_dict
                db['max_sel_dict'] = max_sel_dict

        print("saved to disk")


    print(f"********\nfinished looping through units************")


    new_all_sel_dict = new_sel_dict_layout(all_sel_dict, 'all')
    new_max_sel_dict = new_sel_dict_layout(max_sel_dict, 'max')
    dict_layout = 'new'
    #
    # # todo: Do I need these new dict layouts?  I think not

    if verbose:
        focussed_dict_print(all_sel_dict, 'all_sel_dict')
        # print_nested_round_floats(all_sel_dict, 'all_sel_dict')

    # # save dict
    print("\n\n\n*****************\nanalysis complete\n*****************")

    # # get max_sel dict


    sel_dict = gha_dict

    sel_dict_name = f"{sel_path}/{output_filename}_sel_dict.pickle"

    sel_dict["sel_info"] = {"sel_path": sel_path,
                            'sel_dict_name': sel_dict_name,
                            "all_sel_dict_name": all_sel_dict_name,
                            'max_sel_dict_name': max_sel_dict_name,
                            "correct_items_only": correct_items_only,
                            "all_classes": all_classes,
                            'corr_test_seq_name': corr_test_seq_name,
                            'corr_test_IPC_name': corr_test_IPC_name,
                            "sel_date": int(datetime.datetime.now().strftime("%y%m%d")),
                            "sel_time": int(datetime.datetime.now().strftime("%H%M")),
                            }

    print(f"Saving dict to: {os.getcwd()}")
    pickle_out = open(sel_dict_name, "wb")
    pickle.dump(sel_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()

    focussed_dict_print(sel_dict, "sel_dict")


    '''call from sel_p_unit dict
    for each measure:
        make a df? 
            all units, all timesteps, 1 layer
            all units, 1 timestep, 1 layer
            
        
        get mean and max sel
    
    loop through layers
    
    
    
    make mean and max csv
    
    get top_n units for each sel measure
    
    sort letter sel
    
    '''

    print("\nend of sel script")

    return sel_dict  # , mean_sel_per_NN




###############################

def get_sel_summaries(max_sel_dict_path, verbose=False):
    """
    max sel dict is a nested dict [layers][units][timesteps][measures] = floats

    1. make sel_df: multi-indexed for layer, units, ts as rows.  Cols are all sel measures
    save this


    use sel_df to get mean and max sel scores for whole model
     - have simplified means version to use as summary df.


    2. should have ability to call and at later date get mean and max (using xs) for:
        - each layer (all ts)
        - each timestep (all layers)

    3. Should make highlights dict with the layer, unit, ts of the 3 highest per measure
        - should include units where sel-class is the same for all timesteps.

    4. add placeholder to plot hist of sel scores for a given measure (or measures)

        1.


    :param max_sel_dict_path:
    :return:
    """

    # todo: save csvs, pickle dicts
    '''return
	summary vals and headers for csv

save csv:
sel_df : master df all layers, units, timesteps, all measures
model mean-max df: mean and max vals for all measures and layers and ts

save dicts
hl_df_dict - dataframe per measure

hl_units_dict - nested map to hl units. 
	lists timestep-invariant measures at relevent units
	lists high-scoring measures at relevant unit/timesteps'''

    # # use max_sel_dict_path to get exp_cond_gha_path, sel_dict_name,
    exp_cond_gha_path, sel_dict_name = os.path.split(max_sel_dict_path)
    os.chdir(exp_cond_gha_path)
    current_wd = os.getcwd()

    sel_dict = load_dict(max_sel_dict_path)

    # # sel_p_unit_layout
    '''print("\nORIG max_sel_p_unit dict")
    print(f"\nFirst nest is Layers: {len(list(sel_dict.keys()))} keys.\n{list(sel_dict.keys())}"
          f"\neach with value is a dict.")
    second_layer = sel_dict[list(sel_dict.keys())[0]]
    print(f"\nsecond nest is Units: {len(list(second_layer.keys()))} keys.\n{list(second_layer.keys())}"
          f"\neach with value is a dict.")
    third_layer = second_layer[list(second_layer.keys())[0]]
    print(f"\nThird nest is Timesteps: {len(list(third_layer.keys()))} keys.\n{list(third_layer.keys())}"
          f"\neach with value is a dict.")
    fourth_layer = third_layer[list(third_layer.keys())[0]]
    print(f"\nfourth nest is measures: {len(list(fourth_layer.keys()))} keys.\n{list(fourth_layer.keys())}"
          f"\neach with value is a single item.  {fourth_layer[list(fourth_layer.keys())[0]]}\n")'''

    '''get list of sel measures,
    remove measures where I haven't recorded associated class-label (drop_these_measures)
    remove class-with-highest-sel which are items ending in "_c"'''
    all_sel_measures_list = list(sel_dict[list(sel_dict.keys())[0]][0]['ts0'].keys())

    drop_these_measures = ['max_info_count', 'max_info_thr', 'max_info_sens',
                           'max_info_spec', 'max_info_prec', 'zhou_selects', 'zhou_thr']
    sel_measures_list = [measure for measure in all_sel_measures_list if measure not in drop_these_measures]
    sel_measures_list = [measure for measure in sel_measures_list if measure[-2:] != '_c']
    if verbose:
        print(f"{len(sel_measures_list)} items in sel_measures_list.\n{sel_measures_list}")


    '''reform nested dict before attempting to make df
    https://stackoverflow.com/questions/30384581/nested-dictionary-to-multiindex-pandas-dataframe-3-level
    '''
    reform_nested_sel_dict = {(level1_key, level2_key, level3_key): values
                              for level1_key, level2_dict in sel_dict.items()
                              for level2_key, level3_dict in level2_dict.items()
                              for level3_key, values      in level3_dict.items()}

    sel_df = pd.DataFrame(reform_nested_sel_dict).T
    sel_df_index_names = ['Layer', 'Unit', 'Timestep']
    sel_df.index.set_names(sel_df_index_names, inplace=True)

    # # convert class ('_c') cols to int
    class_cols_list = [measure for measure in all_sel_measures_list if measure[-2:] == '_c']
    class_cols_dict = {i: 'int32' for i in class_cols_list}
    sel_df = sel_df.astype(class_cols_dict)


    if verbose:
        print(f"sel_df:\n{sel_df}")


    '''use sel_df.xs (cross-section) to select info in multi-indexed dataframes'''
    # layer_df = sel_df.xs('hid2', level='Layer')
    # unit_df = sel_df.xs(1, level='Unit')
    # ts_df = sel_df.xs('ts2', level='Timestep')
    # layer_unit_df = sel_df.xs(('hid2', 2), level=('Layer', 'Unit'))
    # layer_ts_df = sel_df.xs(('hid2', 'ts2'), level=('Layer', 'Timestep'))
    # unit_ts_df = sel_df.xs((2, 'ts2'), level=('Unit', 'Timestep'))
    # print(unit_ts_df)


    '''
    summary stats
    
    from max_sel_dict:
    1. use sel_df to get mean and max sel scores for whole model
     - have simplified means version to use as summary df.
        
        
    2. should have ability to call and at later date get mean and max (using xs) for:
        - each layer (all ts)
        - each timestep (all layers)
        
    3. Should make highlights dict with the layer, unit, ts of the 3 highest per measure
        - should include units where sel-class is the same for all timesteps.
        
    4. add placeholder to plot hist of sel scores for a given measure (or measures)'''

    '''dicts should be nested
    means: whole_model, n_layers, n_timesteps'''

    '''1. get means and max for the whole model'''
    sel_measures_df = sel_df[sel_measures_list]
    sel_means_s = sel_measures_df.mean().rename('model_means')
    sel_max_s = sel_measures_df.max().rename('model_max')


    '''1. get values for summary csv'''
    mi_mean = sel_means_s.loc['max_informed']
    mi_max = sel_max_s.loc['max_informed']
    ccma_mean = sel_means_s.loc['ccma']
    ccma_max = sel_max_s.loc['ccma']
    prec_mean = sel_means_s.loc['zhou_prec']
    prec_max = sel_max_s.loc['zhou_prec']
    means_mean = sel_means_s.loc['means']
    means_max = sel_max_s.loc['means']

    for_summary_headers = ["mi_mean", "mi_max", "ccma_mean", "ccma_max",
                           "prec_mean", "prec_max", "means_mean", "means_max"]

    for_summary_vals = [mi_mean, mi_max, ccma_mean, ccma_max,
                        prec_mean, prec_max, means_mean, means_max]


    '''2. get mean and max per layer'''

    layer_names_list = sorted(list(set(sel_df._get_label_or_level_values('Layer'))))
    units_names_list = sorted(list(set(sel_df._get_label_or_level_values('Unit'))))
    ts_names_list = sorted(list(set(sel_df._get_label_or_level_values('Timestep'))))

    looped_arrays = [sel_means_s, sel_max_s]

    if len(layer_names_list) > 1:
        for this_layer in layer_names_list:
            # # select relevant rows from list
            layer_measure_df = sel_measures_df.xs(this_layer, level='Layer')

            # # get means and max vales series
            sel_means_s = sel_measures_df.mean().rename(f'{this_layer}_means')
            sel_max_s = sel_measures_df.max().rename(f'{this_layer}_max')

            looped_arrays.append(sel_means_s)
            looped_arrays.append(sel_max_s)




    '''2. get mean and max per timestep'''
    ts_names_list = sorted(list(set(sel_df._get_label_or_level_values('Timestep'))))
    # looped_arrays = []

    for this_ts in ts_names_list:
        # # select relevant rows from list
        ts_measure_df = sel_measures_df.xs(this_ts, level='Timestep')

        # # get means and max vales series
        sel_means_s = sel_measures_df.mean().rename(f'{this_ts}_means')
        sel_max_s = sel_measures_df.max().rename(f'{this_ts}_max')

        looped_arrays.append(sel_means_s)
        looped_arrays.append(sel_max_s)

    model_mean_max_df = pd.concat(looped_arrays, axis='columns')
    print(f"model_mean_max_df: \n{model_mean_max_df}")
    # ts_mean_max_df = pd.concat(looped_arrays, axis='columns')
    # print(ts_mean_max_df)

    '''3. Should make highlights dict with the layer, unit, ts of the 3 highest per measure'''
    '''
    hl_df_dict: keys: measures; values: dataframe
    
    hl_units_dict: nested keys: layer, unit, timestep; 
                values at unit: list of measures where it is timestep invariant
                values at timesteps: tuple(measure, score, label)
    '''
    hl_df_dict = dict()
    hl_units_dict = dict()

    top_n = 3

    measure = 'roc_auc'

    for measure in sel_measures_list:
        # # if there are lots of 1.0, keep them all
        if len(sel_df[sel_df[measure] == 1.0]) > top_n:
            top_units = sel_df[sel_df[measure] == 1.0]
        else:
            # # just take n highest scores
            top_units = sel_df.nlargest(n=top_n, columns=measure)

        # # reduce df to just keep measure and relevant label
        m_label = f"{measure}_c"
        cols_to_keep = [measure, m_label]
        top_units = top_units[cols_to_keep]

        # # get ranks for scores
        check_values = top_units.loc[:, measure].to_list()
        rank_values = rankdata([-1 * i for i in check_values], method='dense')
        top_units['rank'] = rank_values

        # # append to df_dict
        hl_df_dict[measure] = top_units

        # # loop through top units to populate unit dict
        for index, row in top_units.iterrows():
            # add indices to dict (0: layer, 1: unit, 2: timestep)
            if row.name[0] not in hl_units_dict.keys():
                hl_units_dict[row.name[0]] = dict()
            if row.name[1] not in hl_units_dict[row.name[0]].keys():
                hl_units_dict[row.name[0]][row.name[1]] = dict()
            if row.name[2] not in hl_units_dict[row.name[0]][row.name[1]].keys():
                hl_units_dict[row.name[0]][row.name[1]][row.name[2]] = []

            # # add values to dict (0: measure, 1: label, 2: rank)
            hl_units_dict[row.name[0]][row.name[1]][row.name[2]].append(
                (measure, row.values[0], row.values[1], f'rank_{row.values[2]}'))


    print_nested_round_floats(hl_units_dict, 'hl_units_dict')
    # focussed_dict_print(hl_df_dict, 'hl_df_dict')



    '''3. highlights should include units where sel-class is the same for all timesteps.'''

    for layer in layer_names_list:
        for unit in units_names_list:
            # # get array for this layer, unit - all timesteps
            layer_unit_df = sel_df.xs((layer, unit), level=('Layer', 'Unit'))
            for measure in sel_measures_list:
                # # get max sel label for these timesteps
                check_classes = layer_unit_df.loc[:, f'{measure}_c'].to_list()
                # # if there is only 1 label
                if len(set(check_classes)) == 1:
                    # # add indices to dict
                    if layer not in hl_units_dict.keys():
                        hl_units_dict[layer] = dict()
                    if unit not in hl_units_dict[layer].keys():
                        hl_units_dict[layer][unit] = dict()
                    if 'ts_invar' not in hl_units_dict[layer][unit]:
                        hl_units_dict[layer][unit]['ts_invar'] = []

                    # # add measure name to dict
                    hl_units_dict[layer][unit]['ts_invar'].append(measure)


    print_nested_round_floats(hl_units_dict, 'hl_units_dict')

    return for_summary_headers, for_summary_vals


'''# todo: 4. add placeholder to plot hist of sel scores for a given measure (or measures)'''


max_free_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
            'STM_RNN_test_v30_free_recall/test/all_generator_gha/test/correct_sel/test/' \
            'STM_RNN_test_v30_free_recall_max_sel_p_unit.pickle'
# all_free_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
#             'STM_RNN_test_v30_free_recall/test/all_generator_gha/test/correct_sel/test/' \
#             'STM_RNN_test_v30_free_recall_sel_per_unit.pickle'

# seri_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
#             'STM_RNN_test_v30_serial_recall/test/all_generator_gha/test/correct_sel/test/' \
#             'STM_RNN_test_v30_serial_recall_max_sel_p_unit.pickle'
# seri_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
#             'STM_RNN_test_v30_serial_recall/test/all_generator_gha/test/correct_sel/test/' \
#             'STM_RNN_test_v30_serial_recall_sel_per_unit.pickle'

ser_3l_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
              'STM_RNN_test_v30_serial_recall_3l/test/all_generator_gha/test/correct_sel/' \
              'test/STM_RNN_test_v30_serial_recall_3l_max_sel_p_unit.pickle'


sel_stats = get_sel_summaries(ser_3l_path)







    #
    #     # todo: write results script which will take the input dict/csv,
    #     #  put it into a pandas df, have versions for sort by order of x variable,
    #     #  select top n, etc.  use that to then output a list of units to visualise.
    #     #  Can add call visualise unit
    #
    #
    #     # # # get top three highlights for each feature
    #     # todo: Highlights should include timestep as a variable.
    #     #  e.g., unit 1, timestep 3 has max ROC score
    #     with open(sel_highlights_list_dict_name, "rb") as pickle_load:
    #         # read dict as it is so far
    #         highlights_dict = pickle.load(pickle_load)
    #
    #         highlights_list = list(highlights_dict.keys())
    #
    #     for measure in highlights_list:
    #         # sort max_sel_df by sel measure for this layer
    #         layer_top_3_df = max_sel_df.sort_values(by=measure, ascending=False).reset_index()
    #
    #         # then select info for highlights dict for top 3 items
    #
    #         print(f"layer_top_3_df\n{layer_top_3_df}")
    #         print(f"layer_top_3_df.loc[{measure}]\n{layer_top_3_df[measure]}")
    #
    #         if measure == 'max_informed':
    #             print('max_informed')
    #             # "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
    #             #                                          'prec', 'f1', 'count', 'layer', 'unit']],
    #             for i in range(2):
    #                 top_val = layer_top_3_df[measure].iloc[i]
    #                 if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                'unit'])) > 1:
    #                     top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                         f'{measure}_c'])[i]
    #                     count = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                     'max_info_count'])[i]
    #                     thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                   'max_info_thr'])[i]
    #                     sens = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    'max_info_sens'])[i]
    #                     spec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    'max_info_spec'])[i]
    #                     prec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    'max_info_prec'])[i]
    #                     # f1 = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                     # 'max_info_f1'])[i]
    #                     top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                             'unit'])[i]
    #                 else:
    #                     top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    f'{measure}_c'].item()
    #                     count = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                'max_info_count'].item()
    #                     thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                              'max_info_thr'].item()
    #                     sens = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                               'max_info_sens'].item()
    #                     spec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                               'max_info_spec'].item()
    #                     prec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                               'max_info_prec'].item()
    #                     # f1 = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                     # 'max_info_f1'].item()
    #                     top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                        'unit'].item()
    #
    #                 # new_row = [top_val, top_class, count, thr, sens, spec, prec, f1, layer_name, top_unit_name]
    #                 new_row = [top_val, top_class, count, thr, sens,
    #                            spec, prec, layer_name, top_unit_name]
    #
    #                 # print(f"new_row\n{new_row}")
    #                 # highlights_dict[measure].append(new_row)
    #
    #         elif measure == 'zhou_prec':
    #             print('zhou_prec')
    #             # "zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],
    #             for i in range(2):
    #                 top_val = layer_top_3_df[measure].iloc[i]
    #                 if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                'unit'])) > 1:
    #                     top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                         f'{measure}_c'])[i]
    #                     thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                   'zhou_thr'])[i]
    #                     selects = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                       'zhou_selects'])[i]
    #                     top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                             'unit'])[i]
    #                 else:
    #                     top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    f'{measure}_c'].item()
    #                     thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                              'zhou_thr'].item()
    #                     selects = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                  'zhou_selects'].item()
    #                     top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                        'unit'].item()
    #
    #                 new_row = [top_val, top_class, thr, selects, layer_name, top_unit_name]
    #                 # print(f"new_row\n{new_row}")
    #                 # highlights_dict[measure].append(new_row)
    #
    #         # elif measure == 'corr_coef':
    #         #     print('corr_coef')
    #         #     #   "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],
    #         #     for i in range(2):
    #         #         top_val = layer_top_3_df[measure].iloc[i]
    #         #         if np.isnan(top_val):
    #         #             continue
    #         #         if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
    #         #             top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #         #                                                 f'{measure}_c'])[i]
    #         #             p = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #         #                                         'corr_p'])[i]
    #         #             top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #         #                                                     'unit'])[i]
    #         #         else:
    #         #             top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #         #                                            f'{measure}_c'].item()
    #         #             p = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #         #                                    'corr_p'].item()
    #         #             top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #         #                                                'unit'].item()
    #         #
    #         #         new_row = [top_val, top_class, p, layer_name, top_unit_name]
    #         #         # print(f"new_row\n{new_row}")
    #         #         # highlights_dict[measure].append(new_row)
    #
    #         elif measure == 'means':
    #             print('means')  # "means": [['value', 'class', 'sd', 'layer', 'unit']],
    #             for i in range(2):
    #                 top_val = layer_top_3_df[measure].iloc[i]
    #                 if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
    #                     top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                         f'{measure}_c'])[i]
    #                     sd = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                  'sd'])[i]
    #                     top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                             'unit'])[i]
    #                 else:
    #                     top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    f'{measure}_c'].item()
    #                     sd = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                             'sd'].item()
    #                     top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                        'unit'].item()
    #
    #                 new_row = [top_val, top_class, sd, layer_name, top_unit_name]
    #                 # print(f"new_row\n{new_row}")
    #                 # highlights_dict[measure].append(new_row)
    #
    #         else:  # for most most measures use below
    #             for i in range(2):
    #                 top_val = layer_top_3_df[measure].iloc[i]
    #                 # print('top_val: ', top_val)
    #                 # print(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])
    #                 if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
    #                     top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                         f'{measure}_c'])[i]
    #                     top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                             'unit'])[i]
    #                 else:
    #                     top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                    f'{measure}_c'].item()
    #                     top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
    #                                                        'unit'].item()
    #
    #                 new_row = [top_val, top_class, layer_name, top_unit_name]
    #
    #                 # print(f"new_row\n{new_row}")
    #                 # highlights_dict[measure].append(new_row)
    #
    #         print(f"new_row\n{new_row}")
    #
    #         with open(sel_highlights_list_dict_name, "rb") as pickle_load:
    #             # read dict as it is so far
    #             highlights_dict = pickle.load(pickle_load)
    #
    #         highlights_dict[measure].append(new_row)
    #
    #         print(f"\nhighlights_dict[{measure}]\n{highlights_dict[measure]}")
    #
    #         # save highlights_dict here
    #         with open(sel_highlights_list_dict_name, "wb") as pickle_out:
    #             pickle.dump(highlights_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #         highlights_dict.clear()
    #
    #     print(f"\nFinihshed getting data for highlights_dict")
    #
    # print("\n****** Finished looping through all layers ******")
    #
    # # # add means total
    # lm = pd.read_csv(layer_means_path)
    # # lm.set_index('name')
    # # print("check column names")
    # # print(f"{len(list(lm))} cols\n{list(lm)}")
    # print(lm)
    # print(lm['name'].to_list())
    # print(f"already_done_means_total: {already_done_means_total}")
    # if lm['name'].to_list()[-1] != 'Total':
    #     already_done_means_total = False
    #
    # if not already_done_means_total:
    #     if 'Total' in lm['name'].to_list():
    #         print("total already here, removing and starting again")
    #         lm = lm[lm.name != 'Total']
    #         print(lm)
    #
    #     print("appending total to layer_means csv")
    #     total_means = []
    #     for column_name, column_data in lm.iteritems():
    #         # print(f"column_name: {column_name}")
    #         if column_name == 'name':
    #             # ignore this column
    #             # continue
    #             total_means.append('Total')
    #         elif column_name == 'units':
    #             sum_c = column_data.sum()
    #             total_means.append(sum_c)
    #         else:
    #             mean = column_data.mean()
    #             total_means.append(mean)
    #
    #     totals_s = pd.Series(data=total_means, index=lm.columns, name='Total')
    #     lm = lm.append(totals_s)
    #
    #     lm.to_csv(layer_means_path, index=False)
    #
    # # # make new highlights_df_dict with dataframes
    # highlights_df_dict = dict()
    #
    # # load highlights dict
    # with open(sel_highlights_list_dict_name, "rb") as pickle_load:
    #     # read dict as it is so far
    #     highlights_dict = pickle.load(pickle_load)
    #
    #     for k, v in highlights_dict.items():
    #         hl_df = pd.DataFrame(data=v[1:], columns=v[0])
    #         hl_df.sort_values(by='value', ascending=False, inplace=True)
    #         highlights_df_dict[k] = hl_df
    #         if verbose:
    #             print(f"\n{k} highlights (head)")
    #             print(hl_df)
    #
    # # Save new highlights_df_dict here
    # sel_highlights_df_dict_name = f"{sel_path}/{output_filename}_highlights.pickle"
    # with open(sel_highlights_df_dict_name, "wb") as pickle_out:
    #     pickle.dump(highlights_df_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # sel_highlights_list_dict_name = f"{sel_path}/{output_filename}_highlights.pickle"
    # # pickle_out = open(sel_highlights_list_dict_name, "wb")
    # # pickle.dump(highlights_df_dict, pickle_out)
    # # pickle_out.close()
    #
    # # sel_per_unit_pickle_name = f"{sel_path}/{output_filename}_sel_per_unit.pickle"
    # # pickle_out = open(sel_per_unit_pickle_name, "wb")
    # # pickle.dump(sel_p_unit_dict, pickle_out)
    # # pickle_out.close()
    #
    # # # save dict
    # print("\n\n\n*****************\nanalysis complete\n*****************")
    #
    # master_dict = dict()
    # master_dict["topic_info"] = gha_dict['topic_info']
    # master_dict["data_info"] = gha_dict['data_info']
    # master_dict["model_info"] = gha_dict['model_info']
    # master_dict["training_info"] = gha_dict['training_info']
    # master_dict["GHA_info"] = gha_dict['GHA_info']
    #
    # sel_dict_name = f"{sel_path}/{output_filename}_sel_dict.pickle"
    #
    # master_dict["sel_info"] = {"sel_path": sel_path,
    #                            'sel_dict_name': sel_dict_name,
    #                            # "sel_per_unit_pickle_name": sel_per_unit_pickle_name,
    #                            "sel_per_unit_name": sel_per_unit_db_name,
    #                            'sel_highlights_list_dict_name': sel_highlights_list_dict_name,
    #                            "correct_items_only": correct_items_only,
    #                            "all_classes": all_classes,
    #                            "sel_date": int(datetime.datetime.now().strftime("%y%m%d")),
    #                            "sel_time": int(datetime.datetime.now().strftime("%H%M")),
    #                            }
    #
    # print(f"Saving dict to: {os.getcwd()}")
    # pickle_out = open(sel_dict_name, "wb")
    # pickle.dump(master_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle_out.close()
    #
    # focussed_dict_print(master_dict, "master_dict")
    # # print_nested_round_floats(master_dict)
    #
    # # # save summary csv
    # key_layers_list = [x for x in gha_dict['GHA_info']['gha_key_layers']
    #                    if 'output' not in str.lower(x)]
    # last_hid_layer = key_layers_list[-1]
    #
    # print("lm")
    # print(lm)
    # # mean_roc = lm.at['Total', 'roc_auc']
    # # mean_ap = lm.at['Total', 'ave_prec']
    # # mean_info = lm.at['Total', 'max_informed']
    #
    # # mean_roc = lm.loc[lm['name'] == 'Total', 'roc_auc']
    # # mean_ap = lm.loc[lm['name'] == 'Total', 'ave_prec']
    # # mean_info = lm.loc[lm['name'] == 'Total', 'max_informed']
    #
    # print("\ngetting max sel values from highlights")
    # # todo: add timestep as a variable for these too?
    # # # roc
    # top_roc_df = pd.DataFrame(highlights_dict['roc_auc'])
    # max_sel_header = top_roc_df.iloc[0]
    # top_roc_df = top_roc_df[1:]
    # top_roc_df.columns = max_sel_header
    # top_roc_df = top_roc_df.sort_values(by='value', ascending=False)
    # top_roc = top_roc_df[~top_roc_df.layer.str.contains('utput')]
    # max_roc = top_roc['value'].to_list()[0]
    #
    # # # ap
    # top_ap_df = pd.DataFrame(highlights_dict['ave_prec'])
    # max_sel_header = top_ap_df.iloc[0]
    # top_ap_df = top_ap_df[1:]
    # top_ap_df.columns = max_sel_header
    # top_ap_df = top_ap_df.sort_values(by='value', ascending=False)
    # top_ap = top_ap_df[~top_ap_df.layer.str.contains('utput')]
    # max_ap = top_ap['value'].to_list()[0]
    #
    # # # informedness
    # top_info_df = pd.DataFrame(highlights_dict['max_informed'])
    # max_sel_header = top_info_df.iloc[0]
    # top_info_df = top_info_df[1:]
    # top_info_df.columns = max_sel_header
    # top_info_df = top_info_df.sort_values(by='value', ascending=False)
    # top_info = top_info_df[~top_info_df.layer.str.contains('utput')]
    # max_info = top_info['value'].to_list()[0]
    #
    # # # selectiviy summary
    # run = gha_dict['topic_info']['run']
    # if test_run:
    #     run = 'test'
    #
    # sel_csv_info = [gha_dict['topic_info']['cond'], run, output_filename,
    #                 gha_dict['data_info']['dataset'], gha_dict['GHA_info']['use_dataset'],
    #                 n_layers,
    #                 gha_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable'],
    #                 gha_dict['GHA_info']['scores_dict']['gha_acc'],
    #                 last_hid_layer,
    #                 # mean_roc,
    #                 max_roc,
    #                 # mean_ap,
    #                 max_ap,
    #                 # mean_info,
    #                 max_info]
    #
    # summary_headers = ["cond", "run", "output_filename", "dataset", "use_dataset",
    #                    "n_layers", "hid_units",
    #                    "gha_acc",
    #                    "last_layer",
    #                    # "mean_roc",
    #                    "max_roc",
    #                    # "mean_ap",
    #                    "max_ap",
    #                    # "mean_info",
    #                    "max_info"]
    #
    # exp_path, cond_name = os.path.split(gha_dict['GHA_info']['gha_path'])
    # exp_name = gha_dict['topic_info']['exp_name']
    #
    # os.chdir(exp_path)
    # print(f"exp_path: {exp_path}")
    #
    # if not os.path.isfile(exp_name + "_sel_summary.csv"):
    #     sel_summary = open(exp_name + "_sel_summary.csv", 'w')
    #     mywriter = csv.writer(sel_summary)
    #     mywriter.writerow(summary_headers)
    #     print(f"creating summary csv at: {exp_path}")
    # else:
    #     sel_summary = open(exp_name + "_sel_summary.csv", 'a')
    #     mywriter = csv.writer(sel_summary)
    #     print(f"appending to summary csv at: {exp_path}")
    #
    # mywriter.writerow(sel_csv_info)
    # sel_summary.close()
    #
    # print("\nSanity check for shelve")
    # print(sel_per_unit_db_name)
    # with shelve.open(sel_per_unit_db_name, flag='r') as db:
    #     sel_p_unit_keys = list(db.keys())
    #     print(sel_p_unit_keys)
    #     print(db['fc1'].keys())
    #
    # print("\nend of sel script")
    #
    # return master_dict  # , mean_sel_per_NN

# # #
# print("\nWARNING\n\nrunning test from bottom of sel script\nWARNING")
# gha_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
#                 'VGG_end_aug/all_val_set_gha/vgg_imagenet_GHA_dict.pickle'
#
# # '''sel'''
# sel_dict = ff_sel(gha_dict_path=gha_dict_path, correct_items_only=True, all_classes=True,
#                   test_run=True,
#                   verbose=True)
#
# sel_dict_path = sel_dict['sel_info']['sel_path']
# print("sel_dict_path: ", sel_dict_path)

