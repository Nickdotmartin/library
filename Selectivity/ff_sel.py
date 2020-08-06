import os
import datetime
import copy
import csv
import json

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc
from scipy.stats.stats import pearsonr

from tools.dicts import load_dict, focussed_dict_print
from tools.data import nick_read_csv


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
    if verbose:
        print("\n**** nick_roc_stuff() ****")

    # if class is not empty (no correct items)
    if class_a_size * not_a_size > 0:

        if verbose:
            print(f"this_class: {this_class}, class_a_size: {class_a_size}, not_a_size: {not_a_size}")

        # convert class list to binary one vs all
        if act_func is 'tanh':
            binary_array = [1 if i == this_class else -1 for i in np.array(class_list)]
        else:
            binary_array = [1 if i == this_class else 0 for i in np.array(class_list)]
        hid_act_array = np.array(hid_acts)

        # # get ROC curve
        fpr, tpr, thr = roc_curve(binary_array, hid_act_array)

        # # Use ROC dict stuff to compute all other needed vectors
        tp_count_dict = [class_a_size * i for i in tpr]
        fp_count_dict = [not_a_size * i for i in fpr]
        abv_thr_count_dict = [x + y for x, y in zip(tp_count_dict, fp_count_dict)]
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
        if verbose:
            print(f"\nROC\nno items in class {this_class} (n={class_a_size}) "
                  f"or not_a (n={not_a_size})")
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
    "we can use the correlation between the activation of unit i and the predicted probability for class k as
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

    if verbose:
        print("\n**** class_sel_basics() ****")

    act_values = 'activation'
    if act_func is 'relu':
        act_values = 'normed'
    if act_func is 'sigmoid':
        hi_val_thr = .75

    if not n_classes:
        n_classes = max(list(items_per_cat.keys()))

    if type(n_classes) is int:
        class_list = list(range(n_classes))
    elif type(n_classes) is list:
        class_list = n_classes
        n_classes = len(class_list)

    # # means
    means_dict = dict(this_unit_acts_df.groupby('class')[act_values].mean())

    # # sd.  will give value of nan if there is no variance rather then 0.
    sd_dict = dict(this_unit_acts_df.groupby('class')[act_values].std())
    sd_dict = {k: v if not np.isnan(v) else 0 for k, v in sd_dict.items()}


    # print(f"\nidiot check sd\n"
    #       f"means: {len(means_dict.values())} {means_dict}\n"
    #       f"sd: {len(sd_dict.values())} {sd_dict}\n"
    #       f"items_per_cat: {items_per_cat}\n"
    #       f"this_unit_acts_df:\n{this_unit_acts_df}")
    # for cat in range(n_classes):
    #     print(f"\ncat: {cat}\n"
    #           f"{this_unit_acts_df[this_unit_acts_df['class'] == cat]}")
    # # if all(list(sd_dict.values())) == 'Nan'

    # # non-zero_count
    nz_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values]
                                           > 0.0].groupby('class')[act_values].count())

    if len(list(nz_count_dict.keys())) < n_classes:
        for i in class_list:
            if i not in nz_count_dict:
                nz_count_dict[i] = 0

    # nz_perplexity = sum(1 for i in nz_count_dict.values() if i >= 0)
    non_zero_count_total = this_unit_acts_df[this_unit_acts_df[act_values] > 0][act_values].count()

    # # non-zero prop
    nz_prop_dict = {k: (0 if items_per_cat[k] == 0 else nz_count_dict[k] / items_per_cat[k])
                    for k in items_per_cat.keys() & nz_count_dict}

    # # non_zero precision
    # nz_prec_dict = {k: v / non_zero_count_total for k, v in nz_count_dict.items()}
    # # changed on 13032020 to not divide by zero.
    nz_prec_dict = {k: (0 if v == 0 else v / non_zero_count_total) for k, v in nz_count_dict.items()}

    # # hi val count
    hi_val_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values] >
                                               hi_val_thr].groupby('class')[act_values].count())
    if len(list(hi_val_count_dict.keys())) < n_classes:
        for i in class_list:
            if i not in hi_val_count_dict:
                hi_val_count_dict[i] = 0

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

    copy_dict = copy.copy(class_sel_basics_dict)
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

    if verbose:
        print("\n**** sel_unit_max() ****")
    # is it necessary to copy this - i think so, script failed when I took copy statements out?
    # copy_sel_dict = copy.deepcopy(all_sel_dict)
    copy_sel_dict = copy.copy(all_sel_dict)
    # copy_sel_dict = all_sel_dict

    # focussed_dict_print(copy_sel_dict, 'copy_sel_dict')

    max_sel_dict = dict()

    # # loop through unit dict of sel measure vals for each class
    for measure, class_dict in copy_sel_dict.items():

        # # remove np.NaNs from dict
        clean_dict = {k: class_dict[k] for k in class_dict if not np.isnan(class_dict[k])}


        # # for each sel measure get list of sel values and classes
        measure_c_name = f"{measure}_c"
        classes = list(clean_dict.keys())
        values = list(clean_dict.values())

        # print("\nidiot check\n"
        #       f"measure_c_name: {measure_c_name}\n"
        #       f"classes:{classes}\n"
        #       f"values:{values}"
        #       )

        # # for each sel measure get max value and class
        max_val = max(values)
        max_class = classes[values.index(max_val)]
        # print(measure, measure_c_name)

        # # copy max class and value to max_class_dict
        max_sel_dict[measure] = max_val
        max_sel_dict[measure_c_name] = max_class


    # # remove unnecessary variables rom the dict
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

    max_sel_dict['b_sel_off'] = copy_sel_dict['b_sel_off'][max_sel_dict["b_sel_c"]]
    del max_sel_dict['b_sel_off_c']
    max_sel_dict['b_sel_zero'] = copy_sel_dict['b_sel_zero'][max_sel_dict["b_sel_c"]]
    del max_sel_dict['b_sel_zero_c']
    max_sel_dict['b_sel_pfive'] = copy_sel_dict['b_sel_pfive'][max_sel_dict["b_sel_c"]]
    del max_sel_dict['b_sel_pfive_c']




    # # max corr_coef shold be the absolute max (e.g., including negative) where p < .05.
    # get all values into df
    coef_array = []  # [corr_coef, abs(corr_coef), p, class]
    for coef_k, coef_v in copy_sel_dict['corr_coef'].items():
        abs_coef = abs(coef_v)
        p = copy_sel_dict['corr_p'][coef_k]
        coef_array.append([coef_v, abs_coef, p, coef_k])
    coef_df = pd.DataFrame(data=coef_array, columns=['coef', 'abs', 'p', 'class'])

    # # filter and sort df
    coef_df = coef_df.loc[coef_df['p'] < 0.05]

    if not len(coef_df):  # if there are not items with that p_value
        max_sel_dict['corr_coef'] = float('NaN')
        max_sel_dict['corr_coef_c'] = float('NaN')
        max_sel_dict['corr_p'] = float('NaN')
    else:
        coef_df = coef_df.sort_values(by=['abs'], ascending=False).reset_index()
        max_sel_dict['corr_coef'] = coef_df['coef'].iloc[0]
        max_sel_dict['corr_coef_c'] = coef_df['class'].iloc[0]
        max_sel_dict['corr_p'] = coef_df['p'].iloc[0]

    del max_sel_dict['corr_p_c']

    # # round values
    for k, v in max_sel_dict.items():
        if v is 'flaot':
            max_sel_dict[k] = round(v, 3)

    # print("\n\n\n\nmax sel dict", max_sel_dict)
    # focussed_dict_print(max_sel_dict, 'max_sel_dict')

    return max_sel_dict


#######################################################################################################
def ff_sel(gha_dict_path, correct_items_only=True, all_classes=True,
           layer_classes=("Conv2D", "Dense", "Activation"),
           verbose=False, test_run=False):
    """
    Analyse hidden unit activations.
    1. load dict from study (GHA dict)
    2. from dict get x, y, num of items, IPC, hidden activations etc
    3. load hidden activations.  Check that items were correct - if not - use correct_y-labels
    4. loop through all layers:
        loop through all units:
            loop through all classes:
                run selectivity measures : ccma, zn_AP, pr(full), zhou_prec, top_class, informedness
                save dicts with values for all measures
                save dict with top class/value for all measures

    :param gha_dict_path: path of the gha dict
    :param correct_items_only: Whether selectivity considered incorrect items
    :param all_classes: Whether to test for selectivity of all classes or a subset (e.g., most active classes)
    :param layer_classes: Which layers to analyse
    :param verbose: how much to print to screen
    :param test_run: if True, only do subset, e.g., 3 units from 3 layers

    :return: master dict: contains 'sel_path' (e.g., dir), 'sel_per_unit_pickle_name', 'sel_highlights_pickle_name',

        :return: sel_p_unit_dict. dict with all selectivity results in it.
            all sel values:                 [layer names][unit number][sel measure][class]
            max_sel_p_u values(and class):  [layer names][unit number]['max'][sel measure]
            layer means:                    [layer name]['means'][measure]


        :return: sel_highlights: keys are sel_measures.  Values are DataFrames with the 10 highest scoring units for
                                each sel measure.

    """

    print("\n**** running ff_sel() ****")

    exp_cond_gha_path, gha_dict_name = os.path.split(gha_dict_path)
    os.chdir(exp_cond_gha_path)
    print(f"os.getcwd(): {os.getcwd()}")

    # # part 1. load dict from study (should run with sim, GHA or sel dict)
    gha_dict = load_dict(gha_dict_path)
    focussed_dict_print(gha_dict, 'gha_dict')

    # # load item_correct (y_data)
    item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
    # y_scores_df = pd.read_csv(item_correct_name)
    y_scores_df = nick_read_csv(item_correct_name)
    # use y_df for analysis
    y_df = y_scores_df

    hid_acts_pickle = gha_dict["GHA_info"]["hid_act_files"]['2d']
    n_cats = gha_dict["data_info"]["n_cats"]
    all_items = gha_dict['GHA_info']['scores_dict']['n_items']
    n_correct = gha_dict['GHA_info']['scores_dict']['n_correct']
    n_incorrect = all_items - n_correct
    items_per_cat = gha_dict['GHA_info']['scores_dict']['corr_per_cat_dict']
    trained_for = gha_dict['training_info']['trained_for']
    act_func = gha_dict['model_info']['overview']['act_func']

    # # get basic COI list
    if all_classes is True:
        if type(items_per_cat) is int:
            classes_of_interest = list(range(n_cats))
        else:
            classes_of_interest = list({k: v for (k, v) in items_per_cat.items() if v > 0})
        print(f"classes_of_interest (all classes with correct items): {classes_of_interest}")

    # # # get values for correct/incorrect items (1/0 or True/False)
    full_model_values = y_scores_df.full_model.unique()
    if len(full_model_values) != 2:
        print(f"\nTYPE_ERROR!: what are the scores/acc for items? {full_model_values}")
        print(f"Were there any incorrect responses? n_incorrect = {n_incorrect}")
        print(f"type(n_incorrect): {type(n_incorrect)}")
        if n_incorrect == 0:
            print("\nthere are no incorrect items so all responses are correct")
            correct_item = full_model_values[0]
        else:
            print("not sure what signifies correct items")
    else:
        correct_item = 1
    if 1 not in full_model_values:
        correct_item = True

    # # i need to check whether this analysis should include incorrect items
    # therefore only have correct items so remove all incorrect responses from y_correct df
    gha_incorrect = gha_dict['GHA_info']['gha_incorrect']

    if gha_incorrect:
        if correct_items_only:
            print("\ngha_incorrect: True (I have incorrect responses)\n"
                  "correct_items_only: True (I only want correct responses)")
            print(f"remove {n_incorrect} incorrect from hid_acts & output using y_scores_df.")
            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_item]
            y_df = y_correct_df
            print("use y_correct for y_df")
        else:
            print("\ngha_incorrect: True (I have incorrect responses)\n"
                  "correct_items_only: False (I want incorrect responses)")
            print("no changes needed - don't remove anything from hid_acts, output and use y scores as y_df")
    else:
        if correct_items_only:
            print("\ngha_incorrect: False (I only have correct responses)\n"
                  "correct_items_only: True (I only want correct responses)")
            print("no changes needed - don't remove anything from hid_acts or output.  Use y_correct as y_df")
            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_item]
            y_df = y_correct_df
        else:
            print("\ngha_incorrect: False (I only have correct responses)\n"
                  "correct_items_only: False (I want incorrect responses)")
            print("I can not complete this as desried"
                  "change correct_items_only to True"
                  "for analysis  - don't remove anything from hid_acts, output and use y scores as y_df")
            correct_items_only = True

    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")

    # Output files
    output_filename = gha_dict["topic_info"]["output_filename"]
    print(f"\noutput_filename: {output_filename}")

    # # open hid_acts.pickle
    # todo: move this.  open hid_acts pickle for every layer/unit for imageNet.
    '''
    1. get layer number for last layer
    2. get output layer activations (these are held in memory, but could be loaded for each unit)
    3. * add get list of hid act dict keys - loop thought this rather than actual dict.
    '''

    print("\n**** opening hid_acts.pickle ****")
    with open(hid_acts_pickle, 'rb') as pkl:
        hid_acts_dict = pickle.load(pkl)

    hid_acts_keys_list = list(hid_acts_dict.keys())

    # # get output activations for class correlation
    print(f"hid_acts_pickle: {hid_acts_pickle}\n"
          f"hid_acts_dict: {hid_acts_dict}\n"
          f"hid_acts_dict.keys(): {hid_acts_dict.keys()}")

    # last_layer_num = list(hid_acts_dict.keys())[-1]
    last_layer_num = hid_acts_keys_list[-1]

    print(f"last_layer_num: {last_layer_num}")
    output_layer_acts = hid_acts_dict[last_layer_num]['2d_acts']
    output_layer_df = pd.DataFrame(output_layer_acts)
    if correct_items_only:
        if gha_incorrect:
            output_layer_df['full_model'] = y_scores_df['full_model']
            output_layer_df = output_layer_df.loc[output_layer_df['full_model'] == 1]
            output_layer_df = output_layer_df.drop(['full_model'], axis=1)
            print(f"\nremoving {n_incorrect} incorrect responses from output_layer_df: {output_layer_df.shape}\n")
    # print(f"==> output_layer_df.head(): {output_layer_df.head()}")

    # # close hid act dict to save memory space?
    hid_acts_dict = dict()

    # # where to save files
    current_wd = os.getcwd()
    print(f"current wd: {current_wd}")

    analyse_items = 'all'
    if correct_items_only:
        analyse_items = 'correct'
    sel_folder = f'{analyse_items}_sel'
    if test_run:
        sel_folder = f'{analyse_items}_sel/test'
    sel_path = os.path.join(current_wd, sel_folder)

    if not os.path.exists(sel_path):
        os.makedirs(sel_path)

    # # sel_p_unit_dict
    sel_p_unit_dict = dict()

    layer_sel_mean_dict = dict()

    highlights_dict = {"roc_auc": [['value', 'class', 'layer', 'unit']],

                       "ave_prec": [['value', 'class', 'layer', 'unit']],

                       "pr_auc": [['value', 'class', 'layer', 'unit']],

                       # "nz_ave_prec": [['value', 'class', 'layer', 'unit']],
                       #
                       # "nz_pr_auc": [['value', 'class', 'layer', 'unit']],

                       # "tcs_recall": [['value', 'class', 'thr', 'items', 'layer', 'unit']],

                       "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
                                         'prec', 'layer', 'unit']],

                       "ccma": [['value', 'class', 'layer', 'unit']],

                       "b_sel": [['value', 'class', 'layer', 'unit']],

                       "zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],

                       "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],

                       "means": [['value', 'class', 'sd', 'layer', 'unit']],

                       "nz_count": [['value', 'class', 'layer', 'unit']],

                       "nz_prop": [['value', 'class', 'layer', 'unit']],

                       "nz_prec": [['value', 'class', 'layer', 'unit']],

                       "hi_val_count": [['value', 'class', 'layer', 'unit']],

                       "hi_val_prop": [['value', 'class', 'layer', 'unit']],

                       "hi_val_prec": [['value', 'class', 'layer', 'unit']],

                       # 'all_act_mean': [['value', 'layer', 'unit']],
                       }

    # # loop through dict/layers
    n_layers = len(gha_dict['GHA_info']['model_dict'])

    if test_run:
        layer_counter = 0

    # for key, value in hid_acts_dict.items():
    #     layer_dict = value

    for layer_number in hid_acts_keys_list:
        print("\n**** opening hid_acts.pickle ****")
        with open(hid_acts_pickle, 'rb') as pkl:
            hid_acts_dict = pickle.load(pkl)

        # layer_dict = copy.copy(hid_acts_dict[layer_number])
        layer_dict = hid_acts_dict[layer_number]

        layer_act_list = []

        # # close hid act dict to save memory space?
        hid_acts_dict = dict()

        if layer_dict['layer_class'] not in layer_classes:
            continue  # skip this layer

        if test_run is True:
            layer_counter = layer_counter + 1
            if layer_counter > 3:
                continue

        # layer_number = key
        layer_name = layer_dict['layer_name']

        # if layer_name in ['Output', 'output', 'OUTPUT']:
        #     print("Not analysing output layer - continue")
        #     continue

        hid_acts_array = layer_dict['2d_acts']
        hid_acts_df = pd.DataFrame(hid_acts_array)
        print(f"\nloaded hidden_activation_file: {hid_acts_pickle}, {np.shape(hid_acts_df)}")
        units_per_layer = len(hid_acts_df.columns)

        # # remove incorrect responses
        if correct_items_only:
            if gha_incorrect:
                print(f"\nremoving {n_incorrect} incorrect responses from hid_acts_df: {hid_acts_df.shape}")
                hid_acts_df['full_model'] = y_scores_df['full_model']
                hid_acts_df = hid_acts_df.loc[hid_acts_df['full_model'] == 1]
                hid_acts_df = hid_acts_df.drop(['full_model'], axis=1)
                print(f"(cleaned) hid_acts_df: {hid_acts_df.shape}\n{hid_acts_df.head()}")

        layer_dict = dict()
        max_sel_dict = dict()

        print("\n**** loop through units ****")
        for unit_index, unit in enumerate(hid_acts_df.columns):

            if test_run is True:
                if unit_index > 3:
                    continue

            print(f"\n*************\nrunning layer {layer_number} of {n_layers} ({layer_name}): "
                  f"unit {unit} of {units_per_layer}\n************")

            unit_dict = {'roc_auc': {},
                         'ave_prec': {},
                         'pr_auc': {},
                         # 'nz_ave_prec': {},
                         # 'nz_pr_auc': {},
                         # 'tcs_items': {},
                         # 'tcs_thr': {},
                         # 'tcs_recall': {},
                         'max_informed': {},
                         'max_info_count': {},
                         'max_info_thr': {},
                         'max_info_sens': {},
                         'max_info_spec': {},
                         'max_info_prec': {},
                         # 'max_info_f1': {},
                         'ccma': {},
                         'b_sel': {},
                         'b_sel_off': {},
                         'b_sel_zero': {},
                         'b_sel_pfive': {},
                         'zhou_prec': {},
                         'zhou_selects': {},
                         'zhou_thr': {},
                         'corr_coef': {},
                         'corr_p': {},
                         # 'all_act_mean': {},
                         # 'all_act_sd': {},

                         }

            this_unit_just_acts = list(hid_acts_df.loc[:, unit])

            # # 1st check its not a dead unit
            # if this_unit_just_acts.sum() != 0:  # check for dead units, if dead, all details to 0/na/nan/-999 etc
            if sum(this_unit_just_acts) != 0:  # check for dead units, if dead, all details to 0/na/nan/-999 etc

                print("not dead - sorting activations for analysis")

                # insert act values in middle of labels (item, act, cat)
                this_unit_acts = y_df.copy()

                this_unit_acts.insert(2, column='activation', value=this_unit_just_acts, )

                # # sort by descending hid act values
                this_unit_acts_df = this_unit_acts.sort_values(by='activation', ascending=False)

                # # # normalize activations
                just_act_values = this_unit_acts_df['activation'].tolist()
                max_act = max(just_act_values)
                normed_acts = np.true_divide(just_act_values, max_act)
                this_unit_acts_df.insert(2, column='normed', value=normed_acts)

                # # get overall unit mean activation (not class specific)
                if act_func is 'sigmoid':
                    unit_mean_act = np.mean(just_act_values)
                else:
                    unit_mean_act = np.mean(normed_acts)

                layer_act_list.append(unit_mean_act)


                if verbose is True:
                    print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n{this_unit_acts_df.head()}")

                # # get class_sel_basics
                class_sel_basics_dict = class_sel_basics(this_unit_acts_df,
                                                         items_per_cat,
                                                         n_classes=n_cats,
                                                         verbose=verbose)

                if verbose:
                    focussed_dict_print(class_sel_basics_dict, 'class_sel_basics_dict')

                # # add class_sel_basics_dict to unit dict
                for csb_key, csb_value in class_sel_basics_dict.items():
                    if csb_key == 'total':
                        continue
                    if csb_key == 'perplexity':
                        continue
                    unit_dict[csb_key] = csb_value

                classes_of_interest = list(range(n_cats))
                if all_classes is False:
                    # # I don't want to use all classes, just ones that are worth testing
                    classes_of_interest = coi_list(class_sel_basics_dict, verbose=verbose)

                print('\n**** cycle through classes ****')
                for this_cat in range(len(classes_of_interest)):

                    this_class_size = items_per_cat[this_cat]
                    not_a_size = n_correct - this_class_size

                    if verbose is True:
                        print(f"\nclass_{this_cat}: {this_class_size} items, not_{this_cat}: {not_a_size} items")

                    # # running selectivity measures

                    # # ROC_stuff includes:
                    # roc_auc, ave_prec, pr_auc, nz_ave_prec, nz_pr_auc, top_class_sel, informedness
                    roc_stuff_dict = nick_roc_stuff(class_list=this_unit_acts_df['class'],
                                                    hid_acts=this_unit_acts_df['normed'],
                                                    this_class=this_cat,
                                                    class_a_size=this_class_size, not_a_size=not_a_size,
                                                    verbose=verbose)

                    # # add class_sel_basics_dict to unit dict
                    for roc_key, roc_value in roc_stuff_dict.items():
                        unit_dict[roc_key][this_cat] = roc_value

                    # # ccma
                    class_a = this_unit_acts_df.loc[this_unit_acts_df['class'] == this_cat]
                    class_a_mean = class_a['normed'].mean()
                    not_class_a = this_unit_acts_df.loc[this_unit_acts_df['class'] != this_cat]
                    not_class_a_mean = not_class_a['normed'].mean()
                    ccma = (class_a_mean - not_class_a_mean) / (class_a_mean + not_class_a_mean)
                    unit_dict["ccma"][this_cat] = ccma



                    # # Bowers sel
                    '''
                    test for sel on and off units and give the max.  add variable for b_sel_off
                    '''
                    if verbose:
                        print("\nBowers Sel")

                    # # first check for on units
                    class_a_min = class_a['activation'].min()
                    class_a_max = class_a['activation'].max()
                    not_class_a_max = not_class_a['activation'].max()
                    not_class_a_min = not_class_a['activation'].min()

                    if act_func in ['tanh', 'relu', 'ReLu']:
                        class_a_min = class_a['normed'].min()
                        class_a_max = class_a['normed'].max()
                        not_class_a_max = not_class_a['normed'].max()
                        not_class_a_min = not_class_a['normed'].min()

                    if verbose:
                        print(f"class_a_min: {class_a_min}\n"
                              f"class_a_max: {class_a_max}\n"
                              f"not_class_a_min: {not_class_a_min}\n"
                              f"not_class_a_max: {not_class_a_max}\n")

                    b_sel_on = class_a_min - not_class_a_max
                    b_sel_off = not_class_a_min - class_a_max

                    print(f'\nb_sel_on = class_a_min: {class_a_min} - '
                          f'not_class_a_max: {not_class_a_max} = {b_sel_on}\n'
                          f'\nb_sel_off = not_class_a_min: {not_class_a_min} - '
                          f'class_a_max: {class_a_max} = {b_sel_off}\n')

                    if b_sel_on >= b_sel_off:
                        b_sel = b_sel_on
                        off_unit = 0
                        if verbose:
                            print(f"\nb_sel ON\n"
                                  f"class_a_min: {class_a_min} - not_class_a_max: {not_class_a_max}\n"
                                  f"b_sel: {b_sel}")
                    else:
                        b_sel = b_sel_off
                        off_unit = 1
                        if verbose:
                            print(f"\nb_sel OFF\n"
                                  f"not_class_a_min: {not_class_a_min} - class_a_max: {class_a_max}\n"
                                  f"b_sel: {b_sel}")

                    b_sel_zero = 0
                    b_sel_pfive = 0
                    if b_sel >= 0.0:
                        b_sel_zero = 1
                    if b_sel >= .5:
                        b_sel_pfive = 1

                    unit_dict["b_sel"][this_cat] = b_sel
                    unit_dict["b_sel_off"][this_cat] = off_unit
                    unit_dict["b_sel_zero"][this_cat] = b_sel_zero
                    unit_dict["b_sel_pfive"][this_cat] = b_sel_pfive



                    # # zhou_prec
                    zhou_cut_off = .005
                    if n_correct < 20000:
                        zhou_cut_off = 100 / n_correct
                    zhou_selects = int(n_correct * zhou_cut_off)
                    most_active = this_unit_acts_df.iloc[:zhou_selects]
                    zhou_thr = list(most_active["normed"])[-1]
                    zhou_prec = sum([1 for i in most_active['class'] if i == this_cat]) / zhou_selects
                    unit_dict["zhou_prec"][this_cat] = zhou_prec
                    unit_dict["zhou_selects"][this_cat] = zhou_selects
                    unit_dict["zhou_thr"][this_cat] = zhou_thr



                    # # class correlation
                    class_corr = class_correlation(this_unit_acts=this_unit_acts_df['normed'],
                                                   output_acts=output_layer_df[this_cat], verbose=verbose)
                    unit_dict["corr_coef"][this_cat] = class_corr['coef']
                    unit_dict["corr_p"][this_cat] = class_corr['p']

                if verbose is True:
                    focussed_dict_print(unit_dict, 'unit_dict')

                # # # for each sel variable - get the class with the highest values and add to per_unit
                max_sel_p_unit_dict = sel_unit_max(unit_dict, verbose=verbose)
                unit_dict['max'] = max_sel_p_unit_dict
                max_sel_dict[unit] = max_sel_p_unit_dict

                # # check to see if this unit has any thing for highlights

                print(f"Unit {unit} sel summary")
                focussed_dict_print(max_sel_p_unit_dict, f'max_sel_p_unit_dict, unit {unit}')

                # # ADD UNIT DICT TO LAYER
                layer_dict[unit] = unit_dict


                # get mean activation for layer
                layer_act_mean = np.mean(layer_act_list)
                print(f"layer_act_mean: {layer_act_mean}")

                # # calculate layer sd
                print(f"layer_act_list: {layer_act_list}")
                differences_from_mean_act = [x - layer_act_mean for x in layer_act_list]
                print(f"differences_from_mean_act: {differences_from_mean_act}")
                sum_squared_diff = np.sum([np.square(x) for x in differences_from_mean_act])
                print(f"sum_squared_diff: {sum_squared_diff}")
                layer_sd = np.sqrt(sum_squared_diff / len(layer_act_list))
                print(f"layer_sd: {layer_sd}")


            else:
                print("dead unit found")
                unit_dict = 'dead_unit'

        print("********\nfinished looping through units\n************")

        # # make max per unit csv
        max_sel_csv_name = f'{sel_path}/{output_filename}_{layer_name}_sel_p_unit.csv'
        max_sel_df = pd.DataFrame.from_dict(data=max_sel_dict, orient='index')
        max_sel_df.index.rename('unit', inplace=True)
        max_sel_df.to_csv(max_sel_csv_name)
        print("\n\nmax sel df")
        print(max_sel_df.head())
        # nick_to_csv(max_sel_df, max_sel_csv_name)

        # # get layer_sel_mean_dict
        # # for each unit, for each measure, the max_value.
        # # add all these up for the layer.
        # focussed_dict_print(max_sel_dict, 'max_sel_dict')

        if unit_dict != 'dead_unit':
            for measure in unit_dict:
                layer_unit_sel_list = []
                if measure == 'max':
                    continue
                for unit, unit_sel_dict in max_sel_dict.items():
                    layer_unit_sel_list.append(unit_sel_dict[measure])
                layer_measure_mean = np.mean(layer_unit_sel_list)
                layer_sel_mean_dict[measure] = layer_measure_mean
        # focussed_dict_print(layer_sel_mean_dict, 'layer_sel_mean_dict')
        layer_dict['means'] = layer_sel_mean_dict

        # # layer means
        layer_means_list = list(layer_sel_mean_dict.values())
        layer_means_list = layer_means_list + [layer_act_mean, layer_sd]
        layer_means_list = [layer_name, unit_index + 1] + layer_means_list

        layer_means_headers = list(layer_sel_mean_dict.keys())
        layer_means_headers = ['name', 'units'] + layer_means_headers + ['layer_act_mean', 'mead_act_sd']

        already_done_means = False
        if not os.path.isfile(os.path.join(sel_path, f"{output_filename}_layer_means.csv")):
            layer_means_csv = open(os.path.join(sel_path, f"{output_filename}_layer_means.csv"), 'w')
            mywriter = csv.writer(layer_means_csv)
            mywriter.writerow(layer_means_headers)
            mywriter.writerow(layer_means_list)
            layer_means_csv.close()
            print(f"creating layer_means csv at: {output_filename}")
        else:
            check_it = pd.read_csv(os.path.join(sel_path, f"{output_filename}_layer_means.csv"))
            if 'Total' not in check_it['name'].to_list():
                layer_means_csv = open(os.path.join(sel_path, f"{output_filename}_layer_means.csv"), 'a')
                mywriter = csv.writer(layer_means_csv)
                mywriter.writerow(layer_means_list)
                layer_means_csv.close()
                print(f"appending to layer_means csv at: {output_filename}")
            else:
                print("Already have layer means for this cond")
                already_done_means = True

        # # # get top three highlights for each feature
        highlights_list = list(highlights_dict.keys())
        print(f"highlights_list: {highlights_list}")

        if layer_name not in ['output', 'Output', 'OUTPUT']:
            # don't get highlights for output

            for measure in highlights_list:
                # sort max_sel_df by sel measure for this layer
                layer_top_3_df = max_sel_df.sort_values(by=measure, ascending=False).reset_index()

                # remove output layers
                # print(f"layer_top_3_df:\n{layer_top_3_df}")
                # layer_top_3_df = layer_top_3_df[~layer_top_3_df.layer.str.contains('utput')]

                # then select info for highlights dict for top 3 items

                print(f"layer_top_3_df\n{layer_top_3_df}")
                print(f"layer_top_3_df.loc[{measure}]\n{layer_top_3_df[measure]}")

                # if measure == 'tcs_recall':
                #     print('tcs_recall')  # "tcs_recall": [['value', 'class', 'thr', 'items', 'layer', 'unit']],
                #     for i in range(2):
                #         top_val = layer_top_3_df[measure].iloc[i]
                #         if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                #             top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'])[i]
                #             thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_thr'])[i]
                #             items = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_items'])[i]
                #             top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                #         else:
                #             top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].values.item()
                #             thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_thr'].values.item()
                #             items = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_items'].values.item()
                #             top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].values.item()
                #
                #         new_row = [top_val, top_class, thr, items, layer_name, top_unit_name]
                #         print(f"new_row\n{new_row}")
                #         highlights_dict[measure].append(new_row)

                if measure == 'max_informed':
                    print('max_informed')
                    # "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
                    #                                          'prec', 'f1', 'count', 'layer', 'unit']],
                    for i in range(2):
                        top_val = layer_top_3_df[measure].iloc[i]
                        if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                            top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'])[i]
                            count = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_count'])[i]
                            thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_thr'])[i]
                            sens = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_sens'])[i]
                            spec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_spec'])[i]
                            prec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_prec'])[i]
                            # f1 = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_f1'])[i]
                            top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                        else:
                            top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].values.item()
                            count = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_count'].values.item()
                            thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_thr'].values.item()
                            sens = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_sens'].values.item()
                            spec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_spec'].values.item()
                            prec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_prec'].values.item()
                            # f1 = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_f1'].values.item()
                            top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].values.item()

                        new_row = [top_val, top_class, count, thr, sens, spec, prec, layer_name, top_unit_name]
                        print(f"new_row\n{new_row}")
                        highlights_dict[measure].append(new_row)

                elif measure == 'zhou_prec':
                    print('zhou_prec')
                    # "zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],
                    for i in range(2):
                        top_val = layer_top_3_df[measure].iloc[i]
                        if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                            top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'])[i]
                            thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'zhou_thr'])[i]
                            selects = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'zhou_selects'])[i]
                            top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                        else:
                            top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].values.item()
                            thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'zhou_thr'].values.item()
                            selects = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'zhou_selects'].values.item()
                            top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].values.item()

                        new_row = [top_val, top_class, thr, selects, layer_name, top_unit_name]
                        print(f"new_row\n{new_row}")
                        highlights_dict[measure].append(new_row)

                elif measure == 'corr_coef':
                    print('corr_coef')
                    #   "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],
                    for i in range(2):
                        top_val = layer_top_3_df[measure].iloc[i]
                        if np.isnan(top_val):
                            continue
                        if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                            top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'])[i]
                            p = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'corr_p'])[i]
                            top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                        else:
                            top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].values.item()
                            p = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'corr_p'].values.item()
                            top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].values.item()

                        new_row = [top_val, top_class, p, layer_name, top_unit_name]
                        print(f"new_row\n{new_row}")
                        highlights_dict[measure].append(new_row)

                elif measure == 'means':
                    print('means')  # "means": [['value', 'class', 'sd', 'layer', 'unit']],
                    for i in range(2):
                        top_val = layer_top_3_df[measure].iloc[i]
                        if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                            top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'])[i]
                            sd = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'sd'])[i]
                            top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                        else:
                            top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].values.item()
                            sd = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'sd'].values.item()
                            top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].values.item()

                        new_row = [top_val, top_class, sd, layer_name, top_unit_name]
                        print(f"new_row\n{new_row}")
                        highlights_dict[measure].append(new_row)

                else:  # for most most measures use below
                    for i in range(2):
                        top_val = layer_top_3_df[measure].iloc[i]
                        print('top_val: ', top_val)
                        print(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])
                        if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                            top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'])[i]
                            top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                        elif len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) < 1:
                            top_class = np.nan
                            top_unit_name = np.nan
                        else:
                            top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].values[0]
                            top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].values[0]

                        new_row = [top_val, top_class, layer_name, top_unit_name]

                        print(f"new_row\n{new_row}")
                        highlights_dict[measure].append(new_row)

                print(f"\nhighlights_dict[{measure}]\n{highlights_dict[measure]}")

            print(f"\nEND highlights_dict\n{highlights_dict}")

        # # append layer dict to model dict
        sel_p_unit_dict[layer_name] = layer_dict

        # print(f"layer_sel_mean_dict:\n{layer_sel_mean_dict}")

    # # add means total
    lm_path = os.path.join(sel_path, f"{output_filename}_layer_means.csv")
    lm = pd.read_csv(lm_path, index_col='name')
    if not already_done_means:
        print("appending total to layer_means csv")
        total_means = []

        print(f"\nidiot check\nlm:\n{lm}")

        for column_name, column_data in lm.iteritems():
            print(f"\ncolumn_name: {column_name}")
            if column_name == 'name':
                print("adding 'Total' to name column")
                total_means.append('Total')
            # elif column_data['name'] == 'name':
            #     print("this is the name column")
            elif column_name in ['units', 'unit', 'Units', 'Unit']:
                sum_c = column_data.sum()
                total_means.append(sum_c)
            else:
                mean = column_data.mean()
                total_means.append(mean)

        totals_s = pd.Series(data=total_means, index=lm.columns, name='Total')
        lm = lm.append(totals_s)
        lm.to_csv(lm_path)

    highlights_df_dict = dict()
    for k, v in highlights_dict.items():
        hl_df = pd.DataFrame(data=v[1:], columns=v[0])
        hl_df.sort_values(by='value', ascending=False, inplace=True)
        highlights_df_dict[k] = hl_df
        if verbose:
            print(f"\n{k} highlights (head)")
            print(hl_df)

    sel_highlights_pickle_name = f"{sel_path}/{output_filename}_highlights.pickle"
    pickle_out = open(sel_highlights_pickle_name, "wb")
    pickle.dump(highlights_df_dict, pickle_out)
    pickle_out.close()

    sel_per_unit_pickle_name = f"{sel_path}/{output_filename}_sel_per_unit.pickle"
    pickle_out = open(sel_per_unit_pickle_name, "wb")
    pickle.dump(sel_p_unit_dict, pickle_out)
    pickle_out.close()

    # # save dict
    print("\n\n\n*****************\nanalysis complete\n*****************")

    master_dict = dict()
    master_dict["topic_info"] = gha_dict['topic_info']
    master_dict["data_info"] = gha_dict['data_info']
    master_dict["model_info"] = gha_dict['model_info']
    master_dict["training_info"] = gha_dict['training_info']
    master_dict["GHA_info"] = gha_dict['GHA_info']

    sel_dict_name = f"{sel_path}/{output_filename}_sel_dict.pickle"

    master_dict["sel_info"] = {"sel_path": sel_path,
                               'sel_dict_name': sel_dict_name,
                               "sel_per_unit_pickle_name": sel_per_unit_pickle_name,
                               'sel_highlights_pickle_name': sel_highlights_pickle_name,
                               "correct_items_only": correct_items_only,
                               "all_classes": all_classes, "layer_classes": layer_classes,
                               "sel_date": int(datetime.datetime.now().strftime("%y%m%d")),
                               "sel_time": int(datetime.datetime.now().strftime("%H%M")),
                               }

    print(f"Saving dict to: {os.getcwd()}")
    pickle_out = open(sel_dict_name, "wb")
    pickle.dump(master_dict, pickle_out)
    pickle_out.close()

    focussed_dict_print(master_dict, "master_dict")
    # print_nested_round_floats(master_dict)

    # # save summary csv
    key_layers_list = [x for x in gha_dict['GHA_info']['gha_key_layers'] if 'output' not in str.lower(x)]
    last_hid_layer = key_layers_list[-1]

    print(f"\njust get means of hidden layers\nlm:\n{lm}")
    # # just get means of hidden layers
    # hid_layer_means = lm[~lm.index.str.contains('utput')]
    # hid_layer_means = hid_layer_means[~hid_layer_means.index.str.contains('Total')]
    hid_layer_means = lm.drop(['output', 'Total'])
    print(f"hid_layer_means:\n{hid_layer_means}")

    # mean_roc = lm.loc['Total', 'roc_auc']
    # mean_ap = lm.loc['Total', 'ave_prec']
    # mean_info = lm.loc['Total', 'max_informed']
    # mean_CCMA = lm.loc['Total', 'ccma']
    # mean_bsel = lm.loc['Total', 'b_sel']
    # mean_bsel_off = lm.loc['Total', 'b_sel_off']
    mean_roc = hid_layer_means["roc_auc"].mean()
    mean_ap = hid_layer_means["ave_prec"].mean()
    mean_info = hid_layer_means["max_informed"].mean()
    mean_CCMA = hid_layer_means["ccma"].mean()
    mean_bsel = hid_layer_means["b_sel"].mean()
    mean_nz_prop = hid_layer_means["nz_prop"].mean()
    mean_hi_val_prop = hid_layer_means["hi_val_prop"].mean()
    mean_bsel_zero = hid_layer_means["b_sel_zero"].mean()
    mean_bsel_pfive = hid_layer_means["b_sel_pfive"].mean()
    mean_act = hid_layer_means["layer_act_mean"].mean()
    mead_act_sd = hid_layer_means["mead_act_sd"].mean()




    print("\ngetting max sel values from highlights")
    # # roc
    top_roc_df = pd.DataFrame(highlights_dict['roc_auc'])
    max_sel_header = top_roc_df.iloc[0]
    top_roc_df = top_roc_df[1:]
    top_roc_df.columns = max_sel_header
    top_roc_df = top_roc_df.sort_values(by='value', ascending=False)
    top_roc = top_roc_df[~top_roc_df.layer.str.contains('utput')]
    max_roc = top_roc['value'].to_list()[0]

    # # ap
    top_ap_df = pd.DataFrame(highlights_dict['ave_prec'])
    max_sel_header = top_ap_df.iloc[0]
    top_ap_df = top_ap_df[1:]
    top_ap_df.columns = max_sel_header
    top_ap_df = top_ap_df.sort_values(by='value', ascending=False)
    top_ap = top_ap_df[~top_ap_df.layer.str.contains('utput')]
    max_ap = top_ap['value'].to_list()[0]

    # # informedness
    top_info_df = pd.DataFrame(highlights_dict['max_informed'])
    max_sel_header = top_info_df.iloc[0]
    top_info_df = top_info_df[1:]
    top_info_df.columns = max_sel_header
    top_info_df = top_info_df.sort_values(by='value', ascending=False)
    top_info = top_info_df[~top_info_df.layer.str.contains('utput')]
    max_info = top_info['value'].to_list()[0]

    # # ccma
    top_ccma_df = pd.DataFrame(highlights_dict['ccma'])
    max_sel_header = top_ccma_df.iloc[0]
    top_ccma_df = top_ccma_df[1:]
    top_ccma_df.columns = max_sel_header
    top_ccma_df = top_ccma_df.sort_values(by='value', ascending=False)
    top_ccma = top_ccma_df[~top_ccma_df.layer.str.contains('utput')]
    max_CCMA = top_ccma['value'].to_list()[0]

    # # b_sel
    top_bsel_df = pd.DataFrame(highlights_dict['b_sel'])
    max_sel_header = top_bsel_df.iloc[0]
    top_bsel_df = top_bsel_df[1:]
    top_bsel_df.columns = max_sel_header
    top_bsel_df = top_bsel_df.sort_values(by='value', ascending=False)
    top_bsel = top_bsel_df[~top_bsel_df.layer.str.contains('utput')]
    max_bsel = top_bsel['value'].to_list()[0]

    # # # b_sel_off
    # top_bsel_off_df = pd.DataFrame(highlights_dict['b_sel_off'])
    # max_sel_header = top_bsel_off_df.iloc[0]
    # top_bsel_off_df = top_bsel_off_df[1:]
    # top_bsel_off_df.columns = max_sel_header
    # top_bsel_off_df = top_bsel_off_df.sort_values(by='value', ascending=False)
    # top_bsel_off = top_bsel_off_df[~top_bsel_off_df.layer.str.contains('utput')]
    # max_bsel_off = top_bsel_off['value'].to_list()[0]

    # nz_prop
    top_nz_prop_df = pd.DataFrame(highlights_dict['nz_prop'])
    max_sel_header = top_nz_prop_df.iloc[0]
    top_nz_prop_df = top_nz_prop_df[1:]
    top_nz_prop_df.columns = max_sel_header
    top_nz_prop_df = top_nz_prop_df.sort_values(by='value', ascending=False)
    top_nz_prop = top_nz_prop_df[~top_nz_prop_df.layer.str.contains('utput')]
    max_nz_prop = top_nz_prop['value'].to_list()[0]

    # hi_val_prop
    top_hi_v_df = pd.DataFrame(highlights_dict['hi_val_prop'])
    max_sel_header = top_hi_v_df.iloc[0]
    top_hi_v_df = top_hi_v_df[1:]
    top_hi_v_df.columns = max_sel_header
    top_hi_v_df = top_hi_v_df.sort_values(by='value', ascending=False)
    top_hi_v = top_hi_v_df[~top_hi_v_df.layer.str.contains('utput')]
    max_hi_v_prop = top_hi_v['value'].to_list()[0]

    # # selectiviy summary
    sel_csv_info = [gha_dict['topic_info']['cond'], gha_dict['topic_info']['run'], output_filename,
                    gha_dict['data_info']['dataset'], gha_dict['GHA_info']['use_dataset'],
                    n_layers,
                    gha_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable'],
                    trained_for,
                    gha_dict['GHA_info']['scores_dict']['gha_acc'],
                    last_hid_layer,
                    mean_roc, max_roc,
                    mean_ap, max_ap,
                    mean_info, max_info,
                    mean_CCMA, max_CCMA,
                    mean_bsel, max_bsel,
                    mean_bsel_zero, mean_bsel_pfive,
                    # mean_bsel_off, max_bsel_off,
                    mean_nz_prop, max_nz_prop,
                    mean_hi_val_prop, max_hi_v_prop,
                    mean_act, mead_act_sd
                    ]

    summary_headers = ["cond", "run", "output_filename", "dataset", "use_dataset", "n_layers", "hid_units",
                       "trained_for", "gha_acc",
                       "last_layer",
                       "mean_roc", "max_roc",
                       "mean_ap", "max_ap",
                       "mean_info", "max_info",
                       "mean_CCMA", "max_CCMA",
                       "mean_bsel", "max_bsel",
                       "mean_bsel_0", 'mean_bsel_.5',
                       # "mean_bsel_off", "max_bsel_off",
                       "mean_nz_prop", "max_nz_prop",
                       "mean_hi_val_prop", "max_hi_v_prop",
                       'mean_act', 'sd_act'
                       ]

    exp_path, cond_name = os.path.split(gha_dict['topic_info']['exp_cond_path'])
    exp_name = gha_dict['topic_info']['exp_name']

    os.chdir(exp_path)
    print(f"exp_path: {exp_path}")

    if not os.path.isfile(exp_name + "_sel_summary.csv"):
        sel_summary = open(exp_name + "_sel_summary.csv", 'w')
        mywriter = csv.writer(sel_summary)
        mywriter.writerow(summary_headers)
        print(f"creating summary csv at: {exp_path}")
    else:
        sel_summary = open(exp_name + "_sel_summary.csv", 'a')
        mywriter = csv.writer(sel_summary)
        print(f"appending to summary csv at: {exp_path}")

    mywriter.writerow(sel_csv_info)
    sel_summary.close()

    print("\nend of sel script")

    return master_dict  # , mean_sel_per_NN

# # #
# print("\nWARNING\n\nrunning test from bottom of sel script\nWARNING")
