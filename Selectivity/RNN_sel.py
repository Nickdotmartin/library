import copy
import csv
import datetime
import os
import pickle
import shelve

import h5py
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_curve, auc

from tools.hdf import hdf_df_string_clean
from tools.dicts import load_dict, focussed_dict_print

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

        # # no longer need TCS or NZ_AP, NZ_pr
        # # # to sort out noZero scores
        # nz_ave_prec = np.sum(my_ave_prec_vals_dict[:-1])
        # if len(recall_dict[:-1]) > 1:
        #     nz_pr_auc = auc(recall_dict[:-1], precision_dict[:-1])
        # else:
        #     nz_pr_auc = 0.0
        #
        # # top_class_sel
        # top_class_sel_rows = precision_dict.count(1)
        # # todo: revisit this or remove it.  Should I be setting tcs_thr to 1 for most classes?
        # top_roc_stuff_dict = 'None'
        # if top_class_sel_rows > 0:
        #     top_class_sel_recall = recall_dict[top_class_sel_rows]
        #     top_class_sel_recall_thr = thr[top_class_sel_rows]
        #     top_class_sel_items = int(tp_count_dict[top_class_sel_rows])
        # else:
        #     top_class_sel_recall = 0
        #     top_class_sel_recall_thr = 1
        #     top_class_sel_items = 0

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

        # # no longer need max_info f1
        # if max_informed_prec + max_info_sens == 0.0:
        #     max_info_f1 = 0.0
        # else:
        #     max_info_f1 = 2 * (max_informed_prec * max_info_sens) / (max_informed_prec + max_info_sens)
        # if np.isnan(max_info_f1):
        #     max_info_f1 = 0.0
        # if np.isfinite(max_info_f1):
        #     max_info_f1 = 0.0
        # if type(max_info_f1) is not float:
        #     max_info_f1 = 0.0


    else:  # if there are not items in this class
        roc_auc = ave_prec = pr_auc = nz_ave_prec = nz_pr_auc = 0
        top_class_sel_items = top_class_sel_recall_thr = top_class_sel_recall = 0
        max_informed = max_informed_count = max_informed_thr = 0
        max_info_sens = max_info_spec = max_informed_prec = max_info_f1 = 0

    roc_sel_dict = {'roc_auc': roc_auc,
                    'ave_prec': ave_prec,
                    'pr_auc': pr_auc,
                    # 'nz_ave_prec': nz_ave_prec,
                    # 'nz_pr_auc': nz_pr_auc,
                    # 'tcs_items': top_class_sel_items,
                    # 'tcs_thr': top_class_sel_recall_thr,
                    # 'tcs_recall': top_class_sel_recall,
                    'max_informed': max_informed,
                    'max_info_count': max_informed_count,
                    'max_info_thr': max_informed_thr,
                    'max_info_sens': max_info_sens,
                    'max_info_spec': max_info_spec,
                    'max_info_prec': max_informed_prec,
                    # 'max_info_f1': max_info_f1,
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


def class_sel_basics(this_unit_acts_df, items_per_cat, hi_val_thr=.5,
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
                        (of correct items or all items depending on hid acts)
    :param hi_val_thr: threshold above which an item is considered to be 'strongly active'.
    :param act_func: relu, sigmoid or tanh.  If Relu use normed acts, else use regular.

    :param verbose: how much to print to screen

    :return: class_sel_basics_dict
    """

    print("**** class_sel_basics() ****")

    act_values = 'activation'
    if act_func is 'relu':
        act_values = 'normed'

    # # means
    means_dict = dict(this_unit_acts_df.groupby('class')[act_values].mean())
    sd_dict = dict(this_unit_acts_df.groupby('class')[act_values].std())

    # # non-zero_count
    nz_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values]
                                           > 0.0].groupby('class')[act_values].count())

    for i in range(max(list(items_per_cat.keys()))):
        if i not in nz_count_dict:
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
                                               hi_val_thr].groupby('class')[act_values].count())
    for i in range(max(list(items_per_cat.keys()))):
        if i not in hi_val_count_dict:
            hi_val_count_dict[i] = 0

    hi_val_total = this_unit_acts_df[this_unit_acts_df[act_values] > hi_val_thr][act_values].count()

    # # hi vals precision
    hi_val_prec_dict = {k: v / hi_val_total for k, v in hi_val_count_dict.items()}

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

    # # edit items (where max should be based on main sel value, e.g., not  max threshold).
    # remove unnecessary items
    # max_sel_dict['tcs_thr'] = copy_sel_dict['tcs_thr'][max_sel_dict["tcs_recall_c"]]
    # del max_sel_dict['tcs_thr_c']
    # max_sel_dict['tcs_items'] = copy_sel_dict['tcs_items'][max_sel_dict["tcs_recall_c"]]
    # del max_sel_dict['tcs_items_c']

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
    # max_sel_dict['max_info_f1'] = copy_sel_dict['max_info_f1'][max_sel_dict["max_informed_c"]]
    # del max_sel_dict['max_info_f1_c']

    max_sel_dict['zhou_selects'] = copy_sel_dict['zhou_selects'][max_sel_dict["zhou_prec_c"]]
    del max_sel_dict['zhou_selects_c']
    max_sel_dict['zhou_thr'] = copy_sel_dict['zhou_thr'][max_sel_dict["zhou_prec_c"]]
    del max_sel_dict['zhou_thr_c']

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


####################################################################################################

# @profile
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
                run selectivity measures : CCMA, zn_AP, pr(full), zhou_prec, top_class, informedness
                save dicts with values for all measures
                save dict with top class/value for all measures

    :param gha_dict_path: path of the gha dict
    :param correct_items_only: Whether selectivity considered incorrect items
    :param all_classes: Whether to test for selectivity of all classes or a subset
                        (e.g., most active classes)
    :param layer_classes: Which layers to analyse
    :param verbose: how much to print to screen
    :param test_run: if True, only do subset, e.g., 3 units from 3 layers

    :return: master dict: contains 'sel_path' (e.g., dir),
                                    'sel_per_unit_pickle_name',
                                    'sel_highlights_list_dict_name',

        :return: sel_p_unit_dict. dict with all selectivity results in it.
            all sel values:                 [layer names][unit number][sel measure][class]
            max_sel_p_u values(and class):  [layer names][unit number]['max'][sel measure]
            layer means:                    [layer name]['means'][measure]


        :return: sel_highlights: keys are sel_measures.
                                Values are DataFrames with the 10 highest scoring units for
                                each sel measure.

    """

    # todo: split this into two functions.
    #  1. loop thru model pickle. (to be developed here but
    #  then moved to tools and used else wherer) Can be used with other dicts
    #  to loop through probably (gha, visualization, lesioning etc
    #  2. get unit sel.
    #  the output of 2 is saved to dict by 1. dict now keeps its name.
    #  1. can have a separate hd5 version or add thsat as a funtion for saving.
    # Set save to file regularity by data size.

    # todo: add act_func as a param, can be updated if necessary.

    print("\n**** running ff_sel() ****")

    exp_cond_gha_path, gha_dict_name = os.path.split(gha_dict_path)
    os.chdir(exp_cond_gha_path)
    current_wd = os.getcwd()

    print(f"current_wd: {current_wd}")

    # # part 1. load dict from study (should run with sim, GHA or sel dict)
    gha_dict = load_dict(gha_dict_path)
    focussed_dict_print(gha_dict, 'gha_dict')

    # Output files
    output_filename = gha_dict["topic_info"]["output_filename"]
    print(f"\noutput_filename: {output_filename}")

    hdf_name = f'{output_filename}_gha.h5'

    # # # load item_correct (y_data)
    # todo: check this is correct y label data
    # todo: add letter labels and/or word labels.
    y_scores_df = pd.read_hdf(hdf_name, key='item_correct_df', more='r')
    y_scores_df = hdf_df_string_clean(y_scores_df)
    # use y_df for analysis
    y_df = y_scores_df

    # get other info from dict
    hid_acts_pickle = gha_dict["GHA_info"]["hid_act_files"]['2d']
    n_cats = gha_dict["data_info"]["n_cats"]
    all_items = gha_dict['GHA_info']['scores_dict']['n_items']
    n_correct = gha_dict['GHA_info']['scores_dict']['n_correct']
    n_incorrect = all_items - n_correct
    items_per_cat = gha_dict['GHA_info']['scores_dict']['corr_per_cat_dict']
    timesteps = gha_dict['model_info']["timesteps"]

    # # get basic COI list
    if all_classes is True:
        if type(items_per_cat) is int:
            classes_of_interest = list(range(n_cats))
        else:
            classes_of_interest = list({k: v for (k, v) in items_per_cat.items() if v > 0})

        if len(classes_of_interest) == n_cats:
            print(f"\nall {n_cats} classes have some correct items")
        else:
            print(f"\nclasses_of_interest (all classes with correct items):\n{classes_of_interest}")

    # # # get values for correct/incorrect items (1/0 or True/False)
    full_model_values = y_scores_df.full_model.unique()
    if len(full_model_values) != 2:
        print(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")
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
            print("no changes needed - don't remove anything from hid_acts, output and "
                  "use y scores as y_df")
    else:
        if correct_items_only:
            print("\ngha_incorrect: False (I only have correct responses)\n"
                  "correct_items_only: True (I only want correct responses)")
            print("no changes needed - don't remove anything from hid_acts or output.  "
                  "Use y_correct as y_df")
            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_item]
            y_df = y_correct_df
        else:
            print("\ngha_incorrect: False (I only have correct responses)\n"
                  "correct_items_only: False (I want incorrect responses)")
            print("I can not complete this as desried"
                  "change correct_items_only to True"
                  "for analysis  - don't remove anything from hid_acts, output and "
                  "use y scores as y_df")
            correct_items_only = True

    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")

    # # open hid_acts.pickle
    '''
    1. get layer number for last layer
    2. get output layer activations (these are held in memory, but could be loaded for each unit)
    3. * add get list of hid act dict keys - loop thought this rather than actual dict.
    '''

    print("\n**** opening hid_acts.pickle ****")
    with open(hid_acts_pickle, 'rb') as pkl:
        hid_acts_dict = pickle.load(pkl)

    hid_acts_keys_list = list(hid_acts_dict.keys())

    if verbose:
        print(f"hid_acts_pickle: {hid_acts_pickle}")
        focussed_dict_print(hid_acts_dict, 'hid_acts_dict')
        # print(f"hid_acts_pickle: {hid_acts_pickle}\n"
        #       f"hid_acts_dict: {hid_acts_dict}\n"
        #       f"hid_acts_dict.keys(): {hid_acts_dict.keys()}")

    # # get output activations for class correlation
    last_layer_num = hid_acts_keys_list[-1]
    last_layer_name = hid_acts_dict[last_layer_num]['layer_name']

    with h5py.File(hdf_name, 'r') as my_hdf:
        output_layer_acts = my_hdf['hid_acts_2d'][last_layer_name]
        output_layer_df = pd.DataFrame(output_layer_acts)

    if correct_items_only:
        if gha_incorrect:
            output_layer_df['full_model'] = y_scores_df['full_model']
            output_layer_df = output_layer_df.loc[output_layer_df['full_model'] == 1]
            output_layer_df = output_layer_df.drop(['full_model'], axis=1)
            print(f"\nremoving {n_incorrect} incorrect responses from output_layer_df: "
                  f"{output_layer_df.shape}\n")

    # # close hid act dict to save memory space?
    # hid_acts_dict = dict()

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

    # # save output activations
    # print(f"output_lyer_df shape: {output_layer_df.shape}")
    # print(output_layer_df.head())
    # output_layer_df.to_csv(f'{sel_path}/{output_filename}_output_acts.csv', index=False)

    # # # sel_p_unit_dict
    # sel_per_unit_pickle_name = f"{sel_path}/{output_filename}_sel_per_unit.pickle"
    # if not os.path.isfile(sel_per_unit_pickle_name):
    #     sel_p_unit_dict = dict()
    #     # save sel_p_unit_dict here to be opened and appended to
    #     with open(sel_per_unit_pickle_name, "wb") as pickle_out:
    #         pickle.dump(sel_p_unit_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    # # # sel_p_unit_dict
    sel_per_unit_db_name = f"{sel_path}/{output_filename}_sel_per_unit"
    if not os.path.isfile(sel_per_unit_db_name):
        # sel_p_unit_dict = dict()
        # save sel_p_unit_dict here to be opened and appended to
        with shelve.open(sel_per_unit_db_name, protocol=pickle.HIGHEST_PROTOCOL) as db:
            db['test_key'] = 'test_value'

    # # # layer_sel_mean_dict
    layer_sel_mean_dict = dict()

    # # layer means csv path
    layer_means_path = os.path.join(sel_path, f"{output_filename}_layer_means.csv")

    # # highlights_dict
    sel_highlights_list_dict_name = f"{sel_path}/{output_filename}_hl_list.pickle"
    if not os.path.isfile(sel_highlights_list_dict_name):
        # # save highlights_dict here to be opened and appended to
        highlights_dict = {"roc_auc": [['value', 'class', 'layer', 'unit']],
                           "ave_prec": [['value', 'class', 'layer', 'unit']],
                           "pr_auc": [['value', 'class', 'layer', 'unit']],
                           # "nz_ave_prec": [['value', 'class', 'layer', 'unit']],
                           # "nz_pr_auc": [['value', 'class', 'layer', 'unit']],
                           # "tcs_recall": [['value', 'class', 'thr', 'items', 'layer', 'unit']],
                           "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
                                             'prec',
                                             # 'f1',
                                             'layer', 'unit']],
                           "ccma": [['value', 'class', 'layer', 'unit']],
                           "zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],
                           "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],
                           "means": [['value', 'class', 'sd', 'layer', 'unit']],
                           "nz_count": [['value', 'class', 'layer', 'unit']],
                           "nz_prop": [['value', 'class', 'layer', 'unit']],
                           "nz_prec": [['value', 'class', 'layer', 'unit']],
                           "hi_val_count": [['value', 'class', 'layer', 'unit']],
                           "hi_val_prop": [['value', 'class', 'layer', 'unit']],
                           "hi_val_prec": [['value', 'class', 'layer', 'unit']],
                           }

        with open(sel_highlights_list_dict_name, "wb") as pickle_out:
            pickle.dump(highlights_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    # # loop through dict/layers
    print("\n*** looping through layers ***")

    n_layers = len(gha_dict['GHA_info']['model_dict'])

    if test_run:
        layer_counter = 0

    for layer_number in hid_acts_keys_list:

        layer_dict = hid_acts_dict[layer_number]
        layer_name = layer_dict['layer_name']

        # todo: get act func

        print(f"\nlayer {layer_number}: {layer_name}")

        if layer_dict['layer_class'] not in layer_classes:
            print(f"\tskip this layer!: {layer_dict['layer_class']} not in {layer_classes}")
            continue  # skip this layer

        # # don't run sel on output layer
        if layer_number == last_layer_num:
            print(f"\tskip output layer!: {layer_number} == {last_layer_num}")
            continue

        if test_run is True:
            layer_counter = layer_counter + 1
            if layer_counter > 3:
                print(f"\tskip this layer!: test_run, only running subset of layers")
                continue

        # todo: write results script which will take the input dict/csv,
        #  put it into a pandas df, have versions for sort by order of x variable,
        #  select top n, etc.  use that to then output a list of units to visualise.
        #  Can add call visualise unit



        # # check if layer totals have already been done for sel_p_unit_dict and highlights dict
        sel_p_u_layer_done = False
        # with open(sel_per_unit_pickle_name, "rb") as pickle_load:
        #     sel_p_unit_dict = pickle.load(pickle_load)
        #     sel_p_unit_keys = sel_p_unit_dict.keys()

        with shelve.open(sel_per_unit_db_name) as db:
            sel_p_unit_keys = list(db.keys())
            print(sel_p_unit_keys)

        if layer_name in sel_p_unit_keys:
            sel_p_u_layer_done = True

        highlights_layer_done = False
        with open(sel_highlights_list_dict_name, "rb") as pickle_load:
            # read dict as it is so far
            highlights_dict = pickle.load(pickle_load)
            check_roc = highlights_dict['roc_auc']
        if any(layer_name in sublist for sublist in check_roc):
            highlights_layer_done = True

        layer_means_csv_done = False
        already_done_means_total = False
        if os.path.isfile(layer_means_path):
            layer_means_csv = pd.read_csv(layer_means_path)
            layer_names = layer_means_csv.loc[:, 'name'].to_list()
            if layer_name in layer_names:
                layer_means_csv_done = True
            if 'Total' in layer_names:
                already_done_means_total = True

        layer_done = [sel_p_u_layer_done, highlights_layer_done, layer_means_csv_done]

        if layer_done == [True, True, True]:
            print(f"\tdocs layer has already been completed\n"
                  f"\t(sel_p_u_layer_done, highlights_layer_done, layer_means_csv_done)")
            continue
        else:
            print(f"\tNot yet completed this layer\n"
                  f"\tsel_p_u_layer_done: {sel_p_u_layer_done}\n"
                  f"\thighlights_layer_done: {highlights_layer_done}\n"
                  f"\tlayer_means_csv_done: {layer_means_csv_done}")

        # layer_number = key
        with h5py.File(hdf_name, 'r') as gha_data:
            hid_acts_array = gha_data['hid_acts_2d'][layer_name]
            hid_acts_df = pd.DataFrame(hid_acts_array)

        print(f"\nloaded hidden_activation_file: {hdf_name}, {hid_acts_df.shape}")
        units_per_layer = len(hid_acts_df.columns)

        with shelve.open(sel_per_unit_db_name, protocol=pickle.HIGHEST_PROTOCOL) as db:
            db[layer_name] = dict()

        # # remove incorrect responses
        if correct_items_only:
            if gha_incorrect:
                print(f"\nremoving {n_incorrect} incorrect responses from "
                      f"hid_acts_df: {hid_acts_df.shape}")
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
                         'zhou_prec': {},
                         'zhou_selects': {},
                         'zhou_thr': {},
                         'corr_coef': {},
                         'corr_p': {},
                         }

            this_unit_just_acts = list(hid_acts_df.loc[:, unit])

            # todo: get unit/layer act func

            # # 1st check its not a dead unit
            # check for dead units, if dead, all details to 0/na/nan/-999 etc
            if sum(this_unit_just_acts) != 0:

                print("not dead - sorting activations for analysis")

                # todo: add timesteps (from line 188 of sel_stm_RNN)
                """
                for timestep in range(timesteps):
                    print("unit {} step {} (of {})".format(this_unit, timestep, timesteps))
    
                    one_unit_one_timestep = one_unit_all_timesteps[:, timestep]
                    print(" - one_unit_one_timestep shape: (n seqs) {}".format(np.shape(one_unit_one_timestep)))
    
                    # y_labels_one_timestep_float = combo_data[:, timestep]
                    y_labels_one_timestep_float = Y_labels[:, timestep]
    
                    y_labels_one_timestep = [int(q) for q in y_labels_one_timestep_float]
                    print(" - y_labels_one_timestep shape: {}".format(np.shape(y_labels_one_timestep)))
                    # print(y_labels_one_timestep)
    
                    index = list(range(n_items))
    
                    # insert act values in middle of labels (item, act, cat)
                    this_unit_acts = np.vstack((index, one_unit_one_timestep, y_labels_one_timestep)).T
                    print(" - this_unit_acts shape: {}".format(np.shape(this_unit_acts)))
                    print(f"this_unit_acts: (seq_index, hid_acts, y_label)\n{this_unit_acts}")
                """

                # insert act values in middle of labels (item, act, cat)
                this_unit_acts = y_df.copy()

                this_unit_acts.insert(2, column='activation', value=this_unit_just_acts, )

                # # sort by descending hid act values
                this_unit_acts_df = this_unit_acts.sort_values(by='activation', ascending=False)

                # # # normalize activations
                # todo: I don't need to normalize them if they are tanh units?

                just_act_values = this_unit_acts_df['activation'].tolist()
                max_act = max(just_act_values)
                normed_acts = np.true_divide(just_act_values, max_act)
                this_unit_acts_df.insert(2, column='normed', value=normed_acts)

                if verbose is True:
                    print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n"
                          f"{this_unit_acts_df.head()}")

                # # get class_sel_basics
                # todo: add act_func
                class_sel_basics_dict = class_sel_basics(this_unit_acts_df, items_per_cat,
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
                        print(f"\nclass_{this_cat}: {this_class_size} items, "
                              f"not_{this_cat}: {not_a_size} items")

                    # # running selectivity measures

                    # # ROC_stuff includes:
                    # roc_auc, ave_prec, pr_auc, nz_ave_prec, nz_pr_auc, top_class_sel, informedness
                    # todo: add act_func.  If not relu.  don't use normed acts
                    act_values = 'activation'
                    if act_func is 'relu':
                        act_values = 'normed'

                    roc_stuff_dict = nick_roc_stuff(class_list=this_unit_acts_df['class'],
                                                    hid_acts=this_unit_acts_df[act_values],
                                                    this_class=this_cat,
                                                    class_a_size=this_class_size,
                                                    not_a_size=not_a_size,
                                                    verbose=verbose)

                    # # add class_sel_basics_dict to unit dict
                    for roc_key, roc_value in roc_stuff_dict.items():
                        unit_dict[roc_key][this_cat] = roc_value

                    # # CCMA
                    class_a = this_unit_acts_df.loc[this_unit_acts_df['class'] == this_cat]
                    class_a_mean = class_a[act_values].mean()
                    not_class_a = this_unit_acts_df.loc[this_unit_acts_df['class'] != this_cat]
                    not_class_a_mean = not_class_a[act_values].mean()
                    ccma = (class_a_mean - not_class_a_mean) / (class_a_mean + not_class_a_mean)
                    unit_dict["ccma"][this_cat] = ccma

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
                    # get output activations for class correlation
                    # output_layer_df = pd.read_csv(f'{sel_path}/{output_filename}_output_acts.csv')
                    # print(f"output_lyer_df shape: {output_layer_df.shape}")
                    # print(output_layer_df.head())
                    # print(output_layer_df.iloc[:, this_cat])

                    class_corr = class_correlation(this_unit_acts=this_unit_acts_df[act_values],
                                                   output_acts=output_layer_df[this_cat],
                                                   verbose=verbose)
                    unit_dict["corr_coef"][this_cat] = class_corr['coef']
                    unit_dict["corr_p"][this_cat] = class_corr['p']

                    # del output_layer_df

                if verbose:
                    focussed_dict_print(unit_dict, f'unit_dict {unit}')

                # # # for each sel variable - get the class with the highest values and add to per_unit
                max_sel_p_unit_dict = sel_unit_max(unit_dict, verbose=verbose)
                unit_dict['max'] = max_sel_p_unit_dict
                max_sel_dict[unit] = max_sel_p_unit_dict

                # # check to see if this unit has any thing for highlights
                if verbose:
                    focussed_dict_print(max_sel_p_unit_dict, f'max_sel_p_unit_dict, unit {unit}')

                # # ADD UNIT DICT TO LAYER
                # layer_dict[unit] = unit_dict

                # todo: saving for [layer_name][unit][timestep]

                with shelve.open(sel_per_unit_db_name, protocol=pickle.HIGHEST_PROTOCOL,
                                 writeback=True) as db:
                    db[layer_name][unit] = unit_dict

            else:
                print("dead unit found")
                unit_dict = 'dead_unit'

        print(f"********\n"
              f"finished looping through units in {layer_name}\n"
              f"************")

        # todo: will max sel per unit collapse across timesteps?
        #  or have separate sheets for each timestep?


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

        # todo: will means collapse across timesteps?
        #  or have separate sheets for each timestep?

        if unit_dict != 'dead_unit':
            for measure in list(unit_dict.keys()):
                layer_unit_sel_list = []
                if measure == 'max':
                    continue
                for unit, unit_sel_dict in max_sel_dict.items():
                    layer_unit_sel_list.append(unit_sel_dict[measure])
                layer_measure_mean = np.mean(layer_unit_sel_list)
                layer_sel_mean_dict[measure] = layer_measure_mean

        focussed_dict_print(layer_sel_mean_dict, 'layer_sel_mean_dict')
        # layer_dict['means'] = layer_sel_mean_dict

        # # append layer dict to model dict
        '''make sure I use the right command to read and write here.'''
        # with open(sel_per_unit_pickle_name, "rb") as pickle_load:
        #     # read dict as it is so far
        #     sel_p_unit_dict = pickle.load(pickle_load)
        #
        # # # save sel_p_unit_dict here
        # with open(sel_per_unit_pickle_name, "wb") as pickle_out:
        #     pickle.dump(sel_p_unit_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

        # with shelve.open(sel_per_unit_db_name, protocol=pickle.HIGHEST_PROTOCOL, writeback=True) as db:
        #     db[layer_name] = layer_dict

        with shelve.open(sel_per_unit_db_name, protocol=pickle.HIGHEST_PROTOCOL,
                         writeback=True) as db:
            db[layer_name]['means'] = layer_sel_mean_dict

        # # # clear layer_dict to save memory
        # layer_dict.clear()
        layer_sel_mean_dict.clear()

        # # layer means
        layer_means_list = list(layer_sel_mean_dict.values())
        layer_means_list = [layer_name, unit_index + 1] + layer_means_list

        layer_means_headers = list(layer_sel_mean_dict.keys())
        layer_means_headers = ['name', 'units'] + layer_means_headers

        # already_done_means_total = False
        if not os.path.isfile(layer_means_path):
            # already_done_means_total = False
            layer_means_csv = open(layer_means_path, 'w')
            mywriter = csv.writer(layer_means_csv)
            mywriter.writerow(layer_means_headers)
            mywriter.writerow(layer_means_list)
            layer_means_csv.close()
            print(f"\ncreating layer_means csv at: {output_filename}")
        else:
            check_it = pd.read_csv(layer_means_path)
            if layer_name not in check_it['name'].to_list():
                layer_means_csv = open(layer_means_path, 'a')
                mywriter = csv.writer(layer_means_csv)
                mywriter.writerow(layer_means_list)
                layer_means_csv.close()
                print(f"\nappending to layer_means csv at: {output_filename}")
            else:
                if check_it['name'].to_list()[-1] == 'Total':
                    print("\nAlready have layer means for this cond")
                    already_done_means_total = True

        # # # get top three highlights for each feature
        # todo: Highlights should include timestep as a variable.
        #  e.g., unit 1, timestep 3 has max ROC score
        with open(sel_highlights_list_dict_name, "rb") as pickle_load:
            # read dict as it is so far
            highlights_dict = pickle.load(pickle_load)

            highlights_list = list(highlights_dict.keys())

        for measure in highlights_list:
            # sort max_sel_df by sel measure for this layer
            layer_top_3_df = max_sel_df.sort_values(by=measure, ascending=False).reset_index()

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
            #             top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, f'{measure}_c'].item()
            #             thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_thr'].item()
            #             items = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_items'].item()
            #             top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()
            #
            #         new_row = [top_val, top_class, thr, items, layer_name, top_unit_name]
            #         # print(f"new_row\n{new_row}")
            #         # highlights_dict[measure].append(new_row)

            if measure == 'max_informed':
                print('max_informed')
                # "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
                #                                          'prec', 'f1', 'count', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                   'unit'])) > 1:
                        top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                            f'{measure}_c'])[i]
                        count = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                        'max_info_count'])[i]
                        thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                      'max_info_thr'])[i]
                        sens = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       'max_info_sens'])[i]
                        spec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       'max_info_spec'])[i]
                        prec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       'max_info_prec'])[i]
                        # f1 = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                        # 'max_info_f1'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                                'unit'])[i]
                    else:
                        top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       f'{measure}_c'].item()
                        count = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                   'max_info_count'].item()
                        thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                 'max_info_thr'].item()
                        sens = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                  'max_info_sens'].item()
                        spec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                  'max_info_spec'].item()
                        prec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                  'max_info_prec'].item()
                        # f1 = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                        # 'max_info_f1'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                           'unit'].item()

                    # new_row = [top_val, top_class, count, thr, sens, spec, prec, f1, layer_name, top_unit_name]
                    new_row = [top_val, top_class, count, thr, sens,
                               spec, prec, layer_name, top_unit_name]

                    # print(f"new_row\n{new_row}")
                    # highlights_dict[measure].append(new_row)

            elif measure == 'zhou_prec':
                print('zhou_prec')
                # "zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                   'unit'])) > 1:
                        top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                            f'{measure}_c'])[i]
                        thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                      'zhou_thr'])[i]
                        selects = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                          'zhou_selects'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                                'unit'])[i]
                    else:
                        top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       f'{measure}_c'].item()
                        thr = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                 'zhou_thr'].item()
                        selects = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                     'zhou_selects'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                           'unit'].item()

                    new_row = [top_val, top_class, thr, selects, layer_name, top_unit_name]
                    # print(f"new_row\n{new_row}")
                    # highlights_dict[measure].append(new_row)

            elif measure == 'corr_coef':
                print('corr_coef')
                #   "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if np.isnan(top_val):
                        continue
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                            f'{measure}_c'])[i]
                        p = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                    'corr_p'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                                'unit'])[i]
                    else:
                        top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       f'{measure}_c'].item()
                        p = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                               'corr_p'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                           'unit'].item()

                    new_row = [top_val, top_class, p, layer_name, top_unit_name]
                    # print(f"new_row\n{new_row}")
                    # highlights_dict[measure].append(new_row)

            elif measure == 'means':
                print('means')  # "means": [['value', 'class', 'sd', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                            f'{measure}_c'])[i]
                        sd = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                     'sd'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                                'unit'])[i]
                    else:
                        top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       f'{measure}_c'].item()
                        sd = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                'sd'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                           'unit'].item()

                    new_row = [top_val, top_class, sd, layer_name, top_unit_name]
                    # print(f"new_row\n{new_row}")
                    # highlights_dict[measure].append(new_row)

            else:  # for most most measures use below
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    # print('top_val: ', top_val)
                    # print(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                            f'{measure}_c'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                                'unit'])[i]
                    else:
                        top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                       f'{measure}_c'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val,
                                                           'unit'].item()

                    new_row = [top_val, top_class, layer_name, top_unit_name]

                    # print(f"new_row\n{new_row}")
                    # highlights_dict[measure].append(new_row)

            print(f"new_row\n{new_row}")

            with open(sel_highlights_list_dict_name, "rb") as pickle_load:
                # read dict as it is so far
                highlights_dict = pickle.load(pickle_load)

            highlights_dict[measure].append(new_row)

            print(f"\nhighlights_dict[{measure}]\n{highlights_dict[measure]}")

            # save highlights_dict here
            with open(sel_highlights_list_dict_name, "wb") as pickle_out:
                pickle.dump(highlights_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

            highlights_dict.clear()

        print(f"\nFinihshed getting data for highlights_dict")

    print("\n****** Finished looping through all layers ******")

    # # add means total
    lm = pd.read_csv(layer_means_path)
    # lm.set_index('name')
    # print("check column names")
    # print(f"{len(list(lm))} cols\n{list(lm)}")
    print(lm)
    print(lm['name'].to_list())
    print(f"already_done_means_total: {already_done_means_total}")
    if lm['name'].to_list()[-1] != 'Total':
        already_done_means_total = False

    if not already_done_means_total:
        if 'Total' in lm['name'].to_list():
            print("total already here, removing and starting again")
            lm = lm[lm.name != 'Total']
            print(lm)

        print("appending total to layer_means csv")
        total_means = []
        for column_name, column_data in lm.iteritems():
            # print(f"column_name: {column_name}")
            if column_name == 'name':
                # ignore this column
                # continue
                total_means.append('Total')
            elif column_name == 'units':
                sum_c = column_data.sum()
                total_means.append(sum_c)
            else:
                mean = column_data.mean()
                total_means.append(mean)

        totals_s = pd.Series(data=total_means, index=lm.columns, name='Total')
        lm = lm.append(totals_s)

        lm.to_csv(layer_means_path, index=False)

    # # make new highlights_df_dict with dataframes
    highlights_df_dict = dict()

    # load highlights dict
    with open(sel_highlights_list_dict_name, "rb") as pickle_load:
        # read dict as it is so far
        highlights_dict = pickle.load(pickle_load)

        for k, v in highlights_dict.items():
            hl_df = pd.DataFrame(data=v[1:], columns=v[0])
            hl_df.sort_values(by='value', ascending=False, inplace=True)
            highlights_df_dict[k] = hl_df
            if verbose:
                print(f"\n{k} highlights (head)")
                print(hl_df)

    # Save new highlights_df_dict here
    sel_highlights_df_dict_name = f"{sel_path}/{output_filename}_highlights.pickle"
    with open(sel_highlights_df_dict_name, "wb") as pickle_out:
        pickle.dump(highlights_df_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    # sel_highlights_list_dict_name = f"{sel_path}/{output_filename}_highlights.pickle"
    # pickle_out = open(sel_highlights_list_dict_name, "wb")
    # pickle.dump(highlights_df_dict, pickle_out)
    # pickle_out.close()

    # sel_per_unit_pickle_name = f"{sel_path}/{output_filename}_sel_per_unit.pickle"
    # pickle_out = open(sel_per_unit_pickle_name, "wb")
    # pickle.dump(sel_p_unit_dict, pickle_out)
    # pickle_out.close()

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
                               # "sel_per_unit_pickle_name": sel_per_unit_pickle_name,
                               "sel_per_unit_name": sel_per_unit_db_name,
                               'sel_highlights_list_dict_name': sel_highlights_list_dict_name,
                               "correct_items_only": correct_items_only,
                               "all_classes": all_classes, "layer_classes": layer_classes,
                               "sel_date": int(datetime.datetime.now().strftime("%y%m%d")),
                               "sel_time": int(datetime.datetime.now().strftime("%H%M")),
                               }

    print(f"Saving dict to: {os.getcwd()}")
    pickle_out = open(sel_dict_name, "wb")
    pickle.dump(master_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()

    focussed_dict_print(master_dict, "master_dict")
    # print_nested_round_floats(master_dict)

    # # save summary csv
    key_layers_list = [x for x in gha_dict['GHA_info']['gha_key_layers']
                       if 'output' not in str.lower(x)]
    last_hid_layer = key_layers_list[-1]

    print("lm")
    print(lm)
    # mean_roc = lm.at['Total', 'roc_auc']
    # mean_ap = lm.at['Total', 'ave_prec']
    # mean_info = lm.at['Total', 'max_informed']

    # mean_roc = lm.loc[lm['name'] == 'Total', 'roc_auc']
    # mean_ap = lm.loc[lm['name'] == 'Total', 'ave_prec']
    # mean_info = lm.loc[lm['name'] == 'Total', 'max_informed']

    print("\ngetting max sel values from highlights")
    # todo: add timestep as a variable for these too?
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

    # # selectiviy summary
    run = gha_dict['topic_info']['run']
    if test_run:
        run = 'test'

    sel_csv_info = [gha_dict['topic_info']['cond'], run, output_filename,
                    gha_dict['data_info']['dataset'], gha_dict['GHA_info']['use_dataset'],
                    n_layers,
                    gha_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable'],
                    gha_dict['GHA_info']['scores_dict']['gha_acc'],
                    last_hid_layer,
                    # mean_roc,
                    max_roc,
                    # mean_ap,
                    max_ap,
                    # mean_info,
                    max_info]

    summary_headers = ["cond", "run", "output_filename", "dataset", "use_dataset",
                       "n_layers", "hid_units",
                       "gha_acc",
                       "last_layer",
                       # "mean_roc",
                       "max_roc",
                       # "mean_ap",
                       "max_ap",
                       # "mean_info",
                       "max_info"]

    exp_path, cond_name = os.path.split(gha_dict['GHA_info']['gha_path'])
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

    print("\nSanity check for shelve")
    print(sel_per_unit_db_name)
    with shelve.open(sel_per_unit_db_name, flag='r') as db:
        sel_p_unit_keys = list(db.keys())
        print(sel_p_unit_keys)
        print(db['fc1'].keys())

    print("\nend of sel script")

    return master_dict  # , mean_sel_per_NN

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

