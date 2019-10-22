import os
import sys
import datetime
import copy
import operator
import shelve
from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.stats import pearsonr

sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')
from nick_dict_tools import load_dict, focussed_dict_print, print_nested_round_floats
from nick_data_tools import load_from_datasets, get_dset_path, load_y_data, load_hid_acts, nick_to_csv, nick_read_csv



# # roc_stuff
def nick_roc_stuff(class_list, hid_acts, this_class, classA_size, notA_size, verbose=False):
    """
    compute fpr, tpr, thr

    :param class_list: list of class labels
    :param hid_acts: list of normalized (0:1) hid_act values
    :param this_class: category of interest (which class is signal)
    :param classA_size: number of items from this class
    :param notA_size: number of items not in this class

    :return: roc_dict: fpr, tpr, thr, ROC_AUC
    """

    print("**** nick_roc_stuff() ****")

    if this_class > 0:

        # convert class list to binary one vs all
        binary_array = [1 if i == this_class else 0 for i in np.array(class_list)]
        hid_act_array = np.array(hid_acts)
        n_items = sum([classA_size, notA_size])


        # # get ROC curve
        fpr, tpr, thr = roc_curve(binary_array, hid_act_array)

        # # Use ROC dict stuff to compute all other needed vectors
        tp_count_dict = [classA_size * i for i in tpr]
        fp_count_dict = [notA_size * i for i in fpr]
        abv_thr_count_dict = [x + y for x, y in zip(tp_count_dict, fp_count_dict)]
        prop_above_thr_dict = [i / n_items for i in abv_thr_count_dict]
        precision_dict = [x / y if y else 0 for x, y in zip(tp_count_dict, abv_thr_count_dict)]
        recall_dict = [i / classA_size for i in tp_count_dict]
        recall2_dict = recall_dict[:-1]
        recall2_dict.insert(0, 0)
        recall_increase_dict = [x - y for x, y in zip(recall_dict, recall2_dict)]
        my_ave_prec_vals_dict = [x * y for x, y in zip(precision_dict, recall_increase_dict)]

        # # once we have all vectors, do necessary stats for whole range of activations
        roc_auc = auc(fpr, tpr)
        ave_prec = np.sum(my_ave_prec_vals_dict)
        pr_auc = auc(recall_dict, precision_dict)

        # # to sort out noZero scores
        nZ_ave_prec = np.sum(my_ave_prec_vals_dict[:-1])
        if len(recall_dict[:-1]) > 1:
            nZ_pr_auc = auc(recall_dict[:-1], precision_dict[:-1])
        else:
            nZ_pr_auc = 0.0

        # top_class_sel
        top_class_sel_rows = precision_dict.count(1)
        top_roc_stuff_dict = 'None'
        if top_class_sel_rows > 0:
            top_class_sel_recall = recall_dict[top_class_sel_rows]
            top_class_sel_recall_thr = thr[top_class_sel_rows]
            top_class_sel_items = int(tp_count_dict[top_class_sel_rows])
        else:
            top_class_sel_recall = 0
            top_class_sel_recall_thr = 1
            top_class_sel_items = 0


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

        if max_informed_prec + max_info_sens == 0.0:
            max_info_f1 = 0.0
        else:
            max_info_f1 = 2 * (max_informed_prec * max_info_sens) / (max_informed_prec + max_info_sens)
        if np.isnan(max_info_f1):
            max_info_f1 = 0.0
        if np.isfinite(max_info_f1):
            max_info_f1 = 0.0
        if type(max_info_f1) is not float:
            max_info_f1 = 0.0


    else:  # if there are not items in this class
        roc_auc = ave_prec = pr_auc = nZ_ave_prec = nZ_pr_auc = 0
        top_class_sel_items = top_class_sel_recall_thr = top_class_sel_recall = 0
        max_informed = max_informed_count = max_informed_thr = 0
        max_info_sens = max_info_spec = max_informed_prec = max_info_f1 = 0

    roc_sel_dict = {'roc_auc': roc_auc,
                    'ave_prec': ave_prec,
                    'pr_auc': pr_auc,
                    'nZ_ave_prec': nZ_ave_prec,
                    'nZ_pr_auc': nZ_pr_auc,
                    'tcs_items': top_class_sel_items,
                    'tcs_thr': top_class_sel_recall_thr,
                    'tcs_recall': top_class_sel_recall,
                    'max_informed': max_informed,
                    'max_info_count': max_informed_count,
                    'max_info_thr': max_informed_thr,
                    'max_info_sens': max_info_sens,
                    'max_info_spec': max_info_spec,
                    'max_info_prec': max_informed_prec,
                    'max_info_f1': max_info_f1,
                    }
    
    return roc_sel_dict


# # class correlation
def class_correlation(this_unit_acts, output_acts, verbose=False):
    """
    from: Revisiting the Importance of Individual Units in CNNs via Ablation
    "we can use the correlation between the activation of unit i and the predicted probability for class k as
    the amount of information carried by the unit."

    :param this_unit_acts: normaized activations from the unit
    :param output_acts: activations of the output
    :return: (Pearson's correlation coefficient, 2-tailed p-value)
    """
    print("**** class_correlation() ****")

    coef, p_val = pearsonr(x=this_unit_acts, y=output_acts)
    round_p = round(p_val, 3)
    corr = {"coef": coef, 'p': round_p}

    if verbose:
        print("corr: {}".format(corr))

    return corr


def class_sel_basics(this_unit_acts_df, items_per_cat, hi_val_thr=.5, verbose=False):
    """
    will calculate the following (count & prop, per-class and total)
    1. means: per class and total
    2. sd: per class and total
    3. non_zeros count: per-class and total)
    4. non_zeros prop: (proportion of class) per class and total
    5. above normed thr: count per-class and total
    6. above normed thr: precision per-class (e.g. proportion of items above thr per class

    :param this_unit_acts_df: dataframe containing the activations for this unit
    :param items_per_cat: Number of items in each class (of correct items or all items depending on hid acts)
    :param hi_val_thr: threshold above which an item is considered to be 'strongly active'

    :return: class_sel_basics_dict
    """

    print("**** class_sel_basics() ****")

    # # means
    means_dict = dict(this_unit_acts_df.groupby('class')['normed'].mean())
    # means_dict['total'] = this_unit_acts_df['normed'].mean()

    sd_dict = dict(this_unit_acts_df.groupby('class')['normed'].std())
    # sd_dict['total'] = this_unit_acts_df['normed'].std()

    # # non-zero_count
    n_items = len(this_unit_acts_df.index)
    nz_count_dict = dict(this_unit_acts_df[
                             this_unit_acts_df['normed'] > 0.0].groupby('class')['normed'].count())

    for i in range(max(list(items_per_cat.keys()))):
        if i not in nz_count_dict:
            nz_count_dict[i] = 0

    nz_perplexity = sum(1 for i in nz_count_dict.values() if i >= 0)
    non_zero_count_total = this_unit_acts_df[this_unit_acts_df['normed'] > 0]['normed'].count()
    # nz_count_dict['total'] = non_zero_count_total
    # nz_count_dict['perplexity'] = nz_perplexity

    # # non-zero prop
    nz_prop_dict = {k: (0 if items_per_cat[k] == 0 else nz_count_dict[k] / items_per_cat[k])
                    for k in items_per_cat.keys() & nz_count_dict}
    # nz_prop_dict['total'] = non_zero_count_total/n_items

    # # non_zero precision
    nz_prec_dict = {k: v / non_zero_count_total for k, v in nz_count_dict.items()}

    # # hi val count
    hi_val_count_dict = dict(this_unit_acts_df[
                            this_unit_acts_df['normed'] > hi_val_thr].groupby('class')['normed'].count())
    for i in range(max(list(items_per_cat.keys()))):
        if i not in hi_val_count_dict:
            hi_val_count_dict[i] = 0

    hi_val_total = this_unit_acts_df[this_unit_acts_df['normed'] > hi_val_thr]['normed'].count()

    # # hi vals precision
    hi_val_prec_dict = {k: v / hi_val_total for k, v in hi_val_count_dict.items()}


    hi_val_prop_dict = {k: (0 if items_per_cat[k] == 0 else hi_val_count_dict[k] / items_per_cat[k])
                        for k in items_per_cat.keys() & hi_val_count_dict}

    class_sel_basics_dict = {"means": means_dict, "sd": sd_dict,
                             "nz_count": nz_count_dict, "nz_prop": nz_prop_dict, 'nz_prec': nz_prec_dict,
                             "hi_val_count": hi_val_count_dict, 'hi_val_prop': hi_val_prop_dict,
                             "hi_val_prec": hi_val_prec_dict,
                             # 'hi_val_count_dict': hi_val_count_dict,
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
    :param verbose:  print stuff
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

    coi_list = list(set().union(top_hi_val_cats, lowest_nz_cats, top_mean_cats))

    if verbose is True:
        print("COI\ntop_mean_cats: {}\nlowest_nz_cats: {}\ntop_hi_val_cats:{}\ncoi_list: {}".format(
            top_mean_cats, lowest_nz_cats, top_hi_val_cats, coi_list))

    return coi_list


def sel_unit_max(all_sel_dict, verbose=False):
    """
    Script to take the analysis for multiple classes and
    return the most selective class and value for each selectivity measure.

    :param all_sel_dict: Dict of all selectivity values for this unit
    :return: small dict with just the max class for each measure
    """

    print("**** sel_unit_max() ****")

    copy_sel_dict = copy.deepcopy(all_sel_dict)

    # focussed_dict_print(copy_sel_dict, 'copy_sel_dict')

    max_sel_dict = dict()

    # # loop through unit dict of sel measure vals for each class
    for measure, class_dict in copy_sel_dict.items():
        # # for each sel measure get max value and class
        measure_c_name = "{}_c".format(measure)
        classes = list(class_dict.keys())
        values = list(class_dict.values())
        max_val = max(values)
        max_class = classes[values.index(max_val)]
        # print(measure, measure_c_name)

        # # copy max class and value to max_class_dict
        max_sel_dict[measure] = max_val
        max_sel_dict[measure_c_name] = max_class

    # # edit items (where max should be based on main sel value, e.g., not  max threshold). remove unnecessary items
    max_sel_dict['tcs_thr'] = copy_sel_dict['tcs_thr'][max_sel_dict["tcs_recall_c"]]
    del max_sel_dict['tcs_thr_c']
    max_sel_dict['tcs_items'] = copy_sel_dict['tcs_items'][max_sel_dict["tcs_recall_c"]]
    del max_sel_dict['tcs_items_c']

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
    max_sel_dict['max_info_f1'] = copy_sel_dict['max_info_f1'][max_sel_dict["max_informed_c"]]
    del max_sel_dict['max_info_f1_c']


    max_sel_dict['Zhou_selects'] = copy_sel_dict['Zhou_selects'][max_sel_dict["Zhou_prec_c"]]
    del max_sel_dict['Zhou_selects_c']
    max_sel_dict['Zhou_thr'] = copy_sel_dict['Zhou_thr'][max_sel_dict["Zhou_prec_c"]]
    del max_sel_dict['Zhou_thr_c']


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
@profile
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
    :param all_classes: Whether to test for selectivity of all classes or a subset (e.g., most active classes)
    :param layer_classes: Which layers to analyse
    :param verbose:

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
    print("os.getcwd(): {}".format(os.getcwd()))

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

    # # get basic COI list
    if all_classes is True:
        if type(items_per_cat) is int:
            classes_of_interest = list(range(n_cats))
        else:
            classes_of_interest = list({k: v for (k, v) in items_per_cat.items() if v > 0})
        print("classes_of_interest (all classes with correct items): {}".format(classes_of_interest))

    # # # get values for correct/incorrect items (1/0 or True/False)
    full_model_values = y_scores_df.full_model.unique()
    if len(full_model_values) != 2:
        print("TYPE_ERROR!: what are the scores/acc for items? {}".format(full_model_values))
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
            print("remove {} incorrect from hid_acts & output using y_scores_df.".format(n_incorrect))
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
        print("\ny_df: {}\n{}".format(y_df.shape, y_df.head()))

    # Output files
    output_filename = gha_dict["topic_info"]["output_filename"]
    print("\noutput_filename: " + output_filename)

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

    # print("\n**** opening hid_acts.pickle with shelve****")
    # hid_acts_dict = open(hid_acts_pickle, flag='c', protocol=None, writeback=False)
    # with shelve.open(hid_acts_pickle, flag='c', protocol=None, writeback=False) as shlv:
    #     hid_acts_dict = shlv

    hid_acts_keys_list = list(hid_acts_dict.keys())


    # # get output activations for class correlation
    print("hid_acts_pickle: {}".format(hid_acts_pickle))
    print("hid_acts_dict: {}".format(hid_acts_dict))
    print("hid_acts_dict.keys(): {}".format(hid_acts_dict.keys()))

    # last_layer_num = list(hid_acts_dict.keys())[-1]
    last_layer_num = hid_acts_keys_list[-1]

    print("last_layer_num: {}".format(last_layer_num))
    output_layer_acts = hid_acts_dict[last_layer_num]['2d_acts']
    output_layer_df = pd.DataFrame(output_layer_acts)
    if correct_items_only:
        if gha_incorrect:
            output_layer_df['full_model'] = y_scores_df['full_model']
            output_layer_df = output_layer_df.loc[output_layer_df['full_model'] == 1]
            output_layer_df = output_layer_df.drop(['full_model'], axis=1)
            print("\nremoving {} incorrect responses from output_layer_df: {}\n".format(n_incorrect,
                                                                                        output_layer_df.shape))
    # print("==> output_layer_df.head(): {}".format(output_layer_df.head()))

    # # close hid act dict to save memory space?
    hid_acts_dict = dict()

    # # where to save files
    current_wd = os.getcwd()
    print("current wd: {}".format(current_wd))

    analyse_items = 'all'
    if correct_items_only:
        analyse_items = 'correct'
    sel_folder = '{}_sel'.format(analyse_items)
    if test_run == True:
        sel_folder = '{}_sel/test'.format(analyse_items)
    sel_path = os.path.join(current_wd, sel_folder)

    if not os.path.exists(sel_path):
        os.makedirs(sel_path)


    # # sel_p_unit_dict
    sel_p_unit_dict = dict()

    layer_sel_mean_dict = dict()
    

    highlights_dict = {"roc_auc": [['value', 'class', 'layer', 'unit']],

                       "ave_prec": [['value', 'class', 'layer', 'unit']],

                       "pr_auc": [['value', 'class', 'layer', 'unit']],

                       "nZ_ave_prec": [['value', 'class', 'layer', 'unit']],

                       "nZ_pr_auc": [['value', 'class', 'layer', 'unit']],

                       "tcs_recall": [['value', 'class', 'thr', 'items', 'layer', 'unit']],

                       "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
                                         'prec', 'f1', 'layer', 'unit']],

                       "CCMAs": [['value', 'class', 'layer', 'unit']],

                       "Zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],

                       "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],

                       "means": [['value', 'class', 'sd', 'layer', 'unit']],

                       "nz_count": [['value', 'class', 'layer', 'unit']],

                       "nz_prop": [['value', 'class', 'layer', 'unit']],

                       "nz_prec": [['value', 'class', 'layer', 'unit']],

                       "hi_val_count": [['value', 'class', 'layer', 'unit']],

                       "hi_val_prop": [['value', 'class', 'layer', 'unit']],

                       "hi_val_prec": [['value', 'class', 'layer', 'unit']],
                       }



    # # loop through dict/layers
    n_layers = len(gha_dict['GHA_info']['model_dict'])

    if test_run is True:
        layer_counter = 0

    # for key, value in hid_acts_dict.items():
    #     layer_dict = value

    for layer_number in hid_acts_keys_list:
        print("\n**** opening hid_acts.pickle ****")
        with open(hid_acts_pickle, 'rb') as pkl:
            hid_acts_dict = pickle.load(pkl)

        # layer_dict = copy.copy(hid_acts_dict[layer_number])
        layer_dict = hid_acts_dict[layer_number]

        # # close hid act dict to save memory space?
        hid_acts_dict = dict()

        if layer_dict['layer_class'] not in layer_classes:
            continue  # skip this layer


        if test_run is True:
            layer_counter = layer_counter + 1
            if layer_counter > 5:
                continue

        # layer_number = key
        layer_name = layer_dict['layer_name']
        hid_acts_array = layer_dict['2d_acts']
        hid_acts_df = pd.DataFrame(hid_acts_array)
        print("\nloaded hidden_activation_file: {}, {}".format(hid_acts_pickle, np.shape(hid_acts_df)))
        units_per_layer = len(hid_acts_df.columns)

        # # remove incorrect responses
        if correct_items_only:
            if gha_incorrect:
                print("\nremoving {} incorrect responses from hid_acts_df: {}".format(n_incorrect, hid_acts_df.shape))
                hid_acts_df['full_model'] = y_scores_df['full_model']
                hid_acts_df = hid_acts_df.loc[hid_acts_df['full_model'] == 1]
                hid_acts_df = hid_acts_df.drop(['full_model'], axis=1)
                print("(cleaned) hid_acts_df: {}\n{}".format(hid_acts_df.shape, hid_acts_df.head()))

        layer_dict = dict()
        max_sel_dict = dict()

        print("\n**** loop through units ****")
        for unit_index, unit in enumerate(hid_acts_df.columns):

            if test_run is True:
                if unit_index > 5:
                    continue

            print("\n*************\nrunning layer {} of {} ({}): unit {} of {}\n************".format(
                layer_number, n_layers, layer_name, unit, units_per_layer))

            unit_dict = {'roc_auc': {},
                         'ave_prec': {},
                         'pr_auc': {},
                         'nZ_ave_prec': {},
                         'nZ_pr_auc': {},
                         'tcs_items': {},
                         'tcs_thr': {},
                         'tcs_recall': {},
                         'max_informed': {},
                         'max_info_count': {},
                         'max_info_thr': {},
                         'max_info_sens': {},
                         'max_info_spec': {},
                         'max_info_prec': {},
                         'max_info_f1': {},
                         'CCMAs': {},
                         'Zhou_prec': {},
                         'Zhou_selects': {},
                         'Zhou_thr': {},
                         'corr_coef': {},
                         'corr_p': {},
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

                if verbose is True:
                    print("\nthis_unit_acts_df: {}\n{}".format(this_unit_acts_df.shape, this_unit_acts_df.head()))


                # # get class_sel_basics
                class_sel_basics_dict = class_sel_basics(this_unit_acts_df, items_per_cat, verbose=verbose)

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
                    notA_size = n_correct - this_class_size

                    if verbose is True:
                        print("\nclass_{}: {} items, not_{}: {} items".format(this_cat, this_class_size,
                                                                              this_cat, notA_size))

                    # # running selectivity measures

                    # # ROC_stuff includes:
                    # roc_auc, ave_prec, pr_auc, nZ_ave_prec, nZ_pr_auc, top_class_sel, informedness
                    roc_stuff_dict = nick_roc_stuff(class_list=this_unit_acts_df['class'],
                                                    hid_acts=this_unit_acts_df['normed'],
                                                    this_class=this_cat,
                                                    classA_size=this_class_size, notA_size=notA_size,
                                                    verbose=verbose)

                    # # add class_sel_basics_dict to unit dict
                    for roc_key, roc_value in roc_stuff_dict.items():
                        unit_dict[roc_key][this_cat] = roc_value



                    # # CCMA
                    class_A = this_unit_acts_df.loc[this_unit_acts_df['class'] == this_cat]
                    class_A_mean = class_A['normed'].mean()
                    not_class_A = this_unit_acts_df.loc[this_unit_acts_df['class'] != this_cat]
                    not_class_A_mean = not_class_A['normed'].mean()
                    CCMAs = (class_A_mean - not_class_A_mean) / (class_A_mean + not_class_A_mean)
                    unit_dict["CCMAs"][this_cat] = CCMAs

                    # # Zhou_prec
                    Zhou_cut_off = .005
                    if n_correct < 20000:
                        Zhou_cut_off = 100/n_correct
                    Zhou_selects = int(n_correct * Zhou_cut_off)
                    most_active = this_unit_acts_df.iloc[:Zhou_selects]
                    Zhou_thr = list(most_active["normed"])[-1]
                    Zhou_prec = sum([1 for i in most_active['class'] if i == this_cat]) / Zhou_selects
                    unit_dict["Zhou_prec"][this_cat] = Zhou_prec
                    unit_dict["Zhou_selects"][this_cat] = Zhou_selects
                    unit_dict["Zhou_thr"][this_cat] = Zhou_thr

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

                print("Unit {} sel summary".format(unit))
                focussed_dict_print(max_sel_p_unit_dict, 'max_sel_p_unit_dict, unit {}'.format(unit))

                # # ADD UNIT DICT TO LAYER
                layer_dict[unit] = unit_dict



            else:
                print("dead unit found")
                unit_dict = 'dead_unit'

        print("********\nfinished looping through units\n************")

        # # make max per unit csv

        
        # todo: only save summary docs as csv.  all other output should be numpy, pickle or excel.
        #  or for csv use this https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv
        #  nick_to_csv and nick_read_csv
        max_sel_csv_name = '{}/{}_{}_sel_p_unit.csv'.format(sel_path, output_filename, layer_name)
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
            for measure in list(unit_dict.keys()):
                layer_unit_sel_list = []
                if measure == 'max':
                    continue
                for unit, unit_sel_dict in max_sel_dict.items():
                    layer_unit_sel_list.append(unit_sel_dict[measure])
                layer_measure_mean = np.mean(layer_unit_sel_list)
                layer_sel_mean_dict[measure] = layer_measure_mean
        # focussed_dict_print(layer_sel_mean_dict, 'layer_sel_mean_dict')
        layer_dict['means'] = layer_sel_mean_dict



        # layer_sel_mean_dict[layer_name]
        # layer_means_csv_name = '{}/{}_sel_p_layer.csv'.format(sel_path, output_filename)
        # layer_means_df = pd.DataFrame.from_dict(data=layer_sel_mean_dict, orient='index')
        # layer_means_df.to_csv(layer_means_csv_name)
        # # nick_to_csv(layer_means_df, layer_means_csv_name)


        # # # get top three highlights for each feature




        highlights_list = list(highlights_dict.keys())

        for measure in highlights_list:
            # sort max_sel_df by sel measure for this layer
            layer_top_3_df = max_sel_df.sort_values(by=measure, ascending=False).reset_index()

            # then select info for highlights dict for top 3 items

            print("layer_top_3_df\n{}".format(layer_top_3_df))
            print("layer_top_3_df.loc[measure]\n{}".format(layer_top_3_df[measure]))

            if measure == 'tcs_recall':
                print('tcs_recall')  #  "tcs_recall": [['value', 'class', 'thr', 'items', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = \
                            list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)])[i]
                        thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_thr'])[i]
                        items = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_items'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]
                    else:
                        top_class = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, '{}_c'.format(measure)].item()
                        thr = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, 'tcs_thr'].item()
                        items = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'tcs_items'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()

                    new_row = [top_val, top_class, thr, items, layer_name, top_unit_name]
                    print("new_row\n{}".format(new_row))
                    highlights_dict[measure].append(new_row)

            elif measure == 'max_informed':
                print('max_informed')
                # "max_informed": [['value', 'class', 'count', 'thr', 'sens', 'spec',
                #                                          'prec', 'f1', 'count', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = \
                            list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)])[i]
                        count = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_count'])[i]
                        thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_thr'])[i]
                        sens = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_sens'])[i]
                        spec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_spec'])[i]
                        prec = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_prec'])[i]
                        f1 = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_f1'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]

                    else:
                        top_class = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, '{}_c'.format(measure)].item()
                        count = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_count'].item()
                        thr = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, 'max_info_thr'].item()
                        sens = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_sens'].item()
                        spec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_spec'].item()
                        prec = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_prec'].item()
                        f1 = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'max_info_f1'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()

                    new_row = [top_val, top_class, count, thr, sens, spec, prec, f1, layer_name, top_unit_name]
                    print("new_row\n{}".format(new_row))
                    highlights_dict[measure].append(new_row)

            elif measure == 'Zhou_prec':
                print('Zhou_prec')
                # "Zhou_prec": [['value', 'class', 'thr', 'selects', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = \
                            list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)])[i]
                        thr = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'Zhou_thr'])[i]
                        selects = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'Zhou_selects'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]

                    else:
                        top_class = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, '{}_c'.format(measure)].item()
                        thr = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, 'Zhou_thr'].item()
                        selects = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'Zhou_selects'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()

                    new_row = [top_val, top_class, thr, selects, layer_name, top_unit_name]
                    print("new_row\n{}".format(new_row))
                    highlights_dict[measure].append(new_row)

            elif measure == 'corr_coef':
                print('corr_coef')
                #   "corr_coef": [['value', 'class', 'p', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if np.isnan(top_val):
                        continue
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = \
                            list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)])[i]
                        p = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'corr_p'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]

                    else:
                        top_class = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, '{}_c'.format(measure)].item()
                        p = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, 'corr_p'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()

                    new_row = [top_val, top_class, p, layer_name, top_unit_name]
                    print("new_row\n{}".format(new_row))
                    highlights_dict[measure].append(new_row)

            elif measure == 'means':
                print('means')
                 #                        "means": [['value', 'class', 'sd', 'layer', 'unit']],
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = \
                            list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)])[i]
                        sd = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'sd'])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]

                    else:
                        top_class = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, '{}_c'.format(measure)].item()
                        sd = layer_top_3_df.loc[
                                layer_top_3_df[measure] == top_val, 'sd'].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()

                    new_row = [top_val, top_class, sd, layer_name, top_unit_name]
                    print("new_row\n{}".format(new_row))
                    highlights_dict[measure].append(new_row)

            else:  # for most most measures use below
                for i in range(2):
                    top_val = layer_top_3_df[measure].iloc[i]
                    # print('top_val: ', top_val)
                    # print(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])
                    if len(list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])) > 1:
                        top_class = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)])[i]
                        top_unit_name = list(layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'])[i]

                    else:
                        top_class = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, '{}_c'.format(measure)].item()
                        top_unit_name = layer_top_3_df.loc[layer_top_3_df[measure] == top_val, 'unit'].item()

                    new_row = [top_val, top_class, layer_name, top_unit_name]

                    print("new_row\n{}".format(new_row))
                    highlights_dict[measure].append(new_row)

            print("\nhighlights_dict[{}]\n{}".format(measure, highlights_dict[measure]))

        print("\nEND highlights_dict\n{}".format(highlights_dict))

        # # append layer dict to model dict
        sel_p_unit_dict[layer_name] = layer_dict

        # print("layer_sel_mean_dict:\n{}".format(layer_sel_mean_dict))




    highlights_df_dict = dict()
    for k, v in highlights_dict.items():
        # header= v[0]
        hl_df = pd.DataFrame(data=v[1:], columns=v[0])
        hl_df.sort_values(by='value', ascending=False, inplace=True)
        highlights_df_dict[k] = hl_df
        # hl_df.columns
        if verbose:
            print("\n{} highlights (head)".format(k))
            print(hl_df)


    sel_highlights_pickle_name = "{}/{}_highlights.pickle".format(sel_path, output_filename)
    pickle_out = open(sel_highlights_pickle_name, "wb")
    pickle.dump(highlights_df_dict, pickle_out)
    pickle_out.close()

    sel_per_unit_pickle_name = "{}/{}_sel_per_unit.pickle".format(sel_path, output_filename)
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

    sel_dict_name = "{}/{}_sel_dict.pickle".format(sel_path, output_filename)

    master_dict["sel_info"] = {"sel_path": sel_path,
                               'sel_dict_name': sel_dict_name,
                               "sel_per_unit_pickle_name": sel_per_unit_pickle_name,
                               'sel_highlights_pickle_name': sel_highlights_pickle_name,
                               "correct_items_only": correct_items_only,
                               "all_classes": all_classes, "layer_classes": layer_classes,
                               "sel_date": int(datetime.datetime.now().strftime("%y%m%d")),
                               "sel_time": int(datetime.datetime.now().strftime("%H%M")),
                               }

    print("Saving dict to: {}".format(os.getcwd()))
    pickle_out = open(sel_dict_name, "wb")
    pickle.dump(master_dict, pickle_out)
    pickle_out.close()

    focussed_dict_print(master_dict, "master_dict")
    # print_nested_round_floats(master_dict)

    print("\nend of sel script")

    return master_dict  # , mean_sel_per_NN


# # #
# print("\nWARNING\n\nrunning test from bottom of sel script\nWARNING")
# sel_dict = ff_sel(gha_dict_path='/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/'
#                                 'CIFAR10_models_c6p3_adam_bn/all_test_set_gha/'
#                                 'CIFAR10_models_c6p3_adam_bn_GHA_dict.pickle',
#                   correct_items_only=True, all_classes=True,
#                   verbose=True, test_run=True
#                   )

# test_dict_path = "/home/nm13850/Documents/PhD/python_v2/experiments/train_script_check/train_script_check_fc2_iris/" \
#                  "correct_train_set_gha/correct_sel/train_script_check_fc2_iris_test_dict.pickle"
#
# test_dict = load_dict(test_dict_path)
# max_sel_p_unit_dict = sel_unit_max(test_dict['fc_1'][0])
#
# print("\nmax_sel_p_unit_dict:")
# focussed_dict_print(max_sel_p_unit_dict)

# sel_p_unit_dict = ff_sel(exp_cond_gha_path='train_script_check/train_script_check_fc2_iris/correct_train_set_gha',
#                   correct_items_only=True,
#                   all_classes=False,
#                   verbose=True)

# sel_p_unit_dict = ff_sel(exp_cond_gha_path='train_script_check/train_script_check_fc2_iris_None/correct_train_set_gha',
#                   correct_items_only=True,
#                   all_classes=False,
#                   verbose=True)

# sel_p_unit_dict = ff_sel(exp_cond_gha_path='train_script_check/train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/all_test_set_gha',
#                            correct_items_only=True,
#                            all_classes=False,
#                            verbose=True)

# train_script_check/train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug/all_test_set_gha/train_script_check_con6_pool3_fc1_CIFAR_10_2019_aug_GHA_dict.pickle

# sel_per_unit = ff_sel(exp_cond_gha_path='vgg/vgg_imagenet_20k_30k/all_objects/ILSVRC2012/val_part3_gha',
#                       correct_items_only=False,
#                       all_classes=False,
#                       verbose=True)


'''
get mean sel per layer and plot it

lesion_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/CIFAR10_models_c4p2_adam_bn/' \
                   'all_test_set_gha/lesion/CIFAR10_models_c4p2_adam_bn_lesion_dict.pickle'
sel_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/CIFAR10_models/CIFAR10_models_c4p2_adam_bn/' \
                'all_test_set_gha/correct_sel/CIFAR10_models_c4p2_adam_bn_sel_dict.pickle'


sel_dict = load_dict(sel_dict_path)
focussed_dict_print(sel_dict, "sel_dict", focus_list=['sel_info'])

sel_p_u_dict = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
print(sel_p_u_dict.keys())

print(sel_p_u_dict['conv2d_19'][0].keys())

print(sel_p_u_dict['conv2d_19'][0]['max'])

# measure_list = ['roc_auc', 'ave_prec', 'pr_auc', 'nZ_ave_prec', 'nZ_pr_auc', 'tcs_items', 'tcs_thr', 'tcs_recall', 'max_informed', 'max_info_count', 'max_info_thr', 'max_info_sens', 'max_info_spec', 'max_info_prec', 'max_info_f1', 'CCMAs', 'Zhou_prec', 'Zhou_selects', 'Zhou_thr', 'corr_coef', 'corr_p', 'means', 'sd', 'nz_count', 'nz_prop', 'nz_prec', 'hi_val_count', 'hi_val_prop', 'hi_val_prec', 'max'])
# measure_list = ['roc_auc', 'ave_prec', 'pr_auc', 'tcs_recall', 'max_informed', 'CCMAs', 'Zhou_prec', 'corr_coef']
layers_list = ['conv2d_19', 'activation_25', 'max_pooling2d_10', 'conv2d_20', 'activation_26', 'conv2d_21',
               'activation_27', 'max_pooling2d_11', 'conv2d_22', 'activation_28', 'conv2d_23', 'activation_29',
               'conv2d_24', 'activation_30', 'max_pooling2d_12', 'dense_4', 'activation_31']

measure_list = ['max_informed', 'CCMAs', 'Zhou_prec']

sel_means_dict = {'max_informed': {}, 
                  'CCMAs': {}, 
                  'Zhou_prec': {}}

for measure in measure_list:
    for layer, layer_dict in sel_p_u_dict.items():
        if layer not in layers_list:
            continue
        print("layer: ", layer)
        sel_means_dict[measure][layer] = dict()
        layer_sel_list = []
        for unit, unit_dict in layer_dict.items():
            # print("unit: ", unit)
            max_sel_val = unit_dict['max'][measure]
            # print("max_sel_val:\t", max_sel_val)
            layer_sel_list.append(max_sel_val)
        sel_means_dict[measure][layer]['values'] = layer_sel_list
        sel_means_dict[measure][layer]['mean'] = np.mean(layer_sel_list)

focussed_dict_print(sel_means_dict)


# conv_list = ['conv2d_19', 'conv2d_20', 'conv2d_21', 'conv2d_22', 'conv2d_23', 'conv2d_24', 'dense_4', ]
act_list = ['activation_25', 'activation_26', 'activation_27', 'activation_28', 'activation_29', 'activation_30', 'activation_31']

print(os.getcwd())
sel_dir, _ = os.path.split(sel_dict_path)
os.chdir(sel_dir)
print(os.getcwd())

for measure in measure_list:
    for layer in act_list:
        sns.distplot(sel_means_dict[measure][layer]['values'], kde=False, label=layer)
    # Plot formatting
    plt.legend(prop={'size': 12})
    plt.title('{} per layer'.format(measure))
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')

    plt.savefig("{}_dist_in_act_layers.png".format(measure))
    # plt.show()
    
    '''


