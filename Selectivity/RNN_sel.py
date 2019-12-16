import copy
import csv
import datetime
import json
import os
import pickle
import shelve

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, rankdata
from sklearn.metrics import roc_curve, auc

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.dicts import nested_dict_to_df
from tools.RNN_STM import get_X_and_Y_data_from_seq, seq_items_per_class
from tools.RNN_STM import spell_label_seqs, word_letter_combo_dict
from tools.data import nick_read_csv, find_path_to_dir
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
    "we can use the correlation between the activation of unit i
    and the predicted probability for class k as
    the amount of information carried by the unit."

    :param this_unit_acts: normaized activations from the unit
    :param output_acts: activations of the output
    :param verbose: activations of the output

    :return: (Pearson's correlation coefficient, 2-tailed p-value)
    """
    if verbose:
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
    means_dict = dict(this_unit_acts_df.groupby('label')[act_values].mean())

    # # sd.  will give value of nan if there is no variance rather then 0.
    sd_dict = dict(this_unit_acts_df.groupby('label')[act_values].std())
    sd_dict = {k: v if not np.isnan(v) else 0 for k, v in sd_dict.items()}


    # print(f"\nidiot check sd\n"
    #       f"means: {len(means_dict.values())} {means_dict}\n"
    #       f"sd: {len(sd_dict.values())} {sd_dict}\n"
    #       f"items_per_cat: {items_per_cat}\n"
    #       f"this_unit_acts_df:\n{this_unit_acts_df}")
    # for cat in range(n_classes):
    #     print(f"\ncat: {cat}\n"
    #           f"{this_unit_acts_df[this_unit_acts_df['label'] == cat]}")
    # # if all(list(sd_dict.values())) == 'Nan'

    # # non-zero_count
    nz_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values]
                                           > 0.0].groupby('label')[act_values].count())

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
    nz_prec_dict = {k: v / non_zero_count_total for k, v in nz_count_dict.items()}

    # # hi val count
    hi_val_count_dict = dict(this_unit_acts_df[this_unit_acts_df[act_values] >
                                               hi_val_thr].groupby('label')[act_values].count())
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

    if verbose:
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

        # print("\nidiot check\n"
        #       f"measure: {measure}\nvalues:\n{values}")
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
    :param all_or_max: for each unit/timestep, dict contains either:
        sel values for all classes or just max_sel_class

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


###############################

def get_sel_summaries(max_sel_dict_path,
                      top_n=3,
                      high_sel_thr=1.0,
                      verbose=False):
    """
    max sel dict is a nested dict [layers][units][timesteps][measures] = floats.
    Use this to make the following

    1. max_sel_df: from max_sel_dict make df.  multi-indexed for layer, units, ts as rows.
        Cols are all sel measures

    2. model_mean_max_df: use max_sel_df to get mean and max sel scores for:
        a) whole model, b) each layer and c) each timestep (ts)

    3. for_summ_csv_dict: get model means and max's for a couple of measures
        (max_info, ccma, precision, means) to use on the summary df.

    4. highlights dicts
        a) hl_dfs_dict (highlights_dataframe_dict):
            df with top_n units/timesteps for each sel measure.
            keys: measures; values: dataframes

        b) hl_units_dict (highlights unit dict).  nested keys: [layer][unit][timestep];
            values at unit: list of measures where it is timestep invariant
            (e.g., for a given measure, max sel class is the same at all timesteps)
            values at timesteps: tuple(measure, score, label) from hl_dfs_dict.

    5. get_sel_summaries_dict: details of everything in this function (e.g., paths to files)

    1-4 are all saved directly.
    5 is returned.

    :param max_sel_dict_path:
    :param top_n: how many of the most selective units to save in highlights
    :param high_sel_thr: above what threshold should all units be saved
    :param verbose: How much to print to screen

    :return: get_sel_summaries_dict
    """

    # # use max_sel_dict_path to get exp_cond_gha_path, max_sel_dict_name,
    exp_cond_gha_path, max_sel_dict_name = os.path.split(max_sel_dict_path)
    output_filename = max_sel_dict_name[:-22]

    print(f"\n**** get_sel_summaries ({output_filename}) ****")

    os.chdir(exp_cond_gha_path)

    max_sel_dict = load_dict(max_sel_dict_path)

    focussed_dict_print(max_sel_dict, 'max_sel_dict')

    # # max_sel_p_unit layout
    """print("\nORIG max_sel_p_unit dict")
    print(f"\nFirst nest is Layers: {len(list(max_sel_dict.keys()))} keys."
          f"\n{list(max_sel_dict.keys())}"
          f"\neach with value is a dict.")
    second_layer = max_sel_dict[list(max_sel_dict.keys())[0]]
    print(f"\nsecond nest is Units: {len(list(second_layer.keys()))} keys."
          f"\n{list(second_layer.keys())}"
          f"\neach with value is a dict.")
    third_layer = second_layer[list(second_layer.keys())[0]]
    print(f"\nThird nest is Timesteps: {len(list(third_layer.keys()))} keys."
          f"\n{list(third_layer.keys())}"
          f"\neach with value is a dict.")
    fourth_layer = third_layer[list(third_layer.keys())[0]]
    print(f"\nfourth nest is measures: {len(list(fourth_layer.keys()))} keys."
          f"\n{list(fourth_layer.keys())}"
          f"\neach with value is a single item.  "
          f"{fourth_layer[list(fourth_layer.keys())[0]]}\n")"""

    '''get a list of relevant sel measures'''
    # # list of all keys (sel_measures)
    all_sel_measures_list = list(max_sel_dict[list(max_sel_dict.keys())[0]][0]['ts0'].keys())

    # # remove measures that don't have an associated class-label
    # # also removing nz_count and hi-val count as they have scores > 1.
    drop_these_measures = ['max_info_count', 'max_info_thr', 'max_info_sens',
                           'max_info_spec', 'max_info_prec', 'zhou_selects', 'zhou_thr',
                           # # remove nz and hi val for rnn stuff.  not informative
                           "nz_count", 'nz_count_c', "nz_prop", "nz_prop_c", "nz_prec", "nz_prec_c",
                           'hi_val_count', 'hi_val_count_c',
                           "hi_val_prop", "hi_val_prop_c", "hi_val_prec", "hi_val_prec_c",
                           ]

    sel_measures_list = [measure for measure in all_sel_measures_list if measure not in drop_these_measures]

    # # remove max_sel_class labels associated with sel measures
    sel_measures_list = [measure for measure in sel_measures_list if measure[-2:] != '_c']
    if verbose:
        print(f"{len(sel_measures_list)} items in sel_measures_list.\n{sel_measures_list}")

    '''1. max_sel_df: from max_sel_dict
    reform nested dict first
    https://stackoverflow.com/questions/30384581/nested-dictionary-to-multiindex-pandas-dataframe-3-level
    '''
    # reform_nested_sel_dict = {(level1_key, level2_key, level3_key): values
    #                           for level1_key, level2_dict in max_sel_dict.items()
    #                           for level2_key, level3_dict in level2_dict.items()
    #                           for level3_key, values in level3_dict.items()}
    #
    # max_sel_df = pd.DataFrame(reform_nested_sel_dict).T
    # sel_df_index_names = ['Layer', 'Unit', 'Timestep']
    # max_sel_df.index.set_names(sel_df_index_names, inplace=True)

    max_sel_df = nested_dict_to_df(max_sel_dict)

    # # convert max_sel_class labels ('_c') columns to int
    class_cols_list = [measure for measure in all_sel_measures_list if measure[-2:] == '_c']
    class_cols_dict = {i: 'int32' for i in class_cols_list}
    max_sel_df = max_sel_df.astype(class_cols_dict)

    max_sel_df.to_csv(f'{output_filename}_max_sel.csv')

    if verbose:
        print(f"\nmax_sel_df:\n{max_sel_df}")

    '''use max_sel_df.xs (cross-section) to select info in multi-indexed dataframes'''
    # layer_df = max_sel_df.xs('hid2', level='Layer')
    # unit_df = max_sel_df.xs(1, level='Unit')
    # ts_df = max_sel_df.xs('ts2', level='Timestep')
    # layer_unit_df = max_sel_df.xs(('hid2', 2), level=('Layer', 'Unit'))
    # layer_ts_df = max_sel_df.xs(('hid2', 'ts2'), level=('Layer', 'Timestep'))
    # unit_ts_df = max_sel_df.xs((2, 'ts2'), level=('Unit', 'Timestep'))
    # print(unit_ts_df)

    '''2a. model_mean_max_df: get means and max for the whole model'''
    # # sel_measures_df just contains sel values, not max_sel_class labels
    sel_measures_df = max_sel_df[sel_measures_list]
    model_means_s = sel_measures_df.mean().rename('model_means')
    model_max_s = sel_measures_df.max().rename('model_max')


    # # plot distribution of selectivity scores
    colours = sns.color_palette('husl', n_colors=len(sel_measures_list))
    plt.figure()
    for index, measure in enumerate(sel_measures_list):

        # print(f"\nindex: {index}: measure: {measure}\n{sel_measures_df[measure]}")
        check_values = sel_measures_df[measure].to_list()
        # print(f"check_values: {check_values}")

        sel_measures_df.dropna(subset=[measure], inplace=True)
        # print(f"\ndropna:\n{sel_measures_df[measure]}")

        if not sel_measures_df.empty:
            ax = sns.kdeplot(sel_measures_df[measure], color=colours[index], shade=True)
    plt.legend(sel_measures_list)
    plt.title('Density Plot of Selectivity measures')
    ax.set(xlabel='Selectivity')
    ax.set_xlim(right=1)
    plt.savefig(f"{output_filename}_sel_dist.png")
    # plt.show()
    plt.close()

    # # don't make df yet, put here so layers and timesteps can be added too
    mean_max_arrays = [model_means_s, model_max_s]

    '''2b. model_mean_max_df: get means and max per layer'''
    layer_names_list = sorted(list(set(max_sel_df.index.get_level_values('Layer'))))
    units_names_list = sorted(list(set(max_sel_df.index.get_level_values('Unit'))))
    ts_names_list = sorted(list(set(max_sel_df.index.get_level_values('Timestep'))))

    if len(layer_names_list) > 1:
        for this_layer in layer_names_list:

            # # check that relevant timestep is present
            layer_check = sorted(list(set(sel_measures_df.index.get_level_values('Layer'))))

            if this_layer in layer_check:

                # # select relevant rows from list
                layer_measure_df = sel_measures_df.xs(this_layer, level='Layer')

                # # get means and max vales series
                layer_means_s = layer_measure_df.mean().rename(f'{this_layer}_means')
                layer_max_s = layer_measure_df.max().rename(f'{this_layer}_max')
            else:
                headers = list(sel_measures_df)
                values = [0] * len(headers)
                layer_means_s = pd.Series(dict(zip(headers, values))).rename(f'{this_layer}_means')
                layer_max_s = pd.Series(dict(zip(headers, values))).rename(f'{this_layer}_max')

            mean_max_arrays.append(layer_means_s)
            mean_max_arrays.append(layer_max_s)

    '''2c. model_mean_max_df: get means and max per timestep'''
    if len(ts_names_list) > 1:
        for this_ts in ts_names_list:

            # # check that relevant timestep is present
            ts_check = sorted(list(set(sel_measures_df.index.get_level_values('Timestep'))))
            # print(f"\nts_checl: {ts_check}")

            if this_ts in ts_check:

                # # select relevant rows from list
                ts_measure_df = sel_measures_df.xs(this_ts, level='Timestep')

                # # get means and max vales series
                ts_means_s = ts_measure_df.mean().rename(f'{this_ts}_means')
                ts_max_s = ts_measure_df.max().rename(f'{this_ts}_max')

            else:
                # print("idiot check ts missing")
                headers = list(sel_measures_df)
                values = [0] * len(headers)
                ts_means_s = pd.Series(dict(zip(headers, values))).rename(f'{this_ts}_means')
                ts_max_s = pd.Series(dict(zip(headers, values))).rename(f'{this_ts}_max')

            # print(f"\nmodel_means_s:\n{model_means_s}")

            mean_max_arrays.append(ts_means_s)
            mean_max_arrays.append(ts_max_s)

    model_mean_max_df = pd.concat(mean_max_arrays, axis='columns')

    model_mean_max_df.to_csv(f'{output_filename}_model_mean_max.csv')

    if verbose:
        print(f"model_mean_max_df: \n{model_mean_max_df}")

    '''3. for_summ_csv_dict: summary values from model means and max serieses'''
    for_summ_csv_dict = {"mi_mean": model_means_s.loc['max_informed'],
                         "mi_max": model_max_s.loc['max_informed'],
                         "ccma_mean": model_means_s.loc['ccma'],
                         "ccma_max": model_max_s.loc['ccma'],
                         "prec_mean": model_means_s.loc['zhou_prec'],
                         "prec_max": model_max_s.loc['zhou_prec'],
                         "means_mean": model_means_s.loc['means'],
                         "means_max": model_max_s.loc['means'],
                         "b_sel_mean": model_means_s.loc['b_sel'],
                         "b_sel_max": model_max_s.loc['b_sel'],
                         }

    '''4a. hl_dfs_dict (highlights_dataframe_dict):  
        df with top_n units/timesteps for each sel measure.
        keys: measures; values: dataframes

       4b. hl_units_dict (highlights unit dict).  nested keys: [layer][unit][timestep];
            values at timesteps: tuple(measure, score, label) from hl_dfs_dict.'''

    hl_dfs_dict = dict()
    hl_units_dict = dict()


    for measure in sel_measures_list:
        # # save all units at or above high_sel_threshold
        if len(max_sel_df[max_sel_df[measure] >= high_sel_thr]) > top_n:
            top_units = max_sel_df[max_sel_df[measure] >= high_sel_thr]
        else:
            # make sure column is float
            max_sel_df[measure] = max_sel_df[measure].astype('float')

            # # just take top_n highest scores
            top_units = max_sel_df.nlargest(n=top_n, columns=measure)

        # # only include relevant measure and class label on this df
        m_label = f"{measure}_c"
        cols_to_keep = [measure, m_label]
        top_units = top_units[cols_to_keep]

        # # get ranks for scores
        check_values = top_units.loc[:, measure].to_list()

        if all(check_values) == 0:
            print(f"\nthese values are all 0\nmeasure: {measure}\n{check_values}")
            continue
        if all(check_values) == 0.0:
            print(f"\nthese values are all 0.0\nmeasure: {measure}\n{check_values}")
            continue

        rank_values = rankdata([int(-100 * i) for i in check_values], method='dense')

        top_units['rank'] = rank_values

        # # append to hl_dfs_dict
        hl_dfs_dict[measure] = top_units

        # # loop through top units to populate hl_units_dict
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
                (measure, round(row.values[0], 3), int(row.values[1]), f'rank_{int(row.values[2])}'))

    # save hl_dfs_dict here
    with open(f"{output_filename}_hl_dfs.pickle", "wb") as pickle_out:
        pickle.dump(hl_dfs_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        focussed_dict_print(hl_dfs_dict, 'hl_dfs_dict')

    '''4b. hl_units_dict (highlights unit dict).  nested keys: [layer][unit][timestep];
            values at unit: list of measures where it is timestep invariant'''

    for layer in layer_names_list:
        for unit in units_names_list:
            # # get array for this layer, unit - all timesteps
            layer_unit_df = max_sel_df.xs((layer, unit), level=('Layer', 'Unit'))

            if verbose:
                print(f"\nlayer_unit_df:\n{layer_unit_df}")

            for measure in sel_measures_list:
                # # get max sel label for these timesteps
                check_classes = layer_unit_df.loc[:, f'{measure}_c'].to_list()
                # # if there is only 1 label (i.e., all timesteps have same max_sel_class)
                if len(set(check_classes)) == 1:

                    # # check vals are not all zero
                    check_values = layer_unit_df.loc[:, measure].to_list()

                    if verbose:
                        print(f"\ncheck_values ({measure}): {check_values}")

                    if all(check_values) > 0:

                        # # add indices/keys to dict
                        if layer not in hl_units_dict.keys():
                            hl_units_dict[layer] = dict()
                        if unit not in hl_units_dict[layer].keys():
                            hl_units_dict[layer][unit] = dict()
                        if 'ts_invar' not in hl_units_dict[layer][unit]:
                            hl_units_dict[layer][unit]['ts_invar'] = []

                        # # add measure name to dict
                        hl_units_dict[layer][unit]['ts_invar'].append(measure)

    # save hl_units_dict here as json for easy reading
    # with open(f"{output_filename}_hl_units.pickle", "wb") as pickle_out:
    #     pickle.dump(hl_units_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{output_filename}_hl_units.txt", 'w') as fp:
        json.dump(hl_units_dict, fp, indent=4, separators=(',', ':'))

    if verbose:
        print_nested_round_floats(hl_units_dict, 'hl_units_dict')

    '''5. get_sel_summaries_dict: details of everything in this function 
        (e.g., paths to files) '''

    get_sel_summaries_dict = {
        "max_sel_dict_path": max_sel_dict_path,
        "top_n": top_n,
        "high_sel_thr": high_sel_thr,
        "for_summ_csv_dict": for_summ_csv_dict,
        "max_sel_df_name": f'{output_filename}_max_sel.csv',
        "model_mean_max_df_name": f'{output_filename}_model_mean_max.csv',
        "hl_dfs_dict_name": f"{output_filename}_hl_dfs.pickle",
        "hl_units_dict_name": f"{output_filename}_hl_units.txt",
        "sel_dist_plot_name": f"{output_filename}_sel_dist.png",
        "get_sel_summaries_date": int(datetime.datetime.now().strftime("%y%m%d")),
        "get_sel_summaries_time": int(datetime.datetime.now().strftime("%H%M"))
    }

    return get_sel_summaries_dict


####################################################################################################
def count_sel_units(word_sel_dict_path, measure='b_sel',
                    thresholds=[0.0, .1, .2, .3, .4, .5],
                    just_1st_ts=False,
                    save_csv=True):
    """
    Given a dataset where selectivity has already been run for letters and words.

    1. sel dict, max_sel_per_unit dict for words,letters and combo
    2. re-structure dicts to dfs
    3. count considering each timestep (ts)
        Count n_ts with sel > .0, .1, .2, .3, .4, .5 for words, letters, combo
        Count n_cats in all ts with sel > .0, .1, .2, .3, .4, .5 for words, letters, com
    4. make df/dict just consisisting of units where word/sel values are the same at all timesteps
    5. count considering units where class is the same at all timesteps, using min sel as value
        Count n_ts with sel > .0, .1, .2, .3, .4, .5 for words, letters, combo
        Count n_cats in all ts with sel > .0, .1, .2, .3, .4, .5 for words, letters, comb
    6. option to create or append results to summary doc with scores for each model

    :param word_sel_dict_path:
    :param measure: which selectivity measure to look at.
    :param thresholds: which selectivity thresholds to consider
    :param save_csv: Whether to save a summary doc

    :return: Dict:  Thr [.0, .1, .2, .3, .4, .5]:
                    Level ['word', 'letter', 'combo']
                    count ['timesteps', 'ts_cats', 'invar_units', 'invar_cats']
    """

    # # # 1 load basic info
    # # load sel dict.
    if type(word_sel_dict_path) is dict:
        sel_dict = word_sel_dict_path
    elif type(word_sel_dict_path) is str:
        if os.path.isfile(word_sel_dict_path):
            sel_dict = load_dict(word_sel_dict_path)
        else:
            raise TypeError(f"Sel_dict path should be a dict or path to dict\n"
                            f"{word_sel_dict_path}")
    else:
        raise TypeError(f"Sel_dict path should be a dict or path to dict\n"
                        f"{word_sel_dict_path}")
    focussed_dict_print(sel_dict)


    # # load max sel per unit dict for words and letters.
    sel_path = sel_dict['sel_info']['sel_path']
    word_sel_dict_name = sel_dict['sel_info']['max_sel_dict_name']
    word_sel_dict = load_dict(os.path.join(sel_path, word_sel_dict_name))
    # focussed_dict_print(word_sel_dict, 'word_sel_dict')  #, focus_list=['hid0'])

    if word_sel_dict_name[-3:] == 'txt':
        word_sel_dict_prefix = word_sel_dict_name[:-18]
        word_sel_dict_suffix = word_sel_dict_name[-18:]
    elif word_sel_dict_name[-3:] == 'kle':
        word_sel_dict_prefix = word_sel_dict_name[:-21]
        word_sel_dict_suffix = word_sel_dict_name[-21:]

    letter_sel_dict_name = f"{word_sel_dict_prefix}lett_{word_sel_dict_suffix}"
    letter_sel_dict = load_dict(os.path.join(sel_path, letter_sel_dict_name))
    # focussed_dict_print(letter_sel_dict, 'letter_sel_dict')  #, focus_list=['hid0'])

    combo_dict = word_letter_combo_dict(word_sel_dict_path)
    # focussed_dict_print(combo_dict, 'combo_dict')



    # # # 2 re-structure data
    # # - flatten word_dict and letter_dict, combo into dfs
    # # - keys:     layer, unit, ts, measure
    # # - values:   word_sel, word_class,
    # #             letter_sel, letter_class,
    # #             combo_level (word, letter), combo_sel, combo_class
    word_sel_df = nested_dict_to_df(word_sel_dict)
    sel_columns = [measure, f'{measure}_c']
    word_sel_df = word_sel_df[sel_columns]
    word_sel_df = word_sel_df.astype({measure: 'float32',
                                      f'{measure}_c': 'int32'})
    word_sel_df = word_sel_df.rename(columns={measure: f"word_sel", f'{measure}_c': f'word_c'})
    # print(f"word_sel_df:\n{word_sel_df.head()}")

    letter_sel_df = nested_dict_to_df(letter_sel_dict)
    letter_sel_df = letter_sel_df[sel_columns]
    letter_sel_df = letter_sel_df.astype({measure: 'float32',
                                          f'{measure}_c': 'int32'})
    letter_sel_df = letter_sel_df.rename(columns={measure: f"letter_sel", f'{measure}_c': f'letter_c'})
    # print(f"letter_sel_df:\n{letter_sel_df.head()}")

    combo_sel_df = nested_dict_to_df(combo_dict)
    combo_sel_df = combo_sel_df.rename(columns={'level': 'level',
                                                'sel': f"combo_sel",
                                                'feat': f'combo_c'})
    # print(f"combo_sel_df:\n{combo_sel_df.head()}")




    # # - use vocab dict to create columns for letter_feat, word_feat
    vocab_dict_path = os.path.join(sel_dict['data_info']['data_path'],
                                   sel_dict['data_info']['vocab_dict'])
    vocab_dict = load_dict(vocab_dict_path)
    # focussed_dict_print(vocab_dict)

    # # # get actual words and letters not numeric label
    # # word feat
    word_feat = []
    for cat in list(word_sel_df['word_c']):
        this_feat = vocab_dict[cat]['word']
        word_feat.append(this_feat)
        # print(cat)
    word_sel_df['word_feat'] = word_feat

    letter_id_dict = load_dict(os.path.join(sel_dict['data_info']['data_path'],
                               sel_dict['data_info']['letter_id_dict']))
    # focussed_dict_print(letter_id_dict, 'letter_id_dict')

    # # letter feat
    letter_feat = []
    for cat in list(letter_sel_df['letter_c']):
        this_feat = letter_id_dict[cat]
        letter_feat.append(this_feat)
        # print(cat)
    letter_sel_df['letter_feat'] = letter_feat


    # # - letter in word bool (if letter_feat is in word_feat)
    letter_in_word = []
    for letter, word in zip(letter_feat, word_feat):
        # print(letter, word)
        if letter in word:
            letter_in_word.append(True)
        else:
            letter_in_word.append(False)
    letter_sel_df['letter_in_word'] = letter_in_word



    # # merge dfs
    sel_df = word_sel_df.join(letter_sel_df)
    sel_df = sel_df.join(combo_sel_df)
    # print(f"sel_df:\n{sel_df.head()}")

    # # just considering the first timestep a la bowers
    if just_1st_ts:
        print("\n\nOnly  considering 1st timestep (a la Bowers et al, 2014)\n")
        sel_df = sel_df.xs('ts0', level='Timestep')
    print(f"sel_df:\n{sel_df.head()}")

    sel_count_dict = dict()

    levels = ['word', 'letter']
    # # considering each timestep (ts)
    # # Count timesteps with sel > .0, .1, .2, .3, .4, .5 for words, letters, combo
    # # Count n_cats (ts_cats) in all ts with sel > .0, .1, .2, .3, .4, .5 for words, letters, combo
    for thr in thresholds:
        sel_count_dict[str(thr)] = dict()
        for level in levels:
            sel_count_dict[str(thr)][level] = dict()
            thr_df = sel_df[sel_df[f"{level}_sel"] >= thr]
            class_list = thr_df[f'{level}_c'].to_list()
            sel_count_dict[str(thr)][level]['timesteps'] = len(class_list)
            sel_count_dict[str(thr)][level]['ts_cats'] = len(set(class_list))

            print(f"\nthr: {thr}\nlevel: {level}\ntimesteps: {len(class_list)}\n"
                  f"ts_cats: {len(set(class_list))}\n"
                  f"{thr_df.loc[:, ['letter_sel', 'letter_c', 'letter_feat']]}")



    # # make dict just consisisting of units where word/sel values are the same at all timesteps
    # # - only do single layer output for now
    # # keys: unit
    # # values: level: [word, letter], feature, min_sel
    invar_dict = dict()

    unit_list = list(word_sel_dict['hid0'].keys())
    for unit in unit_list:
        unit_df = sel_df.xs(('hid0', unit))
        # print(f"unit_df: {unit}\n{unit_df}\n")

        # word invar
        print(f"\nUnit_df:\n{unit_df}")
        word_c_list = unit_df['word_c'].to_list()
        if len(set(word_c_list)) == 1:
            invar_dict[unit] = dict()
            # print(f"Invariant for words!\n")
            invar_dict[unit]['word_label'] = word_c_list[0]
            invar_dict[unit]['word_feat'] = unit_df['word_feat'].to_list()[0]
            invar_dict[unit]['word_sel'] = min(unit_df['word_sel'].to_list())

        # letter invar
        letter_c_list = unit_df['letter_c'].to_list()
        if len(set(letter_c_list)) == 1:
            if unit not in invar_dict:
                invar_dict[unit] = dict()
            # print(f"Invariant for letters!\n")
            invar_dict[unit]['letter_label'] = letter_c_list[0]
            invar_dict[unit]['letter_feat'] = unit_df['letter_feat'].to_list()[0]
            invar_dict[unit]['letter_sel'] = min(unit_df['letter_sel'].to_list())
    focussed_dict_print(invar_dict, 'invar_dict')



    invar_df = pd.DataFrame.from_dict(invar_dict, orient='index')
    print(f'invar_df:\n{invar_df}')
    # # considering units where class is the same at all timesteps, using min sel as value
    # # Count invar_units with sel > .0, .1, .2, .3, .4, .5 for words, letters, combo
    # # Count invar_cats in all ts with sel > .0, .1, .2, .3, .4, .5 for words, letters, combo

    # if invar_df.empty:
    #     print("\nthere are no timestep invariant units")

    if not invar_df.empty:
        for thr in thresholds:
            for level in levels:
                if f"{level}_sel" in list(invar_df):
                    thr_invar_df = invar_df[invar_df[f"{level}_sel"] >= thr]
                    if not thr_invar_df.empty:
                        # print(f"{thr} - {level}\n{thr_invar_df}")
                        class_list = thr_invar_df[f'{level}_label'].to_list()
                        class_list = [x for x in class_list if str(x) != 'NaN']
                        sel_count_dict[str(thr)][level]['invar_units'] = len(class_list)
                        sel_count_dict[str(thr)][level]['invar_cats'] = len(set(class_list))
                if 'invar_cats' not in sel_count_dict[str(thr)][level]:
                    sel_count_dict[str(thr)][level]['invar_units'] = 0
                    sel_count_dict[str(thr)][level]['invar_cats'] = 0


    # # option to create or append results to summary doc with scores for each model
    if save_csv:
        # # count csv
        # # flatten count dict into df
        flat_count_df = pd.io.json.json_normalize(sel_count_dict, sep='_')
        flat_count_df.rename(index={0: sel_dict['topic_info']['output_filename']}, inplace=True)
        flat_count_df.index.rename('cond_name', inplace=True)

        print(f"flat_count_df:\n{flat_count_df.head()}")

        # # add details for filtering
        serial_recall = sel_dict['model_info']['overview']['serial_recall']
        x_data_type = sel_dict['model_info']['overview']['x_data_type']
        timesteps = sel_dict['model_info']['overview']['timesteps']
        max_epochs = sel_dict['model_info']['overview']['max_epochs']
        n_cats = sel_dict['data_info']['n_cats']

        flat_count_df = pd.concat([flat_count_df,
                                   pd.DataFrame(
            [[x_data_type, timesteps, serial_recall, max_epochs, n_cats]],
            index=flat_count_df.index,
            columns=['x_data_type', 'timesteps', 'serial_recall', 'max_epochs',
                     'n_cats'])],
            axis=1)


        summary_dir = find_path_to_dir(word_sel_dict_path,
                                       target_dir=sel_dict['topic_info']['exp_name'])
        summary_path = os.path.join(summary_dir, f'{measure}_count_summary.csv')
        if just_1st_ts:
            summary_path = os.path.join(summary_dir, f'{measure}_1ts_count_summary.csv')

        if os.path.isfile(summary_path):
            summary_df = pd.read_csv(summary_path, index_col='cond_name')
            summary_df = summary_df.append(other=flat_count_df)
        else:
            summary_df = flat_count_df

        summary_df.to_csv(summary_path)
        # print(f"summary_df:\n{summary_df}")

    return sel_count_dict

# word_sel_dict_path = '/Users/nickmartin/Documents/PhD/python_v2/experiments/' \
#                 'train_rnn_script_check/test_25112019/correct_sel/' \
#                 'test_25112019_sel_dict.pickle'
#
# sel_count_dict = count_sel_units(word_sel_dict_path, save_csv=True)
#
# focussed_dict_print(sel_count_dict, 'sel_count_dict')



####################################################################

def rnn_sel(gha_dict_path, correct_items_only=True, all_classes=True,
            save_output_to='pickle',
            letter_sel=False,
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
    :param letter_sel: if False, test sel for words (class-labels).
            If True, test for letters (parts) using 'local_word_X' for each word when looping through classes
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


    if letter_sel:
        print("\n**** running rnn_sel() on letters ****")
    else:
        print("\n**** running rnn_sel() on words ****")

    if type(gha_dict_path) is str:
        if os.path.isfile(gha_dict_path):
            print(f"gha_dict_path: {gha_dict_path}")

            # # use gha-dict_path to get exp_cond_gha_path, gha_dict_name,
            exp_cond_gha_path, gha_dict_name = os.path.split(gha_dict_path)
            os.chdir(exp_cond_gha_path)
            current_wd = os.getcwd()

            # # part 1. load dict from study (should run with sim, GHA or sel dict)
            gha_dict = load_dict(gha_dict_path)
        else:
            raise FileNotFoundError(gha_dict_path)

    elif type(gha_dict_path) is dict:
        gha_dict = gha_dict_path
        exp_cond_gha_path = gha_dict['topic_info']['exp_cond_path']
        os.chdir(exp_cond_gha_path)
        current_wd = os.getcwd()
    else:
        raise FileNotFoundError(gha_dict_path)

    assert 'GHA_info' in gha_dict
    focussed_dict_print(gha_dict, 'gha_dict')

    # get topic_info from dict
    output_filename = gha_dict["topic_info"]["output_filename"]
    if letter_sel:
        output_filename = f"{output_filename}_lett"

    # # where to save files
    analyse_items = 'all'
    if correct_items_only:
        analyse_items = 'correct'
    sel_folder = f'{analyse_items}_sel'
    if test_run:
        sel_folder = f'{analyse_items}_sel/test'

    cond_name = gha_dict['topic_info']['output_filename']
    condition_path = find_path_to_dir(long_path=current_wd, target_dir=cond_name)
    sel_path = os.path.join(condition_path, sel_folder)

    if not os.path.exists(sel_path):
        os.makedirs(sel_path)

    if verbose:
        print(f"\ncurrent_wd: {current_wd}")
        print(f"output_filename: {output_filename}")
        print(f"sel_path (to save): {sel_path}")


    # # get data info from dict
    n_cats = gha_dict["data_info"]["n_cats"]
    if verbose:
        print(f"\nthere are {n_cats} word classes")

    if letter_sel:
        n_letters = gha_dict['data_info']["X_size"]
        n_cats = n_letters
        print(f"the are {n_letters} letters classes\nn_cats now set as n_letters")

        letter_id_dict = load_dict(os.path.join(gha_dict['data_info']['data_path'],
                                                'letter_id_dict.txt'))
        print(f"\nletter_id_dict:\n{letter_id_dict}")




    # # get model info from dict
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
        vocab_dict = load_dict(os.path.join(gha_dict['data_info']["data_path"],
                                            gha_dict['data_info']["vocab_dict"]))

    # # I can't do class correlations for letters, (as it is the equivillent of
    # having a dist output for letters
    if letter_sel:
        y_1hot = False

    # # get gha info from dict
    hid_acts_filename = gha_dict["GHA_info"]["hid_act_files"]['2d']


    '''Part 2 - load y, remove incorrect responses'''
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


            # print(f"\nthis_seq: {this_seq}\nget_letters: {get_letters}\nget_words: {get_words}\n")

        y_letters = np.array(y_letters)
        y_words = np.array(y_words)
        if verbose:
            print(f"\ny_letters: {type(y_letters)}  {np.shape(y_letters)}")
            print(f"y_words: {type(y_words)}  {np.shape(y_words)}")

        # print(f"y letters idio check"
        #       f"test_label_seqs: {test_label_seqs}"
        #       f"y_letters[:10]:\n{y_letters[:10]}"
        #       f"y_words[:10]:\n{y_words[:10]}"
        #
        #       f"")

        y_df_headers = [f"ts{i}" for i in range(timesteps)]
        y_scores_df = pd.DataFrame(data=test_label_seqs, columns=y_df_headers)
        y_scores_df['full_model'] = seqs_corr
        if verbose:
            print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")


    # # if not sequence data, load y_labels to go with hid_acts and item_correct for items
    # elif 'item_correct_name' in list(gha_dict['GHA_info']['scores_dict'].keys()):
    elif 'item_correct_name' in gha_dict['GHA_info']['scores_dict']:
        # # load item_correct (y_data)
        item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
        # y_df = pd.read_csv(item_correct_name)
        y_scores_df = nick_read_csv(item_correct_name)

    """# # get rid of incorrect items if required"""
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

    if verbose:
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

            if letter_sel:
                y_letters = y_letters[mask]

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
                raise TypeError("I can not complete this as desried"
                                "change correct_items_only to True"
                                "for analysis  - don't remove anything from hid_acts, output and "
                                "use y scores as y_df")

            # correct_items_only = True

    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")
        print(f"\ntest_label_seqs: {np.shape(test_label_seqs)}")  # \n{test_label_seqs}")
        if letter_sel:
            y_letters = np.asarray(y_letters)
            print(f"y_letters: {np.shape(y_letters)}")  # \n{test_label_seqs}")

    n_correct, timesteps = np.shape(test_label_seqs)
    corr_test_seq_name = f"{output_filename}_{n_correct}_corr_test_label_seqs.npy"
    np.save(corr_test_seq_name, test_label_seqs)
    corr_test_letters_name = 'not_processed_yet'
    if letter_sel:
        corr_test_letters_name = f"{output_filename}_{n_correct}_corr_test_letter_seqs.npy"
        np.save(corr_test_letters_name, y_letters)


    # # get items per class
    IPC_dict = seq_items_per_class(label_seqs=test_label_seqs, vocab_dict=vocab_dict)
    focussed_dict_print(IPC_dict, 'IPC_dict for correct responses')
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
            if 'hid_acts' in hid_acts_dict[last_layer_num]:
                output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts']
            elif 'hid_acts_2d' in hid_acts_dict[last_layer_num]:
                output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts_2d']

        elif '.h5' == hid_acts_filename[-3:]:
            acts_saved_as = 'h5'
            with h5py.File(hid_acts_filename, 'r') as hid_acts_dict:
                hid_acts_keys_list = list(hid_acts_dict.keys())
                last_layer_num = hid_acts_keys_list[-1]
                last_layer_name = hid_acts_dict[last_layer_num]['layer_name']
                # output_layer_acts = hid_acts_dict['hid_acts_2d'][last_layer_name]
                if 'hid_acts' in hid_acts_dict[last_layer_num]:
                    output_layer_acts = hid_acts_dict[last_layer_num]['hid_acts']
                elif 'hid_acts_2d' in hid_acts_dict[last_layer_num]:
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
                              letter_sel=letter_sel,
                              already_completed=already_completed,
                              verbose=verbose,
                              test_run=test_run
                              )

    for index, unit_gha in enumerate(loop_gha):

        if test_run:
            if index == 9:
                break

        unit_ts_dict = {'roc_auc': {}, 'ave_prec': {}, 'pr_auc': {},
                        'max_informed': {}, 'max_info_count': {},
                        'max_info_thr': {}, 'max_info_sens': {},
                        'max_info_spec': {}, 'max_info_prec': {},
                        'ccma': {},
                        'b_sel': {},
                        'b_sel_off': {},
                        'zhou_prec': {},
                        'zhou_selects': {}, 'zhou_thr': {},
                        'corr_coef': {}, 'corr_p': {},
                        }

        # print(f"\n\n{index}:\n{unit_gha}\n")
        sequence_data = unit_gha["sequence_data"]
        y_1hot = unit_gha["y_1hot"]
        act_func = unit_gha["act_func"]
        layer_name = unit_gha["layer_name"]
        unit_index = unit_gha["unit_index"]
        timestep = unit_gha["timestep"]
        ts_name = f"ts{timestep}"
        item_act_label_array = unit_gha["item_act_label_array"]
        IPC_words = IPC_dict['word_p_class_p_ts'][ts_name]
        IPC_letters = IPC_dict['letter_p_class_p_ts'][ts_name]

        # #  make df
        this_unit_acts = pd.DataFrame(data=item_act_label_array,
                                      columns=['item', 'activation', 'label'])
        this_unit_acts_df = this_unit_acts.astype(
            {'item': 'int32', 'activation': 'float', 'label': 'int32'})

        print(f"unit_index, ts: {unit_index}, {ts_name}")
        if verbose:
            print(f"sequence_data: {sequence_data}")
            print(f"y_1hot: {y_1hot}")

        y_letters_1ts = np.array(y_letters[:, timestep])

        if verbose:
            print(f"y_letters_1ts: {np.shape(y_letters_1ts)}")


        if test_run:
            # # get word ids to check results more easily.
            unit_ts_labels = this_unit_acts_df['label'].tolist()
            # print(f"unit_ts_labels:\n{unit_ts_labels}")

            seq_words_df = spell_label_seqs(test_label_seqs=np.asarray(unit_ts_labels),
                                            vocab_dict=vocab_dict, save_csv=False)
            seq_words_list = seq_words_df.iloc[:, 0].tolist()
            # print(f"seq_words_list:\n{seq_words_list}")
            this_unit_acts_df['words'] = seq_words_list
            # print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")


        # sort by descending hid act values (per item)
        this_unit_acts_df = this_unit_acts_df.sort_values(by='activation', ascending=False)
        if letter_sel:
            sorted_items = this_unit_acts_df['item'].to_list()
            sorted_idx = list(range(len(sorted_items)))

            # # unsort sorted_idx back into order of ascending items
            unsorted_order = [b for a, b in sorted(zip(sorted_items, sorted_idx))]

            # # put y_letters into same order as sorted_items
            y_letters_1ts = [b for a, b in sorted(zip(unsorted_order, y_letters_1ts))]
            y_letters_1ts = np.array(y_letters_1ts)
            # print(f"y_letters_1ts: {y_letters_1ts}")
            if verbose:
                print(f"np.shape(y_letters_1ts): {np.shape(y_letters_1ts)}")


        # # # always normalize to range [0, 1] activations if relu
        if act_func in ['relu', 'ReLu', 'Relu']:
            just_act_values = this_unit_acts_df['activation'].tolist()
            max_act = max(just_act_values)
            normed_acts = np.true_divide(just_act_values, max_act)
            this_unit_acts_df.insert(2, column='normed', value=normed_acts)

        # # if tanh, use activation except ccma which needs normed values in range [0, 1]
        if act_func == 'tanh':
            print("act func == tanh")
            just_act_values = this_unit_acts_df['activation'].tolist()
            act_plus_one = [i+1 for i in just_act_values]
            max_act = max(act_plus_one)
            normed_acts = np.true_divide(act_plus_one, max_act)
            this_unit_acts_df.insert(2, column='normed', value=normed_acts)


        if verbose is True:
            print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n")
            print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")

        # # run class_sel_basics here for words, further down for letters
        if not letter_sel:
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
                unit_ts_dict[csb_key] = csb_value

        classes_of_interest = list(range(n_cats))
        if all_classes is False:
            # # I don't want to use all classes, just ones that are worth testing
            classes_of_interest = coi_list(class_sel_basics_dict, verbose=verbose)


        if verbose:
            print('\n**** cycle through classes ****')

        cycle_this = range(len(classes_of_interest))
        if letter_sel:
            cycle_this = range(n_letters)

        for this_cat in cycle_this:

            if letter_sel:
                this_letter = letter_id_dict[this_cat]
                print(f"\nthis_cat: {this_cat} this_letter: {this_letter}")
                if this_letter in IPC_letters:
                    this_class_size = IPC_letters[this_letter]
                else:
                    this_class_size = 0
                not_a_size = n_correct - this_class_size

                # # make binary letter class list
                letter_class_list = y_letters_1ts[:, this_cat]

                # print(f"\n idiot check\n"
                #       f"y_letters: {y_letters.shape}\n{list(y_letters)}"
                #       f"\ny_letters_1ts: {y_letters_1ts.shape}\n{list(y_letters_1ts)}"
                #
                #       # f"y_letters_1ts\n{list(y_letters_1ts)}\n"
                #       f"original lettter class list\n{letter_class_list}\n"
                #       f"this_letter: {this_letter}, this_class_size: {this_class_size}, not_a_size: {not_a_size}\n"
                #       f"{this_unit_acts_df.sort_values(by='words')}\n"
                #       f"")

                # # changes '1's to this_cat
                # print(f"letter_class_list: {letter_class_list}")

                not_this_letter_symbol = 0

                if this_cat == 0:
                    not_this_letter_symbol = -4
                    letter_class_list = [this_cat if i == 1 else not_this_letter_symbol
                                         for i in np.array(letter_class_list)]
                else:
                    letter_class_list = [this_cat if i == 1 else not_this_letter_symbol
                                         for i in np.array(letter_class_list)]
                if verbose:
                    print(f"letter_class_list: {letter_class_list}")
                    # print(f"letter_class_list: {np.shape(letter_class_list)}")

                this_unit_acts_df['label'] = letter_class_list

            else:
                if this_cat in IPC_words:
                    this_class_size = IPC_words[this_cat]
                else:
                    this_class_size = 0
                not_a_size = n_correct - this_class_size

            if verbose:
                print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")


            if verbose is True:
                print(f"\nclass_{this_cat}: {this_class_size} items, "
                      f"not_{this_cat}: {not_a_size} items")


            # # running selectivity measures

            # # run class_sel_basics here for letters, further up page ^ for words
            if letter_sel:
                # # make new IPC_dict
                ts_IPC_letters = copy.copy(IPC_dict['letter_p_class_p_ts'][ts_name])
                # print(f"ts_IPC_letters:\n{ts_IPC_letters}")

                if this_letter in ts_IPC_letters.keys():
                    ts_IPC_this_letter = ts_IPC_letters[this_letter]
                    ts_IPC_this_letter = ts_IPC_letters.pop(this_letter)

                else:
                    ts_IPC_this_letter = 0

                ts_IPC_not_this_letter = sum(list(ts_IPC_letters.values()))

                IPC_binary_letters = {this_cat: ts_IPC_this_letter,
                                      not_this_letter_symbol: ts_IPC_not_this_letter}

                # print(f"ts_IPC_letters: {ts_IPC_letters}")
                # print(f"ts_IPC_this_letter: {ts_IPC_this_letter}")
                # print(f"ts_IPC_not_this_letter: {ts_IPC_not_this_letter}")
                if verbose:
                    print(f"IPC_binary_letters: {IPC_binary_letters}")

                # # get class_sel_basics (class_means, sd, prop > .5, prop @ 0)
                class_sel_basics_dict = class_sel_basics(this_unit_acts_df=this_unit_acts_df,
                                                         items_per_cat=IPC_binary_letters,
                                                         n_classes=[this_cat, not_this_letter_symbol],
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

                    if csb_key not in unit_ts_dict.keys():
                        unit_ts_dict[csb_key] = dict()

                    print(f"\nthis_cat: {this_cat}")
                    print(f"not_this_letter_symbol: {not_this_letter_symbol}")
                    print(f"csb_key: {csb_key}")
                    print(f"csb_value: {csb_value}")

                    if this_cat in csb_value.keys():
                        unit_ts_dict[csb_key][this_cat] = csb_value[this_cat]
                    else:
                        unit_ts_dict[csb_key][this_cat] = 0



                        # # ROC_stuff includes:
            # roc_auc, ave_prec, pr_auc, nz_ave_prec, nz_pr_auc, top_class_sel, informedness

            # # if relu, always use normed values. Otherwise use original values,
            # # except for tanh which must be normalised for ccma
            act_values = 'activation'
            if act_func == 'relu':
                act_values = 'normed'


            roc_stuff_dict = nick_roc_stuff(class_list=this_unit_acts_df['label'],
                                            hid_acts=this_unit_acts_df[act_values],
                                            this_class=this_cat,
                                            class_a_size=this_class_size,
                                            not_a_size=not_a_size,
                                            verbose=verbose)

            if verbose:
                print(f"roc_stuff_dict:\n{roc_stuff_dict}")

            # # add roc_stuff_dict to unit dict
            for roc_key, roc_value in roc_stuff_dict.items():
                unit_ts_dict[roc_key][this_cat] = roc_value

            # # CCMA
            class_a = this_unit_acts_df.loc[this_unit_acts_df['label'] == this_cat]
            not_class_a = this_unit_acts_df.loc[this_unit_acts_df['label'] != this_cat]

            empty_class_list = [class_a.empty, not_class_a.empty]
            an_empty_class = False
            if True in empty_class_list:
                an_empty_class = True

            if verbose:
                print(f"empty_class_list: {empty_class_list}\n"
                      f"an_empty_class: {an_empty_class}")

            if an_empty_class:
                if verbose:
                    print(f"\nCCMA\nno items in class {this_cat} or not this cat\nccma=0")
                ccma = 0
                b_sel = 0
            else:
                class_a_mean = class_a[act_values].mean()
                not_class_a_mean = not_class_a[act_values].mean()
                if act_func == 'tanh':
                    class_a_mean = class_a['normed'].mean()
                    not_class_a_mean = not_class_a['normed'].mean()
                ccma_numerator = class_a_mean - not_class_a_mean
                ccma_denominator = class_a_mean + not_class_a_mean
                ccma = ccma_numerator / ccma_denominator

                if verbose:
                    print(f"\nccma check\n"
                          f"this_class_size: {this_class_size}, not_a_size: {not_a_size}\n"
                          f"class_a:\n{class_a}\nclass_a_mean: {class_a_mean}\n"
                          f"not_class_a:\n{not_class_a}\nnot_class_a_mean: {not_class_a_mean}\n"
                          f"ccma_numerator: {ccma_numerator}\nccma_denominator: {ccma_denominator}\n"
                          f"ccma: {ccma}")

                unit_ts_dict["ccma"][this_cat] = ccma

                # # Bowers sel
                '''
                test for sel on and off units and give the max.  add variable for b_sel_off
                '''
                if verbose:
                    print("\nBowers Sel")

                # # first check for on units
                class_a_min = class_a[act_values].min()
                class_a_max = class_a[act_values].max()
                not_class_a_max = not_class_a[act_values].max()
                not_class_a_min = not_class_a[act_values].min()

                if act_func in ['tanh', 'relu', 'ReLu']:
                    class_a_min = class_a['normed'].min()
                    class_a_max = class_a['normed'].max()
                    not_class_a_max = not_class_a['normed'].max()
                    not_class_a_min = not_class_a['normed'].min()

                b_sel_on = class_a_min - not_class_a_max
                b_sel_off = not_class_a_min - class_a_max

                # print(f'\nb_sel_on = class_a_min: {class_a_min} - '
                #       f'not_class_a_max: {not_class_a_max} = {b_sel_on}\n'
                #       f'\nb_sel_off = not_class_a_min: {not_class_a_min} - '
                #       f'class_a_max: {not_class_a_max} = {class_a_max}\n')

                if b_sel_on >= b_sel_off:
                    b_sel = b_sel_on
                    off_unit = False
                    if verbose:
                        print(f"\nb_sel ON\n"
                              f"class_a_min: {class_a_min} - not_class_a_max: {not_class_a_max}\n"
                              f"b_sel: {b_sel}")
                else:
                    b_sel = b_sel_off
                    off_unit = True
                    if verbose:
                        print(f"\nb_sel OFF\n"
                              f"not_class_a_min: {not_class_a_min} - class_a_max: {class_a_max}\n"
                              f"b_sel: {b_sel}")

            unit_ts_dict["b_sel"][this_cat] = b_sel
            unit_ts_dict["b_sel_off"][this_cat] = off_unit


            # # zhou_prec
            if an_empty_class:
                zhou_prec = zhou_selects = zhou_thr = 0
            else:
                zhou_cut_off = .005
                if n_correct < 20000:
                    zhou_cut_off = 100 / n_correct
                if n_correct < 100:
                    zhou_cut_off = 1 / n_correct
                zhou_selects = int(n_correct * zhou_cut_off)

                if 9 < min(IPC_letters.values()) < 100:
                    zhou_selects = min(IPC_letters.values())

                if verbose:
                    print(f"\nZhou\n{IPC_letters}\n{min(IPC_letters.values())}")


                most_active = this_unit_acts_df.iloc[:zhou_selects]

                if act_func in ['relu', 'ReLu', 'Relu']:
                    zhou_thr = list(most_active["normed"])[-1]
                else:
                    zhou_thr = list(most_active["activation"])[-1]

                zhou_prec = sum([1 for i in most_active['label'] if i == this_cat]) / zhou_selects
            unit_ts_dict["zhou_prec"][this_cat] = zhou_prec
            unit_ts_dict["zhou_selects"][this_cat] = zhou_selects
            unit_ts_dict["zhou_thr"][this_cat] = zhou_thr

            # class correlation
            # get output activations for class correlation
            # # can only run this on y_1hot
            if y_1hot:
                if an_empty_class:
                    class_corr = {'coef': 0, 'p': 1}
                else:
                    output_layer_acts = np.load(output_acts_name)
                    # print(f"np.shape(output_layer_acts): {np.shape(output_layer_acts)}")
                    output_acts_ts = output_layer_acts[:, timestep, :]
                    # print(f"np.shape(output_acts_ts): {np.shape(output_acts_ts)}")

                    class_corr = class_correlation(this_unit_acts=this_unit_acts_df[act_values],
                                                   output_acts=output_acts_ts[:, this_cat],
                                                   verbose=verbose)
                unit_ts_dict["corr_coef"][this_cat] = class_corr['coef']
                unit_ts_dict["corr_p"][this_cat] = class_corr['p']
            else:
                if 'corr_coef' in unit_ts_dict:
                    del unit_ts_dict['corr_coef']
                    del unit_ts_dict['corr_p']

        focussed_dict_print(unit_ts_dict, f'unit_ts_dict: {unit_index} {ts_name}')

        # which class was the highest for each measure
        max_sel_p_unit_dict = sel_unit_max(unit_ts_dict, verbose=verbose)



        # # # # once sel analysis has been done for this hid_act array

        # # sort dicts to save
        # # add layer to all_sel_dict
        if layer_name not in all_sel_dict:
            all_sel_dict[layer_name] = dict()
            max_sel_dict[layer_name] = dict()

        # # add unit index to sel_p_unit dict
        if unit_index not in all_sel_dict[layer_name]:
            all_sel_dict[layer_name][unit_index] = dict()
            max_sel_dict[layer_name][unit_index] = dict()

        # # if not sequences data, add this unit to all_sel_dict
        if not sequence_data:
            all_sel_dict[layer_name][unit_index] = unit_ts_dict
            max_sel_dict[layer_name][unit_index] = max_sel_p_unit_dict

        else:  # # if sequence data
            # # add timestep to max sel_p_unit dict
            if timestep not in all_sel_dict[layer_name][unit_index]:
                all_sel_dict[layer_name][unit_index][ts_name] = dict()
                max_sel_dict[layer_name][unit_index][ts_name] = dict()

            # # add this timestep to all_sel_dict
            all_sel_dict[layer_name][unit_index][ts_name] = unit_ts_dict
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


    if verbose:
        focussed_dict_print(all_sel_dict, 'all_sel_dict')
        # print_nested_round_floats(all_sel_dict, 'all_sel_dict')

    # # save dict
    print("\n\n\n*****************\nanalysis complete\n*****************")

    max_sel_dict_path = os.path.join(sel_path, max_sel_dict_name)
    max_sel_summary = get_sel_summaries(max_sel_dict_path, verbose=verbose)

    # # save selectivity info
    sel_dict = gha_dict

    sel_dict_name = f"{sel_path}/{output_filename}_sel_dict.pickle"

    sel_dict["sel_info"] = {"sel_path": sel_path,
                            'sel_dict_name': sel_dict_name,
                            "all_sel_dict_name": all_sel_dict_name,
                            'max_sel_dict_name': max_sel_dict_name,
                            "correct_items_only": correct_items_only,
                            "all_classes": all_classes,
                            'corr_test_seq_name': corr_test_seq_name,
                            'corr_test_letters_name': corr_test_letters_name,
                            'corr_test_IPC_name': corr_test_IPC_name,
                            'max_sel_summary': max_sel_summary,
                            "sel_date": int(datetime.datetime.now().strftime("%y%m%d")),
                            "sel_time": int(datetime.datetime.now().strftime("%H%M")),
                            }

    print(f"\nSaving sel_dict to: {os.getcwd()}")
    pickle_out = open(sel_dict_name, "wb")
    pickle.dump(sel_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()

    focussed_dict_print(sel_dict, "sel_dict")


    # # making sel_summary_csv
    run = gha_dict['topic_info']['run']
    if test_run:
        run = 'test'

    sel_csv_info = [gha_dict['topic_info']['cond'], run, output_filename,
                    gha_dict['data_info']['dataset'], gha_dict['GHA_info']['use_dataset'],
                    gha_dict['model_info']['overview']['x_data_type'],
                    timesteps,
                    serial_recall,
                    gha_dict['model_info']['overview']['model_name'],
                    n_layers,
                    gha_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable'],
                    letter_sel,
                    gha_dict['GHA_info']['scores_dict']['prop_seq_corr'],
                    round(max_sel_summary['for_summ_csv_dict']['mi_mean'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['mi_max'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['ccma_mean'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['ccma_max'], 3),

                    round(max_sel_summary['for_summ_csv_dict']['b_sel_mean'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['b_sel_max'], 3),

                    round(max_sel_summary['for_summ_csv_dict']['prec_mean'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['prec_max'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['means_mean'], 3),
                    round(max_sel_summary['for_summ_csv_dict']['means_max'], 3),
                    int(datetime.datetime.now().strftime("%y%m%d")),
                    int(datetime.datetime.now().strftime("%H%M")),
                    ]

    summary_headers = ["cond", "run", "output_filename", "dataset", "use_dataset",
                       'X_type',
                       'timesteps',
                       'recall',
                       'model',
                       "n_layers", "hid_units",
                       'letter_sel',
                       "prop_seq_corr",
                       "mi_mean", "mi_max", "ccma_mean", "ccma_max",
                       "b_sel_mean", 'b_sel_max',
                       "prec_mean", "prec_max", "means_mean", "means_max",
                       'sel_date', 'sel_time']

    # # save sel summary in exp folder not condition folder
    exp_name = gha_dict['topic_info']['exp_name']
    exp_path = find_path_to_dir(long_path=sel_path, target_dir=exp_name)
    os.chdir(exp_path)

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

    return sel_dict  # , mean_sel_per_NN
