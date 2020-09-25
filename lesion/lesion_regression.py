import os
import operator
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score, precision_score, \
    recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

import statsmodels.api as sm

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats


# todo: Rather than correlating max_class_drop with selectivity,
#  I need a measure of how selectively impaired the unit was after lesioning.
#  One option is to use max_class_drop / sum(all_classes_that_dropped)

# todo:  I need something about how often the most selective class was the max_class_drop class

# todo: Zhou used spearman correlations rather than pearson

def lesion_sel_regression(lesion_dict_path, sel_dict_path,
                          lesion_meas='prop_change',
                          use_relu=False, test_run=False, verbose=False):
    """
    Script uses lesion dict and sel dict and has a simple logistic regression model to see how well various selectivity
    measures predict max class fail

    :param lesion_dict_path: path to dict from output of lesioning
    :param sel_dict_path: path to dict of output from selectivity
    :param lesion_meas: Which measure to use as to asses impact of lesioning.
                        'Prop_change' is the original 2019 measure: the change in accuracy per class
                                                                    ---------------------------------
                                                                    the class accuracy in unlesioned model

                        'chan_contri' is the new 2020 measure:  change in accuracy per class
                                                                ----------------------------
                                                                total change in acc         (zero if opposite sign).

                        'sign_contri' is the new 2020 measure v2!:
                                        change in accuracy per class
                        -----------------------------------------------------
                        sum of all classes with same sign as the total change  (zero if opposite sign).

                        'class_change': Change (items) per class

                        'just_drops': Drop (items) per class (or zero if class increases)

                        'drop_prop': Class drop as a proportion of sum of all classes that dropped.
                            Classes that increase score zero.  Even if unit Total improves, will still use sum of drops.

    :param use_relu: if False, only uses sel scores for lesioned layers (conv, dense), if true, only uses sel scores
        for ReLu layers, as found with link_layers_dict
    :param test_run: if True, just run two layers and two sel measures
    :param verbose: how much to print to screen

    :return: lesion_regression_dict
    """

    print("\n**** running lesion_sel_regression() ****")
    print(f"sel_dict_path: {sel_dict_path}\nlesion_meas: {lesion_meas}")

    lesion_measure_list = ['prop_change', 'class_change', 'chan_contri', 'sign_contri', 'drop_prop', 'just_drops']
    if lesion_meas not in lesion_measure_list:
        raise ValueError(f"lesion_meas ({lesion_meas}) not recognised.\n"
                         f"Lesion_meas should be one of: {lesion_measure_list}.")

    lesion_dict = load_dict(lesion_dict_path)
    focussed_dict_print(lesion_dict, "lesion dict")

    n_cats = lesion_dict['data_info']['n_cats']
    output_filename = lesion_dict['topic_info']['output_filename']

    lesion_info = lesion_dict['lesion_info']
    lesion_path = lesion_info['lesion_path']

    # # get key layers list
    key_lesion_layers_list = lesion_info['key_lesion_layers_list']

    # # remove output layers from key layers list
    if any("utput" in s for s in key_lesion_layers_list):
        output_layers = [s for s in key_lesion_layers_list if "utput" in s]
        output_idx = []
        for out_layer in output_layers:
            output_idx.append(key_lesion_layers_list.index(out_layer))
        min_out_idx = min(output_idx)
        key_lesion_layers_list = key_lesion_layers_list[:min_out_idx]

    print(f"\nkey_lesion_layers_list\n{key_lesion_layers_list}")

    # # sel_dict
    sel_dict = load_dict(sel_dict_path)
    focussed_dict_print(sel_dict, "lesion sel_dict")

    if key_lesion_layers_list[0] in sel_dict['sel_info']:
        print('\n found old sel dict layout')
        old_sel_dict = True
        sel_info = sel_dict['sel_info']
        short_sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0]['sel'].keys())
        csb_list = list(sel_info[key_lesion_layers_list[0]][0]['class_sel_basics'].keys())
        sel_measures_list = short_sel_measures_list + csb_list
    else:
        print('\n found NEW sel dict layout')
        old_sel_dict = False
        sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
        sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0].keys())

    # # remove measures that I don't want
    sel_measures_to_remove = ['nZ_ave_prec', 'nZ_pr_auc', 'tcs_items', 'tcs_recall',
                              'corr_coef', 'corr_p',
                              'nz_count', 'max_info_f1', 'max_info_count',
                              'max_info_thr', 'max_info_sens', 'max_info_spec',
                              'max_info_prec', 'tcs_thr', 'Zhou_selects', 'Zhou_thr', 'max', ]

    # sel_measures_list = ['roc_auc', 'ave_prec', 'pr_auc', 'max_informed', 'CCMAs', 'Zhou_prec',
    #                      'means', 'sd', 'nz_prop', 'nz_prec', 'hi_val_count', 'hi_val_prop', 'hi_val_prec']


    sel_measures_list = [x for x in sel_measures_list if x not in sel_measures_to_remove]

    if use_relu is True:
        # # get key_relu_layers_list
        key_relu_layers_list = list(sel_info.keys())

        # # # remove unnecessary items from key layers list
        if 'sel_analysis_info' in key_relu_layers_list:
            key_relu_layers_list.remove('sel_analysis_info')

        # # remove output layers from key layers list
        if any("utput" in s for s in key_relu_layers_list):
            output_layers = [s for s in key_relu_layers_list if "utput" in s]
            output_idx = []
            for out_layer in output_layers:
                output_idx.append(key_relu_layers_list.index(out_layer))
            min_out_idx = min(output_idx)
            key_relu_layers_list = key_relu_layers_list[:min_out_idx]
        print(f"\nkey_relu_layers_list\n{key_relu_layers_list}")

        # # put together lists of 1. sel_gha_layers, 2. key_lesion_layers_list.
        n_activation_layers = sum("activation" in layers for layers in key_relu_layers_list)
        n_lesion_layers = len(key_lesion_layers_list)

        if n_activation_layers == n_lesion_layers:
            activation_layers = [layers for layers in key_relu_layers_list if "activation" in layers]
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(activation_layers)))
        elif n_activation_layers == 0:
            print("\nno separate activation layers found - use key_lesion_layers_list")
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(key_lesion_layers_list)))
        else:
            raise TypeError('should be same number of activation layers and lesioned layers')
        if verbose is True:
            focussed_dict_print(link_layers_dict, 'link_layers_dict')

    print(f"\nsel_measures_list\n{sel_measures_list}")

    regression_measures_dict = dict()  # key=measure, val=score

    print("\n\nSaving lesion_regression_dict")
    # # # set dir to save les_sel_rel stuff stuff # # #
    les_sel_reg_path = os.path.join(lesion_dict['GHA_info']['gha_path'], 'les_sel_rel')
    if test_run is True:
        les_sel_reg_path = os.path.join(les_sel_reg_path, 'test')
    if not os.path.exists(les_sel_reg_path):
        os.makedirs(les_sel_reg_path)
    os.chdir(les_sel_reg_path)
    print(f"saving les_sel_rel data to: {les_sel_reg_path}")

    """loop through sel_measures_list
    get long array of all layers concatenated and list all max lesion drops.
    do regression model on this whole thing at once,
    save then repeat for next sel measure"""

    print("\nlooping through network to get selectivity scores as big array")
    sel_measure_counter = 0
    for sel_measure in sel_measures_list:
        print(f"\n{sel_measure}")

        if test_run:
            sel_measure_counter += 1
            if sel_measure_counter > 2:
                continue

        """loop through all layers"""
        all_layer_sel_array = []  # append each sel_layer to this to give array (each unit, classes)

        all_layer_class_drops = []  # # extend each lesion_layer to this to give list of all units in all layers

        layer_counter = 0
        for lesion_layer in key_lesion_layers_list:

            if test_run:
                layer_counter += 1
                if layer_counter > 2:
                    continue

            sel_layer = lesion_layer
            if use_relu:
                sel_layer = link_layers_dict[lesion_layer]

            if verbose:
                print(f"\n\tsel_layer: {sel_layer}\tlesion_layer: {lesion_layer}")

            sel_layer_info = sel_info[sel_layer]

            if 'means' in sel_layer_info.keys():
                del sel_layer_info['means']

            '''get sel_layer unit sel values'''
            # get array of sel values for regression model
            layer_sel_array = []
            for unit, unit_sel in sel_layer_info.items():

                if unit == 'means':
                    continue

                if old_sel_dict:
                    if sel_measure in unit_sel['sel']:
                        sel_items = unit_sel['sel'][sel_measure]
                    elif sel_measure in unit_sel['class_sel_basics']:
                        sel_items = unit_sel['class_sel_basics'][sel_measure]
                else:
                    sel_items = unit_sel[sel_measure]

                # # just check it is just classes in there

                if 'total' in sel_items.keys():
                    del sel_items['total']
                if 'perplexity' in sel_items.keys():
                    del sel_items['perplexity']

                if len(list(sel_items.keys())) != n_cats:
                    # print("\nERROR, {} hasn't got enough classes".format(sel_measure))
                    # print("error found", sel_items)
                    for i in range(n_cats):
                        if i not in sel_items:
                            sel_items[i] = 0.0
                    ordered_dict = dict()
                    for j in range(n_cats):
                        ordered_dict[j] = sel_items[j]

                    sel_items = dict()
                    sel_items = ordered_dict
                    # print('sel_items should be sorted now', sel_items)

                sel_values = list(sel_items.values())

                layer_sel_array.append(sel_values)
                # print(unit, sel_values)

            all_layer_sel_array = all_layer_sel_array + layer_sel_array

            # # lesion stuff
            lesion_per_unit_path = f'{lesion_path}/{output_filename}_{lesion_layer}_{lesion_meas}.csv'

            lesion_per_unit = pd.read_csv(lesion_per_unit_path, index_col=0)
            print(f"lesion_per_unit:\n{lesion_per_unit}")
            lesion_per_unit.drop('total', inplace=True)
            # lesion_per_unit = nick_read_csv(lesion_per_unit_path)
            # lesion_per_unit.set_index(0)
            print("\nlesion_per_unit")
            print(lesion_per_unit)

            lesion_cols = list(lesion_per_unit)

            '''get max class drop per lesion_layer'''
            if lesion_meas in ['prop_change', 'class_change', 'just_drops']:
                # # loop through lesion units (df columns) to find min class drop
                lesion_cat_p_u_dict = dict()
                lesion_unit_cat_list = []
                for index, l_unit in enumerate(lesion_cols):
                    unit_les_val = lesion_per_unit[l_unit].min()
                    unit_les_cat = lesion_per_unit[l_unit].idxmin()
                    lesion_unit_cat_list.append(int(unit_les_cat))
                    lesion_cat_p_u_dict[index] = {'unit': index, "l_min_class": unit_les_cat, 'l_min_drop': unit_les_val}
                    if verbose:
                        print(f"{index}: class: {unit_les_cat}  {unit_les_val}")

            elif lesion_meas in ['chan_contri', 'sign_contri', 'drop_prop']:
                # # loop through lesioned units (df columns) to find max class contri
                lesion_cat_p_u_dict = dict()
                lesion_unit_cat_list = []
                for index, l_unit in enumerate(lesion_cols):
                    unit_les_val = lesion_per_unit[l_unit].max()
                    unit_les_cat = lesion_per_unit[l_unit].idxmax()
                    lesion_unit_cat_list.append(int(unit_les_cat))
                    lesion_cat_p_u_dict[index] = {'unit': index, "l_min_class": unit_les_cat, 'l_min_drop': unit_les_val}
                    if verbose:
                        print(f"{index}: class: {unit_les_cat}  {unit_les_val}")


            # # check for missing values (prob dead relus) (originally just for when using different layers)
            sel_units, classes = np.shape(layer_sel_array)
            les_units = len(lesion_unit_cat_list)

            # if use_relu:  # now trying if for any runs, not just if using different layers
            if sel_units != les_units:
                if len(lesion_unit_cat_list) > sel_units:
                    available_sel_units = list(sel_layer_info.keys())
                    masked_class_drops = [lesion_unit_cat_list[i] for i in available_sel_units]
                    lesion_unit_cat_list = masked_class_drops

                if sel_units != len(lesion_unit_cat_list):
                    raise ValueError(f"unequal number of "
                                     f"sel units {sel_units} and class drop values {len(lesion_unit_cat_list)}")

            # # if there are any NaNs:
            if np.any(np.isnan(layer_sel_array)):
                layer_sel_array = np.nan_to_num(layer_sel_array, copy=False)
            #     if np.any(np.isnan(layer_sel_array)):
            #         print("TRUE still nan")
            # if np.all(np.isfinite(layer_sel_array)):
            #     print("TRUE inf")
            # layer_sel_array = np.array(layer_sel_array)  # [np.isnan(layer_sel_array)] = 0.0
            # layer_sel_array = np.nan_to_num(layer_sel_array, copy=False)
            # layer_sel_array[np.isneginf(layer_sel_array)] = 0

            all_layer_class_drops.extend(lesion_unit_cat_list)

            if verbose:
                print(f"\n\t\tlayer_sel: {np.shape(layer_sel_array)}, "
                      f"all_layers_sel: {np.shape(all_layer_sel_array)}; "
                      f"all_layers_class_drops: {np.shape(all_layer_class_drops)}")

        # # sel lesion model
        '''#  for logistic regression report df in parenthesis (and df error), 
                http://bit.csc.lsu.edu/~jianhua/emrah.pdf
                #  to answer the question of how good a model is and how sound it is, one must  attend  to:  
                #  (a)  overall  model  evaluation (must outperform an intercept only (null) model).
                     Consequently, according to this model, all observations would be predicted to belong in the largest 
                     outcome category.  An improvement over this baseline is examined by using three inferential 
                     statistical tests: the likelihood ratio, score, and Wald tests.

                #  (b)  The  statistical significance of individual regression coefficients (i.e.,βs) is tested using the 
                    Wald chi-square statistic
                #  (c)  goodness-of-fit  statistics (accuracy or roc_auc),and 
                #  (d) validations of predicted probabilities.
                    The four measures of association are Kendall’s Tau-a, Goodman-Kruskal’s Gamma, Somers’s D statistic, and 
                    the c statistic.
                '''

        print("\n*********************************"
              "\nRunning logistic regression model"
              "\n*********************************")

        # # part 2 , use master to do stats
        txt = open(f'{output_filename}_{sel_measure}_{lesion_meas}_les_sel_reg.txt', 'w')

        # x_data = sel measure values
        x_data = all_layer_sel_array
        print(f"x_data: {np.shape(x_data)}")
        print(f"x_data: {type(x_data)}")

        print(f"{len(sel_measures_list)} sel_measures_list: {sel_measures_list}")

        # y = max class drop list
        y = all_layer_class_drops
        print(f"y: {np.shape(y)}")
        # y_data = to_categorical(y, num_classes=n_cats)
        # print(f"y_data: {np.shape(y_data)}")

        n_x_items = np.shape(x_data)[0]
        n_y_items = np.shape(y)[0]

        if n_x_items != n_y_items:
            print(f"\nERROR! number of x's ({n_x_items}) and y's {n_y_items} do not match!")

        txt.write(f"\nsel_measure: {sel_measure}\nx_data: {np.shape(x_data)}")

        # # plot distribution of selectivity score - scatter/violin

        # # get descriptives - mean sel per class

        # # compare means of class sel scores - ANOVA

        # # plot correlations between features
        class_labels = lesion_dict['data_info']['cat_names'].values()
        x_data_df = pd.DataFrame(data=x_data, columns=class_labels)
        # plt.subplots()
        plt.figure()
        ax = sns.heatmap(x_data_df.corr(), annot=True, cmap="RdYlGn", fmt='.1g')
        plt.title("Correlations of selected features")
        ax.set(xlim=(0, n_cats), ylim=(0, n_cats))
        plt.tight_layout()
        # plt.tight_layout(pad=1, h_pad=5.0, )
        plt.savefig(f"{output_filename}_{sel_measure}_{lesion_meas}_feat_corr.png")
        if test_run:
            plt.show()
        plt.close()

        # # plot distribution of max_class_drops - bar plots

        # # split data train/test
        x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(x_data),
                                                            y, test_size=0.2, random_state=2)

        """
        I acn't get the statsmodel version to converge :(

        # # simple regression model
        # sm_model = sm.MNLogit(y_train, x_train)
        sm_model = sm.MNLogit(y_train, sm.add_constant(x_train), missing='raise')
        sm_results = sm_model.fit(method='powell', full_output=True)

        # get model predictions and score AUC
        sm_predict = sm_results.predict(sm.add_constant(x_test)).tolist()
        print(f"sm_predict: {sm_predict}")

        sm_score = sm_model.score(sm_results.params)
        print(f"sm_score: {sm_score}")

        sm_score_obs = sm_model.score_obs(sm_results.params)
        print(f"sm_score_obs: {sm_score_obs}")

        # print("\n sm_model attributes")
        # for attr in dir(sm_model):
        #     if not attr.startswith('_'):
        #         print(attr)
        #
        # sm_score = sm_results
        # print("\n sm_results attributes")
        # for attr in dir(sm_results):
        #     if not attr.startswith('_'):
        #         print(attr)

        # # accuracy and conf matrix
        # sm_bin_pred = [1 if x >= .5 else 0 for x in sm_predict]
        # sm_accuracy = accuracy_score(y_test, sm_bin_pred)
        # print(f"\nsm_accuracy: {sm_accuracy:.3f}")
        # txt.write(f"\n\nsm_accuracy: {sm_accuracy:.3f}")
        #
        # sm_conf_matrix = confusion_matrix(y_test, sm_bin_pred)
        # print(sm_conf_matrix)
        # txt.write(f"\n\nsm_conf_matrix (diagonals are correct)")
        # txt.write(f"\n\tpred_0\tpred_1")
        # txt.write(f"\ntrue_0\t{sm_conf_matrix[0][0]}\t{sm_conf_matrix[0][1]}")
        # txt.write(f"\ntrue_1\t{sm_conf_matrix[1][0]}\t{sm_conf_matrix[1][1]}")

        ## # finisdh off plotting cof matrix

        # print(f"LogisticRegression score: {sm_accuracy:.3f}")
        # print(f"just guessing 'No' everytime would get you {1 - (np.mean(y)):.3f}")
        # print(f'\nThis model does {sm_accuracy - (1 - (np.mean(y))):.3f} better than quessing chance')
        # txt.write(f"\nLogisticRegression score: {sm_accuracy:.3f}")
        # txt.write(f"\njust guessing no everytime would get you {1 - (np.mean(y)):.3f}")
        # txt.write(f'\nThis model does {sm_accuracy - (1 - (np.mean(y))):.3f} better than quessing chance')

        # # model results summary
        print(f"\nsm_results.summary():\n{sm_results.summary()}")
        txt.write(f"\n\nsm_results.summary():\n{sm_results.summary()}")

        '''
        notes about the summary.
        z values are the parameter estimates (coef) divided by their std err.
        p values are calculated with respect to a standard normal distribution (not multiple comparrison corrected)
        '''
        # print("\n sm_results attributes")
        # for attr in dir(sm_results):
        #     if not attr.startswith('_'):
        #         print(attr)

        sm_model_df = int(sm_results.df_model)
        print(f"\nsm_model_df: {sm_model_df:.3f}")
        txt.write(f"\n\nsm_model_df: {sm_model_df:.3f}")
        sm_resid_df = int(sm_results.df_resid)
        print(f"sm_resid_df: {sm_resid_df:.3f}")
        txt.write(f"\nsm_resid_df: {sm_resid_df:.3f}")

        sm_null_ll = sm_results.llnull  # Value of the constant-only loglikelihood
        sm_loglikelihood = sm_results.llf  # Log-likelihood of model
        sm_ll_chi2 = sm_results.llr  # improvement from null, Ll ratio chi-squared statistic; -2*(llnull - llf)
        sm_llr_p_val = sm_results.llr_pvalue  # p val for .llr sm_ll_chi2
        print(f"\nsm_null_ll: {sm_null_ll:.3f}")
        print(f"sm_loglikelihood: {sm_loglikelihood:.3f}")
        print(f"sm_ll_chi2: {sm_ll_chi2:.3f}")
        print(f"sm_llr_p_val: {sm_llr_p_val:.3f}")
        txt.write(f"\n\nsm_null_ll: {sm_null_ll:.3f}")
        txt.write(f"\nsm_loglikelihood: {sm_loglikelihood:.3f}")
        txt.write(f"\nsm_ll_chi2: {sm_ll_chi2:.3f}")
        txt.write(f"\nsm_llr_p_val: {sm_llr_p_val:.3f}")

        print(f"\nX2({sm_model_df} = {sm_ll_chi2:.3f}, p = {sm_llr_p_val:.3f}")
        txt.write(f"\nX2({sm_model_df} = {sm_ll_chi2:.3f}, p = {sm_llr_p_val:.3f}")

        sm_pseudo_r2 = sm_results.prsquared  # McFadden’s pseudo-R-squared (variance explained)
        print(f"\nsm_pseudo_r2: {sm_pseudo_r2:.3f}")
        txt.write(f"\n\nsm_pseudo_r2:{sm_pseudo_r2:.3f}")

        sm_tvalues = sm_results.tvalues  # Return the t-statistic for a given parameter estimate
        sm_p_vals = sm_results.pvalues  # The two-tailed p values for the t-stats of the params
        print(f"\nsm_tvalues:\n{sm_tvalues}")
        print(f"\nsm_p_vals:\n{sm_p_vals}")
        txt.write(f"\n\nsm_tvalues:\n{sm_tvalues}")
        txt.write(f"\n\nsm_p_vals:\n{sm_p_vals}")

        sm_bse = sm_results.bse  # The standard errors of the parameter estimates.
        print(f"\nsm_bse:\n{sm_bse}")
        txt.write(f"\n\nsm_bse:\n{sm_bse}")

        sm_wald_test_terms = sm_results.wald_test_terms()  # Wald tests for terms over multiple predictors

        wald_test_df = pd.DataFrame(data=sm_wald_test_terms.summary_frame())
        wald_sig = []
        for i in sm_wald_test_terms.pvalues:
            if i < .001:
                wald_sig.append('***')
            elif i < .01:
                wald_sig.append('**')
            elif i < .05:
                wald_sig.append('*')
            else:
                wald_sig.append("")
        wald_test_df['sig'] = wald_sig
        print(f"\nwald_test_df:\n{wald_test_df}")
        txt.write(f"\nwald_test_df:\n{wald_test_df}")

        sm_coefficients = sm_results.params
        print(f"\nsm_coefficients:\n{sm_coefficients}")
        txt.write(f"\n\nsm_coefficients:\n{sm_coefficients}")

        sm_conf_int = sm_results.conf_int()
        print(f"\nsm_conf_int:\n{sm_conf_int}")
        txt.write(f"\n\nsm_conf_int:\n{sm_conf_int}")

        # # odds ratio
        sm_odds_ratio = np.exp(sm_coefficients)
        print(f"\nsm_odds_ratio:\n{sm_odds_ratio}")
        txt.write(f"\n\nsm_odds_ratio:\n{sm_odds_ratio}")

        # odds ratios and 95% CI
        sm_params = sm_results.params
        print(f"\nsm_params:\n{sm_params}")
        txt.write(f"\n\nsm_params:\n{sm_params}")

        sm_output_table = sm_conf_int
        sm_output_table['OR'] = sm_odds_ratio
        # sm_output_table = np.exp(sm_output_table)

        sm_output_table['B'] = sm_coefficients
        sm_output_table['SE'] = sm_bse
        sm_output_table.columns = ['B', 'SE', '2.5%', 'OR', '97.5%']
        print(f"\nsm_output_table:\n{sm_output_table}")
        txt.write(f"\n\nsm_output_table:\n{sm_output_table}")

        txt.close()

        # regression_score = sm_accuracy
        # regression_measures_dict[sel_measure] = regression_score
        """

        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial', max_iter=1000, class_weight='balanced').fit(x_train,
                                                                                                        y_train)

        # todo: report more statistics here (get params? predict proba, )

        regression_score = clf.score(x_test, y_test)

        print(f"{sel_measure} regression_score: {regression_score:.2f}")

        # parameters = clf.get_params()
        # print(f"regression parameters:\n{parameters}")

        regression_measures_dict[sel_measure] = regression_score

    # # comparrison with null model (dummy classifier)
    dummy = DummyClassifier(strategy='most_frequent').fit(x_test, y_test)
    dummy_score = dummy.score(x_data, y)
    print(f"{sel_measure} dummy_score: {dummy_score:.2f}")

    regression_measures_dict['dummy_mode'] = dummy_score

    # print("\n\nSaving lesion_regression_dict")
    # # # # set dir to save les_sel_rel stuff stuff # # #
    # les_sel_reg_path = os.path.join(lesion_dict['GHA_info']['gha_path'], 'les_sel_rel')
    # if test_run is True:
    #     les_sel_reg_path = os.path.join(les_sel_reg_path, 'test')
    # if not os.path.exists(les_sel_reg_path):
    #     os.makedirs(les_sel_reg_path)
    # os.chdir(les_sel_reg_path)
    # print(f"saving les_sel_rel data to: {les_sel_reg_path}")

    lesion_regression_dict = lesion_dict
    lesion_regression_dict['regression_info'] = regression_measures_dict
    focussed_dict_print(lesion_regression_dict, 'lesion_regression_dict', focus_list=['regression_info'])

    if use_relu:
        output_filename = output_filename + '_onlyReLu'
    print("output_filename: ", output_filename)

    les_sel_rel_dict_name = f"{output_filename}_{lesion_meas}_les_sel_rel_dict.pickle"
    pickle_out = open(les_sel_rel_dict_name, "wb")
    pickle.dump(lesion_regression_dict, pickle_out)
    pickle_out.close()

    # les_sel_rel_df = pd.DataFrame(data=regression_measures_dict, index=[0], )
    les_sel_rel_df = pd.DataFrame.from_dict(regression_measures_dict, orient='index')
    # nick_to_csv(les_sel_rel_df, f"{output_filename}_les_sel_rel_dict_nm.csv")
    les_sel_rel_df.to_csv(f"{output_filename}_{lesion_meas}_les_sel_rel_dict_pd.csv")

    return lesion_regression_dict


def item_act_fail_regression(sel_dict_path, lesion_dict_path, plot_type='classes',
                             sel_measures=['Zhou_prec', 'CCMAs', 'max_informed'],
                             use_normed_acts=True,
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
            if sel measures = 'best', it will take the top 5 or those performing abobve some threshold (e.g., > .3)
    :param use_normed_acts: if True, use normalised (0-1) activations in analysis, if False use original values.
    :param top_layers: if int, it will just do the top n layers (excluding output).  If not int, will do all layers.
    :param use_relu: if false - use layers from lesion (e.g., conv), if True, use activation following lesioned layer.
    :param verbose: how much to print to screen
    :param test_run: if True just run subset (e.g., two units from two layers)

    :return: print and save plots
    """

    print("\n**** running item_act_fail_regression()****")

    # # lesion dict
    lesion_dict = load_dict(lesion_dict_path)
    focussed_dict_print(lesion_dict, 'lesion_dict')

    # # get key_lesion_layers_list
    lesion_info = lesion_dict['lesion_info']
    lesion_path = lesion_info['lesion_path']
    key_lesion_layers_list = lesion_info['key_lesion_layers_list']

    # # remove unnecessary items from key layers list
    if 'highlights' in key_lesion_layers_list:
        key_lesion_layers_list.remove('highlights')
    if any("utput" in s for s in key_lesion_layers_list):
        output_layers = [s for s in key_lesion_layers_list if "utput" in s]
        output_idx = []
        for out_layer in output_layers:
            output_idx.append(key_lesion_layers_list.index(out_layer))
        min_out_idx = min(output_idx)
        key_lesion_layers_list = key_lesion_layers_list[:min_out_idx]
    print(f"\nkey_lesion_layers_list\n{key_lesion_layers_list}")

    # # sel_dict
    sel_dict = load_dict(sel_dict_path)
    if key_lesion_layers_list[0] in sel_dict['sel_info']:
        print('\nfound OLD sel dict layout')
        key_relu_layers_list = list(sel_dict['sel_info'].keys())
        old_sel_dict = True
        sel_info = sel_dict['sel_info']
        short_sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0]['sel'].keys())
        csb_list = list(sel_info[key_lesion_layers_list[0]][0]['class_sel_basics'].keys())
        sel_measures_list = short_sel_measures_list + csb_list
    else:
        print('\n found NEW sel dict layout')
        old_sel_dict = False
        sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
        sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0].keys())
        key_relu_layers_list = list(sel_info.keys())
        # print(sel_info.keys())

    n_conv_layers = len(key_lesion_layers_list)
    n_layers = n_conv_layers

    # # get key_relu_layers_list
    if use_relu:
        # # if model has separate activation layers (e.g., conv, activation, dense, activation)
        #        get scores from activation layers
        # # remove unnecessary items from key layers list
        if 'sel_analysis_info' in key_relu_layers_list:
            key_relu_layers_list.remove('sel_analysis_info')
        if any("utput" in s for s in key_relu_layers_list):
            output_layers = [s for s in key_relu_layers_list if "utput" in s]
            output_idx = []
            for out_layer in output_layers:
                output_idx.append(key_relu_layers_list.index(out_layer))
            min_out_idx = min(output_idx)
            key_relu_layers_list = key_relu_layers_list[:min_out_idx]
        print(f"\nkey_relu_layers_list\n{key_relu_layers_list}")

        # # put together lists of 1. sel_relu_layers, 2. key_lesion_layers_list.
        n_activation_layers = sum("activation" in layers for layers in key_relu_layers_list)

        if n_activation_layers == n_conv_layers:
            n_layers = n_activation_layers
            activation_layers = [layers for layers in key_relu_layers_list if "activation" in layers]
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(activation_layers)))
        elif n_activation_layers == 0:
            print("\nno separate activation layers found - use key_lesion_layers_list")
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(key_lesion_layers_list)))
        else:
            print(f"n_activation_layers: {n_activation_layers}\n{key_relu_layers_list}"
                  f"\nn_conv_layers: {n_conv_layers}\n{key_lesion_layers_list}")
            raise TypeError('should be same number of activation layers and lesioned layers')

        if verbose is True:
            focussed_dict_print(link_layers_dict, 'link_layers_dict')

    # # # get info
    output_filename = sel_dict['topic_info']['output_filename']

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
    # else:
    #     hid_act_items = 'all'
    #     if not sel_dict['GHA_info']['gha_incorrect']:
    #         hid_act_items = 'correct'
    #
    #     use_dataset = sel_dict['GHA_info']['use_dataset']
    #     gha_folder = '{}_{}_gha'.format(hid_act_items, use_dataset)
    #     hid_acts_path = os.path.join(exp_cond_path, gha_folder, hid_acts_pickle_name)

    with open(hid_acts_path, 'rb') as pkl:
        hid_acts_dict = pickle.load(pkl)
    print("\nopened hid_acts.pickle")
    # print(hid_acts_dict.keys())

    # # dict to get the hid_acts_dict key for each layer based on its name
    get_hid_acts_number_dict = dict()
    for key, value in hid_acts_dict.items():
        hid_acts_dict_layer = value['layer_name']
        hid_acts_layer_number = key
        get_hid_acts_number_dict[hid_acts_dict_layer] = hid_acts_layer_number

    # # get sel measures to use
    if sel_measures == 'all':
        print("Using all sel measures")
        sel_measures = sel_measures_list
    elif sel_measures == 'best':
        print("Using best sel measures")
        # sort sel measures and take top 5
        les_sel_reg_path = os.path.join(lesion_dict['GHA_info']['gha_path'], 'les_sel_rel',
                                        f"{output_filename}_les_sel_rel_dict.pickle")
        if os.path.isfile(les_sel_reg_path):
            les_sel_reg_dict = load_dict(les_sel_reg_path)
            reg_info = les_sel_reg_dict['regression_info']
            top_n = 5
            sorted_sel_measures = sorted(reg_info.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
            sel_measures = [measure[0] for measure in sorted_sel_measures]
        else:
            raise ValueError("Could not find les sel reg dict to get best measures")

    print(f"\nsel_measures\n{sel_measures}")

    # # new master page [layer, unit, item, class, hid_act, normed_act, item_change, prec, CCMA, info)
    master_headers = ['layer', 'unit', 'item', 'class', 'hid_acts', 'normed_acts', 'l_fail'] + sel_measures

    master_df = pd.DataFrame(data=None, columns=master_headers)
    print(master_df.head())

    # # where to save files
    save_plots_name = 'item_act_fail_reg'
    if plot_type is "OneVsAll":
        save_plots_name = f'item_act_fail_reg/{sel_measures}'
    if use_relu:
        save_plots_name = 'item_act_fail_reg_ReLu'
    item_fail_reg_path = os.path.join(lesion_dict['GHA_info']['gha_path'], save_plots_name)
    if test_run is True:
        item_fail_reg_path = os.path.join(item_fail_reg_path, 'test')
    if not os.path.exists(item_fail_reg_path):
        os.makedirs(item_fail_reg_path)
    os.chdir(item_fail_reg_path)
    print(f"\nitem_fail_reg_path: {os.getcwd()}")

    # # check for master list
    already_got_master = False
    if use_relu:
        master_filename = f"{output_filename}_item_act_fail_ReLu_MASTER.pickle"
    else:
        master_filename = f"{output_filename}_item_act_fail_MASTER.pickle"

    if os.path.isfile(master_filename):
        master_df = pickle.load(open(master_filename, "rb"))
        print("\nAlready have a master_df")
        master_cols = list(master_df)
        if set(sel_measures).issubset(master_cols):
            print("Master df contains all the right sel measures")
            already_got_master = True
        else:
            print(f"but it does not have all these sel measures: {sel_measures}")

    if not already_got_master:
        print("\n\n**********************"
              "\nlooping through layers"
              "\n**********************\n")

        # for layer_index, (relu_layer_name, conv_layer_name) in enumerate(link_layers_dict.items()):
        for layer_index, conv_layer_name in enumerate(reversed(key_lesion_layers_list)):

            if test_run:
                if layer_index > 2:
                    continue

            if type(top_layers) is int:
                if top_layers < n_activation_layers:
                    if layer_index > top_layers:
                        continue

            use_layer_name = conv_layer_name
            if use_relu:
                use_layer_name = link_layers_dict[conv_layer_name]

            # todo: adapt this for hdf5 files

            use_layer_number = get_hid_acts_number_dict[use_layer_name]
            hid_acts_dict_layer = hid_acts_dict[use_layer_number]

            if use_layer_name != hid_acts_dict_layer['layer_name']:
                print(f"conv_layer_name: {conv_layer_name}\nuse_layer_name: {use_layer_name}\n"
                      f"use_layer_number: {use_layer_number}\n"
                      f"hid_acts_dict_layer['layer_name']: {hid_acts_dict_layer['layer_name']}")
                focussed_dict_print(get_hid_acts_number_dict, 'get_hid_acts_number_dict')
                raise TypeError("use_layer_number and hid_acts_dict_layer['layer_name'] should match!")

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
            item_change_name = f"{lesion_path}/{output_filename}_{conv_layer_name}_item_change.csv"
            item_change_df = pd.read_csv(item_change_name, header=0, dtype=int, index_col=0)

            if verbose:
                print("\n*******************************************"
                      f"\n{layer_index}. use layer {use_layer_number}: "
                      f"{use_layer_name} \tlesion layer: {conv_layer_name}"
                      "\n*******************************************")

                print(f"\n\thid_acts {use_layer_name} shape: {hid_acts_df.shape}\n"
                      f"\tloaded: {output_filename}_{conv_layer_name}_item_change.csv: {item_change_df.shape}")

            units_per_layer = len(hid_acts_df.columns)

            print("\n\n\t**** loop through units ****")
            for unit_index, unit in enumerate(hid_acts_df.columns):

                if test_run:
                    if unit_index > 2:
                        continue

                conv_layer_and_unit = f"{conv_layer_name}.{unit}"

                print(f"\n\n*************\nrunning layer {layer_index} of {n_layers}: "
                      f"unit {unit} of {units_per_layer}\n************")

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
                    print(f"\n\tall items - unit_df: {unit_df.shape}")

                # # remove rows where network failed originally and after lesioning this unit - uninteresting
                old_df_length = len(unit_df)
                unit_df = unit_df.loc[unit_df['item_change'] != 0]
                if verbose is True:
                    n_fail_fail = old_df_length - len(unit_df)
                    print(f"\n\t{n_fail_fail} fail-fail items removed - new shape unit_df: {unit_df.shape}")

                # # remove rows where network failed originally and passed after lesioning this unit
                old_df_length = len(unit_df)
                unit_df = unit_df.loc[unit_df['item_change'] != 2]
                if verbose is True:
                    n_fail_pass = old_df_length - len(unit_df)
                    print(f"\n\t{n_fail_pass} fail-pass items removed - new shape unit_df: {unit_df.shape}")

                # # make item fail specific column
                l_fail = [1 if lf == -1 else 0 for lf in unit_df['item_change']]
                unit_df.insert(loc=6, column='l_fail', value=l_fail)
                unit_df = unit_df.drop(columns="item_change")

                print(f"\nunit df:\n{unit_df.head()}")

                # # getting sel measures
                for measure in sel_measures:
                    # # includes if statement since some units have not score (dead relu?)
                    if old_sel_dict:
                        sel_measure_dict = sel_dict['sel_info'][use_layer_name][unit][measure]
                    else:
                        if unit in sel_info[use_layer_name]:
                            sel_measure_dict = sel_info[use_layer_name][unit][measure]

                    unit_df[measure] = unit_df['class'].map(sel_measure_dict)

                master_df = master_df.append(unit_df, ignore_index=True, sort=True)
                print(f"master_df: {master_df.shape}")

        print("\n\n********************************"
              "\nfinished looping through layers"
              "\n********************************\n")

    print(f"master_df: {master_df.shape}")

    # drop columns from master df that I don't need
    data = master_df.drop(columns=['layer', 'unit', 'item', 'class'])  # , 'normed_acts'])

    bin_log_reg = (output_filename, data)

    #######################################################
    # # part 2 , use master to do stats
    if use_relu:
        txt = open(f'{output_filename}_ReLu_item_act_fail.txt', 'w')
    else:
        txt = open(f'{output_filename}_item_act_fail.txt', 'w')

    use_acts = 'hid_acts'
    print(f"\nFor analysis, using {use_acts} from ['hid_acts', 'normed_acts']")
    txt.write(f"\nFor analysis, using {use_acts} from ['hid_acts', 'normed_acts']")

    if use_normed_acts:
        data = master_df.drop(columns=['hid_acts'])
        use_acts = 'normed_acts'
    else:
        data = master_df.drop(columns=['normed_acts'])

    # fill in any NaNs
    if data.isnull().values.any():
        data = data.fillna(0)

    print(f"columns to analyse: {list(master_df)}")

    txt.write(f"\nsel_measures: {sel_measures}\n\nuse_relu: {use_relu}\n"
              f"data shape: {data.shape}\ndata columns: {list(data.columns)}\n")

    # plot distribution of selectivity scores
    colours = sns.color_palette('husl', n_colors=len(sel_measures))
    plt.figure()
    for index, measure in enumerate(sel_measures):
        ax = sns.kdeplot(data[measure], color=colours[index], shade=True)
    plt.legend(sel_measures)
    plt.title('Density Plot of Selectivity measures')
    ax.set(xlabel='Selectivity')
    ax.set_xlim(right=1)
    if use_relu:
        plt.savefig(f"{output_filename}_ReLu_sel_dist.png")
    else:
        plt.savefig(f"{output_filename}_sel_dist.png")
    if test_run:
        plt.show()
    plt.close()

    # plot distribution of hid acts
    plt.figure()
    ax = sns.kdeplot(data[use_acts], color="darkturquoise", shade=True)
    plt.legend([use_acts])
    plt.title('Density Plot of hidden activations')
    ax.set(xlabel='Hidden Activations')
    ax.set_xlim(left=0)
    if use_relu:
        plt.savefig(f"{output_filename}_ReLu_{use_acts}_dist.png")
    else:
        plt.savefig(f"{output_filename}_{use_acts}_dist.png")
    if test_run:
        plt.show()
    plt.close()

    # # compare means
    print("\nget descriptives")
    # print(data.groupby('l_fail').mean())
    l_fail = data[data['l_fail'] == 1]
    l_fail_mean = l_fail[use_acts].mean()
    n_failed = len(l_fail)

    l_passed = data[data['l_fail'] == 0]
    l_passed_mean = l_passed[use_acts].mean()
    n_passed = len(l_passed)

    # # did items fail at zero?
    test_failed = data[data.l_fail == 1]
    failed_at_zero = data[(data[use_acts] == 0.0) & (data.l_fail == 1)]

    # plot distribution of normed_acts for pass and fail
    test_passes = data[data.l_fail == 0]
    plt.figure()
    ax = sns.kdeplot(test_passes[use_acts], color="orange", shade=True, label='passed')
    sns.kdeplot(test_failed[use_acts], color="darkturquoise", shade=True, label='failed')
    plt.title('Density Plot of hidden activations')
    plt.legend([f'passed n={n_passed}', f'failed n={n_failed}'])
    ax.set(xlabel='Normalized Hidden Activations (0:1 per unit)')
    ax.set_xlim([0, 1])
    if use_relu:
        plt.savefig(f"{output_filename}_ReLu_fail_dist.png")
    else:
        plt.savefig(f"{output_filename}_fail_dist.png")
    if test_run:
        plt.show()
    plt.close()

    if failed_at_zero.empty:
        print(f"no items failed with zero activations\n"
              f"The lowest activation was: {test_failed[use_acts].min():.3f}")
        txt.write(f"\nno items failed with zero activations\n"
                  f"The lowest activation was: {test_failed[use_acts].min():.3f}")
    else:
        n_zero_fail, _ = failed_at_zero.shape
        print(f"{n_zero_fail} items failed despite having zero activation")
        txt.write(f"{n_zero_fail} items failed despite having zero activation")

    print("\n\nDescriptives")
    print(f"\nproportion failed: {data['l_fail'].mean():.3f}, n = {n_failed}"
          f"\nproportion passed: {1 - (data['l_fail'].mean()):.3f}, n = {n_passed}")

    txt.write("\n\nDescriptives")
    txt.write(f"\nproportion failed: {data['l_fail'].mean():.3f}, n = {n_failed}"
              f"\nproportion passed: {1 - (data['l_fail'].mean()):.3f}, n = {n_passed}")

    print(f"\nt-test\nl_fail_mean: {l_fail_mean:.3f}; l_passed_mean: {l_passed_mean:.3f}")
    txt.write(f"\n\nt-test\nl_fail_mean: {l_fail_mean:.3f}; l_passed_mean: {l_passed_mean:.3f}")

    t_test_t, t_test_p = stats.ttest_ind(l_fail[use_acts].values, l_passed[use_acts].values)
    if t_test_p < .001:
        print(f"\nt_test_t: {t_test_t:.3f}; t_test_p < .001")
        txt.write(f"\nt_test_t: {t_test_t:.3f}; t_test_p < .001")
    else:
        print(f"\nt_test_t: {t_test_t:.3f}; t_test_p: {t_test_p:.3f}")
        txt.write(f"\nt_test_t: {t_test_t:.3f}; t_test_p: {t_test_p:.3f}")

    # # ROC
    print("\ndata ROC")
    y = [1 if i == 1 else 0 for i in np.array(np.ravel(data['l_fail'].values))]
    x_roc = np.ravel(data[use_acts])

    fpr, tpr, thr = roc_curve(y, x_roc, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # plot ROC
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
    plt.title('ROC curve of activations and failed-when-lesioned')
    plt.legend(loc="lower right")
    if use_relu:
        plt.savefig(f"{output_filename}_ReLu_data_ROC.png")
    else:
        plt.savefig(f"{output_filename}_data_ROC.png")
    if test_run:
        plt.show()
    plt.close()

    print(f"ROC_AUC: {roc_auc}")
    print("ROC AUC is equal to the probability that a random positive example (failed)\n"
          "will be ranked above a random negative example (passed).")
    txt.write(f"\n\nROC AUC\nROC_AUC: {roc_auc:.3f} is equal to the probability that a random positive "
              f"example (failed)\nwill be ranked above a random negative example (passed).")

    '''
    https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
    4.2.1. Model evaluation based on simple train/test split using train_test_split() function
    sklearn output wasn't as detailed as I'd like so I've tried to build an equivillent to their RFE
    model using statsmodel 
    '''

    # # statsmodel version
    # simple models
    # sm_model = sm.Logit(y_train, x_train)
    # sm_model = sm.Logit(y_train, sm.add_constant(x_train))
    # sm_results = sm_model.fit()

    print("\n\n******************************\nStatsmodel Logistic regression\n"
          "Recursive Feature Elimination\n******************************\n")
    txt.write("\n\nStatsmodel Logistic regression\nRecursive Feature Elimination\n")

    selected_features = [use_acts] + sel_measures
    x_sm_selected_feat = data[selected_features]
    print("\nx_sm_selected_feat: ", list(x_sm_selected_feat))

    y = [1 if i == 1 else 0 for i in np.array(np.ravel(data['l_fail'].values))]

    x_train, x_test, y_train, y_test = train_test_split(x_sm_selected_feat, y, test_size=0.2, random_state=2)

    # # recursive feature elimination
    dropped_features = []
    previous_score, new_score = 0.0, 0.0

    feature_counter = []
    score_counter = []

    while len(selected_features) > 0:
        print("\n**********************************************************")
        print(f"\tprevious score {previous_score:.3f}, new score {new_score:.3f}")
        print(f"\t{len(selected_features)} selected_features: {selected_features}")

        # train model on selected features
        x_rfe = x_train[selected_features]
        x_rfe = sm.add_constant(x_rfe)

        sm_rfe_model = sm.Logit(y_train, x_rfe)
        sm_results = sm_rfe_model.fit()

        # get model predictions and score AUC
        sm_predict = sm_results.predict(sm.add_constant(x_test[selected_features])).tolist()
        [fpr, tpr, thr] = roc_curve(y_test, sm_predict)
        sm_score_auc = auc(fpr, tpr)
        new_score = sm_score_auc
        print(f"\tnew_score: {new_score:.3f}")

        # # for making a plot of score vs n_features
        feature_counter.append(len(selected_features))
        score_counter.append(new_score)

        if round(previous_score, 3) > round(new_score, 3):
            print(f"\tprevious score {previous_score} > new score {new_score}")
            selected_features.append(dropped_features[-1])
            print("\tselected_features: ", selected_features)
            break

        sm_p_vals = sm_results.pvalues  # The two-tailed p values for the t-stats of the params
        sm_p_vals = sm_p_vals.drop(['const'])
        # print(f"\nsm_p_vals:\n{sm_p_vals}")

        max_p_val = sm_p_vals.max()
        max_p_feature = sm_p_vals.idxmax()

        if max_p_val > 0.05:
            print(f"\n\tdrop max_p_feature: {max_p_feature} = {max_p_val:.3f}")
            selected_features.remove(max_p_feature)
            dropped_features.append(max_p_feature)
            previous_score = new_score

        else:
            print("\n\tall features now have p < .05")
            break

    print("\n\nRecursive Feature Elimination")
    print(f"Optimal number of features: {len(selected_features)}"
          f"\nSelected features: {selected_features}")
    txt.write("\n\nRecursive Feature Elimination with Cross validation")
    txt.write(f"\nOptimal number of features: {len(selected_features)}"
              f"\nSelected features: {selected_features}")

    # Plot number of features VS. cross-validation scores
    print(f"feature_counter: {feature_counter}\nscore_counter: {score_counter}")
    y_pad = (max(score_counter) - min(score_counter)) / 10

    plt.figure()
    ax = sns.lineplot(x=feature_counter, y=score_counter, markers='o', sort=False)
    plt.xlabel("Number of features selected")
    plt.ylabel("Accuracy score")
    plt.title('Number of features VS. cross-validation scores')
    ax.set_xlim(max(feature_counter), min(feature_counter))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(min(score_counter) - y_pad, max(score_counter) + y_pad)
    if use_relu:
        plt.savefig(f"{output_filename}_ReLu_RFE_score.png")
    else:
        plt.savefig(f"{output_filename}_RFE_score.png")
    if test_run:
        plt.show()
    plt.close()

    # # plot correlations between features
    # plt.subplots()
    plt.figure()
    ax = sns.heatmap(x_train[selected_features].corr(), annot=True, cmap="RdYlGn")
    plt.title("Correlations of selected features")
    ax.set(xlim=(0, len(selected_features)), ylim=(0, len(selected_features)))
    plt.tight_layout()
    # plt.tight_layout(pad=1, h_pad=5.0, )
    if use_relu:
        plt.savefig("{}_ReLu_feat_corr.png".format(output_filename))
    else:
        plt.savefig("{}_feat_corr.png".format(output_filename))
    if test_run:
        plt.show()
    plt.close()

    # # plot score AUC
    print(f"\nsm_predict_auc: {sm_score_auc}")
    txt.write(f"\n\nsm_predict_auc: {sm_score_auc}")

    idx = np.min(np.where(tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95

    plt.figure()
    plt.plot(fpr, tpr, color='coral', label=f"ROC curve (area = {auc(fpr, tpr):0.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Score Auc: y_test vs y-predicted-probability')
    plt.legend(loc="lower right")
    if use_relu:
        plt.savefig(f"{output_filename}_ReLu_model_AUC.png")
    else:
        plt.savefig(f"{output_filename}_model_AUC.png")
    if test_run:
        plt.show()
    plt.close()

    print(f"\nUsing a threshold of {thr[idx]:.3f} guarantees a sensitivity of {tpr[idx]:.3f}\n"
          f"and a specificity of {1 - fpr[idx]:.3f}, "
          f"i.e. a false positive rate of {np.array(fpr[idx]) * 100:.2f}%.")
    txt.write(f"\n\nUsing a threshold of {thr[idx]:.3f} guarantees a sensitivity of {tpr[idx]:.3f}\n"
              f"and a specificity of {1 - fpr[idx]:.3f}, "
              f"i.e. a false positive rate of {np.array(fpr[idx]) * 100:.2f}%.")

    # # seaborn regression plot
    # fig, ax = plt.subplot()
    plt.figure()
    ax = sns.regplot(x=x_test[use_acts], y=y_test, logistic=True, y_jitter=.03, n_boot=100)
    plt.ylabel("Item failed when unit lesioned")
    plt.title(f'Regression plot of item fails and {use_acts}')
    plt.savefig(f"{output_filename}_regplot.png")
    if test_run:
        plt.show()
    plt.close()

    # # accuracy and conf matrix
    sm_bin_pred = [1 if x >= .5 else 0 for x in sm_predict]
    sm_accuracy = accuracy_score(y_test, sm_bin_pred)
    print(f"\nsm_accuracy: {sm_accuracy:.3f}")
    txt.write(f"\n\nsm_accuracy: {sm_accuracy:.3f}")

    sm_conf_matrix = confusion_matrix(y_test, sm_bin_pred)
    print(sm_conf_matrix)
    txt.write(f"\n\nsm_conf_matrix (diagonals are correct)")
    txt.write(f"\n\tpred_0\tpred_1")
    txt.write(f"\ntrue_0\t{sm_conf_matrix[0][0]}\t{sm_conf_matrix[0][1]}")
    txt.write(f"\ntrue_1\t{sm_conf_matrix[1][0]}\t{sm_conf_matrix[1][1]}")

    print(f"\n\nLogistic regression with {selected_features}")
    print(f"LogisticRegression score: {sm_accuracy:.3f}")
    print(f"just guessing 'No' everytime would get you {1 - (np.mean(y)):.3f}")
    print(f'\nThis model does {sm_accuracy - (1 - (np.mean(y))):.3f} better than quessing chance')
    txt.write(f"\n\nLogistic regression with {selected_features}")
    txt.write(f"\nLogisticRegression score: {sm_accuracy:.3f}")
    txt.write(f"\njust guessing no everytime would get you {1 - (np.mean(y)):.3f}")
    txt.write(f'\nThis model does {sm_accuracy - (1 - (np.mean(y))):.3f} better than quessing chance')

    # # model results summary
    print(f"\nsm_results.summary():\n{sm_results.summary()}")
    txt.write(f"\n\nsm_results.summary():\n{sm_results.summary()}")

    '''
    notes about the summary.
    z values are the parameter estimates (coef) divided by their std err.
    p values are calculated with respect to a standard normal distribution (not multiple comparrison corrected)
    '''
    # print("\n sm_results attributes")
    # for attr in dir(sm_results):
    #     if not attr.startswith('_'):
    #         print(attr)

    sm_model_df = int(sm_results.df_model)
    print(f"\nsm_model_df: {sm_model_df:.3f}")
    txt.write(f"\n\nsm_model_df: {sm_model_df:.3f}")
    sm_resid_df = int(sm_results.df_resid)
    print(f"sm_resid_df: {sm_resid_df:.3f}")
    txt.write(f"\nsm_resid_df: {sm_resid_df:.3f}")

    sm_null_ll = sm_results.llnull  # Value of the constant-only loglikelihood
    sm_loglikelihood = sm_results.llf  # Log-likelihood of model
    sm_ll_chi2 = sm_results.llr  # improvement from null, Ll ratio chi-squared statistic; -2*(llnull - llf)
    sm_llr_p_val = sm_results.llr_pvalue  # p val for .llr sm_ll_chi2
    print(f"\nsm_null_ll: {sm_null_ll:.3f}")
    print(f"sm_loglikelihood: {sm_loglikelihood:.3f}")
    print(f"sm_ll_chi2: {sm_ll_chi2:.3f}")
    print(f"sm_llr_p_val: {sm_llr_p_val:.3f}")
    txt.write(f"\n\nsm_null_ll: {sm_null_ll:.3f}")
    txt.write(f"\nsm_loglikelihood: {sm_loglikelihood:.3f}")
    txt.write(f"\nsm_ll_chi2: {sm_ll_chi2:.3f}")
    txt.write(f"\nsm_llr_p_val: {sm_llr_p_val:.3f}")

    print(f"\nX2({sm_model_df} = {sm_ll_chi2:.3f}, p = {sm_llr_p_val:.3f}")
    txt.write(f"\nX2({sm_model_df} = {sm_ll_chi2:.3f}, p = {sm_llr_p_val:.3f}")

    sm_pseudo_r2 = sm_results.prsquared  # McFadden’s pseudo-R-squared (variance explained)
    print(f"\nsm_pseudo_r2: {sm_pseudo_r2:.3f}")
    txt.write(f"\n\nsm_pseudo_r2:{sm_pseudo_r2:.3f}")

    sm_tvalues = sm_results.tvalues  # Return the t-statistic for a given parameter estimate
    sm_p_vals = sm_results.pvalues  # The two-tailed p values for the t-stats of the params
    print(f"\nsm_tvalues:\n{sm_tvalues}")
    print(f"\nsm_p_vals:\n{sm_p_vals}")
    txt.write(f"\n\nsm_tvalues:\n{sm_tvalues}")
    txt.write(f"\n\nsm_p_vals:\n{sm_p_vals}")

    sm_bse = sm_results.bse  # The standard errors of the parameter estimates.
    print(f"\nsm_bse:\n{sm_bse}")
    txt.write(f"\n\nsm_bse:\n{sm_bse}")

    sm_wald_test_terms = sm_results.wald_test_terms()  # Wald tests for terms over multiple predictors

    wald_test_df = pd.DataFrame(data=sm_wald_test_terms.summary_frame())
    wald_sig = []
    for i in sm_wald_test_terms.pvalues:
        if i < .001:
            wald_sig.append('***')
        elif i < .01:
            wald_sig.append('**')
        elif i < .05:
            wald_sig.append('*')
        else:
            wald_sig.append("")
    wald_test_df['sig'] = wald_sig
    print(f"\nwald_test_df:\n{wald_test_df}")
    txt.write(f"\nwald_test_df:\n{wald_test_df}")

    sm_coefficients = sm_results.params
    print(f"\nsm_coefficients:\n{sm_coefficients}")
    txt.write(f"\n\nsm_coefficients:\n{sm_coefficients}")

    sm_conf_int = sm_results.conf_int()
    print(f"\nsm_conf_int:\n{sm_conf_int}")
    txt.write(f"\n\nsm_conf_int:\n{sm_conf_int}")

    # # odds ratio
    sm_odds_ratio = np.exp(sm_coefficients)
    print(f"\nsm_odds_ratio:\n{sm_odds_ratio}")
    txt.write(f"\n\nsm_odds_ratio:\n{sm_odds_ratio}")

    # odds ratios and 95% CI
    sm_params = sm_results.params
    print(f"\nsm_params:\n{sm_params}")
    txt.write(f"\n\nsm_params:\n{sm_params}")

    sm_output_table = sm_conf_int
    sm_output_table['OR'] = sm_odds_ratio
    # sm_output_table = np.exp(sm_output_table)

    sm_output_table['B'] = sm_coefficients
    sm_output_table['SE'] = sm_bse
    sm_output_table.columns = ['B', 'SE', '2.5%', 'OR', '97.5%']
    print(f"\nsm_output_table:\n{sm_output_table}")
    txt.write(f"\n\nsm_output_table:\n{sm_output_table}")

    txt.close()

    if not already_got_master:
        if use_relu:
            master_df.to_pickle(f"{output_filename}_item_act_fail_ReLu_MASTER.pickle")
        else:
            master_df.to_pickle(f"{output_filename}_item_act_fail_MASTER.pickle")

    print("\nEnd of script")


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
    :param sel_measures: list of measures to test or 'all'
    :param verbose: how much to print to screen
    :param test_run: if True...

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
    lesion_highlights = lesion_info["lesion_highlights"]
    # focussed_dict_print(lesion_highlights)
    key_lesion_layers_list = list(lesion_highlights.keys())
    # # remove unnecessary items from key layers list
    # # remove unnecessary items from key layers list
    if 'highlights' in key_lesion_layers_list:
        key_lesion_layers_list.remove('highlights')
    if 'output' in key_lesion_layers_list:
        key_lesion_layers_list.remove('output')
    if 'Output' in key_lesion_layers_list:
        key_lesion_layers_list.remove('Output')
    print(f"\nkey_lesion_layers_list: {key_lesion_layers_list}")

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

    '''get rid of measures that I've had trouble with'''
    if 'nz_count' in auto_sel_measures:
        auto_sel_measures.remove('nz_count')
    if 'max_info_f1' in auto_sel_measures:
        auto_sel_measures.remove('max_info_f1')

    if use_relu is True:
        # # get key_relu_layers_list
        key_relu_layers_list = list(sel_info.keys())
        # # remove unnecessary items from key layers list
        if 'sel_analysis_info' in key_relu_layers_list:
            key_relu_layers_list.remove('sel_analysis_info')
        if 'output' in key_relu_layers_list:
            output_idx = key_relu_layers_list.index('output')
            key_relu_layers_list = key_relu_layers_list[:output_idx]
        if 'Output' in key_relu_layers_list:
            output_idx = key_relu_layers_list.index('Output')
            key_relu_layers_list = key_relu_layers_list[:output_idx]

        # # put together lists of 1. sel_gha_layers, 2. key_lesion_layers_list.
        n_activation_layers = sum("activation" in layers for layers in key_relu_layers_list)
        n_lesion_layers = len(key_lesion_layers_list)

        if n_activation_layers == n_lesion_layers:
            n_layers = n_activation_layers
            activation_layers = [layers for layers in key_relu_layers_list if "activation" in layers]
            link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(activation_layers)))
        else:
            raise TypeError('should be same number of activation layers and lesioned layers')

        if verbose is True:
            focussed_dict_print(link_layers_dict, 'link_layers_dict')

    print("\n\nSaving lesion_corr_dict")
    # # where to save files
    save_plots_name = 'sel_les_corr'
    if use_relu:
        save_plots_name = 'sel_les_corr_ReLu'
    sel_les_corr_path = os.path.join(lesion_dict['GHA_info']['gha_path'], save_plots_name)
    if test_run is True:
        sel_les_corr_path = os.path.join(sel_les_corr_path, 'test')
    if not os.path.exists(sel_les_corr_path):
        os.makedirs(sel_les_corr_path)
    os.chdir(sel_les_corr_path)
    print(f"\nsel_les_corr_path: {os.getcwd()}")

    if sel_measures == 'all':
        sel_measures_list = auto_sel_measures
    else:
        sel_measures_list = sel_measures

    print(f"\nsel_measures_list\n{sel_measures_list}")

    sel_corr_dict = dict()  # key=measure, val=score
    for i in key_lesion_layers_list:
        sel_corr_dict[i] = dict()
    sel_corr_dict['all_layers'] = dict()

    """loop through sel_measures_list
    get long array of all layers concatenated and list all max lesion drops.
    do correlations on this whole thing at once,
    save then repeat for next sel measure"""

    print("\nlooping through network to get selectivity scores as big array")

    sel_measure_counter = 0
    for sel_measure in sel_measures_list:

        print(f"\n\n******{sel_measure}******")

        if test_run:
            sel_measure_counter += 1
            if sel_measure_counter > 2:
                continue

        """loop through all layers"""
        all_layer_sel_array = []  # append each sel_layer to this to give array (each unit, classes)

        all_layer_class_drops = []  # # extend each lesion_layer to this to give list of all units in all layers

        all_les_sel_pairs = []  # max_class_lesion_drop and sel for that class

        # measure_sel_corr_dict = dict()  # key=measure, val=score
        # sel_corr_dict[lesion_layer] = dict()

        for test_layer_count, lesion_layer in enumerate(key_lesion_layers_list):

            if test_run:
                if test_layer_count > 2:
                    continue

            sel_layer = lesion_layer
            if use_relu:
                sel_layer = link_layers_dict[lesion_layer]

            if verbose:
                print(f"\n\tsel_layer: {sel_layer}\tlesion_layer: {lesion_layer}")

            sel_layer_info = sel_info[sel_layer]

            layer_les_sel_pairs = []

            # sel_corr_dict[lesion_layer] = dict()

            # # lesion stuff
            # # conv2d_6
            lesion_per_unit_path = f'{lesion_path}/{output_filename}_{lesion_layer}_prop_change.csv'

            lesion_per_unit = pd.read_csv(lesion_per_unit_path, index_col=0)
            # lesion_per_unit = nick_read_csv(lesion_per_unit_path)
            # lesion_per_unit.set_index(0)

            lesion_cols = list(lesion_per_unit)
            # print("\nlesion_per_unit")
            # print(lesion_cols)
            # print(lesion_per_unit.head())

            '''get max class drop per lesion_layer'''
            # # loop through lesion units (df columns) to find min class drop
            lesion_cat_p_u_dict = dict()
            lesion_unit_cat_list = []
            for index, l_unit in enumerate(lesion_cols):
                # print(lesion_per_unit[l_unit])
                unit_les_val = lesion_per_unit[l_unit].min()
                unit_les_cat = lesion_per_unit[l_unit].idxmin()
                lesion_unit_cat_list.append(int(unit_les_cat))
                # print("{}: class: {}  {}".format(index, unit_les_cat, unit_les_val))
                lesion_cat_p_u_dict[index] = {'unit': index, "l_min_class": unit_les_cat, 'l_min_drop': unit_les_val}

            '''get sel_layer sel values'''
            # get array of sel values
            layer_sel_array = []

            for unit, unit_sel in sel_layer_info.items():

                if unit == 'means':
                    continue

                if old_sel_dict:
                    if sel_measure in unit_sel['sel']:
                        sel_items = unit_sel['sel'][sel_measure]
                    elif sel_measure in unit_sel['class_sel_basics']:
                        sel_items = unit_sel['class_sel_basics'][sel_measure]
                else:
                    sel_items = unit_sel[sel_measure]

                # # just check it is just classes in there
                if 'total' in sel_items.keys():
                    del sel_items['total']
                if 'perplexity' in sel_items.keys():
                    del sel_items['perplexity']

                if len(list(sel_items.keys())) != n_cats:
                    # print("\nERROR, {} hasn't got enough classes".format(sel_measure))
                    # print("error found", sel_items)
                    for i in range(n_cats):
                        if i not in sel_items:
                            sel_items[i] = 0.0
                    ordered_dict = dict()
                    for j in range(n_cats):
                        ordered_dict[j] = sel_items[j]

                    sel_items = dict()
                    sel_items = ordered_dict
                    # print('sel_items should be sorted now', sel_items)

                sel_values = list(sel_items.values())

                layer_sel_array.append(sel_values)

                les_drop_class = int(lesion_cat_p_u_dict[unit]['l_min_class'])
                drop_class_sel = sel_values[les_drop_class]

                all_les_sel_pairs.append([lesion_cat_p_u_dict[unit]['l_min_drop'], drop_class_sel])
                layer_les_sel_pairs.append([lesion_cat_p_u_dict[unit]['l_min_drop'], drop_class_sel])

                # print(unit, sel_values)

            # print("\nlayer_sel_array\n{}".format(layer_sel_array))
            # print("\t\tlayer_sel_array: {}".format(np.shape(layer_sel_array)))

            all_layer_sel_array = all_layer_sel_array + layer_sel_array
            # print("\t\tall_layer_sel_array: {}".format(np.shape(all_layer_sel_array)))

            # # lesion stuff went here...

            # # check for missing values (prob dead relus) when using different layers
            sel_units, classes = np.shape(layer_sel_array)
            les_units = len(lesion_unit_cat_list)
            if use_relu:
                if sel_units != les_units:
                    print("\n\number of units is wrong")
                    print(f"sel_units: {sel_units}\nles_units: {les_units}")

                    if len(lesion_unit_cat_list) > sel_units:
                        available_sel_units = list(sel_layer_info.keys())
                        masked_class_drops = [lesion_unit_cat_list[i] for i in available_sel_units]
                        lesion_unit_cat_list = masked_class_drops

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

            all_layer_class_drops.extend(lesion_unit_cat_list)
            # print("\t\tall_layer_class_drops: {}".format(np.shape(all_layer_class_drops)))

            # focussed_dict_print(lesion_cat_p_u_dict)
            # print("\nlesion_unit_cat_list\n{}".format(lesion_unit_cat_list))
            if verbose:
                print(f"\t\tlayer_sel: {np.shape(layer_sel_array)}; "
                      f"all_layers_sel: {np.shape(all_layer_sel_array)}; "
                      f"all_layers_class_drops: {np.shape(all_layer_class_drops)}")

            # # layer correlations
            # # # sel lesion correlations
            print("layer_les_sel_pairs shape: ", np.shape(layer_les_sel_pairs))
            max_lesion_vals = [i[0] for i in layer_les_sel_pairs]
            sel_pair_score = [i[1] for i in layer_les_sel_pairs]
            print("max_lesion_vals shape: ", np.shape(max_lesion_vals))

            corr_coef, corr_p = stats.pearsonr(sel_pair_score, max_lesion_vals)

            print(f"{sel_measure} corr: {corr_coef}, p = {corr_p}")

            sel_corr_dict[lesion_layer][f'{sel_measure}_corr_coef'] = corr_coef
            sel_corr_dict[lesion_layer][f'{sel_measure}_corr_p'] = corr_p

            sns.regplot(x=sel_pair_score, y=max_lesion_vals)
            plt.ylabel("Max class drop")
            plt.xlabel("selectivity score")
            plt.suptitle(f"{lesion_layer} class drop vs {sel_measure}")
            if round(corr_p, 3) == 0.000:
                plt.title(f"r={corr_coef:.3f}, p<.001")
            else:
                plt.title(f"r={corr_coef:.3f}, p={corr_p:.3f}")
            print(os.getcwd())
            if use_relu:
                plt.savefig(f"{sel_les_corr_path}/{output_filename}_{lesion_layer}_{sel_measure}_ReLu_corr.png")
            else:
                plt.savefig(f"{sel_les_corr_path}/{output_filename}_{lesion_layer}_{sel_measure}_corr.png")
            plt.close()

            # sel_corr_dict[lesion_layer] = measure_sel_corr_dict

        # # # sel lesion correlations
        # todo: Zhou did spearman not pearson.
        #  For this I need to rank selectivity and max class drops and do spearman on these ranks.
        # I am not sure if I need to calculate ranks or whether scipy does that for me scipy.stats.spearmanr
        print("all_les_sel_pairs shape: ", np.shape(all_les_sel_pairs))
        max_lesion_vals = [i[0] for i in all_les_sel_pairs]
        sel_pair_score = [i[1] for i in all_les_sel_pairs]
        print("max_lesion_vals shape: ", np.shape(max_lesion_vals))

        corr_coef, corr_p = stats.pearsonr(sel_pair_score, max_lesion_vals)

        print(f"{sel_measure} corr: {corr_coef:.3f}, p = {corr_p:.3f}")

        # all_layer_sel_corr_dict = dict()
        # sel_corr_dict['all_layers'] = dict()

        sel_corr_dict['all_layers'][f'{sel_measure}_corr_coef'] = corr_coef
        sel_corr_dict['all_layers'][f'{sel_measure}_corr_p'] = corr_p

        sns.regplot(x=sel_pair_score, y=max_lesion_vals)
        plt.ylabel("Max class drop")
        plt.xlabel("selectivity score")
        plt.suptitle(f"class drop vs {sel_measure}")
        if round(corr_p, 3) == 0.000:
            plt.title(f"r={corr_coef:.3f}, p<.001")
        else:
            plt.title(f"r={corr_coef:.3f}, p={corr_p:.3f}")
        print(os.getcwd())
        if use_relu:
            plt.savefig(f"{sel_les_corr_path}/{output_filename}_{sel_measure}_ReLu_corr.png")
        else:
            plt.savefig(f"{sel_les_corr_path}/{output_filename}_{sel_measure}_corr.png")
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

    les_reg_dict_name = f"{sel_les_corr_path}/{output_filename}_les_sel_corr_dict.pickle"
    pickle_out = open(les_reg_dict_name, "wb")
    pickle.dump(lesion_corr_dict, pickle_out)
    pickle_out.close()

    # les_reg_df = pd.DataFrame(data=sel_corr_dict, index=[0])
    les_reg_df = pd.DataFrame.from_dict(data=sel_corr_dict, orient='index')

    print(les_reg_df.head())
    # nick_to_csv(les_reg_df, "{}/{}_les_sel_corr_dict_nm.csv".format(sel_les_corr_path, output_filename))
    les_reg_df.to_csv(f"{sel_les_corr_path}/{output_filename}_les_sel_corr_dict_pd.csv")

    best_corr_list = les_reg_df.loc['all_layers', :].tolist()

    print(f"\nbest_corr_list: {best_corr_list}")

    if test_run:
        sel_measures_list = sel_measures_list[:2]
    best_corr_array = np.array(best_corr_list).reshape(int(len(best_corr_list) / 2), 2)
    best_corr_df = pd.DataFrame(data=best_corr_array, index=sel_measures_list, columns=['corr_coef', 'corr_p'])

    print(f"\nbest_corr_df:\n{best_corr_df}")

    print(f"\nbest_corr_df['corr_p']:\n{best_corr_df['corr_p']}")

    print(f"\nbest_corr_df['corr_p'].to_list():\n{best_corr_df['corr_p'].to_list()}")

    corr_stars = []
    for i in best_corr_df['corr_p'].to_list():
        print(i)
        if i < .001:
            corr_stars.append(3)
        elif i < .01:
            corr_stars.append(2)
        elif i < .05:
            corr_stars.append(1)
        else:
            corr_stars.append(0)
    print(f"\ncorr_stars:\n{corr_stars}")

    best_corr_df['sig_stars'] = corr_stars

    print(f"\nbest_corr_df: {best_corr_df}")

    abs_coef = [abs(i) for i in best_corr_df['corr_coef'].to_list()]
    best_corr_df['abs_coef'] = abs_coef

    print(f"\nabs_coef: {abs_coef}")

    best_corr_df = best_corr_df.sort_values(by=['sig_stars', 'abs_coef'], ascending=False)

    print(f"\nbest_corr_df:\n{best_corr_df}")

    return lesion_corr_dict

#
# #
# # # #########################
# print("\nWARNING _ RUNNING SCRIPT FROM BOTTOM OF PAGE!!!\n\n\n\n\n")
