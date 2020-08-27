import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os.path

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats


'''
This script takes in a summary csv, and loops through cond names.
It then opens the sel_p_u csv for each condition
It takes the useful cols from sel_p_u and adds the condition info from the summary, 
and puts this into one master df with sel scores for each unit.

collapse chanProp and chanDist into one since they turned out the same.

# This scripts also makes a COMBO column.
# If B_sel > 0:
#     combo = MI + Bsel
# else:
#     combo = MI 

'''


# def master_sel_from_sel_p_u(exp_path, ):
topic_name = 'cos_sim_3_vary'

# exp path should contain sel_summary and cond folders
exp_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
           'within_between_dist_july2020/cos_sim_3_vary/'

sel_sum_name = 'cos_sim_3_vary_sel_summary.csv'

layer_names = ['hid0']

# summary_csv columns to use
sum_headers = ['output_filename', 'V5', 'hid_units', 'V1', 'V2',
               'cond', 'trained_for', 'gha_acc',
               'mean_info', 'mean_bsel', 'mean_bsel_0',
               'max_bsel', 'mean_bsel_.5',
               # 'mean_combo', 'max_combo'
               ]

# sel_p_u columns to use
sel_p_u_headers = [
            "unit",
            # "roc_auc", "ave_prec", "pr_auc",
            "max_informed", "max_informed_c",
            # "max_info_sens",
            # "max_info_count", "max_info_thr", "max_info_spec", "max_info_prec",
            # "ccma",
            "b_sel", "b_sel_c",
            # "b_sel_off", "b_sel_zero",	"b_sel_pfive",
            # "zhou_prec", "zhou_selects", "zhou_thr",
            # "corr_coef", "corr_p",
            "means", "means_c"
            # "sd",
            # "nz_count", "nz_prec",
            # "nz_prop", "hi_val_prop",
            # "hi_val_count", "hi_val_prec"
            ]

sel_summary_path = os.path.join(exp_path, sel_sum_name)

summary_df = pd.read_csv(sel_summary_path, usecols=sum_headers)
print(f"\ntopic name {topic_name}")
print(f"summary_df.shape: {summary_df.shape}\n\n{summary_df.head()}")

master_list = []

# for row in summary_df.head().itertuples():
for row in summary_df.itertuples():
    # print(row)
    cond_name = row.output_filename

    for layer in layer_names:

        sel_p_u_path = f'{exp_path}/' \
                       f'{cond_name}/' \
                       f'correct_train_set_gha/correct_sel/' \
                       f'{cond_name}_{layer}_sel_p_unit.csv'

        sel_p_u = pd.read_csv(sel_p_u_path, delimiter=',', header=0, usecols=sel_p_u_headers)

        # # collapse chanDist and chanProp into one since they are the same.
        # print(f"row.V1: {row.V1}")
        if row.V1 == 'bin':
            cont_bin = 'bin'
        elif row.V1 in ['chanProp', 'chanDist', 'cont', 'Cont']:
            cont_bin = 'cont'
        else:
            raise ValueError(f'Unknown dtype: {row.V1}')

        # # condition info to go with unit sel
        sel_p_u['filename'] = cond_name
        sel_p_u['n_units'] = row.hid_units
        sel_p_u['trained_for'] = row.trained_for
        # sel_p_u['dtype'] = row.V1
        sel_p_u['dtype'] = cont_bin
        sel_p_u['proto_var'] = row.V2
        # sel_p_u['within'] = row.V3
        # sel_p_u['cos_sim'] = row.V4
        sel_p_u['act_func'] = row.V5
        sel_p_u['gha_acc'] = row.gha_acc

        # print(sel_p_u.head())
        # print(list(sel_p_u))

        master_list.append(sel_p_u)

master_df = pd.concat(master_list, axis=0, ignore_index=True)
master_df = master_df[['filename', 'n_units', 'act_func',
                       'dtype',
                       # 'between', 'within', 'cos_sim',
                       'trained_for', 'gha_acc',
                       'unit',
                       'max_informed', 'max_informed_c',
                       # 'max_info_sens', 'ccma',
                       'b_sel', 'b_sel_c',
                       # 'b_sel_off', 'b_sel_zero', 'b_sel_pfive',
                       'means', 'means_c',
                       # 'sd', 'nz_prop', 'hi_val_prop'
                       ]]


# make combo column, if bsel > 0, add bsel and MI, else just MI
master_df['combo'] = np.where(master_df['b_sel'] >= 0,
                              master_df['b_sel'] + master_df['max_informed'],
                              master_df['max_informed'])

print(master_df.shape)
print(master_df.head())

master_name = f'{topic_name}_sel_p_u_master.csv'
master_path = os.path.join(exp_path, master_name)
print(f"saving master to: {master_path}")

master_df.to_csv(master_path, index=False, header=True)