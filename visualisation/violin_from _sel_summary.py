import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os.path


"""This script should:
1. Get summary csv (top) or master_sel_p_u (bottom)
2. Decide which conditions I want to include (which sizes, datasets, networks etc)
    networks    use list_of_conditions and list of condition names 
    sizes       use min_hid_units and max_hid_units
    datasets    use selected_datasets
3. input how I want the data to be grouped
    by dataset and by network type
4. produce a barplot showing selectivities by dataset for each measure

5. Variables of interest:
    V1: data_type
    V2: between
    V3: Within
    V4: dset_cond (e.g., HBHW, MBLW etc)
    V5: act_func
    cond: cond-name
    trained_for: epochs (0-500)
    gha_acc:
    mean_info: mean max info per model
    mean_bsel: mean b-sel per model
    max_bsel: per model
    mean_bsel_0: proportion of units with bsel > 0 per model
    mean_bsel_.5: proportion of units with bsel > .5 per model

8. could the script be improved by using dctionaries to access information rather than lists/dfs?    


    """
# # part 1 # # load summary csv - check conditions

topic_name = 'cos_sim2'
# topic_name = 'cos_sim_3_vary'

experiment_chapter = 'within_between_dist_july2020'
summary_df_name = f"{topic_name}_sel_summary"  # don't need to include the .csv bit
summary_df_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
                 f'{experiment_chapter}/{topic_name}/{summary_df_name}.csv'

per_condition_headers = ['V5', 'V1', 'hid_units', 'V4', 'V2', 'V3',
                         'cond', 'trained_for', 'gha_acc',
                         'mean_info', 'mean_bsel', 'mean_bsel_0',
                         'max_bsel', 'mean_bsel_.5',
                         # 'mean_combo', 'max_combo'
                         ]

summary_df = pd.read_csv(summary_df_path, usecols=per_condition_headers)
print("\ntopic name {}".format(topic_name))
print("summary_df.shape: {}".format(summary_df.shape))

summary_df = summary_df.rename(columns={"V5": "act_func", "V1": "data_type", "V4": "dset_cond",
                                       "V2": "between", "V3": "within"})
print("summary_df.columns.values: {}\n".format(summary_df.columns.values))

print(summary_df.head())

bin_cont = [("bin" if b == "bin" else "cont") for b in summary_df['data_type']]
summary_df['data_type'] = bin_cont
# # # make pivot table
# summary_pivot = pd.pivot_table(summary_df, index=['data_type', 'act_func', 'hid_units'],
#                                columns=['dataset'], values=['cond'], aggfunc='count', margins=True)
# print("summary_pivot {}".format(summary_pivot))
# summary_pivot.to_csv("{}_summary_pivot.csv".format(topic_name))


# # part 2. decide on how to group the data (dependent variables)
# # decide whether to include simulations based on 'data_type', 'hid_units', 'act_func', 'dataset', 'gha_acc',
use_dset_type = 'cont'  # 'all', 'bin', 'cont'
use_act_func = 'sigm'  # 'all', 'sigm', 'ReLu
min_hid_units = 10
max_hid_units = 500
min_gha_acc = .15

# # sanity check on layers
get_values = summary_df.act_func.unique()
for i in range(len(get_values)):
    get_value_count = summary_df.act_func.value_counts()[get_values[i]]
    print("{} value: {} {}".format(i, get_values[i], get_value_count))
print(get_values)

if use_dset_type is 'all':
    print("use all dset_types")
else:
    print(f"use_dset_type: {use_dset_type}")
    summary_df = summary_df.loc[summary_df['data_type'] == use_dset_type]
    print(f"summary_df.shape: {summary_df.shape}")

if use_act_func is 'all':
    print("use all act_funcs")
else:
    print(f"use_act_func: {use_act_func}")
    summary_df = summary_df.loc[summary_df['act_func'] == use_act_func]
    print(f"summary_df.shape: {summary_df.shape}")


if (use_act_func == 'all') and (use_dset_type == 'all'):
    plots_dir = 'ALL_summ'
    plot_title_use = 'All'
elif use_act_func is 'sigm':
    plots_dir = f'{use_dset_type}_sig'
    plot_title_use = f'{use_dset_type} data, sigmoid units'

elif use_act_func is 'ReLu':
    plots_dir = f'{use_dset_type}_relu'
    plot_title_use = f'{use_dset_type} data, relu units'
else:
    raise ValueError(f"Where should these plots go?\n"
                     f"use_dset_type: {use_dset_type}\n"
                     f"use_act_func: {use_act_func}")

plots_path = f'/home/nm13850/Documents/PhD/python_v2/experiments/{experiment_chapter}/' \
             f'{topic_name}/Plots/{plots_dir}/'

if not os.path.isdir(plots_path):
    os.makedirs(plots_path)

summary_df = summary_df[summary_df['hid_units'].between(min_hid_units, max_hid_units, inclusive=True)]
# print("select_hid_units.shape: {}".format(select_hid_units.shape))
summary_df = summary_df[summary_df['gha_acc'] > min_gha_acc]
# print("summary_df.shape: {}".format(summary_df.shape))

filtered_df = summary_df
print(f"\nsummary_df.shape: {summary_df.shape}\n\n{summary_df.head()}")


# IV = ['dset_cond', 'act_func', 'data_type', 'between', 'within', 'hid_units']
IV = ['dset_cond']
# IV = ['data_type', 'hid_units']

DV = ["trained_for"]  #, "gha_acc", "mean_info",
      # "mean_bsel", "mean_bsel_0", "max_bsel", 'mean_bsel_.5', 'mean_combo', 'max_combo']

font_size = 10
counter = 0

for var_name in IV:
    for measure in DV:
        plot_type = 'violin'
        use_order = False
        counter += 1
        variable = var_name
        print(f"{counter}:\tIV: {var_name}\tDV: {measure}")

        if var_name is 'dset_cond':
            order = ["HBHW", "MBHW", "MBMW", "LBHW", "LBMW", "LBLW"]
            use_order = True
            by = 'between and within class similarity'

        elif var_name is 'dset':
            order = ["pro_sm_", "pro_med"]
            use_order = True
            by = 'Prototype difference'

        elif var_name is 'act_func':
            by = 'activation function'

        elif var_name is 'data_type':
            # order = ['bin', 'chanDist', 'chanProp']
            order = ['bin', 'cont']
            use_order = True
            by = 'data type'

        elif var_name is 'between':
            order = ['HB', 'MB', 'LB']
            use_order = True
            by = 'between class similarity'

        elif var_name is 'within':
            order = ['HW', 'MW', 'LW']
            use_order = True
            by = 'within class similarity'

        elif var_name is 'hid_units':
            by = 'layer size'

        # elif var_name is 'output_class':
        #     by = 'Frequency'
        #     variable = list(range(n_cats))
        #     # plot_type = 'countplot'

        else:
            raise ValueError(f'unknown independent var_name: {var_name}')


        # measure (DV)
        print(f"measure: {measure}\nmeasure[-2:]: {measure[-2:]}")

        if measure is 'combo':
            m_name = 'Combo selectivity'

        elif measure is 'max_informed':
            m_name = 'Maximum Informedness'

        elif measure is 'max_info_sens':
            m_name = 'Sensitivity at Max. info'

        elif measure is 'b_sel':
            m_name = "Bowers' selectivity"

        elif measure is 'means':
            m_name = 'Mean activation'

        elif measure is 'sd':
            m_name = 'Std deviation (activation)'

        elif measure is 'nz_prop':
            m_name = 'Prop. of items w activation > 0'

        elif measure is 'hi_val_prop':
            m_name = 'Prop. of items w high activation'

        elif measure is 'trained_for':
            m_name = 'epochs'

        elif measure[-2:] == '_c':
            measure_root = measure[:-2]
            m_name = f'Class with highest {measure_root}'
            plot_type = 'countplot'

        else:
            raise ValueError(f'unknown measure\n'
                             f'measure: {measure}')

        print(f"plot_type: {plot_type}")

        # plt.figure(figsize=(3, 2.571))
        plt.figure(figsize=(3.5, 3))
        # plt.figure(figsize=(4.08, 3.5))


        if plot_type == 'violin':

            if use_order:
                ax = sns.violinplot(x=variable, y=measure, data=summary_df, cut=0,
                                    order=order,
                                    scale="count",
                                    # showmeans=True,
                                    )
            else:
                ax = sns.violinplot(x=variable, y=measure, data=summary_df, cut=0,
                                    scale="count",

                                    # showmeans=True,
                                    )

            # # # to show means - its ugly
            # ax = sns.pointplot(x=variable, y=measure, data=summary_df,
            #                    estimator=np.mean, join=False, ci=None,
            #                    # markers="+",
            #                    color='grey',
            #                    scale=1.5)


        elif plot_type == 'countplot':
            ax = sns.countplot(x=measure, data=summary_df)

        ax.set_title(f"{plot_title_use}: {m_name}\nby {by}", fontsize=font_size, wrap=True)

        if var_name is 'data_typetype':
            # ax.set_xticklabels(["Binary", "Cont-chanDist", "Cont-chanProp"])
            ax.set_xticklabels(["Binary", "Cont"], fontsize=font_size)

        elif var_name is 'act_func':
            ax.set_xticklabels(["ReLu", "Sigmoid"], fontsize=font_size)
        elif var_name in ['between', 'within']:
            ax.set_xticklabels(['High', 'Medium', 'Low'], fontsize=font_size)

        elif var_name is 'dset_cond':
            ax.set_xticklabels(labels=order, rotation=20)


        ax.set_xlabel(f"{by}", fontsize=font_size)
        ax.set_ylabel(f"{m_name}", fontsize=font_size)

        if measure is 'combo':
            ax.axhline(1.0, ls='--', color='grey')
        elif measure is 'b_sel':
            ax.axhline(0.0, ls='--', color='grey')

        # plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        # plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        # plt.subplots_adjust(wspace=0)
        plt.tight_layout()
        # plt.tight_layout(
            # pad=0.2,
            # w_pad=0.5, h_pad=1.0
        # )

        # show means associated with plot
        # print(f"\nHere's the means:\n{summary_df.groupby(variable)[measure].mean()}")
        print(summary_df.groupby(variable)[measure].mean())
        print(summary_df.groupby(variable).mean())
        # print()
        # print()



        plot_name = f'{topic_name}_{plots_dir}_{measure}_by_{var_name}'
        # plt.show()
        plt.savefig(os.path.join(plots_path, plot_name))
        plt.close()

        print(f"save as: {os.path.join(plots_path, plot_name)}")




print("script finished")


# # # part 1 # # load sel per unit summary csv - check conditions
# topic_name = 'cos_sim2'
# experiment_chapter = 'within_between_dist_july2020'
# master_df_name = f"{topic_name}_sel_p_u_master"  # don't need to include the .csv bit
# master_df_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
#                  f'{experiment_chapter}/{topic_name}/{topic_name}_sel_p_u_master.csv'
#
# # # part 2. decide on how to group the data (dependent variables)
# # # decide whether to include simulations based on 'data_type', 'hid_units', 'act_func', 'dataset', 'gha_acc',
# use_dset_type = 'all'  # 'all', 'bin', 'cont', 'chanprop', 'chanDist'
# use_act_func = 'all'  # 'all', 'sigm' 'ReLu'
# min_hid_units = 9
# max_hid_units = 501
# min_gha_acc = .15
# n_cats = 10
#
# # # variables
# IV = [
#     # 'dset'
#     'cos_sim',
#     #   'act_func',
#     # 'dtype',
#     #   'between', 'within',
#     # 'n_units'
#     # 'output_class'
#       ]
# # DV = ["trained_for", "gha_acc",
# #       'combo', "max_informed", "max_info_sens", "b_sel",
# #       "means", 'sd', 'nz_prop', 'hi_val_prop']
# # DV = ['combo', "max_informed", "max_info_sens", "b_sel",
# #       "trained_for", "means", 'sd', 'nz_prop', 'hi_val_prop']
# DV = ["means", "max_informed", "b_sel"]
# # DV = [
# # # #     # "means",
# # # #     "means_c",
# # # #      "max_informed",
# # # #     "max_informed_c",
# # # #     # "b_sel",
# # # #     "b_sel_c"
# #     'nz_prop', 'hi_val_prop'
# #         ]
#
#
# # # plot details
# font_size = 10
#
#
#
# master_df = pd.read_csv(master_df_path)
# print(f"\ntopic name {topic_name}")
# print(f"master_df.shape: {master_df.shape}")
# print(master_df.head())
#
# for index, header in enumerate(list(master_df)):
#     print(index, header)
# # print(list(master_df))
#
#
#
#
# if (use_act_func == 'all') and (use_dset_type == 'all'):
#     plots_dir = 'ALL_sel_p_u'
#     plot_title_use = 'All'
# elif use_act_func is 'sigm':
#     plots_dir = f'{use_dset_type}_sig'
#     plot_title_use = f'{use_dset_type} data, sigmoid units'
#
# elif use_act_func is 'ReLu':
#     plots_dir = f'{use_dset_type}_relu'
#     plot_title_use = f'{use_dset_type} data, relu units'
#
# else:
#     raise ValueError(f"Where should these plots go?\n"
#                      f"use_dset_type: {use_dset_type}\n"
#                      f"use_act_func: {use_act_func}")
#
# plots_path = f'/home/nm13850/Documents/PhD/python_v2/experiments/{experiment_chapter}/' \
#              f'{topic_name}/Plots/{plots_dir}/'
#
# if not os.path.isdir(plots_path):
#     os.makedirs(plots_path)
#
# # # sanity check on layers
# # get_values = master_df.dset.unique()
# # for i in range(len(get_values)):
# #     get_value_count = master_df.dset.value_counts()[get_values[i]]
# #     print("{} value: {} {}".format(i, get_values[i], get_value_count))
# # print(get_values)
#
# if use_dset_type is 'all':
#     print("use all dset_types")
# else:
#     print(f"use_dset_type: {use_dset_type}")
#     master_df = master_df.loc[master_df['dtype'] == use_dset_type]
#     print(f"master_df.shape: {master_df.shape}")
#
# if use_act_func is 'all':
#     print("use all act_funcs")
# else:
#     print(f"use_act_func: {use_act_func}")
#     master_df = master_df.loc[master_df['act_func'] == use_act_func]
#     print(f"master_df.shape: {master_df.shape}")
#
#
# master_df = master_df[master_df['n_units'].between(min_hid_units, max_hid_units, inclusive=True)]
# # print("select_hid_units.shape: {}".format(select_hid_units.shape))
# master_df = master_df[master_df['gha_acc'] > min_gha_acc]
# # print("master_df.shape: {}".format(master_df.shape))
#
# filtered_df = master_df
# print(f"\nmaster_df.shape: {master_df.shape}\n\n{master_df.head()}")
#
#
# # IV_datasets_df.to_csv("{}IV_datasets_df.csv".format(topic_name))
#
#
# counter = 0
#
# for var_name in IV:
#     for measure in DV:
#         plot_type = 'violin'
#         use_order = False
#         counter += 1
#         variable = var_name
#         print(f"{counter}:\tIV: {var_name}\tDV: {measure}")
#
#         if var_name is 'cos_sim':
#             order = ["HBHW", "MBHW", "MBMW", "LBHW", "LBMW", "LBLW"]
#             use_order = True
#             by = 'between and within class similarity'
#
#         elif var_name is 'dset':
#             order = ["pro_sm_", "pro_med"]
#             use_order = True
#             by = 'Prototype difference'
#
#         elif var_name is 'act_func':
#             by = 'activation function'
#
#         elif var_name is 'dtype':
#             # order = ['bin', 'chanDist', 'chanProp']
#             order = ['bin', 'cont']
#             use_order = True
#             by = 'data type'
#
#         elif var_name is 'between':
#             order = ['HB', 'MB', 'LB']
#             use_order = True
#             by = 'between class similarity'
#
#         elif var_name is 'within':
#             order = ['HW', 'MW', 'LW']
#             use_order = True
#             by = 'within class similarity'
#
#         elif var_name is 'n_units':
#             by = 'layer size'
#
#         elif var_name is 'output_class':
#             by = 'Frequency'
#             variable = list(range(n_cats))
#             # plot_type = 'countplot'
#
#         else:
#             raise ValueError(f'unknown independent var_name: {var_name}')
#
#
#         # measure (DV)
#         print(f"measure: {measure}\nmeasure[-2:]: {measure[-2:]}")
#
#         if measure is 'combo':
#             m_name = 'Combo selectivity'
#
#         elif measure is 'max_informed':
#             m_name = 'Maximum Informedness'
#
#         elif measure is 'max_info_sens':
#             m_name = 'Sensitivity at Max. info'
#
#         elif measure is 'b_sel':
#             m_name = "Bowers' selectivity"
#
#         elif measure is 'means':
#             m_name = 'Mean activation'
#
#         elif measure is 'sd':
#             m_name = 'Std deviation (activation)'
#
#         elif measure is 'nz_prop':
#             m_name = 'Prop. of items w activation > 0'
#
#         elif measure is 'hi_val_prop':
#             m_name = 'Prop. of items w high activation'
#
#         elif measure is 'trained_for':
#             m_name = 'epochs'
#
#         elif measure[-2:] == '_c':
#             measure_root = measure[:-2]
#             m_name = f'Class with highest {measure_root}'
#             plot_type = 'countplot'
#
#         else:
#             raise ValueError(f'unknown measure\n'
#                              f'measure: {measure}')
#
#         print(f"plot_type: {plot_type}")
#
#         plt.figure(figsize=(3.5, 3))
#         # plt.figure(figsize=(4.08, 3.5))
#
#         if plot_type == 'violin':
#             if use_order:
#                 ax = sns.violinplot(x=variable, y=measure, data=master_df, cut=0,
#                                     scale="count",
#                                     order=order)
#             else:
#                 ax = sns.violinplot(x=variable, y=measure, data=master_df, cut=0,
#                                     scale="count",
#                                     )
#
#         elif plot_type == 'countplot':
#             ax = sns.countplot(x=measure, data=master_df)
#
#         ax.set_title(f"{plot_title_use}: {m_name}\nby {by}", fontsize=font_size, wrap=True)
#
#         if var_name is 'dtype':
#             # ax.set_xticklabels(["Binary", "Cont-chanDist", "Cont-chanProp"])
#             ax.set_xticklabels(["Binary", "Cont"], fontsize=font_size)
#
#         elif var_name is 'act_func':
#             ax.set_xticklabels(["ReLu", "Sigmoid"], fontsize=font_size)
#         elif var_name in ['between', 'within']:
#             ax.set_xticklabels(['High', 'Medium', 'Low'], fontsize=font_size)
#
#         elif var_name is 'cos_sim':
#             ax.set_xticklabels(labels=order, rotation=20)
#
#
#         ax.set_xlabel(f"{by}", fontsize=font_size)
#         ax.set_ylabel(f"{m_name}", fontsize=font_size)
#
#         if measure is 'combo':
#             ax.axhline(1.0, ls='--', color='grey')
#         elif measure is 'b_sel':
#             ax.axhline(0.0, ls='--', color='grey')
#
#         # plt.tight_layout(rect=[0, 0.03, 1, 0.90])
#         # plt.tight_layout(rect=[0, 0.01, 1, 0.99])
#         # plt.subplots_adjust(wspace=0)
#         plt.tight_layout()
#         # plt.tight_layout(
#         #     pad=0.4, w_pad=0.5, h_pad=1.0
#         # )
#         # plt.show()
#         plot_name = f'{topic_name}_{plots_dir}_{measure}_by_{var_name}'
#         plt.savefig(os.path.join(plots_path, plot_name))
#         plt.close()
#
#
#
# print("script finished")
