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
    V1: dset_type
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
# # # part 1 # # load summary csv - check conditions
# summary_df_name = "cos_sim2_sel_summary"  # don't need to include the .csv bit
# 
# topic_name = 'cos_sim2'
# 
# per_condition_headers = ['V5', 'V1', 'hid_units', 'V4', 'V2', 'V3',
#                          'cond', 'trained_for', 'gha_acc',
#                          'mean_info', 'mean_bsel', 'mean_bsel_0',
#                          'max_bsel', 'mean_bsel_.5',
#                          'mean_combo', 'max_combo'
#                          ]
# 
# summary_df = pd.read_csv("{}.csv".format(summary_df_name), usecols=per_condition_headers)
# print("\ntopic name {}".format(topic_name))
# print("summary_df.shape: {}".format(summary_df.shape))
# 
# summary_df = summary_df.rename(columns={"V5": "act_func", "V1": "dset_type", "V4": "dset_cond",
#                                        "V2": "between", "V3": "within"})
# print("summary_df.columns.values: {}\n".format(summary_df.columns.values))
# 
# print(summary_df.head())
# 
# # # # make pivot table
# # summary_pivot = pd.pivot_table(summary_df, index=['dset_type', 'act_func', 'hid_units'],
# #                                columns=['dataset'], values=['cond'], aggfunc='count', margins=True)
# # print("summary_pivot {}".format(summary_pivot))
# # summary_pivot.to_csv("{}_summary_pivot.csv".format(topic_name))
# 
# 
# # # part 2. decide on how to group the data (dependent variables)
# # # decide whether to include simulations based on 'dset_type', 'hid_units', 'act_func', 'dataset', 'gha_acc',
# use_dset_type = 'all'  # 'chanprop', 'chanDist'
# use_act_func = 'all'  # 'sigmoid'
# min_hid_units = 10
# max_hid_units = 500
# min_gha_acc = .15
# 
# # # sanity check on layers
# get_values = summary_df.within.unique()
# for i in range(len(get_values)):
#     get_value_count = summary_df.within.value_counts()[get_values[i]]
#     print("{} value: {} {}".format(i, get_values[i], get_value_count))
# print(get_values)
# 
# if use_dset_type is 'all':
#     print("use all dset_types")
# else:
#     print(f"use_dset_type: {use_dset_type}")
#     summary_df = summary_df.loc[summary_df['dset_type'] == use_dset_type]
#     print(f"summary_df.shape: {summary_df.shape}")
# 
# if use_act_func is 'all':
#     print("use all act_funcs")
# else:
#     print(f"use_act_func: {use_act_func}")
#     summary_df = summary_df.loc[summary_df['act_func'] == use_act_func]
#     print(f"summary_df.shape: {summary_df.shape}")
# 
# 
# summary_df = summary_df[summary_df['hid_units'].between(min_hid_units, max_hid_units, inclusive=True)]
# # print("select_hid_units.shape: {}".format(select_hid_units.shape))
# summary_df = summary_df[summary_df['gha_acc'] > min_gha_acc]
# # print("summary_df.shape: {}".format(summary_df.shape))
# 
# filtered_df = summary_df
# print(f"\nsummary_df.shape: {summary_df.shape}\n\n{summary_df.head()}")
# 
# #
# # # # # sort datasets
# # print("\n********\nDATASETS\n********")
# # orig_datasets = remove_low_acc['dataset'].unique()
# #
# # # # # To type in new dataset names as I go, use this
# # # new_dataset_names = []
# # # for i in range(len(orig_datasets)):
# # #     print("{}: {}".format(i, orig_datasets[i]))
# # #     new_name = input("\atype new name for this datasets\npress ENTER when finished")
# # #     new_dataset_names.append(new_name)
# # # print("new_dataset_names {}".format(new_dataset_names))
# #
# # # # If I already know new dataset names
# # new_dataset_names = ['MNIST', 'MNIST arb50', 'MNIST arb100', 'Iris', 'Iris arb50', 'Iris arb100']
# #
# # # # Stick with original names
# # # new_dataset_names = orig_datasets
# #
# # number_of_datasets = len(orig_datasets)
# # print("number_of_datasets {}".format(number_of_datasets))
# #
# # # for i in range(number_of_datasets):
# #     # print("{}: {} aka {}".format(i, orig_datasets[i], new_dataset_names[i]))
# #
# # # # select the datasets I want to include
# # # # can also chose the order here
# # # # to use ALL datasets choose # selected_datasets = list(range(number_of_datasets))
# # # selected_datasets = list(range(number_of_datasets))
# #
# # # # If I already know which datasets to include use this
# # selected_datasets = [3, 4, 5, 0, 1, 2]  # NEW_arb1
# #
# # # # # To manually enter datasets and order..
# # # selected_datasets = [int(x) for x in input("In the correct order, enter the number of the datasets to use\n"
# # #                                            "separate with spaces\npress ENTER when finished").split()]
# #
# # number_of_selected_datasets = len(selected_datasets)
# # selected_dataset_orig_names = [orig_datasets[i] for i in selected_datasets]
# # selected_dataset_new_names = [new_dataset_names[i] for i in selected_datasets]
# # # print(selected_dataset_new_names)
# #
# # print("\n{} selected for analysis".format(number_of_selected_datasets))
# # for i in range(number_of_selected_datasets):
# #     this_selected_dataset = selected_datasets[i]
# #     print("{}: {}\t\t\taka: {}".format(this_selected_dataset, selected_dataset_new_names[i],
# #                                        selected_dataset_orig_names[i]))
# #
# # IV_datasets_df = remove_low_acc[remove_low_acc['dataset'].isin(selected_dataset_orig_names)]
# #
# #
# # # # # add new dataset names to df
# # # # also add what manipulation was performed
# # # # and what the original dataset was
# # df_rows, df_cols = IV_datasets_df.shape
# # # print("\nIV_datasets_df.shape: {}".format(IV_datasets_df.shape))
# # print("{} simulations incuded in analysis".format(df_rows))
# # new_dset_names = []
# #
# # roots = ['iris', 'MNIST']
# # n_roots = len(roots)
# # manipulations = ['arb50', 'arb100']
# # n_manip = len(manipulations)
# #
# # dset_root = []
# # dset_manip = []
# #
# # for i in range(df_rows):
# #     # # update new dataset name
# #     this_dataset = IV_datasets_df.iloc[i]['dataset']
# #     which_dset = np.where(IV_datasets_df.iloc[i]['dataset'] == orig_datasets)[0][0]
# #     new_dset = new_dataset_names[which_dset]
# #     new_dset_names.append(new_dset)
# #
# #     # update dataset root
# #     root_matches = 0
# #     for j in range(len(roots)):
# #         if roots[j] in new_dset:
# #             # print("from root {}".format(roots[j]))
# #             root_idx = roots[j]
# #             root_matches += 1
# #     if root_matches == 1:
# #         dset_root.append(root_idx)
# #     else:
# #         dset_root.append('ERROR')
# #
# #     # update dataset manipulation
# #     manip_matches = 0
# #     for k in range(len(manipulations)):
# #         if manipulations[k] in new_dset:
# #             # print("from root {}".format(roots[k]))
# #             manipulations_idx = manipulations[k]
# #             manip_matches += 1
# #     if manip_matches == 1:
# #         dset_manip.append(manipulations_idx)
# #     elif manip_matches == 0:
# #         dset_manip.append("original")
# #     else:
# #         dset_manip.append('ERROR')
# #
# #
# # # # # add df_cols signalling dataset parent (e.g., was is based on iris or mnist)
# # IV_datasets_df.insert(6, 'new_dset_names', new_dset_names)
# # IV_datasets_df.insert(7, 'dset_root', dset_root)
# # IV_datasets_df.insert(8, 'dset_manip', dset_manip)
# #
# # # # #
# # # make col for sel > .5
# # # #
# # mean_above_five = []
# # for index, row in IV_datasets_df.iterrows():
# #     mean_above_five.append(100 - row.mean_below_five)
# # IV_datasets_df['mean_above_five'] = mean_above_five
# #
# #
# # print("\nIV_datasets_df.shape: {}".format(IV_datasets_df.shape))
# # print("\nIV_datasets_df.columns.values: {}".format(IV_datasets_df.columns.values))
# # print(IV_datasets_df.head())
# #
# # IV_datasets_df.to_csv("{}IV_datasets_df.csv".format(topic_name))
# IV = ['dset_cond', 'act_func', 'dset_type', 'between', 'within', 'hid_units']
# DV = ["trained_for", "gha_acc", "mean_info",
#       "mean_bsel", "mean_bsel_0", "max_bsel", 'mean_bsel_.5', 'mean_combo', 'max_combo']
# #
# # between_level = 'dset_root'
# # within_level = 'dset_manip'
# # interactions = 'new_dset_names'
# #
# # comparrison = ['dataset_type', 'data_manipulations', 'dataset_type and data_manipulations']
# # comparrison_level_names = ["between", "within", 'interactions']
# # comparing = ['dset_root', 'dset_manip', 'datasets']
# #
# # print("\nBetween: {}, levels: {}\nitems: {}".format(between_level, n_roots, roots))
# # print("\nWithin: {}, levels: {}\nitems: {}".format(within_level, n_manip, manipulations))
# # print("\nInteractions: {}, levels: {}\nitems: {}".format(interactions, number_of_selected_datasets,
# #                                                          selected_dataset_new_names))
# #
# #
# # # # # Stats_dict
# # stats_dict = {"topic": topic_name, "use_dset_type": use_dset_type, "use_act_func": use_act_func,
# #               "min_hid_units": min_hid_units, "max_hid_units": max_hid_units, "min_gha_acc": min_gha_acc,
# #               "n_conds": df_rows,
# #               "DVs": DV,
# #               'interactions': {"label": interactions,
# #                                "levels": number_of_selected_datasets,
# #                                "variables": selected_dataset_new_names,
# #                                "old_names": selected_dataset_orig_names},
# #               'dset_roots': IV_datasets_df['dset_root'].unique(),
# #               'dset_manips': IV_datasets_df['dset_manip'].unique()
# #               }
# #
# # print("\n**** stats_dict ****")
# # for key, value in stats_dict.items():
# #     print("{0}: {1}".format(key, value))
# #
# #
# # print("\n********\nmake COI count plots\n********")
# 
# ax = sns.violinplot(x="dset_cond", y="max_combo", data=summary_df, cut=0,
#                     # order=['HW', 'MW', 'LW']
#                     order=["HBHW", "MBHW", "MBMW", "LBHW", "LBMW", "LBLW"]
#                     )
# # 'proportion of units with Bsel > 0'
# ax.set_title("Max combo_sel by between and within similarity")
# # ax.set_xticklabels(["Binary", "Cont-chanProp", "Cont-chanProp"])
# # ax.set_xticklabels(["ReLu", "Sigmoid"])
# # ax.set_xticklabels(['High (n=540)', 'Medium (n=360)', 'Low (n=180)'])
# 
# ax.set_xlabel("between and within similiarity")
# ax.set_ylabel("Max Combo selectivity")
# # plt.axvline(x=best_info_thr, linestyle="--", color='grey')
# ax.axhline(1.0, ls='--', color='grey')
# plt.show()
# 
# 
# #
# # this_dataframe = IV_datasets_df
# # # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# #
# # # # 1. Bowers sel
# # ax1 = sns.barplot(x="new_dset_names", y="%_units_w_sel", data=this_dataframe,
# #                   order=selected_dataset_new_names)  # , ax=axes[0, 0])
# # ax1.set_title("Mean percentage of units with selectivity > 0")
# # ax1.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # ax1.set_ylabel("% selective units")
# # # ax1.legend(loc='best', fancybox=True, framealpha=0.5)
# # plt.tight_layout()
# # plt.savefig("{}_mean_Bowers_sel.png".format(topic_name))
# # plt.close()
# #
# # # # 2. Zhou COI
# # ax2 = sns.barplot(x="new_dset_names", y="mean_partial_prop", data=this_dataframe)
# #                     # order=selected_dataset_orig_names)  # , , ax=axes[0, 1])
# # ax2.set_title("Mean Precision")
# # ax2.set_xticklabels(labels=selected_dataset_new_names, rotation=12)
# # ax2.set_ylabel("Precision")
# # # ax2.legend(loc='best', fancybox=True, framealpha=0.5)
# # plt.tight_layout()
# # plt.savefig("{}_mean_Zhou_sel.png".format(topic_name))
# # # plt.show()
# # plt.close()
# #
# # # # 3. MORCOS COI
# # ax3 = sns.barplot(x="new_dset_names", y="mean_morcos_sel_v_items", data=this_dataframe)
# #                     # order=selected_dataset_orig_names)  # , , ax=axes[1, 0])
# # ax3.set_title("Mean Class Conditional Mean Activation")
# # ax3.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # ax3.set_ylabel("CCMA")
# # # ax3.legend(loc='best', fancybox=True, framealpha=0.5)
# # plt.tight_layout()
# #
# # plt.savefig("{}_mean_Morcos_sel.png".format(topic_name))
# # plt.close()
# #
# # # # 4. ROC COI
# # ax4 = sns.barplot(x="new_dset_names", y="mean_max_AUC", data=this_dataframe)
# # # , order=selected_dataset_orig_names)  # , , ax=axes[1, 1])
# # ax4.set_title("Mean ROC AUC selectivity by dataset")
# # ax4.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # ax4.set_ylabel("ROC_AUC")
# # plt.tight_layout()
# #
# # plt.savefig("{}_mean_ROC_AUC_sel.png".format(topic_name))
# # plt.close()
# #
# # # # 5. lifetime sparsity
# # # ax4 = sns.barplot(x="new_dset_names", y="mean_below_five", data=this_dataframe)
# # # # , order=selected_dataset_orig_names)  # , , ax=axes[1, 1])
# # # ax4.set_title("mean percent of items active < .5 per unit")
# # # ax4.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # # ax4.set_ylabel("lifetime_sparsity")
# # # plt.savefig("{}_mean_lifetime_sparsity_sel.png".format(topic_name))
# # # plt.close()
# #
# # ax4 = sns.barplot(x="new_dset_names", y="mean_above_five", data=this_dataframe)
# # # , order=selected_dataset_orig_names)  # , , ax=axes[1, 1])
# # ax4.set_title("mean percent of items active > .5 per unit")
# # ax4.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # ax4.set_ylabel("lifetime_sparsity")
# # plt.tight_layout()
# #
# # plt.savefig("{}_mean_lifetime_sparsity_sel.png".format(topic_name))
# # plt.close()
# #
# #
# # # # 4. time to train
# # ax4 = sns.barplot(x="new_dset_names", y="trained_for", data=this_dataframe)
# # ax4.set_title("Mean time to train by dataset")
# # ax4.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # ax4.set_ylabel("epochs")
# # plt.tight_layout()
# #
# # plt.savefig("{}_mean_training.png".format(topic_name))
# # plt.close()
# #
# # # # 4. max acc
# # ax4 = sns.barplot(x="new_dset_names", y="gha_acc", data=this_dataframe)
# # ax4.set_title("Mean accuracy by dataset")
# # ax4.set_xticklabels(labels=selected_dataset_new_names, rotation=10)
# # ax4.set_ylabel("accuracy")
# # plt.tight_layout()
# #
# # plt.savefig("{}_mean_accuracy.png".format(topic_name))
# # plt.close()
# 
# print("script finished")

# # part 1 # # load summary csv - check conditions
topic_name = 'cos_sim2'
experiment_chapter = 'within_between_dist_july2020'
master_df_name = f"{topic_name}_sel_p_u_master"  # don't need to include the .csv bit
master_df_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
                 f'{experiment_chapter}/{topic_name}/{topic_name}_sel_p_u_master.csv'

# # part 2. decide on how to group the data (dependent variables)
# # decide whether to include simulations based on 'dset_type', 'hid_units', 'act_func', 'dataset', 'gha_acc',
use_dset_type = 'all'  # 'all', 'bin', 'cont', 'chanprop', 'chanDist'
use_act_func = 'all'  # 'all', 'sigm' 'ReLu'
min_hid_units = 10
max_hid_units = 500
min_gha_acc = .15
n_cats = 10

# # variables
IV = [
    # 'dset'
    'cos_sim',
    #   'act_func',
    # 'dtype',
    #   'between', 'within',
    # 'n_units'
    # 'output_class'
      ]
# DV = ["trained_for", "gha_acc",
#       'combo', "max_informed", "max_info_sens", "b_sel",
#       "means", 'sd', 'nz_prop', 'hi_val_prop']
# DV = ['combo', "max_informed", "max_info_sens", "b_sel",
#       "trained_for", "means", 'sd', 'nz_prop', 'hi_val_prop']
# DV = ["trained_for", "means", "max_informed", "b_sel"]
DV = [
# #     # "means",
# #     "means_c",
# #      "max_informed",
# #     "max_informed_c",
# #     # "b_sel",
# #     "b_sel_c"
    'nz_prop', 'hi_val_prop'
        ]


# # plot details
font_size = 10



master_df = pd.read_csv(master_df_path)
print(f"\ntopic name {topic_name}")
print(f"master_df.shape: {master_df.shape}")
print(master_df.head())

for index, header in enumerate(list(master_df)):
    print(index, header)
# print(list(master_df))




if (use_act_func == 'all') and (use_dset_type == 'all'):
    plots_dir = 'ALL_sel_p_u'
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

# # sanity check on layers
# get_values = master_df.dset.unique()
# for i in range(len(get_values)):
#     get_value_count = master_df.dset.value_counts()[get_values[i]]
#     print("{} value: {} {}".format(i, get_values[i], get_value_count))
# print(get_values)

if use_dset_type is 'all':
    print("use all dset_types")
else:
    print(f"use_dset_type: {use_dset_type}")
    master_df = master_df.loc[master_df['dtype'] == use_dset_type]
    print(f"master_df.shape: {master_df.shape}")

if use_act_func is 'all':
    print("use all act_funcs")
else:
    print(f"use_act_func: {use_act_func}")
    master_df = master_df.loc[master_df['act_func'] == use_act_func]
    print(f"master_df.shape: {master_df.shape}")


master_df = master_df[master_df['n_units'].between(min_hid_units, max_hid_units, inclusive=True)]
# print("select_hid_units.shape: {}".format(select_hid_units.shape))
master_df = master_df[master_df['gha_acc'] > min_gha_acc]
# print("master_df.shape: {}".format(master_df.shape))

filtered_df = master_df
print(f"\nmaster_df.shape: {master_df.shape}\n\n{master_df.head()}")


# IV_datasets_df.to_csv("{}IV_datasets_df.csv".format(topic_name))


counter = 0

for var_name in IV:
    for measure in DV:
        plot_type = 'violin'
        use_order = False
        counter += 1
        variable = var_name
        print(f"{counter}:\tIV: {var_name}\tDV: {measure}")

        if var_name is 'cos_sim':
            order = ["HBHW", "MBHW", "MBMW", "LBHW", "LBMW", "LBLW"]
            use_order = True
            by = 'between and within class similarity'

        elif var_name is 'dset':
            order = ["pro_sm_", "pro_med"]
            use_order = True
            by = 'Prototype difference'

        elif var_name is 'act_func':
            by = 'activation function'

        elif var_name is 'dtype':
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

        elif var_name is 'n_units':
            by = 'layer size'

        elif var_name is 'output_class':
            by = 'Frequency'
            variable = list(range(n_cats))
            # plot_type = 'countplot'

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

        # plt.figure(figsize=(3.5, 3))
        plt.figure(figsize=(4.08, 3.5))

        if plot_type == 'violin':
            if use_order:
                ax = sns.violinplot(x=variable, y=measure, data=master_df, cut=0,
                                    order=order)
            else:
                ax = sns.violinplot(x=variable, y=measure, data=master_df, cut=0)

        elif plot_type == 'countplot':
            ax = sns.countplot(x=measure, data=master_df)

        ax.set_title(f"{plot_title_use}: {m_name} by {by}", fontsize=font_size, wrap=True)

        if var_name is 'dtype':
            # ax.set_xticklabels(["Binary", "Cont-chanDist", "Cont-chanProp"])
            ax.set_xticklabels(["Binary", "Cont"], fontsize=font_size)

        elif var_name is 'act_func':
            ax.set_xticklabels(["ReLu", "Sigmoid"], fontsize=font_size)
        elif var_name in ['between', 'within']:
            ax.set_xticklabels(['High', 'Medium', 'Low'], fontsize=font_size)

        elif var_name is 'cos_sim':
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
        # plt.tight_layout()
        # plt.tight_layout(
        #     pad=0.4, w_pad=0.5, h_pad=1.0
        # )
        # plt.show()
        plot_name = f'{topic_name}_{plots_dir}_{measure}_by_{var_name}'
        plt.savefig(os.path.join(plots_path, plot_name))
        plt.close()



print("script finished")
