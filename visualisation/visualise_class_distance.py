import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os.path

"""
This script will take in a dataset or list of datasets.

Load distance details csv (with between and within mean per class.


Plot dataset mean or class means onto plot with between and within on x and y axis

"""


# dset_dist_details_path = '/home/nm13850/Documents/PhD/python_v2/experiments/within_between_dist_july2020/New_data/similarity_details/cont_pro_med_v3_r_vary_max_cos_sim.csv'
dist_details_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
                    'within_between_dist_july2020/New_data/similarity_details/'

# dset_path = '/home/nm13850/Documents/PhD/python_v2/experiments/within_between_dist_july2020/New_data/datasets'
# dset_name = 'cont_pro_med_v3_r_vary_max_cos_sim'
# '.csv'
# for index, filename in enumerate(os.listdir(dist_details_path)):
#     if 'load_dict' in filename:
    # print(filename)

n_cats = 10

print_this = 'classes'  # 'means', 'classes'

filename_list = [
    # "bin_b84_w87_HBHW_v1_cos_sim.csv",
    # "bin_b82_w87_HBHW_v2_cos_sim.csv",
    # "bin_b74_w85_MBHW_v1_cos_sim.csv",
    # "bin_b74_w85_MBHW_v2_cos_sim.csv",
    # "bin_b71_w76_MBMW_v1_cos_sim.csv",
    # "bin_b72_w75_MBMW_v2_cos_sim.csv",
    # "bin_b50_w87_LBHW_v1_cos_sim.csv",
    # "bin_b50_w87_LBHW_v2_cos_sim.csv",
    # "bin_b50_w76_LBMW_v1_cos_sim.csv",
    # "bin_b50_w76_LBMW_v2_cos_sim.csv",
    # "bin_b50_w54_LBLW_v1_cos_sim.csv",
    # "bin_b50_w55_LBLW_v2_cos_sim.csv",
    #
    # "chanDist_b83_w86_HBHW_v1_cos_sim.csv",
    # "chanDist_b83_w86_HBHW_v2_cos_sim.csv",
    # "chanDist_b74_w86_MBHW_v1_cos_sim.csv",
    # "chanDist_b74_w86_MBHW_v2_cos_sim.csv",
    # "chanDist_b66_w74_MBMW_v1_cos_sim.csv",
    # "chanDist_b65_w73_MBMW_v2_cos_sim.csv",
    # "chanDist_b51_w86_LBHW_v1_cos_sim.csv",
    # "chanDist_b51_w86_LBHW_v2_cos_sim.csv",
    # "chanDist_b50_w73_LBMW_v1_cos_sim.csv",
    # "chanDist_b50_w73_LBMW_v2_cos_sim.csv",
    # "chanDist_b45_w54_LBLW_v1_cos_sim.csv",
    # "chanDist_b45_w54_LBLW_v2_cos_sim.csv",
    # "chanProp_b84_w86_HBHW_v1_cos_sim.csv",
    # "chanProp_b84_w87_HBHW_v2_cos_sim.csv",
    # "chanProp_b74_w87_MBHW_v1_cos_sim.csv",
    # "chanProp_b73_w87_MBHW_v2_cos_sim.csv",
    # "chanProp_b67_w76_MBMW_v1_cos_sim.csv",
    # "chanProp_b66_w76_MBMW_v2_cos_sim.csv",
    # "chanProp_b48_w86_LBHW_v1_cos_sim.csv",
    # "chanProp_b49_w87_LBHW_v2_cos_sim.csv",
    # "chanProp_b48_w75_LBMW_v1_cos_sim.csv",
    # "chanProp_b46_w75_LBMW_v2_cos_sim.csv",
    # "chanProp_b42_w51_LBLW_v1_cos_sim.csv",
    # "chanProp_b42_w51_LBLW_v2_cos_sim.csv",



    "bin_pro_sm_v1_r_vary_max_cos_sim.csv",
    "bin_pro_sm_v2_r_vary_max_cos_sim.csv",
    "bin_pro_sm_v3_r_vary_max_cos_sim.csv",
    "bin_pro_med_v1_r_vary_max_cos_sim.csv",
    "bin_pro_med_v2_r_vary_max_cos_sim.csv",
    "bin_pro_med_v3_r_vary_max_cos_sim.csv",

    "cont_pro_sm_v1_r_vary_max_cos_sim.csv",
    "cont_pro_sm_v2_r_vary_max_cos_sim.csv",
    "cont_pro_sm_v3_r_vary_max_cos_sim.csv",
    "cont_pro_med_v1_r_vary_max_cos_sim.csv",
    "cont_pro_med_v2_r_vary_max_cos_sim.csv",
    "cont_pro_med_v3_r_vary_max_cos_sim.csv"
    ]


# plt.figure()
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set(xlim=(0.35, 1))
ax1.set(ylim=(0.35, 1))


data_for_plots = []


counter = 0
for filename in filename_list:
    counter += 1
    print(filename)

    dtype = 'cont'
    if 'bin' in filename:
        dtype = 'bin'

    version = 'V1'
    if 'V2' in filename:
        version = 'V2'
    elif 'v2' in filename:
        version = 'V2'
    elif 'V3' in filename:
        version = 'V3'
    elif 'v3' in filename:
        version = 'V3'

    data_set = 'HBHW'
    if 'MBHW' in filename:
        data_set = 'MBHW'
    elif 'MBMW' in filename:
        data_set = 'MBMW'
    elif 'LBHW' in filename:
        data_set = 'LBHW'
    elif 'LBMW' in filename:
        data_set = 'LBMW'
    elif 'LBLW' in filename:
        data_set = 'LBLW'
    elif 'pro_sm' in filename:
        data_set = 'pro_sm'
    elif 'pro_med' in filename:
        data_set = 'pro_med'

    # load distance document
    dist_df_path = os.path.join(dist_details_path, filename)
    dist_df = pd.read_csv(dist_df_path)

    print(dist_df.head())

    if print_this is 'means':
        between = dist_df.iloc[10]['between']
        within = dist_df.iloc[10]['within']

        dataset_details = [filename, dtype, data_set, version, between, within]
        data_for_plots.append(dataset_details)
        # print(between, within)
        #
        # ax = sns.scatterplot(x=between, y=within,
        #              # hue="year", size="mass",
        #              # palette=cmap, sizes=(10, 200),
        #              # data=planets
        #                      )
        # plt.show()

    elif print_this is 'classes':
        print('classes')

        for cat in range(n_cats):
            between = dist_df.iloc[cat]['between']
            within = dist_df.iloc[cat]['within']
            dataset_details = [filename, dtype, data_set, version, cat, between, within]
            data_for_plots.append(dataset_details)
        #
        # print(f"\ncheck df\nP{dist_df}")
        dist_df = dist_df.drop([10, 11])
        # print(f"\ncheck df\nP{dist_df}")

        ax1 = sns.lineplot(x='between', y='within',
                           color='gainsboro',
                           # style='dtype',
                           # hue='dtype',
                           data=dist_df, dashes=True)

        # plt.show()

    else:
        print("do you want dataset means or class values?")




if print_this is 'means':
    headers = ["filename", "dtype", "data_set", "version", "Between", "Within"]
    master_df = pd.DataFrame(data=data_for_plots, columns=headers)
    print(master_df.head())

    ax = sns.scatterplot(x="Between", y="Within",
                         hue="data_set",
                         s=100,
                         style='dtype',

                         # palette=cmap, sizes=(10, 200),
                         data=master_df)
    ax.set_xlim([.4, 1.0])
    ax.set_ylim([.4, 1.0])

    ax.set_title('Between and Within-class cosine similarity per dataset')
    # , fontsize=font_size, wrap=True)

    plt.legend(loc='lower right')

elif print_this is 'classes':
    print('classes')
    headers = ["filename", "dtype", "data_set", "version", "class", "Between", "Within"]
    master_df = pd.DataFrame(data=data_for_plots, columns=headers)
    print(master_df.head())

    ax1 = sns.scatterplot(x="Between", y="Within",
                         hue="data_set",
                         s=100,
                         style='dtype',

                         # palette=cmap, sizes=(10, 200),
                         data=master_df)
    # ax.set_xlim([.35, 1.0])
    # ax.set_ylim([.35, 1.0])

    # ax.set_title('Between and Within-class cosine similarity per dataset')
    # , fontsize=font_size, wrap=True)

    plt.legend(loc='lower right')

    ax1.set(xlim=(0.35, 1))
    ax1.set(ylim=(0.35, 1))
    ax1.set_title(f"Sim2: Dataset between and within class similarity")
    ax1.set_xlabel("Between class cosine-similarity")
    ax1.set_ylabel("Within class cosine-similarity")

plt.show()
