from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, cosine_distances
import numpy as np
import scipy
from itertools import combinations
import json
import pandas as pd
import os
import csv
from tools.data import running_on_laptop, switch_home_dirs


# # # toy data
# # a = np.random.choice([0, 1], size=(2, 10), p=[0.2, 0.8])
# # b = np.random.choice([0, 1], size=(2, 10), p=[0.4, 0.6])
# # c = np.random.choice([0, 1], size=(2, 10), p=[0.5, 0.5])
# # d = np.random.choice([0, 1], size=(2, 10), p=[0.6, 0.4])
# # e = np.random.choice([0, 1], size=(2, 10), p=[0.8, 0.2])
# # f = np.random.choice([0, 1], size=(2, 10), p=[0.9, 0.1])
# #
# # z = np.concatenate((a, b, c, d, e, f), axis=0)
#
#
# # file_name = "y_50_by_50_binary_dense_50.csv"
# # file_name = "HB_HW__cont_dataset1"
# # file_name = "MNISTmini_X"
#
# file_path = "/home/nm13850/Documents/PhD/python_v2/experiments/within_between_dist/datasets"
# os.chdir(file_path)
# print(f"os.getcwd(): {os.getcwd()}")
# file_name = "cont_dataset_LBLW1"
# file_type = ".csv"
# file = file_name + file_type
# load_file = np.loadtxt(file, delimiter=",")
#
#
# '''
# To turn this into a function I need...
# input data - as array.  not sure if/how wel this will work for colour images - need to be careful with indexing.
# n_cats
# IPC dict
#
# Not sure I can make it a function...
#
# '''
# # # enter either 'cos_sim, 'cos_dist' or 'taxi'
# # loaded_dataset = z
# loaded_dataset = load_file
# distance = 'cos_sim'
# n_cats = 50
# IPC_dict = None
# # IPC_dict = {0: 3, 1: 3, 2: 4, 3: 2}
# dataset_name = file_name
#
#
#
# dataset = np.asarray(loaded_dataset)
# items, features = np.shape(dataset)
# print(f'\ndataset: {dataset}')
# print(f'items, features: {items}, {features}')
#
# # add IPC dict here if class_sizes are not equal
# if IPC_dict is None:
#     cat_size = int(items/n_cats)
#     IPC_dict = {i: cat_size for i in range(n_cats)}
#     print(f'\nequal size IPC dict\n{IPC_dict}')
#
# # separate out the individual classes
# # start with class inidices list containing zero, index of the first class
# class_indices = [0]
# IPC_vals = list(IPC_dict.values())
# print(f'\nIPC_vals: {IPC_vals}')
# for i in range(n_cats):
#     next_val = class_indices[-1] + IPC_vals[i]
#     class_indices.append(next_val)
#
# #  list of items numbers to start each class
# start_indices = class_indices[:n_cats]
# # print(f'\nstart_indices: {start_indices}')
#
# # list of indices to end each class
# end_indices = class_indices[1:]
# # print(f'end_indices: {end_indices}')
#
# # 1. define classes as slices of dataset array
# class_list = []
# names_list = []
#
# for cat in range(n_cats):
#     this_name = f'class_{cat}'
#     names_list.append(this_name)
#
#     this_class = dataset[start_indices[cat]:end_indices[cat], :]
#     class_list.append(this_class)
#
#     # print(f'\n{this_name}\n{this_class}\n')
#
# # within class similarities
# # 3. make empty list to store results.
# within_list = []
#
# for index, this_cat in enumerate(class_list):
#     # print(f'\ngetting within class cos_sim for {names_list[index]}')
#
#     # will do all pairwise comparrisons within the given category
#     if distance in ['cos_sim', 'cosine_similarity', 'cosine_sim', 'cos_similarity']:
#         within_cat = cosine_similarity(this_cat)
#         # the SIMILARITY between two identical vectors will be 1
#     elif distance in ['cos_dist', 'cosine_distance', 'cosine_dist', 'cos_distance']:
#         within_cat = cosine_distances(this_cat)
#         # this DISTANCE between two identical vectors will be 0
#         # Cosine_distance = 1 - cosine_similarity
#     elif distance in ['manhattan', 'taxi']:
#         within_cat = manhattan_distances(this_cat)
#     else:
#         raise ValueError('must input a valid distance name')
#
#     # print(within_cat)
#
#     # just take the triangle since this analysis compares items with themselves
#     triangle_indices = np.triu_indices(IPC_dict[index], 1)
#     values_for_descriptives = (within_cat[triangle_indices])
#     # print(values_for_descriptives)
#
#     data_similarity_descriptives = scipy.stats.describe(values_for_descriptives, axis=None)
#     mean_sim = str(np.round(data_similarity_descriptives.mean, decimals=2))
#     print(f"\nBetween group mean {distance} for {names_list[index]}: {mean_sim}")
#
#     within_list.append(mean_sim)
#
# print(f'\nwithin_list ({distance}): {within_list}\n')
#
#
# # between class similarities.
# print('\nbetween class similarities')
# '''
# For each pair of classes
# - get the similarities of each item in one class to each item in the other class.
# - take the average of the whole matrix (not just the triangle) to get the
# mean similaritiy between these two classes.
#
# These mean between class similarities go into an n_cats x n_cats-1 matrix.
# (n_cats-1 because I am not going to have diagonals comparing classes with themselves.
# Each row shows a classes similarity to all other classes.
# - Take the average of each row to a get a class's mean between class similarity.
#
# Example below shows 4 classes (rows) and the values show which other class is being compared.
# e.g., class1 is compared with classes 2, 3, 4.  Class2 is compared with classes 1, 3, 4.
#         compA   compB   compC
# class1: 2       3       4
# class2: 1       3       4
# class3: 1       2       4
# class4: 1       2       3
# '''
#
# class_pairs_list = list(combinations(class_list, 2))
# class_names_list = list(combinations(names_list, 2))
# class_index_list = list(combinations(range(n_cats), 2))
# print(f'running {len(class_index_list)} between class comparrrions.\n{class_index_list}')
# between_array = np.zeros(shape=(n_cats, n_cats-1))
#
# for index, cat_pair in enumerate(class_pairs_list):
#     cat_a = cat_pair[0]
#     cat_name_a = class_names_list[index][0]
#
#     cat_b = cat_pair[1]
#     cat_name_b = class_names_list[index][1]
#
#     print(f'\nbetween class {distance} for: {cat_name_a} and {cat_name_b}')
#
#
#     # # do all pairwise comparrisons between the classes
#     if distance in ['cos_sim', 'cosine_similarity', 'cosine_sim', 'cos_similarity']:
#         between_pairs_matrix = cosine_similarity(X=cat_a, Y=cat_b)
#     elif distance in ['cos_dist', 'cosine_distance', 'cosine_dist', 'cos_distance']:
#         between_pairs_matrix = cosine_distances(X=cat_a, Y=cat_b)
#     elif distance in ['manhattan', 'taxi']:
#         between_pairs_matrix = manhattan_distances(X=cat_a, Y=cat_b)
#     else:
#         raise ValueError('must input a valid distance name')
#
#     print(f'{between_pairs_matrix}')
#     mean_between_pair = np.mean(between_pairs_matrix)
#     print(f'mean_between_pair: {mean_between_pair}')
#
#     # append to between array in both (ofset) diagonals
#     idxA, idxB = class_index_list[index]
#     print(f'add to matrix position: {idxA}, {idxB}')
#     between_array[idxA, idxB-1] = mean_between_pair
#     between_array[idxB, idxA] = mean_between_pair
#
# print(f"\nbetween_array:\n{between_array}")
#
# print(f'\nmean between class {distance}')
# between_list = []
# for index in range(n_cats):
#     this_row = between_array[index]
#     this_mean = np.mean(this_row)
#     between_list.append(this_mean)
#     print(index, this_mean)
#
# print("I want to get the mean of the between list and the within list")
# dset_between_mean = np.mean(between_list)
# dset_between_sd = np.std(between_list)
# print(f"dataset mean between class distance: {dset_between_mean} std.dev: {dset_between_sd}")
#
# print(f"check within list:\n{within_list}")
# within_list_num = [float(i) for i in within_list]
# print(f"check within_list_num:\n{within_list_num}")
#
# dset_within_mean = np.mean(within_list_num)
# dset_within_sd = np.std(within_list_num)
# print(f"dataset mean within class distance: {dset_within_mean} std.dev: {dset_within_sd}")
#
#
# # # save output.
# '''for each class:
#     mean within
#     mean between
#     paired between
# '''
# names_list.append('Dset_means')
# names_list.append('Dset_sd')
# within_list.append(dset_within_mean)
# within_list.append(dset_within_sd)
# between_list.append(dset_between_mean)
# between_list.append(dset_between_sd)
#
# class_sim_dict = {
#                   'class': names_list,
#                   'between': between_list,
#                   'within': within_list}
# class_sim_df = pd.DataFrame(class_sim_dict)
# print(class_sim_df)
# class_sim_df.to_csv(f'{dataset_name}_{distance}.csv', index_label='class', )

def get_cos_sim(dset, n_cats, dtype, dset_name, version, sim_type, IPC_dict=None):
    """
    This will take a dataset and calculate the cosine similiarity within and
    between classes, producing a csv with results and updating a main doc.

    :param dset: data to be tested, csv, (pd or np array?)
    :param n_cats: number of classes (items per-class calculated as items/classes)
    :param dtype: binary, chan_dist or chanProp.  only needed for labelling
    :param dset_name: of dataset eg HBHW, HBLW, LBHW, LBLW
    :param version: number with 2 versions of each type
    :param sim_type: Describe the similarity e.g., HBHW or vary etc
    :param IPC_dict: defalt = None.  if the number of items per class is not
                    equal, enter a dict


    """
    print("\nrunning ** get_cos_sim()**")

    file_path = "/home/nm13850/Documents/PhD/python_v2/experiments/" \
                "within_between_dist_july2020/New_data/"
    if running_on_laptop():
        file_path = '/Users/nickmartin/Library/Mobile Documents/com~apple~CloudDocs/' \
                    'Documents/PhD/python_v2/experiments/' \
                    'within_between_dist_july2020/New_data/'

    save_path = os.path.join(file_path, 'similarity_details')

    # # enter either 'cos_sim, 'cos_dist' or 'taxi'
    distance = 'cos_sim'

    dataset = np.asarray(dset)
    items, features = np.shape(dataset)
    print(f'\ndataset: {dataset}')
    print(f'items, features: {items}, {features}')

    # add IPC dict here if class_sizes are not equal
    if IPC_dict is None:
       cat_size = int(items / n_cats)
       IPC_dict = {i: cat_size for i in range(n_cats)}
       print(f'\nequal size IPC dict\n{IPC_dict}')
    else:
        print("using IPC dict")

    # separate out the individual classes
    # start with class inidices list containing zero, index of the first class
    class_indices = [0]
    IPC_vals = list(IPC_dict.values())
    print(f'\nIPC_vals: {IPC_vals}')
    for i in range(n_cats):
       next_val = class_indices[-1] + IPC_vals[i]
       class_indices.append(next_val)

    #  list of items numbers to start each class
    start_indices = class_indices[:n_cats]
    # print(f'\nstart_indices: {start_indices}')

    # list of indices to end each class
    end_indices = class_indices[1:]
    # print(f'end_indices: {end_indices}')

    # 1. define classes as slices of dataset array
    class_list = []
    names_list = []

    for cat in range(n_cats):
        this_name = f'class_{cat}'
        names_list.append(this_name)

        this_class = dataset[start_indices[cat]:end_indices[cat], :]
        class_list.append(this_class)

        # print(f'\n{this_name}\n{this_class}\n')

    # within class similarities
    # 3. make empty list to store results.
    within_list = []

    for index, this_cat in enumerate(class_list):
        # print(f'\ngetting within class cos_sim for {names_list[index]}')

        # will do all pairwise comparrisons within the given category
        if distance in ['cos_sim', 'cosine_similarity', 'cosine_sim', 'cos_similarity']:
            within_cat = cosine_similarity(this_cat)
            # the SIMILARITY between two identical vectors will be 1
        elif distance in ['cos_dist', 'cosine_distance', 'cosine_dist', 'cos_distance']:
            within_cat = cosine_distances(this_cat)
            # this DISTANCE between two identical vectors will be 0
            # Cosine_distance = 1 - cosine_similarity
        elif distance in ['manhattan', 'taxi']:
            within_cat = manhattan_distances(this_cat)
        else:
            raise ValueError('must input a valid distance name')

        # print(within_cat)

        # just take the triangle since this analysis compares items with themselves
        triangle_indices = np.triu_indices(IPC_dict[index], 1)
        values_for_descriptives = (within_cat[triangle_indices])
        # print(values_for_descriptives)

        data_similarity_descriptives = scipy.stats.describe(values_for_descriptives, axis=None)
        mean_sim = str(np.round(data_similarity_descriptives.mean, decimals=2))
        print(f"\nWithin group mean {distance} for {names_list[index]}: {mean_sim}")

        within_list.append(mean_sim)

    print(f'\nwithin_list ({distance}): {within_list}\n')

    # between class similarities.
    print('\nbetween class similarities')
    '''
    For each pair of classes
    - get the similarities of each item in one class to each item in the other class.
    - take the average of the whole matrix (not just the triangle) to get the 
    mean similaritiy between these two classes.
    
    These mean between class similarities go into an n_cats x n_cats-1 matrix.
    (n_cats-1 because I am not going to have diagonals comparing classes with themselves.  
    Each row shows a classes similarity to all other classes.
    - Take the average of each row to a get a class's mean between class similarity.
    
    Example below shows 4 classes (rows) and the values show which other class is being compared.
    e.g., class1 is compared with classes 2, 3, 4.  Class2 is compared with classes 1, 3, 4.
           compA   compB   compC
    class1: 2       3       4
    class2: 1       3       4
    class3: 1       2       4
    class4: 1       2       3
    '''

    class_pairs_list = list(combinations(class_list, 2))
    class_names_list = list(combinations(names_list, 2))
    class_index_list = list(combinations(range(n_cats), 2))
    print(f'running {len(class_index_list)} between class comparrrions.\n{class_index_list}')
    between_array = np.zeros(shape=(n_cats, n_cats - 1))

    for index, cat_pair in enumerate(class_pairs_list):
        cat_a = cat_pair[0]
        cat_name_a = class_names_list[index][0]

        cat_b = cat_pair[1]
        cat_name_b = class_names_list[index][1]

        print(f'\nbetween class {distance} for: {cat_name_a} and {cat_name_b}')

        # # do all pairwise comparrisons between the classes
        if distance in ['cos_sim', 'cosine_similarity', 'cosine_sim', 'cos_similarity']:
            between_pairs_matrix = cosine_similarity(X=cat_a, Y=cat_b)
        elif distance in ['cos_dist', 'cosine_distance', 'cosine_dist', 'cos_distance']:
            between_pairs_matrix = cosine_distances(X=cat_a, Y=cat_b)
        elif distance in ['manhattan', 'taxi']:
            between_pairs_matrix = manhattan_distances(X=cat_a, Y=cat_b)
        else:
            raise ValueError('must input a valid distance name')

        print(f'{between_pairs_matrix}')
        mean_between_pair = np.mean(between_pairs_matrix)
        print(f'mean_between_pair: {mean_between_pair}')

        # append to between array in both (ofset) diagonals
        idxA, idxB = class_index_list[index]
        print(f'add to matrix position: {idxA}, {idxB}')
        between_array[idxA, idxB - 1] = mean_between_pair
        between_array[idxB, idxA] = mean_between_pair

    print(f"\nbetween_array:\n{between_array}")

    print(f'\nmean between class {distance}')
    between_list = []
    for index in range(n_cats):
        this_row = between_array[index]
        this_mean = np.mean(this_row)
        between_list.append(this_mean)
        print(index, this_mean)

    print("I want to get the mean of the between list and the within list")
    dset_between_mean = np.mean(between_list)
    dset_between_sd = np.std(between_list)
    print(f"dataset mean between class distance: {dset_between_mean} std.dev: {dset_between_sd}")

    print(f"check within list:\n{within_list}")
    within_list_num = [float(i) for i in within_list]
    print(f"check within_list_num:\n{within_list_num}")

    dset_within_mean = np.mean(within_list_num)
    dset_within_sd = np.std(within_list_num)
    print(f"dataset mean within class distance: {dset_within_mean} std.dev: {dset_within_sd}")

    # # save output.
    '''for each class:
       mean within
       mean between
       paired between 
    '''
    names_list.append('Dset_means')
    names_list.append('Dset_sd')
    within_list.append(dset_within_mean)
    within_list.append(dset_within_sd)
    between_list.append(dset_between_mean)
    between_list.append(dset_between_sd)

    class_sim_dict = {
       'class': names_list,
       'between': between_list,
       'within': within_list}
    class_sim_df = pd.DataFrame(class_sim_dict)
    print(class_sim_df)
    csv_name = f'{dset_name}_{distance}.csv'
    csv_path = os.path.join(save_path, csv_name)
    class_sim_df.to_csv(csv_path, index_label='class', )

    # check if similiarity summary exists
    similarity_info = [dtype, dset_name, sim_type, version, n_cats,
                       dset_between_mean, dset_between_sd,
                       dset_within_mean, dset_within_sd]
    print(f"similarity_info:\n{similarity_info}")


    # check if training_info.csv exists
    summary_name = 'similarity_summary.csv'
    print(f"\nlooking for file:\n{os.path.join(save_path, summary_name)}")
    if not os.path.isfile(os.path.join(save_path, summary_name)):
        print("making summary page")
        headers = ["dtype", "dset_name", 'sim_type', "version", "n_cats",
                   "mean_b", "sd_b", "mean_w", "sd_w"]

        similarity_overview = open(os.path.join(save_path, summary_name), 'w')
        mywriter = csv.writer(similarity_overview)
        mywriter.writerow(headers)
    else:
        print("appending to summary page")
        similarity_overview = open(os.path.join(save_path, summary_name), 'a')
        mywriter = csv.writer(similarity_overview)

    mywriter.writerow(similarity_info)
    similarity_overview.close()

    return_dict = {"dtype": dtype,
                   "dset_name": dset_name,
                   'sim_type': sim_type,
                   "version": version,
                   "n_cats": n_cats,
                   "dset_between_mean": dset_between_mean,
                   "dset_between_sd": dset_between_sd,
                   "dset_within_mean": dset_within_mean,
                   "dset_within_sd": dset_within_sd
                   }

    return return_dict

    # print(f"similiarity summary:\n{similarity_overview}")

#####################################

print("\n\nthere is stuff at the bottom of the page")
datasets_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
                  'within_between_dist_july2020/New_data/datasets'
counter = 0
for index, file_name in enumerate(os.listdir(datasets_path)):
    if 'load_dict' in file_name:
        continue
    elif 'labels_10cats_50IPC' in file_name:
        continue
    else:
        counter += 1
        print(counter, file_name)

#
#
# dset_names = ["bin_b82_w87_HBHW_v2.csv",
#              "bin_b84_w87_HBHW_v1.csv",
#              "bin_b74_w85_MBHW_v1.csv",
#              "bin_b74_w85_MBHW_v2.csv",
#              "bin_b71_w76_MBMW_v1.csv",
#              "bin_b72_w75_MBMW_v2.csv",
#              "bin_b50_w87_LBHW_v1.csv",
#              "bin_b50_w87_LBHW_v2.csv",
#              "bin_b50_w76_LBMW_v1.csv",
#              "bin_b50_w76_LBMW_v2.csv",
#              "bin_b50_w54_LBLW_v1.csv",
#              "bin_b50_w55_LBLW_v2.csv",
#              "chanProp_b84_w86_HBHW_v1.csv",
#              "chanProp_b84_w87_HBHW_v2.csv",
#              "chanProp_b83_w86_HBHW_v1.csv",
#              "chanProp_b74_w87_MBHW_v1.csv",
#              "chanProp_b73_w87_MBHW_v2.csv",
#              "chanProp_b67_w76_MBMW_v1.csv",
#              "chanProp_b66_w76_MBMW_v2.csv",
#              "chanProp_b48_w86_LBHW_v1.csv",
#              "chanProp_b49_w87_LBHW_v2.csv",
#              "chanProp_b48_w75_LBMW_v1.csv",
#              "chanProp_b46_w75_LBMW_v2.csv",
#              "chanProp_b42_w51_LBLW_v1.csv",
#              "chanProp_b42_w51_LBLW_v2.csv",
#              "chanDist_b83_w86_HBHW_v1.csv",
#              "chanDist_b83_w86_HBHW_v2.csv",
#              "chanDist_b74_w86_MBHW_v1.csv",
#              "chanDist_b74_w86_MBHW_v2.csv",
#              "chanDist_b66_w74_MBMW_v1.csv",
#              "chanDist_b65_w73_MBMW_v2.csv",
#              "chanDist_b51_w86_LBHW_v1.csv",
#              "chanDist_b51_w86_LBHW_v2.csv",
#              "chanDist_b50_w73_LBMW_v1.csv",
#              "chanDist_b50_w73_LBMW_v2.csv",
#              "chanDist_b45_w54_LBLW_v1.csv",
#              "chanDist_b45_w54_LBLW_v2.csv",
#              "bin_pro_sm_v1_r_vary_max.csv",
#              "bin_pro_sm_v2_r_vary_max.csv",
#              "bin_pro_sm_v3_r_vary_max.csv",
#              "cont_pro_sm_v1_r_vary_max.csv",
#              "cont_pro_sm_v2_r_vary_max.csv",
#              "cont_pro_sm_v3_r_vary_max.csv",
#              "bin_pro_med_v1_r_vary_max.csv",
#              "bin_pro_med_v2_r_vary_max.csv",
#              "bin_pro_med_v3_r_vary_max.csv",
#              "cont_pro_med_v1_r_vary_max.csv",
#              "cont_pro_med_v2_r_vary_max.csv",
#              "cont_pro_med_v3_r_vary_max.csv",
#              ]
#
# for file_name in dset_names:


    dset_name = file_name[:-4]
    print(dset_name)

    load_path = os.path.join(datasets_path, file_name)
    load_file = np.loadtxt(load_path, delimiter=",")
    dataset = np.asarray(load_file)

    print(np.shape(dataset))

    dtype = 'bin'
    if 'cont' in dset_name:
        dtype = 'cont'
    elif 'chanProp' in dset_name:
        dtype = 'cont'
    elif 'chanDist' in dset_name:
        dtype = 'cont'

    if 'v1' in dset_name:
        version = 'v1'
    elif 'v2' in dset_name:
        version = 'v2'
    elif 'v3' in dset_name:
        version = 'v3'
    else:
        raise ValueError("version unknown")

    if 'HBHW' in dset_name:
        dset_sims = 'HBHW'
    elif 'MBHW' in dset_name:
        dset_sims = 'MBHW'
    elif 'MBMW' in dset_name:
        dset_sims = 'MBMW'
    elif 'LBHW' in dset_name:
        dset_sims = 'LBHW'
    elif 'LBMW' in dset_name:
        dset_sims = 'LBMW'
    elif 'LBLW' in dset_name:
        dset_sims = 'LBLW'
    elif 'pro_sm' in dset_name:
        dset_sims = 'pro_sm_vary'
    elif 'pro_med' in dset_name:
        dset_sims = 'pro_med_vary'
    else:
        raise ValueError('what dataset similarity type is this')

    print(f"{dset_name}\t{dtype}\t{version}\t{dset_sims}")

    get_cos_sim(dset=dataset, n_cats=10, dtype=dtype,
                dset_name=dset_name, version=version, sim_type=dset_sims)
    print("finihsed :)")
#
