import csv
import os.path
import sys
import numpy as np
import pandas as pd

# from tools.dicts import load_dict


def nick_to_csv(df, path):
    """
    :param df: dataframe
    :param path: save path
    :return: nothing
    """

    '''https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv'''
    # Prepend dtypes to the top of df
    df2 = df.copy()
    if 'unit' in list(df2):
        df2 = df2.astype({'unit': int})
    if 'class' in list(df2):
        df2 = df2.astype({'class': int})
    df2.loc[-1] = df2.dtypes
    df2.index = df2.index + 1
    df2.sort_index(inplace=True)
    # Then save it to a csv
    df2.to_csv(path, index=False)

def nick_read_csv(path):
    '''https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv'''

    # # Nicks version 14th june 2019
    # Read types first line of csv
    dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    try:
        # Read the rest of the lines with the types from above
        open_csv = pd.read_csv(path, dtype=dtypes, skiprows=[1], index_col=0)
    except:
        # # if there are not dtypes saved in csv
        # # load without headers
        open_csv = pd.read_csv(path, header=None)

        # #check to see if first row contains headers
        header_row = list(open_csv.iloc[0])

        # # if pandas had added an idex, the first cell might be empty
        if np.isnan(header_row[0]):
            header_row[0] = 'index'

        if all(isinstance(element, str) for element in header_row):
            # first row is all strings
            open_csv = pd.read_csv(path, header=0, index_col=0)

        elif list(header_row) == list(range(header_row.size)):
            # zero indexed headers
            open_csv = pd.read_csv(path, header=0, index_col=0)

        elif list(header_row) == list(range(1, header_row.size + 1)):
            # one indexed headers
            open_csv = pd.read_csv(path, header=0, index_col=0)

    return open_csv


    # # simple version (Aaron N. Brock)
    # # Read types first line of csv
    # dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    # # Read the rest of the lines with the types from above
    # return pd.read_csv(path, dtype=dtypes, skiprows=[1])

    # modification cincin21
    # # # Read types first line of csv
    # dtypes = {key:value for (key,value) in pd.read_csv(path,
    #           nrows=1).iloc[0].to_dict().items() if 'date' not in value}
    #
    # parse_dates = [key for (key,value) in pd.read_csv(path,
    #                nrows=1).iloc[0].to_dict().items() if 'date' in value]
    # # Read the rest of the lines with the types from above
    # return pd.read_csv(path, dtype=dtypes, parse_dates=parse_dates, skiprows=[1])


def load_x_data(x_filename):
    """will open the xdata from npy or csv format and return a numpy.ndarray"""
    print("\n**** load_x_data() ****")

    if x_filename[-3:] == 'csv':
        x_data = np.loadtxt(x_filename, delimiter=",")
        print("loaded X as csv: {} {}".format(x_filename, np.shape(x_data)))
    elif x_filename[-3:] == 'npy':
        x_data = np.load(x_filename)
        print("loaded X as npy: {} {}".format(x_filename, np.shape(x_data)))
    else:
        print("unknown X_data file type: {}".format(x_filename[-3:]))
    return x_data



def load_y_data(y_label_path):
    """will open the y_data from npy or csv format and return:
        1. a pd.dataframe of [item, class] or [item, class, filename, class_name]
        2. a list of y class labels"""

    print("\n**** load_y_data() ****")

    if y_label_path[-3:] == 'csv':
        with open(y_label_path) as csv_file:
            y_array = list(csv.reader(csv_file, delimiter=','))
            items, columns = np.shape(y_array)
            if columns == 2:
                item_and_class_df = pd.DataFrame(data=y_array, columns=['item', 'class'])
                item_and_class_df.astype('int32').dtypes
            elif columns == 4:
                item_and_class_df = pd.DataFrame(data=y_array, columns=['item', 'class', 'filename', 'class_name'])
                item_and_class_df.astype({'item': 'int32',
                                          'class': 'int32',
                                          'filename': 'string',
                                          'class_name': 'string', }).dtypes

            else:
                print("unknown columns in y_array")

            # # csv is a TEXT file format so all values are strings!
            item_and_class_df['class'] = [int(i) for i in item_and_class_df['class'].tolist()]
            item_and_class_df['item'] = [int(i) for i in item_and_class_df['item'].tolist()]

        print("loaded y_label_path from csv: {} {}".format(y_label_path, item_and_class_df.shape))

    elif y_label_path[-3:] == 'npy':
        y_array = np.load(y_label_path)
        items, columns = np.shape(y_array)
        if columns == 2:
            item_and_class_df = pd.DataFrame({'item': y_array[:, 0], 'class': y_array[:, 1]})
        elif columns == 4:
            item_and_class_df = pd.DataFrame({'item': y_array[:, 0], 'class': y_array[:, 1],
                                              'filename': y_array[:, 2], 'class_name': y_array[:, 3]})
        else:
            print("unknown columns in y_array")
        print("loaded y_label_path as npy: {} {}".format(y_label_path, item_and_class_df.shape))

    else:
        print("unknown y_label_path file type: {}".format(y_label_path[-3:]))

    y_label_list = [int(i) for i in item_and_class_df['class'].tolist()]
    # item_and_class_df['class'] = y_label_list
    return item_and_class_df, y_label_list


# def load_data(data_dict, x_y, test_train_val='train_set'):
#     """
#     load x or y data from data_dict.  Can specify train, test, val
#     If y returns a dict with {"y_df": y_df, 'y_label_list': y_label_list}
#     :param data_dict: str(dict_name) or dict
#     :param x_y: str specific X/x or Y/y
#     :param test_train_val: str 'train_set', 'test_set' or 'val_set'
#     :return: dataset
#     """
#
#     print("\n****load_data()****")
#
#     if type(data_dict) is str:
#         data_dict = load_dict(data_dict)
#
#     if x_y in ['x', 'X']:
#         get_dataset = 'X_data'
#     elif x_y in ['y', 'Y']:
#         get_dataset = 'Y_data'
#     else:
#         print("ERROR x_y should be either 'x'/'X' or 'y'/'Y'")
#
#     #  find main dataset
#     # could be only data avaliable(data_dict), train(data_dict['train_set']),
#     # could be test(data_dict['test_set']) or val(data_dict['val_set'])
#     data_info_location = data_dict
#     if get_dataset in data_dict.keys():  # is it in first set of keys?
#         pass
#         # # get data in first set of keys
#     elif 'data_info' in data_dict.keys():  # data_dict['data_info']
#         if get_dataset in data_dict['data_info'].keys():
#             data_info_location = data_dict['data_info']
#             # # get_data in second set of keys (nested in data_info
#         elif test_train_val in data_dict['data_info'].keys():  # data_dict['data_info']['test_train_val']
#             if get_dataset in data_dict['data_info'][test_train_val].keys():
#                 data_info_location = data_dict['data_info'][test_train_val]
#                 # # get_data in 3rd set of keys nested in 'data_info' and 'train_test_val
#             else:
#                 print("not found ln96 get_dataset in data_dict['data_info'][test_train_val].keys():")
#
#         else:
#             print("not found ln99 test_train_val in data_dict['data_dict'].keys():")
#
#     elif test_train_val in data_dict.keys():  # data_dict['test_train_val']
#         data_info_location = data_dict[test_train_val]
#         # # get_data in second set of keys (nested in test_train_val)
#
#     else:
#         ValueError("\ntraining data not found")
#
#
#     dset_filename = data_info_location[get_dataset]
#
#     # # if X
#     if get_dataset is 'X_data':
#         if dset_filename[-3:] == 'csv':
#             dataset = np.loadtxt(dset_filename, delimiter=",")
#             print("loaded X as csv: {} {}".format(dset_filename, np.shape(dataset)))
#         elif dset_filename[-3:] == 'npy':
#             dataset = np.load(dset_filename)
#             print("loaded X as npy: {} {}".format(dset_filename, np.shape(dataset)))
#         else:
#             print("unknown X_data file type: {}".format(dset_filename[-3:]))
#
#     # # if y
#     if get_dataset is 'Y_data':
#         print("dset_filename: {}".format(dset_filename))
#
#         # print("load_y_data returns a dict with y_df and y_label_list")
#         # if dset_filename == 'use_to_categorical':
#         #     print("ERROR: {}".format(dset_filename))
#         dset_filename = data_info_location['Y_labels']
#         #     print("FIXED: {}".format(dset_filename))
#
#         if dset_filename[-3:] == 'csv':
#             item_and_class_df = pd.read_csv(dset_filename, header=None, names=["item", "class"])
#             y_label_list = item_and_class_df['class'].tolist()
#             dataset = {"y_df": item_and_class_df, 'y_label_list': y_label_list}
#             print("loaded y_filename as csv: {} {}".format(dset_filename, item_and_class_df.shape))
#         elif dset_filename[-3:] == 'npy':
#             y_labels_items = np.load(dset_filename)
#             item_and_class_df = pd.DataFrame({'item': y_labels_items[:, 0], 'class': y_labels_items[:, 1]})
#             y_label_list = item_and_class_df['class'].tolist()
#             dataset = {"y_df": item_and_class_df, 'y_label_list': y_label_list}
#             print("loaded y_filename as npy: {} {}".format(dset_filename, item_and_class_df.shape))
#         else:
#             print("unknown y_filename file type: {}".format(dset_filename[-3:]))
#
#     return dataset


def load_data_no_dict(dataset_name):
    """
    load x or y data from filename (data has no dict).
    If y returns a dict with {"y_df": y_df, 'y_label_list': y_label_list}
    :param dataset_name: str(name)
    :param x_y: str specific X/x or Y/y
    :param test_train_val: str 'train_set', 'test_set' or 'val_set'
    :return: dataset
    """

    if dataset_name[-3:] == 'csv':
        dataset = np.loadtxt(dataset_name, delimiter=",")
        print("loaded dataset as csv: {} {}".format(dataset_name, np.shape(dataset)))
    elif os.path.isfile("{}.csv".format(dataset_name)):
        dataset = np.loadtxt("{}.csv".format(dataset_name), delimiter=",")
        print("loaded dataset as csv: {} {}".format(dataset_name, np.shape(dataset)))

    elif dataset_name[-3:] == 'npy':
        dataset = np.load(dataset_name)
        print("loaded dataset as npy: {} {}".format(dataset_name, np.shape(dataset)))
    elif os.path.isfile("{}.npy".format(dataset_name)):
        dataset = np.load("{}.npy".format(dataset_name))
        print("loaded dataset as npy: {} {}".format(dataset_name, np.shape(dataset)))
    else:
        print("unknown dataset file type: {}".format(dataset_name[-3:]))

    return dataset



def load_hid_acts(hid_act_filename):
    """
    :param hid_act_filename: string
    :return: hid_act_df
    """
    print("\n**** load_hid_acts() ****")
    hid_act_df = None
    if hid_act_filename[-3:] == 'npy':
        hid_act_np = np.load(hid_act_filename)
        hid_act_df = pd.DataFrame(hid_act_np)
    elif hid_act_filename[-3:] == 'csv':
        hid_act_df = pd.read_csv(hid_act_filename, header=None)
    else:
        try:
            hid_act_np = np.load("{}.npy".format(hid_act_filename))
            hid_act_df = pd.DataFrame(hid_act_np)
        except FileNotFoundError:
            try:
                hid_act_df = pd.read_csv("{}.csv".format(hid_act_filename))
            except FileNotFoundError:
                print("hidden_activation_file not found")

    return hid_act_df


def sort_cycle_duplicates_list(list_to_sort, verbose=False):
    """this function takes a list of values and sorts them ascending but looping through instances of each value.
    the list [3, 1, 2, 1, 3, 2, 3] becomes [1, 2, 3, 1, 2, 3, 3].
    """
    sorted_sample = []
    sample_values = list(set(list_to_sort))

    if verbose is True:
        print("values in list_to_sort: {}".format(sample_values))

    counter = 0
    while len(list_to_sort) >= 1:
        for value in sample_values:
            if value in list_to_sort:
                np.random.random_sample.remove(value)
                sorted_sample.append(value)
                # print(random_sample, sorted_sample)
                if verbose is True:
                    print("\n{}. value: {}\n{}\n{}".format(counter, value, sorted_sample, list_to_sort))
                    counter = counter + 1

    # print(random_sample)
    print("sorted_sample: {}".format(sorted_sample))
    return sorted_sample



def sort_cycle_duplicates_df(df_to_sort, duplicated_value, sort_1st, sort_2nd, verbose=False):
    """this function will take a dataframe containing a now with duplicate values -
    (e.g., zero activations from hid_act).
    Data will be sorted by sort_1st variable (e.g., activations).
    Then rows with duplicate values in sort_1st column (e.g., zeros)
    will by cycled through by the sort_2nd column (e.g., class_label so that if they were first in the order
    [3, 1, 2, 1, 3, 2, 3] becomes [1, 2, 3, 1, 2, 3, 3]"""

    if verbose is True:
        print("df_to_sort\n{}".format(df_to_sort))

    # # first split dataframe into duplicates and other values in sort_1st col
    duplicates_to_sort = pd.DataFrame(data=df_to_sort[df_to_sort.loc[:, sort_1st] == duplicated_value],
                                      columns=df_to_sort.columns)
    values_to_sort = pd.DataFrame(data=df_to_sort[df_to_sort.loc[:, sort_1st] != duplicated_value])

    # # non-duplicate values can easily be sorted
    srtd_values_df = values_to_sort.sort_values(by=sort_1st, ascending=False)

    # # now sort duplicates...
    # # empty df to append duplicates into into
    cycled_duplicates = pd.DataFrame(columns=df_to_sort.columns)

    # # list of unique values from sort_2nd - e.g., what classes are present at zero
    sample_values = list(set(duplicates_to_sort.loc[:, sort_2nd]))
    if verbose is True:
        print("\nsample_values\n{}".format(sample_values))

    duplicate_rows, df_cols = duplicates_to_sort.shape

    while duplicate_rows > 0:  # until we have removed all rows
        for value in sample_values:
            remaining_values_list = list(duplicates_to_sort.loc[:, sort_2nd])
            if value in remaining_values_list:

                # # find where this value remains in the list
                row_idx = remaining_values_list.index(value)

                # # move row to sorted array
                row_to_copy = duplicates_to_sort.iloc[row_idx]
                cycled_duplicates = cycled_duplicates.append(row_to_copy)

                # # remove that row from original array
                duplicates_to_sort.drop(duplicates_to_sort.index[row_idx], inplace=True)
                duplicate_rows, df_cols = duplicates_to_sort.shape
                #
                if verbose is True:
                    counter = 0
                    print("\nvalue: {}"
                          "\nunsorted: {}\nsorted: {}".format(value, remaining_values_list,
                                                              np.round(list(cycled_duplicates.loc[:, sort_2nd]))))

    sorted_cycled_df = pd.concat([srtd_values_df, cycled_duplicates]).reset_index(drop=True)
    if verbose is True:
        print("\nsorted_cycled_df\n{}".format(sorted_cycled_df))

    return sorted_cycled_df


def get_dset_path(cond_name):
    """
    if I don't have the dataset path in my dict, just look it up here.
    It will search a string for the name of one of the commonly used datasets

    :param cond_name: string containing the name of dataset
    :return: path to that dataset (from dsets folder onwards)
    """

    dset_path = None
    dataset_path_dict = {'cifar': 'objects/CIFAR_10/CIFAR_10_2019',
                         'iris': 'other_classification/iris/orig/iris',
                         'mnist': 'digits/MNIST_June18/orig_data/mnist'}
    dsets = dataset_path_dict.keys()

    if any(dset in cond_name.lower() for dset in dsets):
        for dset in dsets:
            if dset in cond_name.lower():
                dset_path = dataset_path_dict[dset]
                print("\nget_dset_path() found {}".format(dset))
    else:
        print("\n\nERROR! DATASET NO FOUND\nLooking for datasets: {}\nin: {}".format(dsets, cond_name))

    return dset_path




def find_path_to_dir(long_path, target_dir, recursion_path=None):
    """
    When I want to save something (e.g., a summary.csv) in a parent folder,
    but I am not sure how many levels up it is, use this to find the path.

    If it can not find the target_dir in the first pass (long_path), it will
    use 'recursion_path' on subsequent loops, allowing the original 'long_path'
    to be returned in the error message.

    :param long_path: path of a child, grandchild, great-grandchild folder for target_dir.
    :param target_dir: the name of the  parent/grandparent/great-grandparent of longpath
    :param recursion_path: Default: None.  The path to check in all recursive loops.
        Do not enter anything for this variable when calling the function.

    :return: path to the target_directory.
    """

    prefix, target_dir = os.path.split(target_dir)
    if prefix == '':
        print(f"prefix: {prefix}, target_dir: {target_dir}")
    elif target_dir == '':
        prefix,  target_dir = target_dir, prefix
        print(f"prefix: {prefix}, target_dir: {target_dir}")
    else:
        print(f"neither empty\n"
              f"prefix: {prefix}, target_dir: {target_dir}")




    # # on first pass check the long-path.
    if recursion_path is None:
        tail, head = os.path.split(long_path)

    # # on subsequent loops use the recursion path
    else:
        tail, head = os.path.split(recursion_path)

    # # if target_dir has been found return path to target_dir
    if head == target_dir:
        if recursion_path is None:

            return long_path
        else:

            return recursion_path

    # # if the path can not be split. raise an error
    if head is '':
        raise ValueError(f"target_dir {target_dir} not found in {long_path}")

    else:
        # print(f"checking path:  {tail}\t\thead: {head}")

        return find_path_to_dir(long_path, target_dir, tail)

    # # # on first pass check the long-path.
    # if recursion_path is None:
    #     tail, head = os.path.split(long_path)
    #
    # # # on subsequent loops use the recursion path
    # else:
    #     tail, head = os.path.split(recursion_path)
    #
    # # # if target_dir has been found return path to target_dir
    # if head == target_dir:
    #     if recursion_path is None:
    #         return long_path
    #     else:
    #         return recursion_path
    #
    # # # if the path can not be split. raise an error
    # if head is '':
    #     raise ValueError(f"target_dir {target_dir} not found in {long_path}")
    #
    # else:
    #     # print(f"checking path:  {tail}\t\thead: {head}")
    #     return find_path_to_dir(long_path, target_dir, tail)

# exp_cond_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/sim1_3/0r0_dist_let_seq7_RNN_Free_v300/training'
# exp_name = 'STM_RNN/sim1_3'  #'STM_RNN/sim1_3'
# exp_path = find_path_to_dir(long_path=exp_cond_path, target_dir=exp_name)
# print(exp_path)



def running_on_laptop(verbose=True):
    """
    Check if I am on my laptop (not my work machine), might need to change paths
    :param verbose:
    :return:
    """
    # if verbose:
    if sys.executable[:18] == '/Users/nickmartin/':
        print("Script is running on Nick's laptop")
    else:
        print("Script is not running on Nick's laptop")
    return sys.executable[:18] == '/Users/nickmartin/'

def switch_home_dirs(path_to_change):
    """
    Try this module anytime I am having a problem on my laptop with uni paths.

    :param path_to_change:
    :return: new path: to try out
    """

    laptop_path = '/Users/nickmartin/Documents/PhD/python_v2/'

    GPU_path = '/home/nm13850/Documents/PhD/python_v2/'

    if laptop_path == path_to_change[:len(laptop_path)]:
        snip_end = path_to_change[len(laptop_path):]
        new_path = os.path.join(laptop_path, snip_end)
    elif GPU_path == path_to_change[:len(GPU_path)]:
        snip_end = path_to_change[len(GPU_path):]
        new_path = os.path.join(laptop_path, snip_end)
    else:
        print(f"path not found in laptop or GPU paths\n{path_to_change}")

    return new_path

# this_path = '/Users/nickmartin/Documents/PhD/python_v2/datasets/RNN/bowers14_rep'
#
# path_check = switch_home_dirs(this_path)
#
# print(f"output: {path_check}")