import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

from nick_dict_tools import focussed_dict_print

tools_date = int(datetime.datetime.now().strftime("%y%m%d"))
tools_time = int(datetime.datetime.now().strftime("%H%M"))


def hdf_df_string_clean(df):
    """
    Where pandas loads a df if sometimes doesn't decode bytes to strings.
    In some cases, strings are double bytes (b'b").  This should sort that.

    :param df: dataframe to check for bytes column
    :return: cleaned df
    """

    # # convert troublesome columns to string
    if type(df.loc[0, 'class_name']) is np.bytes_:
        df["filename"] = df["filename"].str.decode('utf-8')
        df["class_name"] = df["class_name"].str.decode('utf-8')

    if type(df.loc[0, 'class_name']) is str:
        if df.loc[0, 'class_name'][0:2] in ['b"', "b'"]:
            df["filename"] = df["filename"].str[2:-1]
            df["class_name"] = df["class_name"].str[2:-1]

    # print(f"\ndf.head():\n{df.head()}")

    return df


def h5py_data_batches(data_hdf_path='/home/nm13850/Documents/PhD/python_v2/datasets/'
                                    'objects/ILSVRC2012/imagenet_hdf5/imageNet2012Val.h5',
                      total_items=50000,
                      batch_size=50,
                      x_path='x_data',
                      use_y_data=False,  # could be ['y_labels, 'y_df]
                      y_label_path='y_labels',
                      y_df_path='y_df',
                      use_vgg_colours=True,
                      verbose=False,
                      ):

    """
    Script to load batches of data from hdf5 file.  Either just x_data, or x and y_data.
    :param data_hdf_path: path to hdf5 file
    :param total_items: all items in dataset
    :param batch_size:
    :param x_path: on hdf5 file
    :param use_y_data: [False, y_labels, y_df]
    :param y_label_path: on hdf5 file
    :param y_df_path: on hdf5 file (note if made with Pandas, it might also need ['table']
    :param use_vgg_colours: preprocess RBG to BRG
    :param verbose: If True, print details to screen


    :Yield: batch of data loaded from hdf5.
            Whether y_data is included will depend on 'use_y_data'
    """


    batchsize = batch_size
    batches_in_data = total_items//batchsize

    for i in range(batches_in_data):

        with h5py.File(data_hdf_path, 'r') as dataset:

            # whole dataset would be i in range 1000 with batchsize 50
            # (50 images should fit in the memory easily)
            idx_from = i * batchsize
            idx_to = (i + 1) * batchsize
            print(f"\n{i}: from {idx_from} to: {idx_to}")

            x_data = dataset[x_path][idx_from:idx_to, ...]
            if use_vgg_colours:
                x_data = preprocess_input(x_data)


            if use_y_data == 'y_labels':
                y_labels = dataset[y_label_path][idx_from:idx_to, ...]

                if verbose:
                    print(f"x_data: {x_data.shape}")
                    print(f"y_labels: {y_labels.shape}\n{y_labels}")

                yield x_data, y_labels


            elif use_y_data == 'y_df':
                y_df_tuples = dataset[y_df_path]['table'][idx_from:idx_to, ...]

                # convert list fo tuples to list of lists
                y_df_lists = [list(elem) for elem in y_df_tuples]

                y_df = pd.DataFrame(y_df_lists, columns=['item', 'cat', 'filename', 'class_name'])
                y_df = y_df.set_index('item')

                if verbose:
                    print(f"x_data: {x_data.shape}")
                    print(f"y_df: {y_df.shape}")
                    print(f"y_df: {y_df}")


                yield x_data, y_df

            else:
                if verbose:
                    print(f"x_data: {x_data.shape}")

                yield x_data




def hdf_pred_scores(model,
                    output_filename,
                    data_hdf_path='/home/nm13850/Documents/PhD/python_v2/datasets/'
                                  'objects/ILSVRC2012/imagenet_hdf5/imageNet2012Val.h5',
                    total_items=50000,
                    batch_size=16,
                    x_path='x_data',
                    y_df_path='y_df',
                    use_vgg_colours=True,
                    df_name='item_correct_df',
                    test_run=False,
                    verbose=False,
                    ):
    """
    Script to get predictions from slices of X data on a model.
    For each slice, then also get the item correct on this slice.

    save pred_out, item correct etc into new hdf file as I go.

    keep a counter of n_items and n_correct to give accuracy
    or just get acc on the whole thing on the end.

    :param model:
    :param output_filename:
    :param data_hdf_path: path to hdf5 file
    :param total_items: all items in dataset
    :param batch_size:
    :param x_path: on hdf5 file
    :param y_df_path: on hdf5 file (note if made with Pandas, it might also need ['table']
    :param test_run: If True, don't run whole dataset, just first 64 items
    :param use_vgg_colours: preprocess RBG to BRG

    :param verbose: If True, print details to screen

    :return:
    """

    if test_run:
        total_items = 64

    batchsize = batch_size
    batches_in_data = total_items//batchsize

    # # list of all incorrect items added to slice-by-slice
    incorrect_items = []

    for i in range(batches_in_data):
        # # step through the data in slices/batches

        with h5py.File(data_hdf_path, 'r') as dataset:
            # # open the hdf with the x and y data

            # # get indices to slice to/from
            idx_from = i * batchsize
            idx_to = (i + 1) * batchsize
            print(f"\n{i}: from {idx_from} to: {idx_to}")

            # # slice x_data
            x_data = dataset[x_path][idx_from:idx_to, ...]

            # # preprocess colours from RGB to BGR
            if use_vgg_colours:
                x_data = preprocess_input(x_data)

            # # slice y data
            y_df_tuples = dataset[y_df_path]['table'][idx_from:idx_to, ...]
            # convert list fo tuples to list of lists
            y_df_lists = [list(elem) for elem in y_df_tuples]

            y_df = pd.DataFrame(y_df_lists, columns=['item', 'cat', 'filename', 'class_name'])
            y_df = y_df.set_index('item')

            if verbose:
                print(f"x_data: {x_data.shape}")
                print(f"y_df: {y_df.shape}")
                print(f"y_df: {y_df}")

            # yield x_data, y_df

            # # get the true cat labels for this slice
            true_cat = [int(i) for i in y_df['cat'].to_numpy()]

            # # get predictions (per cat) and then pred_labels
            pred_vals = model.predict(x_data)
            pred_cat = np.argmax(pred_vals, axis=1)

            # # # get item correct and scores (per cat and total)
            n_items, n_cats = np.shape(pred_vals)

            slice_incorrect_items = [x for x, y in zip(pred_cat, true_cat) if x != y]
            incorrect_items.extend(slice_incorrect_items)
            item_score = [1 if x == y else 0 for x, y in zip(pred_cat, true_cat)]


            # # append item correct to new hdf file
            item_correct_df = y_df  # .copy()

            # # convert troublesome columns to string
            item_correct_df["filename"] = item_correct_df["filename"].map(str)
            item_correct_df["class_name"] = item_correct_df["class_name"].map(str)

            # # add item_correct column ['full)_model] to item_correct df
            item_correct_df.insert(2, column="full_model", value=item_score)



            if verbose:
                print("item_correct_df.shape: {}".format(item_correct_df.shape))
                print("len(item_score): {}".format(len(item_score)))
                print(item_correct_df.dtypes)
                # print(item_correct_df.head())

            # # make output hdf to store item_correct df
            with pd.HDFStore(f"{output_filename}_gha.h5") as store:

                line_len_dict = {
                                 # 'item': 0,
                                 'cat': 0, 'full_model': 0,
                                 'filename': 35, 'class_name': 35}

                print(store.keys())
                if f"/{df_name}" not in store.keys():
                    print(f"creating blank df in {df_name} on store")

                    store.put(f'{df_name}',
                              pd.DataFrame(data=None,
                                           columns=['item', 'cat', 'full_model', 'filename', 'class_name']),
                              format='t', append=True, min_itemsize=line_len_dict)


                # # I'm having problems with line length
                # # trying to work out why
                line_len_check_dict = {}
                for c in item_correct_df:
                    if item_correct_df[c].dtype == 'object':
                        max_len = item_correct_df[c].map(len).max()
                        print(f'Max length of column {c}: {max_len}')
                        line_len_check_dict[c] = max_len
                    else:
                        max_len = 0
                        print(f'Not a string column {c}: {max_len}')
                        line_len_check_dict[c] = max_len

                line_lengths = list(line_len_check_dict.values())
                max_line = max(line_lengths)

                if max_line > 30:
                    focussed_dict_print(line_len_check_dict, 'line_len_check_dict')


                store.append(f'/{df_name}', item_correct_df, min_itemsize=line_len_dict)

                if verbose:
                    print(f"store['item_correct_df'].shape: {store[f'/{df_name}'].shape}")


    print("\nfinished looping through dataset")
    incorrect_items_s = pd.Series(incorrect_items)

    # add incorrect items to output hdf
    incorrect_items_name = 'incorrect_items'
    if df_name != 'item_correct_df':
        incorrect_items_name = f'incorrect_{df_name}'

    with pd.HDFStore(f"{output_filename}_gha.h5") as store:

        store.put(incorrect_items_name, incorrect_items_s)

        print(f"store.keys(): {store.keys()}")

        # if df_name in store.keys():
        #     pass
        # elif f"/{df_name}" in store.keys():
        #     df_name = f'/{df_name}'

        item_correct_df = store[f'/{df_name}']

    print(f"item_correct_df.shape: {item_correct_df.shape}")
    # print(item_correct_df.head())

    full_model = [int(i) for i in item_correct_df['full_model'].to_numpy()]
    fm_correct = np.sum(full_model)
    fm_items = len(full_model)

    gha_acc = np.around(fm_correct / fm_items, decimals=3)

    print("\nitems: {}\ncorrect: {}\nincorrect: {}\naccuracy: {}".
          format(fm_items, fm_correct, fm_items - fm_correct, gha_acc))

    # # get count_correct_per_class
    corr_per_cat_dict = dict()
    for cat in range(n_cats):
        corr_per_cat_dict[cat] = len(item_correct_df[(item_correct_df['cat'] == cat) &
                                                     (item_correct_df['full_model'] == 1)])

    # # # are any categories missing?
    category_fail = sum(value == 0 for value in corr_per_cat_dict.values())
    category_low = sum(value < 3 for value in corr_per_cat_dict.values())
    n_cats_correct = n_cats - category_fail

    scores_dict = {"n_items": fm_items, "n_correct": fm_correct, "gha_acc": gha_acc,
                   "category_fail": category_fail, "category_low": category_low,
                   "n_cats_correct": n_cats_correct,
                   "corr_per_cat_dict": corr_per_cat_dict,
                   # "item_correct_name": item_correct_name,
                   # "flat_conf_name": flat_conf_name,
                   "scores_date": tools_date, 'scores_time': tools_time}
    

    return item_correct_df, scores_dict, incorrect_items


def hdf_gha(model,
            layer_name,
            layer_number,
            layer_class,
            output_filename,
            data_hdf_path='/home/nm13850/Documents/PhD/python_v2/datasets/'
                          'objects/ILSVRC2012/imagenet_hdf5/imageNet2012Val.h5',
            total_items=50000,
            batch_size=16,
            x_path='x_data',
            use_vgg_colours=True,
            gha_incorrect=True,
            test_run=False,
            verbose=False,
            ):

    print('getting layer')


    converted_to_2d = False  # set to True if 4d acts have been converted to 2d

    # model to record hid acts
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


    # # x data slice generator goes here
    if test_run:
        total_items = 64

    batchsize = batch_size
    batches_in_data = total_items//batchsize

    for i in range(batches_in_data):
        # # step through the data in slices/batches

        # # get indices to slice to/from
        idx_from = i * batchsize
        idx_to = (i + 1) * batchsize
        print(f"\n{i}: from {idx_from} to: {idx_to}")

        with h5py.File(data_hdf_path, 'r') as dataset:
            # # open the hdf with the x and y data

            # # slice x_data
            x_data = dataset[x_path][idx_from:idx_to, ...]

            # # preprocess colours from RGB to BGR
            if use_vgg_colours:
                x_data = preprocess_input(x_data)

            if verbose:
                print(f"x_data: {x_data.shape}")


        # # # which items to run gha on - all or just correct items
        gha_items = x_data

        if not gha_incorrect:  # If I want ALL items including those classified incorrectly
            print("\n**** REMOVE INCORRECT FROM X DATA ****")
            # # filter out incorrect items from that slice
            output_hdf_name = f"{output_filename}_gha.h5"

            item_correct_df = pd.read_hdf(path_or_buf=output_hdf_name,
                                          key='item_correct_df',
                                          mode='r+',
                                          start=idx_from,
                                          stop=idx_to,
                                          )

            incorrect_items = item_correct_df['full_model'].tolist()
            mask = np.ones(len(x_data), dtype=bool)
            mask[incorrect_items] = False
            gha_items = x_data[mask]  # correct items only

        print(f"gha_items: (incorrect items={gha_incorrect}) {np.shape(gha_items)}")



        # # # use predict (not predict_generator) on x data
        intermediate_output = intermediate_layer_model.predict(gha_items, verbose=1)

        layer_acts_shape = np.shape(intermediate_output)

        if len(layer_acts_shape) == 2:
            acts_2d = intermediate_output

            items, kernels = np.shape(intermediate_output)


        elif len(layer_acts_shape) == 4:
            # # reduce dimensions from (items, width, height, depth/n_kernels)
            #     convert to (items, n_kernels) using max per kernel
            # acts_2d = kernel_to_2d(intermediate_output, reduce_type='max', verbose=True)

            layer_activation_4d = intermediate_output

            print('\n**** kernel_to_2d GHA() ****')
            items, width, height, kernels = np.shape(layer_activation_4d)

            if verbose:
                print(f"\t{kernels} kernels, shape ({width}, {height})")

            # # to save all item averages per conv filter make: layer_mean_acts
            layer_mean_acts = np.empty((items, 0))

            # # loop through conv filters
            for kernel in range(kernels):
                this_kernel = layer_activation_4d[:, :, :, kernel]

                # # to save averages per item as computed
                kernel_means = []
                for item in range(items):
                    kernel_acts = this_kernel[item]

                    # if reduce_type is 'mean':
                    #     kernel_mean = np.mean(kernel_acts)
                    # else:  # use max
                    kernel_mean = np.amax(kernel_acts)

                    kernel_means.append(kernel_mean)

                # # append column to layer means
                layer_mean_acts = np.column_stack((layer_mean_acts, kernel_means))

                if verbose:
                    print(f"\t{kernel}. layer_mean_acts: {np.shape(layer_mean_acts)} {type(layer_mean_acts)}")

            acts_2d = layer_mean_acts

            layer_acts_shape = np.shape(acts_2d)
            converted_to_2d = True

        else:
            print("\n\n\n\nSHAPE ERROR - UNEXPECTED DIMENSIONS\n\n\n\n")
            acts_2d = 'SHAPE_ERROR'
            layer_acts_shape = 'NONE'

        # # store data in hid_act dict
        hid_act_2d_dict = {'layer_name': layer_name, 'layer_class': layer_class,
                           "layer_shape": layer_acts_shape}

        if converted_to_2d:
            hid_act_2d_dict['converted_to_2d'] = True

        if verbose:
            print(f"\nlayer{layer_number}. hid_act_2d_dict: {layer_acts_shape}\n")



        # # store hid acts
        with h5py.File(f"{output_filename}_gha.h5", 'a') as store:

            # # create a group to store hid acts in, with layer names as keys
            hid_acts_group = store.require_group('hid_acts_2d')

            # # make an empty dataset of size items X kernels
            print(f"hid_acts_group.keys(): {hid_acts_group.keys()}")
            if layer_name not in hid_acts_group.keys():
                hid_acts_store = hid_acts_group.require_dataset(name=layer_name,
                                                                shape=(items, kernels),
                                                                dtype=float,
                                                                maxshape=(None, kernels),
                                                                compression="gzip")
            else:
                hid_acts_store = hid_acts_group[layer_name]

            # resize dataset to accomodate new data
            print(f"hid_acts_store prev size: {hid_acts_store.shape}")
            hid_acts_store.resize((idx_to, kernels))

            # append hid acts to dataset at right location
            hid_acts_store[idx_from:idx_to] = acts_2d
            print(f"hid_acts_store new size: {hid_acts_store.shape}")




        # # save distplot of activations
        sns.distplot(np.ravel(acts_2d))
        plt.title(str(layer_name))
        plt.savefig(f"{output_filename}_{layer_name}_act_distplot.png")
        plt.close()

    return hid_act_2d_dict
