import csv
import datetime
import os.path
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.RNN_STM import get_label_seqs, get_test_scores, get_layer_acts


def kernel_to_2d(layer_activation_4d, reduce_type='max', verbose=False):
    """
    To perform selectivity analysis 'per unit', 4d layer need to be reduced to 2d.
    where shape is (items, width, height, depth/n_kernels)
    convert to (items, n_kernels)

    :param layer_activation_4d: the GHA of a filter/kernel (conv/pool) layer with 4d (e.g., shape: (1, 2, 3, 4))
    :param reduce_type: the method for simplifying the kernel e.g. max, mean etc
    :param verbose: whether to print intermediate steps to screen

    :return: 2d hid acts - 1 float per kernel per item
    """
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

            if reduce_type is 'mean':
                kernel_mean = np.mean(kernel_acts)

            else:  # use max
                kernel_mean = np.amax(kernel_acts)

            kernel_means.append(kernel_mean)

        # # append column to layer means
        layer_mean_acts = np.column_stack((layer_mean_acts, kernel_means))

        if verbose:
            print(f"\t{kernel}. layer_mean_acts: {np.shape(layer_mean_acts)} {type(layer_mean_acts)}")

    return layer_mean_acts


######################
# @profile
def rnn_gha(sim_dict_path,
            gha_incorrect=True,
            use_dataset='train_set',
            get_layer_list=None,
            exp_root='/home/nm13850/Documents/PhD/python_v2/experiments/',
            verbose=False,
            test_run=False
            ):
    """
    gets activations from hidden units.

    1. load simulation dict (with data info) (*_load_dict.pickle)
        sim_dict can be fed in from sim script, or loaded separately
    2. load model - get structure and details
    3. run dataset through once, recording accuracy per item/class
    4. run on 2nd model to get hid acts

    :param sim_dict_path: path to the dictionary for this experiment condition
    :param get_classes: which types of layer are we interested in?
            I've changed this to just use certain layer names rather than layer classes.
    :param gha_incorrect: GHA for ALL items (True) or just correct items (False)
    :param use_dataset: GHA for train/test data
    :param get_layer_list: if None, gha all layers, else list of layer names to gha
    :param exp_root: root to save experiments
    :param verbose:
    :param test_run: Set test = True to just do one unit per layer

    :return: dict with hid acts per layer.  saved as dict so different shaped arrays don't matter too much
    """

    print('**** ff_gha GHA() ****')

    # # # PART 1 # # #
    # # load details from dict
    if os.path.isfile(sim_dict_path):
        print(f"sim_dict_path: {sim_dict_path}")
        sim_dict = load_dict(sim_dict_path)
        full_exp_cond_path, sim_dict_name = os.path.split(sim_dict_path)

    elif os.path.isfile(os.path.join(exp_root, sim_dict_path)):
        sim_dict_path = os.path.join(exp_root, sim_dict_path)
        print(f"sim_dict_path: {sim_dict_path}")
        sim_dict = load_dict(sim_dict_path)
        full_exp_cond_path, sim_dict_name = os.path.split(sim_dict_path)
    else:
        raise FileNotFoundError(sim_dict_path)

    os.chdir(full_exp_cond_path)
    print(f"set_path to full_exp_cond_path: {full_exp_cond_path}")

    focussed_dict_print(sim_dict, 'sim_dict')


    # # # load datasets
    data_dict = sim_dict['data_info']
    if use_dataset is 'generator':
        vocab_dict = load_dict(os.path.join(data_dict["data_path"],
                                            data_dict["vocab_dict"]))
        n_cats = data_dict["n_cats"]
        x_data_path = sim_dict['training_info']['x_data_path']
        y_data_path = sim_dict['training_info']['y_data_path']
        n_items = 'unknown'

    else:
        # load data from somewhere
        n_items = data_dict["n_items"]
        n_cats = data_dict["n_cats"]
        hdf5_path = sim_dict['topic_info']["dataset_path"]

        x_data_path = hdf5_path
        y_data_path = '/home/nm13850/Documents/PhD/python_v2/datasets/' \
                      'objects/ILSVRC2012/imagenet_hdf5/y_df.csv'

        seq_data = pd.read_csv(data_dict["seqs"], header=None, names=['seq1', 'seq2', 'seq3'])
        print(f"\nseq_data: {seq_data.shape}\n{seq_data.head()}")

        X_data = np.load(data_dict["X_data"])
        print("\nshape of X_data: {}".format(np.shape(X_data)))

        Y_labels = np.loadtxt(data_dict["Y_labels"], delimiter=',').astype('int8')
        print(f"\nY_labels:\n{Y_labels}")
        print(np.shape(Y_labels))

        Y_data = to_categorical(Y_labels, num_classes=30)
        print(f"\nY_data:\n{Y_data}")
        print(np.shape(Y_data))

    # # # data preprocessing
    # # # if network is cnn but data is 2d (e.g., MNIST)
    # if len(np.shape(x_data)) != 4:
    #     if sim_dict['model_info']['overview']['model_type'] == 'cnn':
    #         width, height = sim_dict['data_info']['image_dim']
    #         x_data = x_data.reshape(x_data.shape[0], width, height, 1)
    #         print(f"\nRESHAPING x_data to: {np.shape(x_data)}")

    # # other details
    hid_units = sim_dict['model_info']['layers']['hid_layers']['hid_totals']["analysable"]
    optimizer = sim_dict['model_info']["overview"]["optimizer"]
    loss_func = sim_dict['model_info']["overview"]["loss_func"]
    batch_size = sim_dict['model_info']["overview"]["batch_size"]
    timesteps = sim_dict['model_info']["overview"]["timesteps"]
    serial_recall = sim_dict['model_info']["overview"]["serial_recall"]
    x_data_type = sim_dict['model_info']["overview"]["x_data_type"]
    end_seq_cue = sim_dict['model_info']["overview"]["end_seq_cue"]
    act_func = sim_dict['model_info']["overview"]["act_func"]
    input_dim = data_dict["X_size"]
    output_dim = data_dict["n_cats"]

    # Output files
    output_filename = sim_dict["topic_info"]["output_filename"]
    print(f"\nOutput file: {output_filename}")



    # # # # PART 2 # # #
    print("\n**** THE MODEL ****")
    model_name = sim_dict['model_info']['overview']['trained_model']
    loaded_model = load_model(model_name)
    model_details = loaded_model.get_config()
    # print_nested_round_floats(model_details)
    focussed_dict_print(model_details, 'model_details')

    n_layers = len(model_details['layers'])
    model_dict = dict()

    # # turn off "trainable" and get useful info
    for layer in range(n_layers):
        # set to not train
        model_details['layers'][layer]['config']['trainable'] = 'False'

        if verbose:
            print(f"Model layer {layer}: {model_details['layers'][layer]}")

        # # get useful info
        layer_dict = {'layer': layer,
                      'name': model_details['layers'][layer]['config']['name'],
                      'class': model_details['layers'][layer]['class_name']}

        if 'units' in model_details['layers'][layer]['config']:
            layer_dict['units'] = model_details['layers'][layer]['config']['units']
        if 'activation' in model_details['layers'][layer]['config']:
            layer_dict['act_func'] = model_details['layers'][layer]['config']['activation']
        if 'filters' in model_details['layers'][layer]['config']:
            layer_dict['filters'] = model_details['layers'][layer]['config']['filters']
        if 'kernel_size' in model_details['layers'][layer]['config']:
            layer_dict['size'] = model_details['layers'][layer]['config']['kernel_size'][0]
        if 'pool_size' in model_details['layers'][layer]['config']:
            layer_dict['size'] = model_details['layers'][layer]['config']['pool_size'][0]
        if 'strides' in model_details['layers'][layer]['config']:
            layer_dict['strides'] = model_details['layers'][layer]['config']['strides'][0]
        if 'rate' in model_details['layers'][layer]['config']:
            layer_dict["rate"] = model_details['layers'][layer]['config']['rate']

        # # set and save layer details
        model_dict[layer] = layer_dict

    # # my model summary
    model_df = pd.DataFrame.from_dict(data=model_dict, orient='index',
                                      columns=['layer', 'name', 'class', 'act_func',
                                               'units', 'filters', 'size', 'strides', 'rate'], )

    print(f"\nmodel_df\n{model_df}")

    # # make new df with just layers of interest
    if get_layer_list is None:
        key_layers_df = model_df
        get_layer_list = model_df['name'].tolist()


    key_layers_df = model_df.loc[model_df['name'].isin(get_layer_list)]

    key_layers_df.reset_index(inplace=True)
    del key_layers_df['index']
    key_layers_df.index.name = 'index'
    key_layers_df = key_layers_df.drop(columns=['size', 'strides', 'rate'])

    # # add column ('n_units_filts')to say how many things needs gha per layer (number of units or filters)
    # # add zeros to rows with no units or filters
    key_layers_df.loc[:, 'n_units_filts'] = key_layers_df.units.fillna(0) + key_layers_df.filters.fillna(0)

    # print(f"\nkey_layers_df:\n{key_layers_df}")

    key_layers_df.loc[:, "n_units_filts"] = key_layers_df["n_units_filts"].astype(int)

    # # get to total number of units or filters in key layers of the network
    key_n_units_fils = sum(key_layers_df['n_units_filts'])

    print(f"\nkey_layers_df:\n{key_layers_df.head()}")
    print(f"key_n_units_fils: {key_n_units_fils}")

    '''i currently get output layer, make sure I keep this in to make sure I can do class correlation'''

    # # # set dir to save gha stuff # # #
    hid_act_items = 'all'
    if not gha_incorrect:
        hid_act_items = 'correct'
    gha_folder = f'{hid_act_items}_{use_dataset}_gha'

    if test_run:
        gha_folder = os.path.join(gha_folder, 'test')
    gha_path = os.path.join(full_exp_cond_path, gha_folder)

    if not os.path.exists(gha_path):
        os.makedirs(gha_path)
    os.chdir(gha_path)
    print(f"\nsaving hid_acts to: {gha_path}")



    # # # PART 3 get_scores() # # #
    loaded_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

    test_label_seqs = get_label_seqs(n_labels=n_cats, seq_len=timesteps,
                                     serial_recall=serial_recall, n_seqs=10*batch_size)

    test_label_name = f"{output_filename}_{np.shape(test_label_seqs)[0]}_test_label_seqs.npy"
    print(f"test_label_name: {test_label_name}")
    np.save(test_label_name, test_label_seqs)

    scores_dict = get_test_scores(model=loaded_model, data_dict=data_dict,
                                  test_label_seqs=test_label_seqs,
                                  serial_recall=serial_recall,
                                  x_data_type=x_data_type,
                                  end_seq_cue=end_seq_cue,
                                  batch_size=batch_size,
                                  verbose=verbose)

    mean_IoU = scores_dict['mean_IoU']
    prop_seq_corr = scores_dict['prop_seq_corr']


    # # PART 5
    print("\n**** Get Hidden unit activations ****")
    hid_acts_dict = dict()

    # # loop through key layers df
    gha_key_layers = []
    for index, row in key_layers_df.iterrows():
        if test_run:
            if index > 3:
                continue

        layer_number, layer_name, layer_class = row['layer'], row['name'], row['class']
        print(f"\n{layer_number}. name: {layer_name}; class: {layer_class}")

        # if layer_class not in get_classes:  # no longer using this - skip class types not in list
        if layer_name not in get_layer_list:  # skip layers/classes not in list
            continue

        else:
            # record hid acts
            layer_activations = get_layer_acts(model=loaded_model,
                                               layer_name=layer_name,
                                               data_dict=data_dict,
                                               test_label_seqs=test_label_seqs,
                                               serial_recall=serial_recall,
                                               x_data_type=x_data_type,
                                               end_seq_cue=end_seq_cue,
                                               batch_size=batch_size,
                                               verbose=verbose
                                               )

            layer_acts_shape = np.shape(layer_activations)

            converted_to_2d = False  # set to True if 4d acts have been converted to 2d
            if len(layer_acts_shape) == 2:
                hid_acts = layer_activations

            elif len(layer_acts_shape) == 3:
                if not serial_recall:
                    ValueError(f"layer_acts_shape: {layer_acts_shape}"
                               f"\n3d expected only for serial recall")
                else:
                    hid_acts = layer_activations

            # elif len(layer_acts_shape) == 4:  # # call mean_act_conv
            #     hid_acts = kernel_to_2d(layer_activations, verbose=True)
            #     layer_acts_shape = np.shape(hid_acts)
            #     converted_to_2d = True

            else:
                ValueError(f"Unexpected number of dimensions for layer activations {layer_acts_shape}")


            hid_acts_dict[index] = {'layer_name': layer_name, 'layer_class': layer_class,
                                    "layer_shape": layer_acts_shape, 'hid_acts': hid_acts}

            if converted_to_2d:
                hid_acts_dict[index]['converted_to_2d'] = True

            print(f"\nlayer {index}. layer_acts_shape: {layer_acts_shape}\n")

            # # save distplot for sanity check
            sns.distplot(np.ravel(hid_acts))
            plt.title(str(layer_name))
            plt.savefig(f"{layer_name}_act_distplot.png")
            plt.close()


        print("\n**** saving info to summary page and dictionary ****")

        hid_act_filenames = {'2d': None, 'any_d': None}
        dict_2d_save_name = f'{output_filename}_hid_act.pickle'
        with open(dict_2d_save_name, "wb") as pkl:  # 'wb' mean 'w'rite the file in 'b'inary mode
            pickle.dump(hid_acts_dict, pkl)
        # np.save(dict_2d_save_name, hid_acts_dict)
        hid_act_filenames['2d'] = dict_2d_save_name


    cond = sim_dict["topic_info"]["cond"]
    run = sim_dict["topic_info"]["run"]
    if test_run:
        run = 'test'

    hid_units = sim_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable']

    trained_for = sim_dict["training_info"]["trained_for"]
    end_accuracy = sim_dict["training_info"]["acc"]
    dataset = sim_dict["data_info"]["dataset"]
    gha_date = int(datetime.datetime.now().strftime("%y%m%d"))
    gha_time = int(datetime.datetime.now().strftime("%H%M"))

    # gha_acc = scores_dict['gha_acc']
    # n_cats_correct = scores_dict['n_cats_correct']

    # # GHA_info_dict
    gha_dict_name = f"{output_filename}_GHA_dict.pickle"
    gha_dict_path = os.path.join(gha_path, gha_dict_name)

    gha_dict = {"topic_info": sim_dict['topic_info'],
                "data_info": sim_dict['data_info'],
                "model_info": sim_dict['model_info'],
                "training_info": sim_dict['training_info'],
                "GHA_info": {"use_dataset": use_dataset,
                             'x_data_path': x_data_path,
                             'y_data_path': y_data_path,
                             'gha_path': gha_path,
                             'gha_dict_path': gha_dict_path,
                             "gha_incorrect": gha_incorrect,
                             "hid_act_files": hid_act_filenames,
                             'gha_key_layers': gha_key_layers,
                             'key_n_units_fils': key_n_units_fils,
                             "gha_date": gha_date, "gha_time": gha_time,
                             "scores_dict": scores_dict,
                             }
                }

    with open(gha_dict_name, "wb") as pickle_out:
        pickle.dump(gha_dict, pickle_out)

    if verbose:
        focussed_dict_print(gha_dict, 'gha_dict', ['GHA_info'])


    # make a list of dict names to do sel on
    if not os.path.isfile(f"{output_filename}_dict_list_for_sel.csv"):
        dict_list = open(f"{output_filename}_dict_list_for_sel.csv", 'w')
        mywriter = csv.writer(dict_list)
    else:
        dict_list = open(f"{output_filename}_dict_list_for_sel.csv", 'a')
        mywriter = csv.writer(dict_list)

    mywriter.writerow([gha_dict_name[:-7]])
    dict_list.close()

    print(f"\nadded to list for selectivity analysis: {gha_dict_name[:-7]}")

    gha_info = [cond, run, output_filename, n_layers, hid_units, dataset, use_dataset,
                gha_incorrect, n_cats,
                timesteps,
                x_data_type,
                act_func,
                serial_recall,
                trained_for, end_accuracy, mean_IoU, prop_seq_corr,
                test_run, gha_date, gha_time]

    # # check if gha_summary.csv exists
    # # save summary file in exp folder (grandparent dir to gha folder: exp/cond/gha)
    # to move up to parent just use '..' rather than '../..'

    # exp_name = exp_dir.strip('/')
    exp_name = sim_dict['topic_info']['exp_name']

    os.chdir('../..')
    exp_path = os.getcwd()

    if not os.path.isfile(exp_name + "_GHA_summary.csv"):
        gha_summary = open(exp_name + "_GHA_summary.csv", 'w')
        mywriter = csv.writer(gha_summary)
        summary_headers = ["cond", "run", 'filename', "n_layers", "hid_units", "dataset", "GHA_on",
                           'incorrect', "n_cats",
                           "timesteps",
                           "x_data_type",
                           "act_func",
                           "serial_recall",
                           "trained_for", "train_acc", "mean_IoU", "prop_seq_corr",
                           "test_run", "gha_date", "gha_time"]

        mywriter.writerow(summary_headers)
        print(f"creating summary csv at: {exp_path}")

    else:
        gha_summary = open(exp_name + "_GHA_summary.csv", 'a')
        mywriter = csv.writer(gha_summary)
        print(f"appending to summary csv at: {exp_path}")

    mywriter.writerow(gha_info)
    gha_summary.close()

    print("\nend of ff_gha")

    return gha_info, gha_dict

###############################
# print("\n\n\n\n\nWarning\n\n\n\n\nrunning script from bottom of page!\n\n\!!!!!")
#
# gha_info, gha_dict = ff_gha(sim_dict_path='/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                           'VGG_end_aug/vgg_imagenet_sim_dict.pickle',
#                             gha_incorrect=True,
#                             use_dataset='val_set',
#                             exp_root='/home/nm13850/Documents/PhD/python_v2/experiments/',
#                             verbose=True,
#                             test_run=True
#                             )
# print("\n\n\n\n\nWarning\n\n\n\n\nrunning script from bottom of page!\n\n\!!!!!")
#
