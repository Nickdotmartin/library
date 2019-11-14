import os
import numpy as np
import more_itertools
import pandas as pd
from tensorflow.keras.models import load_model, Model

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats



def get_label_seqs(n_labels=30, seq_len=3, serial_recall=False, n_seqs=1):
    """
    Generate random sequences of labels for STM_RNN project.

    :param n_labels: number of labels to select from
    :param seq_len: Number of labels to select for each seq
    :param serial_recall: For serial recall a sequence may contian an item more than once e.g. [0, 1, 0]
        For free recall and item may only appear in each sequence once.
    :param n_seqs: number of sequences to generate.

    :return: 2d numpy array (n_seqs, seq_len)
    """
    class_list = np.arange(n_labels)

    sequences = []

    if n_seqs == 1:
        if serial_recall:
            sequences = more_itertools.random_product(class_list, repeat=seq_len)
        else:
            sequences = more_itertools.random_permutation(iterable=class_list, r=seq_len)
    else:
        for s in range(n_seqs):
            if serial_recall:
                this_seq = more_itertools.random_product(class_list, repeat=seq_len)
            else:
                this_seq = more_itertools.random_permutation(iterable=class_list, r=seq_len)
            sequences.append(this_seq)

    return np.array(sequences)

# # # test get_label_seqs
# test_get_label_seqs = get_label_seqs(n_labels=3, seq_len=3, serial_recall=False, n_seqs=3)
# print(f"\ntest_get_label_seqs: {type(test_get_label_seqs)}  {np.shape(test_get_label_seqs)}")
# print(test_get_label_seqs)
# print(f"test_get_label_seqs[0]: {type(test_get_label_seqs[0])}  {np.shape(test_get_label_seqs)[0]}")


def get_X_and_Y_data_from_seq(vocab_dict,
                              seq_line,
                              serial_recall=False,
                              x_data_type='dist_letter_X',
                              end_seq_cue=False
                              ):
    """
    Take a single Y_Label_seq and return the corresponding X and Y data.

    :param vocab_dict: Dict containing the codes for Y and Y data
    :param seq_line: The Y_label_seq to get the X and Y data for
    :param x_data_type: 'local_word_X', 'local_letter_X', 'dist_letter_X'.
    :param serial_recall: For serial recall, Y array is made of 1hot vectors
                    If False, make single n-hot array where n=seq len.  e.g., activate all words simultaneously
    :param end_seq_cue: if True, add input unit which is activated for the last item in each seq

    :return: numpy arrays of X_data and y_data as specified in vocab
    """
    # # X data
    # # add an additional unit which is only activated on the last item of each seq
    if end_seq_cue:
        # get seq_len to know which is last item of seq
        seq_len = len(seq_line) - 1
        # print(f"seq_len: {seq_len}")

        x_data = []
        for index, item in enumerate(seq_line):
            this_word = vocab_dict[item][x_data_type]

            if index < seq_len:
                this_word = this_word + [0]
            else:
                this_word = this_word + [1]

            x_data.append(this_word)
    else:
        # x_data = [vocab_dict[item][x_data_type] for item in seq_line]
        x_data = []
        for item in seq_line:
            this_word = vocab_dict[item][x_data_type]
            x_data.append(this_word)

    # # Y data
    if serial_recall:
        # y_data = [vocab_dict[item]['local_word_X'] for item in seq_line]
        # get n_items
        n_items = max(list(vocab_dict.keys())) + 1

        y_data = []
        for category in seq_line:
            # make blank array of right length
            y_vector = [0]*n_items
            # change relevant items to 1
            y_vector[category] = 1
            y_data.append(y_vector)

    else:
        # get n_items
        n_items = max(list(vocab_dict.keys())) + 1

        # make blank array of right length
        y_data = [0]*n_items

        # change relevant items to 1
        for category in seq_line:
            y_data[category] = 1


    return np.array(x_data), np.array(y_data)

# # # test get_X_and_Y_data_from_seq
# print("\ntest get_X_and_Y_data_from_seq")
# vocab_dict = load_dict('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep/vocab_30_dict.txt')
#
# # # single sequence
# # this_seq = [0, 1, 2]
# # get_x, get_y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
# #                                          seq_line=this_seq,
# #                                          x_data_type='dist_letter_X',
# #                                          serial_recall=False,
# #                                          end_seq_cue=False)
# # print(f"x: {get_x}\ny:{get_y}")
#
# # many sequences
# # these_seqs = get_label_seqs(n_labels=3, seq_len=3, serial_recall=False, n_seqs=3)
# # these_seqs = [[0, 2, 3], [5, 3, 6], [0, 4, 2]]
# these_seqs = [[0], [2], [3], [5], [3], [6], [0], [4], [2]]
# x_seqs = []
# y_seqs = []
# # for index, this_seq in enumerate(these_seqs):
# for this_seq in these_seqs:
#
#     get_x, get_y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict, seq_line=this_seq, serial_recall=False)
#     x_seqs.append(get_x)
#     y_seqs.append(get_y)
#     # print(f"\n{index}: {this_seq}\nx:{get_x}\ny: {get_y}")
#
# x_seqs = np.array(x_seqs)
# print(f"\nx_seqs: {type(x_seqs)}  {np.shape(x_seqs)}")
# print(x_seqs)
# y_seqs = np.array(y_seqs)
# print(f"\ny_seqs: {type(y_seqs)}  {np.shape(y_seqs)}")
# print(y_seqs)



# make data generator
def generate_STM_RNN_seqs(data_dict,
                          seq_len,
                          batch_size=16,
                          serial_recall=False,
                          x_data_type='dist_letter_X',
                          end_seq_cue=False,
                          ):
    """
    https://keras.io/models/model/#fit_generator see example

    Will generate a list of seqs e.g., [[1, 2, 3], [6, 2, 4]...
    and make X_data and Y_data from vocab_dict.
    Note that this generator relies on two functions
    (e.g., can produce sequences infinteley,
    although the other functions work one at a time.)

    :param data_dict: dict for this dataset with links to vocab_dict
    :param seq_len: Or time-steps.  number of items per seq.
    :param batch_size: Generator outputs in batches - this sets their size
    :param serial_recall: default=false. free recall, y is a single n-hot vector where n=seq_len.
        In this conditon a seq can not contain repeated items. For y make a single n-hot array where n=seq len.
        e.g., activate all words simultaneously
        If serial_recall=True, y is a (classes, n_seqs) array of 1hot vectors, which can contain repeated items.
            for y, use append local_word_X
    :param x_data_type: 'local_word_X', 'local_letter_X', 'dist_letter_X'.
                Note for 1hot Y data use local_word_X
    :param end_seq_cue: if True, add input unit which is activated for the last item in each seq

    :Yeild: X_array and Y_array
    """

    # load vocab dict
    vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))
    n_cats = data_dict['n_cats']
    class_list = list(range(n_cats))

    while True:      # this allows it to go on forever
        x_batch = []
        y_batch = []

        for items in range(batch_size):
            # generate random seq of numbers from class_list
            if serial_recall:
                this_seq = more_itertools.random_product(class_list, repeat=seq_len)
            else:
                this_seq = more_itertools.random_permutation(iterable=class_list, r=seq_len)

            # get input and output data from vocab dict for this_seq
            get_X, get_Y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                     seq_line=this_seq,
                                                     x_data_type=x_data_type,
                                                     serial_recall=serial_recall,
                                                     end_seq_cue=end_seq_cue
                                                     )
            x_batch.append(get_X)
            y_batch.append(get_Y)

        # yeild returns a generator not an array.
        yield np.asarray(x_batch), np.asarray(y_batch)

# ######################################
# print("\ntest generate_STM_RNN_seqs")
#
# generate_data = generate_STM_RNN_seqs(data_dict=load_dict('/home/nm13850/Documents/PhD/python_v2/'
#                                                           'datasets/RNN/bowers14_rep/'
#                                                           'vocab_30_load_dict.txt'),
#                                       seq_len=3,
#                                       batch_size=4,
#                                       serial_recall=True,
#                                       x_data_type='dist_letter_X',
#                                       end_seq_cue=True
#                                       )
#
# print("\ntesting generator for ten iterations")
#
# y_labels = []
# x_data = []
# y_data = []
#
# for i in range(10):
#     this_seq = next(generate_data)
#     x_data.append(this_seq[0])
#     y_data.append(this_seq[1])
#
# x_data = np.array(x_data)
# print(f"\nx_data: {type(x_data)}  {np.shape(x_data)}")
# # print(x_data)
# y_data = np.array(y_data)
# print(f"\ny_data: {type(y_data)}  {np.shape(y_data)}")
# # print(y_data)



# # get test scores.
def get_test_scores(model, data_dict, test_label_seqs,
                    serial_recall=False,
                    end_seq_cue=False,
                    batch_size=16,
                    verbose=True):
    """
    Get test scores for the model for test_labels.

    1. First get x and y data from get_x_and_Y_data_from_seq
    2. predictions
    3. convert these into right format for analysis (list of predicted labels e.g., [[12, 5, 2], [...)
        this might be different for free recall and serial recall
    4. get accuracy
        - meanIoU.  For each seq get IoU.  intersection = matching labels from pred and true.
                                           Union = All labels from pred and true.
        - prop seq correct.  How many sequences is have IoU = 1.
    5. return scores_dict = {"n_seqs": n_seqs,
                             "mean_IoU": mean_IoU,
                             "prop_seq_corr": prop_seq_corr,
                             }


    For free recall - IoU is already set up/ (sort Y-True)
    For serial, turn predictions into seq labels (don't sort)

    :param model: trained model
    :param data_dict: dict with relevant info including link to vocab dict
    :param test_label_seqs: sequence of labels to test on
    :param serial_recall: Type of recall
    :param end_seq_cue:
    :param batch_size:
    :param verbose:

    :return: scores_dict = {"n_seqs": n_seqs,
                             "mean_IoU": mean_IoU,
                             "prop_seq_corr": prop_seq_corr,
                            }
    """

    print("\n**** get_test_scores() ****")

    # # load x and y data from vocab dict.
    vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))


    # with test_labels get x and y data from get_x_and_Y_data_from_seq
    labels_test = test_label_seqs
    if len(np.shape(labels_test)) == 2:
        n_seqs, seq_len = np.shape(labels_test)

    x_test = []
    y_test = []
    for this_seq in labels_test:
        get_x, get_y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                 seq_line=this_seq,
                                                 serial_recall=serial_recall,
                                                 end_seq_cue=end_seq_cue
                                                 )
        x_test.append(get_x)
        y_test.append(get_y)

    x_test = np.array(x_test).astype(np.float32)
    # x_test = np.dtype(np.int32)
    y_test = np.array(y_test).astype(np.float32)
    # y_test = np.dtype(np.int32)

    if verbose:
        print(f"\nx_test: {np.shape(x_test)}\ny_test: {np.shape(y_test)}\n"
              f"labels_test: {np.shape(labels_test)}")

    print(f"type(x_test): {type(x_test)}")
    print(f"type(x_test[0][0][0]): {type(x_test[0][0][0])}")

    # # get class labels for predictions
    if serial_recall:
        # print("predicting classes")
        all_pred_labels = model.predict_classes(x_test, batch_size=batch_size, verbose=1)
        # print(f"all_pred_labels: {np.shape(all_pred_labels)}")

        # # sanitycheck
        # pred_y_values = model.predict(x_test, batch_size=batch_size, verbose=verbose)


    else:
        # print("predicting y_values")
        pred_y_values = model.predict(x_test, batch_size=batch_size, verbose=verbose)
        # print(f"pred_y_values: {np.shape(pred_y_values)}")

        # # get labels for classes where value is greater than .5
        all_pred_labels = []
        for seq in range(n_seqs):
            these_pred_labels = np.argwhere(pred_y_values[seq] > .5)

            # flatted predictions
            if len(np.shape(these_pred_labels)) > 1:
                these_pred_labels = np.ravel(these_pred_labels).tolist()
            all_pred_labels.append(these_pred_labels)

    # if verbose:
    #     print(f"\nlabels_test: {np.shape(labels_test)}"
    #           # f"\n{labels_test}"
    #           f"\nall_pred_labels: {np.shape(all_pred_labels)}"
    #           # f"\n{all_pred_labels}"
    #           )

        # print("sanity check - first item")
        # pred_round = [np.round(i, 2) for i in pred_y_values[0]]
        # print(f"\n\n\nlabels_test: {labels_test[0]}\n"
        #       f"all_pred_labels: {all_pred_labels[0]}\n"
        #       f"pred_round: {pred_round}\n\n\n")

    print("\nIoU acc")
    iou_scores = []
    seq_corr_list = []
    for seq in range(n_seqs):

        true_labels = labels_test[seq, :]
        pred_labels = all_pred_labels[seq]

        # make set to get intersection and union
        pred_labels = set(pred_labels)

        intersection = pred_labels.intersection(true_labels)
        union = pred_labels.union(true_labels)

        IoU = len(intersection) / len(union)
        iou_scores.append(IoU)

        if IoU == 1.0:
            seq_corr_list.append(int(1))
        else:
            seq_corr_list.append(int(0))

        if verbose:
            print(f"{seq}: pred: {pred_labels} true: {true_labels} len(intersection): {len(intersection)} IoU: {IoU}")

    # get the average of all IoUs (per seq/batch etc
    mean_IoU = sum(iou_scores) / len(iou_scores)

    # # get prop of seqs where IoU == 1.0
    n_seq_corr = sum(seq_corr_list)
    print(f"n_seq_corr: {n_seq_corr}")
    print(f"len(seq_corr_list): {len(seq_corr_list)}")
    prop_seq_corr = n_seq_corr / len(seq_corr_list)

    scores_dict = {"n_seqs": n_seqs,
                   "mean_IoU": mean_IoU,
                   "prop_seq_corr": prop_seq_corr,
                   "n_seq_corr": n_seq_corr,
                   "seq_corr_list": seq_corr_list
                   }

    if verbose:
        focussed_dict_print(scores_dict, 'scores_dict')

    return scores_dict

####################
# print("\nTesting get_test_scores")
# data_dict = load_dict('/home/nm13850/Documents/PhD/python_v2/datasets/'
#                       'RNN/bowers14_rep/vocab_30_data_load_dict.txt')
# vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))
#
# n_cats = 30
# timesteps = 3
# serial_recall = False
# x_data_type = 'dist_letter_X'
# end_seq_cue = False
# batch_size = 16
# verbose=True
#
# from tensorflow.keras.models import load_model
# model = load_model("/home/nm13850/Documents/PhD/python_v2/experiments/"
#                    "STM_RNN/STM_RNN_test_v30_free_recall/"
#                    "STM_RNN_test_v30_free_recall_model.hdf5")
#
# # test_label_seqs = np.array([[0, 2, 3], [5, 3, 6], [0, 4, 2], [11, 12, 13]])
# #
# test_label_seqs = get_label_seqs(n_labels=n_cats, seq_len=timesteps,
#                                  serial_recall=serial_recall, n_seqs=100)
#
# # # call get test accracy(serial_recall,
# test_score_dict = get_test_scores(model=model, data_dict=data_dict, test_label_seqs=test_label_seqs,
#                 serial_recall=serial_recall,
#                 x_data_type=x_data_type,
#                 end_seq_cue=end_seq_cue,
#                 # batch_size=batch_size,
#                 verbose=verbose)
#
# # print(test_score_dict)

# # get test scores.
def get_layer_acts(model, layer_name, data_dict, test_label_seqs,
                   serial_recall=False,
                   end_seq_cue=False,
                   batch_size=16,
                   verbose=True):
    """
    Get test hidden activations for the model for test_labels.

    The layer being recorded from will ALWAYS have return_sequences=True,
    so that I can record activations fro each timestep.

    1. First get x and y data from get_x_and_Y_data_from_seq
    2. change return_sequences=True, if necessary
    3. predictions to get layer activations

    :param model: trained model
    :param layer_name: name of layer to get activations from
    :param data_dict: dict with relevant info including link to vocab dict
    :param test_label_seqs: sequence of labels to test on
    :param serial_recall: Type of recall
    :param end_seq_cue: whether extra input unit is added
    :param batch_size: batch to predict at once
    :param verbose:

    :return: np.array of activations at each timestep
    """

    print(f"\n**** get_layer_acts({layer_name}) ****")


    # # load x and y data from vocab dict.
    vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))

    # with test_label_seqs get x and y data from get_x_and_Y_data_from_seq
    x_test = []
    y_test = []
    for this_seq in test_label_seqs:
        get_x, get_y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                 seq_line=this_seq,
                                                 serial_recall=serial_recall,
                                                 end_seq_cue=end_seq_cue)
        x_test.append(get_x)
        y_test.append(get_y)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if verbose:
        print(f"\nx_test: {np.shape(x_test)}\ny_test: {np.shape(y_test)}\n"
              f"test_label_seqs: {np.shape(test_label_seqs)}")

    # # make new model
    gha_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


    # # get hid acts for each timestep even if output is free-recall
    # print("\nchanging layer attribute: return_sequnces")
    # print(f"target layer_name: {layer_name}")
    for layer in gha_model.layers:
        # set to return sequences = True
        # print(f"this layer name: {layer.name}")
        if layer.name == layer_name:
            # print("layer matches")
            try:
                layer.return_sequences = True
                # print(layer.name, layer.return_sequences)
            except AttributeError:
                print(f"can't change return seqs on {layer.name}")

    # # get dict for updated model
    gha_model_config = gha_model.get_config()
    # focussed_dict_print(gha_model_config)

    # # I have to reload the model from the dict for the changes to actually take place!
    gha_model = Model.from_config(gha_model_config)

    layer_activations = gha_model.predict(x_test, batch_size=batch_size, verbose=verbose)

    if verbose:
        print(f"layer_activations: {np.shape(layer_activations)}")

    return layer_activations

# ####################
# print("\nTesting get_layer_acts")
# free_model_dict = {"model_path": '/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                  'STM_RNN/STM_RNN_test_v30_free_recall/test/'
#                                  'STM_RNN_test_v30_free_recall_model.hdf5',
#                    'layer_name': 'hid0',
#                    'test_label_seqs_name': '/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                            'STM_RNN/STM_RNN_test_v30_free_recall/test/all_generator_gha/test/'
#                                            'STM_RNN_test_v30_free_recall_320_test_label_seqs.npy',
#                    'sim_dict_path': '/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                     'STM_RNN/STM_RNN_test_v30_free_recall/test/'
#                                     'STM_RNN_test_v30_free_recall_sim_dict.txt'
#                    }
# seri_model_dict = {"model_path": '/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                  'STM_RNN/STM_RNN_test_v30_serial_recall/test/'
#                                  'STM_RNN_test_v30_serial_recall_model.hdf5',
#                    'layer_name': 'hid0',
#                    'test_label_seqs_name': '/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                            'STM_RNN/STM_RNN_test_v30_serial_recall/test/all_generator_gha/test/'
#                                            'STM_RNN_test_v30_serial_recall_320_test_label_seqs.npy',
#                    'sim_dict_path': '/home/nm13850/Documents/PhD/python_v2/experiments/'
#                                     'STM_RNN/STM_RNN_test_v30_serial_recall/test/'
#                                     'STM_RNN_test_v30_serial_recall_sim_dict.txt'
#                    }
#
# test_dict = free_model_dict
# sim_dict = load_dict(test_dict['sim_dict_path'])
# data_dict = sim_dict['data_info']
# loaded_model = load_model(test_dict['model_path'])
# test_label_seqs = np.load(test_dict['test_label_seqs_name'])
#
# serial_recall = sim_dict['model_info']['overview']['serial_recall']
# x_data_type = sim_dict['model_info']['overview']['x_data_type']
# end_seq_cue = sim_dict['model_info']['overview']['end_seq_cue']
# batch_size = sim_dict['model_info']['overview']['batch_size']
#
# test_get_layer_acts = get_layer_acts(model=loaded_model,
#                                      layer_name='hid0',
#                                      data_dict=data_dict,
#                                      test_label_seqs=test_label_seqs,
#                                      serial_recall=serial_recall,
#                                      x_data_type=x_data_type,
#                                      end_seq_cue=end_seq_cue,
#                                      batch_size=batch_size,
#                                      verbose=True)
#
# print(f"end\nlayer_acts: {np.shape(test_get_layer_acts)}")


def seq_items_per_class(label_seqs, vocab_dict):
    """
    Script to calculate number of items per class in a given sequence of items.

    Output is 4 dicts
    1. Word count overall
    2. word count per timestep

    3. letter count overall
    4. letter count per timestep.

    :param label_seqs: Sequence of labels to get IPC for.
    :param vocab_dict: Vocab dict containing letter codes for words.

    :return: IPC_dict
    """
    print("\n****running seq_items_per_class()****")

    # print(f"\nseq_labels: {np.shape(seq_labels)}")
    # print(f"\nseq_labels:\n{seq_labels}")
    # focussed_dict_print(vocab_dict, 'vocab_dict')

    # # get variables
    n_seqs, n_ts = np.shape(label_seqs)
    ts_labels = [f"ts{i}" for i in range(n_ts)]

    # # 1. word ocurrence in whole seq
    whole_seq = np.ravel(label_seqs)
    unique, counts = np.unique(whole_seq, return_counts=True)
    word_p_class_all = dict(zip(unique, counts))
    # print(f"\nword_p_class_all\n{word_p_class_all}")


    # # 2. word_p_class_p_ts
    word_p_class_p_ts = dict()
    for index, ts in enumerate(ts_labels):
        ts_items = label_seqs[:, index]
        unique, counts = np.unique(ts_items, return_counts=True)
        word_p_class_ts = dict(zip(unique, counts))
        word_p_class_p_ts[ts] = word_p_class_ts
    # focussed_dict_print(word_p_class_p_ts, 'word_p_class_p_ts')

    # # get array with letters, shape (word_len, n_seqs, n_ts)
    seq_letters = []
    for ts in range(n_ts):
        ts_items = label_seqs[:, ts]
        ts_letters = []
        for seq in range(n_seqs):
            word = ts_items[seq]
            letter_vector = vocab_dict[word]['letters']
            ts_letters.append(letter_vector)
        seq_letters.append(ts_letters)

    # # 3. letter occurence in whole seq
    whole_seq = np.ravel(seq_letters)
    unique, counts = np.unique(whole_seq, return_counts=True)
    letter_p_class_all = dict(zip(unique, counts))
    # print(f"\nletter_p_class_all\n{letter_p_class_all}")

    # # # 4. letter occurence per timestep
    letter_p_class_p_ts = dict()
    for index, ts in enumerate(ts_labels):
        ts_letters = np.ravel(seq_letters[index])
        unique, counts = np.unique(ts_letters, return_counts=True)
        letter_p_class_ts = dict(zip(unique, counts))
        letter_p_class_p_ts[ts] = letter_p_class_ts
    # focussed_dict_print(letter_p_class_p_ts, 'letter_p_class_p_ts')

    IPC_seq_dict = {'word_p_class_all': word_p_class_all,
                    'word_p_class_p_ts': word_p_class_p_ts,
                    'letter_p_class_all': letter_p_class_all,
                    'letter_p_class_p_ts': letter_p_class_p_ts,
                    }

    return IPC_seq_dict


#####
# print(f"\n\n\ntesting IPC from bottom of script\n\n\n")
#
# seq_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
#                  'STM_RNN/STM_RNN_test_v30_free_recall/test/all_generator_gha/test/' \
#                  'STM_RNN_test_v30_free_recall_320_test_label_seqs.npy'
# seq_labels = np.load(seq_label_path)
# # print(f"seq_labels: {np.shape(seq_labels)}")
#
# vocab_dict = load_dict(os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep',
#                                     'vocab_30_dict.txt'))
# # focussed_dict_print(vocab_dict)
#
# IPC_dict = seq_items_per_class(label_seqs=seq_labels, vocab_dict=vocab_dict)
#
# focussed_dict_print(IPC_dict, 'IPC_dict')


def spell_label_seqs(test_label_seqs, vocab_dict,
                     test_label_name='test_label_words', save_csv=True):
    """
    input a list of sequence labels and return list of english words.

    :param test_label_seqs: sequence of ints to convert to words
    :param vocab_dict: vocab dict to map ints to words
    :param test_label_name: name to save file
    :param save_csv: whether to save file

    :return: spelled_label_seqs_df: dataframe of words with
                                    timesteps as columns, items as rows
    """

    if save_csv:
        if test_label_name is None:
            raise ValueError("Enter a name/path to save the file")

    if type(test_label_seqs) == str:
        if os.path.isfile(test_label_seqs):
            test_labels = np.load(test_label_seqs)
    elif type(test_label_seqs) == np.ndarray:
        test_labels = test_label_seqs
    else:
        raise TypeError("test_label_seqs should be path or np.ndarray")
    # print(test_labels)
    # print(np.shape(test_labels))
    # print(len(np.shape(test_labels)))
    # print(type(test_labels))
    if len(np.shape(test_labels)) is 2:
        items, seqs = np.shape(test_labels)
    elif len(np.shape(test_labels)) is 1:
        seqs = 1
    else:
        raise ValueError("expecting 1d or 2d np.ndarray for test_labels\n"
                         f"This array is shape {np.shape(test_labels)}")

    # focussed_dict_print(vocab_dict)

    df_headers = [f"ts{seq}" for seq in list(range(seqs))]
    test_label_df = pd.DataFrame(data=test_labels,
                                 columns=df_headers)
    # print(test_label_df.head())
    # test_label_df.to_csv(os.path.join(np_dir, csv_name), index=False)

    spelled_label_seqs = []
    for index, row in test_label_df.iterrows():
        row_seqs = []
        # print(index, row.tolist())
        for label in row.tolist():
            this_word = vocab_dict[label]['word']
            # print(label, this_word)
            row_seqs.append(this_word)
        spelled_label_seqs.append(row_seqs)


    spelled_label_seqs_df = pd.DataFrame(data=spelled_label_seqs,
                                         columns=df_headers)
    # print(spelled_label_seqs_df.head())

    if save_csv:
        spelled_label_seqs_df.to_csv(test_label_name, index=False)

    return spelled_label_seqs_df

# print("\n\n\n*********testing spell_label_seqs()\n\n")
# free_test_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/STM_RNN_test_v30_free_recall/test/all_generator_gha/test/STM_RNN_test_v30_free_recall_lett_320_corr_test_label_seqs.npy'
# seri_test_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/STM_RNN_test_v30_serial_recall/test/all_generator_gha/test/STM_RNN_test_v30_serial_recall_320_corr_test_label_seqs.npy'
# seri_3l_test_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/STM_RNN_test_v30_serial_recall_3l/test/all_generator_gha/test/STM_RNN_test_v30_serial_recall_3l_320_corr_test_label_seqs.npy'
#
# this_one = seri_3l_test_label_path
# np_dir, np_name = os.path.split(this_one)
# csv_name = os.path.join(np_dir, f"{np_name[:-8]}spelled.csv")
#
# vocab_dict = load_dict(os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep',
#                                     'vocab_30_dict.txt'))
# get_words = spell_label_seqs(test_label_seqs=this_one,
#                              test_label_name=csv_name,
#                              vocab_dict=vocab_dict, save_csv=True)


##########################################

def letter_in_seq(letter, test_label_seqs, vocab_dict):
    """
    input a list of sequence labels and a letter to search for.
    Return a binary array showing whether the letter in question is present in each time.

    :param letter:  to search for
    :param test_label_seqs: items to search in
    :param vocab_dict: to check it all out.

    :return: binary numpy array
    """

    print(f"letter: {letter}\n"
          f"test_label_seqs: {np.shape(test_label_seqs)}\n"
          # f"{test_label_seqs}"
          )

    letter_id_dict = load_dict('/home/nm13850/Documents/PhD/python_v2/datasets/'
                               'RNN/bowers14_rep/letter_id_dict.txt')

    if type(letter) is int:
        letter_id = letter
        letter = letter_id_dict[letter_id]
    elif type(letter) is str:
        all_letters = list(letter_id_dict.values())
        print(all_letters)
        letter_id = all_letters.index(letter)
    else:
        raise TypeError("letter to search for should be string or int")

    print(f"letter: {letter}  id: {letter_id}")

    letter_present_list = []
    for row in test_label_seqs:
        # print(row)
        new_row = []
        for item in row:
            spelled_word = vocab_dict[item]['letters']
            if letter in spelled_word:
                print(spelled_word, letter)
                new_row.append(1)
            else:
                print(spelled_word, )
                new_row.append(0)
        letter_present_list.append(new_row)

    letter_present_array = np.array(letter_present_list)

    return letter_present_array

# print("\n\n\n*********testing letter_in_seq()\n\n")
# free_test_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
#                        'STM_RNN_test_v30_free_recall/test/all_generator_gha/test/' \
#                        'STM_RNN_test_v30_free_recall_lett_320_corr_test_label_seqs.npy'
# seri_test_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
#                        'STM_RNN_test_v30_serial_recall/test/all_generator_gha/test/' \
#                        'STM_RNN_test_v30_serial_recall_320_corr_test_label_seqs.npy'
# seri_3l_test_label_path = '/home/nm13850/Documents/PhD/python_v2/experiments/STM_RNN/' \
#                           'STM_RNN_test_v30_serial_recall_3l/test/all_generator_gha/test/' \
#                           'STM_RNN_test_v30_serial_recall_3l_320_corr_test_label_seqs.npy'
#
# this_one = free_test_label_path
# np_dir, np_name = os.path.split(this_one)
# # csv_name = os.path.join(np_dir, f"{np_name[:-8]}spelled.csv")
#
# test_label_seqs = np.load(free_test_label_path)
#
# vocab_dict = load_dict(os.path.join('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep',
#                                     'vocab_30_dict.txt'))
# get_letters = letter_in_seq(letter='ea', test_label_seqs=test_label_seqs, vocab_dict=vocab_dict)
