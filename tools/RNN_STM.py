import os
import numpy as np
import more_itertools
from tools.dicts import load_dict, focussed_dict_print



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
        x_data = [vocab_dict[item][x_data_type] for item in seq_line]

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

# # test get_X_and_Y_data_from_seq
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
# these_seqs = get_label_seqs(n_labels=3, seq_len=3, serial_recall=False, n_seqs=3)
# # these_seqs = [[0, 2, 3], [5, 3, 6], [0, 4, 2]]
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
    :param vocab_dict: Dict containing the codes for Y and Y data
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
    :param sequences_to_use: if None, generate sequences.  Else, add sequences here.

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
                    x_data_type='dist_letter_X',
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

    # todo: think about checking accuracy per item


    :param model: trained model
    :param data_dict: dict with relevant info including link to vocab dict
    :param test_label_seqs: sequence of labels to test on
    :param serial_recall: Type of recall
    :param x_data_type:
    :param end_seq_cue:
    :param batch_size:
    :param verbose:

    :return: scores_dict = {"n_seqs": n_seqs,
                             "mean_IoU": mean_IoU,
                             "prop_seq_corr": prop_seq_corr,
                            }
    """

    print("\n**** get_test_scores() ****")

    # # idiot check
    print(f"batch_size: {batch_size}")

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

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if verbose:
        print(f"\nx_test: {np.shape(x_test)}\ny_test: {np.shape(y_test)}\n"
              f"labels_test: {np.shape(labels_test)}")


    # # get class labels for predictions
    if serial_recall:
        # print("predicting classes")
        all_pred_labels = model.predict_classes(x_test, batch_size=batch_size, verbose=1)
        # print(f"all_pred_labels: {np.shape(all_pred_labels)}")

        # # sanitycheck
        pred_y_values = model.predict(x_test, batch_size=batch_size, verbose=verbose)


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

    if verbose:
        print(f"\nlabels_test: {np.shape(labels_test)}\n{labels_test}")
        print(f"\nall_pred_labels: {np.shape(all_pred_labels)}\n{all_pred_labels}")

        print("sanity check - first item")
        pred_round = [np.round(i, 2) for i in pred_y_values[0]]
        print(f"\n\n\nlabels_test: {labels_test[0]}\nall_pred_labels: {all_pred_labels[0]}\npred_round: {pred_round}\n\n\n")

    print("\nIoU acc")
    iou_scores = []
    for seq in range(n_seqs):

        true_labels = labels_test[seq, :]
        pred_labels = all_pred_labels[seq]

        # make set to get intersection and union
        pred_labels = set(pred_labels)

        intersection = pred_labels.intersection(true_labels)
        union = pred_labels.union(true_labels)

        IoU = len(intersection) / len(union)
        iou_scores.append(IoU)

        if verbose:
            # print(f"\n{seq}\nintersection: len({len(intersection)}): {intersection}\n"
            #       f"union: len({len(union)}):  {union}\n"
            #       f"IoU: {IoU}\niou_scores: {iou_scores}")

            print(f"\n{seq}: pred: {pred_labels} true: {true_labels} len(intersection): {len(intersection)} IoU: {IoU}")

    # get the average of all IoUs (per seq/batch etc
    mean_IoU = sum(iou_scores) / len(iou_scores)

    # # get prop of seqs where IoU == 1.0
    prop_seq_corr = iou_scores.count(1.0) / len(iou_scores)

    scores_dict = {"n_seqs": n_seqs,
                   "mean_IoU": mean_IoU,
                   "prop_seq_corr": prop_seq_corr,
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