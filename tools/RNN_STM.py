import os
import numpy as np
import itertools
import more_itertools
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats



def get_label_seqs(n_labels=30, seq_len=3, repetitions=False, n_seqs=1, cycles=False,
                   ):
    """
    Generate random sequences of labels for STM_RNN project.

    :param n_labels: number of labels to select from
    :param seq_len: Number of labels to select for each seq
    :param repetitions: if True a sequence may contain an item more than once e.g. [0, 1, 0]
        If False, items may only appear in each sequence once.
    :param n_seqs: number of sequences to generate.
    :param cycles: default=False: All seqs same len,
                    True: seqs of [1, 2, 3,... n]

    :return: 2d numpy array (n_seqs, seq_len)
    """
    class_list = np.arange(n_labels)

    sequences = []

    if cycles:
        max_len = seq_len
        seq_len_list = (list(range(1, max_len + 1)))
        seq_len_gen = itertools.cycle(seq_len_list)

        for s in range(n_seqs):
            this_len = next(seq_len_gen)
            # print(f'seq: {s}, this_len: {this_len}')

            if repetitions:
                # print('repetions')
                this_seq = more_itertools.random_product(class_list, repeat=this_len)
            else:
                # print('no repetions')
                if n_labels < max_len:
                    print(f"Can not produce seqs (max_len {max_len}) with no repetitions using only {n_labels} labels")
                this_seq = more_itertools.random_permutation(iterable=class_list, r=this_len)

            sequences.append(this_seq)

        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                               maxlen=max_len,
                                                               dtype='int32',
                                                               padding='post',
                                                               truncating='pre',
                                                               value=n_labels)
        # print(f"padded:\n{padded}")
        sequences = padded


    elif n_seqs == 1:
        if repetitions:
            sequences = more_itertools.random_product(class_list, repeat=seq_len)
        else:
            sequences = more_itertools.random_permutation(iterable=class_list, r=seq_len)
    else:
        for s in range(n_seqs):
            if repetitions:
                # print('repetions')
                this_seq = more_itertools.random_product(class_list, repeat=seq_len)
            else:
                # print('no repetions')
                this_seq = more_itertools.random_permutation(iterable=class_list, r=seq_len)
            sequences.append(this_seq)

    return np.array(sequences)


# # # test get_label_seqs
# test_get_label_seqs = get_label_seqs(n_labels=30, seq_len=8, repetitions=False, n_seqs=16,
#                                      cycles=True)
# print(f"\ntest_get_label_seqs: {type(test_get_label_seqs)}  {np.shape(test_get_label_seqs)}")
# print(test_get_label_seqs)


def get_X_and_Y_data_from_seq(vocab_dict,
                              seq_line,
                              serial_recall=False,
                              output_type='classes',
                              x_data_type='dist_letter_X',
                              end_seq_cue=False,
                              train_cycles=False,
                              pad_label=None,
                              ):
    """
    Take a single Y_Label_seq and return the corresponding X and Y data.

    :param vocab_dict: Dict containing the codes for Y and Y data
    :param seq_line: The sequence of y_labels to get the X and Y data for.
    :param x_data_type: 'local_word_X', 'local_letter_X', 'dist_letter_X'.
    :param serial_recall: if True, Y array is a list of vectors
                        If False, y-array is a single vector  e.g., activate all words simultaneously
    :param output_type: default 'classes': output units represent class labels
                        'letters' output units correspond to letters
    :param end_seq_cue: if True, add input unit which is activated for the last item in each seq
    :param train_cycles: default=False: All seqs same len,
                    True: seqs of [1, 2, 3,... n]
    :return: numpy arrays of X_data and y_data as specified in vocab
    """

    if type(output_type) is not str:
        raise ValueError(f"output_type should be string: 'classes', or 'letters'")
    if output_type is 'words':
        output_type = 'classes'
    elif output_type not in ['classes', 'letters']:
        raise ValueError(f"output_type should be string: 'classes', or 'letters'")

    if not serial_recall and output_type == 'letters':
        raise ValueError(f"You can not have serial_recall=False and output_type='letters'\n"
                         f"This would require reporting all letters in the list at a single timestep")

    # # X data
    if train_cycles:
        # # some items will be padded with class n_cats (e.g., 30), can be multiple pads per list
        # print(f"train_cycles: {train_cycles}")

        # figure out which is last item in the sequence
        if type(pad_label) is int:
            last_item = pad_label
            # print(f"last_item: {last_item}")
        else:
            raise ValueError(f"for train cycles enter int for pad_label which gives the class of the pad")

        # # make end_seq_x data
        x_size = len(vocab_dict[0][x_data_type])
        word_pad = [0] * x_size
        # print(f"word_pad: {word_pad}")

        x_data = []
        for index, item in enumerate(seq_line):
            if item in vocab_dict:
                this_word = vocab_dict[item][x_data_type]
            elif item == last_item:
                this_word = word_pad

            if end_seq_cue:
                # this doesn't give a specific cue at the end of the seq
                this_word = this_word + [0]

            x_data.append(this_word)


    elif end_seq_cue == True:
        # original code from before I started on cycles.
        # assumes a single last item which contains only the cue.
        # # add an additional unit which is only activated on the last item of each seq
        # # not sure this really words...
        # get last_item to know which is last item of seq
        last_item = len(seq_line) - 1
        # print(f"last_item: {last_item}")

        x_data = []
        for index, item in enumerate(seq_line):
            this_word = vocab_dict[item][x_data_type]

            if index < last_item:
                this_word = this_word + [0]
            else:
                this_word = this_word + [0]

            x_data.append(this_word)
    else:
        # x_data = [vocab_dict[item][x_data_type] for item in seq_line]
        x_data = []
        # print(f"line 87 seq_line: {np.shape(seq_line)} {type(seq_line)} {seq_line}")
        for item in seq_line:
            this_word = vocab_dict[item][x_data_type]
            x_data.append(this_word)

    # # Y data
    if serial_recall:
        # # output will be a list of vectors - regardless of contents (1hot or multilabel)
        # y_data = [vocab_dict[item]['local_word_X'] for item in seq_line]
        # get n_items
        n_items = max(list(vocab_dict.keys())) + 1

        y_data = []
        for index, item in enumerate(seq_line):

            if output_type is 'classes':
                if train_cycles:
                    n_items = n_items + 1
                # # 1hot class labels for each vector
                # make blank array of right length
                y_vector = [0]*n_items
                # change relevant items to 1
                y_vector[item] = 1
                y_data.append(y_vector)

            elif output_type is 'letters':
                # # use local-letter-x for output vectors (but with no end_seq_cue)
                if train_cycles:
                    if item in vocab_dict:
                        this_word = vocab_dict[item]['local_letter_X']
                    elif item == last_item:
                        this_word = word_pad
                else:
                    this_word = vocab_dict[item]['local_letter_X']

                if end_seq_cue:
                    # this doesn't give a specific cue at the end of the seq
                    this_word = this_word + [0]
                y_data.append(this_word)

    else:
        # # free-recall
        # get n_items
        n_items = max(list(vocab_dict.keys())) + 1

        # make blank array of right length
        y_data = [0]*n_items

        # change relevant items to 1
        for item in seq_line:
            y_data[item] = 1


    return np.array(x_data), np.array(y_data)

# # test get_X_and_Y_data_from_seq
# print("\ntest get_X_and_Y_data_from_seq")
# vocab_dict = load_dict('/home/nm13850/Documents/PhD/python_v2/datasets/RNN/bowers14_rep/vocab_30_dict.txt')
#
# # # single sequence
# # this_seq = [0, 1, 2]
# #
# # get_x, get_y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
# #                                          seq_line=this_seq,
# #                                          x_data_type='dist_letter_X',
# #                                          serial_recall=True,
# #                                          output_type='letters',
# #                                          end_seq_cue=True,
# #                                          train_cycles=3,
# #                                          )
# # # print(f"x: {get_x}\ny:{get_y}")
# # print(f"y:{get_y}")
#
# # # many sequences
# # # these_seqs = get_label_seqs(n_labels=3, seq_len=3, serial_recall=False, n_seqs=3)
# # these_seqs = [[0, 2, 3], [5, 3, 6], [0, 4, 2]]
# # # these_seqs = [[0], [2], [3], [5], [3], [6], [0], [4], [2]]
# n_cats = 30
# these_seqs = get_label_seqs(n_labels=n_cats, seq_len=8, repetitions=False, n_seqs=16,
#                                      cycles=True)
# print(f"these_seqs:\n{these_seqs}")
# x_seqs = []
# y_seqs = []
# # for index, this_seq in enumerate(these_seqs):
# for this_seq in these_seqs:
#
#     get_x, get_y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict, seq_line=this_seq,
#                                              serial_recall=True,
#                                              output_type='letters',
#                                              end_seq_cue=False,
#                                              train_cycles=True,
#                                              pad_label=n_cats)
#     x_seqs.append(get_x)
#     y_seqs.append(get_y)
#     print(f"\n: {this_seq}\nx:{get_x}\ny: {get_y}")
#
# # x_seqs = np.array(x_seqs)
# # print(f"\nx_seqs: {type(x_seqs)}  {np.shape(x_seqs)}")
# # print(x_seqs)
# # y_seqs = np.array(y_seqs)
# # print(f"\ny_seqs: {type(y_seqs)}  {np.shape(y_seqs)}")
# # print(y_seqs)



# make data generator
def generate_STM_RNN_seqs(data_dict,
                          seq_len,
                          batch_size=16,
                          repetitions=False,
                          serial_recall=False,
                          output_type='classes',
                          x_data_type='dist_letter_X',
                          end_seq_cue=False,
                          train_cycles=False,
                          pad_label=None,
                          verbose=False,
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
    :param serial_recall: if True, Y array is a list of vectors
                        If False, y-array is a single vector  e.g., activate all words simultaneously
    :param output_type: default 'classes': output units represent class labels
                        'letters' output units correspond to letters
    :param x_data_type: 'local_word_X', 'local_letter_X', 'dist_letter_X'.
                Note for 1hot Y data use local_word_X
    :param end_seq_cue: if True, add input unit which is activated for the last item in each seq
    :param train_cycles: if False, all lists lengths = timesteps.
                        If True, train on varying length, [1, 2, 3,...timesteps].
    :Yeild: X_array and Y_array
    """

    if verbose:
        print(f'\n*** running generate_STM_RNN_seqs() ***')
        print(f"seq_len={seq_len}\nbatch_size={batch_size}\nrepetitions={repetitions}\n"
              f"serial_recall={serial_recall}\noutput_type={output_type}\n"
              f"x_data_type={x_data_type}\nend_seq_cue={end_seq_cue}\ntrain_cycles: {train_cycles}")

    # load vocab dict
    vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))
    n_cats = data_dict['n_cats']
    class_list = list(range(n_cats))

    while True:      # this allows it to go on forever
        x_batch = []
        y_batch = []

        if train_cycles:
            # # generate a whole batch of seqs at once, then get x and y one-at-a-time
            pad_label = n_cats
            batch_of_seqs = get_label_seqs(n_labels=n_cats, seq_len=seq_len, repetitions=repetitions,
                                           n_seqs=batch_size,
                                           cycles=True)

        for items in range(batch_size):

            if train_cycles:
                this_seq = batch_of_seqs[items]
            else:
                this_seq = get_label_seqs(n_labels=n_cats, seq_len=seq_len, repetitions=repetitions)

            # get input and output data from vocab dict for this_seq
            get_X, get_Y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                     seq_line=this_seq,
                                                     x_data_type=x_data_type,
                                                     serial_recall=serial_recall,
                                                     output_type=output_type,
                                                     end_seq_cue=end_seq_cue,
                                                     train_cycles=train_cycles,
                                                     pad_label=pad_label,
                                                     )


            x_batch.append(get_X)
            y_batch.append(get_Y)

            # print(f"testing end seq cue")
            # print(f"get_X: {get_X}")
            # print(f"get_Y: {get_Y}")

            if verbose:
                print(f'\nthis seq: {this_seq}\n'
                      f'x_batch:\n{x_batch}\n'
                      f'y_batch:\n{y_batch}\n')

        # yeild returns a generator not an array.
        yield np.asarray(x_batch), np.asarray(y_batch)

######################################
# print("\ntest generate_STM_RNN_seqs")
#
# # data_dict_path = '/home/nm13850/Documents/PhD/python_v2/datasets/' \
# #                  'RNN/bowers14_rep/vocab_30_load_dict.txt'
# data_dict_path = '/home/nm13850/Documents/PhD/python_v2/datasets/' \
#                  'RNN/bowers14_rep/vocab_30_data_load_dict.txt'
# data_dict = load_dict(data_dict_path)
#
# generate_data = generate_STM_RNN_seqs(data_dict=data_dict,
#                                       seq_len=3,
#                                       batch_size=1,
#                                       serial_recall=False,
#                                       output_type='words',
#                                       x_data_type='dist_letter_X',
#                                       end_seq_cue=True,
#                                       train_cycles=False,
#                                       verbose=True,
#                                       )
# # test_get_label_seqs = get_label_seqs(n_labels=30, seq_len=8, repetitions=False, n_seqs=16,
# #                                      cycles=True)
# # print(f"\ntest_get_label_seqs: {type(test_get_label_seqs)}  {np.shape(test_get_label_seqs)}")
# # print(test_get_label_seqs)
# print("\ntesting generator for ten iterations")
#
# y_labels = []
# x_data = []
# y_data = []
#
# for i in range(3):
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

# make data generator
def generate_STM_seq2seqs(data_dict,
                          seq_len,
                          batch_size=16,
                          repetitions=False,
                          serial_recall=False,
                          output_type='classes',
                          x_data_type='dist_letter_X',
                          end_seq_cue=False,
                          train_cycles=False,
                          pad_label=None,
                          verbose=False,
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
    :param serial_recall: if True, Y array is a list of vectors
                        If False, y-array is a single vector  e.g., activate all words simultaneously
    :param output_type: default 'classes': output units represent class labels
                        'letters' output units correspond to letters
    :param x_data_type: 'local_word_X', 'local_letter_X', 'dist_letter_X'.
                Note for 1hot Y data use local_word_X
    :param end_seq_cue: if True, add input unit which is activated for the last item in each seq
    :param end_seq_cue: if True, add input unit which is activated for the last item in each seq
    :param train_cycles: if False, all lists lengths = timesteps.
                        If True, train on varying length, [1, 2, 3,...timesteps].
    :Yeild: X1_array, X2 and Y_array
    """

    if verbose:
        print(f'\n*** running def generate_STM_RNN_seqs() ***')
        print(f"seq_len={seq_len}\nbatch_size={batch_size}\nrepetitions={repetitions}\n"
              f"serial_recall={serial_recall}\noutput_type={output_type}\n"
              f"x_data_type={x_data_type}\nend_seq_cue={end_seq_cue}\ntrain_cycles: {train_cycles}")

    # load vocab dict
    vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))
    n_cats = data_dict['n_cats']
    class_list = list(range(n_cats))

    while True:      # this allows it to go on forever
        x_batch = []
        y_batch = []

        if train_cycles:
            # # generate a whole batch of seqs at once, then get x and y one-at-a-time
            pad_label = n_cats
            batch_of_seqs = get_label_seqs(n_labels=n_cats, seq_len=seq_len, repetitions=repetitions,
                                           n_seqs=batch_size,
                                           cycles=True)

        for items in range(batch_size):

            if train_cycles:
                this_seq = batch_of_seqs[items]
            else:
                this_seq = get_label_seqs(n_labels=n_cats, seq_len=seq_len, repetitions=repetitions)

            # get input and output data from vocab dict for this_seq
            get_X, get_Y = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                     seq_line=this_seq,
                                                     x_data_type=x_data_type,
                                                     serial_recall=serial_recall,
                                                     output_type=output_type,
                                                     end_seq_cue=end_seq_cue,
                                                     train_cycles=train_cycles,
                                                     pad_label=pad_label,
                                                     )

            print(f'get_X: {get_X}\n')
            list_len, x_len = np.shape(get_X)

            shifted_x = get_X[:-1]
            print(f'shifted_x: {shifted_x}')

            x2_cue = np.asarray([0] * x_len)
            # change relevant items to 1
            x2_cue[x_len-1] = 1

            print(f'x2_cue: {x2_cue}')

            # X2 = x2_cue + get_X[1:]
            X2 = np.vstack((x2_cue, shifted_x))
            print(f'X2: {X2}')



            x_batch.append(get_X)
            y_batch.append(get_Y)

            if verbose:
                print(f'\nthis seq: {this_seq}\n'
                      f'x_batch:\n{x_batch}\n'
                      f'y_batch:\n{y_batch}\n')

        # yeild returns a generator not an array.
        yield np.asarray(x_batch), np.asarray(y_batch)

######################################
# print("\ntest generate_STM_seq2seqs")
#
# # data_dict_path = '/home/nm13850/Documents/PhD/python_v2/datasets/' \
# #                  'RNN/bowers14_rep/vocab_30_load_dict.txt'
# data_dict_path = '/home/nm13850/Documents/PhD/python_v2/datasets/' \
#                  'RNN/bowers14_rep/vocab_30_data_load_dict.txt'
# data_dict = load_dict(data_dict_path)
#
# generate_data = generate_STM_seq2seqs(data_dict=data_dict,
#                                       seq_len=3,
#                                       batch_size=1,
#                                       serial_recall=False,
#                                       output_type='words',
#                                       x_data_type='dist_letter_X',
#                                       end_seq_cue=True,
#                                       train_cycles=False,
#                                       verbose=True,
#                                       )
# # test_get_label_seqs = get_label_seqs(n_labels=30, seq_len=8, repetitions=False, n_seqs=16,
# #                                      cycles=True)
# # print(f"\ntest_get_label_seqs: {type(test_get_label_seqs)}  {np.shape(test_get_label_seqs)}")
# # print(test_get_label_seqs)
# print("\ntesting generator for ten iterations")
#
# y_labels = []
# x_data = []
# y_data = []
#
# for i in range(3):
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

# get test scores.
def get_test_scores(model, data_dict, test_label_seqs,
                    x_data_type='dist_letter_X',
                    serial_recall=False,
                    output_type='classes',
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

    if verbose:
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
                                                 output_type=output_type,
                                                 x_data_type=x_data_type,
                                                 end_seq_cue=end_seq_cue,
                                                 train_cycles=False,
                                                 pad_label=None,
                                                 )
        x_test.append(get_x)
        y_test.append(get_y)

    x_test = np.array(x_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    if verbose:
        print(f"\nlabels_test: {np.shape(labels_test)}\n{labels_test[0]}\n"
              f"\nx_test: {np.shape(x_test)}\n{x_test[0]}\n"
              f"\ny_test: {np.shape(y_test)}\n{y_test[0]}\n")

    # print(f"type(x_test): {type(x_test)}")
    # print(f"type(x_test[0][0][0]): {type(x_test[0][0][0])}")




    # # get class labels for predictions
    if serial_recall:
        # print("predicting classes")
        if output_type is 'letters':
            print(f"output_type: {output_type}\n")

            # print("predicting y_values")
            pred_y_values = model.predict(x_test, batch_size=batch_size, verbose=verbose)

            if verbose:
                print(f"\npred_y_values: {np.shape(pred_y_values)}\n{pred_y_values[0]}\n")

            # # get labels for classes where value is greater than .5
            all_pred_labels = []


            for index, seq in enumerate(pred_y_values):
                # print(f"\n{index}. seq: {seq}\n")
                these_pred_labels = []
                for idx, item in enumerate(seq):
                    # binarise at .5
                    bin_item = [1. if x > .5 else 0. for x in item]
                    print(f"\n\nbin_item: {bin_item}")

                    # print(f"index: {index}, idx: {idx}, item:\n{item}\nbin_item:\n{bin_item}\n")
                    true_item = y_test[index][idx]
                    print(f"true_item: {true_item}")

                    true_label = labels_test[index][idx]
                    print(f"true_label: {true_label}")


                    if np.array_equal(bin_item, true_item):
                        pred_label = true_item
                    elif sum(bin_item) == 0.0:
                        pred_label = -999
                    else:
                        pred_label = -true_label

                    print(f"pred_label: {pred_label}")

                    these_pred_labels.append(pred_label)

                    # print(f"these_pred_labels: \n{these_pred_labels}")


                all_pred_labels.append(these_pred_labels)




        else:
            print(f"output_type: {output_type}")

            all_pred_labels = model.predict_classes(x_test, batch_size=batch_size, verbose=1)


        # # sanitycheck
        # pred_y_values = model.predict(x_test, batch_size=batch_size, verbose=verbose)


    else:
        # print("predicting y_values")
        pred_y_values = model.predict(x_test, batch_size=batch_size, verbose=verbose)

        if verbose:
            print(f"pred_y_values: {np.shape(pred_y_values)}")

        # # get labels for classes where value is greater than .5
        all_pred_labels = []
        for seq in range(n_seqs):
            these_pred_labels = np.argwhere(pred_y_values[seq] > .5)

            # flatted predictions
            if len(np.shape(these_pred_labels)) > 1:
                these_pred_labels = np.ravel(these_pred_labels).tolist()
            all_pred_labels.append(these_pred_labels)

        # print(f"\npred_y_values: {np.shape(pred_y_values)}")
        # print(f"y_test: {np.shape(y_test)}")
        # for seq in range(n_seqs):
        #     print(f"\npred_y_values: {pred_y_values[seq]}")
        #     print(f"all_pred_labels: {all_pred_labels[seq]}")
        #     print(f"y_test: {y_test[seq]}")

    if verbose:
        print(f"all_pred_labels: {np.shape(all_pred_labels)}\n{all_pred_labels[0]}\n")
        print(f"y_test: {np.shape(y_test)}")

    if verbose:
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
    # print(f"n_seq_corr: {n_seq_corr}")
    # print(f"len(seq_corr_list): {len(seq_corr_list)}")
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
# timesteps = 7
# serial_recall = True
# x_data_type = 'dist_letter_X'
# output_type = 'letters'
# end_seq_cue = False
# batch_size = 16
# verbose=True
#
# from tensorflow.keras.models import load_model
# model = load_model("/Users/nickmartin/Documents/PhD/python_v2/experiments/"
#                    "test_b16/test_acc_25112019/training/test_acc_25112019_model.hdf5")
#
# # test_label_seqs = np.array([[0, 2, 3], [5, 3, 6], [0, 4, 2], [11, 12, 13]])
# #
# # test_label_seqs = get_label_seqs(n_labels=n_cats, seq_len=timesteps,
# #                                  serial_recall=serial_recall, n_seqs=batch_size)
# # data_path = '/Users/nickmartin/Documents/PhD/python_v2/datasets/RNN/bowers16_rep'
# test_seq_path = '/Users/nickmartin/Documents/PhD/python_v2/datasets/' \
#                 'RNN/bowers16_rep/seq7_v30_960_test_seq_labels.npy'
# test_label_seqs = np.load(test_seq_path)
#
# # # call get test accracy(serial_recall,
# test_score_dict = get_test_scores(model=model, data_dict=data_dict, test_label_seqs=test_label_seqs,
#                                   serial_recall=serial_recall,
#                                   x_data_type=x_data_type,
#                                   output_type=output_type,
#                                   end_seq_cue=end_seq_cue,
#                                   batch_size=batch_size,
#                                   verbose=verbose)
#
# print(test_score_dict)

##########################################
# # free_rec_acc
"""
Input is y_pred and y_true arrays.

If free recall these will be a single vector per item

Covert y_pred to binary array where elements are greater than .5

convert y+pred and y-true to lists of indices where vector is 1.

Compare these to get IoU list

get mean of IoU list


"""

def free_rec_acc(y_true, y_pred, get_prop_corr=False):
    """
    1. Input is y_pred and y_true arrays.
        for free recall these will be a single vector per item
    2. Covert y_pred to binary array where elements are greater than .5
    3. convert y+pred and y-true to lists of indices where vector is 1.
    4. Compare these to get IoU list

    5. either: get mean of IoU list
                get proportion of items where IoU == 1.0

    :param y_true: array of correct class labels per sequence
    :param y_pred: array of predicted labels per sequence
    :param get_prop_corr: If False, return mean_IoU,
                        if True, return proportion of seqs where IoU == 1.0

    :return: accuracy (either mean_IoU or prop_corr)
    """

    if np.shape(y_true) != np.shape(y_pred):
        print(f"\ny_pred\ntype: {type(y_pred)}\n{y_pred}")
        print(f"\ny_true\ntype: {type(y_true)}\n{y_true}")
        raise ValueError(f"y_true ({np.shape(y_true)}) and y_pred ({np.shape(y_pred)}) should be same shape")

    n_seqs, n_cats = np.shape(y_pred)

    # # check shapes
    # print(f"\ny_pred: {np.shape(y_pred)}.  y_true: {np.shape(y_true)}")

    # # get labels for classes where value is greater than .5
    predicted_labels = []
    true_labels = []
    for seq in range(n_seqs):
        these_pred_labels = np.argwhere(y_pred[seq] > .5)
        these_true_labels = np.argwhere(y_true[seq] > .5)

        # flatted predictions
        if len(np.shape(these_pred_labels)) > 1:
            these_pred_labels = np.ravel(these_pred_labels).tolist()
        predicted_labels.append(these_pred_labels)

        # flatted predictions
        if len(np.shape(these_true_labels)) > 1:
            these_true_labels = np.ravel(these_true_labels).tolist()
        true_labels.append(these_true_labels)

    # # check labels
    # for seq in range(n_seqs):
        # print(f"\npredicted_labels: {predicted_labels[seq]}")
        # print(f"true_labels: {true_labels[seq]}")
    # print(f"\npredicted_labels: {predicted_labels}")
    # print(f"true_labels: {true_labels}")

    labels_test = true_labels

    iou_scores = []
    seq_corr_list = []
    count_corr = 0
    for seq in range(n_seqs):

        these_true_labels = labels_test[seq]
        # make set to get intersection and union
        these_pred_labels = set(predicted_labels[seq])

        intersection = these_pred_labels.intersection(these_true_labels)
        union = these_pred_labels.union(these_true_labels)

        IoU = len(intersection) / len(union)

        if get_prop_corr:
            if IoU == 1.0:
                count_corr += 1
        else:
            iou_scores.append(IoU)

        # if IoU == 1.0:
        #     seq_corr_list.append(int(1))
        # else:
        #     seq_corr_list.append(int(0))
        #
        # if verbose:
        #     print(f"{seq}: pred: {these_pred_labels} true: {these_true_labels} len(intersection): {len(intersection)} IoU: {IoU}")

    if get_prop_corr:
        # get proportion of seqs where IoU == 1.0
        accuracy = count_corr/n_seqs
    else:
        # get the average of all IoUs (per seq/batch etc
        accuracy = sum(iou_scores) / len(iou_scores)

    return accuracy

# ####################
# print("\nTesting free_rec_acc")
# # data_dict = load_dict('/home/nm13850/Documents/PhD/python_v2/datasets/'
# #                       'RNN/bowers14_rep/vocab_300_data_load_dict.txt')
# # vocab_dict = load_dict(os.path.join(data_dict['data_path'], data_dict['vocab_dict']))
#
# n_cats = 300
# timesteps = 3
# serial_recall = False
# x_data_type = 'dist_letter_X'
# end_seq_cue = False
# batch_size = 32
# verbose=True
#
# y_true = np.load('/home/nm13850/Documents/PhD/python_v2/ideas/y_true.npy')
# y_pred = np.load('/home/nm13850/Documents/PhD/python_v2/ideas/y_pred.npy')
#
# # # call get test accracy(serial_recall,
# IoU = free_rec_acc(y_true=y_true, y_pred=y_pred, get_prop_corr=False)
#
# print(f"\noutput of IoU: {IoU}")




####################


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
          # f"test_label_seqs: {np.shape(test_label_seqs)}\n"
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
        # print(f'row: {row}')
        # print(f'shape: {np.shape(row)}')
        # print(len(row))
        if type(row) is int:
            row = [row]

        new_row = []
        for item in row:
            spelled_word = vocab_dict[item]['letters']
            if letter in spelled_word:
                # print(spelled_word, letter)
                new_row.append(1)
            else:
                # print(spelled_word, )
                new_row.append(0)
        letter_present_list.append(new_row)

    letter_present_array = np.array(letter_present_list)

    return letter_present_array




def word_letter_combo_dict(sel_dict_path, measure='b_sel', save_combo_dict=True):
    '''
    This function combines the sel dicts for selectivity to words and letters
    into a single dict
    '''

    print('\n** word_letter_combo_dict() **')


    # 1. load sel dict
    if type(sel_dict_path) is dict:
        sel_dict = sel_dict_path
    elif type(sel_dict_path) is str:
        if os.path.isfile(sel_dict_path):
            sel_dict = load_dict(sel_dict_path)
    else:
        raise TypeError(f"Sel_dict path should be a dict or path to dict\n"
                        f"{sel_dict_path}")

    # focussed_dict_print(sel_dict)

    # 2. load sel_p_unit dicts for words and letters
    sel_path = sel_dict['sel_info']['sel_path']
    word_sel_dict_name = sel_dict['sel_info']['max_sel_dict_name']
    # print(f"word sel dict name; {word_sel_dict_name}")
    word_sel_dict = load_dict(os.path.join(sel_path, word_sel_dict_name))
    # focussed_dict_print(word_sel_dict, 'word_sel_dict')  #, focus_list=['hid0'])

    if word_sel_dict_name[-3:] == 'txt':
        word_sel_dict_prefix = word_sel_dict_name[:-18]
        word_sel_dict_suffix = word_sel_dict_name[-18:]
    elif word_sel_dict_name[-3:] == 'kle':
        word_sel_dict_prefix = word_sel_dict_name[:-21]
        word_sel_dict_suffix = word_sel_dict_name[-21:]

    letter_sel_dict_name = f"{word_sel_dict_prefix}lett_{word_sel_dict_suffix}"
    letter_sel_dict = load_dict(os.path.join(sel_path, letter_sel_dict_name))
    # focussed_dict_print(letter_sel_dict, 'letter_sel_dict')  #, focus_list=['hid0'])

    # print(f'\nidiot check\n'
    #       f'word_sel_dict inspection\n'
    #       f'word_sel_dict["hid0"].keys():\n {word_sel_dict["hid0"].keys()}\n\n'
    #       f'word_sel_dict["hid0"][0].keys():\n '
    #       f'{word_sel_dict["hid0"][0].keys()}\n\n'
    #       )

    # check if combo dict already exists
    combo_sel_dict_name = f"{word_sel_dict_prefix}combo_{word_sel_dict_suffix}"

    if combo_sel_dict_name[-6:] == 'pickle':
        combo_sel_dict_name = f'{combo_sel_dict_name[:-6]}txt'

    combo_sel_dict_path = os.path.join(sel_path, combo_sel_dict_name)

    if os.path.isfile(combo_sel_dict_path):
        print("Combo_dict already exists")
        combo_dict = load_dict(combo_sel_dict_path)
        return combo_dict

    combo_dict = dict()

    # 3. make new dict with same structure as unit_hl dicts but this one has:
    # keys: layer, unit, ts
    # values: letter/word, feature number, feature, sel_value

    for layer, units in word_sel_dict.items():
        # print(layer)
        combo_dict[layer] = dict()
        for unit, steps in units.items():
        #     print(unit, steps)
        #     print(f'\nsteps["ts0"]:\n{steps["ts0"]}')
        #     print(f'\nsteps["ts1"]:\n{steps["ts1"]}')

            combo_dict[layer][unit] = dict()
            for ts, scores in steps.items():
                word_sel_val = scores[measure]
                letter_sel_val = letter_sel_dict[layer][unit][ts][measure]
                letter_sel_class = letter_sel_dict[layer][unit][ts][f"{measure}_c"]
                word_sel_class = scores[f"{measure}_c"]

                max_sel_rule = letter_sel_val > 0

                if max_sel_rule:
                    combo = {'level': 'letter', 'sel': letter_sel_val, 'feat': letter_sel_class}
                else:
                    combo = {'level': 'word', 'sel': word_sel_val, 'feat': word_sel_class}

                combo_dict[layer][unit][ts] = combo
    # focussed_dict_print(combo_dict, 'combo_dict')

    # 4. return new dict

    return combo_dict

# sel_dict_path = '/Users/nickmartin/Documents/PhD/python_v2/experiments/' \
#                 'train_rnn_script_check/test_25112019/correct_sel/' \
#                 'test_25112019_sel_dict.pickle'
#
# combo_dict = word_letter_combo_dict(sel_dict_path)
#
# focussed_dict_print(combo_dict, 'combo_dict')



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
