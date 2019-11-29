import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from operator import itemgetter

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.RNN_STM import get_X_and_Y_data_from_seq, seq_items_per_class, spell_label_seqs
from tools.RNN_STM import word_letter_combo_dict
from tools.data import nick_read_csv, find_path_to_dir
from tools.network import loop_thru_acts




def simple_plot_rnn(gha_dict_path,
                    plot_what='all',
                    measure='b_sel',
                    letter_sel=False,
                    correct_items_only=True,
                    verbose=False, test_run=False, 
                    show_plots=False):
    """
    
    :param gha_dict_path: or gha_dict
    :param plot_what: 'all' or 'highlights' or dict[layer_names][units][timesteps] 
    :param measure: selectivity measure to focus on if hl_dict provided
    :param letter_sel: focus on level of words or letters
    :param correct_items_only: remove items that were incorrect
    :param verbose:
    :param test_run: just 3 plots
    :param show_plots: 

    :return: 
    """



    print("\n**** running simple_plot_rnn() ****")


    if os.path.isfile(gha_dict_path):
        # # use gha-dict_path to get exp_cond_gha_path, gha_dict_name,
        exp_cond_gha_path, gha_dict_name = os.path.split(gha_dict_path)
        os.chdir(exp_cond_gha_path)

        # # part 1. load dict from study (should run with sim, GHA or sel dict)
        gha_dict = load_dict(gha_dict_path)

    elif type(gha_dict_path) is dict:
        gha_dict = gha_dict_path
        exp_cond_gha_path = os.getcwd()

    else:
        raise FileNotFoundError(gha_dict_path)

    if verbose:
        focussed_dict_print(gha_dict, 'gha_dict')

    # get topic_info from dict
    output_filename = gha_dict["topic_info"]["output_filename"]
    if letter_sel:
        output_filename = f"{output_filename}_lett"

    # # where to save files
    plots_folder = 'plots'
    cond_name = gha_dict['topic_info']['output_filename']
    condition_path = find_path_to_dir(long_path=exp_cond_gha_path, target_dir=cond_name)
    plots_path = os.path.join(condition_path, plots_folder)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    # os.chdir(plots_path)


    if verbose:
        print(f"\noutput_filename: {output_filename}")
        print(f"plots_path (to save): {plots_path}")
        print(f"os.getcwd(): {os.getcwd()}")


    # # get data info from dict
    n_words = gha_dict["data_info"]["n_cats"]
    n_letters = gha_dict["data_info"]["X_size"]
    if verbose:
        print(f"the are {n_words} word classes")

    if letter_sel:
        n_letters = gha_dict['data_info']["X_size"]
        n_words = n_letters
        print(f"the are {n_letters} letters classes\nn_words now set as n_letters")

        letter_id_dict = load_dict(os.path.join(gha_dict['data_info']['data_path'],
                                                'letter_id_dict.txt'))
        print(f"\nletter_id_dict:\n{letter_id_dict}")




    # # get model info from dict
    model_dict = gha_dict['model_info']['config']
    if verbose:
        focussed_dict_print(model_dict, 'model_dict')

    timesteps = gha_dict['model_info']["overview"]["timesteps"]
    vocab_dict = load_dict(os.path.join(gha_dict['data_info']["data_path"],
                                        gha_dict['data_info']["vocab_dict"]))


    '''Part 2 - load y, sort out incorrect resonses'''
    print("\n\nPart 2: loading labels")
    # # load y_labels to go with hid_acts and item_correct for sequences
    if 'seq_corr_list' in gha_dict['GHA_info']['scores_dict']:
        n_seqs = gha_dict['GHA_info']['scores_dict']['n_seqs']
        n_seq_corr = gha_dict['GHA_info']['scores_dict']['n_seq_corr']
        n_incorrect = n_seqs - n_seq_corr

        test_label_seq_name = gha_dict['GHA_info']['y_data_path']
        seqs_corr = gha_dict['GHA_info']['scores_dict']['seq_corr_list']

        test_label_seqs = np.load(f"{test_label_seq_name}labels.npy")

        if verbose:
            print(f"test_label_seqs: {np.shape(test_label_seqs)}")
            print(f"seqs_corr: {np.shape(seqs_corr)}")
            print(f"n_seq_corr: {n_seq_corr}")

        if letter_sel:
            # # get 1hot item vectors for 'words' and 3 hot for letters
            '''Always use serial_recall True. as I want a separate 1hot vector for each item.
            Always use x_data_type 'local_letter_X' as I want 3hot vectors'''
            y_letters = []
            y_words = []
            for this_seq in test_label_seqs:
                get_letters, get_words = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                                   seq_line=this_seq,
                                                                   serial_recall=True,
                                                                   end_seq_cue=False,
                                                                   x_data_type='local_letter_X')
                y_letters.append(get_letters)
                y_words.append(get_words)

            y_letters = np.array(y_letters)
            y_words = np.array(y_words)
            if verbose:
                print(f"\ny_letters: {type(y_letters)}  {np.shape(y_letters)}")
                print(f"y_words: {type(y_words)}  {np.shape(y_words)}")


        y_df_headers = [f"ts{i}" for i in range(timesteps)]
        y_scores_df = pd.DataFrame(data=test_label_seqs, columns=y_df_headers)
        y_scores_df['full_model'] = seqs_corr
        if verbose:
            print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")


    # # if not sequence data, load y_labels to go with hid_acts and item_correct for items
    elif 'item_correct_name' in gha_dict['GHA_info']['scores_dict']:
        # # load item_correct (y_data)
        item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
        # y_df = pd.read_csv(item_correct_name)
        y_scores_df = nick_read_csv(item_correct_name)




    """# # get rid of incorrect items if required"""
    print("\n\nRemoving incorrect responses")
    # # # get values for correct/incorrect items (1/0 or True/False)
    item_correct_list = y_scores_df['full_model'].tolist()
    full_model_values = list(set(item_correct_list))

    correct_symbol = 1
    if len(full_model_values) != 2:
        TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")
    if 1 not in full_model_values:
        if True in full_model_values:
            correct_symbol = True
        else:
            TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")

    print(f"len(full_model_values): {len(full_model_values)}")
    print(f"correct_symbol: {correct_symbol}")

    # # i need to check whether this analysis should include incorrect items (True/False)
    gha_incorrect = gha_dict['GHA_info']['gha_incorrect']

    # # get item indeces for correct and incorrect items
    item_index = list(range(n_seq_corr))

    incorrect_items = []
    correct_items = []
    for index in range(len(item_correct_list)):
        if item_correct_list[index] == 0:
            incorrect_items.append(index)
        else:
            correct_items.append(index)
    if correct_items_only:
        item_index == correct_items

    if gha_incorrect:
        if correct_items_only:
            if verbose:
                print("\ngha_incorrect: True (I have incorrect responses)\n"
                      "correct_items_only: True (I only want correct responses)")
                print(f"remove {n_incorrect} incorrect from hid_acts & output using y_scores_df.")
                print("use y_correct for y_df")

            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_symbol]
            y_df = y_correct_df

            mask = np.ones(shape=len(seqs_corr), dtype=bool)
            mask[incorrect_items] = False
            test_label_seqs = test_label_seqs[mask]

            if letter_sel:
                y_letters = y_letters[mask]

        else:
            if verbose:
                print("\ngha_incorrect: True (I have incorrect responses)\n"
                      "correct_items_only: False (I want incorrect responses)")
                print("no changes needed - don't remove anything from hid_acts, output and "
                      "use y scores as y_df")
    else:
        if correct_items_only:
            if verbose:
                print("\ngha_incorrect: False (I only have correct responses)\n"
                      "correct_items_only: True (I only want correct responses)")
                print("no changes needed - don't remove anything from hid_acts or output.  "
                      "Use y_correct as y_df")
            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_symbol]
            y_df = y_correct_df
        else:
            if verbose:
                print("\ngha_incorrect: False (I only have correct responses)\n"
                      "correct_items_only: False (I want incorrect responses)")
                raise TypeError("I can not complete this as desried"
                                "change correct_items_only to True"
                                "for analysis  - don't remove anything from hid_acts, output and "
                                "use y scores as y_df")

            # correct_items_only = True

    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")
        print(f"\ntest_label_seqs: {np.shape(test_label_seqs)}")  # \n{test_label_seqs}")
        # if letter_sel:
        #     y_letters = np.asarray(y_letters)
        #     print(f"y_letters: {np.shape(y_letters)}")  # \n{test_label_seqs}")

    n_correct, timesteps = np.shape(test_label_seqs)
    corr_test_seq_name = f"{output_filename}_{n_correct}_corr_test_label_seqs.npy"
    np.save(corr_test_seq_name, test_label_seqs)
    corr_test_letters_name = 'not_processed_yet'
    if letter_sel:
        corr_test_letters_name = f"{output_filename}_{n_correct}_corr_test_letter_seqs.npy"
        np.save(corr_test_letters_name, y_letters)


    # # get items per class
    IPC_dict = seq_items_per_class(label_seqs=test_label_seqs, vocab_dict=vocab_dict)
    focussed_dict_print(IPC_dict, 'IPC_dict')
    corr_test_IPC_name = f"{output_filename}_{n_correct}_corr_test_IPC.pickle"
    with open(corr_test_IPC_name, "wb") as pickle_out:
        pickle.dump(IPC_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    # # how many times is each item represented at each timestep.
    word_p_class_p_ts = IPC_dict['word_p_class_p_ts']
    letter_p_class_p_ts = IPC_dict['letter_p_class_p_ts']
    
    for i in range(timesteps):
        n_words_p_ts = len(word_p_class_p_ts[f"ts{i}"].keys())
        n_letters_p_ts = len(letter_p_class_p_ts[f"ts{i}"].keys())

        print(f"ts{i}) words:{n_words_p_ts}/{n_words}\tletters: {n_letters_p_ts}/{n_letters}")
        # print(word_p_class_p_ts[f"ts{i}"].keys())

    # # sort plot_what
    print(f"\nplotting: {plot_what}")

    if type(plot_what) is str:
        if plot_what == 'all':
            hl_dict = dict()
        elif os.path.isfile(plot_what):
            hl_dict = load_dict(plot_what)
            """plot_what should be:\n
                    i. 'all'\n
                    ii. path to highlights dict\n
                    iii. highlights_dict\n
                    iv. dict with structure [layers][units][timesteps]"""

    elif type(plot_what) is dict:
        hl_dict = plot_what
    else:
        raise ValueError("plot_what should be\n"
                         "i. 'all'\n"
                         "ii. path to highlights dict\n"
                         "iii. highlights_dict\n"
                         "iv. dict with structure [layers][units][timesteps]")

    if hl_dict:
        focussed_dict_print(hl_dict, 'hl_dict')

    '''save results
    either make a new empty place to save.
    or load previous version and get the units I have already completed'''
    os.chdir(plots_path)


    '''
    part 3   - get gha for each unit
    '''
    loop_gha = loop_thru_acts(gha_dict_path=gha_dict_path,
                              correct_items_only=correct_items_only,
                              letter_sel=letter_sel,
                              verbose=verbose,
                              test_run=test_run
                              )

    test_run_counter = 0
    for index, unit_gha in enumerate(loop_gha):

        print(f"\nindex: {index}")

        # print(f"\n\n{index}:\n{unit_gha}\n")
        sequence_data = unit_gha["sequence_data"]
        y_1hot = unit_gha["y_1hot"]
        act_func = unit_gha["act_func"]
        layer_name = unit_gha["layer_name"]
        unit_index = unit_gha["unit_index"]
        timestep = unit_gha["timestep"]
        ts_name = f"ts{timestep}"
        item_act_label_array = unit_gha["item_act_label_array"]
        IPC_words = IPC_dict['word_p_class_p_ts'][ts_name]
        IPC_letters = IPC_dict['letter_p_class_p_ts'][ts_name]


        # # only plot units of interest according to hl dict
        if hl_dict:
            if layer_name not in hl_dict:
                print(f"{layer_name} not in hl_dict")
                continue
            if unit_index not in hl_dict[layer_name]:
                print(f"unit {unit_index} not in hl_dict[{layer_name}]")
                continue
            if ts_name not in hl_dict[layer_name][unit_index]:
                print(f"{ts_name} not in hl_dict[{layer_name}][{unit_index}]")
                continue

            unit_hl_info = [x for x in hl_dict[layer_name][unit_index][ts_name]
                            if x[0] == measure]
            if len(unit_hl_info) == 0:
                print(f"{measure} not in hl_dict[{layer_name}][{unit_index}][{ts_name}]")
                continue

            if 'ts_invar' in hl_dict[layer_name][unit_index]:
                if measure not in hl_dict[layer_name][unit_index]['ts_invar']:
                    print(f"{measure} not in hl_dict[{layer_name}][{unit_index}]['ts_invar']")
                    continue

            if test_run:
                if test_run_counter == 3:
                    break
                test_run_counter += 1

            unit_hl_info = list(unit_hl_info[0])

            print(f"plotting {layer_name} {unit_index} {ts_name} "
                  f"{unit_hl_info}")

            print(f"\nsequence_data: {sequence_data}")
            print(f"y_1hot: {y_1hot}")
            print(f"unit_index: {unit_index}")
            print(f"timestep: {timestep}")
            print(f"ts_name: {ts_name}")

            # # selective_for_what
            sel_idx = unit_hl_info[2]
            if letter_sel:
                sel_for = 'letter'
                sel_item = letter_id_dict[sel_idx]
            else:
                sel_for = 'word'
                sel_item = vocab_dict[sel_idx]['word']

            # # add in sel item
            unit_hl_info.insert(3, sel_item)

            # # change rank to int
            rank_str = unit_hl_info[4]
            unit_hl_info[4] = int(rank_str[5:])


            # hl_text = f'measure\tvalue\tclass\t{sel_for}\trank\n'
            hl_keys = ['measure: ', 'value: ', 'label: ', f'{sel_for}: ', 'rank: ']
            hl_text = ''
            for idx, info in enumerate(unit_hl_info):
                key = hl_keys[idx]
                str_info = str(info)
                # hl_text = ''.join([hl_text, str_info[1:-1], '\n'])
                hl_text = ''.join([hl_text, key, str_info, '\n'])

            print(f"\nhl_text: {hl_text}")

        else:
            print("no hl_dict")

        # #  make df
        this_unit_acts = pd.DataFrame(data=item_act_label_array,
                                      columns=['item', 'activation', 'label'])
        this_unit_acts_df = this_unit_acts.astype(
            {'item': 'int32', 'activation': 'float', 'label': 'int32'})



        if letter_sel:
            y_letters_1ts = np.array(y_letters[:, timestep])
            print(f"y_letters_1ts: {np.shape(y_letters_1ts)}")
            # print(f"y_letters_1ts: {y_letters_1ts}")


        # if test_run:
        # # get word ids to check results more easily.
        unit_ts_labels = this_unit_acts_df['label'].tolist()
        # print(f"unit_ts_labels:\n{unit_ts_labels}")

        seq_words_df = spell_label_seqs(test_label_seqs=np.asarray(unit_ts_labels),
                                        vocab_dict=vocab_dict, save_csv=False)
        seq_words_list = seq_words_df.iloc[:, 0].tolist()
        # print(f"seq_words_df:\n{seq_words_df}")
        this_unit_acts_df['words'] = seq_words_list
        # print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")


        # # get labels for selective item
        if letter_sel:
            sel_item_list = y_letters_1ts[:, sel_idx]
        else:
            sel_item_list = [1 if x == sel_item else 0 for x in seq_words_list]
        this_unit_acts_df['sel_item'] = sel_item_list



        # sort by ascending word labels
        this_unit_acts_df = this_unit_acts_df.sort_values(by='words', ascending=True)

        if verbose is True:
            print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n")
            print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")


        # # make simple plot
        title = f"{layer_name} unit{unit_index} {ts_name} (of {timesteps})"

        print(f"title: {title}")

        if hl_dict:
            # fig = plt.figure()
            # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
            # spotty_axis = plt.subplot(gs[0])
            # text_box = plt.subplot(gs[1])

            gridkw = dict(width_ratios=[2, 1])
            fig, (spotty_axis, text_box) = plt.subplots(1, 2, gridspec_kw=gridkw)
            spot_plot = sns.catplot(x='activation', y="words", hue='sel_item',
                                    data=this_unit_acts_df,
                                    ax=spotty_axis, orient='h', kind="strip",
                                    jitter=1, dodge=True, linewidth=.5,
                                    palette="Set2", marker="D", edgecolor="gray")  # , alpha=.25)
            text_box.text(0.0, -0.01, hl_text, fontsize=10, clip_on=False)
            text_box.axes.get_yaxis().set_visible(False)
            text_box.axes.get_xaxis().set_visible(False)
            text_box.patch.set_visible(False)
            text_box.axis('off')
            spotty_axis.get_legend().set_visible(False)
            spotty_axis.set_xlabel("Unit activations")
            fig.suptitle(title)
            plt.close()  # extra blank plot
        else:
            g = sns.catplot(x='activation', y="words", data=this_unit_acts_df,
                            orient='h', kind="strip",
                            jitter=1, dodge=True, linewidth=.5,
                            palette="Set2", marker="D", edgecolor="gray")  # , alpha=.25)
            plt.xlabel("Unit activations")
            plt.suptitle(title)
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])


        if letter_sel:
            save_name = f"{plots_path}/{output_filename}_{layer_name}_{unit_index}_{ts_name}_lett_spotty.png"
        else:
            save_name = f"{plots_path}/{output_filename}_{layer_name}_{unit_index}_{ts_name}_spotty.png"
        plt.savefig(save_name)
        if show_plots:
            plt.show()
        plt.close()


    print("\nend of simple_plot_rnn script")


###########################
def plot_all_units(sel_dict_path,
                    plot_what='all',
                    measure='b_sel',
                    letter_sel=True,
                    correct_items_only=True,
                    verbose=True, test_run=True,
                    show_plots=False):
    """

    given a cond name

        load dicts and other info (BOTH Letter and word sel dicts)
        choose sel per unit (letter if > 0, else word)

        specify grid shape
        loop thru units, appending to axis

        plot units vertically, timesteps horizontally
        e.g., unit 0: ts0, ts1, ts2, ts3, ts4, ts5, ts6, ts7


    :param sel_dict_path: or gha_dict
    :param plot_what: 'all' or 'highlights' or dict[layer_names][units][timesteps]
    :param measure: selectivity measure to focus on if hl_dict provided
    :param letter_sel: focus on level of words or letters
    :param correct_items_only: remove items that were incorrect
    :param verbose:
    :param test_run: just 9 plots
    :param show_plots:

    :return:
    """

    print("\n**** running plot_all_units() ****")

    if os.path.isfile(sel_dict_path):
        # # use gha-dict_path to get exp_cond_gha_path, gha_dict_name,
        exp_cond_gha_path, gha_dict_name = os.path.split(sel_dict_path)
        os.chdir(exp_cond_gha_path)

        # # part 1. load dict from study (should run with sim, GHA or sel dict)
        gha_dict = load_dict(sel_dict_path)

    elif type(sel_dict_path) is dict:
        gha_dict = sel_dict_path
        exp_cond_gha_path = os.getcwd()

    else:
        raise FileNotFoundError(sel_dict_path)

    if verbose:
        focussed_dict_print(gha_dict, 'gha_dict')

    # get topic_info from dict
    output_filename = gha_dict["topic_info"]["output_filename"]
    if letter_sel:
        output_filename = f"{output_filename}_lett"

    # # where to save files
    plots_folder = 'plots'
    cond_name = gha_dict['topic_info']['output_filename']
    condition_path = find_path_to_dir(long_path=exp_cond_gha_path, target_dir=cond_name)
    plots_path = os.path.join(condition_path, plots_folder)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    # os.chdir(plots_path)

    if verbose:
        print(f"\noutput_filename: {output_filename}")
        print(f"plots_path (to save): {plots_path}")
        print(f"os.getcwd(): {os.getcwd()}")

    # # get data info from dict
    n_words = gha_dict["data_info"]["n_cats"]
    n_letters = gha_dict["data_info"]["X_size"]
    if verbose:
        print(f"the are {n_words} word classes")

    if letter_sel:
        n_letters = gha_dict['data_info']["X_size"]
        n_words = n_letters
        print(f"the are {n_letters} letters classes\nn_words now set as n_letters")

        letter_id_dict = load_dict(os.path.join(gha_dict['data_info']['data_path'],
                                                'letter_id_dict.txt'))
        print(f"\nletter_id_dict:\n{letter_id_dict}")

    # # get model info from dict
    model_dict = gha_dict['model_info']['config']
    if verbose:
        focussed_dict_print(model_dict, 'model_dict')

    timesteps = gha_dict['model_info']["overview"]["timesteps"]
    vocab_dict = load_dict(os.path.join(gha_dict['data_info']["data_path"],
                                        gha_dict['data_info']["vocab_dict"]))

    '''Part 2 - load y, sort out incorrect resonses'''
    print("\n\nPart 2: loading labels")
    # # load y_labels to go with hid_acts and item_correct for sequences
    if 'seq_corr_list' in gha_dict['GHA_info']['scores_dict']:
        n_seqs = gha_dict['GHA_info']['scores_dict']['n_seqs']
        n_seq_corr = gha_dict['GHA_info']['scores_dict']['n_seq_corr']
        n_incorrect = n_seqs - n_seq_corr

        test_label_seq_name = gha_dict['GHA_info']['y_data_path']
        seqs_corr = gha_dict['GHA_info']['scores_dict']['seq_corr_list']

        test_label_seqs = np.load(f"{test_label_seq_name}labels.npy")

        if verbose:
            print(f"test_label_seqs: {np.shape(test_label_seqs)}")
            print(f"seqs_corr: {np.shape(seqs_corr)}")
            print(f"n_seq_corr: {n_seq_corr}")

        if letter_sel:
            # # get 1hot item vectors for 'words' and 3 hot for letters
            '''Always use serial_recall True. as I want a separate 1hot vector for each item.
            Always use x_data_type 'local_letter_X' as I want 3hot vectors'''
            y_letters = []
            y_words = []
            for this_seq in test_label_seqs:
                get_letters, get_words = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
                                                                   seq_line=this_seq,
                                                                   serial_recall=True,
                                                                   end_seq_cue=False,
                                                                   x_data_type='local_letter_X')
                y_letters.append(get_letters)
                y_words.append(get_words)

            y_letters = np.array(y_letters)
            y_words = np.array(y_words)
            if verbose:
                print(f"\ny_letters: {type(y_letters)}  {np.shape(y_letters)}")
                print(f"y_words: {type(y_words)}  {np.shape(y_words)}")

        y_df_headers = [f"ts{i}" for i in range(timesteps)]
        y_scores_df = pd.DataFrame(data=test_label_seqs, columns=y_df_headers)
        y_scores_df['full_model'] = seqs_corr
        if verbose:
            print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")


    # # if not sequence data, load y_labels to go with hid_acts and item_correct for items
    elif 'item_correct_name' in gha_dict['GHA_info']['scores_dict']:
        # # load item_correct (y_data)
        item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
        # y_df = pd.read_csv(item_correct_name)
        y_scores_df = nick_read_csv(item_correct_name)

    """# # get rid of incorrect items if required"""
    print("\n\nRemoving incorrect responses")
    # # # get values for correct/incorrect items (1/0 or True/False)
    item_correct_list = y_scores_df['full_model'].tolist()
    full_model_values = list(set(item_correct_list))

    correct_symbol = 1
    if len(full_model_values) != 2:
        TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")
    if 1 not in full_model_values:
        if True in full_model_values:
            correct_symbol = True
        else:
            TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")

    print(f"len(full_model_values): {len(full_model_values)}")
    print(f"correct_symbol: {correct_symbol}")

    # # i need to check whether this analysis should include incorrect items (True/False)
    gha_incorrect = gha_dict['GHA_info']['gha_incorrect']

    # get item indeces for correct and incorrect items
    item_index = list(range(n_seq_corr))

    incorrect_items = []
    correct_items = []
    for index in range(len(item_correct_list)):
        if item_correct_list[index] == 0:
            incorrect_items.append(index)
        else:
            correct_items.append(index)
    if correct_items_only:
        item_index == correct_items

    if gha_incorrect:
        if correct_items_only:
            if verbose:
                print("\ngha_incorrect: True (I have incorrect responses)\n"
                      "correct_items_only: True (I only want correct responses)")
                print(f"remove {n_incorrect} incorrect from hid_acts & output using y_scores_df.")
                print("use y_correct for y_df")

            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_symbol]
            y_df = y_correct_df

            mask = np.ones(shape=len(seqs_corr), dtype=bool)
            mask[incorrect_items] = False
            test_label_seqs = test_label_seqs[mask]

            if letter_sel:
                y_letters = y_letters[mask]

        else:
            if verbose:
                print("\ngha_incorrect: True (I have incorrect responses)\n"
                      "correct_items_only: False (I want incorrect responses)")
                print("no changes needed - don't remove anything from hid_acts, output and "
                      "use y scores as y_df")
    else:
        if correct_items_only:
            if verbose:
                print("\ngha_incorrect: False (I only have correct responses)\n"
                      "correct_items_only: True (I only want correct responses)")
                print("no changes needed - don't remove anything from hid_acts or output.  "
                      "Use y_correct as y_df")
            y_correct_df = y_scores_df.loc[y_scores_df['full_model'] == correct_symbol]
            y_df = y_correct_df
        else:
            if verbose:
                print("\ngha_incorrect: False (I only have correct responses)\n"
                      "correct_items_only: False (I want incorrect responses)")
                raise TypeError("I can not complete this as desried"
                                "change correct_items_only to True"
                                "for analysis  - don't remove anything from hid_acts, output and "
                                "use y scores as y_df")

            # correct_items_only = True

    if verbose is True:
        print(f"\ny_df: {y_df.shape}\n{y_df.head()}")
        print(f"\ntest_label_seqs: {np.shape(test_label_seqs)}")  # \n{test_label_seqs}")
        # if letter_sel:
        #     y_letters = np.asarray(y_letters)
        #     print(f"y_letters: {np.shape(y_letters)}")  # \n{test_label_seqs}")

    n_correct, timesteps = np.shape(test_label_seqs)
    corr_test_seq_name = f"{output_filename}_{n_correct}_corr_test_label_seqs.npy"
    np.save(corr_test_seq_name, test_label_seqs)
    corr_test_letters_name = 'not_processed_yet'
    if letter_sel:
        corr_test_letters_name = f"{output_filename}_{n_correct}_corr_test_letter_seqs.npy"
        np.save(corr_test_letters_name, y_letters)

    # # get items per class
    IPC_dict = seq_items_per_class(label_seqs=test_label_seqs, vocab_dict=vocab_dict)
    focussed_dict_print(IPC_dict, 'IPC_dict')
    corr_test_IPC_name = f"{output_filename}_{n_correct}_corr_test_IPC.pickle"
    with open(corr_test_IPC_name, "wb") as pickle_out:
        pickle.dump(IPC_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    # # how many times is each item represented at each timestep.
    word_p_class_p_ts = IPC_dict['word_p_class_p_ts']
    letter_p_class_p_ts = IPC_dict['letter_p_class_p_ts']

    for i in range(timesteps):
        n_words_p_ts = len(word_p_class_p_ts[f"ts{i}"].keys())
        n_letters_p_ts = len(letter_p_class_p_ts[f"ts{i}"].keys())

        print(f"ts{i}) words:{n_words_p_ts}/{n_words}\tletters: {n_letters_p_ts}/{n_letters}")
        # print(word_p_class_p_ts[f"ts{i}"].keys())

    # # # sort plot_what
    # print(f"\nplotting: {plot_what}")
    # 
    # if type(plot_what) is str:
    #     if plot_what == 'all':
    #         hl_dict = dict()
    #     elif os.path.isfile(plot_what):
    #         hl_dict = load_dict(plot_what)
    #         """plot_what should be:\n
    #                 i. 'all'\n
    #                 ii. path to highlights dict\n
    #                 iii. highlights_dict\n
    #                 iv. dict with structure [layers][units][timesteps]"""
    # 
    # elif type(plot_what) is dict:
    #     hl_dict = plot_what
    # else:
    #     raise ValueError("plot_what should be\n"
    #                      "i. 'all'\n"
    #                      "ii. path to highlights dict\n"
    #                      "iii. highlights_dict\n"
    #                      "iv. dict with structure [layers][units][timesteps]")
    # 
    # if hl_dict:
    #     focussed_dict_print(hl_dict, 'hl_dict')

    # get max sel per unit for words or letters
    combo_dict = word_letter_combo_dict(sel_dict_path)
    focussed_dict_print(combo_dict, 'combo_dict')


    '''save results
    either make a new empty place to save.
    or load previous version and get the units I have already completed'''
    os.chdir(plots_path)

    print_nested_round_floats(gha_dict, 'gha_dict')
    n_units = gha_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable']

    test_run_value = 9
    max_rows = 20
    if test_run:
        n_rows = test_run_value//timesteps
        if test_run_value % timesteps > 0:
            n_rows = test_run_value//timesteps + 1
        n_plots = test_run_value
        n_pages = 1
    else:
        n_rows = max_rows
        n_plots = n_rows * timesteps
        n_pages = (n_units * timesteps)//n_plots

    print(f'\nn_rows: {n_rows}, n_plots: {n_plots}, n_pages: {n_pages}')



    '''
    part 3   - get gha for each unit
    '''
    loop_gha = loop_thru_acts(gha_dict_path=sel_dict_path,
                              correct_items_only=correct_items_only,
                              letter_sel=letter_sel,
                              verbose=verbose,
                              test_run=test_run
                              )


    # for page in range(n_pages):

    fig, axes = plt.subplots(nrows=n_rows, ncols=timesteps,
                             sharex=True, sharey=True)  # , squeeze=True)

    axes_zip = list(zip(range(1, n_plots + 1), axes.flatten()))

    for i in range(n_plots):
        this_idx = axes_zip[i][0]
        this_ax = axes_zip[i][1]
        print(this_idx, this_ax)

    fig.suptitle(f'{cond_name} All units & timesteps')

    test_run_counter = 0
    for index, unit_gha in enumerate(loop_gha):

        if unit_gha["unit_index"] != 0:
            continue

        print(f"\nindex: {index}")

        if test_run:
            if test_run_counter == test_run_value:
                break
            test_run_counter += 1


        sequence_data = unit_gha["sequence_data"]
        y_1hot = unit_gha["y_1hot"]
        act_func = unit_gha["act_func"]
        layer_name = unit_gha["layer_name"]
        unit_index = unit_gha["unit_index"]
        timestep = unit_gha["timestep"]
        ts_name = f"ts{timestep}"
        item_act_label_array = unit_gha["item_act_label_array"]
        IPC_words = IPC_dict['word_p_class_p_ts'][ts_name]
        IPC_letters = IPC_dict['letter_p_class_p_ts'][ts_name]




        # #  make df
        this_unit_acts = pd.DataFrame(data=item_act_label_array,
                                      columns=['item', 'activation', 'label'])
        this_unit_acts_df = this_unit_acts.astype(
            {'item': 'int32', 'activation': 'float', 'label': 'int32'})

        ax_idx = axes_zip[index][0]
        ax = axes_zip[index][1]
        print(f"\nax_idx: {ax_idx}: ax: {ax}")
        # print(f"\nax: {ax}")

        plot_index = (unit_index*timesteps + timestep + 1)

        print(f"\nsubplot: row{n_units} col{timesteps} #{plot_index}")

        # # for this unit - get sel stats from combo dict
        ts_dict = combo_dict[layer_name][unit_index][ts_name]
        sel_level = ts_dict['level']
        sel_value = round(ts_dict['sel'], 3)
        sel_feat = ts_dict['feat']
        print(f"sel_level: {sel_level}, sel_value: {sel_value}, sel_feat: {sel_feat}")

        # # get sel_feat
        # # selective_for_what
        sel_idx = sel_feat
        if sel_level == 'letter':
            sel_item = letter_id_dict[sel_feat]
        else:
            sel_item = vocab_dict[sel_feat]['word']

        if sel_level == 'letter':
            y_letters_1ts = np.array(y_letters[:, timestep])
            print(f"y_letters_1ts: {np.shape(y_letters_1ts)}")
            # # use this to just get a binary array of whether a letter is present?
            sel_item_list = y_letters_1ts[:, sel_idx]
        else:
            # # sort class label list
            class_labels = this_unit_acts['label'].to_list()

            # sel_item_list = [1 if x == sel_item else 0 for x in seq_words_list]
            sel_item_list = [1 if x == sel_feat else 0 for x in class_labels]

        this_unit_acts_df['sel_item'] = sel_item_list
        print(f"this_unit_acts_df:\n{this_unit_acts_df}")

        # I am going to need to get
        # 1. list of labels per items for words and letters - look at line 440
        # 2. vocab and letter ids - line 899 in rnn_sel



        # if sel_level == 'word':
        #     feat_list =
        #     , just use sel_feat.
        # else:
        #     y_letters_1ts = np.array(y_letters[:, timestep])
        #     print(f"y_letters_1ts: {np.shape(y_letters_1ts)}")
        #     # print(f"y_letters_1ts: {y_letters_1ts}")

        # # if letter level, I might have to go thrugh each word to see if that
        # letter in question is in each word.  use letter_in_seq

        # fig.add_subplot(n_units, timesteps, plot_index)

        sns.catplot(x='activation', y="label",
                    hue='sel_item',
                    data=this_unit_acts_df,
                    ax=ax,
                    orient='h', kind="strip",
                    jitter=1, dodge=True, linewidth=.5,

                    # palette="Set2", marker="D", edgecolor="gray"
                    )  # , alpha=.25)

        ax.set_xlim([0, 1])
        ax.set_yticks([])
        ax.set_title(f'u{unit_index}-{timestep}\n{sel_item}: {sel_value}', fontsize=8)
        ax.get_legend().set_visible(False)

        ax.set(xlabel='', ylabel='')

        # sort labels for left and bottom plots
        if index % timesteps == 0:
            ax.set_ylabel("Features")
        if index >= n_plots - timesteps:
            ax.set_xlabel("Activation")

        plt.close()


    # fig.tight_layout()

    # if n_pages == 1:
    save_name = f"{plots_path}/{output_filename}_all plots.png"
    # else:
    #     save_name = f"{plots_path}/{output_filename}_all plots_{page}of{n_pages}.png"
    plt.savefig(save_name)


    if show_plots:
        plt.show()

    print("\nend of plot_all_units script")



    #
    #     # # only plot units of interest according to hl dict
    #     if hl_dict:
    #         if layer_name not in hl_dict:
    #             print(f"{layer_name} not in hl_dict")
    #             continue
    #         if unit_index not in hl_dict[layer_name]:
    #             print(f"unit {unit_index} not in hl_dict[{layer_name}]")
    #             continue
    #         if ts_name not in hl_dict[layer_name][unit_index]:
    #             print(f"{ts_name} not in hl_dict[{layer_name}][{unit_index}]")
    #             continue
    #
    #         unit_hl_info = [x for x in hl_dict[layer_name][unit_index][ts_name]
    #                         if x[0] == measure]
    #         if len(unit_hl_info) == 0:
    #             print(f"{measure} not in hl_dict[{layer_name}][{unit_index}][{ts_name}]")
    #             continue
    #
    #         if 'ts_invar' in hl_dict[layer_name][unit_index]:
    #             if measure not in hl_dict[layer_name][unit_index]['ts_invar']:
    #                 print(f"{measure} not in hl_dict[{layer_name}][{unit_index}]['ts_invar']")
    #                 continue
    #
    #         if test_run:
    #             if test_run_counter == 3:
    #                 break
    #             test_run_counter += 1
    #
    #         unit_hl_info = list(unit_hl_info[0])
    #
    #         print(f"plotting {layer_name} {unit_index} {ts_name} "
    #               f"{unit_hl_info}")
    #
    #         print(f"\nsequence_data: {sequence_data}")
    #         print(f"y_1hot: {y_1hot}")
    #         print(f"unit_index: {unit_index}")
    #         print(f"timestep: {timestep}")
    #         print(f"ts_name: {ts_name}")
    #
    #         # # selective_for_what
    #         sel_idx = unit_hl_info[2]
    #         if letter_sel:
    #             sel_for = 'letter'
    #             sel_item = letter_id_dict[sel_idx]
    #         else:
    #             sel_for = 'word'
    #             sel_item = vocab_dict[sel_idx]['word']
    #
    #         # # add in sel item
    #         unit_hl_info.insert(3, sel_item)
    #
    #         # # change rank to int
    #         rank_str = unit_hl_info[4]
    #         unit_hl_info[4] = int(rank_str[5:])
    #
    #         # hl_text = f'measure\tvalue\tclass\t{sel_for}\trank\n'
    #         hl_keys = ['measure: ', 'value: ', 'label: ', f'{sel_for}: ', 'rank: ']
    #         hl_text = ''
    #         for idx, info in enumerate(unit_hl_info):
    #             key = hl_keys[idx]
    #             str_info = str(info)
    #             # hl_text = ''.join([hl_text, str_info[1:-1], '\n'])
    #             hl_text = ''.join([hl_text, key, str_info, '\n'])
    #
    #         print(f"\nhl_text: {hl_text}")
    #
    #     # #  make df
    #     this_unit_acts = pd.DataFrame(data=item_act_label_array,
    #                                   columns=['item', 'activation', 'label'])
    #     this_unit_acts_df = this_unit_acts.astype(
    #         {'item': 'int32', 'activation': 'float', 'label': 'int32'})
    #
    #     if letter_sel:
    #         y_letters_1ts = np.array(y_letters[:, timestep])
    #         print(f"y_letters_1ts: {np.shape(y_letters_1ts)}")
    #         # print(f"y_letters_1ts: {y_letters_1ts}")
    #
    #     # if test_run:
    #     # # get word ids to check results more easily.
    #     unit_ts_labels = this_unit_acts_df['label'].tolist()
    #     # print(f"unit_ts_labels:\n{unit_ts_labels}")
    #
    #     seq_words_df = spell_label_seqs(test_label_seqs=np.asarray(unit_ts_labels),
    #                                     vocab_dict=vocab_dict, save_csv=False)
    #     seq_words_list = seq_words_df.iloc[:, 0].tolist()
    #     # print(f"seq_words_df:\n{seq_words_df}")
    #     this_unit_acts_df['words'] = seq_words_list
    #     # print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")
    #
    #     # # get labels for selective item
    #     if letter_sel:
    #         sel_item_list = y_letters_1ts[:, sel_idx]
    #     else:
    #         sel_item_list = [1 if x == sel_item else 0 for x in seq_words_list]
    #     this_unit_acts_df['sel_item'] = sel_item_list
    #
    #     # sort by ascending word labels
    #     this_unit_acts_df = this_unit_acts_df.sort_values(by='words', ascending=True)
    #
    #     if verbose is True:
    #         print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n")
    #         print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")
    #
    #     # # make simple plot
    #     title = f"{layer_name} unit{unit_index} {ts_name} (of {timesteps})"
    #
    #     print(f"title: {title}")
    #
    #     if hl_dict:
    #         # fig = plt.figure()
    #         # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    #         # spotty_axis = plt.subplot(gs[0])
    #         # text_box = plt.subplot(gs[1])
    #
    #         gridkw = dict(width_ratios=[2, 1])
    #         fig, (spotty_axis, text_box) = plt.subplots(1, 2, gridspec_kw=gridkw)
    #         spot_plot = sns.catplot(x='activation', y="words", hue='sel_item',
    #                                 data=this_unit_acts_df,
    #                                 ax=spotty_axis, orient='h', kind="strip",
    #                                 jitter=1, dodge=True, linewidth=.5,
    #                                 palette="Set2", marker="D", edgecolor="gray")  # , alpha=.25)
    #         text_box.text(0.0, -0.01, hl_text, fontsize=10, clip_on=False)
    #         text_box.axes.get_yaxis().set_visible(False)
    #         text_box.axes.get_xaxis().set_visible(False)
    #         text_box.patch.set_visible(False)
    #         text_box.axis('off')
    #         spotty_axis.get_legend().set_visible(False)
    #         spotty_axis.set_xlabel("Unit activations")
    #         fig.suptitle(title)
    #         plt.close()  # extra blank plot
    #     else:
    #         g = sns.catplot(x='activation', y="words", data=this_unit_acts_df,
    #                         orient='h', kind="strip",
    #                         jitter=1, dodge=True, linewidth=.5,
    #                         palette="Set2", marker="D", edgecolor="gray")  # , alpha=.25)
    #         plt.xlabel("Unit activations")
    #         plt.suptitle(title)
    #         plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    #
    #     if letter_sel:
    #         save_name = f"{plots_path}/{output_filename}_{layer_name}_{unit_index}_{ts_name}_lett_spotty.png"
    #     else:
    #         save_name = f"{plots_path}/{output_filename}_{layer_name}_{unit_index}_{ts_name}_spotty.png"
    #     plt.savefig(save_name)
    #     if show_plots:
    #         plt.show()
    #     plt.close()
    #

# # # # PART 1 # # #
# sel_dict_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
#                 'STM_RNN/STM_RNN_test_v30_free_recall/test/all_generator_gha/test/' \
#                 'STM_RNN_test_v30_free_recall_GHA_dict.pickle'
#
# hl_path = '/home/nm13850/Documents/PhD/python_v2/experiments/' \
#           'STM_RNN/STM_RNN_test_v30_free_recall/correct_sel/test/' \
#           'STM_RNN_test_v30_free_recall_hl_units.pickle'
#
# plot_this = {'hid0': {1: {"ts2": ['reason for plotting can go here']},
#                       2: {'ts0': ['why did i want to plot this']}}}
#
# simple_plot_rnn(sel_dict_path,
#                 plot_what='all',
#                 show_plots=True,
#                 verbose=True, test_run=True)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.datasets import load_iris
#
# data = load_iris()
# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.subplots_adjust(hspace=0.5)
# fig.suptitle('Distributions of Iris Features')
#
# axis_zip = list(zip(range(1, 5), axes.flatten()))
# print(axis_zip)
# print(axis_zip[0])
# print(axis_zip[0][1])
#
# # for ax, feature, name in zip(axes.flatten(), data.data.T, data.feature_names):
# for i in range(4):
#     ax = axis_zip[i][1]
#     feature = data.data.T[i]
#     name = data.feature_names[i]
#     print(ax, feature, name)
#     sns.distplot(feature, ax=ax, bins=len(np.unique(data.data.T[0]))//2)
#     ax.set(title=name[:-4].upper(), xlabel='cm')
# plt.show()
