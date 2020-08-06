import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import ptitprince as pt
from itertools import zip_longest

from tools.dicts import load_dict, focussed_dict_print, print_nested_round_floats
from tools.data import load_y_data, load_hid_acts, nick_read_csv, find_path_to_dir
from tools.network import loop_thru_acts

from tools.RNN_STM import get_X_and_Y_data_from_seq, seq_items_per_class, spell_label_seqs



# # or, if ptitprince is NOT installed
# from ptitprince import PtitPrince as pt


def simple_plot(gha_dict_path,
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

    print("\n**** running simple_plot() ****")

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
    # if letter_sel:
    #     output_filename = f"{output_filename}_lett"

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
    n_cats = gha_dict["data_info"]["n_cats"]
    X_size = gha_dict["data_info"]["X_size"]
    n_items = gha_dict['GHA_info']['scores_dict']['n_items']
    if verbose:
        print(f"the are {n_cats} classes")

    # if letter_sel:
    #     X_size = gha_dict['data_info']["X_size"]
    #     n_cats = X_size
    #     print(f"the are {X_size} letters classes\nn_cats now set as X_size")
    #
    #     letter_id_dict = load_dict(os.path.join(gha_dict['data_info']['data_path'],
    #                                             'letter_id_dict.txt'))
    #     print(f"\nletter_id_dict:\n{letter_id_dict}")

    # # get model info from dict
    model_dict = gha_dict['model_info']['config']
    hid_layers = gha_dict['model_info']['layers']['totals']['hid_layers']
    n_units = gha_dict['model_info']['layers']['hid_layers']['hid_totals']['analysable']

    if verbose:
        focussed_dict_print(model_dict, 'model_dict')

    if 'timesteps' in gha_dict['model_info']['overview']:
        timesteps = gha_dict['model_info']["overview"]["timesteps"]
    else:
        timesteps = 1
    # vocab_dict = load_dict(os.path.join(gha_dict['data_info']["data_path"],
    #                                     gha_dict['data_info']["vocab_dict"]))

    # get gha info from dict
    n_correct = gha_dict['GHA_info']['scores_dict']['n_correct']
    n_incorrect = n_items - n_correct

    '''Part 2 - load y, sort out incorrect resonses'''
    print("\n\nPart 2: loading labels")
    # # load y_labels to go with hid_acts and item_correct for sequences
    # if 'seq_corr_list' in gha_dict['GHA_info']['scores_dict']:
    #     n_seqs = gha_dict['GHA_info']['scores_dict']['n_seqs']
    #     n_seq_corr = gha_dict['GHA_info']['scores_dict']['n_seq_corr']
    #     n_incorrect = n_seqs - n_seq_corr
    #
    #     test_label_seq_name = gha_dict['GHA_info']['y_data_path']
    #     seqs_corr = gha_dict['GHA_info']['scores_dict']['seq_corr_list']
    #
    #     test_label_seqs = np.load(f"{test_label_seq_name}labels.npy")
    #
    #     if verbose:
    #         print(f"test_label_seqs: {np.shape(test_label_seqs)}")
    #         print(f"seqs_corr: {np.shape(seqs_corr)}")
    #         print(f"n_seq_corr: {n_seq_corr}")

        # if letter_sel:
        #     # # get 1hot item vectors for 'words' and 3 hot for letters
        #     '''Always use serial_recall True. as I want a separate 1hot vector for each item.
        #     Always use x_data_type 'local_letter_X' as I want 3hot vectors'''
        #     y_letters = []
        #     y_words = []
        #     for this_seq in test_label_seqs:
        #         get_letters, get_words = get_X_and_Y_data_from_seq(vocab_dict=vocab_dict,
        #                                                            seq_line=this_seq,
        #                                                            serial_recall=True,
        #                                                            end_seq_cue=False,
        #                                                            x_data_type='local_letter_X')
        #         y_letters.append(get_letters)
        #         y_words.append(get_words)
        #
        #     y_letters = np.array(y_letters)
        #     y_words = np.array(y_words)
        #     if verbose:
        #         print(f"\ny_letters: {type(y_letters)}  {np.shape(y_letters)}")
        #         print(f"y_words: {type(y_words)}  {np.shape(y_words)}")

        # y_df_headers = [f"ts{i}" for i in range(timesteps)]
        # y_scores_df = pd.DataFrame(data=test_label_seqs, columns=y_df_headers)
        # y_scores_df['full_model'] = seqs_corr
        # if verbose:
        #     print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")


    # # if not sequence data, load y_labels to go with hid_acts and item_correct for items
    # el
    if 'item_correct_name' in gha_dict['GHA_info']['scores_dict']:
        # # load item_correct (y_data)
        item_correct_name = gha_dict['GHA_info']['scores_dict']['item_correct_name']
        # y_df = pd.read_csv(item_correct_name)
        y_scores_df = nick_read_csv(item_correct_name)

        if verbose:
            print(f"\ny_scores_df: {y_scores_df.shape}\n{y_scores_df.head()}")


    """# # get rid of incorrect items if required"""
    print("\n\nRemoving incorrect responses")
    # # # get values for correct/incorrect items (1/0 or True/False)
    item_correct_list = y_scores_df['full_model'].tolist()
    full_model_values = list(set(item_correct_list))

    correct_symbol = 1
    if len(full_model_values) != 2:
        TypeError(f"TYPE_ERROR!: what are the scores/acc for items? {full_model_values}")

        if n_incorrect == 0:
            print("\nthere are no incorrect items so all responses are correct")
            correct_symbol = full_model_values[0]
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
    # item_index = list(range(n_seq_corr))
    item_index = list(range(gha_dict['data_info']['n_items']))


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

            mask = np.ones(shape=len(n_correct), dtype=bool)
            mask[incorrect_items] = False
            test_label_seqs = item_correct_list[mask]

            # if letter_sel:
            #     y_letters = y_letters[mask]

        else:
            y_df = y_scores_df
            test_label_seqs = item_correct_list
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
            # test_label_seqs = item_correct_list
            test_label_seqs = y_df['class']

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
        print(f"\ntest_label_seqs: {np.shape(test_label_seqs)}\n{test_label_seqs}")
        # if letter_sel:
        #     y_letters = np.asarray(y_letters)
        #     print(f"y_letters: {np.shape(y_letters)}")  # \n{test_label_seqs}")


    # # do I need to make test label seqs have shape n_correct, timesteps?
    # print(f"test_label_seqs shape\n{np.shape(test_label_seqs)}\n{test_label_seqs}")
    # reshaped_tls = np.reshape(test_label_seqs, (n_correct, timesteps))
    # print(f"reshaped_tls shape\n{np.shape(reshaped_tls)}\n{reshaped_tls}")

    # n_correct, timesteps = np.shape(test_label_seqs)
    corr_test_seq_name = f"{output_filename}_{n_correct}_corr_test_label_seqs.npy"
    np.save(corr_test_seq_name, test_label_seqs)
    corr_test_letters_name = 'not_processed_yet'
    # if letter_sel:
    #     corr_test_letters_name = f"{output_filename}_{n_correct}_corr_test_letter_seqs.npy"
    #     np.save(corr_test_letters_name, y_letters)

    # # get items per class
    # IPC_dict = seq_items_per_class(label_seqs=test_label_seqs, vocab_dict=vocab_dict)

    if correct_items_only:
        IPC_dict = gha_dict['GHA_info']['scores_dict']['corr_per_cat_dict']
    else:
        # if includes incorrect items
        if type(gha_dict['data_info']['items_per_cat']) is dict:
            IPC_dict = gha_dict['data_info']['items_per_cat']
        elif type(gha_dict['data_info']['items_per_cat']) is int:
            class_list = range(gha_dict['data_info']['n_cats'])
            items_per_cat = gha_dict['data_info']['items_per_cat']
            IPC_dict = {cat: items_per_cat for cat in class_list}

        else:
            print(f"Can not get IPC - check data type:\n"
                  f"{type(gha_dict['data_info']['items_per_cat'])}")


    focussed_dict_print(IPC_dict, 'IPC_dict')
    corr_test_IPC_name = f"{output_filename}_{n_correct}_corr_test_IPC.pickle"
    with open(corr_test_IPC_name, "wb") as pickle_out:
        pickle.dump(IPC_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

    # # how many times is each item represented at each timestep.
    # if timesteps > 1:
    # word_p_class_p_ts = IPC_dict['word_p_class_p_ts']
    # letter_p_class_p_ts = IPC_dict['letter_p_class_p_ts']
    #
    # for i in range(timesteps):
    #     n_words_p_ts = len(word_p_class_p_ts[f"ts{i}"].keys())
    #     n_letters_p_ts = len(letter_p_class_p_ts[f"ts{i}"].keys())
    #
    #     print(f"ts{i}) words:{n_words_p_ts}/{n_cats}\tletters: {n_letters_p_ts}/{X_size}")
    #     # print(word_p_class_p_ts[f"ts{i}"].keys())

    # # sort plot_what
    print(f"\nplotting: {plot_what}")

    if type(plot_what) is str:
        if plot_what == 'all':
            hl_dict = dict()

            # # # add model full model structure to hl_dict
            # if letter_sel:
            #     sel_per_unit_dict_path = f'{exp_cond_gha_path}/{cond_name}_lett_sel_per_unit.pickle'
            # else:
            sel_per_unit_dict_path = f'{exp_cond_gha_path}/{cond_name}_sel_per_unit.pickle'

            if os.path.isfile(sel_per_unit_dict_path):
                sel_per_unit_dict = load_dict(sel_per_unit_dict_path)

                for layer in list(sel_per_unit_dict.keys()):
                    hl_dict[layer] = dict()
                    for unit in sel_per_unit_dict[layer].keys():
                        hl_dict[layer][unit] = dict()
                        # for ts in sel_per_unit_dict[layer][unit].keys():
                        if measure in sel_per_unit_dict[layer][unit]:
                            class_sel_dict = sel_per_unit_dict[layer][unit][measure]
                            key_max = max(class_sel_dict, key=class_sel_dict.get)
                            val_max = class_sel_dict[key_max]
                            hl_entry = (measure, val_max, key_max, 'rank_1')
                            hl_dict[layer][unit] = list()
                            hl_dict[layer][unit].append(hl_entry)



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
    mean_activations = []


    loop_gha = loop_thru_acts(gha_dict_path=gha_dict_path,
                              correct_items_only=correct_items_only,
                              # letter_sel=letter_sel,
                              verbose=verbose,
                              test_run=test_run
                              )

    test_run_counter = 0
    for index, unit_gha in enumerate(loop_gha):

        print(f"\nindex: {index}")

        # print(f"\n\n{index}:\n{unit_gha}\n")
        sequence_data = unit_gha["sequence_data"]
        y_1hot = unit_gha["y_1hot"]
        layer_name = unit_gha["layer_name"]
        unit_index = unit_gha["unit_index"]
        timestep = unit_gha["timestep"]
        ts_name = f"ts{timestep}"
        item_act_label_array = unit_gha["item_act_label_array"]



        # print(f"check item_act_label_array\nI want class label not whether it was correct\n"
        #       f"{item_act_label_array}")

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

            # # list comp version fails so use for loop
            # unit_hl_info = [x for x in hl_dict[layer_name][unit_index][ts_name]
            #                 if x[0] == measure]
            unit_hl_info = []
            print('check line 377')
            for x in hl_dict[layer_name][unit_index][ts_name]:
                print(x)
                if x[0] == measure:
                    unit_hl_info.append(x)

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
            # if letter_sel:
            #     sel_for = 'letter'
            #     sel_item = letter_id_dict[sel_idx]
            # else:
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

        # if letter_sel:
        #     y_letters_1ts = np.array(y_letters[:, timestep])
        #     print(f"y_letters_1ts: {np.shape(y_letters_1ts)}")
        #     # print(f"y_letters_1ts: {y_letters_1ts}")

        # if test_run:
        # # get word ids to check results more easily.
        unit_ts_labels = this_unit_acts_df['label'].tolist()
        print(f"unit_ts_labels:\n{unit_ts_labels}")

        # seq_words_df = spell_label_seqs(test_label_seqs=np.asarray(unit_ts_labels),
        #                                 vocab_dict=vocab_dict, save_csv=False)
        # seq_words_list = seq_words_df.iloc[:, 0].tolist()
        # # print(f"seq_words_df:\n{seq_words_df}")
        # this_unit_acts_df['words'] = seq_words_list
        # # print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")
        #
        # # # get labels for selective item
        # # if letter_sel:
        # #     sel_item_list = y_letters_1ts[:, sel_idx]
        # #
        # # else:
        # sel_item_list = [1 if x == sel_item else 0 for x in seq_words_list]
        #
        # this_unit_acts_df['sel_item'] = sel_item_list
        #
        # # sort by ascending word labels
        # this_unit_acts_df = this_unit_acts_df.sort_values(by='words', ascending=True)

        # # get class labels
        this_unit_acts_df['class'] = test_label_seqs

        # # chec mean act
        acts_mean = np.mean(list(this_unit_acts_df['activation']))
        mean_activations.append(acts_mean)



        if verbose is True:
            print(f"\nthis_unit_acts_df: {this_unit_acts_df.shape}\n")
            print(f"this_unit_acts_df:\n{this_unit_acts_df.head()}")

        # # make simple plot
        title = f"{layer_name} unit{unit_index} {ts_name} (of {timesteps})"

        print(f"title: {title}")

        if hl_dict:
            gridkw = dict(width_ratios=[2, 1])
            fig, (spotty_axis, text_box) = plt.subplots(1, 2, gridspec_kw=gridkw)
            sns.catplot(x='activation', y="words", hue='sel_item',
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
            # sns.catplot(x='activation', y="words", data=this_unit_acts_df,
            sns.catplot(x='activation', y="class", data=this_unit_acts_df,
                        orient='h', kind="strip",
                        jitter=1, dodge=True, linewidth=.5,
                        palette="Set2", marker="D", edgecolor="gray")  # , alpha=.25)
            plt.xlabel("Unit activations")
            plt.suptitle(title)
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        # if letter_sel:
        #     save_name = f"{plots_path}/" \
        #                 f"{output_filename}_{layer_name}_{unit_index}_{ts_name}" \
        #                 f"_{measure}_lett.png"
        # else:
        save_name = f"{plots_path}/" \
                    f"{output_filename}_{layer_name}_{unit_index}_{ts_name}" \
                    f"_{measure}_word.png"

        plt.savefig(save_name)
        if show_plots:
            plt.show()
        plt.close()

    # # mean activations
    total_mean_act = np.mean(mean_activations)
    print(f"\n{cond_name}: total mean: {total_mean_act}\n{mean_activations}")


    print("\nend of simple_plot script")





"""based on raincloud_item_change_21022019.py"""

def raincloud_compare_dset(sel_dict_path,
              # lesion_dict_path,
              plot_type='classes',
              coi_measure='c_informed',
              top_layers='all',
              selected_units=False,
              plots_dir='simple_rain_plots',
              plot_fails=False,
              plot_class_change=False,
              normed_acts=False,
              layer_act_dist=False,
              verbose=False, test_run=False,
              ):
    """
    With visualise units with raincloud plot.  has distributions (cloud), individual activations (raindrops), boxplot
     to give median and interquartile range.  Also has plot of zero activations, scaled by class size.  Will show items
     that are affected by lesioning in different colours.

     I only have lesion data for [conv2d, dense] layers
     I have GHA and sel data from  [conv2d, activation, max_pooling2d, dense] layers

     so for each lesioned layer [conv2d, dense] I will use the following activation layer to take GHA and sel data from.

     Join these into groups using the activation number as the layer numbers.
     e.g., layer 1 (first conv layer) = conv2d_1 & activation_1.  layer 7 (first fc layer) = dense1 & activation 7)

    :param sel_dict_path:  path to selectivity dict
    # :param lesion_dict_path: path to lesion dict
    :param plot_type: all classes or OneVsAll.  if n_cats > 10, should automatically revert to oneVsAll.
    :param coi_measure: measure to use when choosing which class should be the coi.  Either the best performing sel
            measures (c_informed, c_ROC) or max class drop from lesioning.
    :param top_layers: if int, it will just do the top n layers (excluding output).  If not int, will do all layers.
    :param selected_units: default is to test all units on all layers.  But If I just want individual units, I should be
                    able to input a dict with layer names as keys and a list for each unit on that layer.
                    e.g., to just get unit 216 from 'fc_1' use selected_units={'fc_1': [216]}.
    :param plots_dir: where to save plots
    :param plot_fails: If False, just plots correct items, if true, plots items that failed after lesioning in RED
    :param plot_class_change: if True, plots proportion of items correct per class.
    :param normed_acts: if False use actual activation values, if True, normalize activations 0-1
    :param layer_act_dist: plot the distribution of all activations on a given layer.
                                This should already have been done in GHA
    :param verbose: how much to print to screen
    :param test_run: if True, just plot two units from two layers, if False, plot all (or selected units)

    returns nothings, just saves the plots
    """

    print("\n**** running visualise_units()****")

    if not selected_units:
        print(f"selected_units?: {selected_units}\n"
              "running ALL layers and units")
    else:
        print(focussed_dict_print(selected_units, 'selected_units'))
    # if type(selected_units) is dict:
    #     print("dict found")

    # # # lesion dict
    # # if lesion_dict_path is not None:
    # lesion_dict = load_dict(lesion_dict_path)
    # focussed_dict_print(lesion_dict, 'lesion_dict')
    #
    # # # get key_lesion_layers_list
    # lesion_info = lesion_dict['lesion_info']
    # lesion_path = lesion_info['lesion_path']
    # lesion_highlighs = lesion_info["lesion_highlights"]
    # key_lesion_layers_list = list(lesion_highlighs.keys())
    #
    # # # remove unnecesary items from key layers list
    # if 'highlights' in key_lesion_layers_list:
    #     key_lesion_layers_list.remove('highlights')
    # # if 'output' in key_lesion_layers_list:
    # #     key_lesion_layers_list.remove('output')
    # # if 'Output' in key_lesion_layers_list:
    # #     key_lesion_layers_list.remove('Output')
    #
    # # # remove output layers from key layers list
    # if any("utput" in s for s in key_lesion_layers_list):
    #     output_layers = [s for s in key_lesion_layers_list if "utput" in s]
    #     output_idx = []
    #     for out_layer in output_layers:
    #         output_idx.append(key_lesion_layers_list.index(out_layer))
    #     min_out_idx = min(output_idx)
    #     key_lesion_layers_list = key_lesion_layers_list[:min_out_idx]
    #
    # class_labels = list(lesion_dict['data_info']['cat_names'].values())
    #
    # # # sel_dict
    # sel_dict = load_dict(sel_dict_path)
    # if key_lesion_layers_list[0] in sel_dict['sel_info']:
    #     print('\nfound old sel dict layout')
    #     key_gha_sel_layers_list = list(sel_dict['sel_info'].keys())
    #     old_sel_dict = True
    #     # sel_info = sel_dict['sel_info']
    #     # short_sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0]['sel'].keys())
    #     # csb_list = list(sel_info[key_lesion_layers_list[0]][0]['class_sel_basics'].keys())
    #     # sel_measures_list = short_sel_measures_list + csb_list
    # else:
    #     print('\nfound NEW sel dict layout')
    #     old_sel_dict = False
    #     sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
    #     # sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0].keys())
    #     key_gha_sel_layers_list = list(sel_info.keys())
    #     # print(sel_info.keys())
    #
    # # # get key_gha_sel_layers_list
    # # # # remove unnecesary items from key layers list
    # # if 'sel_analysis_info' in key_gha_sel_layers_list:
    # #     key_gha_sel_layers_list.remove('sel_analysis_info')
    # # if 'output' in key_gha_sel_layers_list:
    # #     output_idx = key_gha_sel_layers_list.index('output')
    # #     key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]
    # # if 'Output' in key_gha_sel_layers_list:
    # #     output_idx = key_gha_sel_layers_list.index('Output')
    # #     key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]
    #
    # # # remove output layers from key layers list
    # if any("utput" in s for s in key_gha_sel_layers_list):
    #     output_layers = [s for s in key_gha_sel_layers_list if "utput" in s]
    #     output_idx = []
    #     for out_layer in output_layers:
    #         output_idx.append(key_gha_sel_layers_list.index(out_layer))
    #     min_out_idx = min(output_idx)
    #     key_gha_sel_layers_list = key_gha_sel_layers_list[:min_out_idx]
    #     # key_layers_df = key_layers_df.loc[~key_layers_df['name'].isin(output_layers)]
    #
    # # # put together lists of 1. sel_gha_layers, 2. key_lesion_layers_list.
    # n_activation_layers = sum("activation" in layers for layers in key_gha_sel_layers_list)
    # n_lesion_layers = len(key_lesion_layers_list)
    #
    # if n_activation_layers == n_lesion_layers:
    #     # # for models where activation and conv (or dense) are separate layers
    #     n_layers = n_activation_layers
    #     activation_layers = [layers for layers in key_gha_sel_layers_list if "activation" in layers]
    #     link_layers_dict = dict(zip(reversed(activation_layers), reversed(key_lesion_layers_list)))
    #
    # elif n_activation_layers == 0:
    #     print("\nno separate activation layers found - use key_lesion_layers_list")
    #     n_layers = len(key_lesion_layers_list)
    #     link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(key_lesion_layers_list)))
    #
    # else:
    #     print(f"n_activation_layers: {n_activation_layers}\n{key_gha_sel_layers_list}")
    #     print("n_lesion_layers: {n_lesion_layers}\n{key_lesion_layers_list}")
    #     raise TypeError('should be same number of activation layers and lesioned layers')
    #
    # if verbose is True:
    #     focussed_dict_print(link_layers_dict, 'link_layers_dict')
    #
    # # # # get info
    # exp_cond_path = sel_dict['topic_info']['exp_cond_path']
    # output_filename = sel_dict['topic_info']['output_filename']
    #
    # # # load data
    # # # check for training data
    # use_dataset = sel_dict['GHA_info']['use_dataset']
    #
    # n_cats = sel_dict['data_info']["n_cats"]
    #
    # if use_dataset in sel_dict['data_info']:
    #     # n_items = sel_dict["data_info"][use_dataset]["n_items"]
    #     items_per_cat = sel_dict["data_info"][use_dataset]["items_per_cat"]
    # else:
    #     # n_items = sel_dict["data_info"]["n_items"]
    #     items_per_cat = sel_dict["data_info"]["items_per_cat"]
    # if type(items_per_cat) is int:
    #     items_per_cat = dict(zip(list(range(n_cats)), [items_per_cat] * n_cats))
    #
    # if plot_type != 'OneVsAll':
    #     if n_cats > 20:
    #         plot_type = 'OneVsAll'
    #         print("\n\n\nWARNING!  There are lots of classes, it might make a messy plot"
    #               "Switching to OneVsAll\n")
    #
    # if sel_dict['GHA_info']['gha_incorrect'] == 'False':
    #     # # only gha for correct items
    #     # n_items = sel_dict['GHA_info']['scores_dict']['n_correct']
    #     items_per_cat = sel_dict['GHA_info']['scores_dict']['corr_per_cat_dict']
    #
    # # # load hid acts dict called hid_acts.pickle
    # """
    # Hid_acts dict has numbers as the keys for each layer.
    # Some layers (will be missing) as acts only recorded from some layers (e.g., [17, 19, 20, 22, 25, 26, 29, 30])
    # hid_acts_dict.keys(): dict_keys([0, 1, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 19, 20, 22, 25, 26, 29, 30])
    # hid_acts_dict[0].keys(): dict_keys(['layer_name', 'layer_class', 'layer_shape', '2d_acts', 'converted_to_2d'])
    # In each layer there is ['layer_name', 'layer_class', 'layer_shape', '2d_acts']
    # For 4d layers (conv, pool) there is also, key, value 'converted_to_2d': True
    # """
    #
    # # # check if I have saved the location to this file
    # hid_acts_pickle_name = sel_dict["GHA_info"]["hid_act_files"]['2d']
    # if 'gha_path' in sel_dict['GHA_info']:
    #     gha_path = sel_dict['GHA_info']['gha_path']
    #     hid_acts_path = os.path.join(gha_path, hid_acts_pickle_name)
    # else:
    #     hid_act_items = 'all'
    #     if not sel_dict['GHA_info']['gha_incorrect']:
    #         hid_act_items = 'correct'
    #
    #     gha_folder = f'{hid_act_items}_{use_dataset}_gha'
    #     hid_acts_path = os.path.join(exp_cond_path, gha_folder, hid_acts_pickle_name)
    # with open(hid_acts_path, 'rb') as pkl:
    #     hid_acts_dict = pickle.load(pkl)
    # print("\nopened hid_acts.pickle")
    #
    # # # # visualizing distribution of activations
    # # if layer_act_dist:
    # #     print("\nPlotting the distributions of activations for each layer")
    # #     for k, v in hid_acts_dict.items():
    # #         print("\nPlotting distribution of layer acts")
    # #         layer_act_dist_dir = 'layer_act_dist'
    # #         print(hid_acts_dict[k]['layer_name'])
    # #         hid_acts = hid_acts_dict[k]['2d_acts']
    # #         print(np.shape(hid_acts))
    # #         sns.distplot(np.ravel(hid_acts))
    # #         plt.title(str(hid_acts_dict[k]['layer_name']))
    # #         dist_plot_name = "{}_{}_layer_act_distplot.png".format(output_filename, hid_acts_dict[k]['layer_name'])
    # #         plt.savefig(os.path.join(plots_dir, layer_act_dist_dir, dist_plot_name))
    # #         # plt.show()
    # #         plt.close()
    #
    # # # dict to get the hid_acts_dict key for each layer based on its name
    # get_hid_acts_number_dict = dict()
    # for key, value in hid_acts_dict.items():
    #     hid_acts_layer_name = value['layer_name']
    #     hid_acts_layer_number = key
    #     get_hid_acts_number_dict[hid_acts_layer_name] = hid_acts_layer_number
    #
    # # # where to save files
    # save_plots_name = plots_dir
    # if plot_type is "OneVsAll":
    #     save_plots_name = f'{plots_dir}/{coi_measure}'
    # save_plots_dir = lesion_dict['GHA_info']['gha_path']
    # save_plots_path = os.path.join(save_plots_dir, save_plots_name)
    # if test_run:
    #     save_plots_path = os.path.join(save_plots_path, 'test')
    # if not os.path.exists(save_plots_path):
    #     os.makedirs(save_plots_path)
    # os.chdir(save_plots_path)
    # print(f"\ncurrent wd: {os.getcwd()}")
    #
    # if layer_act_dist:
    #     layer_act_dist_path = os.path.join(save_plots_path, 'layer_act_dist')
    #     if not os.path.exists(layer_act_dist_path):
    #         os.makedirs(layer_act_dist_path)
    #
    #
    # print("\n\n**********************"
    #       "\nlooping through layers"
    #       "\n**********************\n")
    #
    # for layer_index, (gha_layer_name, lesion_layer_name) in enumerate(link_layers_dict.items()):
    #
    #     if test_run:
    #         if layer_index > 2:
    #             continue
    #
    #     if type(top_layers) is int:
    #         if top_layers < n_activation_layers:
    #             if layer_index > top_layers:
    #                 continue
    #
    #
    #     # print(f"\nwhich units?: {selected_units}")
    #     # if selected_units != 'all':
    #     if selected_units is not False:
    #         if gha_layer_name not in selected_units:
    #             print(f"\nselected_units only, skipping layer {gha_layer_name}")
    #             continue
    #         else:
    #             print(f"\nselected_units only, from {gha_layer_name}")
    #             # print(f"\t{gha_layer_name} in {list(selected_units.keys())}")
    #             this_layer_units = selected_units[gha_layer_name]
    #             print(f"\trunning units: {this_layer_units}")
    #
    #     gha_layer_number = get_hid_acts_number_dict[gha_layer_name]
    #     layer_dict = hid_acts_dict[gha_layer_number]
    #
    #     if gha_layer_name != layer_dict['layer_name']:
    #         raise TypeError("gha_layer_name (from link_layers_dict) and layer_dict['layer_name'] should match! ")
    #     hid_acts_array = layer_dict['2d_acts']
    #     hid_acts_df = pd.DataFrame(hid_acts_array, dtype=float)
    #
    #     # # visualizing distribution of activations
    #     if layer_act_dist:
    #         hid_acts = layer_dict['2d_acts']
    #         print(f"\nPlotting distribution of activations {np.shape(hid_acts)}")
    #         sns.distplot(np.ravel(hid_acts))
    #         plt.title(f"{str(layer_dict['layer_name'])} activation distribution")
    #         dist_plot_name = "{}_{}_layer_act_distplot.png".format(output_filename, layer_dict['layer_name'])
    #         plt.savefig(os.path.join(layer_act_dist_path, dist_plot_name))
    #         if test_run:
    #             plt.show()
    #         plt.close()
    #
    #
    #     # # load item change details
    #     """# # four possible states
    #         full model      after_lesion    code
    #     1.  1 (correct)     0 (wrong)       -1
    #     2.  0 (wrong)       0 (wrong)       0
    #     3.  1 (correct)     1 (correct)     1
    #     4.  0 (wrong)       1 (correct)     2
    #
    #     """
    #     item_change_df = pd.read_csv(f"{lesion_path}/{output_filename}_{lesion_layer_name}_item_change.csv",
    #                                  header=0, dtype=int, index_col=0)
    #
    #     prop_change_df = pd.read_csv(f'{lesion_path}/{output_filename}_{lesion_layer_name}_prop_change.csv',
    #                                  header=0,
    #                                  # dtype=float,
    #                                  index_col=0)
    #
    #     if verbose:
    #         print("\n*******************************************"
    #               f"\n{layer_index}. gha layer {gha_layer_number}: {gha_layer_name} \tlesion layer: {lesion_layer_name}"
    #               "\n*******************************************")
    #         # focussed_dict_print(hid_acts_dict[layer_index])
    #         print(f"\n\thid_acts {gha_layer_name} shape: {hid_acts_df.shape}")
    #         print(f"\tloaded: {output_filename}_{lesion_layer_name}_item_change.csv: {item_change_df.shape}")
    #
    #     units_per_layer = len(hid_acts_df.columns)
    #
    #     print("\n\n\t**** loop through units ****")
    #     for unit_index, unit in enumerate(hid_acts_df.columns):
    #
    #         if test_run:
    #             if unit_index > 2:
    #                 continue
    #
    #         # if selected_units != 'all':
    #         if selected_units is not False:
    #             if unit not in this_layer_units:
    #                 # print(f"skipping unit {gha_layer_name} {unit}")
    #                 continue
    #             else:
    #                 print(f"\nrunning unit {gha_layer_name} {unit}")
    #
    #         # # check unit is in sel_per_unit_dict
    #         if unit in sel_info[gha_layer_name].keys():
    #             if verbose:
    #                 print("found unit in dict")
    #         else:
    #             print("unit not in dict\n!!!!!DEAD RELU!!!!!!!!\n...on to the next unit\n")
    #             continue
    #
    #         lesion_layer_and_unit = f"{lesion_layer_name}.{unit}"
    #         output_layer_and_unit = f"{lesion_layer_name}_{unit}"
    #
    #         print("\n\n*************\n"
    #               f"running layer {layer_index} of {n_layers} ({gha_layer_name}): unit {unit} of {units_per_layer}\n"
    #               "************")
    #
    #         # # make new df with just [item, hid_acts*, class, item_change*] *for this unit
    #         unit_df = item_change_df[["item", "class", lesion_layer_and_unit]].copy()
    #         # print(hid_acts_df)
    #         this_unit_hid_acts = hid_acts_df.loc[:, unit]
    #
    #
    #         # # check for dead relus
    #         if sum(np.ravel(this_unit_hid_acts)) == 0.0:
    #             print("\n\n!!!!!DEAD RELU!!!!!!!!...on to the next unit\n")
    #             continue
    #
    #         if verbose:
    #             print(f"\tnot a dead unit, hid acts sum: {sum(np.ravel(this_unit_hid_acts)):.2f}")
    #
    #         unit_df.insert(loc=1, column='hid_acts', value=this_unit_hid_acts)
    #         unit_df = unit_df.rename(index=str, columns={lesion_layer_and_unit: 'item_change'})
    #
    #         if verbose is True:
    #             print(f"\n\tall items - unit_df: {unit_df.shape}")
    #
    #         # # remove rows where network failed originally and after lesioning this unit - uninteresting
    #         old_df_length = len(unit_df)
    #         unit_df = unit_df.loc[unit_df['item_change'] != 0]
    #         if verbose is True:
    #             n_fail_fail = old_df_length - len(unit_df)
    #             print(f"\n\t{n_fail_fail} fail-fail items removed - new shape unit_df: {unit_df.shape}")
    #
    #         # # get items per class based on their occurences in the dataframe.
    #         # # this includes fail-pass, pass-pass and pass-fail - but not fail-fail
    #         no_fail_fail_ipc = unit_df['class'].value_counts(sort=False)
    #
    #         df_ipc = dict()
    #         for i in range(n_cats):
    #             df_ipc[i] = no_fail_fail_ipc[i]
    #
    #         # # # calculate the proportion of items that failed.
    #         # # # this is not the same as total_unit_change (which takes into account fail-pass as well as pass-fail)
    #         # df_ipc_total = sum(df_ipc.values())
    #         # l_failed_df = unit_df[(unit_df['item_change'] == -1)]
    #         # l_failed_count = len(l_failed_df)
    #         #
    #         # print("\tdf_ipc_total: {}".format(df_ipc_total))
    #         # print("\tl_failed_count: {}".format(l_failed_count))
    #
    #         # # getting max_class_drop
    #         max_class_drop_col = prop_change_df.loc[:, str(unit)]
    #         total_unit_change = max_class_drop_col['total']
    #         max_class_drop_col = max_class_drop_col.drop(labels=['total'])
    #         max_class_drop_val = max_class_drop_col.min()
    #         max_drop_class = max_class_drop_col.idxmin()
    #         print(f"\n\tmax_class_drop_val: {max_class_drop_val}\n"
    #               f"\tmax_drop_class: {max_drop_class}\n"
    #               f"\ttotal_unit_change: {total_unit_change}")
    #
    #         # # getting best sel measure (max_informed)
    #         main_sel_name = 'informedness'
    #
    #         # # includes if statement since some units have not score (dead relu?)
    #         if old_sel_dict:
    #             main_sel_val = sel_dict['sel_info'][gha_layer_name][unit]['max']['informed']
    #             main_sel_class = int(sel_dict['sel_info'][gha_layer_name][unit]['max']['c_informed'])
    #         else:
    #             # print(sel_info[gha_layer_name][unit]['max'])
    #             main_sel_val = sel_info[gha_layer_name][unit]['max']['max_informed']
    #             main_sel_class = int(sel_info[gha_layer_name][unit]['max']['max_informed_c'])
    #
    #         print(f"\tmain_sel_val: {main_sel_val}")
    #         print(f"\tmain_sel_class: {main_sel_class}")
    #
    #         # # coi stands for Class Of Interest
    #         # # if doing oneVsAll I need to have a coi measure. (e.g., clas with max informed 'c_informed')
    #         if plot_type is "OneVsAll":
    #
    #             # # get coi
    #             if coi_measure == 'max_class_drop':
    #                 coi = max_drop_class
    #             elif coi_measure == 'c_informed':
    #                 coi = main_sel_class
    #             else:
    #                 coi = int(sel_dict['sel_info'][gha_layer_name][unit]['max'][coi_measure])
    #             print(f"\n\tcoi: {coi}  ({coi_measure})")
    #
    #             # # get new class labels based on coi, OneVsAll
    #             all_classes_col = unit_df['class'].astype(int)
    #
    #             one_v_all_class_list = [1 if x is coi else 0 for x in all_classes_col]
    #             print(f"\tall_classes_col: {len(all_classes_col)}  one_v_all_class_list: {len(one_v_all_class_list)}")
    #
    #             if 'OneVsAll' not in list(unit_df):
    #                 print("\tadding 'OneVsAll'")
    #                 print("\treplacing all classes with 'OneVsAll'class column")
    #                 unit_df['class'] = one_v_all_class_list
    #
    #
    #         min_act = unit_df['hid_acts'].min()
    #
    #         if normed_acts:
    #             if min_act >= 0.0:
    #                 print("\nnormalising activations")
    #                 this_unit_normed_acts = np.divide(unit_df['hid_acts'], unit_df['hid_acts'].max())
    #                 unit_df['normed'] = this_unit_normed_acts
    #                 print(unit_df.head())
    #             else:
    #                 print("\ncan not do normed acts on this unit")
    #                 normed_acts = False
    #
    #
    #         # # # did any items fail that were previously at zero
    #         print(f"\n\tsmallest activation on this layer was {min_act}")
    #         l_failed_df = unit_df[(unit_df['item_change'] == -1)]
    #         l_failed_df = l_failed_df.sort_values(by=['hid_acts'])
    #
    #         min_failed_act = l_failed_df['hid_acts'].min()
    #         print(f"\n\tsmallest activation of items that failed after lesioning was {min_failed_act}")
    #         if min_failed_act == 0.0:
    #             fail_zero_df = l_failed_df.loc[l_failed_df['hid_acts'] == 0.0]
    #             fail_zero_count = len(fail_zero_df.index)
    #             print(f"\n\tfail_zero_df: {fail_zero_count} items\n\t{fail_zero_df.head()}")
    #             fail_zero_df.to_csv(f"{output_filename}_{gha_layer_name}_{unit}_fail_zero_df.csv", index=False)
    #
    #
    #         # # make plot of class changes
    #         # if plot_fails is True:
    #         if plot_class_change:
    #             class_prop_change = prop_change_df.iloc[:-1, unit].to_list()
    #             print(f"\n\tclass_prop_change: {class_prop_change}")
    #
    #             # change scale if there are big changes
    #             class_change_x_min = -.5
    #             if min(class_prop_change) < class_change_x_min:
    #                 class_change_x_min = min(class_prop_change)
    #
    #             class_change_x_max = .1
    #             if max(class_prop_change) > class_change_x_max:
    #                 class_change_x_max = max(class_prop_change)
    #
    #             class_change_curve = sns.barplot(x=class_prop_change, y=class_labels, orient='h')
    #             class_change_curve.set_xlim([class_change_x_min, class_change_x_max])
    #             class_change_curve.axvline(0, color="k", clip_on=False)
    #             plt.subplots_adjust(left=0.15)  # just to fit the label 'automobile' on
    #
    #             print(f'\nclass num: {class_prop_change.index(min(class_prop_change))}, '
    #                   f'class label: {class_labels[class_prop_change.index(min(class_prop_change))]}, '
    #                   f'class_val: {min(class_prop_change):.2f}'
    #                   )
    #
    #             plt.title(f"{lesion_layer_and_unit}\n"
    #                       f"total change: {total_unit_change:.2f} "
    #                       f"max_class ({class_labels[class_prop_change.index(min(class_prop_change))]}): "
    #                       f"{min(class_prop_change):.2f}")
    #             plt.savefig(f"{output_filename}_{output_layer_and_unit}_class_prop_change.png")
    #
    #             if test_run:
    #                 plt.show()
    #
    #             plt.close()

    sel_dict = load_dict(sel_dict_path)

    data_dir, cond_name = os.path.split(sel_dict_path)
    cond_name = cond_name[:-16]
    print(f'cond_name: {cond_name}')
    print(f'data_dir: {data_dir}')


    for layer in selected_units:
        # # cycle thru layers to print from
        print(f'layer: {layer}')
        for unit in selected_units[layer]:
            # cycle thru units to print from
            print(f'unit: {unit}')

            correct_y_label_path = str(sel_dict['GHA_info']['correct_Y_labels'])
            correct_y_label_path = os.path.join(data_dir, correct_y_label_path)
            print(f'correct_y_label_path: {correct_y_label_path}')

            correct_y_labels = pd.read_csv(correct_y_label_path, names=['item', 'class'],
                                           dtype='int32', )
            # correct_y_labels = load_y_data(correct_y_label_path)
            print(f'correct_y_labels:\n{correct_y_labels.head()}')

            gha_paths = sel_dict['GHA_info']['hid_act_files']
            if type(gha_paths) is str:
                gha_path = gha_paths
            elif type(gha_paths) is list:
                if len(gha_paths) == 1:
                    gha_path = gha_paths
                else:
                    gha_path = gha_paths[layer-1]

            gha_path = os.path.join(data_dir, gha_path)
            # hid_acts = np.genfromtxt(gha_path, delimiter=',')
            hid_acts = load_hid_acts(gha_path)

            print(f'gha: {np.shape(hid_acts)}\n{hid_acts}')

            # # # # # # # # # # # #
            # # raincloud plots # #
            # # # # # # # # # # # #

            # # # plot title
            if plot_fails:
                title = f"Layer: {gha_layer_name} Unit: {unit}\nmax_class_drop: {max_class_drop_val:.2f} " \
                        f"({max_drop_class}), total change: {total_unit_change:.2f}\n" \
                        f"{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"

                if plot_type == "OneVsAll":
                    title = f"Layer: {gha_layer_name} Unit: {unit} class: {coi}\n" \
                            f"max_class_drop: {max_class_drop_val:.2f} ({max_drop_class}), " \
                            f"total change: {total_unit_change:.2f}" \
                            "\n{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"
            else:
                title = f"Layer: {gha_layer_name} Unit: {unit}\n" \
                        f"{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"

                if plot_type == "OneVsAll":
                    title = f"Layer: {gha_layer_name} Unit: {unit} class: {coi}\n" \
                            f"{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"
            print(f"\ntitle:\n{title}")

            # # # load main dataframe
            raincloud_data = unit_df
            # print(raincloud_data.head())

            plot_y_vals = "class"
            # use_this_ipc = items_per_cat
            use_this_ipc = df_ipc

            if plot_type is "OneVsAll":
                print("\t\n\n\nUSE OneVsAll mode")
                n_cats = 2
                items_per_coi = use_this_ipc[coi]
                other_items = sum(df_ipc.values()) - items_per_coi
                use_this_ipc = {0: other_items, 1: items_per_coi}
                print(f"\tcoi {coi}, items_per_cat {items_per_cat}")

            # # # choose colours
            use_colours = 'tab10'
            if 10 < n_cats < 21:
                use_colours = 'tab20'
            elif n_cats > 20:
                print("\tERROR - more classes than colours!?!?!?")
            sns.set_palette(palette=use_colours, n_colors=n_cats)

            # Make MULTI plot
            fig = plt.figure(figsize=(10, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
            zeros_axis = plt.subplot(gs[0])
            rain_axis = plt.subplot(gs[1])

            # # # # # # # # # # # #
            # # make zeros plot # #
            # # # # # # # # # # # #

            # 1. get biggest class size (for max val of plot)
            max_class_size = max(use_this_ipc.values())
            print(f"\tmax_class_size: {max_class_size}")

            # 2. get list or dict of zeros per class
            zeros_dict = {}
            for k in range(n_cats):
                if plot_type is "OneVsAll":
                    plot_names = ["all_others", f"class_{coi}"]
                    this_name = plot_names[k]
                    this_class = unit_df.loc[unit_df['OneVsAll'] == k]
                    zero_count = 0 - (this_class['hid_acts'] == 0).sum()
                    zeros_dict[this_name] = zero_count
                else:
                    this_class = unit_df.loc[unit_df['class'] == k]
                    zero_count = 0 - (this_class['hid_acts'] == 0).sum()
                    zeros_dict[k] = zero_count

            # zd_classes = list(zeros_dict.keys())
            # zd_classes = list(lesion_dict['data_info']['cat_names'].values())
            zd_zero_count = list(zeros_dict.values())

            if verbose:
                print(f"\n\tzeros_dict:{zeros_dict.values()}, use_this_ipc:{use_this_ipc.values()}")

            zd_zero_perc = [x / y * 100 if y else 0 for x, y in zip(zeros_dict.values(), use_this_ipc.values())]

            zd_data = {"class": class_labels, "zero_count": zd_zero_count, "zero_perc": zd_zero_perc}

            zeros_dict_df = pd.DataFrame.from_dict(data=zd_data)

            # zero_plot
            sns.catplot(x="zero_perc", y="class", data=zeros_dict_df, kind="bar", orient='h', ax=zeros_axis)

            zeros_axis.set_xlabel("% at zero (height reflects n items)")

            zeros_axis.set_xlim([-100, 0])

            # # set width of bar to reflect class size
            new_heights = [x / max_class_size for x in use_this_ipc.values()]
            print(f"\tuse_this_ipc: {use_this_ipc}\n\tnew_heights: {new_heights}")

            # def change_height(zeros_axis, new_value):
            patch_count = 0
            for patch in zeros_axis.patches:
                current_height = patch.get_height()
                make_new_height = current_height * new_heights[patch_count]
                diff = current_height - make_new_height

                if new_heights[patch_count] < 1.0:
                    # print("{}. current_height {}, new_height: {}".format(patch, current_height, make_new_height))

                    # # change the bar height
                    patch.set_height(make_new_height)

                    # # recenter the bar
                    patch.set_y(patch.get_y() + diff * .65)

                patch_count = patch_count + 1


            zeros_axis.set_xticklabels(['100', '50', ''])
            # zeros_axis.xaxis.set_major_locator(plt.MaxNLocator(1))
            plt.close()

            # # # # # # # # #
            # # raincloud # #
            # # # # # # # # #

            data_values = "hid_acts"  # float
            if normed_acts:
                data_values = 'normed'
            data_class = plot_y_vals  # class
            orientation = "h"  # orientation

            # cloud_plot
            pt.half_violinplot(data=raincloud_data, bw=.1, linewidth=.5, cut=0., width=1, inner=None,
                               orient=orientation, x=data_values, y=data_class, scale="count")  # scale="area"

            """# # rain_drops - plot 3 separate plots so that they are interesting items are ontop of pass-pass
            # # zorder is order in which items are printed
            # # item_change: 1 ('grey') passed before and after lesioning
            # # -1 ('red') passed in full model but failed when lesioned
            # # 2 ('green') failed in full model but passed in lesioning"""
            fail_palette = {1: "silver", -1: "red", 2: "green", 0: "orange"}


            # # separate rain drops for pass pass,
            pass_pass_df = unit_df[(unit_df['item_change'] == 1)]
            pass_pass_drops = sns.stripplot(data=pass_pass_df, x=data_values, y=data_class, jitter=1, zorder=1,
                                            size=2, orient=orientation)  # , hue='item_change', palette=fail_palette)

            if plot_fails is True:

                '''I'm not using this atm, but if I want to plot items that originally failed and later passed'''
                # # separate raindrop for fail pass
                # fail_pass_df = unit_df[(unit_df['item_change'] == 2)]
                # if not fail_pass_df.empty:
                #     fail_pass_drops = sns.stripplot(data=fail_pass_df, x=data_values, y=data_class, jitter=1,
                #                                     zorder=3, size=4, orient=orientation, hue='item_change',
                #                                     palette=fail_palette, edgecolor='gray', linewidth=.4, marker='s',
                #                                     label='')

                # # separate raindrops for pass fail
                if not l_failed_df.empty:
                    # pass_fail_drops
                    sns.stripplot(data=l_failed_df, x=data_values, y=data_class, jitter=1, zorder=4, size=4,
                                  orient=orientation, hue='item_change', palette=fail_palette, edgecolor='white',
                                  linewidth=.4, marker='s')

            # box_plot
            sns.boxplot(data=raincloud_data, color="gray", orient=orientation, width=.15, x=data_values,
                        y=data_class, zorder=2, showbox=False,
                        # boxprops={'facecolor': 'none', "zorder": 2},
                        showfliers=False, showcaps=False,
                        whiskerprops={'linewidth': .01, "zorder": 2}, saturation=1,
                        # showwhiskers=False,
                        medianprops={'linewidth': .01, "zorder": 2},
                        showmeans=True,
                        meanprops={"marker": "*", "markerfacecolor": "white", "markeredgecolor": "black"}
                        )

            # # Finalize the figure
            rain_axis.set_xlabel("Unit activations")
            if normed_acts:
                rain_axis.set_xlabel("Unit activations (normalised)")

            # new_legend_text = ['l_passed', 'l_failed']
            new_legend_text = ['l_failed']

            leg = pass_pass_drops.axes.get_legend()
            if leg:
                # in here because leg is None if no items changed when this unit was lesioned
                for t, l in zip(leg.texts, new_legend_text):
                    t.set_text(l)

            # # hid ticks and labels from rainplot
            plt.setp(rain_axis.get_yticklabels(), visible=False)
            rain_axis.axes.get_yaxis().set_visible(False)

            # # put plots together
            max_activation = max(this_unit_hid_acts)
            min_activation = min(this_unit_hid_acts)
            if normed_acts:
                max_activation = max(this_unit_normed_acts)
                min_activation = min(this_unit_normed_acts)

            max_x_val = max_activation * 1.05
            layer_act_func = None
            for k, v in lesion_dict['model_info']['layers']['hid_layers'].items():
                if v['name'] == gha_layer_name:
                    layer_act_func = v['act_func']
                    break
            if layer_act_func in ['relu', 'Relu', 'ReLu']:
                min_x_val = 0
            elif min_activation > 0.0:
                min_x_val = 0
            else:
                min_x_val = min_activation

            rain_axis.set_xlim([min_x_val, max_x_val])
            rain_axis.get_shared_y_axes().join(zeros_axis, rain_axis)
            fig.subplots_adjust(wspace=0)

            fig.suptitle(title, fontsize=12).set_position([.5, 1.0])  # .set_bbox([])  #

            # # add y axis back onto rainplot
            plt.axvline(x=min_x_val, linestyle="-", color='black', )

            # # add marker for max informedness
            if 'info' in coi_measure:
                if old_sel_dict:
                    normed_info_thr = sel_dict['sel_info'][gha_layer_name][unit]['max']['thr_informed']
                else:
                    print(sel_info[gha_layer_name][unit]['max'])
                    normed_info_thr = sel_info[gha_layer_name][unit]['max']['max_info_thr']

                if normed_acts:
                    best_info_thr = normed_info_thr
                else:
                    # unnormalise it
                    best_info_thr = normed_info_thr * max(this_unit_hid_acts)
                print(f"\tbest_info_thr: {best_info_thr}")
                plt.axvline(x=best_info_thr, linestyle="--", color='grey')

            # sns.despine(right=True)

            if plot_type is "OneVsAll":
                plt.savefig(f"{output_filename}_{gha_layer_name}_{unit}_cat{coi}_raincloud.png")

            else:
                plt.savefig(f"{output_filename}_{gha_layer_name}_{unit}_raincloud.png")

            if test_run:
                plt.show()

            print("\n\tplot finished\n")

            # # clear for next round
            plt.close()

    # # plt.show()
    print("End of script")

########

def raincloud_w_fail(sel_dict_path, lesion_dict_path, plot_type='classes', coi_measure='c_informed', top_layers='all',
                     selected_units=False,
                     plots_dir='simple_rain_plots',
                     plot_fails=False,
                     plot_class_change=False,
                     normed_acts=False,
                     layer_act_dist=False,
                     verbose=False, test_run=False,
                     ):
    """
    With visualise units with raincloud plot.  has distributions (cloud), individual activations (raindrops), boxplot
     to give median and interquartile range.  Also has plot of zero activations, scaled by class size.  Will show items
     that are affected by lesioning in different colours.

     I only have lesion data for [conv2d, dense] layers
     I have GHA and sel data from  [conv2d, activation, max_pooling2d, dense] layers

     so for each lesioned layer [conv2d, dense] I will use the following activation layer to take GHA and sel data from.

     Join these into groups using the activation number as the layer numbers.
     e.g., layer 1 (first conv layer) = conv2d_1 & activation_1.  layer 7 (first fc layer) = dense1 & activation 7)

    :param sel_dict_path:  path to selectivity dict
    :param lesion_dict_path: path to lesion dict
    :param plot_type: all classes or OneVsAll.  if n_cats > 10, should automatically revert to oneVsAll.
    :param coi_measure: measure to use when choosing which class should be the coi.  Either the best performing sel
            measures (c_informed, c_ROC) or max class drop from lesioning.
    :param top_layers: if int, it will just do the top n mayers (excluding output).  If not int, will do all layers.
    :param selected_units: default is to test all units on all layers.  But If I just want individual units, I should be
                    able to input a dict with layer names as keys and a list for each unit on that layer.
                    e.g., to just get unit 216 from 'fc_1' use selected_units={'fc_1': [216]}.
    :param plots_dir: where to save plots
    :param plot_fails: If False, just plots correct items, if true, plots items that failed after lesioning in RED
    :param plot_class_change: if True, plots proportion of items correct per class.
    :param normed_acts: if False use actual activation values, if True, normalize activations 0-1
    :param layer_act_dist: plot the distribution of all activations on a given layer.
                                This should already have been done in GHA
    :param verbose: how much to print to screen
    :param test_run: if True, just plot two units from two layers, if False, plot all (or selected units)

    returns nothings, just saves the plots
    """

    print("\n**** running visualise_units()****")

    if not selected_units:
        print(f"selected_units?: {selected_units}\n"
              "running ALL layers and units")
    else:
        print(focussed_dict_print(selected_units, 'selected_units'))
    # if type(selected_units) is dict:
    #     print("dict found")

    # # lesion dict
    lesion_dict = load_dict(lesion_dict_path)
    focussed_dict_print(lesion_dict, 'lesion_dict')

    # # get key_lesion_layers_list
    lesion_info = lesion_dict['lesion_info']
    lesion_path = lesion_info['lesion_path']
    lesion_highlighs = lesion_info["lesion_highlights"]
    key_lesion_layers_list = list(lesion_highlighs.keys())

    # # remove unnecesary items from key layers list
    if 'highlights' in key_lesion_layers_list:
        key_lesion_layers_list.remove('highlights')
    # if 'output' in key_lesion_layers_list:
    #     key_lesion_layers_list.remove('output')
    # if 'Output' in key_lesion_layers_list:
    #     key_lesion_layers_list.remove('Output')

    # # remove output layers from key layers list
    if any("utput" in s for s in key_lesion_layers_list):
        output_layers = [s for s in key_lesion_layers_list if "utput" in s]
        output_idx = []
        for out_layer in output_layers:
            output_idx.append(key_lesion_layers_list.index(out_layer))
        min_out_idx = min(output_idx)
        key_lesion_layers_list = key_lesion_layers_list[:min_out_idx]

    class_labels = list(lesion_dict['data_info']['cat_names'].values())

    # # sel_dict
    sel_dict = load_dict(sel_dict_path)
    if key_lesion_layers_list[0] in sel_dict['sel_info']:
        print('\nfound old sel dict layout')
        key_gha_sel_layers_list = list(sel_dict['sel_info'].keys())
        old_sel_dict = True
        # sel_info = sel_dict['sel_info']
        # short_sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0]['sel'].keys())
        # csb_list = list(sel_info[key_lesion_layers_list[0]][0]['class_sel_basics'].keys())
        # sel_measures_list = short_sel_measures_list + csb_list
    else:
        print('\nfound NEW sel dict layout')
        old_sel_dict = False
        sel_info = load_dict(sel_dict['sel_info']['sel_per_unit_pickle_name'])
        # sel_measures_list = list(sel_info[key_lesion_layers_list[0]][0].keys())
        key_gha_sel_layers_list = list(sel_info.keys())
        # print(sel_info.keys())

    # # get key_gha_sel_layers_list
    # # # remove unnecesary items from key layers list
    # if 'sel_analysis_info' in key_gha_sel_layers_list:
    #     key_gha_sel_layers_list.remove('sel_analysis_info')
    # if 'output' in key_gha_sel_layers_list:
    #     output_idx = key_gha_sel_layers_list.index('output')
    #     key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]
    # if 'Output' in key_gha_sel_layers_list:
    #     output_idx = key_gha_sel_layers_list.index('Output')
    #     key_gha_sel_layers_list = key_gha_sel_layers_list[:output_idx]

    # # remove output layers from key layers list
    if any("utput" in s for s in key_gha_sel_layers_list):
        output_layers = [s for s in key_gha_sel_layers_list if "utput" in s]
        output_idx = []
        for out_layer in output_layers:
            output_idx.append(key_gha_sel_layers_list.index(out_layer))
        min_out_idx = min(output_idx)
        key_gha_sel_layers_list = key_gha_sel_layers_list[:min_out_idx]
        # key_layers_df = key_layers_df.loc[~key_layers_df['name'].isin(output_layers)]

    # # put together lists of 1. sel_gha_layers, 2. key_lesion_layers_list.
    n_activation_layers = sum("activation" in layers for layers in key_gha_sel_layers_list)
    n_lesion_layers = len(key_lesion_layers_list)

    if n_activation_layers == n_lesion_layers:
        # # for models where activation and conv (or dense) are separate layers
        n_layers = n_activation_layers
        activation_layers = [layers for layers in key_gha_sel_layers_list if "activation" in layers]
        link_layers_dict = dict(zip(reversed(activation_layers), reversed(key_lesion_layers_list)))

    elif n_activation_layers == 0:
        print("\nno separate activation layers found - use key_lesion_layers_list")
        n_layers = len(key_lesion_layers_list)
        link_layers_dict = dict(zip(reversed(key_lesion_layers_list), reversed(key_lesion_layers_list)))

    else:
        print(f"n_activation_layers: {n_activation_layers}\n{key_gha_sel_layers_list}")
        print("n_lesion_layers: {n_lesion_layers}\n{key_lesion_layers_list}")
        raise TypeError('should be same number of activation layers and lesioned layers')

    if verbose is True:
        focussed_dict_print(link_layers_dict, 'link_layers_dict')

    # # # get info
    exp_cond_path = sel_dict['topic_info']['exp_cond_path']
    output_filename = sel_dict['topic_info']['output_filename']

    # # load data
    # # check for training data
    use_dataset = sel_dict['GHA_info']['use_dataset']

    n_cats = sel_dict['data_info']["n_cats"]

    if use_dataset in sel_dict['data_info']:
        # n_items = sel_dict["data_info"][use_dataset]["n_items"]
        items_per_cat = sel_dict["data_info"][use_dataset]["items_per_cat"]
    else:
        # n_items = sel_dict["data_info"]["n_items"]
        items_per_cat = sel_dict["data_info"]["items_per_cat"]
    if type(items_per_cat) is int:
        items_per_cat = dict(zip(list(range(n_cats)), [items_per_cat] * n_cats))

    if plot_type != 'OneVsAll':
        if n_cats > 20:
            plot_type = 'OneVsAll'
            print("\n\n\nWARNING!  There are lots of classes, it might make a messy plot"
                  "Switching to OneVsAll\n")

    if sel_dict['GHA_info']['gha_incorrect'] == 'False':
        # # only gha for correct items
        # n_items = sel_dict['GHA_info']['scores_dict']['n_correct']
        items_per_cat = sel_dict['GHA_info']['scores_dict']['corr_per_cat_dict']

    # # load hid acts dict called hid_acts.pickle
    """
    Hid_acts dict has numbers as the keys for each layer.
    Some layers (will be missing) as acts only recorded from some layers (e.g., [17, 19, 20, 22, 25, 26, 29, 30])
    hid_acts_dict.keys(): dict_keys([0, 1, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 19, 20, 22, 25, 26, 29, 30])
    hid_acts_dict[0].keys(): dict_keys(['layer_name', 'layer_class', 'layer_shape', '2d_acts', 'converted_to_2d'])
    In each layer there is ['layer_name', 'layer_class', 'layer_shape', '2d_acts']
    For 4d layers (conv, pool) there is also, key, value 'converted_to_2d': True
    """

    # # check if I have saved the location to this file
    hid_acts_pickle_name = sel_dict["GHA_info"]["hid_act_files"]['2d']
    if 'gha_path' in sel_dict['GHA_info']:
        gha_path = sel_dict['GHA_info']['gha_path']
        hid_acts_path = os.path.join(gha_path, hid_acts_pickle_name)
    else:
        hid_act_items = 'all'
        if not sel_dict['GHA_info']['gha_incorrect']:
            hid_act_items = 'correct'

        gha_folder = f'{hid_act_items}_{use_dataset}_gha'
        hid_acts_path = os.path.join(exp_cond_path, gha_folder, hid_acts_pickle_name)
    with open(hid_acts_path, 'rb') as pkl:
        hid_acts_dict = pickle.load(pkl)
    print("\nopened hid_acts.pickle")

    # # # visualizing distribution of activations
    # if layer_act_dist:
    #     print("\nPlotting the distributions of activations for each layer")
    #     for k, v in hid_acts_dict.items():
    #         print("\nPlotting distribution of layer acts")
    #         layer_act_dist_dir = 'layer_act_dist'
    #         print(hid_acts_dict[k]['layer_name'])
    #         hid_acts = hid_acts_dict[k]['2d_acts']
    #         print(np.shape(hid_acts))
    #         sns.distplot(np.ravel(hid_acts))
    #         plt.title(str(hid_acts_dict[k]['layer_name']))
    #         dist_plot_name = "{}_{}_layer_act_distplot.png".format(output_filename, hid_acts_dict[k]['layer_name'])
    #         plt.savefig(os.path.join(plots_dir, layer_act_dist_dir, dist_plot_name))
    #         # plt.show()
    #         plt.close()

    # # dict to get the hid_acts_dict key for each layer based on its name
    get_hid_acts_number_dict = dict()
    for key, value in hid_acts_dict.items():
        hid_acts_layer_name = value['layer_name']
        hid_acts_layer_number = key
        get_hid_acts_number_dict[hid_acts_layer_name] = hid_acts_layer_number

    # # where to save files
    save_plots_name = plots_dir
    if plot_type is "OneVsAll":
        save_plots_name = f'{plots_dir}/{coi_measure}'
    save_plots_dir = lesion_dict['GHA_info']['gha_path']
    save_plots_path = os.path.join(save_plots_dir, save_plots_name)
    if test_run:
        save_plots_path = os.path.join(save_plots_path, 'test')
    if not os.path.exists(save_plots_path):
        os.makedirs(save_plots_path)
    os.chdir(save_plots_path)
    print(f"\ncurrent wd: {os.getcwd()}")

    if layer_act_dist:
        layer_act_dist_path = os.path.join(save_plots_path, 'layer_act_dist')
        if not os.path.exists(layer_act_dist_path):
            os.makedirs(layer_act_dist_path)


    print("\n\n**********************"
          "\nlooping through layers"
          "\n**********************\n")

    for layer_index, (gha_layer_name, lesion_layer_name) in enumerate(link_layers_dict.items()):

        if test_run:
            if layer_index > 2:
                continue

        if type(top_layers) is int:
            if top_layers < n_activation_layers:
                if layer_index > top_layers:
                    continue


        # print(f"\nwhich units?: {selected_units}")
        # if selected_units != 'all':
        if selected_units is not False:
            if gha_layer_name not in selected_units:
                print(f"\nselected_units only, skipping layer {gha_layer_name}")
                continue
            else:
                print(f"\nselected_units only, from {gha_layer_name}")
                # print(f"\t{gha_layer_name} in {list(selected_units.keys())}")
                this_layer_units = selected_units[gha_layer_name]
                print(f"\trunning units: {this_layer_units}")

        gha_layer_number = get_hid_acts_number_dict[gha_layer_name]
        layer_dict = hid_acts_dict[gha_layer_number]

        if gha_layer_name != layer_dict['layer_name']:
            raise TypeError("gha_layer_name (from link_layers_dict) and layer_dict['layer_name'] should match! ")
        hid_acts_array = layer_dict['2d_acts']
        hid_acts_df = pd.DataFrame(hid_acts_array, dtype=float)

        # # visualizing distribution of activations
        if layer_act_dist:
            hid_acts = layer_dict['2d_acts']
            print(f"\nPlotting distribution of activations {np.shape(hid_acts)}")
            sns.distplot(np.ravel(hid_acts))
            plt.title(f"{str(layer_dict['layer_name'])} activation distribution")
            dist_plot_name = "{}_{}_layer_act_distplot.png".format(output_filename, layer_dict['layer_name'])
            plt.savefig(os.path.join(layer_act_dist_path, dist_plot_name))
            if test_run:
                plt.show()
            plt.close()


        # # load item change details
        """# # four possible states
            full model      after_lesion    code
        1.  1 (correct)     0 (wrong)       -1
        2.  0 (wrong)       0 (wrong)       0
        3.  1 (correct)     1 (correct)     1
        4.  0 (wrong)       1 (correct)     2

        """
        item_change_df = pd.read_csv(f"{lesion_path}/{output_filename}_{lesion_layer_name}_item_change.csv",
                                     header=0, dtype=int, index_col=0)

        prop_change_df = pd.read_csv(f'{lesion_path}/{output_filename}_{lesion_layer_name}_prop_change.csv',
                                     header=0,
                                     # dtype=float,
                                     index_col=0)

        if verbose:
            print("\n*******************************************"
                  f"\n{layer_index}. gha layer {gha_layer_number}: {gha_layer_name} \tlesion layer: {lesion_layer_name}"
                  "\n*******************************************")
            # focussed_dict_print(hid_acts_dict[layer_index])
            print(f"\n\thid_acts {gha_layer_name} shape: {hid_acts_df.shape}")
            print(f"\tloaded: {output_filename}_{lesion_layer_name}_item_change.csv: {item_change_df.shape}")

        units_per_layer = len(hid_acts_df.columns)

        print("\n\n\t**** loop through units ****")
        for unit_index, unit in enumerate(hid_acts_df.columns):

            if test_run:
                if unit_index > 2:
                    continue

            # if selected_units != 'all':
            if selected_units is not False:
                if unit not in this_layer_units:
                    # print(f"skipping unit {gha_layer_name} {unit}")
                    continue
                else:
                    print(f"\nrunning unit {gha_layer_name} {unit}")

            # # check unit is in sel_per_unit_dict
            if unit in sel_info[gha_layer_name].keys():
                if verbose:
                    print("found unit in dict")
            else:
                print("unit not in dict\n!!!!!DEAD RELU!!!!!!!!\n...on to the next unit\n")
                continue

            lesion_layer_and_unit = f"{lesion_layer_name}.{unit}"
            output_layer_and_unit = f"{lesion_layer_name}_{unit}"

            print("\n\n*************\n"
                  f"running layer {layer_index} of {n_layers} ({gha_layer_name}): unit {unit} of {units_per_layer}\n"
                  "************")

            # # make new df with just [item, hid_acts*, class, item_change*] *for this unit
            unit_df = item_change_df[["item", "class", lesion_layer_and_unit]].copy()
            # print(hid_acts_df)
            this_unit_hid_acts = hid_acts_df.loc[:, unit]


            # # check for dead relus
            if sum(np.ravel(this_unit_hid_acts)) == 0.0:
                print("\n\n!!!!!DEAD RELU!!!!!!!!...on to the next unit\n")
                continue

            if verbose:
                print(f"\tnot a dead unit, hid acts sum: {sum(np.ravel(this_unit_hid_acts)):.2f}")

            unit_df.insert(loc=1, column='hid_acts', value=this_unit_hid_acts)
            unit_df = unit_df.rename(index=str, columns={lesion_layer_and_unit: 'item_change'})

            if verbose is True:
                print(f"\n\tall items - unit_df: {unit_df.shape}")

            # # remove rows where network failed originally and after lesioning this unit - uninteresting
            old_df_length = len(unit_df)
            unit_df = unit_df.loc[unit_df['item_change'] != 0]
            if verbose is True:
                n_fail_fail = old_df_length - len(unit_df)
                print(f"\n\t{n_fail_fail} fail-fail items removed - new shape unit_df: {unit_df.shape}")

            # # get items per class based on their occurences in the dataframe.
            # # this includes fail-pass, pass-pass and pass-fail - but not fail-fail
            no_fail_fail_ipc = unit_df['class'].value_counts(sort=False)

            df_ipc = dict()
            for i in range(n_cats):
                df_ipc[i] = no_fail_fail_ipc[i]

            # # # calculate the proportion of items that failed.
            # # # this is not the same as total_unit_change (which takes into account fail-pass as well as pass-fail)
            # df_ipc_total = sum(df_ipc.values())
            # l_failed_df = unit_df[(unit_df['item_change'] == -1)]
            # l_failed_count = len(l_failed_df)
            #
            # print("\tdf_ipc_total: {}".format(df_ipc_total))
            # print("\tl_failed_count: {}".format(l_failed_count))

            # # getting max_class_drop
            max_class_drop_col = prop_change_df.loc[:, str(unit)]
            total_unit_change = max_class_drop_col['total']
            max_class_drop_col = max_class_drop_col.drop(labels=['total'])
            max_class_drop_val = max_class_drop_col.min()
            max_drop_class = max_class_drop_col.idxmin()
            print(f"\n\tmax_class_drop_val: {max_class_drop_val}\n"
                  f"\tmax_drop_class: {max_drop_class}\n"
                  f"\ttotal_unit_change: {total_unit_change}")

            # # getting best sel measure (max_informed)
            main_sel_name = 'informedness'

            # # includes if statement since some units have not score (dead relu?)
            if old_sel_dict:
                main_sel_val = sel_dict['sel_info'][gha_layer_name][unit]['max']['informed']
                main_sel_class = int(sel_dict['sel_info'][gha_layer_name][unit]['max']['c_informed'])
            else:
                # print(sel_info[gha_layer_name][unit]['max'])
                main_sel_val = sel_info[gha_layer_name][unit]['max']['max_informed']
                main_sel_class = int(sel_info[gha_layer_name][unit]['max']['max_informed_c'])

            print(f"\tmain_sel_val: {main_sel_val}")
            print(f"\tmain_sel_class: {main_sel_class}")

            # # coi stands for Class Of Interest
            # # if doing oneVsAll I need to have a coi measure. (e.g., clas with max informed 'c_informed')
            if plot_type is "OneVsAll":

                # # get coi
                if coi_measure == 'max_class_drop':
                    coi = max_drop_class
                elif coi_measure == 'c_informed':
                    coi = main_sel_class
                else:
                    coi = int(sel_dict['sel_info'][gha_layer_name][unit]['max'][coi_measure])
                print(f"\n\tcoi: {coi}  ({coi_measure})")

                # # get new class labels based on coi, OneVsAll
                all_classes_col = unit_df['class'].astype(int)

                one_v_all_class_list = [1 if x is coi else 0 for x in all_classes_col]
                print(f"\tall_classes_col: {len(all_classes_col)}  one_v_all_class_list: {len(one_v_all_class_list)}")

                if 'OneVsAll' not in list(unit_df):
                    print("\tadding 'OneVsAll'")
                    print("\treplacing all classes with 'OneVsAll'class column")
                    unit_df['class'] = one_v_all_class_list


            min_act = unit_df['hid_acts'].min()

            if normed_acts:
                if min_act >= 0.0:
                    print("\nnormalising activations")
                    this_unit_normed_acts = np.divide(unit_df['hid_acts'], unit_df['hid_acts'].max())
                    unit_df['normed'] = this_unit_normed_acts
                    print(unit_df.head())
                else:
                    print("\ncan not do normed acts on this unit")
                    normed_acts = False


            # # # did any items fail that were previously at zero
            print(f"\n\tsmallest activation on this layer was {min_act}")
            l_failed_df = unit_df[(unit_df['item_change'] == -1)]
            l_failed_df = l_failed_df.sort_values(by=['hid_acts'])

            min_failed_act = l_failed_df['hid_acts'].min()
            print(f"\n\tsmallest activation of items that failed after lesioning was {min_failed_act}")
            if min_failed_act == 0.0:
                fail_zero_df = l_failed_df.loc[l_failed_df['hid_acts'] == 0.0]
                fail_zero_count = len(fail_zero_df.index)
                print(f"\n\tfail_zero_df: {fail_zero_count} items\n\t{fail_zero_df.head()}")
                fail_zero_df.to_csv(f"{output_filename}_{gha_layer_name}_{unit}_fail_zero_df.csv", index=False)


            # # make plot of class changes
            # if plot_fails is True:
            if plot_class_change:
                class_prop_change = prop_change_df.iloc[:-1, unit].to_list()
                print(f"\n\tclass_prop_change: {class_prop_change}")

                # change scale if there are big changes
                class_change_x_min = -.5
                if min(class_prop_change) < class_change_x_min:
                    class_change_x_min = min(class_prop_change)

                class_change_x_max = .1
                if max(class_prop_change) > class_change_x_max:
                    class_change_x_max = max(class_prop_change)

                class_change_curve = sns.barplot(x=class_prop_change, y=class_labels, orient='h')
                class_change_curve.set_xlim([class_change_x_min, class_change_x_max])
                class_change_curve.axvline(0, color="k", clip_on=False)
                plt.subplots_adjust(left=0.15)  # just to fit the label 'automobile' on

                print(f'\nclass num: {class_prop_change.index(min(class_prop_change))}, '
                      f'class label: {class_labels[class_prop_change.index(min(class_prop_change))]}, '
                      f'class_val: {min(class_prop_change):.2f}'
                      )

                plt.title(f"{lesion_layer_and_unit}\n"
                          f"total change: {total_unit_change:.2f} "
                          f"max_class ({class_labels[class_prop_change.index(min(class_prop_change))]}): "
                          f"{min(class_prop_change):.2f}")
                plt.savefig(f"{output_filename}_{output_layer_and_unit}_class_prop_change.png")

                if test_run:
                    plt.show()

                plt.close()



            # # # # # # # # # # # #
            # # raincloud plots # #
            # # # # # # # # # # # #

            # # # plot title
            if plot_fails:
                title = f"Layer: {gha_layer_name} Unit: {unit}\nmax_class_drop: {max_class_drop_val:.2f} " \
                        f"({max_drop_class}), total change: {total_unit_change:.2f}\n" \
                        f"{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"

                if plot_type == "OneVsAll":
                    title = f"Layer: {gha_layer_name} Unit: {unit} class: {coi}\n" \
                            f"max_class_drop: {max_class_drop_val:.2f} ({max_drop_class}), " \
                            f"total change: {total_unit_change:.2f}" \
                            "\n{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"
            else:
                title = f"Layer: {gha_layer_name} Unit: {unit}\n" \
                        f"{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"

                if plot_type == "OneVsAll":
                    title = f"Layer: {gha_layer_name} Unit: {unit} class: {coi}\n" \
                            f"{main_sel_name}: {main_sel_val:.2f} ({main_sel_class})"
            print(f"\ntitle:\n{title}")

            # # # load main dataframe
            raincloud_data = unit_df
            # print(raincloud_data.head())

            plot_y_vals = "class"
            # use_this_ipc = items_per_cat
            use_this_ipc = df_ipc

            if plot_type is "OneVsAll":
                print("\t\n\n\nUSE OneVsAll mode")
                n_cats = 2
                items_per_coi = use_this_ipc[coi]
                other_items = sum(df_ipc.values()) - items_per_coi
                use_this_ipc = {0: other_items, 1: items_per_coi}
                print(f"\tcoi {coi}, items_per_cat {items_per_cat}")

            # # # choose colours
            use_colours = 'tab10'
            if 10 < n_cats < 21:
                use_colours = 'tab20'
            elif n_cats > 20:
                print("\tERROR - more classes than colours!?!?!?")
            sns.set_palette(palette=use_colours, n_colors=n_cats)

            # Make MULTI plot
            fig = plt.figure(figsize=(10, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
            zeros_axis = plt.subplot(gs[0])
            rain_axis = plt.subplot(gs[1])

            # # # # # # # # # # # #
            # # make zeros plot # #
            # # # # # # # # # # # #

            # 1. get biggest class size (for max val of plot)
            max_class_size = max(use_this_ipc.values())
            print(f"\tmax_class_size: {max_class_size}")

            # 2. get list or dict of zeros per class
            zeros_dict = {}
            for k in range(n_cats):
                if plot_type is "OneVsAll":
                    plot_names = ["all_others", f"class_{coi}"]
                    this_name = plot_names[k]
                    this_class = unit_df.loc[unit_df['OneVsAll'] == k]
                    zero_count = 0 - (this_class['hid_acts'] == 0).sum()
                    zeros_dict[this_name] = zero_count
                else:
                    this_class = unit_df.loc[unit_df['class'] == k]
                    zero_count = 0 - (this_class['hid_acts'] == 0).sum()
                    zeros_dict[k] = zero_count

            # zd_classes = list(zeros_dict.keys())
            # zd_classes = list(lesion_dict['data_info']['cat_names'].values())
            zd_zero_count = list(zeros_dict.values())

            if verbose:
                print(f"\n\tzeros_dict:{zeros_dict.values()}, use_this_ipc:{use_this_ipc.values()}")

            zd_zero_perc = [x / y * 100 if y else 0 for x, y in zip(zeros_dict.values(), use_this_ipc.values())]

            zd_data = {"class": class_labels, "zero_count": zd_zero_count, "zero_perc": zd_zero_perc}

            zeros_dict_df = pd.DataFrame.from_dict(data=zd_data)

            # zero_plot
            sns.catplot(x="zero_perc", y="class", data=zeros_dict_df, kind="bar", orient='h', ax=zeros_axis)

            zeros_axis.set_xlabel("% at zero (height reflects n items)")

            zeros_axis.set_xlim([-100, 0])

            # # set width of bar to reflect class size
            new_heights = [x / max_class_size for x in use_this_ipc.values()]
            print(f"\tuse_this_ipc: {use_this_ipc}\n\tnew_heights: {new_heights}")

            # def change_height(zeros_axis, new_value):
            patch_count = 0
            for patch in zeros_axis.patches:
                current_height = patch.get_height()
                make_new_height = current_height * new_heights[patch_count]
                diff = current_height - make_new_height

                if new_heights[patch_count] < 1.0:
                    # print("{}. current_height {}, new_height: {}".format(patch, current_height, make_new_height))

                    # # change the bar height
                    patch.set_height(make_new_height)

                    # # recenter the bar
                    patch.set_y(patch.get_y() + diff * .65)

                patch_count = patch_count + 1


            zeros_axis.set_xticklabels(['100', '50', ''])
            # zeros_axis.xaxis.set_major_locator(plt.MaxNLocator(1))
            plt.close()

            # # # # # # # # #
            # # raincloud # #
            # # # # # # # # #

            data_values = "hid_acts"  # float
            if normed_acts:
                data_values = 'normed'
            data_class = plot_y_vals  # class
            orientation = "h"  # orientation

            # cloud_plot
            pt.half_violinplot(data=raincloud_data, bw=.1, linewidth=.5, cut=0., width=1, inner=None,
                               orient=orientation, x=data_values, y=data_class, scale="count")  # scale="area"

            """# # rain_drops - plot 3 separate plots so that they are interesting items are ontop of pass-pass
            # # zorder is order in which items are printed
            # # item_change: 1 ('grey') passed before and after lesioning
            # # -1 ('red') passed in full model but failed when lesioned
            # # 2 ('green') failed in full model but passed in lesioning"""
            fail_palette = {1: "silver", -1: "red", 2: "green", 0: "orange"}


            # # separate rain drops for pass pass,
            pass_pass_df = unit_df[(unit_df['item_change'] == 1)]
            pass_pass_drops = sns.stripplot(data=pass_pass_df, x=data_values, y=data_class, jitter=1, zorder=1,
                                            size=2, orient=orientation)  # , hue='item_change', palette=fail_palette)

            if plot_fails is True:

                '''I'm not using this atm, but if I want to plot items that originally failed and later passed'''
                # # separate raindrop for fail pass
                # fail_pass_df = unit_df[(unit_df['item_change'] == 2)]
                # if not fail_pass_df.empty:
                #     fail_pass_drops = sns.stripplot(data=fail_pass_df, x=data_values, y=data_class, jitter=1,
                #                                     zorder=3, size=4, orient=orientation, hue='item_change',
                #                                     palette=fail_palette, edgecolor='gray', linewidth=.4, marker='s',
                #                                     label='')

                # # separate raindrops for pass fail
                if not l_failed_df.empty:
                    # pass_fail_drops
                    sns.stripplot(data=l_failed_df, x=data_values, y=data_class, jitter=1, zorder=4, size=4,
                                  orient=orientation, hue='item_change', palette=fail_palette, edgecolor='white',
                                  linewidth=.4, marker='s')

            # box_plot
            sns.boxplot(data=raincloud_data, color="gray", orient=orientation, width=.15, x=data_values,
                        y=data_class, zorder=2, showbox=False,
                        # boxprops={'facecolor': 'none', "zorder": 2},
                        showfliers=False, showcaps=False,
                        whiskerprops={'linewidth': .01, "zorder": 2}, saturation=1,
                        # showwhiskers=False,
                        medianprops={'linewidth': .01, "zorder": 2},
                        showmeans=True,
                        meanprops={"marker": "*", "markerfacecolor": "white", "markeredgecolor": "black"}
                        )

            # # Finalize the figure
            rain_axis.set_xlabel("Unit activations")
            if normed_acts:
                rain_axis.set_xlabel("Unit activations (normalised)")

            # new_legend_text = ['l_passed', 'l_failed']
            new_legend_text = ['l_failed']

            leg = pass_pass_drops.axes.get_legend()
            if leg:
                # in here because leg is None if no items changed when this unit was lesioned
                for t, l in zip(leg.texts, new_legend_text):
                    t.set_text(l)

            # # hid ticks and labels from rainplot
            plt.setp(rain_axis.get_yticklabels(), visible=False)
            rain_axis.axes.get_yaxis().set_visible(False)

            # # put plots together
            max_activation = max(this_unit_hid_acts)
            min_activation = min(this_unit_hid_acts)
            if normed_acts:
                max_activation = max(this_unit_normed_acts)
                min_activation = min(this_unit_normed_acts)

            max_x_val = max_activation * 1.05
            layer_act_func = None
            for k, v in lesion_dict['model_info']['layers']['hid_layers'].items():
                if v['name'] == gha_layer_name:
                    layer_act_func = v['act_func']
                    break
            if layer_act_func in ['relu', 'Relu', 'ReLu']:
                min_x_val = 0
            elif min_activation > 0.0:
                min_x_val = 0
            else:
                min_x_val = min_activation

            rain_axis.set_xlim([min_x_val, max_x_val])
            rain_axis.get_shared_y_axes().join(zeros_axis, rain_axis)
            fig.subplots_adjust(wspace=0)

            fig.suptitle(title, fontsize=12).set_position([.5, 1.0])  # .set_bbox([])  #

            # # add y axis back onto rainplot
            plt.axvline(x=min_x_val, linestyle="-", color='black', )

            # # add marker for max informedness
            if 'info' in coi_measure:
                if old_sel_dict:
                    normed_info_thr = sel_dict['sel_info'][gha_layer_name][unit]['max']['thr_informed']
                else:
                    print(sel_info[gha_layer_name][unit]['max'])
                    normed_info_thr = sel_info[gha_layer_name][unit]['max']['max_info_thr']

                if normed_acts:
                    best_info_thr = normed_info_thr
                else:
                    # unnormalise it
                    best_info_thr = normed_info_thr * max(this_unit_hid_acts)
                print(f"\tbest_info_thr: {best_info_thr}")
                plt.axvline(x=best_info_thr, linestyle="--", color='grey')

            # sns.despine(right=True)

            if plot_type is "OneVsAll":
                plt.savefig(f"{output_filename}_{gha_layer_name}_{unit}_cat{coi}_raincloud.png")

            else:
                plt.savefig(f"{output_filename}_{gha_layer_name}_{unit}_raincloud.png")

            if test_run:
                plt.show()

            print("\n\tplot finished\n")

            # # clear for next round
            plt.close()

    # # plt.show()
    print("End of script")

####################################################################################
# print("\n!\n!\n!TEST RUN FROM BOTTOM OF SCRIPT")
