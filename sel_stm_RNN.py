import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical

# # # PART 1 # # #
GHA_dict_filename = "test_vocab_30_seq3_test_Y_GHA_dict"
GHA_dict = json.loads(open("{}.txt".format(GHA_dict_filename)).read())

print("\n**** SIMULATION DICTIONARY ****")
# for key, value in GHA_dict.items():
#     print("{0} : {1}".format(key, value))
for key, value in GHA_dict.items():
    if 'dict' in str(type(value)):
        print("{} :...".format(key))
        for key1, value1 in value.items():
            print(" - {} : {}".format(key1, value1))
    else:
        print("{} : {}".format(key, value))


# load other info
topic_name = GHA_dict['topic_info']['topic_name']
GHA_set = GHA_dict['GHA_info']['dataset']

# 1. find main dataset
data_info_location = GHA_dict['data_info'][GHA_set]
if 'X_data' in data_info_location.keys():
    print("\ntraining data found in GHA_dict['data_info'][{}]".format(GHA_set))
elif 'X_data' in GHA_dict.keys():
    print("\ntraining data found in data_dict")
    data_info_location = GHA_dict['data_info']
else:
    print("\ntraining data not found")
    # sys.exit()

# 2. load main dataset stuff
# # seq_data = np.loadtxt(data_info_location["seqs"], delimiter=',')
# seq_data = pd.read_csv(data_info_location["seqs"], header=None, names=['seq1', 'seq2', 'seq3'])
# print(f"\nseq_data: {seq_data.shape}\n{seq_data.head()}")

# # combo_data = np.loadtxt(data_info_location["combos"], delimiter=',')
# combo_data = pd.read_csv(data_info_location["combos"])
# print(f"\ncombo_data: {combo_data.shape}\n{combo_data.head()}")


# X_data = np.load(data_info_location["X_data"])
# print("\nshape of X_data: {}".format(np.shape(X_data)))

# # "Y_labels": "Y_label_seqs.csv", first work is 
Y_labels = np.loadtxt(data_info_location["Y_labels"], delimiter=',').astype('int8')
print(f"\nY_labels:\n{Y_labels}")
print(np.shape(Y_labels))

# Y_data = to_categorical(Y_labels, num_classes=30)
# print(f"\nY_data:\n{Y_data}")
# print(np.shape(Y_data))


# # other details
n_items = data_info_location["n_items"]
n_cats = GHA_dict["data_info"]["n_cats"]
hid_layers = GHA_dict["model_info"]["hid_layers"]
hid_units = GHA_dict['model_info']["hid_units"]  # for output units change this to 10
units_per_layer = int(hid_units / hid_layers)  # GHA_dict["model_info"]["units_per_layer"]
timesteps = GHA_dict['model_info']["timesteps"]



hid_act_files = GHA_dict["GHA_info"]["hid_act_files"]
hid_acts = np.load(hid_act_files)
print("hid_acts shape: {}".format(np.shape(hid_acts)))


# # # # idiot_check on output data
# out_unit_acts_name = "test_1000_combos_train_out_act.npy"
# out_acts = np.load(out_unit_acts_name)
# print("out_acts shape: {}".format(np.shape(out_acts)))
#
# # y_labels = combo_data
#
# hid_acts = out_acts

# Output file
output_filename = GHA_dict['topic_info']["output_filename"] + "_hid_units"


selectivity_output = []

print("\n*********Running Selectivity******")
for layer in range(hid_layers):
    layer_name = layer + 1
    print("\nLayer {} of {}".format(layer_name, hid_layers))
    if hid_layers == 1:  # just one per-unit file
        # # also find correct hid-act file (diff naming convention for 4 layers)
        hidden_activation_file = GHA_dict["GHA_info"]["hid_act_files"]  # open correct hid act file
    elif hid_layers == 4:  # make per unit file for each layer
        hidden_activation_file = GHA_dict["GHA_info"]["hid_act_files"][layer]
        # hidden_activation_file = hidden_activation_file_list[layer]

    # open hid_act file for this layer
    if hidden_activation_file[-3:] == 'csv':
        hid_act = np.genfromtxt(hidden_activation_file, delimiter=',')
        print("loaded hid_act as csv: {}".format(hidden_activation_file))
    elif hidden_activation_file[-3:] == 'npy':
        hid_act = np.load(hidden_activation_file)
        print("loaded hid_act as npy: {}".format(hidden_activation_file))
    elif hidden_activation_file[-3:] == 'act':
        hid_act = np.load(hidden_activation_file + '.npy')
        print("loaded hid_act as npy: {}".format(hidden_activation_file))
    else:
        print("unknown hid_act file type: {}".format(hidden_activation_file))
        if hid_layers == 1:  # sometimes it is looking in wrong place
            hidden_activation_file = GHA_dict["GHA_info"]["hid_act_files"][0]
            if hidden_activation_file[-3:] == 'csv':
                hid_act = np.genfromtxt(hidden_activation_file, delimiter=',')
                print("loaded hid_act as csv with [0]: {}".format(hidden_activation_file))
            elif hidden_activation_file[-3:] == 'npy':
                hid_act = np.load(hidden_activation_file)
                print("loaded hid_act as npy with [0]: {}".format(hidden_activation_file))
            elif hidden_activation_file[-3:] == 'act':
                hid_act = np.load(hidden_activation_file + '.npy')
                print("loaded hid_act as npy with [0]: {}".format(hidden_activation_file))
            else:
                print("unknown hid_act file type with [0]: {}".format(hidden_activation_file))


    # start looping through each unit in this layer
    print(f"np.shape(hid_act) (seqs, timesteps, units): {np.shape(hid_act)}")

    # for i in range(units_per_layer):
    for i in range(units_per_layer):
        this_unit = i
        print("\n******************\n***** unit {} *****\n******************".format(i))
        print("testing for dead unit")
        dead_unit = 0

        # print("hid_acts shape: {}".format(np.shape(hid_acts)))
        # print("combo_data shape: {}".format(np.shape(combo_data)))

        one_unit_all_timesteps = hid_acts[:, :, this_unit]
        # print("one unit all timesteps shape: {}".format(np.shape(one_unit_all_timesteps)))
        print(f"np.shape(one_unit_all_timesteps) (seqs, timesteps): {np.shape(one_unit_all_timesteps)}")

        if np.sum(one_unit_all_timesteps) == 0:  # check for dead units, if dead, all details to 0
            print("dead unit found")

            # # all descriptives of this unit to 0 (or whatever is appropriate)
            biggest_internal_gap_dict = {'gap_size': 0, 'subsequent_item': {'activation': 0}}
            dead_unit = 1

            insertion_list = [int(n_items / n_cats)] * n_cats  # so not flagged as clusters
            internal_list = [0] * n_cats
            selectivity_list = [0] * n_cats
            class_mean_list = [0] * n_cats
            class_sd_list = [0] * n_cats
            class_AUC_list = [0] * n_cats

            unit_mean_act = unit_sd_act = min_act = max_act = max_act_items = biggest_internal_gap = gap_from = \
                PREFERENCE = off_pref_gap = preferred_items = GEN_SEL = PERPLEXITY = cats_of_interest = \
                mode_top_cat_instances = SENSITIVITY = SPECIFICITY = NiClaSel = 0

            switch_like = strong_sel = single_sel = totem = clustered_unit = n_clusters = banded_unit = internal_cat = \
                sel_off = sel_on = sel_unit = double_unit = Bow_sel_score = \
                top_n_size = partial_cat_count = partial_cat_prop = \
                morcos_sel_v_items = final_inverted = final_opt_thr = TPR_at_opt_thr = FPR_at_opt_thr = \
                Zhou_dis_AUC = Morcos_dis_AUC = uninterpretable = 0

            norm_mean = norm_median = seventy_fifth_act = ninetieth_act = 0

            final_max_AUC = .5

            min_act_class = max_item = max_act_class = Bow_sel_cat = mode_cat_pref_group = partial_cat = \
                highest_CCMA = final_max_AUC_class = -999

            min_act_items = n_items
            below_two_five = below_five = below_seven_five = below_nine = 100



        else:  # if not dead, do selectivity analysis
            print("not a dead unit, running selectivity")

            # # maybe here look at different time steps?
            # start looping through each timestep for this unit
            for timestep in range(timesteps):
                print("unit {} step {} (of {})".format(this_unit, timestep, timesteps))

                one_unit_one_timestep = one_unit_all_timesteps[:, timestep]
                print(" - one_unit_one_timestep shape: (n seqs) {}".format(np.shape(one_unit_one_timestep)))

                # y_labels_one_timestep_float = combo_data[:, timestep]
                y_labels_one_timestep_float = Y_labels[:, timestep]

                y_labels_one_timestep = [int(q) for q in y_labels_one_timestep_float]
                print(" - y_labels_one_timestep shape: {}".format(np.shape(y_labels_one_timestep)))
                # print(y_labels_one_timestep)

                index = list(range(n_items))

                # insert act values in middle of labels (item, act, cat)
                this_unit_acts = np.vstack((index, one_unit_one_timestep, y_labels_one_timestep)).T
                print(" - this_unit_acts shape: {}".format(np.shape(this_unit_acts)))
                print(f"this_unit_acts: (seq_index, hid_acts, y_label)\n{this_unit_acts}")
                #
                # # this_unit_acts = np.insert(y_labels, 1, hid_act[:, this_unit], axis=1)  # item, activation, category
                # print(this_unit_acts[:, 0])

                # # # # # #
                # NC_sel  #
                # # # # # #
                print("running non-class-selectivity")
                # # normalize activations
                just_act_values = this_unit_acts[:, 1]
                max_act = max(just_act_values)

                # todo: I don't need to normalize them if they are tanh units?
                normed_acts = just_act_values / max_act

                # # get central tendency
                norm_mean = np.around(np.mean(normed_acts), decimals=3)
                norm_sd = np.around(np.std(normed_acts), decimals=3)
                norm_median = np.around(np.median(normed_acts), decimals=3)
                print("mean {} median {}".format(norm_mean, norm_median))

                # # # get activation at these points - 50th percentile, 75th, 90th
                # sorted_norm_acts = sorted(normed_acts)
                # print(n_items)
                # seventy_fifth_act = np.around(sorted_norm_acts[int((n_items / 100) * 75)], decimals=3)
                # ninetieth_act = np.around(sorted_norm_acts[int((n_items / 100) * 90)], decimals=3)
                # # print("75% of acts below: {}".format(seventy_fifth_act))
                # # print("90% of acts below: {}".format(ninetieth_act))

                # # get proportion below these values
                below_two_five = np.around(sum(i < .25 for i in normed_acts) / n_items * 100, decimals=2)
                below_five = np.around(sum(i < .5 for i in normed_acts) / n_items * 100, decimals=2)
                below_seven_five = np.around(sum(i < .75 for i in normed_acts) / n_items * 100, decimals=2)
                below_nine = np.around(sum(i < .9 for i in normed_acts) / n_items * 100, decimals=2)
                print("percentage of items below activation:\n.25: {}%\n.5: {}%\n.75: {}%\n.9: {}%"
                      "".format(below_two_five, below_five, below_seven_five, below_nine))

                # # # # # #
                # ROC AUC #
                # # # # # #
                print("running ROC AUC")
                # todo: It is nt clear that ROC, Mex_info etc are useful for tanh units.  check this
                # # # get data
                AUC_scores = this_unit_acts[:, 1]  # hid unit activations
                # AUC_class = this_unit_acts[:, 2]  # class labels
                # AUC_class = y_labels_one_timestep
                AUC_class = [int(q) for q in this_unit_acts[:, 2]]
                print("shape AUC score: {}, class: {}".format(np.shape(AUC_scores), np.shape(AUC_class)))


                unique_classes = np.unique(AUC_class)
                print("unique classes: {}".format(len(unique_classes)))

                # Binarize the output  - will look like a 1-hot output array
                # AUC_classes = label_binarize(AUC_class, classes=range(n_cats))
                AUC_classes = label_binarize(AUC_class, classes=range(n_cats))

                # # # MULTICLASS ROC AUC
                # Compute ROC curve and ROC area for each class
                fpr_dict = dict()
                tpr_dict = dict()
                thrshlds_dict = dict()
                roc_auc_dict = dict()
                opt_thr_dict = dict()
                abs_AUC = []
                inverted_AUC = []

                # # running ROC for all classes
                for b in range(n_cats):
                    # comput ROC_AUC for this class
                    fpr_dict[b], tpr_dict[b], thrshlds_dict[b] = roc_curve(AUC_classes[:, b], AUC_scores, pos_label=1)
                    roc_auc_dict[b] = auc(fpr_dict[b], tpr_dict[b])

                    # # # added these two lines to deal with missing classes
                    # print("\nclass {} ROC_AUC {}".format(b, roc_auc_dict[b]))
                    # if type(roc_auc_dict[b]) != float:
                    #     roc_auc_dict[b] = .5
                    print("\nclass {} ROC_AUC {}".format(b, roc_auc_dict[b]))

                    # get optimal threshold  # works for top of distribution
                    optimal_threshold = thrshlds_dict[b][np.argmax(tpr_dict[b] - fpr_dict[b])]

                    # update lists
                    for_abs_AUC = roc_auc_dict[b]
                    # abs_AUC.append()
                    inverted_AUC.append(0)
                    # print("class {} normal {}".format(i, roc_auc_dict[i]))

                    # Negative AUC  - if AUC < .5, then it is an inverted predictor - i.g., guess the opposite
                    if roc_auc_dict[b] < .5:
                        invert_AUC = 1 - roc_auc_dict[b]
                        for_abs_AUC = invert_AUC
                        inverted_AUC[b] = 1
                        print("class {} inverted {} to {}".format(b, roc_auc_dict[b], invert_AUC))

                        # select lower threshold by swapping tpr and fpr - works for bottom of distribution
                        inv_optimal_threshold = thrshlds_dict[b][np.argmax(fpr_dict[b] - tpr_dict[b])]
                        optimal_threshold = inv_optimal_threshold

                    abs_AUC.append(for_abs_AUC)
                    # update list with optimal threshold
                    opt_thr_dict[b] = optimal_threshold
                    print("optimal_threshold {}".format(optimal_threshold))

                # all raw (not abs) AUC scores - use on output.
                class_AUC_list = list(roc_auc_dict.values())
                print("\nclass_AUC_list {}".format(class_AUC_list))

                # # get max auc
                max_of_roc_auc = roc_auc_dict[max(roc_auc_dict, key=roc_auc_dict.get)]
                max_cat_of_roc_auc = list(roc_auc_dict.keys())[list(roc_auc_dict.values()).index(max_of_roc_auc)]

                max_abs_AUC = max(abs_AUC)

                # # get maximum values
                # # first check raw AUC_scores, then go off absolute values
                final_max_AUC = max_of_roc_auc
                final_max_AUC_class = max_cat_of_roc_auc
                final_inverted = 0
                # print("optimal_threshold {}".format(optimal_threshold))
                print("max roc_auc {} max abs AUC {}".format(max_of_roc_auc, max_abs_AUC))

                # # if best classifier is negative...
                if max_of_roc_auc < max_abs_AUC:
                    # print("abs_AUC {}".format(abs_AUC))
                    # print("inverted_list: {}".format(inverted_AUC))
                    final_max_AUC_class = abs_AUC.index(max_abs_AUC)
                    final_max_AUC = max_abs_AUC  # use this to show the 'inverted value'
                    # final_max_AUC = class_AUC_list[final_max_AUC_class]  # use this to show the 'actual value'
                    final_inverted = inverted_AUC[final_max_AUC_class]

                # # optimum threshld to use for this as a classifier
                final_opt_thr = opt_thr_dict[final_max_AUC_class]

                # # # get true pos and false pos rate at optimum threshold
                loc_of_opt_thr = np.where(thrshlds_dict[final_max_AUC_class] == final_opt_thr)[0][0]
                TPR_at_opt_thr = tpr_dict[final_max_AUC_class][loc_of_opt_thr]
                FPR_at_opt_thr = fpr_dict[final_max_AUC_class][loc_of_opt_thr]

                # # if best classifier is negative...change TPR and FPR
                if final_inverted == 1:
                    TPR_at_opt_thr = 1 - tpr_dict[final_max_AUC_class][loc_of_opt_thr]
                    FPR_at_opt_thr = 1 - fpr_dict[final_max_AUC_class][loc_of_opt_thr]

                # print("loc_of_opt_thr {}".format(loc_of_opt_thr))
                print("1. final_max_AUC {}".format(final_max_AUC))
                print("2. final_max_AUC_class {}".format(final_max_AUC_class))
                print("3. final_inverted {}".format(final_inverted))
                print("4. final_opt_thr {}".format(final_opt_thr))
                print("5. TPR_at_opt_thr {}".format(TPR_at_opt_thr))
                print("6. FPR_at_opt_thr {}".format(FPR_at_opt_thr))

                # # # # # #
                # end AUC #
                # # # # # #
                per_unit_headers = ['Unit', 'layer', 'timestep',
                                    'dead', 'mean_act', 'sd_act',
                                    'final_max_AUC', 'final_max_AUC_class',
                                    'below_five']
                unit_details = [this_unit, layer, timestep,
                                dead_unit, norm_mean, norm_sd,
                                final_max_AUC, final_max_AUC_class,
                                below_five]

                selectivity_output.append(unit_details)

                print("Sel output shape: {}\n".format(np.shape(selectivity_output)))
                # # unit_details_pd = pd.DataFrame(data=unit_details, columns=per_unit_headers)
                # unit_details_pd = pd.Series(data=unit_details, name="unit_details_pd")
                #
                # print(unit_details_pd)
                #
                # sel_all_units = sel_all_units.append(unit_details_pd)  # append to mega-list


# selectivity per unit pd
per_unit_headers = ['Unit', 'layer', 'timestep',
                    'dead', 'mean_act', 'sd_act',
                    'final_max_AUC', 'final_max_AUC_class',
                    'below_five']
sel_all_units = pd.DataFrame(columns=per_unit_headers, data=selectivity_output)

sel_all_units.to_csv("{}_sel_all_units.csv".format(output_filename))

print("script finished")
