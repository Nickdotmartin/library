from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, cosine_distances
import numpy as np
import scipy
from itertools import combinations
import json
import pandas as pd


# toy data
a = np.random.choice([0, 1], size=(2, 10), p=[0.2, 0.8])
b = np.random.choice([0, 1], size=(2, 10), p=[0.4, 0.6])
c = np.random.choice([0, 1], size=(2, 10), p=[0.5, 0.5])
d = np.random.choice([0, 1], size=(2, 10), p=[0.6, 0.4])
e = np.random.choice([0, 1], size=(2, 10), p=[0.8, 0.2])
f = np.random.choice([0, 1], size=(2, 10), p=[0.9, 0.1])

z = np.concatenate((a, b, c, d, e, f), axis=0)




# file_name = "y_50_by_50_binary_dense_50.csv"
# file_name = "HB_HW__cont_dataset1"
file_name = "MNISTmini_X"
file_type = ".csv"
file = file_name + file_type
load_file = np.loadtxt(file, delimiter=",")


'''
To turn this into a function I need...
input data - as array.  not sure if/how wel this will work for colour images - need to be careful with indexing.
n_cats
IPC dict

Not sure I can make it a function...

'''
# # enter either 'cos_sim, 'cos_dist' or 'taxi'
# loaded_dataset = z
loaded_dataset = load_file
distance = 'cos_sim'
n_cats = 10
IPC_dict = None
# IPC_dict = {0: 3, 1: 3, 2: 4, 3: 2}
dataset_name = 'MNISTmini'



dataset = np.asarray(loaded_dataset)
items, features = np.shape(dataset)
print(f'\ndataset: {dataset}')
print(f'items, features: {items}, {features}')

# add IPC dict here if class_sizes are not equal
if IPC_dict == None:
    cat_size = int(items/n_cats)
    IPC_dict = {i: cat_size for i in range(n_cats)}
    print(f'\nequal size IPC dict\n{IPC_dict}')

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
    elif distance in ['cos_dist', 'cosine_distance', 'cosine_dist', 'cos_distance']:
        within_cat = cosine_distances(this_cat)
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
    print(f"\nBetween group mean {distance} for {names_list[index]}: {mean_sim}")

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
between_array = np.zeros(shape=(n_cats, n_cats-1))

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
    between_array[idxA, idxB-1] = mean_between_pair
    between_array[idxB, idxA] = mean_between_pair

print(f"\nbetween_array:\n{between_array}")

print(f'\nmean between class {distance}')
between_list = []
for index in range(n_cats):
    this_row = between_array[index]
    this_mean = np.mean(this_row)
    between_list.append(this_mean)
    print(index, this_mean)

# # save output.
'''for each class:
    mean within
    mean between
    paired between 
'''
class_sim_dict = {
    # 'class': names_list,
                  'within': within_list,
                  'between': between_list}
class_sim_df = pd.DataFrame(class_sim_dict)
print(class_sim_df)
class_sim_df.to_csv(f'{dataset_name}_{distance}.csv', index_label='class', )
