import pickle
import json
import os.path
import datetime


# TODO: make save dict as pickle or json function (or hd5f?)
# todo: add filepath as optional arg for load dict.


tools_date = int(datetime.datetime.now().strftime("%y%m%d"))
tools_time = int(datetime.datetime.now().strftime("%H%M"))


def simple_dict_print(dict_to_print):
    """prints a dictionary
    1. one key-value pair per row (even if value is int or dict)
    """
    for key, value in dict_to_print.items():
        print("{} : {}".format(key, value))



def print_nested_round_floats(dict_to_print, dict_title='focussed_dict_print'):
    """prints a dictionary
    1. indents nested layers (upto a depth of 4)
    2. rounds floats to 2dp."""

    print(f"\n** {dict_title} **")


    for key, value in dict_to_print.items():
        if 'dict' in str(type(value)):
            print("{} :...".format(key))
            for key1, value1 in value.items():
                if 'dict' in str(type(value1)):
                    print("    {} :...".format(key1))
                    for key2, value2 in value1.items():
                        if 'dict' in str(type(value2)):
                            print("        {} :...".format(key2))
                            for key3, value3 in value2.items():
                                if 'dict' in str(type(value3)):
                                    print("            {} :...".format(key3))
                                    for key4, value4 in value3.items():
                                        if 'float' in str(type(value3)):
                                            print("                {} : {:.2f}".format(key4, value4))
                                        else:
                                            print("                {} : {}".format(key4, value4))
                                else:
                                    if 'float' in str(type(value3)):
                                        print("            {} : {:.2f}".format(key3, value3))
                                    else:
                                        print("            {} : {}".format(key3, value3))
                        else:
                            if 'float' in str(type(value2)):
                                print("        {} : {:.2f}".format(key2, value2))
                            else:
                                print("        {} : {}".format(key2, value2))
                else:
                    if 'float' in str(type(value1)):
                        print("    {} : {:.2f}".format(key1, value1))
                    else:
                        print("    {} : {}".format(key1, value1))
        else:
            if 'float' in str(type(value)):
                print("{} : {:.2f}".format(key, value))
            else:
                print("{} : {}".format(key, value))


def focussed_dict_print(dict_to_print, dict_title='focussed_dict_print', focus_list=[]):
    """will print a dict in one of two ways.
    1. most key-values pairs will be printed on one line each (simple dict).
    2. If key in focus list:
        print nested dict"""
    print("\n** {} **".format(dict_title))

    for key, value in dict_to_print.items():
        if key not in focus_list:
            print("{} : {}".format(key, value))
        else:
            # print("\n* {} :...".format(key))
            print_nested_round_floats(value, dict_title=key)


def is_int(number):
    """will try to convert value to int"""
    try:
        num = int(number)
    except ValueError:
        return False
    return True


def is_float(number):
    """will try to convert value to float"""
    try:
        num = float(number)
    except ValueError:
        return False
    return True


def json_key_to_int(input_dict, verbose=False):
    """Returns an dictionary where keys that should be ints are returned as ints.
    A key that should be an int but isn't is found if:
        1.  the key was saved as a float in the generation process (e.g., units 1.0, 2.0, 3.0).
        2. the key was saved as a string when saving as json (e.g., units '1.0', '2.0', '3.0')

       Arguments:
           - input_dict: count be nested to any depth

       Returns:
           - output dict - same dimensions, just keys changed
    """
    # print("**** json_key_to_int() ****")

    output_dict = {}
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            if type(value) is dict:
                # recursively check for nested dicts first
                value = json_key_to_int(value)

            # # if no further nested levels remain
            if is_int(key) is True:
                new_key = int(key)
                output_dict[new_key] = value
            elif is_float(key) is True:
                new_key = int(float(key))
                output_dict[new_key] = value
            else:
                new_key = key
                output_dict[new_key] = value

            if verbose is True:
                if type(value) is dict:
                    print("{} {} -> {} {}\n".format(key, type(key), new_key, type(new_key)))
                else:
                    print("    {} {} -> {} {}".format(key, type(key), new_key, type(new_key)))

    return output_dict


def load_dict(dict_name):
    """function to take a dict_name string and search for the corresponding file.
    It will open the dict from pickle or json.
    It will run convert str(int) to int() on keys that json changed.

    it will check for an exact match for the string, or
    for {}_load_dict - which is the naming convention for dataset dicts"""

    print("\n**** load_dict() ****")

    json_dict = False
    if os.path.isfile("{}_load_dict.pickle".format(dict_name)):
        loaded_dict = pickle.load(open(dict_name + "_load_dict.pickle", "rb"))
        print("loaded: {}_load_dict.pickle".format(dict_name))

    elif os.path.isfile("{}.pickle".format(dict_name)):
        loaded_dict = pickle.load(open(dict_name + ".pickle", "rb"))
        print("loaded: {}.pickle".format(dict_name))

    elif os.path.isfile("{}_load_dict.txt".format(dict_name)):
        loaded_dict = json.loads(open("{}_load_dict.txt".format(dict_name)).read())
        print("loaded: {}_load_dict.txt".format(dict_name))
        json_dict = True

    elif os.path.isfile("{}.txt".format(dict_name)):
        loaded_dict = json.loads(open("{}.txt".format(dict_name)).read())
        print("loaded: {}.txt".format(dict_name))
        json_dict = True

    elif os.path.isfile("{}".format(dict_name)):
        if dict_name[-7:] == '.pickle':
            loaded_dict = pickle.load(open(dict_name, "rb"))
            print("loaded: {}".format(dict_name))

        elif dict_name[-4:] == '.txt':
            loaded_dict = json.loads(open(dict_name).read())
            print("loaded: {}".format(dict_name))
            json_dict = True

    if json_dict is True:
        loaded_dict = json_key_to_int(loaded_dict)

    return loaded_dict



def load_dict_from_data(data_path):
    """function to take a data_path (from datasets folder) of dict.
    It will join the path to the main datapath and extract the dset name.
    It will open the dict from pickle or json.
    It will run convert str(int) to int() on keys that json changed.

    it will check for an exact match for the string, or
    for {}_load_dict - which is the naming convention for dataset dicts"""

    dataset_root = '/home/nm13850/Documents/PhD/python_v2/datasets/'

    dset_dir, dset_name = os.path.split(data_path)
    look_here = os.path.join(dataset_root, dset_dir)
    os.chdir(look_here)

    print("\n**** load_dict({}) ****".format(dset_name))
    print("current wd: {}".format(os.getcwd()))

    json_dict = False
    if os.path.isfile("{}_load_dict.pickle".format(dset_name)):
        loaded_dict = pickle.load(open(dset_name + "_load_dict.pickle", "rb"))
        print("loaded: {}_load_dict.pickle".format(dset_name))

    elif os.path.isfile("{}.pickle".format(dset_name)):
        loaded_dict = pickle.load(open(dset_name + ".pickle", "rb"))
        print("loaded: {}.pickle".format(dset_name))

    elif os.path.isfile("{}_load_dict.txt".format(dset_name)):
        loaded_dict = json.loads(open("{}_load_dict.txt".format(dset_name)).read())
        print("loaded: {}_load_dict.txt".format(dset_name))
        json_dict = True

    else:  # #  os.path.isfile("{}.txt".format(dset_name)):
        loaded_dict = json.loads(open("{}.txt".format(dset_name)).read())
        print("loaded: {}.txt".format(dset_name))
        json_dict = True

    if json_dict is True:
        loaded_dict = json_key_to_int(loaded_dict)

    return loaded_dict


