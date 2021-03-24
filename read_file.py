from parse_expr import sexp
import pyparsing
from pprint import pprint
import glob
from other_parser import parse
import numpy as np

global_num_vars = []

def read_file(path):
    with open(path, "r") as file:
        f = file.readlines()

    return f


def get_quantifier_candidates(file):
    for e, line in enumerate(file):
        if line.startswith("(quantifier_candidates"):
            index = e

    # Changed this to include a stopping line 0303

    candidates = file[index:]

    candidates_string = "".join([k.strip() for k in candidates])
    # print(candidates_string)
    return sexp.parseString(candidates_string).asList()

def get_quantifier_candidates_new_parser(file):

    for e, line in enumerate(file):
        if line.startswith("(quantifier_candidates"):
            index = e

    # Changed this to include a stopping line 0303

    candidates = file[index:]

    candidates_string = "".join([k.strip() for k in candidates])

    return parse(candidates_string)


def get_candidate_infos(file):

    for e, line in enumerate(file):
        if line.startswith("(candidate_infos"):
            index = e

    candidate_infos = file[index:]

    candidate_info_string = "".join([k.strip() for k in candidate_infos])

    return sexp.parseString(candidate_info_string).asList()


def to_tuple(lst):
    """
    Some recursive nested list to nested tuple conversion with list type check
    """
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def depthCount(x):
    return 1 + max(map(depthCount, x)) if x and isinstance(x, tuple) else 0


def extract_candidates(variable_candidate_list):
    """"
    Get a nested list containing all the candidates for a specific variable,
    Output a dictionary mapping candidate terms to lists of features

    It's a dictionary because we need to be able to match against terms later to collect the
    output variable (usefulness)
    """

    candidates = variable_candidate_list[2:]  # strip away nodetype and variable number
    feature_dictionary = {}

    for candidate in candidates:
        _, term, age, phase, relevant = candidate

        if type(term) == list:
            term = to_tuple(term)  # we need the term to be hashable

        feature_dictionary[term] = [age[1], phase[1], relevant[1], depthCount(term)]

    return feature_dictionary


def extract_useful(feature_dictionary_list, useful):
    # print("FDL: ", feature_dictionary_list)
    for feature_dictionary in feature_dictionary_list:
        for candidate in feature_dictionary:
            feature_dictionary[candidate] = [
                feature_dictionary[candidate],
                0,
            ]  # put 0s everywhere

    if len(useful) > 1:

        useful = useful[1:]


        if len(useful[0]) == 1:
            for useful_instantiation in useful:
                term = useful_instantiation[0]
                if type(term) == list:
                    term = to_tuple(term)


                feature_dictionary_list[0][term][1] = 1  # if useful, set class to 1.
        else:
            for useful_instantiation in useful:
                term_list = useful_instantiation

                for e, term in enumerate(term_list):
                    if type(term) == list:
                        term = to_tuple(term)

                    feature_dictionary_list[e][term][1] = 1


    return feature_dictionary_list


def collect_features(candidates):
    # Get a list of all the candidates

    # This is a list of all the quantified parts, which contains lists of instantiations for each variable in the part
    list_quantified_parts = candidates[0]
    print(list_quantified_parts)
    sample_list = []
    for quantified_part in list_quantified_parts[1:]:

        quantified_part = quantified_part[1:]  # disregard 'candidates' label
        print(quantified_part)
        multivariable_dict_list = []
        for item in quantified_part:
            print(item)
            # if item[0] == "forall":
            if item[0] == ('forall',):
                print("TRUE")
                pass
            # elif item[0] == "variable":
            elif item[0] == ('variable',):


                feature_dict = extract_candidates(item)
                multivariable_dict_list.append(feature_dict)
            # elif item[0] == "all_successful_instantiations":
            elif item[0] == ('all_successful_instantiations',):
                pass
            # elif item[0] == "useful_instantiations":
            elif item[0] == ('useful_instantiations',):

                # print("NUM: ", len(multivariable_dict_list))
                global_num_vars.append(len(multivariable_dict_list))
                feature_dict_final = extract_useful(multivariable_dict_list, item)
                for fd in feature_dict_final:

                    sample_list += list(fd.values())


    return sample_list


def convert_files_to_data(folder):

    folder += "/*"

    file_list = glob.glob(folder)
    exceptioncounter = 0
    samples = []
    for file in file_list:
        print(file)
        try:
            fi = read_file(file)
            # candidates = get_quantifier_candidates(fi)
            candidates = get_quantifier_candidates_new_parser(fi)
            print(candidates)
            sl = collect_features(candidates)
            samples += sl
        except pyparsing.ParseException as e:

            print(e)
            exceptioncounter += 1

    print(samples)

    # print(file_list)
    print("Number of parsing exceptions: ", exceptioncounter)
    x = []
    y = []
    for x_s, y_s in samples:
        x.append(x_s)
        y.append(y_s)

    x_array = np.array(x)
    y_array = np.array(y)

    np.save("x_only_tried_perm0_nord.npy", x_array)
    np.save("y_only_tried_perm0_nord.npy", y_array)

import numpy as np
# print(np.mean(global_num_vars))
convert_files_to_data("/home/jelle/projects/smt_quantifier_prediction/data_folders/data_only_qtried/easy_UFNIA_180_4_cvc4ql_perm0_nord/")
