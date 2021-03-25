from featurizer import *
import joblib

with open('databad/data/declarations.log','r') as f:
    log = f.readlines()

with open('databad/data/quantifier.txt','r') as f:
    quantifier = f.read()

extracted_data_per_formula,var_term_counts = get_parsed_format(quantifier,log)
split_ixs = []