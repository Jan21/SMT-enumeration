from featurizer import *
import joblib
from pysmt.smtlib.parser import SmtLibParser,get_formula,Tokenizer
from pysmt.smtlib.script import SmtLibCommand, SmtLibScript
import copy
import pysmt.smtlib.commands as smtcmd
from io import StringIO

def get_script(selff,script):
    #selff._reset()
    res = SmtLibScript()
    for cmd in selff.get_command_generator(script):
        res.add_command(cmd)
    res.annotations = selff.cache.annotations
    return res


def get_parsed_script(parser,toparse,cache,env):
    cstring = cStringIO(toparse)
    #parser._reset()
    #parser.env = copy.deepcopy(env)
    parser.cache = copy.deepcopy(cache)
    parser.env = copy.deepcopy(env)
    parsed = get_script(parser,cstring)
    cache = copy.deepcopy(parser.cache)
    return parsed, cache
def get_decl(parser,toparse,cache,env):
    parsed,cache = get_parsed_script(parser,toparse,cache,env)
    env = copy.deepcopy(parser.env)
    return cache,env
def get_graph_rep(parser,toparse):
    parsed = get_script(parser,toparse)
    f = parsed.get_strict_formula()
    c = cStringIO()
    pr = CustomSmtPrinter(c)
    e, n = pr.walk(f)
    return e,n

# def get_graph_rep(toparse,formula):
#     toparselines = toparse.split("\n")
#     toparse1 = "\n".join(toparselines[:-5])
#     toparse2 = "\n".join(toparselines[-5:])
#     cstring = cStringIO(toparse)
#     parsed = parser.get_script(cstring)
#     parser_cache = copy.deepcopy(parser.cache)
#     parsed = get_script(parser,cStringIO(toparse2))
#     #env.formula_manger = copy.deepcopy(f_man_decl)
#     parser._reset()
#     parser.cache = copy.deepcopy(parser_cache)
#     parsed = get_script(parser,cStringIO(formula))
#     f = parsed.get_strict_formula()
#     c = cStringIO()
#     pr = CustomSmtPrinter(c)
#     e, n = pr.walk(f)
#     return e,n

def get_parsed_format(quantifier,log):
    declarations = ""
    sorts = ""
    for l in log:
        if l.startswith('(declare-fun') or l.startswith('(declare-constant'):
            declarations += l
        if l.startswith('(declare-sort'):
            sorts += l
    parser = SmtLibParser(interactive=True)
    formula_str = quantifier
    formula = cStringIO(f"{formula_str}")
    div_mod_declarations = '(declare-fun __div (Int Int) Int)\n (declare-fun __mod (Int Int) Int)\n'
    toparse = sorts + declarations + div_mod_declarations + formula_str + "(check-sat)"
    toparse = formula_str
    parser = SmtLibParser()
    res = parser.get_script(cStringIO(toparse))
    f = res.get_strict_formula()
    c = cStringIO()
    pr = CustomSmtPrinter(c)
    e, n = pr.walk(f)
    exp = parser.get_expression(Tokenizer(formula))
    f = exp.get_strict_formula()
    c = cStringIO()
    pr = CustomSmtPrinter(c)
    e, n = pr.walk(f)
    #formula = get_script(parser,cStringIO(sorts+formula))
    #formula = get_formula(cStringIO(formula),env)
    e, n1 = get_graph_rep(parser, cStringIO(toparse+formula))
    e, n2 = get_graph_rep(parser, cStringIO(formula))
    cntr = Counter({'SYMBOL': 6, 'NOT': 4, 'EQUALS': 4, 'INT_CONSTANT': 4, 'LE': 3, 'FUNCTION': 2, 'PLUS': 2, 'FORALL': 1,
             'AND': 1})
    assert Counter(n1.values()) == Counter(n2.values()) == cntr
    extracted_data_per_formula = {"formula_graph":{'edges':e,'nodes_dic':n},'terms':[]}
    var_term_count = []
    for var in candidates[2:]:
        term_count = 0
        for var_info in var[2:]:
            base_features = {k[0]:v for k,v in var_info[2:]}
            var = var_info[1]    
            if var == ('null',):
                continue
            var_str = print_str(var)
            term = f"(assert ( = {var_str} {var_str}))\n (check-sat)\n"
            toparse = declarations + term
            e, n = get_graph_rep(toparse)            
            extracted_data_per_formula['terms'].append({"base":base_features,'edges':e,'nodes_dic':n})
            term_count += 1
        var_term_count.append(term_count)
    return extracted_data_per_formula,var_term_count


with open('data/dec_problem.txt','r') as f:
    log = f.readlines()

with open('data/pokus8.smt2','r') as f:
    quantifier = f.read()

extracted_data_per_formula,var_term_counts = get_parsed_format(quantifier,log)
split_ixs = []