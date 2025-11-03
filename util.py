
import numpy as np
from sgfmill import sgf as _sgf
from uuid import uuid4
import string
relevant_keys = ['move','order','pv','scoreMean','winrate','utility','utilityLcb','visits']
def top_k(moveInfos,k):
    s = [ {k:x[k] for k in relevant_keys} for x in moveInfos]
    s = sorted(s,key=lambda x: x['order'])[:k]
    return s
def mk_sgf_letters_bimap():
    sgf_letters = [ x for x in string.ascii_lowercase if x != 'i']
    sgf_letters_bimap =dict()
    for i,c in zip(range(1,1+len(sgf_letters)), sgf_letters):
        sgf_letters_bimap[i] = c.upper()
        sgf_letters_bimap[c.lower()] = i
        sgf_letters_bimap[c.upper()] = i
    return sgf_letters_bimap
bm = mk_sgf_letters_bimap()
def load_sgf_file(pathname:str):
    with open(pathname,'r') as f:
        st = f.read()
        g = _sgf.Sgf_game.from_string(st)
    return g
def mkid():
    return str(uuid4())
xchg_color = { "W":"B", "B":"W" }
parity_player = {"W":1, "B": 0}
sign = lambda x: 1 if x=='B' else -1
def _list_oc(board):
    def f(p):
        return [p[0].upper(), f"{bm[1+p[1][1]]}{1+p[1][0]}"]
    return [f(p) for p in board.list_occupied_points()]

def random_key(d):
    n = len(d)
    keys=list(d.keys())
    return keys[np.random.choice(n)]

def uct(self_visits, child_keys_and_values):
    numerator = 2*np.log(1+self_visits)
    lst = [ (c, mu + np.sqrt(numerator/denom)) for (c,(mu,denom)) in child_keys_and_values]
    m = np.max([x[1] for x in lst])
    lst_filtered = [ x[0] for x in lst if x[1] == m]
    child = lst_filtered[np.random.choice(len(lst_filtered))]
    return child
