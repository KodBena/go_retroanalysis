from util import load_sgf_file,mkid, xchg_color, parity_player,_list_oc,sign,random_key,uct
from problem import bm,mk_sgf_letters_bimap,Problem
from stats import *
SGF_FILE="/home/bork/1sgf/retroanalysis/34_54_73_pincer_1.sgf"

def sgf_to_query_dict(sgf):
    bm = mk_sgf_letters_bimap()
    out=dict()
    out['id'] = mkid()
    props = sgf.get_root().get_raw_property_map()
    board=boards.Board(19);
    board.apply_setup(*sgf.root.get_setup_stones())

    out['rules'] = 'tromp-taylor'
    out['komi'] = sgf.get_komi()
    out['boardXSize'] = sgf.get_size()
    out['boardYSize'] = sgf.get_size()
    out['moves'] = []
    out['maxVisits'] = 1
    out['includePolicy'] = True
    p = Problem(sgf)
    out['initialStones'] = p.initials
    #out['initialPlayer'] = 'B'
    out['initialPlayer'] = p.relevant_player # ???
    comparison_query = p.comparison_query
    reference_query = p.reference_query
    out['avoidMoves'] = p.avoidMoves
    xdata = dict(ref = reference_query, comp = comparison_query,
        relevant_player = p.relevant_player,
        initial_counts = p.initial_counts,
        passes=p.passes)

    alpha = xdata['initial_counts']['B'] - xdata['initial_counts']['W']
    neg_alpha = -alpha
    match xdata['relevant_player']:
        case 'B':
            b_passes = max(0, -neg_alpha)
            w_passes = max(0, neg_alpha)
        case 'W':
            b_passes = max(0, -(neg_alpha + 1))
            w_passes = max(0, neg_alpha + 1)
    assert b_passes == xdata['passes']['B'], f"b_passes: {b_passes},w_passes: {w_passes} vs {xdata['passes']}"
    assert w_passes == xdata['passes']['W']
    return out,xdata,board

def test_query_invariant(query,xdata):
    stones = query['initialStones']
    result = 0 if xdata['relevant_player'] == 'B' else 1
    return (result == sum([sign(x[0]) for x in stones]))

def mk_forbidden_vertices(query):
    qq = { p['player']:p['moves'] for p in query['avoidMoves']}
    forbidden_vertices =  sorted(list(set(sum(qq.values(),[]))))
    return forbidden_vertices

def test_initial_condition(query):
    qq = { p['player']:p['moves'] for p in query['avoidMoves']}
    forbidden_vertices =  sorted(list(set(sum(qq.values(),[]))))
    avoidstones = [ [player, vertex] for player,lst in qq.values() for vertex in lst ]
    return avoidstones



from node import *

def mk_root_node(initial_query, xdata,board):
    node = Node(board, 'root',parent=None, parent_query = initial_query, action=None,xdata=xdata)
    node.forbidden_vertices = mk_forbidden_vertices(initial_query)
    node.balance = xdata['initial_counts']
    return node

def get_initial_data(pathname):
    sgf = load_sgf_file(pathname);
    initial_query,xdata,board=sgf_to_query_dict(sgf);
    return initial_query, xdata,board

from kgclient import KataGoClient


def init_interactive_test():
    initial_query, xdata,board = get_initial_data(SGF_FILE)
    katago = KataGoClient('ipc:///tmp/kg.sock')
    return initial_query,xdata, board, katago

def start_test(random_drop = False, m=10):
    initial_query, xdata, board, katago = init_interactive_test();
    node = mk_root_node(initial_query, xdata,board);
    node.random_drop = random_drop
    node.katago = katago
    for i in range(m):
        r,v = node.apply()
        if not r:
            return v
        if 0 == (i % 16):
            print(f"i = {i}")
    return node

