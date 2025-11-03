class MoveInfosException(Exception):
    pass
class DefaultPolicyException(Exception):
    pass
class TreePolicyException(Exception):
    pass
class BorkException(Exception):
    pass
K_BUDGET=15
D_BUDGET=35
from copy import deepcopy
from functools import lru_cache,cached_property
from util import top_k,mkid,uct,random_key,xchg_color,_list_oc
from sgfmill.common import move_from_vertex as mvv
from sgfmill import boards
from sgfmill import sgf
import numpy as np
default_policy_stats = dict(tries=0, success = 0)
bests = [dict(num_best=0,node=None,value=0)]
solutions = [dict(num_solution=0,node=None,value=0)]
from collections import defaultdict, deque
import time
timestamp = lambda: time.clock_gettime(time.CLOCK_REALTIME)

now = timestamp()
leaf_statistics = dict(count=0,last=now, first=now)

fail_counts = defaultdict(int)

import time
import threading
from collections import Counter
import shutil


class CLIHistogram:
    def __init__(self, bins, minval, maxval, width=50, char='▇'):
        self.bins = bins
        self.minval = minval
        self.maxval = maxval
        self.width = width
        self.char = char
        self.counts = [0]*bins
        self.lock = threading.Lock()

    def add(self, val):
        with self.lock:
            if val < self.minval:
                idx = 0
            elif val >= self.maxval:
                idx = self.bins-1
            else:
                idx = int((val - self.minval) / (self.maxval - self.minval) * self.bins)
            self.counts[idx] += 1

    def print(self):
        with self.lock:
            maxcount = max(self.counts) if self.counts else 1
            cols = shutil.get_terminal_size((80,20)).columns
            bar_max = min(self.width, cols - 20)
            for i, c in enumerate(self.counts):
                bar_len = int(c / maxcount * bar_max)
                lo = self.minval + (self.maxval - self.minval) * (i / self.bins)
                hi = self.minval + (self.maxval - self.minval) * ((i+1) / self.bins)
                print(f"{lo:6.2f}-{hi:6.2f} | {self.char * bar_len} ({c})")

histo = CLIHistogram(24,0.5,1.1)

class Action:
    def __init__(self,to_move=None, which_move=None):
        self.to_move = to_move
        self.which_move = which_move
        self.kg_query_format = [to_move,which_move]
class Node:
    def __init__(self, xdata, board, id, parent=None, parent_query = None, action=None):
        self.xdata = xdata
        self.board = board
        self.action = action
        self.taboo_list = []
        self.id = id
        if parent is None:
            assert parent_query is not None
            self.query = parent_query
            self.parent = self
            self.to_move = xdata['relevant_player']
            self.depth=0
            self.pass_audit = dict(B=0,W=0)
            self.last_good_node = None
        else:
            self.parent = parent
            self.pass_audit = deepcopy(parent.pass_audit)
            self.query = deepcopy(parent.query)
            self.to_move = xchg_color[parent.to_move]
            self.balance = deepcopy(parent.balance)
            if action:
                self.balance[action.to_move] += 1
            else:
                self.pass_audit[parent.to_move] += 1
#            if self.pass_audit['B'] == self.xdata['passes']['B'] and self.pass_audit['W'] == self.xdata['passes']['W'] and self.get_root().xdata['relevant_player']:
#                print(f" found possibly terminal node. checking versus condition: {self.is_terminal}")
            self.update_query(action)
            self.depth = parent.depth + int(action is not None)
        self.visits = 0
        self._outcome = 0
        self.children = dict()
        self.q_value = 0
    def move_history(self):
        out = []
        node = self
        while True:
            z = None if not node.action else node.action.kg_query_format
            out.insert(0,z)
            node = node.parent
            if node.is_root:
                break
        return out

    def is_ancestor(self,other):
        node = self
        while not node.is_root:
            node = node.parent
            if other == node:
                return True
        return False

    @lru_cache()
    def get_root(self):
        node = self
        while not node.is_root:
            node = node.parent
        return node
    def initial_context(self):
        return self.get_root().query['initialStones']

    def update_query(self,action):
        self.query['id'] = mkid()
        self.query['initialPlayer'] = self.to_move
        if action is None:
            pass
        else:
            self.board.play(*mvv(action.which_move,19),action.to_move.lower())
            assert self.to_move == xchg_color[action.to_move]
        if self.is_move_node:
            if 'avoidMoves' in self.query:
                self.query['avoidMoves'][0]['untilDepth'] = max (1, self.query['avoidMoves'][0]['untilDepth'] - 1)
                self.query['avoidMoves'][1]['untilDepth'] = max (1, self.query['avoidMoves'][1]['untilDepth'] - 1)
                a = self.query['avoidMoves'][0]['untilDepth']
                b = self.query['avoidMoves'][1]['untilDepth']
                if a<=1 and b<=1:
                    del self.query['avoidMoves']
                elif self.get_root().random_drop and ((np.random.rand() < 1.0/a) and (len(self.parent.children) < K_BUDGET)):
                    del self.query['avoidMoves']
#                    print(f"added short-circuit node")
                    self.parent.children[(mkid(), 'pass' if action is None else action.which_move)] = None
        self.query['initialStones'] = _list_oc(self.board)

    @property
    def is_root(self):
        return self.parent == self

    @cached_property
    def is_move_node(self):
        return self.is_balanced

    @property
    def is_setup_node(self):
        return not self.is_move_node

    @cached_property
    def is_viable(self):
        if self.is_balanced and self.to_move == self.get_root().xdata['relevant_player']:
            self.get_root().last_good_node = self
            return True
        return False

    @cached_property
    def is_terminal(self):
        if self.is_viable and 'avoidMoves' not in self.query.keys():
            return True
        else:
            return False

    @cached_property
    def is_leaf(self):
        return 0 == len(self.children)

    @cached_property
    def is_balanced(self):
        match self.to_move:
            case 'B':
                return self.balance['B'] == self.balance['W']
            case 'W':
                return self.balance['B'] == (self.balance['W'] + 1)


    def dump_sgf(self,gain,penalty,self_winrate,pvs=None,name=None):
        d=defaultdict(list)
        for z in self.query['initialStones']:
            d[z[0]].append(mvv(z[1], 19))
        u=sgf.Sgf_game(19);
        r=u.get_root()
        r.set_setup_stones(d["B"],d["W"],[])
        r.add_comment_text(f"gain: {gain} penalty: {penalty} self_winrate: {self_winrate}")
        r.set('KM',self.query['komi'])
        r.set('TR',[mvv(self.xdata['comp'],19)])
#        if pvs:
#            a=u.extend_main_sequence()
#            color = self.xdata['relevant_player']
#            assert pvs[0][0] in self.either
#            for vertex in pvs[0]:
#                a.set_move(color.lower(), mvv(vertex,19))
#                color = xchg_color[color]
#            for pv in pvs[1:]:
#                assert pv[0] in self.either
#                color = self.xdata['relevant_player']
#                c = u.get_root().new_child()
#                for vertex in pv:
#                    c.set_move(color.lower(),mvv(vertex,19))
#                    color = xchg_color[color]
#                    c = c.new_child()
        pathname = f'output/{self.query['id'] if name is None else name}.sgf'
        with open(pathname,'wb') as f:
            f.write(u.serialise())
        print(f"dumped {pathname} ({gain}, penalty {penalty} winrate {self_winrate})")
   
    def __getitem__(self,child):
        return self.children[child]
    def __setitem__(self,child,value):
        self.children[child] = value

    @property
    def outcome(self,num_visits=1600):
        if self._outcome:
            return self._outcome
        assert self.is_viable
#######################
        katago = self.get_root().katago
        query = deepcopy(self.query)
        query['maxVisits'] = num_visits
        query['analysisPVLen'] = 8
        query['initialPlayer'] = self.xdata['relevant_player']
        self.full_output = katago.query_raw(query)
        self.winrate = self.full_output['rootInfo']['winrate']
        winrate_regularizer = 0.1 * (self.winrate - 0.5)**2
        #winrate_regularizer = 0.25 * (self.winrate - self.get_root().winrate)**2
        moveInfos = self.full_output['moveInfos']
        moveInfos = sorted(moveInfos, key=lambda x:x['order'])
        moveInfos_lcb = sorted(moveInfos, key=lambda x:x['utilityLcb'])
        lcb_range = moveInfos_lcb[0]['utilityLcb'] - moveInfos_lcb[-1]['utilityLcb']
        moveInfos_dict = { m['move']:m for m in moveInfos }
        assert moveInfos[0]['order'] == 0
#        best_score = moveInfos[0]['utilityLcb']
#######################
        comp,ref = self.xdata['comp'],self.xdata['ref']
        either = set([comp]).union(ref)
        self.either = either
        if comp not in moveInfos_dict:
            outcome = outcome_r = 0
        else:
            cmi = moveInfos_dict[comp]
#            print(f" either = {either}")
#            pvs = [ moveInfos_dict[k]['pv'] for k in moveInfos_dict if k in either]
#            pvks = [(k, moveInfos_dict[k]['pv']) for k in moveInfos_dict if k in either]
#            print(f"pvks = {pvks}")
            l = list(moveInfos_dict[k]['winrate'] for k in moveInfos_dict if k in either)
            #lcb_ = moveInfos_dict[comp]['utilityLcb']
            mu = 0 if len(l) == 0 else np.max(l)
            #outcome = 0.5*(lcb_ - mu)/lcb_range
            ref_visits = sum(moveInfos_dict[x]['visits'] for x in ref if x in moveInfos_dict)
            outcome_vs_ref1 = (cmi['visits'] / (cmi['visits'] + ref_visits))
            outcome_vs_ref2 = cmi['winrate'] - mu
            outcome_vs_ref = 0.7*outcome_vs_ref2 + 0.3 * outcome_vs_ref1
            outcome_vs_order = ((len(moveInfos) - cmi['order']) / len(moveInfos))**2
            outcome = 0.8*outcome_vs_order + 0.2*outcome_vs_ref

            outcome_r = outcome - winrate_regularizer

            if outcome_r > bests[-1]['value'] or moveInfos[0]['move'] == comp:

                if moveInfos[0]['move'] == comp:
                    prev_sol = solutions[-1]
                    solutions.append(dict())
                    solutions[-1]['num_solution'] = prev_sol['num_solution'] + 1
                    solutions[-1]['node'] = self
                    solutions[-1]['value'] = outcome_r
                    name = f"solution_{solutions[-1]['num_solution']}_{self.winrate:.3f}"
                else:
                    prev_best = bests[-1]
                    bests.append(dict())
                    bests[-1]['num_best'] = prev_best['num_best'] + 1
                    bests[-1]['node'] = self
                    bests[-1]['value'] = outcome_r
                    name = f"best_{bests[-1]['num_best']}_{self.winrate:.3f}"

                self.dump_sgf(outcome, winrate_regularizer, self.winrate, pvs = None, name=name)
        self._outcome = outcome_r
        leaf_statistics['count'] += 1
        count = leaf_statistics['count']
        last = leaf_statistics['last']
        now = timestamp()
        histo.add(outcome_r)
        if now - last > 20:
            print(f"{count} evals, {count/(now - leaf_statistics['first'])} per second")
            leaf_statistics['last'] = now
            histo.print()
        return outcome_r

    def get_tabooed_visits(self):
        s = sum(self.children[c].visits for c in self.children if c not in self.taboo_list and self.children[c])
        return max(1,s)
#    def get_tabooed_q_value(self):
#        tabooed_visits = self.get_tabooed_visits()
#        s = sum(self.children[c].q_value for c in self.children if c not in self.taboo_list)
#        return s

    def _child_values(self):
        tabooed_visits = self.get_tabooed_visits()
        def arf(child):
            if self[child] is None:
                mu = 0
                denominator = 1
            else:
                node = self[child]
                mu = node.q_value / (1 + tabooed_visits)
                denominator = 1 + node.visits
            return (mu,denominator)
        return [(child, arf(child)) for child in self.children if child not in self.taboo_list ]
    def select(self,k_budget=K_BUDGET,loss_thr=0.3,type='random'):
        if len(self.children) == 0:
            self._build_child_nodes(k_budget=k_budget,loss_thr=loss_thr)
        match type:
            case 'random':
                child = random_key({c:v for c,v in self.children.items() if c not in self.taboo_list})
            case 'UCT':
                child = uct(self.get_tabooed_visits(), self._child_values())
        return child
    def tree_policy(self):
        assert not self.is_terminal
        node = self
        i=0
        while not node.is_terminal:
#            assert len(self.children) > 0
            child = node.select(type='UCT')
            i += 1
            if node[child] is None:
                try:
                    node.expand(child)
                except MoveInfosException as e:
                    pass
                    #print(f"{e}: failed to expand node!")
#                    del self.children[child]
                return node[child]
            node = node[child]
        return node
    def apply(self):
        w=None
        v=None
        for i in range(8):
            try:
                v = self.tree_policy();
            except TreePolicyException as e:
                print(f"apply: TreePolicyException")
                pass
            except BorkException as e:
                print(f" bork exception (select -> _build_child_nodes -> _get_candidate_moves ")
                continue
            if not v:
                continue
            try:
                w = v.default_policy();
            except DefaultPolicyException as e:
                pass
            except BorkException as e:
                pass
            if w:
                w.backprop();
                return (True,None)
        return (False, v)

    def expand(self,child):
        xdata = deepcopy(self.xdata)
        if child[1] == 'pass':
            action = None
            xdata['passes'][self.to_move] -= 1
            assert xdata['passes']['B'] >= 0
            assert xdata['passes']['W'] >= 0
        else:
            action = Action(to_move=self.to_move,which_move=child[1])

        board_ =self.board.copy()
        node = Node(xdata,board_,child, parent=self,action=action)
        self[child]=node
        node._build_child_nodes()
    def default_policy(self,maxdepth = D_BUDGET):
        depth = 0
        node = self
        default_policy_stats['tries'] += 1
        default_policy_history = self.move_history()
        while not node.is_terminal:
            depth +=1
            if depth >= maxdepth:
                frac = default_policy_stats['success'] / default_policy_stats['tries']
                mh = node.move_history()
                blacks = [ x[1] for x in mh if x is not None and x[0] == 'B']
                whites = [ x[1] for x in mh if x is not None and x[0] == 'W']
#                print(f"exceeded depth budget (depth >= {maxdepth}) ({frac} success rate), bailing")
#                print(f"move history was {mh} (#B = {len(blacks)}, #W = {len(whites)})")
                last_good_node = self.get_root().last_good_node
                if last_good_node and last_good_node.is_ancestor(self):
                    return last_good_node
                else:
                    raise DefaultPolicyException('no good last node')
            try:
                child = node.select(type='random')
                default_policy_history.append(child[1])
            except MoveInfosException:
                print(f"default_policy: MoveInfosException 1; from node.select")
                raise DefaultPolicyException
            if node.children[child] is None:
                try:
                    node.expand(child)
                except MoveInfosException as e:
                    print(f"default_policy: MoveInfosException 2; from node.expand")
                    raise DefaultPolicyException
            node = node.children[child]
        if self.is_terminal:
            assert node == self
        default_policy_stats['success'] += 1
        return node
    def _get_candidate_moves(self,k_budget,loss_thr,num_visits=20):
        katago = self.get_root().katago
        query = deepcopy(self.query)
        query['maxVisits'] = num_visits
        if self.is_root:
            print(f"using many visits for root node")
            query['maxVisits'] = 3200
        #query['maxVisits'] = num_visits
        output = self.candidate_output = katago.query_raw(query)
        self.query_output = output
        self.winrate = output['rootInfo']['winrate']
        self.output_dict = dict()
        for moveInfo in output['moveInfos']:
            self.output_dict[moveInfo['move']] = moveInfo
        tops = top_k(output['moveInfos'],k_budget)
        best_score = tops[0]['scoreMean']
        moves = [ x['move'] for x in tops if x['scoreMean'] - best_score < loss_thr]
        if self.xdata['passes'][self.to_move] > 0:
            assert 'pass' not in moves
            moves.append('pass')
        forbidden = self.get_root().forbidden_vertices
        moves = [x for x in moves if x not in forbidden]
        if len(moves) <= 0:
            self.place_taboo()
            raise BorkException
        self.candidate_moves = moves
        self.candidate_pvs = { k:self.output_dict[k]['pv'] for k in moves if k != 'pass'}
#        uu = self.candidate_pvs
#        uu = [ x for x in uu if 0 == len(set(uu).intersection(forbidden)) ]
#        dist1 = sorted([f"{len(x)}" for x in self.candidate_pvs])
#        dist2 = sorted([f"{len(x)}" for x in uu])
#        print(f"{" ".join(dist1)} ({" ".join(dist2)}) ")
        if self.is_root:
            print(f"root node got {len(moves)} number of moves")
        return moves
    def place_taboo(self):
        self.parent.taboo_list.append(self.id)
        node = self.parent
        while len(node.children) == len(node.taboo_list):
            for x in node.taboo_list:
                assert x in node.children
            node.parent.taboo_list.append(node.id)
            node = node.parent
#            print(f"recursed in place_taboo")
    def _build_child_nodes(self,k_budget=8,loss_thr=0.3):
        assert self.is_leaf
        moves = self._get_candidate_moves(k_budget,loss_thr)
        for move in moves:
            self.children[(mkid(), move)] = None
    def backprop(self):
        self.get_root().katago
        assert self.is_viable
        delta_vp = self.outcome
        node = self
        while True:
            id = node.query['id']
            node.visits += 1
            node.q_value += delta_vp
            if node.is_root:
                break
            node = node.parent


