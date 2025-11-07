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
ROOT_VISITS=6400
TERMINAL_VISITS=1600
CANDIDATE_VISITS=40
OUTCOME_PVLEN=8
CANDIDATE_PVLEN=8
LOSS_THRESHOLD = 0.3    # how bad of a move we accept to search, in point loss vs preferred move


visit_weight = 0.3
winrate_weight = 1.0-visit_weight
soft_outcome_weight = 0.2
hard_outcome_weight = 1.0 - soft_outcome_weight
winrate_lambda = 0.8

from copy import deepcopy
from functools import lru_cache,cached_property
from util import top_k,mkid,uct,random_key,xchg_color,_list_oc,timestamp,parity_player
from sgfmill.common import move_from_vertex as mvv
from sgfmill import boards,sgf
import numpy as np
default_policy_stats = dict(tries=0, success = 0)
bests = [dict(num_best=0,node=None,value=0)]
solutions = [dict(num_solution=0,node=None,value=0)]
from collections import defaultdict, deque

now = timestamp()
leaf_statistics = dict(count=0,last_histo=now, last_tree=now, first=now)

fail_counts = defaultdict(int)

from stats import *
from histo import CLIHistogram
histo = CLIHistogram(12,0.7,1.0)

class Action:
    def __init__(self,to_move=None, which_move=None):
        self.to_move = to_move
        self.which_move = which_move
        self.kg_query_format = [to_move,which_move]

def delete_tree(root):
    if root.is_leaf:
        if hasattr(root,'until_depth'):
            del root.until_depth;
        if hasattr(root,'parent'):
            root.parent = None
        if hasattr(root,'xdata'):
            del root.xdata
        if hasattr(root,'board'):
            del root.board
        if hasattr(root,'action'):
            del root.action
        if hasattr(root,'id'):
            del root.id
        if hasattr(root,'query'):
            del root.query
        if hasattr(root,'balance'):
            del root.balance
    for k,v in root.children.items():
        if v is not None:
            assert isinstance(v,Node)
            delete_tree(v)
    del root.children
    if hasattr(root,'taboo_list'):
        del root.taboo_list

class Node:
    def __init__(self, board, id, parent=None, parent_query = None, action=None,xdata=None):
        self.board = board
        self.action = action
        self.taboo_list = []    #XXX: deprecated, just delete the corresponding child entry(?)
        self.id = id
        if parent is None:
            self.relevant_player = xdata['relevant_player']
            self.passes = deepcopy(xdata['passes'])
            assert parent_query is not None
            self.until_depth=dict()
            if 'B' != parent_query['avoidMoves'][0]['player']:
                parent_query['avoidMoves'][0],parent_query['avoidMoves'][1] = parent_query['avoidMoves'][1],parent_query['avoidMoves'][0]
            assert 'B' == parent_query['avoidMoves'][0]['player']
            self.until_depth['B'] = parent_query['avoidMoves'][0]['untilDepth']
            self.until_depth['W'] = parent_query['avoidMoves'][1]['untilDepth']
            self.query = parent_query
            self.parent = self
            self.to_move = xdata['relevant_player']
            self.last_good_node = None

            comp,ref = xdata['comp'],xdata['ref']
            either = set([comp]).union(ref)
            self.either = either
            self.comp = comp
            self.ref = ref
        else:
            self.passes = deepcopy(parent.passes)
            self.until_depth = deepcopy(parent.until_depth)
            self.parent = parent
            self.to_move = xchg_color[parent.to_move]
            self.balance = deepcopy(parent.balance)
            if action:
                self.balance[action.to_move] += 1
            else:
                self.passes[parent.to_move] -= 1
                assert self.passes[parent.to_move] >= 0
            self.until_depth['B'] = max (0, self.until_depth['B'] - 1)
            self.until_depth['W'] = max (0, self.until_depth['W'] - 1)
            if action is None:
                pass
            else:
                self.board.play(*mvv(action.which_move,19),action.to_move.lower())
                assert self.to_move == xchg_color[action.to_move]
            a = self.until_depth['B']
            b = self.until_depth['W']
            rv = np.random.rand()
            if self.is_balanced and (max(a,b) > 0) and self.get_root().random_drop and rv < 2.0/(a+b) and len(self.parent.children) < K_BUDGET:
                self.parent.children[(mkid(), 'pass' if action is None else action.which_move)] = None  # XXX
                self.until_depth['B'] = 0
                self.until_depth['W'] = 0
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


    @property
    def is_root(self):
        return self.parent == self


    @cached_property
    def is_viable(self):
        if self.is_balanced and self.to_move == self.get_root().relevant_player:
            self.get_root().last_good_node = self
            assert self.passes['W'] == 0 and self.passes['B'] == 0
            return True
        return False

    @cached_property
    def is_terminal(self):
        a = self.until_depth['B']
        b = self.until_depth['W']
        if self.is_viable and max(a,b) == 0:
            return True
        else:
            return False

    @cached_property
    def is_leaf(self):
        return 0 == len(self.children)

    @cached_property
    def is_balanced(self):
        return self.balance['B'] == (self.balance['W'] + parity_player[self.to_move])

    def get_updated_query(self):
        action = self.action
        query = deepcopy(self.get_root().query)
        query['id'] = mkid()
        query['initialPlayer'] = self.to_move
        if self.is_balanced and 'avoidMoves' in query:
            query['avoidMoves'][0]['untilDepth'] = self.until_depth['B']
            query['avoidMoves'][1]['untilDepth'] = self.until_depth['W']
            if self.until_depth['B'] == 0 and self.until_depth['W'] == 0:
                del query['avoidMoves']
        query['initialStones'] = _list_oc(self.board)
        return query

    def dump_sgf(self,gain,penalty,self_winrate,pvs=None,name=None):
        d=defaultdict(list)
        query=self.get_updated_query()
        for z in query['initialStones']:
            d[z[0]].append(mvv(z[1], 19))
        u=sgf.Sgf_game(19);
        r=u.get_root()
        r.set_setup_stones(d["B"],d["W"],[])
        r.add_comment_text(f"gain: {gain} penalty: {penalty} self_winrate: {self_winrate}")
        r.set('KM',query['komi'])
        r.set('TR',[mvv(self.get_root().comp,19)])
        pathname = f'output/{query['id'] if name is None else name}.sgf'
        with open(pathname,'wb') as f:
            f.write(u.serialise())
        print(f"dumped {pathname} ({gain}, penalty {penalty} winrate {self_winrate})")
   
    def __getitem__(self,child):
        return self.children[child]
    def __setitem__(self,child,value):
        self.children[child] = value    #XXX

    def mk_outcome_query(self,num_visits=TERMINAL_VISITS):
        query = self.get_updated_query()
        query['maxVisits'] = num_visits
        query['analysisPVLen'] = OUTCOME_PVLEN
        query['initialPlayer'] = self.get_root().relevant_player
        return query

    def parse_outcome_query(self,full_output):
        winrate = full_output['rootInfo']['winrate']
        moveInfos = sorted(full_output['moveInfos'],key=lambda x:x['order'])
        r=self.get_root()
        either,comp = r.either,r.comp
        moveInfos_dict = { m['move']:m for m in moveInfos }
        pvs = [ moveInfos_dict[k]['pv'] for k in moveInfos_dict if k in either]
        pvks = [(k, moveInfos_dict[k]['pv']) for k in moveInfos_dict if k in either]
        evaluable_node = comp in moveInfos_dict
        is_solution = (comp == moveInfos[0]['move'])
        return dict(winrate=winrate,
                    MI=moveInfos,
                    MID=moveInfos_dict,
                    pvs=pvs,
                    pvks=pvks,
                    evaluable_node=evaluable_node,
                    is_solution = is_solution)
    def _outcome_from_parsed(self,parsed):
        winrate_regularizer = winrate_lambda * (parsed['winrate'] - 0.5)**2
        r=self.get_root()
        either,comp,ref = r.either,r.comp,r.ref
        cmi = parsed['MID'][comp]
        l = list(parsed['MID'][k]['winrate'] for k in parsed['MID'] if k in either)
        mu = 0 if len(l) == 0 else np.max(l)
        ref_visits = sum(parsed['MID'][x]['visits'] for x in ref if x in parsed['MID'])
        visits_vs_ref = (cmi['visits'] / (cmi['visits'] + ref_visits))
        winrate_vs_ref = cmi['winrate'] - mu
        outcome_vs_ref = winrate_weight*winrate_vs_ref + visit_weight * visits_vs_ref
        outcome_vs_order = ((len(parsed['MI']) - cmi['order']) / len(parsed['MI']))
        outcome = hard_outcome_weight*outcome_vs_order + soft_outcome_weight*outcome_vs_ref

        outcome_r = outcome - winrate_regularizer
        #this kind of "cheating" just distorts the evaluations too much; it is better
        #to shape the reward appropriately so that order == 0 in a natural way gets
        #a very high reward. Otherwise what seems to happen is that the winrate is all over
        #the place in a way it isn't if we just find solutions naturally without this type of
        #nudge.
        #outcome_r = (np.random.uniform()*1e-6+bests[-1]['value']) if parsed['is_solution'] else outcome_r
        return outcome,winrate_regularizer,outcome_r

    def _report_trigger(self,outcome_r,winrate,is_solution=None):
        if is_solution:
            prev_sol = solutions[-1]
            solutions.append(dict())
            solutions[-1]['num_solution'] = prev_sol['num_solution'] + 1
            solutions[-1]['node'] = self
            solutions[-1]['value'] = outcome_r
            name = f"solution_{solutions[-1]['num_solution']}_{winrate:.3f}"
        else:
            prev_best = bests[-1]
            bests.append(dict())
            bests[-1]['num_best'] = prev_best['num_best'] + 1
            bests[-1]['node'] = self
            bests[-1]['value'] = outcome_r
            name = f"best_{bests[-1]['num_best']}_{winrate:.3f}"
        return name

    @property
    def outcome(self,num_visits=TERMINAL_VISITS):
        if self._outcome:
            return self._outcome
        assert self.is_viable
        katago = self.get_root().katago
        qy = self.mk_outcome_query(num_visits)
        self.full_output = katago.query_raw(self.mk_outcome_query(num_visits))
        parsed = self.parse_outcome_query(self.full_output)
        self.winrate = parsed['winrate']
        if not parsed['evaluable_node']:
            outcome = outcome_r = 0
        else:
            outcome,winrate_regularizer,outcome_r = self._outcome_from_parsed(parsed)

            if outcome_r > bests[-1]['value'] or parsed['is_solution']:

                name = self._report_trigger(outcome_r,
                                            self.winrate,
                                            is_solution = parsed['is_solution'])
                self.dump_sgf(outcome, winrate_regularizer, self.winrate, pvs = None, name=name)
        self._outcome = outcome_r
        self._histo_trigger(timestamp(),outcome_r)
        return outcome_r
    def _histo_trigger(self,now, outcome_r, max_delta=20):
        leaf_statistics['count'] += 1
        count = leaf_statistics['count']
        histo.add(outcome_r)
        last_histo = leaf_statistics['last_histo']
        if now - last_histo > max_delta:
            print(f"{count} evals, {count/(now - leaf_statistics['first'])} per second")
            leaf_statistics['last_histo'] = now
            histo.print()
    def get_tabooed_visits(self):
        if len(self.taboo_list) == 0:
            return self.visits
        s = sum(self.children[c].get_tabooed_visits() for c in self.children if c not in self.taboo_list and self.children[c])
        return max(1,s)

    def _child_values(self):
        tabooed_visits = self.get_tabooed_visits()
        def arf(child):
            if self[child] is None:
                mu = 0
                denominator = 1
            else:
                node = self[child]
                mu = node.q_value / (1 + node.get_tabooed_visits())
                denominator = 1 + node.visits
            return (mu,denominator)
        if len(self.taboo_list) > 0:
            print(self.taboo_list)
            print(self.children.keys())
        for child in self.children:
            assert child not in self.taboo_list
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
        n=64
        for i in range(n):
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
            else:
                print(f" {i}/{n}")
        return (False, v)

    def expand(self,child):
        if child[1] == 'pass':
            action = None
        else:
            action = Action(to_move=self.to_move,which_move=child[1])

        board_ =self.board.copy()
        node = Node(board_,child, parent=self,action=action,xdata=None)
        self[child]=node
        node._build_child_nodes()
        if all(x is not None for x in self.children.values()):
            del self.board
            self.board = None
    def default_policy(self,maxdepth = D_BUDGET):
        depth = 0
        node = self
        default_policy_stats['tries'] += 1
        default_policy_history = self.move_history()
        while not node.is_terminal:
            depth +=1
            if depth >= maxdepth:
#                frac = default_policy_stats['success'] / default_policy_stats['tries']
#                print(f"exceeded depth budget (depth >= {maxdepth}) ({frac} success rate), bailing")
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
    def _get_candidate_moves(self,k_budget,loss_thr,num_visits=CANDIDATE_VISITS):
        katago = self.get_root().katago
        query = self.get_updated_query()
        query['maxVisits'] = num_visits
        if self.is_root:
            print(f"using many visits for root node")
            query['maxVisits'] = ROOT_VISITS
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
        if self.passes[self.to_move] > 0:
            assert 'pass' not in moves
            moves.append('pass')
        forbidden = self.get_root().forbidden_vertices
        moves = [x for x in moves if x not in forbidden]
        if len(moves) <= 0:
            self.place_taboo()
            raise BorkException
        self.candidate_moves = moves
        self.candidate_pvs = { k:self.output_dict[k]['pv'] for k in moves if k != 'pass'}
        if self.is_root:
            print(f"root node got {len(moves)} number of moves")
        return moves
    def place_taboo(self):
        print("placing taboo")
        self.parent.taboo_list.append(self.id)
        node = self.parent
        while len(node.children) == len(node.taboo_list):
            assert all(x in node.children for x in node.taboo_list)
            node.parent.taboo_list.append(node.id)
            node = node.parent
            print(f"recursing in place_taboo")
    def _build_child_nodes(self,k_budget=8,loss_thr=LOSS_THRESHOLD):
        assert self.is_leaf
        moves = self._get_candidate_moves(k_budget,loss_thr)
        for move in moves:
            self.children[(mkid(), move)] = None
    def backprop(self):
        assert self.is_viable
        delta_vp = self.outcome
        node = self
        while True:
            node.visits += 1
            node.q_value += delta_vp
            if node.is_root:
                break
            node = node.parent



