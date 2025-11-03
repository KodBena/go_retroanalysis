import json
from util import bm, mk_sgf_letters_bimap
def mkrect(ref_point, width, height):
    bm = mk_sgf_letters_bimap()
    x0 = bm[ref_point[0]]
    y0 = int(ref_point[1])
    out = []
    for col in range(width):
        for row in range(height):
            npx = bm[x0 + col]
            npy = str(y0 + row)
            out.append(f'{npx}{npy}')
    return out

class Problem:
    def __init__(self,sgf):
        d = json.loads(sgf.get_root().get('C'))
        self.comparison_query = d['comp']
        self.reference_query = d['ref'][0].split(',')
        #self.reference_query = d['ref']
        self.avoid = dict()
        for player in ['B','W']:
            avoid = d['avoid'][player]
            u = set()
            if 'points' in avoid:
                u = u.union(avoid['points'])
            if 'rects' in avoid:
                u = u.union(sum([mkrect(*x) for x in avoid['rects']],[]))
            self.avoid[player] = dict(moves=sorted(list(u)), untilDepth=avoid['depth'])

        self.relevant_player = d['player']

        initials = []
        initial_counts = dict()
        initial_dict = dict()
        for setup_color in ["B","W"]:
            w = sgf.root.get("A"+setup_color)
            initial_counts[setup_color] = len(w)
            initial_dict[setup_color] = [ [setup_color, f"{bm[y+1].upper()}{1+x}"] for (x,y) in w ]

            initials += [ [setup_color, f"{bm[y+1].upper()}{1+x}"] for (x,y) in w ]
        self.initials = initials
        self.initial_dict = initial_dict
        self.initial_counts = initial_counts
        black_passes = white_passes = 0
        match self.relevant_player:
            case 'B':
                if initial_counts['B'] != initial_counts['W']:
                    delta = initial_counts['B'] - initial_counts['W']
                    if delta > 0:
                        black_passes = delta
                    else:
                        white_passes = -delta
            case 'W':
                if initial_counts['B'] != initial_counts['W'] + 1:
                    delta = initial_counts['B'] - (initial_counts['W'] + 1)
                    if delta > 0:
                        black_passes = delta
                    else:
                        white_passes = -delta
        self.passes = dict(B=black_passes,W=white_passes)

    @property
    def avoidMoves(self):
        d=self.avoid
        return [dict(player=player, moves=d[player]['moves'], untilDepth=d[player]['untilDepth']) for player in ['B','W']]
