import unittest
import numpy as np
from experiment import sieve, layouts, board, valid_origins, distances

class ResearchTests(unittest.TestCase):
    def test_known_primes(self):
        self.assertEqual(np.flatnonzero(sieve(30)).tolist(),[2,3,5,7,11,13,17,19,23,29])
        self.assertEqual(int(sieve(100000).sum()),9592)
    def test_spiral(self):
        xy=layouts(25)['ulam']
        self.assertEqual(xy[1:10].tolist(),[[0,0],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
        self.assertEqual(len(set(map(tuple,xy[1:]))),25)
    def test_against_exhaustive_square(self):
        labels=sieve(1000)
        xy=layouts(1000)['ulam']; g,xy=board(xy,labels)
        anchors=np.flatnonzero(labels & valid_origins(g,xy,4))
        for mode in ('corner','boundary'):
            measured=distances(g,xy,anchors,labels,mode,4)
            for i,a in enumerate(anchors):
                x,y=xy[a]
                for j,(sx,sy) in enumerate(((1,1),(-1,1),(-1,-1),(1,-1))):
                    expected=5
                    for r in range(1,5):
                        points=[(r,r)] if mode=='corner' else [(u,v) for u in range(r+1) for v in range(r+1) if u or v]
                        if any(labels[g[x+sx*u,y+sy*v]] for u,v in points):
                            expected=r; break
                    self.assertEqual(measured[i,j],expected)
    def test_empty_search_is_censored(self):
        xy=layouts(1000)['ulam']; labels=np.zeros(1001,bool); g,xy=board(xy,labels)
        anchors=np.flatnonzero(valid_origins(g,xy,3))
        self.assertTrue(np.all(distances(g,xy,anchors,labels,'boundary',3)==4))

if __name__=='__main__': unittest.main()
