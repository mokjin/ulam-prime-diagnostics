import unittest
import numpy as np
from experiment import layouts
from path_structure import value,segments,signature

class PathTests(unittest.TestCase):
    def test_inverse_against_walk(self):
        xy=layouts(10000)['ulam']
        for n in range(1,10001): self.assertEqual(value(*map(int,xy[n])),n)
    def test_paths_cross_axes_and_ring_corners(self):
        for x in range(-5,6):
            for y in range(-5,6):
                for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)):
                    parts=segments(x,y,sx,sy,30)
                    self.assertEqual([r for p in parts for r in range(p['start'],p['end']+1)],list(range(1,31)))
                    for p in parts:
                        a,b,c=p['coefficients']
                        for r in range(p['start'],p['end']+1): self.assertEqual(a*r*r+b*r+c,value(x+sx*r,y+sy*r))
    def test_same_occupancy_different_removal(self):
        a=signature([7,13]); b=signature([9,15])
        self.assertEqual(a['occupied_residues']['3'],b['occupied_residues']['3'])
        self.assertEqual(a['removed_per_prime']['3'],0)
        self.assertEqual(b['removed_per_prime']['3'],2)
    def test_small_primes_and_overlap(self):
        s=signature([2,3,5,7,11,30,49])
        self.assertEqual(s['survivors'],5)
        self.assertEqual(s['removal_mask_counts'][7],1)
        self.assertEqual(sum(s['removal_mask_counts']),7)

if __name__=='__main__': unittest.main()
