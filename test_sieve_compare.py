import unittest
import numpy as np
from sieve_compare import probability,survival,hazard
from path_structure import segments

class ModelTests(unittest.TestCase):
    def test_conditional_density(self):
        self.assertAlmostEqual(probability(np.array([101]),(2,3,5))[0],3.75/np.log(101))
        self.assertAlmostEqual(probability(np.array([101]),(2,3,5,7,11))[0],min(1,4.8125/np.log(101)))
    def test_prime_exceptions(self):
        np.testing.assert_array_equal(probability(np.array([1,2,3,5,7,11,30,49]),(2,3,5,7,11)),[0,1,1,1,1,1,0,0])
    def test_hazard_missing(self):
        np.testing.assert_allclose(hazard(np.array([1,.5,0,0])),[.5,1,np.nan],equal_nan=True)
    def test_survival_product(self):
        n=np.array([[101,103]])
        q=probability(n,())
        np.testing.assert_allclose(survival(n,(),1),[[1,1-q[0,0],(1-q[0,0])*(1-q[0,1])]])
    def test_segment_finite_counts(self):
        for part in segments(1,1,-1,-1,20):
            a,b,c=part['coefficients']; vals=[a*r*r+b*r+c for r in range(part['start'],part['end']+1)]
            for ell in (2,3,5,7,11):
                count=sum(n>ell and n%ell==0 for n in vals)
                self.assertEqual(part['finite_removed_count'][str(ell)],count)
                self.assertEqual(part['finite_removed_rate'][str(ell)],count/len(vals))

if __name__=='__main__':unittest.main()
