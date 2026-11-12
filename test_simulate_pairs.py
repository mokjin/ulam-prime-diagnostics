import unittest
import numpy as np
from simulate_pairs import restricted_time,contrast_weights

class SimulationTests(unittest.TestCase):
    def test_censoring_and_first_hit(self):
        hits=np.zeros((4,20),bool);hits[0,0]=True;hits[1,19]=True;hits[3,4:7]=True
        np.testing.assert_array_equal(restricted_time(hits),[1,20,20,5])
    def test_contrast_is_difference_of_group_means(self):
        n=np.full(5,2);direction=np.zeros(5);band=np.zeros(5);score=np.array([0,0,1,1,1])
        w,cells=contrast_weights(n,direction,band,score,minimum=2)
        residual=np.array([1,3,2,4,6])
        self.assertAlmostEqual(w@residual,2)
        self.assertAlmostEqual(w.sum(),0)
        self.assertEqual(len(cells),1)
    def test_no_common_support(self):
        w,cells=contrast_weights(np.full(3,2),np.zeros(3),np.zeros(3),np.zeros(3),minimum=1)
        self.assertEqual(cells,[]);np.testing.assert_array_equal(w,0)
    def test_shared_integer_consistency(self):
        numbers=np.array([[101,103],[107,101]])
        unique,index=np.unique(numbers,return_inverse=True)
        values=np.array([True,False,False])
        hits=values[index].reshape(numbers.shape)
        self.assertEqual(hits[0,0],hits[1,1])

if __name__=='__main__':unittest.main()
