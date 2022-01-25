# -*- coding: utf-8 -*-

################
# DO NOT REMOVE
# Versions
# numpy==1.18.0
################
import numpy as np

class TDAgent(object):
    def __init__(self):
        pass
       
    def get_expec_value(self):
        expec_value = [0]* len(self.state_values)
        expec_rewards = [0]* len(self.state_values)
        one_minus_proba = 1-self.proba
        # expec_value[0] = self.state_values[0]
        expec_rewards[1] = self.proba * self.state_returns[0] + one_minus_proba * self.state_returns[1]
        expec_value[1] = self.proba * self.state_values[1] + one_minus_proba * self.state_values[2] + expec_rewards[1]
        expec_rewards[2] = self.proba*(self.state_returns[0] + self.state_returns[2]) + one_minus_proba * (self.state_returns[1]+self.state_returns[3])
        expec_value[2] =  self.state_values[3] + expec_rewards[2]
        for state_num in range(3 , 6): # from 3 to last state: 5
            expec_rewards[state_num] =  expec_rewards[state_num -1] + self.state_returns[state_num + 1 ]
            expec_value[state_num ] =  expec_rewards[state_num ] + self.state_values[state_num +1 ]
        return expec_value

    def get_error(self, lambda_value, expec_value):
        error = 0
        one_minus_lbd = 1 - lambda_value
        for k in range(1,5): # from 1 to 4
            error += lambda_value **(k - 1) * expec_value[k]
        error = error *  one_minus_lbd
        error += (lambda_value**4-1) * expec_value[5]
        return error

    def solve(self,p,V,rewards):
        self.proba= p
        self.state_values = V
        self.state_returns= rewards
        try_lambda = 0.5  # initial value of lambda
        thera = 0.002
        step = 0.1
        expec_value = self.get_expec_value()
        last_sign_dirt = 1
        while True:
            error_now = self.get_error( try_lambda,expec_value)
            grat = (self.get_error(try_lambda + step,expec_value) - error_now)/step
            sign_dirt = np.sign(grat) * np.sign(error_now)
            if abs(error_now) < thera:
                return np.round(try_lambda,3)
            else:
                if (sign_dirt != last_sign_dirt):
                    step = 0.5 * step
                try_lambda += -step* sign_dirt                  
                last_sign_dirt = sign_dirt


## DO NOT MODIFY THIS CODE.  This code will ensure that your submission
## will work proberly with the autograder

import unittest

class TestTDNotebook(unittest.TestCase):
    def test_case_1(self):
        agent = TDAgent()
        np.testing.assert_almost_equal(
            agent.solve(
                p=0.81,
                V=[0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
                rewards=[7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6]
            ),
            0.622,
            decimal=3
        )
        
    def test_case_2(self):
        agent = TDAgent()
        np.testing.assert_almost_equal(
            agent.solve(
                p=0.22,
                V=[12.3, -5.2, 0.0, 25.4, 10.6, 9.2, 0.0],
                rewards=[-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1]
            ),
            0.519,
            decimal=3
        )
        
    def test_case_3(self):
        agent = TDAgent()

        np.testing.assert_almost_equal(
            agent.solve(
                p=0.64,
                V=[-6.5, 4.9, 7.8, -2.3, 25.5, -10.2, 0.0],
                rewards=[-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9]
            ),
            0.207,
            decimal=3
        )

unittest.main(argv=[''], verbosity=2, exit=False)

