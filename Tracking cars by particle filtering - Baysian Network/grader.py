#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""


import random

import graderUtil
import util
import collections
import copy
grader = graderUtil.Grader()
exactInference = grader.load('exactInference')
particleFilter = grader.load('particleFilter')


############################################################
# Manual problems

# .addBasicPart is a basic test and always run.
#grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf'), 0)

############################################################
# Problem 5.1: Incorporating observations (10 points)

def test1a():
    ei = exactInference.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.requireIsEqual(0.030841805296, ei.belief.getProb(0, 0))
    grader.requireIsEqual(0.00073380582967, ei.belief.getProb(2, 4))
    grader.requireIsEqual(0.0269846478431, ei.belief.getProb(4, 7))
    grader.requireIsEqual(0.0129150762582, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.requireIsEqual(0.00000261584106271, ei.belief.getProb(0, 0))
    grader.requireIsEqual(0.000924335357194, ei.belief.getProb(2, 4))
    grader.requireIsEqual(0.0295673460685, ei.belief.getProb(4, 7))
    grader.requireIsEqual(0.000102360275238, ei.belief.getProb(5, 9))


grader.addBasicPart('5.1a-0', test1a, 10)


############################################################
# Problem 5.2: Incorporating transitions and observations (10 points)

def test2a():
    ei = exactInference.ExactInference(30, 13)

    # test 1
    ei.elapseTime()
    ei.belief.normalize()
    grader.requireIsEqual(0.0105778989624, ei.belief.getProb(16, 6))
    grader.requireIsEqual(0.00250560512469, ei.belief.getProb(18, 7))
    grader.requireIsEqual(0.0165024135157, ei.belief.getProb(21, 7))
    grader.requireIsEqual(0.0178755550388, ei.belief.getProb(8, 4))

    # test 2
    ei.elapseTime()
    ei.belief.normalize()
    grader.requireIsEqual(0.0138327373012, ei.belief.getProb(16, 6))
    grader.requireIsEqual(0.00257237608713, ei.belief.getProb(18, 7))
    grader.requireIsEqual(0.0232612833688, ei.belief.getProb(21, 7))
    grader.requireIsEqual(0.0176501876956, ei.belief.getProb(8, 4))

grader.addBasicPart('5.2a-0', test2a, 10)


############################################################
# Problem 5.3: Particle Filtering (30 points)

def test3a_0():
    random.seed(3)
    pf = particleFilter.ParticleFilter(30, 13)

    pf.observe(555, 193, 800)

    grader.requireIsEqual(0.005, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.045, pf.belief.getProb(21, 5))
    grader.requireIsEqual(0.95, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(8, 4))
    
    pf.observe(525, 193, 830)

    grader.requireIsEqual(0.0, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.0, pf.belief.getProb(21, 5))
    grader.requireIsEqual(1.0, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(8, 4))
    

grader.addBasicPart('5.3a-0', test3a_0, 10)

def test3a_1():
    random.seed(3)
    pf = particleFilter.ParticleFilter(30, 13)
    grader.requireIsEqual(69, len(pf.particles)) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.requireIsEqual(200, sum(pf.particles.values())) # Do not lose particles
    grader.requireIsEqual(60, len(pf.particles)) # Most particles lie on the same (row, col) locations

    grader.requireIsEqual(7, pf.particles[(3,9)])
    grader.requireIsEqual(0, pf.particles[(2,10)])
    grader.requireIsEqual(6, pf.particles[(8,4)])
    grader.requireIsEqual(0, pf.particles[(12,6)])
    grader.requireIsEqual(2, pf.particles[(7,8)])
    grader.requireIsEqual(0, pf.particles[(11,6)])
    grader.requireIsEqual(2, pf.particles[(18,7)])
    grader.requireIsEqual(1, pf.particles[(20,5)])

    pf.elapseTime()
    grader.requireIsEqual(200, sum(pf.particles.values())) # Do not lose particles
    grader.requireIsEqual(57, len(pf.particles)) # Slightly more particles lie on the same (row, col) locations

    grader.requireIsEqual(6, pf.particles[(3,9)])
    grader.requireIsEqual(0, pf.particles[(2,10)]) # 0 --> 0
    grader.requireIsEqual(7, pf.particles[(8,4)])
    grader.requireIsEqual(0, pf.particles[(12,6)])
    grader.requireIsEqual(1, pf.particles[(7,8)])
    grader.requireIsEqual(2, pf.particles[(11,6)])
    grader.requireIsEqual(0, pf.particles[(18,7)]) # 0 --> 1
    grader.requireIsEqual(3, pf.particles[(20,5)]) # 1 --> 0

grader.addBasicPart('5.3a-1', test3a_1, 10)

def test3a_2():
    random.seed(3)
    pf = particleFilter.ParticleFilter(30, 13)
    grader.requireIsEqual(69, len(pf.particles)) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.requireIsEqual(60, len(pf.particles)) # Most particles lie on the same (row, col) locations
    pf.observe(555, 193, 800)

    grader.requireIsEqual(200, sum(pf.particles.values())) # Do not lose particles
    grader.requireIsEqual(3, len(pf.particles)) # Most particles lie on the same (row, col) locations
    grader.requireIsEqual(0.075, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.1, pf.belief.getProb(21, 5))
    grader.requireIsEqual(0.0, pf.belief.getProb(21, 6))
    grader.requireIsEqual(0.825, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(22, 7))

    pf.elapseTime()
    grader.requireIsEqual(6, len(pf.particles)) # Most particles lie on the same (row, col) locations

    pf.observe(660, 193, 50)
    grader.requireIsEqual(0.0, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.0, pf.belief.getProb(21, 5))
    grader.requireIsEqual(0.18, pf.belief.getProb(21, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.8, pf.belief.getProb(22, 7))

grader.addBasicPart('5.3a-2', test3a_2, 10)
grader.grade()
