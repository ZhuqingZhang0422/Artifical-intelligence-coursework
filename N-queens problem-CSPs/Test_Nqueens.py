import submission
csp = submission.create_nqueens_csp()
alg = submission.BacktrackingSearch()
alg.solve(csp)
print (alg.optimalAssignment)