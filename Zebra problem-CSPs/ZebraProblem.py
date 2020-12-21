import submission
csp = submission.zebra_csp()
alg = submission.BacktrackingSearch()
alg.solve(csp)
#print(csp.domains())
print (alg.optimalAssignment)