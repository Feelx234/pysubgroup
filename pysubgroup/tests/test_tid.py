from scipy.io import arff

import pysubgroup as ps
import pandas as pd
from timeit import default_timer as timer


data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NominalTarget('class', b'bad')
searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, result_set_size=10, depth=5, qf=ps.ChiSquaredQF())


start = timer()
result = ps.BSD().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))


print ("******")
start = timer()
result = ps.TID_SD(use_sets=True).execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))


print ("******")
start = timer()
result = ps.TID_SD().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))
