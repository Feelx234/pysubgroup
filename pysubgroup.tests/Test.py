import cProfile
import heapq
import time

from scipy.io import arff

from Selectors import NominalSelector
import SubgroupDiscoveryTask, Selectors, BSD, SimpleDFS


data = arff.loadarff('C:/test/credit-g.arff') [0]
# print meta['class']
target = NominalSelector ('class', b'bad')

# sel2 = NominalSelector ('checking_status', 'no checking')

# sg = Subgroup(NominalSelector ('class', "bad"), [NominalSelector ('checking_status', "'no checking'")])
# qf = WRAccQF()
# print qf.evaluateFromDataset(data, sg)
# print sel.covers(data[0])
# print sum(1 for i in data if sel.covers(i))

searchSpace = Selectors.createSelectors(data, intervals_only=False)
searchSpace = [x for x in searchSpace if x.attributeName != target.attributeName]
print (searchSpace)

task = SubgroupDiscoveryTask.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=3)
algo = SimpleDFS.SimpleDFS()
start_time = time.clock()
result = algo.execute(task)
stop_time = time.clock()
print (stop_time - start_time, "seconds")
# cProfile.run('result = algo.execute(task)')

print (result)
result = heapq.nlargest(len(result), result)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))   

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
