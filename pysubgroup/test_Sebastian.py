import pandas as pd
import pysubgroup as ps
import pickle as pickle
windows_prefix= 'E:/tmp/'
linux_prefix = '/home/felixstamm/'

prefix = linux_prefix
df = pd.read_pickle(prefix+'dump_felix.pkl.gzip', compression='gzip')
path = '/home/felix/selectors.pickle'
selectors = ps.create_selectors(df, ignore=['address','balance_2019-04'])
#with open(path,'wb') as f:
#    pickle.dump(selectors, f)
#with open('/home/felix/selectors.pickle','rb') as f:
#    selectors = pickle.load(f)

qf = ps.StandardQFNumeric(0.5, estimator='order')
qf.min_size = 100
task = ps.SubgroupDiscoveryTask(df, ps.NumericTarget('balance_2019-04'), selectors, qf, 10, 3)
algo = ps.Apriori(show_progress=True, use_parallelisation=True)
result = algo.execute(task)
for q, sg in result:
    print(q, sg)