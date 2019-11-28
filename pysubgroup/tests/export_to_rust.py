import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
import numpy as np
def setUp():
    NS_telephone = ps.NominalSelector("own_telephone", b"yes")
    NS_foreign_worker = ps.NominalSelector("foreign_worker", b"yes")
    NS_other_parties = ps.NominalSelector("other_parties", b"none")
    NS_personal = ps.NominalSelector("personal_status", b'male single')
    NS_job = ps.NominalSelector("job", b'high qualif/self emp/mgmt')
    NS_class = ps.NominalSelector("class", b"bad")

    o = [[NS_telephone],
            [NS_foreign_worker, NS_telephone],
            [NS_other_parties, NS_telephone],
            [NS_foreign_worker, NS_telephone, NS_personal],
            [NS_telephone, NS_personal],
            [NS_foreign_worker, NS_other_parties, NS_telephone],
            [NS_job],
            [NS_class, NS_telephone],
            [NS_foreign_worker, NS_job],
            [NS_foreign_worker, NS_other_parties, NS_telephone, NS_personal]
            ]
    result = list(map(ps.Conjunction, o))
    qualities = [
        383476.7679999999,
        361710.05800000014,
        345352.9920000001,
        338205.08,
        336857.8220000001,
        323586.28200000006,
        320306.81600000005,
        300963.84599999996,
        299447.332,
        297422.98200000013]

    data = get_credit_data()
    for _ in range(10):
        data = data.append(data, ignore_index = True)
    target = ps.NumericTarget('credit_amount')
    searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['credit_amount'])
    searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
    searchSpace = searchSpace_Nominal + searchSpace_Numeric
    return ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(1, False, 'sum')))
task = setUp()
task.data = task.data.sort_values(by=task.target.target_variable, ascending=False)
arr = np.array([sel.covers(task.data) for sel in task.search_space], dtype=bool)
print(arr.shape)
target = np.array(task.data[task.target.target_variable].to_numpy(), dtype = np.uint16)
print(np.sum(np.abs(task.data[task.target.target_variable].to_numpy()-target)))
print(np.sum(target))
if True:
    with open(r'E:\tmp\target.npy','wb') as f:
        np.save(f, target)

    with open(r'E:\tmp\arr.npy','wb') as f:
        np.save(f, arr)

print(task.search_space[65])
import time
start_time = time.time()
result = ps.DFSNumeric().execute(task)
print("--- %s seconds ---" % (time.time() - start_time))
print(result)
print(np.count_nonzero(result[0][1].covers(task.data)))