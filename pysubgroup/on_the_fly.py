import numpy as np
import time
n=1500
s=np.random.rand(n)
t=np.sort(np.random.rand(n))[::-1]
my_mean=np.mean(t)
inds_a = np.argsort(s)
ts = np.empty(n,dtype=int)
for i, val in enumerate(inds_a):
    ts[val] = i
from tqdm import tqdm
class tmp:
    def __init__(self):
        self.best =- 100
        self.best_l = 0
        self.best_r = n
        self.min_size = 0
        self.num_calls = 0

    def run(self,t_2_s, s_2_t, t_in, left, right):
        self.f(t_2_s, s_2_t, t_in, left, right, first=True)
        self.f(t_2_s, s_2_t, t_in, left, right, first=False)

    def get_estimate(self, t_2_s, s_2_t, t_in, left, right):
        a = 1
        ns=np.arange(1, 1+right-left)
        bs=np.zeros(len(t_in),dtype=bool)
        bs[s_2_t[left:right]] = True
        arr = t_in[bs]
        qualities = np.power(ns, a)*(np.cumsum(arr)/ns - my_mean)


        estimate = np.max(qualities)
        curr_qual = qualities[-1]
        return curr_qual, estimate,bs,qualities

    def f(self,t_2_s, s_2_t, t_in, left, right, first=True):
        self.num_calls += 1

        curr_qual, estimate, bs, qualities = self.get_estimate(t_2_s, s_2_t, t_in, left, right)


        #print(estimate, self.best, right-left)
        #
        # print(left, right, estimate, curr_qual)
        if curr_qual > self.best:
            self.best = curr_qual
            self.best_l = left
            self.best_r = right
        print(left, right, estimate, curr_qual)
        if estimate > self.best:
            if first:
                pos = np.argmax(qualities)
                #print(left, right, estimate, curr_qual)

                arr2 = (t_2_s[bs])[0:pos+1]
                if len(arr2) == 1:
                    l=arr2[0]
                    r=arr2[0]
                else:    
                    l=np.min(arr2)
                    r=np.max(arr2)
                assert(l>=left)
                assert(r<=right)
                assert(r-l < right - left)
                if r-l > 0:
                    self.f(t_2_s, s_2_t, t, l, r, True)
            else:
                if right - left > self.min_size:
                    self.f(t_2_s, s_2_t, t, left+1, right, False)
                    self.f(t_2_s, s_2_t, t, left, right-1, False)

    def naive(self, s_2_t, t_in):
        for left in range(0,len(t_in)):
            for right in range(left + 1, len(t_in)):
                a=1
                ns=right-left
                bs=np.zeros(len(t_in),dtype=bool)
                bs[s_2_t[left:right]] = True
                arr = t_in[bs]
                curr_qual = (ns ** a) * (np.sum(arr)/ns - my_mean)

                if curr_qual > self.best:
                    self.best = curr_qual
                    self.best_l = left
                    self.best_r = right


    def apriori(self,t_2_s, s_2_t, t_in):
        previous_left = [0]
        lookup=np.zeros(len(t_in), dtype=bool)
        previous_promising=[]
        for depth in tqdm(range(0, len(t_in))):
            lookup[previous_left]=True
            candidates = []
            for l in range(depth+1):
                if l==0:
                    if lookup[l]:
                        candidates.append(l)
                elif l==depth:
                    if lookup[l-1]:
                        candidates.append(l)
                else:
                    if lookup[l] and lookup[l-1]:
                        candidates.append(l)
            delta = len(t_in) - depth
            previous_left.clear()
            for left in candidates:
                self.num_calls+=1
                right = left+delta
                curr_qual, estimate, _, _ = self.get_estimate(t_2_s, s_2_t, t_in, left, right)

                if curr_qual > self.best:
                    self.best = curr_qual
                    self.best_l = left
                    self.best_r = right
                if estimate > self.best:
                    previous_promising.append((estimate, left))
            for estimate, left in previous_promising:
                if estimate > self.best:
                    previous_left.append(left)
            previous_promising.clear()
            
            



print(n*(n/1)/2)

algo = tmp()
start_time = time.time()
algo.naive(inds_a, t)
print('naive'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)





algo = tmp()
start_time = time.time()
algo.apriori(ts, inds_a, t)
print('algorithm'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)









