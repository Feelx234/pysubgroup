import numpy as np
import time
n=1000
s=np.random.rand(n)
t=np.sort(np.random.rand(n))[::-1]
my_mean=np.mean(t)
inds_a = np.argsort(s)
ts = np.empty(n,dtype=int)
for i, val in enumerate(inds_a):
    ts[val] = i
from tqdm import tqdm
from itertools import chain
class tmp:
    def __init__(self):
        self.best =- 100
        self.best_l = 0
        self.best_r = n
        self.min_size = 0
        self.num_calls = 0
        self.num_candidates=[]
        self.a = 1

    def run(self,t_2_s, s_2_t, t_in, left, right):
        self.f(t_2_s, s_2_t, t_in, left, right, first=True)
        self.f(t_2_s, s_2_t, t_in, left, right, first=False)

    def get_estimate(self, t_2_s, s_2_t, t_in, left, right):
        ns=np.arange(1, 1+right-left)
        bs=np.zeros(len(t_in),dtype=bool)
        bs[s_2_t[left:right]] = True
        arr = t_in[bs]
        qualities = np.power(ns, self.a)*(np.cumsum(arr)/ns - my_mean)


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
        t = t_in[s_2_t]
        t_cumsum = np.cumsum(t)-t[0]
        for width in range(1, len(t_in)):
            power = (width ** self.a)
            for left in range(0, len(t_in)-width):
                right = left + width
                curr_qual = power * ((t_cumsum[right]-t_cumsum[left])/width - my_mean)

                if curr_qual > self.best:
                    self.best = curr_qual
                    self.best_l = left
                    self.best_r = right


    def apriori(self, t_2_s, s_2_t, t_in, start=None, stop=None, previous_left_in=None):
        if start is None:
            start = len(t_in)
        if stop is None:
            stop = 0
        if previous_left_in is not None:
            previous_left = previous_left_in
        else:
            previous_left = list(range(len(t_in)-start+1))


        lookup = np.zeros(len(t_in), dtype=bool)
        previous_promising = []
        for length in range(start, stop, -1):
            lookup[:] = False
            if len(previous_left) == 0:
                break
            self.num_candidates.append((self.best, len(previous_left), length))
            lookup[previous_left] = True
            candidates = []
            #print(previous_left)
            #print(length)
            for l in previous_left:
                if l == (len(lookup)-length):
                    candidates.append(l)
                else:
                    candidates.append(l)
                    if not lookup[l+1]:
                        candidates.append(l+1)
            previous_left.clear()
            for left in candidates:
                self.num_calls += 1
                right = left+length
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
        return previous_left


    def get_bounds(self, t_in, bound):
        t_mean = np.mean(t_in)
        for n_max, v in enumerate(t_in-t_mean):
            if v < 0:
                break
        t_cumsum = np.cumsum(t_in - t_mean)-(t_in[0]+t_mean)
        val = t_cumsum[n_max]
        for min_chop_off in range(n_max, 0, -1):
            val -= (t_in[min_chop_off]-t_mean)
            if  val < bound:
                break
        t_averaged = t_in - t_mean
        t_cumsum_bad = np.cumsum(t_averaged[::-1])
        t_cumsum_bad -= t_cumsum_bad[0]
        for min_general in range(n_max+1,len(t_in)):
            if t_cumsum[min_general] < bound:
                break
        
        print(n_max)
        print('lower bound', min_chop_off)
        print('upper_bound', n_max + min_general)
        return n_max, min_chop_off, min_general


    def bounded(self, t_2_s, s_2_t, t_in):
        n_max, min_chop_off, min_general = self.get_bounds(t_in,1)
        self.generalising_apriori(t_2_s, s_2_t, t_in, start=n_max+min_general, previous_left_in = None)
        self.apriori(t_2_s, s_2_t, t_in, start=min_chop_off, previous_left_in = None)
        print(self.best)
        n_max, min_chop_off, min_general = self.get_bounds(t_in,self.best)
        self.generalising_apriori(t_2_s, s_2_t, t_in, start=n_max+min_general, previous_left_in = None)
        self.apriori(t_2_s, s_2_t, t_in, start=min_chop_off, previous_left_in = None)


    def generalising_apriori(self, t_2_s, s_2_t, t_in, start=None, stop=None, previous_left_in=None):
        t=t_in[s_2_t]
        t_cumsum = np.cumsum(t)-t[0]
        if start is None:
            start = 1
        if stop is None:
            stop = len(t_in)
        if previous_left_in is not None:
            previous_left = previous_left_in
        else:
            previous_left = list(range(len(t_in)-start-1))
        lookup = np.zeros(len(t_in), dtype=bool)
        previous_promising = []
        for depth in range(start, stop):
            lookup[:] = False
            if len(previous_left) == 0:
                break
            self.num_candidates.append((self.best, len(previous_left), depth))
            lookup[previous_left] = True
            candidates = []
            for l in previous_left:
                if l == 0:
                    if not lookup[l]:
                        candidates.append(l)
                else:
                    candidates.append(l-1)
                    if not lookup[l]:
                        candidates.append(l)
            delta = depth
            previous_left.clear()
            #print()
            for left in candidates:
                #print(left)
                #print(delta)
                self.num_calls += 1
                right = left+delta
                size = right-left
                #print(size)
                curr_qual = (size ** self.a) * ((t_cumsum[right]-t_cumsum[left])/size - my_mean)
                arr = np.cumsum( np.sort(np.hstack([t[:left],t[right:]]))[::-1])
                sizes = np.arange(size, len(t_in))
                estimate = np.max(sizes ** self.a * ((t_cumsum[right] - t_cumsum[left]+arr)/sizes - my_mean))
                #print(curr_qual, estimate)
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
        return previous_left

    def double_apriori(self, t_2_s, s_2_t, t_in):
        top = int(len(t_in)/2)
        bottom = top - 1
        previous_bottom = None
        previous_top = None
        while bottom > 0:
            if previous_bottom is None or len(previous_bottom):
                previous_bottom = self.apriori(t_2_s, s_2_t, t_in, start=bottom, stop=max(bottom-10,0), previous_left_in=previous_bottom)
            bottom = max(bottom-10,0)

            if previous_top is None or len(previous_top):
                previous_top = self.generalising_apriori(t_2_s, s_2_t, t_in, start=top, stop=min(top+10,len(t_in)), previous_left_in=previous_top)
            top = max(top+10,0)
            print(bottom, self.best)


print(n*(n/1)/2)

algo = tmp()
start_time = time.time()
algo.naive(inds_a, t)
print('naive'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)

algo = tmp()
start_time = time.time()
algo.bounded(ts, inds_a, t)
print('bounded'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)
#for tpl in algo.num_candidates:
#    print(tpl)


algo = tmp()
start_time = time.time()
algo.double_apriori(ts, inds_a, t)
print('apriori'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)
for tpl in algo.num_candidates:
    if tpl[1]+tpl[2]<n-1:
        print(tpl)








