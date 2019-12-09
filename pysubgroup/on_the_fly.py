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
from itertools import chain, product
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

    def naive(self, t_in, s_2_t=None):
        if s_2_t is None:
            t = t_in
        else:
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

    def naive_plus(self, s_2_t, t_in):
        t = t_in[s_2_t]
        t_cumsum = np.cumsum(t)-t[0]
        _, min_width, max_width = self.get_bounds(t_in, self.best)
        for width in range(1, len(t_in)):
            if width < min_width:
                continue
            if width > max_width:
                break
            power = (width ** self.a)
            for left in range(0, len(t_in)-width):
                right = left + width
                curr_qual = power * ((t_cumsum[right]-t_cumsum[left])/width - my_mean)

                if curr_qual > self.best:
                    self.best = curr_qual
                    self.best_l = left
                    self.best_r = right
                    _, min_width, max_width = self.get_bounds(t_in, self.best)


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
        for min_chop_off in range(0, n_max):
            if  val - t_cumsum[min_chop_off] < bound:
                break
        t_averaged = t_in - t_mean
        t_cumsum_bad = np.cumsum(t_averaged[::-1])
        t_cumsum_bad -= t_cumsum_bad[0]
        for min_general in range(0, n_max):
            if val + t_cumsum_bad[min_general] < bound:
                break

        print(n_max)
        print('lower loose bound', n_max - min_chop_off)
        print('upper loose bound', len(t_in)-min_general)
        #return n_max, n_max - min_chop_off, len(t_in)-min_general

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
        print('lower strict bound', min_chop_off)
        print('upper strict bound', min_general)
        return n_max, min_chop_off, min_general
        


    def bounded(self, t_2_s, s_2_t, t_in):
        n_max, min_chop_off, min_general = self.get_bounds(t_in,30)
        self.generalising_apriori(t_2_s, s_2_t, t_in, start=min_general, previous_left_in = None)
        self.apriori(t_2_s, s_2_t, t_in, start=min_chop_off, previous_left_in = None)
        print(self.best)
        n_max, min_chop_off, min_general = self.get_bounds(t_in,self.best)
        self.generalising_apriori(t_2_s, s_2_t, t_in, start=min_general, previous_left_in = None)
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

    def left_right_order(self, L, R):
        l=0
        r=0
        out=np.empty(len(L)+len(R))
        while l < len(L) and r < len(R):
            if L[l] > R[r]:
                out[l+r] = L[l]
                l+=1
            else:
                out[l+r] = R[r]
                r+=1
        while l<len(L):
            out[l+r] = L[l]
            l+=1
        while r<len(R):
            out[l+r] = R[r]
            r+=1
        return out[:l+r]



    def generalising_apriori2(self, t_2_s, s_2_t, t_in, start=None, stop=None, previous_left_in=None):
        t = t_in[s_2_t]
        
        t_mean = np.mean(t)
        t_cumsum = np.cumsum(t)-t[0]
        k=10
        test = t-t_mean
        print(np.sum(test))
        real_maxes_right, indices_right,sums=self.compute_sums(k, test)
        #for l,index in zip(real_maxes_right, indices_right):
        #    print(l)
        #    print(index)

        real_maxes_left, indices_left,_=self.compute_sums(k, test[::-1])
        real_maxes_left=list(reversed(real_maxes_left))
        indices_left=list(reversed(indices_left))
        #for l in real_maxes_left:
        #    print(l)

        best_score = []
        right_estimates=[]
        left_estimates=[]
        for left_list, right_list in zip(real_maxes_left, real_maxes_right):
            left = 0
            if len(left_list) >0:
                left=np.max(left_list)
            right = 0
            if len(right_list) >0:
                right=np.max(right_list)
            right_estimates.append(right)
            left_estimates.append(left)
        print(sums)
        lr_estimates=[l+r+s for l,r,s in zip(left_estimates, right_estimates,sums)]
        print(lr_estimates)
        print(np.max(lr_estimates))

                
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
            sizes = np.arange(delta, len(t_in))
            sizes_a = sizes ** self.a
            best_est=-100
            for left in candidates:
                #print(left)
                #print(delta)
                self.num_calls += 1
                right = left+delta
                size = right-left
                #print(size)
                curr_qual = (size ** self.a) * ((t_cumsum[right]-t_cumsum[left])/size - my_mean)
                #arr = np.cumsum(self.left_right_order(t[left:0:-1], t[right:]))
                
                estimate = ((t_cumsum[right]-t_cumsum[left])/size - my_mean) + test[left]
                best_est=max(estimate,best_est)
                #print(curr_qual, estimate)
                if curr_qual > self.best:
                    self.best = curr_qual
                    self.best_l = left
                    self.best_r = right
                if estimate > self.best:
                    previous_promising.append((estimate, left))
            print(best_est)
            for estimate, left in previous_promising:
                if estimate > self.best:
                    previous_left.append(left)
            previous_promising.clear()
        return previous_left


    def compute_sums(self, k, arr_in):
        maxs = np.empty(k)
        sums = np.empty(k)
        max_index=np.empty(k,dtype=int)
        for i, arr in enumerate(np.split(arr_in, k)):
            cs = np.cumsum(arr)
            max_index[i]=np.argmax(cs)
            maxs[i] = np.max(cs)
            sums[i] = cs[-1]

        sums2=np.cumsum(sums)
        real_maxes = []
        real_indcies = []
        for i in range(k):
            l=[]
            indices=[]
            for j in range(i, k):
                l.append(sums2[j-1]-sums2[i] + maxs[j])
                indices.append(max_index[j])
            real_maxes.append(l)
            real_indcies.append(indices)
        return real_maxes, real_indcies, sums

    def to_max(self, arr, indices, k, default_index):
        the_values=[]
        the_indices=[]
        size_per_arr= int(default_index/k)
        l=[np.arange(i, len(arr[0])+1,dtype=int)*size_per_arr for i in range(len(arr))]
        for curr_list, indices, position in zip(arr, indices, l):
            val = 0
            index = default_index
            if len(curr_list) > 0:
                val = np.max(curr_list)
                index = position[np.argmax(curr_list)] + indices[np.argmax(curr_list)]
            the_values.append(val)
            the_indices.append(index)
        return the_values, the_indices

    def the_real_stuff(self, t_2_s, s_2_t, t_in, start=None, stop=None, previous_left_in=None):
        t = t_in[s_2_t]
        
        t_mean = np.mean(t)
        t_cumsum = np.cumsum(t)-t[0]
        k=20
        test = t-t_mean
        #print(np.sum(test))
        real_maxes_right, indices_right,sums=self.compute_sums(k, test)
        #for l,index in zip(real_maxes_right, indices_right):
        #    print(l)
        #    print(index)

        real_maxes_left, indices_left,_=self.compute_sums(k, test[::-1])
        
        #for l in real_maxes_left:
        #    print(l)

        best_score = []
        right_estimates, right_indices = self.to_max(real_maxes_right, indices_right, k, len(t))
        left_estimates, left_indices = self.to_max(real_maxes_left, indices_left, k, len(t))
        left_estimates = list(reversed(left_estimates))
        left_indices = [len(t)-i for i in reversed(left_indices)]
        #print()
        #print()
        #print(right_indices)
        #print(left_estimates)
        real_maxes_left=list(reversed(real_maxes_left))
        indices_left=list(reversed(indices_left))
        #print([l+r for l,r in zip(left_estimates[1:], right_estimates[:-1])])
        #print([l+r for l,r in zip(left_estimates[:-1], right_estimates[1:])])
        lr_estimates=[l+r+s for l,r,s in zip(left_estimates, right_estimates, sums)]
        lr_indices=[(l, r) for l,r in zip(left_indices, right_indices)]
        #print(sums)
        #print(lr_estimates)
        #print(lr_indices)
        #print(np.max(lr_estimates))
        best_indices = lr_indices[np.argmax(lr_estimates)]
        #print(best_indices)
        self.best_l = best_indices[0]
        self.best_r = best_indices[1]
        self.best = np.max(lr_estimates)


        for i, arr in enumerate(np.split(t, k)):
            self.naive(arr[1:-1])
    

    
            


print(n*(n/1)/2)

algo = tmp()
start_time = time.time()
algo.naive(t, inds_a)
print('naive'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)


algo = tmp()
start_time = time.time()
algo.the_real_stuff(ts, inds_a, t)
print('stuff'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)
#for tpl in algo.num_candidates:
#    print(tpl)
quit()

algo = tmp()
start_time = time.time()
algo.double_apriori(ts, inds_a, t)
print('apriori'+" --- %s seconds ---" % (time.time() - start_time))
print(algo.best_l, algo.best_r, algo.best, algo.num_calls)
for tpl in algo.num_candidates:
    if tpl[1]+tpl[2]<n-1:
        print(tpl)








