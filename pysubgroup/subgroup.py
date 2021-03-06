'''
Created on 28.04.2016

@author: lemmerfn
'''
from abc import ABC, abstractmethod
import weakref
from functools import total_ordering
import numpy as np
import pandas as pd
import pysubgroup as ps



@total_ordering
class SelectorBase(ABC):
    __refs__ = weakref.WeakSet()
    def __new__(cls, *args, **kwargs):

        tmp = super().__new__(cls)

        tmp.set_descriptions(*args, **kwargs)
        if tmp in SelectorBase.__refs__:
            for ref in SelectorBase. __refs__:
                if ref == tmp:
                    return ref
        return tmp

    def __init__(self):
        SelectorBase.__refs__.add(self)

    def __eq__(self, other):
        if other is None:
            return False
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return self._hash #pylint: disable=no-member

    @abstractmethod
    def set_descriptions(self, *args, **kwargs):
        pass


def get_cover_array_and_size(subgroup, data_len=None, data=None):
    if hasattr(subgroup, "representation"):
        cover_arr = subgroup
        size = subgroup.size_sg
    elif isinstance(subgroup, slice):
        cover_arr = subgroup
        if data_len is None:
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                raise ValueError("if you pass a slice, you need to pass either data_len or data")
        # https://stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
        size = len(range(*subgroup.indices(data_len)))
    elif hasattr(subgroup, '__array_interface__'):
        cover_arr = subgroup
        type_char = subgroup.__array_interface__['typestr'][1]
        if type_char == 'b': # boolean indexing is used
            size = np.count_nonzero(cover_arr)
        elif type_char == 'u' or type_char == 'i': # integer indexing
            size = subgroup.__array_interface__['shape'][0]
        else:
            print(type_char)
            raise NotImplementedError(f"Currently a typechar of {type_char} is not supported.")
    else:
        assert isinstance(data, pd.DataFrame)
        cover_arr = subgroup.covers(data)
        size = np.count_nonzero(cover_arr)
    return cover_arr, size


def get_size(subgroup, data_len=None, data=None):
    if hasattr(subgroup, "representation"):
        size = subgroup.size_sg
    elif isinstance(subgroup, slice):
        if data_len is None:
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                raise ValueError("if you pass a slice, you need to pass either data_len or data")
        # https://stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
        size = len(range(*subgroup.indices(data_len)))
    elif hasattr(subgroup, '__array_interface__'):
        type_char = subgroup.__array_interface__['typestr'][1]
        if type_char == 'b': # boolean indexing is used
            size = np.count_nonzero(subgroup)
        elif type_char == 'u' or type_char == 'i': # integer indexing
            size = subgroup.__array_interface__['shape'][0]
        else:
            print(type_char)
            raise NotImplementedError(f"Currently a typechar of {type_char} is not supported.")
    else:
        assert isinstance(data, pd.DataFrame)
        size = np.count_nonzero(subgroup.covers(data))
    return size


class EqualitySelector(SelectorBase):
    def __init__(self, attribute_name, attribute_value, selector_name=None):
        if attribute_name is None:
            raise TypeError()
        if attribute_value is None:
            raise TypeError()
        self._attribute_name = attribute_name
        self._attribute_value = attribute_value
        self._selector_name = selector_name
        self.set_descriptions(self._attribute_name, self._attribute_value, self._selector_name)
        super().__init__()

    @property
    def attribute_name(self):
        return self._attribute_name

    @property
    def attribute_value(self):
        return self._attribute_value

    def set_descriptions(self, attribute_name, attribute_value, selector_name=None): # pylint: disable=arguments-differ
        self._hash, self._query, self._string = EqualitySelector.compute_descriptions(attribute_name, attribute_value, selector_name=selector_name)

    @classmethod
    def compute_descriptions(cls, attribute_name, attribute_value, selector_name):
        if isinstance(attribute_value, (str, bytes)):
            query = str(attribute_name) + "==" + "'" + str(attribute_value) + "'"
        elif np.isnan(attribute_value):
            query = attribute_name + ".isnull()"
        else:
            query = str(attribute_name) + "==" + str(attribute_value)
        if selector_name is not None:
            string_ = selector_name
        else:
            string_ = query
        hash_value = hash(query)
        return (hash_value, query, string_)

    def __repr__(self):
        return self._query

    def covers(self, data):
        row = data[self.attribute_name].to_numpy()
        if pd.isnull(self.attribute_value):
            return pd.isnull(row)
        return row == self.attribute_value

    def __str__(self, open_brackets="", closing_brackets=""):
        return open_brackets + self._string + closing_brackets

    @property
    def selectors(self):
        return (self,)


class NegatedSelector(SelectorBase):
    def __init__(self, selector):
        self._selector = selector
        self.set_descriptions(selector)
        super().__init__()

    def covers(self, data_instance):
        return np.logical_not(self._selector.covers(data_instance))

    def __repr__(self):
        return self._query

    def __str__(self, open_brackets="", closing_brackets=""):
        return "NOT " + self._selector.__str__(open_brackets, closing_brackets)

    def set_descriptions(self, selector):  # pylint: disable=arguments-differ
        self._query = "(not " + repr(selector) + ")"
        self._hash = hash(repr(self))

    @property
    def attribute_name(self):
        return self._selector.attribute_name

    @property
    def selectors(self):
        return self._selector.selectors


# Including the lower bound, excluding the upper_bound
class IntervalSelector(SelectorBase):
    def __init__(self, attribute_name, lower_bound, upper_bound, selector_name=None):
        self._attribute_name = attribute_name
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.selector_name = selector_name
        self.set_descriptions(attribute_name, lower_bound, upper_bound, selector_name)
        super().__init__()

    @property
    def attribute_name(self):
        return self._attribute_name

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    def covers(self, data_instance):
        val = data_instance[self.attribute_name].to_numpy()
        return np.logical_and(val >= self.lower_bound, val < self.upper_bound)

    def __repr__(self):
        return self._query

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self._string

    @classmethod
    def compute_descriptions(cls, attribute_name, lower_bound, upper_bound, selector_name=None):
        if selector_name is None:
            _string = cls.compute_string(attribute_name, lower_bound, upper_bound, rounding_digits=2)
        else:
            _string = selector_name
        _query = cls.compute_string(attribute_name, lower_bound, upper_bound, rounding_digits=None)
        _hash = _query.__hash__()
        return (_hash, _query, _string)

    def set_descriptions(self, attribute_name, lower_bound, upper_bound, selector_name=None):  # pylint: disable=arguments-differ
        self._hash, self._query, self._string = IntervalSelector.compute_descriptions(attribute_name, lower_bound, upper_bound, selector_name=selector_name)

    @classmethod
    def compute_string(cls, attribute_name, lower_bound, upper_bound, rounding_digits):
        if rounding_digits is None:
            formatter = "{}"
        else:
            formatter = "{0:." + str(rounding_digits) + "f}"
        ub = upper_bound
        lb = lower_bound
        if ub % 1:
            ub = formatter.format(ub)
        if lb % 1:
            lb = formatter.format(lb)

        if lower_bound == float("-inf") and upper_bound == float("inf"):
            repre = attribute_name + "= anything"
        elif lower_bound == float("-inf"):
            repre = attribute_name + "<" + str(ub)
        elif upper_bound == float("inf"):
            repre = attribute_name + ">=" + str(lb)
        else:
            repre = attribute_name + ": [" + str(lb) + ":" + str(ub) + "["
        return repre

    @property
    def selectors(self):
        return (self,)


@total_ordering
class Subgroup():
    def __init__(self, target, subgroup_description):
        # If its already a BinaryTarget object, we are fine, otherwise we create a new one
        # if (isinstance(target, BinaryTarget) or isinstance(target, NumericTarget)):
        #    self.target = target
        # else:
        #    self.target = BinaryTarget(target)

        # If its already a SubgroupDescription object, we are fine, otherwise we create a new one
        self.target = target
        self.subgroup_description = subgroup_description

        # initialize empty cache for statistics
        self.statistics = {}

    def __repr__(self):
        return "<<" + repr(self.target) + "; D: " + repr(self.subgroup_description) + ">>"

    def covers(self, instance):
        cover_arr, _ = ps.get_cover_array_and_size(self.subgroup_description, len(instance), instance)
        if not isinstance(cover_arr, type(np.array)):
            arr = np.zeros(len(instance), dtype=bool)
            arr[cover_arr] = True
            return arr
        else:
            return cover_arr

    def __getattr__(self, name):
        return getattr(self.subgroup_description, name)

    def __eq__(self, other):
        if other is None:
            return False
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return repr(self) < repr(other)

    def get_base_statistics(self, data):
        return self.target.get_base_statistics(self, data)

    def calculate_statistics(self, data):
        return self.target.calculate_statistics(self, data)


def create_selectors(data, nbins=5, intervals_only=True, ignore=None):
    if ignore is None:
        ignore = []
    sels = create_nominal_selectors(data, ignore)
    sels.extend(create_numeric_selectors(data, nbins, intervals_only, ignore=ignore))
    return sels


def create_nominal_selectors(data, ignore=None):
    if ignore is None:
        ignore = []
    nominal_selectors = []
    # for attr_name in [x for x in data.select_dtypes(exclude=['number']).columns.values if x not in ignore]:
    #    nominal_selectors.extend(create_nominal_selectors_for_attribute(data, attr_name))
    nominal_dtypes = data.select_dtypes(exclude=['number'])
    dtypes = data.dtypes
    # print(dtypes)
    for attr_name in [x for x in nominal_dtypes.columns.values if x not in ignore]:
        nominal_selectors.extend(create_nominal_selectors_for_attribute(data, attr_name, dtypes))
    return nominal_selectors


def create_nominal_selectors_for_attribute(data, attribute_name, dtypes=None):
    nominal_selectors = []
    for val in pd.unique(data[attribute_name]):
        nominal_selectors.append(EqualitySelector(attribute_name, val))
    # setting the is_bool flag for selector
    if dtypes is None:
        dtypes = data.dtypes
    if dtypes[attribute_name] == 'bool':
        for s in nominal_selectors:
            s.is_bool = True
    return nominal_selectors


def create_numeric_selectors(data, nbins=5, intervals_only=True, weighting_attribute=None, ignore=None):
    if ignore is None:
        ignore = []
    numeric_selectors = []
    for attr_name in [x for x in data.select_dtypes(include=['number']).columns.values if x not in ignore]:
        numeric_selectors.extend(create_numeric_selectors_for_attribute(
            data, attr_name, nbins, intervals_only, weighting_attribute))
    return numeric_selectors


def create_numeric_selectors_for_attribute(data, attr_name, nbins=5, intervals_only=True, weighting_attribute=None):
    numeric_selectors = []
    data_not_null = data[data[attr_name].notnull()]

    uniqueValues = np.unique(data_not_null[attr_name])
    if len(data_not_null.index) < len(data.index):
        numeric_selectors.append(EqualitySelector(attr_name, np.nan))

    if len(uniqueValues) <= nbins:
        for val in uniqueValues:
            numeric_selectors.append(EqualitySelector(attr_name, val))
    else:
        cutpoints = ps.equal_frequency_discretization(data, attr_name, nbins, weighting_attribute)
        if intervals_only:
            old_cutpoint = float("-inf")
            for c in cutpoints:
                numeric_selectors.append(IntervalSelector(attr_name, old_cutpoint, c))
                old_cutpoint = c
            numeric_selectors.append(IntervalSelector(attr_name, old_cutpoint, float("inf")))
        else:
            for c in cutpoints:
                numeric_selectors.append(IntervalSelector(attr_name, c, float("inf")))
                numeric_selectors.append(IntervalSelector(attr_name, float("-inf"), c))

    return numeric_selectors


def remove_target_attributes(selectors, target):
    result = []
    for sel in selectors:
        if not sel.get_attribute_name() in target.get_attributes():
            result.append(sel)
    return result
