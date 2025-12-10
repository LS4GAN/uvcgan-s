import copy

class LossMetrics:

    def __init__(self, values = None, n = 0):
        self._values = values
        self._n      = n

    @property
    def values(self):
        if self._values is None:
            return None

        return { k : v / self._n for (k,v) in self._values.items() }

    def update(self, values):
        if self._values is None:
            self._values = copy.deepcopy(values)
        else:
            for k,v in values.items():
                self._values[k] += v

        self._n += 1

    def join(self, other, other_prefix = None):
        self_dict  = self.values
        other_dict = other.values

        if other_prefix is not None:
            other_dict = {
                other_prefix + k : v for (k, v) in other_dict.items()
            }

        values_dict = { **self_dict, **other_dict }

        return LossMetrics(values_dict, n = 1)

