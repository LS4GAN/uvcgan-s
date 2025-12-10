import torch
from torchvision.transforms import Normalize
from .select import extract_name_kwargs

class DataNorm:

    def normalize(self, x):
        raise NotImplementedError

    def denormalize(self, y):
        raise NotImplementedError

    def normalize_nograd(self, x):
        with torch.no_grad():
            return self.normalize(x)

    def denormalize_nograd(self, y):
        with torch.no_grad():
            return self.denormalize(y)

class Standardizer(DataNorm):

    def __init__(self, mean, stdev):
        super().__init__()
        # fwd: (x - m) / s
        # bkw: s * y + m

        bkw_mean  = [ -m/s for (m, s) in zip(mean, stdev) ]
        bkw_stdev = [ 1/s  for s in stdev ]

        self._norm_fwd = Normalize(mean, stdev)
        self._norm_bkw = Normalize(bkw_mean, bkw_stdev)

    def normalize(self, x):
        return self._norm_fwd(x)

    def denormalize(self, y):
        return self._norm_bkw(y)

class ScaleNorm(DataNorm):

    def __init__(self, scale = 1.0):
        super().__init__()
        self._scale = scale

    def normalize(self, x):
        return self._scale * x

    def denormalize(self, y):
        return y / self._scale

class LogNorm(DataNorm):

    def __init__(self, clip_min = None, bias = None):
        super().__init__()

        self._clip_min = clip_min
        self._bias     = bias

    def normalize(self, x):
        if self._clip_min is not None:
            x = x.clip(min = self._clip_min, max = None)

        if self._bias is not None:
            x = x + self._bias

        return torch.log(x)

    def denormalize(self, y):
        x = torch.exp(y)

        if self._bias is not None:
            x = x - self._bias

        return x

class SymLogNorm(DataNorm):
    """
    To ensure continuity of the function and its derivative:
        y = scale * x                       if (x <= T)
        y = scale * T * (log(x/T) + 1)      if (x >  T)

    Inverse:
        x = y / scale                       if (y <= scale * T)
        x = T * exp(y / (scale * T) - 1)    otherwise
    """

    def __init__(self, threshold = 1.0, scale = 1.0):
        super().__init__()

        self._scale = scale
        self._T     = threshold
        self._inv_T = scale * threshold

    def normalize(self, x):
        x_abs = x.abs()

        y_lin = self._scale * x
        y_log = torch.sign(x) * self._inv_T * (torch.log(x_abs / self._T) + 1)

        return torch.where(x_abs > self._T, y_log, y_lin)

    def denormalize(self, y):
        y_abs = y.abs()

        x_lin = y / self._scale
        x_log = torch.sign(y) * self._T * torch.exp(y_abs / self._inv_T - 1)

        return torch.where(y_abs > self._inv_T, x_log, x_lin)

class MinMaxScaler(DataNorm):

    def __init__(self, feature_min, feature_max, dim):
        super().__init__()

        self._min = feature_min
        self._max = feature_max
        self._dim = dim

    @staticmethod
    def align_shapes(source, target, dim):
        if source.ndim == 0:
            source = source.unsqueeze(0)

        n_before = dim
        n_after  = target.ndim - (dim + source.ndim)

        return source.reshape(
            ( 1, ) * n_before + source.shape + ( 1, ) * n_after
        )

    def normalize(self, x):
        fmin = torch.tensor(self._min, dtype = x.dtype, device = x.device)
        fmax = torch.tensor(self._max, dtype = x.dtype, device = x.device)

        fmin = MinMaxScaler.align_shapes(fmin, x, self._dim)
        fmax = MinMaxScaler.align_shapes(fmax, x, self._dim)

        return (x - fmin) / (fmax - fmin)

    def denormalize(self, y):
        fmin = torch.tensor(self._min, dtype = y.dtype, device = y.device)
        fmax = torch.tensor(self._max, dtype = y.dtype, device = y.device)

        fmin = MinMaxScaler.align_shapes(fmin, y, self._dim)
        fmax = MinMaxScaler.align_shapes(fmax, y, self._dim)

        return (y * (fmax - fmin) + fmin)

class DoublePrecision(DataNorm):

    def __init__(self, norm):
        self._norm = norm

    def normalize(self, x):
        old_dtype = x.dtype

        result = x.double()
        result = self._norm.normalize(result)

        return result.to(dtype = old_dtype)

    def denormalize(self, y):
        old_dtype = y.dtype

        result = y.double()
        result = self._norm.denormalize(result)

        return result.to(dtype = old_dtype)

class Compose(DataNorm):

    def __init__(self, norms):
        self._norms = norms

    def normalize(self, x):
        for norm in self._norms:
            x = norm.normalize(x)

        return x

    def denormalize(self, y):
        for norm in reversed(self._norms):
            y = norm.denormalize(y)

        return y

def select_single_data_normalization(norm):
    name, kwargs = extract_name_kwargs(norm)

    if name == 'double-precision':
        return DoublePrecision(select_data_normalization(**kwargs))

    if name == 'scale':
        return ScaleNorm(**kwargs)

    if name == 'log':
        return LogNorm(**kwargs)

    if name == 'symlog':
        return SymLogNorm(**kwargs)

    if name == 'standardize':
        return Standardizer(**kwargs)

    if name == 'min-max-scaler':
        return MinMaxScaler(**kwargs)

    raise ValueError(f"Unknown data normalization '{name}'")

def select_data_normalization(norm):
    if norm is None:
        return None

    if isinstance(norm, (tuple, list)):
        norm = [ select_single_data_normalization(n) for n in norm ]
        return Compose(norm)

    return select_single_data_normalization(norm)

