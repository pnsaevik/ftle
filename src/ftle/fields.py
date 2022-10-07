import numpy as np
import xarray as xr


class Fields:
    def __init__(self, funcdict):
        self._funcdict = dict()
        for k in funcdict:
            self[k] = funcdict[k]

    def __getitem__(self, item):
        return self._funcdict[item]

    def __setitem__(self, key, value):
        self._funcdict[key] = value

    def keys(self):
        return self._funcdict.keys()

    def values(self):
        return self._funcdict.values()

    def items(self):
        return self._funcdict.items()

    def __contains__(self, item):
        return item in self._funcdict

    @staticmethod
    def from_dict(funcdict):
        return Fields(funcdict)

    @staticmethod
    def from_dataset(dset):
        funcdict = dict()
        for k, v in dset.data_vars.items():
            fn = get_interp_func_from_xr_data_array(v)
            funcdict[k] = fn

        return Fields(funcdict)


def get_interp_func_from_xr_data_array(darr, mapping=None):
    if mapping is not None:
        darr = darr.rename(mapping)

    def mkvar(np_arr):
        return xr.Variable('pid', np_arr)

    def fn(t, z, y, x):
        coords = dict(x=mkvar(x), y=mkvar(y), z=mkvar(z), t=mkvar(t))
        return darr.interp(**{k: coords[k] for k in darr.dims})

    if darr.dims == ():
        fn.dtype = darr.dtype
    else:
        fn.dtype = np.dtype('f8')
    return fn
