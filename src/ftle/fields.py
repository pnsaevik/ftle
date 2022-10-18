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

    @staticmethod
    def from_roms_dataset(dset, remove_coords=True, posix_time=True, z_depth=False):
        """
        Create Fields object from ROMS dataset

        :param dset: A ROMS dataset (xarray.Dataset)

        :param remove_coords: True if interpolation should take fractional zero-based
        indices as input instead of dataset coordinates (default = True)

        :param posix_time: True if the time variable should be converted to posix
        timestamps (default = True)

        :param z_depth: True if interpolation should take depth as input instead of
        S-coordinates (default = False)

        :return: A Fields object
        """
        return from_roms_dataset(dset, remove_coords, posix_time, z_depth)


def from_roms_dataset(dset, remove_coords=True, posix_time=True, z_depth=False):
    funcdict = dict()

    mappings = dict(
        s_rho='z',
        s_w='z',
        eta_rho='y',
        xi_rho='x',
        eta_u='y',
        xi_u='x',
        eta_v='y',
        xi_v='x',
        ocean_time='t',
    )

    offsets = dict(
        s_w=0.5,
        xi_u=-0.5,
        eta_v=-0.5,
    )

    nearests = dict(
        u={'y'},
        v={'x'},
    )

    if remove_coords:
        data_vars = {k: xr.DataArray(v, coords={}) for k, v in dset.variables.items()}
    else:
        data_vars = {**dict(dset.data_vars), **dict(dset.coords)}

    # --- Start create coordinate transform
    from . import coords
    roms_crs = coords.fourdim_crs_from_roms_grid(dset, z_coord='index', t_coord='index')

    if z_depth:
        input_vert_crs = coords.NegativePlainVertCRS()
    else:
        input_vert_crs = roms_crs.vert_crs

    input_horz_crs = roms_crs.horz_crs
    input_time_crs = roms_crs.time_crs

    input_crs = coords.FourDimCRS(input_horz_crs, input_vert_crs, input_time_crs)
    trans = coords.FourDimTransform(input_crs, roms_crs).transform
    # --- End create coordinate transform

    if posix_time and 'ocean_time' in data_vars:
        epoch = np.datetime64('1970-01-01', 'us')
        one_sec = np.timedelta64(1000000, 'us')
        posix = (data_vars['ocean_time'] - epoch) / one_sec
        if not remove_coords:
            posix.coords['ocean_time'] = posix.variable
        data_vars['ocean_time'] = posix

    for k, v in data_vars.items():
        mapping = {dim: mappings[dim] for dim in v.dims if dim in mappings}
        offset = {mappings[dim]: offsets[dim] for dim in v.dims if dim in offsets}
        nearest = nearests.get(k, ())

        fn = get_interp_func_from_xr_data_array(v, mapping, offset, nearest, trans)

        funcdict[k] = fn

    return Fields(funcdict)


def _get_vtrans_function(func, vtrans):
    def fn(t, z, y, x):
        new_z = vtrans.z(x, y, -z, t)
        return func(t, new_z, y, x)

    return fn


def get_interp_func_from_xr_data_array(darr, mapping=None, offset=None, nearest=(), transform=None):
    """
    Create 4D interpolation function from grid variables

    :param darr:

    Input data. Should be an xarray.DataArray. Dimensions that are named
    't', 'z', 'y' or 'x' are interpolated. Other dimension names are permittible if a
    `mapping` argument is given.
    The input data need not include all 4 dimensions. Missing dimensions are ignored
    during interpolation.
    If `darr` has coordinates, these are the basis for the interpolation. Otherwise,
    a zero-based range index is implied. For instance, fn(0, 0, 0, 0) should return
    the first element of `darr` if the array has no coordinates.

    :param mapping: Mapping from `darr` dimension names to ('t', 'z', 'y', 'x')

    :param offset: A mapping of offset values. Offset values are added to the coordinates
    before interpolation. For instance, fn(100, 0, 0, 0) returns the first element of
    `darr` if there are no coordinates and offset = {'t': -100}.

    :param nearest: A subset of {'t', 'z', 'y', 'x'}. Specifies dimensions which should
    be interpolated using the `nearest` algorithm.

    :param transform: A transform function fn(xx, yy, zz, tt) which returns a tuple
    (x2, y2, z2, t2). The transform is applied to the input coordinates before
    interpolation.

    :return: A function fn(t, z, y, x) which samples the array value at fractional
    coordinates.
    """
    if mapping is not None:
        darr = darr.rename(mapping)

    def mkvar(np_arr):
        return xr.Variable('pid', np_arr)

    def shift(coords_in):
        """Shift coordinates by the specified offset if it exists"""
        return {k: v + offset[k] if k in offset else v for k, v in coords_in.items()}

    def fn(t, z, y, x):
        if transform is not None:
            x, y, z, t = transform(xx=x, yy=y, zz=z, tt=t)

        coords = dict(x=mkvar(x), y=mkvar(y), z=mkvar(z), t=mkvar(t))
        coords = {k: coords[k] for k in darr.dims}
        if offset:
            coords = shift(coords)

        for k in nearest:
            coords[k] = np.round(coords[k])

        return darr.interp(**coords)

    def fn_singledim(t, z, y, x):
        v = next(vv for vv in (t, z, y, x) if vv.shape != ())
        return xr.broadcast(darr, xr.DataArray(mkvar(v)))[0]

    fn.dtype = np.dtype('f8')
    fn_singledim.dtype = darr.dtype

    if darr.dims == ():
        return fn_singledim
    else:
        return fn
