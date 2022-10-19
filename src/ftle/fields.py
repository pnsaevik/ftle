import numpy as np


class Fields:
    """
    A set of named, four-dimensional fields.

    A four-dimensional field is simply a function fn(x, y, z, t) which accepts numpy
    arrays as its input and returns an xarray.DataArray as its output.

    To generate fields from linear interpolation of gridded datasets, use the
    *from_dataset* or the *from_roms_dataset* function.
    """
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
    def from_roms_dataset(dset, xy_coords='index', z_coords='index', t_coords='index'):
        """
        Create Fields object from ROMS dataset

        :param dset: A ROMS dataset (xarray.Dataset)

        :param xy_coords: ('index' or 'lonlat') Horizontal input coordinate system.
        - 'index': Use 'eta_rho' and 'xi_rho' as input coordinates. For instance, x = 0
        means xi_rho = 0 and xi_u = -0.5, while x = 0.5 means xi_rho = 0.5 and xi_u = 0.
        - 'lonlat': Use longitude and latitude as input coordinates. For instance, y = 60
        means a latitude of 60 degrees north, while x = -5 means a longitude of 5 degrees
        west. Conversion between lat/lon and grid coordinates is done using linear
        interpolation of the lat_rho and lon_rho arrays.

        :param z_coords: ('index', 'depth' or 'S-coord') Vertical input coordinate system.
        - 'index': Use 's_rho' as input coordinate. For instance, z = 0 means s_rho = 0
        and s_w = 0.5, while z = -0.5 means s_rho = -0.5 and s_w = 0.
        - 'depth': Use meters below surface as input coordinate. For instance, if the
        total depth is 100, then z = 100 means s_rho = -0.5 and s_w = 0.

        :param t_coords: ('index', 'posix' or 'numpy') Time input coordinate system.
        - 'index': Use the index of 'ocean_time' as input coordinate. For instance, t = 0
        means the first time index and t = 0.5 means the value halfway between the first
        and second time indices.
        - 'posix': Use the number of seconds since 1970-01-01 (disregarding leap seconds)
        as input coordinate. For instance, t = 3600 is the date 1970-01-01T01.
        - 'numpy': Use numpy dates as input coordinates.

        :return: A Fields object
        """
        return from_roms_dataset(dset, xy_coords, z_coords, t_coords)


def from_roms_dataset(dset, xy_coords='index', z_coords='index', t_coords='index'):
    # The output dict of interpolation functions
    funcdict = dict()

    # Conversion between roms names and coordinate names
    mappings = dict(
        s_rho='z', s_w='z', eta_rho='y', xi_rho='x', eta_u='y', xi_u='x', eta_v='y',
        xi_v='x', ocean_time='t',
    )

    # Off-by-one-half corrections for variables that are defined on grid faces
    offsets = dict(s_w=0.5, xi_u=-0.5, eta_v=-0.5)

    # Use nearest neighbour interpolation in lateral direction for horizontal velocities
    nearests = dict(u={'y'}, v={'x'})

    # Strip variables of coordinates to facilitate index-based interpolation
    data_vars = data_vars_without_coords(dset)

    # Create four-dimensional transform object
    from .coords import FourDimTransform
    if {xy_coords, z_coords, t_coords} != {'index'}:
        trans = FourDimTransform.from_roms(dset, xy_coords, z_coords, t_coords).transform
    else:
        trans = None

    # Create one interpolation function per variable
    for k, v in data_vars.items():
        mapping = {dim: mappings[dim] for dim in v.dims if dim in mappings}
        offset = {mappings[dim]: offsets[dim] for dim in v.dims if dim in offsets}
        nearest = nearests.get(k, ())

        fn = get_interp_func_from_xr_data_array(v, mapping, offset, nearest, trans)

        funcdict[k] = fn

    return Fields(funcdict)


def data_vars_without_coords(dset):
    """
    Return a dict of all variables in the dataset, as xarray.DataArray objects, with
    all the coordinates removed.
    :param dset: An xarray.Dataset object
    :return: A dict where the keys are the variable names and the values are
    xarray.DataArray objects.
    """
    import xarray as xr
    data_vars = {k: xr.DataArray(v, coords={}) for k, v in dset.variables.items()}
    return data_vars


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

    :param mapping: Mapping from `darr` dimension names to ('x', 'y', 'z', 't')

    :param offset: A mapping of offset values. Offset values are added to the coordinates
    before interpolation. For instance, fn(100, 0, 0, 0) returns the first element of
    `darr` if there are no coordinates and offset = {'t': -100}.

    :param nearest: A subset of {'z', 'y', 'z', 't'}. Specifies dimensions which should
    be interpolated using the `nearest` algorithm.

    :param transform: A transform function fn(xx, yy, zz, tt) which returns a tuple
    (x2, y2, z2, t2). The transform is applied to the input coordinates before
    interpolation.

    :return: A function fn(x, y, z, t) which samples the array value at fractional
    coordinates.
    """
    if mapping is not None:
        darr = darr.rename(mapping)

    def mkvar(np_arr):
        import xarray as xr
        return xr.Variable('pid', np_arr)

    def shift(coords_in):
        """Shift coordinates by the specified offset if it exists"""
        return {k: v + offset[k] if k in offset else v for k, v in coords_in.items()}

    def fn(x, y, z, t):
        import xarray as xr

        if transform is not None:
            x, y, z, t = transform(xx=x, yy=y, zz=z, tt=t)

        coords = dict(x=mkvar(x), y=mkvar(y), z=mkvar(z), t=mkvar(t))
        coords = {k: coords[k] for k in darr.dims}
        if offset:
            coords = shift(coords)

        for k in nearest:
            coords[k] = np.round(coords[k])

        darr_interp = darr.interp(**coords)

        if dateconv:
            darr_interp = xr.DataArray(
                data=epoch + (one_sec * darr_interp.values).astype('timedelta64[ns]'),
                coords=darr_interp.coords, dims=darr_interp.dims, name=darr_interp.name,
                attrs=darr_interp.attrs,
            )

        return darr_interp

    def fn_singledim(x, y, z, t):
        import xarray as xr
        v = next(vv for vv in (x, y, z, t) if vv.shape != ())
        return xr.broadcast(darr, xr.DataArray(mkvar(v)))[0]

    # If input array is a datetime, we must convert it to a number so that interpolation works
    if np.issubdtype(darr.dtype, np.datetime64):
        epoch = np.datetime64('1970-01-01').astype(darr.dtype)
        one_sec = np.timedelta64(1, 's')
        dateconv = True
        darr = (darr - epoch) / one_sec
        fn.dtype = np.dtype('datetime64[ns]')
    else:
        dateconv = False
        fn.dtype = np.dtype('f8')

    fn_singledim.dtype = darr.dtype

    if darr.dims == ():
        return fn_singledim
    else:
        return fn
