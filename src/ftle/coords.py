import cftime
import pyproj
from scipy.ndimage import map_coordinates
import numpy as np
from contextlib import contextmanager


def named_crs(name):
    proj_strings = dict(
        svim="+proj=stere +R=6371000.0 +lat_0=90 +lat_ts=60 +lon_0=58 +x_0=3988000 +y_0=2548000 +to_meter=4000",
        nk800="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3192800 +y_0=1784000 +ellps=WGS84 +to_meter=800",
        a01="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2992640 +y_0=1720320 +ellps=WGS84 +to_meter=160",
        a02="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3080640 +y_0=1656320 +ellps=WGS84 +to_meter=160",
        a03="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3080640 +y_0=1485120 +ellps=WGS84 +to_meter=160",
        a04="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3016640 +y_0=1438720 +ellps=WGS84 +to_meter=160",
        a05="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2896640 +y_0=1427520 +ellps=WGS84 +to_meter=160",
        a06="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2704640 +y_0=1373120 +ellps=WGS84 +to_meter=160",
        a07="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2571840 +y_0=1453120 +ellps=WGS84 +to_meter=160",
        a08="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2368640 +y_0=1429120 +ellps=WGS84 +to_meter=160",
        a09="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2248640 +y_0=1429120 +ellps=WGS84 +to_meter=160",
        a10="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=2072640 +y_0=1408320 +ellps=WGS84 +to_meter=160",
        a11="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=1848640 +y_0=1432320 +ellps=WGS84 +to_meter=160",
        a12="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=1600640 +y_0=1507520 +ellps=WGS84 +to_meter=160",
        a13="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=1400640 +y_0=1688320 +ellps=WGS84 +to_meter=160",
    )

    if isinstance(name, str) and name.lower() in proj_strings:
        return pyproj.CRS.from_proj4(proj_strings[name.lower()])
    else:
        return pyproj.CRS.from_user_input(name)


def bilin_inv(f, g, F, G, maxiter=7, tol=1.0e-7):
    """Inverse bilinear interpolation

    f, g : scalars or arrays of same shape
    F, G : 2D arrays of the same shape

    returns x, y : shaped like f and g
    such that F and G linearly interpolated to x, y
    returns f and g

    """
    imax, jmax = F.shape

    # initial guess
    x = np.zeros_like(f) + 0.5 * imax
    y = np.zeros_like(f) + 0.5 * jmax

    for t in range(maxiter):
        i = x.astype("i4").clip(0, imax - 2)
        j = y.astype("i4").clip(0, jmax - 2)
        p = x - i
        q = y - j

        # Bilinear estimate of F[x,y] and G[x,y]
        Fs = (
            (1 - p) * (1 - q) * F[i, j]
            + p * (1 - q) * F[i + 1, j]
            + (1 - p) * q * F[i, j + 1]
            + p * q * F[i + 1, j + 1]
        )
        Gs = (
            (1 - p) * (1 - q) * G[i, j]
            + p * (1 - q) * G[i + 1, j]
            + (1 - p) * q * G[i, j + 1]
            + p * q * G[i + 1, j + 1]
        )

        H = (Fs - f) ** 2 + (Gs - g) ** 2

        if np.all(H < tol):
            break

        # Estimate Jacobi matrix
        Fx = (1 - q) * (F[i + 1, j] - F[i, j]) + q * (F[i + 1, j + 1] - F[i, j + 1])
        Fy = (1 - p) * (F[i, j + 1] - F[i, j]) + p * (F[i + 1, j + 1] - F[i + 1, j])
        Gx = (1 - q) * (G[i + 1, j] - G[i, j]) + q * (G[i + 1, j + 1] - G[i, j + 1])
        Gy = (1 - p) * (G[i, j + 1] - G[i, j]) + p * (G[i + 1, j + 1] - G[i + 1, j])

        # Newton-Raphson step
        # Jinv = np.linalg.inv([[Fx, Fy], [Gx, Gy]])
        # incr = - np.dot(Jinv, [Fs-f, Gs-g])
        # x = x + incr[0], y = y + incr[1]
        det = Fx * Gy - Fy * Gx
        x -= (Gy * (Fs - f) - Fy * (Gs - g)) / det
        y -= (-Gx * (Fs - f) + Fx * (Gs - g)) / det

    return x, y


class HorzCRS:
    def __init__(self):
        pass

    def xy(self, lat, lon, z, t):
        raise NotImplementedError()

    def latlon(self, x, y, z, t):
        raise NotImplementedError()

    @staticmethod
    def from_name(name):
        return PyprojHorzCRS(named_crs(name))

    @staticmethod
    def from_array(lat, lon):
        return ArrayHorzCRS(lat, lon)


class PlainHorzCRS(HorzCRS):
    def __init__(self):
        super().__init__()

    def latlon(self, x, y, z, t):
        return x, y

    def xy(self, lat, lon, z, t):
        return lat, lon


class ArrayHorzCRS(HorzCRS):
    """A horizontal coordinate system based on a set of lat/lon arrays"""
    def __init__(self, lat, lon):
        """
        Returns a horizontal coordinate system based on a set of lat/lon arrays

        The lat/lon arrays are 2-dimensional arrays where the first axis is the y
        coordinate and the second one is the x coordinate.

        :param lat: The latitude of each grid point (y, x)
        :param lon: The longitude of each grid point (y, x)
        """
        super().__init__()
        self.lat = lat
        self.lon = lon

    def latlon(self, x, y, z, t):
        lat = map_coordinates(self.lat, [y, x], order=1)
        lon = map_coordinates(self.lon, [y, x], order=1)
        return lat, lon

    def xy(self, lat, lon, z, t):
        y, x = bilin_inv(lat, lon, self.lat, self.lon)
        return x, y


class PyprojHorzCRS(HorzCRS):
    def __init__(self, crs):
        super().__init__()
        self.crs = crs
        self._latlon_transform = None

    def _get_latlon_transform(self):
        if self._latlon_transform is None:
            wgs84 = pyproj.CRS.from_epsg(4326)
            self._latlon_transform = pyproj.Transformer.from_crs(self.crs, wgs84)
        return self._latlon_transform

    def xy(self, lat, lon, z, t):
        from pyproj.enums import TransformDirection
        transform = self._get_latlon_transform().transform
        return transform(lat, lon, direction=TransformDirection('INVERSE'))

    def latlon(self, x, y, z, t):
        transform = self._get_latlon_transform().transform
        return transform(x, y)


class TimeCRS:
    def posix(self, x, y, z, t):
        raise NotImplementedError()

    def t(self, x, y, z, posix):
        raise NotImplementedError()

    def to_array(self, x, y, z, t):
        return posix_to_numpy(self.posix(x, y, z, t))

    @staticmethod
    def from_array(numpy_times, t):
        return ArrayTimeCRS(numpy_times, t)

    @staticmethod
    def from_cf(ncatts):
        if 'calendar' in ncatts:
            return CFTimeCRS(ncatts['units'], ncatts['calendar'])
        else:
            return CFTimeCRS(ncatts['units'])


def numpy_to_posix(numpy_times):
    npdates = np.asarray(numpy_times).astype('datetime64[us]')
    epoch = np.datetime64('1970-01-01', 'us')
    one_sec = np.timedelta64(1000000, 'us')
    return (npdates - epoch) / one_sec


def posix_to_numpy(posix_times):
    epoch = np.datetime64('1970-01-01', 'us')
    one_sec = np.timedelta64(1000000, 'us')
    return epoch + posix_times * one_sec


class ArrayTimeCRS(TimeCRS):
    def __init__(self, numpy_times, t):
        super().__init__()

        self._posix = numpy_to_posix(numpy_times)
        self._tcoord = t

        from scipy.interpolate import UnivariateSpline
        self._interp_fwd = UnivariateSpline(self._posix, t, k=1, s=0, ext=0)
        self._interp_bwd = UnivariateSpline(t, self._posix, k=1, s=0, ext=0)

    def posix(self, x, y, z, t):
        return self._interp_bwd(t)

    def t(self, x, y, z, posix):
        return self._interp_fwd(posix)


class PlainTimeCRS(TimeCRS):
    def __init__(self):
        super().__init__()

    def posix(self, x, y, z, t):
        return t

    def t(self, x, y, z, posix):
        return posix


class CFTimeCRS(TimeCRS):
    def __init__(self, units, calendar="standard"):
        super().__init__()
        self.units = units
        self.calendar = calendar

    def posix(self, x, y, z, t):
        import cftime
        pydates = cftime.num2date(
            times=t,
            units=self.units,
            calendar=self.calendar,
            only_use_python_datetimes=True,
            only_use_cftime_datetimes=False,
        )
        npdates = np.array(pydates).astype('datetime64[us]')
        return numpy_to_posix(npdates)

    def t(self, x, y, z, posix):
        pydates = posix_to_numpy(posix).astype(object)
        return cftime.date2num(dates=pydates, units=self.units, calendar=self.calendar)


class VertCRS:
    def __init__(self):
        pass

    def depth(self, x, y, z, t):
        """
        Returns the depth (nonnegative, in meters) at the given grid coordinates
        """
        raise NotImplementedError()

    def z(self, x, y, depth, t):
        """
        Returns the vertical coordinate at the given depth and horizontal coordinates
        """
        raise NotImplementedError()

    @staticmethod
    def from_array(depth, z):
        """
        Returns a vertical coordinate system based on a depth array.

        The depth array is a 3-dimensional array where the first axis is the vertical
        axis and the remaining ones are the horizontal axes. The array should be
        organized so that depth[k, j, i] < depth[k + 1, j, i] <= 0 for all i, j, k.

        The z array is a 1-dimensional array (matching axis 0 of the depth array)
        containing the vertical coordinates for each depth level.

        :param depth: Depth in meters
        :param z: Vertical coordinate for each depth level
        """
        return ArrayVertCRS(depth, z)


class ArrayVertCRS(VertCRS):
    """
    A vertical coordinate system based on a depth array.
    """

    def __init__(self, depth, z):
        """
        Returns a vertical coordinate system based on a depth array.

        The depth array is a 3-dimensional array where the first axis is the vertical
        axis and the remaining ones are the horizontal axes. The array should be
        organized so that depth[k, j, i] < depth[k + 1, j, i] <= 0 for all i, j, k.

        The z array is a 1-dimensional array (matching axis 0 of the depth array)
        containing the vertical coordinates for each depth level.

        :param depth: Depth in meters
        :param z: Vertical coordinate for each depth level
        """
        self._depth = depth
        self._zcoord = z
        super().__init__()

    def _get_k_from_z(self, z):
        kcoord = np.arange(len(self._zcoord))
        return np.interp(z, self._zcoord, kcoord)

    def _get_z_from_k(self, k):
        return map_coordinates(self._zcoord, [k], order=1)

    def _get_depth_from_xyk(self, x, y, k):
        return map_coordinates(self._depth, [k, y, x], order=1)

    def depth(self, x, y, z, t):
        k = self._get_k_from_z(z)
        return self._get_depth_from_xyk(x, y, k)

    def z(self, x, y, depth, t):
        kmax = self._depth.shape[0]  # Number of vertical levels

        shape = (kmax, np.size(x))
        xx = np.broadcast_to(x.ravel(), shape)
        yy = np.broadcast_to(y.ravel(), shape)
        kk = np.broadcast_to(np.arange(kmax)[:, np.newaxis], shape)

        d = map_coordinates(self._depth, [kk, yy, xx], order=1)
        k_int = np.less(d, depth).sum(axis=0)
        k_int = np.maximum(np.minimum(k_int, kmax - 1), 1)
        depth_0 = self._get_depth_from_xyk(x, y, k_int - 1)
        depth_1 = self._get_depth_from_xyk(x, y, k_int)

        k_float = k_int - (depth_1 - depth) / (depth_1 - depth_0)
        return self._get_z_from_k(k_float)


def searchsorted(a, v, crd=None, side='left'):
    if crd is None:
        b = a[:, np.newaxis]
    else:
        b = a[(slice(None), ) + tuple(crd)]

    operator = dict(left=np.less, right=np.less_equal)[side]

    return operator(b, v).sum(axis=0)


class PlainVertCRS(VertCRS):
    def __init__(self):
        super().__init__()

    def depth(self, x, y, z, t):
        return z

    def z(self, x, y, depth, t):
        return depth


class FourDimCRS:
    def __init__(self, horz_crs: HorzCRS, vert_crs: VertCRS, time_crs: TimeCRS):
        self.horz_crs = horz_crs
        self.vert_crs = vert_crs
        self.time_crs = time_crs
        self.posix = self.time_crs.posix
        self.t = self.time_crs.t
        self.depth = self.vert_crs.depth
        self.z = self.vert_crs.z
        self.xy = self.horz_crs.xy
        self.latlon = self.horz_crs.latlon

    @staticmethod
    def from_roms_grid(fname_or_dset):
        with open_file_or_dset(fname_or_dset) as dset:
            return fourdim_crs_from_roms_grid(dset)


def create_depth_array_from_roms_dataset(dset):
    def interleave_w_and_rho_points(w_arr, rho_arr):
        return np.stack([w_arr, np.r_[rho_arr, 0]]).T.ravel()[:-1]

    c = interleave_w_and_rho_points(dset.Cs_w.values, dset.Cs_r.values)
    s = np.linspace(-1, 0, len(c))
    h = dset.h.values.ravel()
    hc = dset.hc.values.item()
    vtrans = dset.Vtransform.values.item()

    outshape = (len(c),) + dset.h.shape

    if vtrans == 1:  # Default transform by Song and Haidvogel
        A = hc * (s - c)[:, None]
        B = np.outer(c, h)
        return (A + B).reshape(outshape)

    elif vtrans == 2:  # New transform by Shchepetkin
        N = hc * s[:, None] + np.outer(c, h)
        D = 1.0 + hc / h
        return (N / D).reshape(outshape)

    else:
        raise ValueError(f'Unknkown Vtransform: {vtrans}')


class FourDimTransform:
    def __init__(self, crs_from: FourDimCRS, crs_to: FourDimCRS):
        self.crs_from = crs_from
        self.crs_to = crs_to

    def transform(self, xx, yy, zz, tt):
        # Horizontal coordinates
        if self.crs_from.horz_crs is self.crs_to.horz_crs:
            x2, y2 = xx, yy
        else:
            lat, lon = self.crs_from.latlon(xx, yy, zz, tt)
            x2, y2 = self.crs_to.xy(lat, lon, zz, tt)

        # Vertical coordinates
        if self.crs_from.vert_crs is self.crs_to.vert_crs:
            z2 = zz
        else:
            depth = self.crs_from.depth(xx, yy, zz, tt)
            z2 = self.crs_to.z(xx, yy, depth, tt)

        # Time coordinates
        if self.crs_from.time_crs is self.crs_to.time_crs:
            t2 = tt
        else:
            t_posix = self.crs_from.posix(xx, yy, zz, tt)
            t2 = self.crs_to.t(xx, yy, zz, t_posix)

        return x2, y2, z2, t2

    @staticmethod
    def from_roms_grid(file_or_dset):
        with open_file_or_dset(file_or_dset) as dset:
            fourdim_crs_from_roms_grid(dset)


def fourdim_crs_from_roms_grid(dset):
    horz_crs = HorzCRS.from_array(dset.lat_rho.values, dset.lon_rho.values)

    depth_array = create_depth_array_from_roms_dataset(dset)
    s = np.linspace(-1, 0, depth_array.shape[0])
    vert_crs = VertCRS.from_array(depth_array, s)

    np_times = dset.ocean_time.values
    time_crs = TimeCRS.from_array(np_times, np.arange(len(np_times)))

    return FourDimCRS(horz_crs, vert_crs, time_crs)


@contextmanager
def open_file_or_dset(d):
    if isinstance(d, str):
        import xarray as xr
        with xr.open_dataset(d) as dset:
            yield dset
    else:
        yield d
