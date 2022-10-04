import pyproj
from scipy.ndimage import map_coordinates
import numpy as np


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


class VertCRS:
    def __init__(self):
        pass

    def depth(self, x, y, z):
        """
        Returns the depth (nonnegative, in meters) at the given grid coordinates
        """
        raise NotImplementedError()

    def z(self, x, y, depth):
        """
        Returns the vertical coordinate at the given depth and horizontal coordinates
        """
        raise NotImplementedError()


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

    def depth(self, x, y, z):
        k = self._get_k_from_z(z)
        return self._get_depth_from_xyk(x, y, k)

    def z(self, x, y, depth):
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


