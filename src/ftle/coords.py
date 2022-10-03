import pyproj
from scipy.ndimage import map_coordinates


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

    def __init__(self, depth):
        """
        Returns a vertical coordinate system based on a depth array.

        The depth array is a 3-dimensional array where the first axis is the vertical
        axis and the remaining ones are the horizontal axes.

        :param depth: The depth of each grid point (nonnegative), in meters
        """
        self._depth = depth
        super().__init__()

    def depth(self, x, y, z):
        return map_coordinates(self._depth, [z, y, x], order=1)

