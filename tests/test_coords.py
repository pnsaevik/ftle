from ftle import coords
import pyproj
import numpy as np


class Test_named_crs:
    def test_can_return_epsg_projections(self):
        crs = coords.named_crs(4326)  # Regular lat/lon projection
        assert crs.name == "WGS 84"

    def test_can_return_ogr_projections(self):
        crs = coords.named_crs("urn:ogc:def:crs:OGC:1.3:CRS84")
        assert crs.name == "WGS 84 (CRS84)"

    def test_all_named_projections_are_valid(self):
        names = ['nk800', 'svim'] + [f'a{i:02}' for i in range(1, 14)]
        for name in names:
            crs = coords.named_crs(name)
            assert isinstance(crs, pyproj.CRS)


class Test_ArrayVertCRS_depth:
    def test_can_interpolate_when_depth_increases_with_z(self):
        depth_array = np.arange(24, dtype='f4').reshape((2, 3, 4))
        crs = coords.ArrayVertCRS(depth_array)
        d = crs.depth(x=[0, 0.5, 0, 0], y=[0, 0, 0.5, 0], z=[0, 0, 0, .5])
        assert d.tolist() == [0.0, 0.5, 2.0, 6.0]

    def test_can_interpolate_when_depth_decreases_with_z(self):
        depth_array = np.flip(np.arange(24, dtype='f4').reshape((2, 3, 4)), axis=0)
        assert depth_array[0, 0, 0] == 12
        crs = coords.ArrayVertCRS(depth_array)
        d = crs.depth(x=[0, 0.5, 0, 0], y=[0, 0, 0.5, 0], z=[0, 0, 0, .5])
        assert d.tolist() == [12.0, 12.5, 14.0, 6.0]

