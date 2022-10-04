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
    def test_can_interpolate(self):
        depth_array = -24 + np.arange(24, dtype='f4').reshape((2, 3, 4))
        z_coord = np.array([0, 1000])
        crs = coords.ArrayVertCRS(depth_array, z_coord)
        d = crs.depth(x=[0, 0.5, 0, 0], y=[0, 0, 0.5, 0], z=[0, 0, 0, 500])
        assert d.tolist() == [-24, -23.5, -22, -18]

    def test_extrapolates_constant_value(self):
        depth_array = np.array([-4, -3, -2, -1, 0]).reshape((-1, 1, 1))
        z_coord = np.arange(depth_array.size)
        crs = coords.ArrayVertCRS(depth_array, z_coord)
        d = crs.depth(x=[0, 0, 0, 0], y=[0, 0, 0, 0], z=[0, 1, -1, 6])
        assert d.tolist() == [-4, -3, -4, 0]


class Test_ArrayVertCRS_z:
    def test_can_return_fractional_z_values(self):
        depth_array = np.array([-24, -16, -8]).reshape((3, 1, 1))
        z_coord = np.array([0., 1000., 2000.])
        crs = coords.ArrayVertCRS(depth_array, z_coord)
        x = np.array([0, 0, 0, 0, 0])
        y = np.array([0, 0, 0, 0, 0])
        depth = -np.array([24, 22, 20, 16, 12])
        z = crs.z(x=x, y=y, depth=depth)
        assert z.tolist() == [0, 250, 500, 1000, 1500]

    def test_can_interpolate_fractional_xy_values(self):
        depth_array = -16 + np.arange(16, dtype='f4').reshape((2, 2, 4))
        z_coord = np.array([0., 1000.])
        crs = coords.ArrayVertCRS(depth_array, z_coord)
        x = np.array([0, 0, 0, 0, 0])
        y = np.array([0, 0.25, 0.5, 0.75, 1])  # maxdepth goes from -8 to -16
        depth = -np.array([8, 8, 8, 8, 8])
        z = crs.z(x=x, y=y, depth=depth)
        assert z.tolist() == [1000, 875, 750, 625, 500]


class Test_searchsorted:
    def test_matches_numpy_for_onedim_increasing_array(self):
        a = np.array([0, 1, 5, 10, 50, 100])
        v = [-1, 0, .5, 1, 1.5, 100, 101]
        np_result = np.searchsorted(a, v)
        result = coords.searchsorted(a, v)
        assert result.tolist() == np_result.tolist()

    def test_matches_numpy_for_onedim_increasing_array_if_side_right(self):
        a = np.array([0, 1, 5, 10, 50, 100])
        v = [-1, 0, .5, 1, 1.5, 100, 101]
        np_result = np.searchsorted(a, v, side='right')
        result = coords.searchsorted(a, v, side='right')
        assert result.tolist() == np_result.tolist()

    def test_matches_numpy_for_twodim_increasing_array(self):
        a = np.array([
            [0, 1, 5, 10, 50, 100],
            [1000, 1001, 1005, 1010, 1050, 1100],
        ]).T
        v = np.array([
            [-1, 0, .5, 1, 2, 100, 101],
            [999, 1000, 1000.5, 1001, 1002, 1100, 1101],
        ])
        i = np.array([[0] * 7, [1] * 7])
        np_result_0 = np.searchsorted(a[:, 0], v[0])
        np_result_1 = np.searchsorted(a[:, 1], v[1])
        result = coords.searchsorted(a, v.ravel(), [i.ravel()])
        assert result.tolist() == np_result_0.tolist() + np_result_1.tolist()
