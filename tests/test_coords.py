from ftle import coords
import pyproj


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
