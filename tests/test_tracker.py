from ftle import tracker
import numpy as np
import pytest


class Test_advect:
    @pytest.fixture(scope='class')
    def params(self):
        def u(xx, *_):
            return np.ones_like(xx)

        def v(xx, *_):
            return 2*np.ones_like(xx)

        def w(xx, *_):
            return 3*np.ones_like(xx)

        x = np.array([1, 2, 3, 4], dtype='f4')
        y = np.array([4, 5, 6, 7], dtype='f4')
        z = np.array([7, 8, 9, 10], dtype='f4')
        t = 10
        dt = 1

        return x, y, z, t, u, v, w, dt

    def test_correct_when_linear_field_and_order_1(self, params):
        x, y, z, t, u, v, w, dt = params
        x2, y2, z2, t2 = tracker.advect(x, y, z, t, u, v, w, dt, order=1)

        assert x2.tolist() == [2, 3, 4, 5]
        assert y2.tolist() == [6, 7, 8, 9]
        assert z2.tolist() == [10, 11, 12, 13]
        assert t2 == t + dt

    def test_correct_when_linear_field_and_order_2(self, params):
        x, y, z, t, u, v, w, dt = params
        x2, y2, z2, t2 = tracker.advect(x, y, z, t, u, v, w, dt, order=2)

        assert x2.tolist() == [2, 3, 4, 5]
        assert y2.tolist() == [6, 7, 8, 9]
        assert z2.tolist() == [10, 11, 12, 13]
        assert t2 == t + dt

    def test_correct_when_linear_field_and_order_4(self, params):
        x, y, z, t, u, v, w, dt = params
        x2, y2, z2, t2 = tracker.advect(x, y, z, t, u, v, w, dt, order=4)

        assert x2.tolist() == [2, 3, 4, 5]
        assert y2.tolist() == [6, 7, 8, 9]
        assert z2.tolist() == [10, 11, 12, 13]
        assert t2 == t + dt

    def test_returns_verbatim_z_if_no_vertical_velocity(self, params):
        x, y, z, t, u, v, w, dt = params
        x2, y2, z2, t2 = tracker.advect(x, y, z, t, u, v, None, dt, order=1)

        assert x2.tolist() == [2, 3, 4, 5]
        assert y2.tolist() == [6, 7, 8, 9]
        assert z2 is z
        assert t2 == t + dt
