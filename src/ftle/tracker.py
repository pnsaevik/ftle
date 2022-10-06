import numpy as np


def advect(x, y, z, t, u, v, w, dt, order):
    def velocity(crd_in, t_in):
        return np.stack([u(*crd_in, t_in), v(*crd_in, t_in), w(*crd_in, t_in)])

    c0 = np.stack([x, y, z])

    if order == 1:
        c_final = c0 + velocity(c0, t) * dt

    elif order == 2:
        t1 = t + 0.5 * dt
        c1 = c0 + 0.5 * dt * velocity(c0, t)
        c_final = c0 + dt * velocity(c1, t1)

    elif order == 4:
        v1 = velocity(c0, t)
        c1 = c0 + 0.5 * dt * v1

        v2 = velocity(c1, t + 0.5 * dt)
        c2 = c0 + 0.5 * dt * v2

        v3 = velocity(c2, t + 0.5 * dt)
        c3 = c0 + v3 * dt

        v4 = velocity(c3, t + dt)

        c_final = c0 + (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)

    else:
        raise NotImplementedError(f'Unknown integration order: {order}')

    return c_final[0], c_final[1], c_final[2], t + dt
