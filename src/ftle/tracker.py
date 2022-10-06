def advection(x, y, u, v, dt, order=1):
    if order == 1:
        x2 = x + u.interp(x=x, y=y) * dt
        y2 = y + v.interp(x=x, y=y) * dt
    else:
        raise NotImplementedError(f'Unknown integration order: {order}')

    return x2, y2
