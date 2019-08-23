import matplotlib

def movefigure(fig, pos):
    """ Moves pyplot figure window to new position where top left corner is at pixel coordinates (px, py)"""
    px, py = pos
    backend = matplotlib.get_backend()
    if backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((px, py))
    elif backend in ['GTK', 'GTKAgg', 'GTKCairo', 'GTK3Agg', 'GTK3Cairo', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo']:
        fig.canvas.manager.window.move(px, py)
    elif backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+{}+{}".format(px, py))
    else:
        raise NotImplementedError
