import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

keeps_these_objects_away_from_the_gc = []

def plotpause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def plotshow():
    """ Equivalent of plt.show(), however does not force focus on windows 
    and should exit on ctrl-c """
    plt.show()
    interval = 0.01
    try:
        while True:
            backend = plt.rcParams['backend']
            if backend in matplotlib.rcsetup.interactive_bk:
                figManager = matplotlib._pylab_helpers.Gcf.get_active()
                if figManager is not None:
                    canvas = figManager.canvas
                    if canvas.figure.stale:
                        canvas.draw()
                    canvas.start_event_loop(interval)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting wait loop.")

def button_callback(event):
    print("Closing all windows")
    plt.close('all')
def plot_closeall_button():
    """ If the returned button object is not collected somewhere,
    it will be GCed and the button unresponsive. """
    fig = plt.figure("closeall_button")
    ax = plt.axes()
    bca = Button(ax, 'Close All', color=(1., 0.8, 0.8, 1))
    bca.on_clicked(button_callback)
    keeps_these_objects_away_from_the_gc.append(bca)
    return bca

