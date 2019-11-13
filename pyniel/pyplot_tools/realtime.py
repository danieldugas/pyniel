import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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
    fig = plt.figure("closeall_button")
    plt.plot([0], [0])
    bca = Button(plt.gca(), 'Close All', color=(1., 0.8, 0.8, 1))
    cid = fig.canvas.mpl_connect('button_press_event', button_callback)

