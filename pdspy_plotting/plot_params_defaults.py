import numpy as np

source = "" # name of the source
v_width = 6.0 # velocity range to plot in km/s
models = []  # models to plot
base_path = ".." # root directory of models
plot_type = 'full' # determine to make one plot (full) or split into seperate blue and red shifted plots (seperate)
fontsize = 7
ncol = 7
levels = np.linspace(6.0, 31.0, 10)
contour_levels = {"data": (6.0, 31.0),
                  "model": (4.0, 31.0),
                  "residuals": (7.0, 31.0),
                  "nlevels": 10}
plot_size = (12, 3)
ticks = [-1.5, 0, 1.5]
ncpu=8 # number of cpus
