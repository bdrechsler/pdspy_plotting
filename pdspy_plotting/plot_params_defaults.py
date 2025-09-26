import numpy as np

source = "" # name of the source
v_width = 6.0 # velocity range to plot in km/s
models = []  # models to plot
base_path = ".." # root directory of models
plot_type = 'full' # determine to make one plot (full) or split into seperate blue and red shifted plots (seperate)
fontsize = 7
ncol = 7
levels = np.linspace(6.0, 31.0, 10)
negative_levels = np.linspace(-31.0, -6.0, -10)
ncpu=8 # number of cpus
