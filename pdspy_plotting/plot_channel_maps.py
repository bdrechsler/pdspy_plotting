from pdspy.interferometry import Visibilities, clean, average
from pdspy.constants.physics import c
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy
import numpy as np

class Transform:
    def __init__(self, xmin, xmax, dx, fmt):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)

def plot_channel_maps_bd(visibilities, model_adj, params, params_adj, residuals,
                         v_start, v_end,
                         index=0, fig=None, image="data", contours="model", 
                         vmin=None, vmax=None, levels=None, negative_levels=None, 
                         image_cmap="viridis", contours_colors=None, fontsize="medium",
                         show_velocity=True, show_xlabel=True, 
                         show_ylabel=True, skip=0,v_width=None,uvtaper=None):
    
    # set up figure if not was provided
    nrows = visibilities["nrows"][index]
    ncols = visibilities["ncols"][index]
    if fig == None:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,9))
    else:
        fig, ax = fig

    # calculate velocity of each image
    line_freq = float(visibilities["freq"][index])*1.0e9 #Hz
    freq_array = visibilities["image"][index].freq
    v = c * (line_freq - freq_array) / line_freq

    # set the ticks
    ticks = visibilities["image_ticks"][index]

    # first plot image, then plot contours
    for i in range(2):
        if i == 0:
            plot_type = image
        else:
            plot_type = contours

        # if plotting the data, load in the image
        if plot_type == "data":
            plot_image = visibilities["image"][index]
        # if plotting the model, get the image from the model
        elif plot_type == "model":
            plot_image = model_adj.images[visibilities["lam"][index]]
        # use input residuals
        elif plot_type == "residuals":
            plot_image = residuals

        # set vmin and vmax if none are given
        if i == 0:
            if vmin is None:
                vmin = np.nanmin(plot_image.image)
            if vmax is None:
                vmax = np.nanmax(plot_image.image)
        
        # set contour levels if none are given
        if i == 1:
            if levels is None:
                levels = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95]) * plot_image.image.max()

    # Get the correct range of pixels for making the sub-image.

        if plot_type == "data" or plot_type == "residuals":
            xmin, xmax = int(round(visibilities["image_npix"][index]/2+\
                    visibilities["x0"][index]/\
                    visibilities["image_pixelsize"][index]+ \
                    params_adj["x0"]/visibilities["image_pixelsize"][index]+ \
                    ticks[0]/visibilities["image_pixelsize"][index])), \
                    int(round(visibilities["image_npix"][index]/2+\
                    visibilities["x0"][index]/\
                    visibilities["image_pixelsize"][index]+ \
                    params_adj["x0"]/visibilities["image_pixelsize"][index]+ \
                    ticks[-1]/visibilities["image_pixelsize"][index]))

            ymin, ymax = int(round(visibilities["image_npix"][index]/2-\
                    visibilities["y0"][index]/\
                    visibilities["image_pixelsize"][index]- \
                    params_adj["y0"]/visibilities["image_pixelsize"][index]+ \
                    ticks[0]/visibilities["image_pixelsize"][index])), \
                    int(round(visibilities["image_npix"][index]/2-\
                    visibilities["y0"][index]/\
                    visibilities["image_pixelsize"][index]- \
                    params_adj["y0"]/visibilities["image_pixelsize"][index]+ \
                    ticks[-1]/visibilities["image_pixelsize"][index]))
        else:
            xmin, xmax = int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[0]/visibilities["image_pixelsize"][index])), \
                    int(round(visibilities["image_npix"][index]/2+1 +\
                    ticks[-1]/visibilities["image_pixelsize"][index]))
            ymin, ymax = int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[0]/visibilities["image_pixelsize"][index])), \
                    int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[-1]/visibilities["image_pixelsize"][index]))
            
        # loop through channels and plot
        # starting and ending indicies
        start = (np.abs(v/1e5 - v_start)).argmin()
        end = (np.abs(v/1e5 - v_end)).argmin()
   
        plot_chans = np.linspace(start, end, ncols)

        for k in range(nrows):
            for l in range(ncols):
                # ind = (k*ncols + l)*(skip+1) + visibilities["ind0"][index] + start
                ind = int(plot_chans[l])

                if ind >= v.size:
                    ax[k, l].set_axis_off()
                    print('Index greater than array size, skipping', v.size, ind)
                    continue

                # set up color map
                if image_cmap == "BlueToRed":
                    if v[ind]/1e5 < params["v_sys"]:
                        cdict1 = {'red':   ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0)),
                                  'green': ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0)),
                                  'blue':  ((0.0, 1.0, 1.0),
                                            (1.0, 1.0, 1.0))}
                        blues = LinearSegmentedColormap('blues', cdict1)
                        cmap = blues
                    else:
                        cdict2 = {'red':   ((0.0, 1.0, 1.0),
                                            (1.0, 1.0, 1.0)),
                                  'green': ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0)),
                                  'blue':  ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0))}
                        reds = LinearSegmentedColormap('reds', cdict2)
                        cmap = reds

                    scalefn = lambda x: (numpy.arctan(x*7-4)+1.3) / \
                            (numpy.arctan(1*7-4)+1.3)
                else:
                    cmap = image_cmap
                    scalefn = lambda x: 1.

                # on first loop plot image
                # on second loop plot contours
                if i == 0:
                    # plot the image
                    ax[k, l].imshow(plot_image.image[ymin:ymax, xmin:xmax, \
                                    ind, 0]*scalefn(abs(v[ind]/1e5 - params["v_sys"])),
                                    origin="lower", interpolation="nearest", vmin=vmin,
                                    vmax=vmax, cmap=cmap)
                    # add a + to the center of the image
                    xloc = (xmin + xmax)/2 - xmin
                    yloc = (ymin + ymax)/2 - ymin
                    ax[k, l].plot(xloc, yloc, 'k+', markersize=8)

                    # Add the velocity to the map.

                    if show_velocity:
                        if k == 0 and l == 0:
                            if abs(v[ind]/1e5) < 0.001:
                                print("Hello!")
                                v[ind] = +0.
                            txt = ax[k,l].annotate(r"$v=%{0:s}$ km s$^{{-1}}$".\
                                    format(visibilities["fmt"][index]) % \
                                    (round(v[ind]/1e5,1)), xy=(0.98,0.85), \
                                    xycoords='axes fraction', \
                                    horizontalalignment="right", \
                                    fontsize=fontsize)
                        else:
                            txt = ax[k,l].annotate(r"$%{0:s}$ km s$^{{-1}}$".\
                                    format(visibilities["fmt"][index]) % \
                                    (v[ind]/1e5), xy=(0.98,0.85), \
                                    xycoords='axes fraction', \
                                    horizontalalignment="right", \
                                    fontsize=fontsize)
                            
                    # fix the axes labels
                    transformx = ticker.FuncFormatter(Transform(xmin, xmax,\
                                visibilities["image_pixelsize"][index],'%.1f"'))
                    transformy = ticker.FuncFormatter(Transform(ymin, ymax,\
                            visibilities["image_pixelsize"][index],'%.1f"'))

                    ax[k,l].set_xticks((xmin + xmax)/2+\
                            ticks[1:-1]/visibilities["image_pixelsize"]\
                            [index]-xmin)
                    ax[k,l].set_yticks((ymin + ymax)/2+\
                            ticks[1:-1]/visibilities["image_pixelsize"]\
                            [index]-ymin)

                    ax[k,l].get_xaxis().set_major_formatter(transformx)
                    ax[k,l].get_yaxis().set_major_formatter(transformy)

                    # Adjust the tick labels.

                    ax[k,l].tick_params(labelsize=fontsize,direction='in')

                    # Add a label to the x-axis.

                    if show_xlabel:
                        ax[-1,l].set_xlabel("$\Delta$RA", fontsize=fontsize)
                    
                    ax[k, l].set_adjustable('box')

                    # label the y axis
                    if show_ylabel:
                        ax[k, 0].set_ylabel("$\Delta$Dec", fontsize=fontsize)

                # plot the contours
                elif i == 1:
                    if len(numpy.where(plot_image.image[ymin:ymax,\
                                xmin:xmax,ind,0] > levels.min())[0]) > 0:
                            ax[k,l].contour(plot_image.image[ymin:ymax,\
                                    xmin:xmax,ind,0], levels=levels, \
                                    colors=contours_colors)
                    # Plot the negative contours, if requested.
                    if negative_levels is not None:
                        if len(numpy.where(plot_image.image[ymin:ymax,\
                                xmin:xmax,ind,0] < negative_levels.max())\
                                [0]) > 0:
                            ax[k,l].contour(plot_image.image[ymin:ymax,\
                                    xmin:xmax,ind,0], \
                                    levels=negative_levels, \
                                    linestyles="--", colors=contours_colors)
                            
    # return figure and axes
    return fig, ax

        
        

