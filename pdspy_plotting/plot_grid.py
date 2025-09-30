
import numpy as np
import matplotlib.pyplot as plt
from .plot_channel_maps import plot_channel_maps_bd

def plot_grid(model, visibilities, m_adj, params, params_adj, residual, v_start, v_end,
              index, image_cmap, contours_colors, fontsize,
              skip, v_width, plot_name, nrows, ncols, levels, negative_levels, outdir):
    plt.close()
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, dpi=300)
    for n in range(3):
        # first, plot the data
        if n == 0:
            print('plot data')
            plot_type = "data"

            show_velocity= True
            show_beam = False

            # define the plot range vmin and vmax
            vmin = 0.0
            vmax = np.nanmax(visibilities['image'][0].image) # max of image

            # define contour levels
            sigma = np.nanstd(visibilities['image'][0].image)
            # levels = np.linspace(6.0, 31.0, 10) * sigma
            # negative_levels = np.linspace(-31.0, -6.0, 10) * sigma
            levels *= sigma
            negative_levels *= sigma

            # determine how much to clean
            maxiter = 600
            threshold = 0.5

            # label plot as an observation
            ax[n, 0].text(0.5, 0.2, 'Observations')

            # get name of molecule
            molecule = str(visibilities['lam'][0])
            
            # title the plot with the name of the molecule
            ax[n, 0].set_title(molecule)

            # dont show the x label
            show_xlabel=False
        
        # next, plot the model
        elif n == 1:
            print('plot clean model')
            plot_type = "model"

            show_velocity = True
            show_beam = False

            # define the plot range vmin and vmax
            vmin = 0.0
            vmax = np.nanmax(visibilities['image'][0].image) # max of image

            # define contour levels
            sigma = np.nanstd(visibilities['image'][0].image)
            # levels = np.linspace(4.0, 31.0, 10) * sigma
            # negative_levels = np.linspace(-31.0, -4.0, 10) * sigma
            levels *= sigma
            negative_levels *= sigma

            # determine how much to clean
            maxiter = 1000
            threshold = 0.

            # label plot as a model
            ax[n, 0].text(0.5, 0.2, 'Model')

            # don't show the x label
            show_xlabel = False

        # lastly, plot the residuals
        elif n == 2:
            plot_type = "residuals"
            print('plot residuals')

            # label plot as a residual
            ax[n, 0].text(0.5, 0.2, 'Residuals')

            # define contour levels
            sigma = np.nanstd(visibilities['image'][0].image)
            levels *= sigma
            negative_levels *= sigma
            # levels = np.linspace(7.0, 31.0, 10) * sigma
            # negative_levels = np.linspace(-31.0, -7.0, 10) * sigma

            # show the x label for the bottom row
            show_xlabel = True

        # create the plot
        plot_channel_maps_bd(visibilities, m_adj, params, params_adj, residual, v_start, v_end,
                            index=0, fig=(fig, ax[n:n+1, :]), image=plot_type, contours=plot_type,
                            vmin=vmin, vmax=vmax, levels=levels, negative_levels=negative_levels,
                            image_cmap='BlueToRed', contours_colors='k', fontsize=7,
                            show_velocity=show_velocity, show_xlabel=show_xlabel,
                            skip=1, v_width=v_width)
        
    # set figure size
    fig.set_size_inches((10.5, 4.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.08,
                        wspace=0., hspace=0.0)
    fig.tight_layout()

    plt.savefig(outdir + "{0}_{1}.png".format(model, plot_name))
