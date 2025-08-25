#!/usr/bin/env python

import pdspy.plotting as plotting
import pdspy.interferometry as uv
import pdspy.modeling as modeling
import pdspy.utils as utils
import pdspy.imaging as imaging
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import sys
import os
from mpi4py import MPI
import argparse
from .residuals import create_residual_image
from .plot_channel_maps import plot_channel_maps_bd

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default=None, help='path to plot_params.py')
    parser.add_argument('-s', '--source', default=None, help='source name')
    parser.add_argument('-w', '--width', default=6, help='width of velocity range to plot')
    parser.add_argument('-p', '--path', default="../", help='root directory of models')
    parser.add_argument('-m', '--models', nargs='+', default=['exptaper'], help='models to plot')
    parser.add_argument('-t', '--type', choices=['full', 'serperate', 'both'], default='both',
                        help='determine if one plot is made or separeate blue/red-shifted plots, or both')
    parser.add_argument('-n', '--ncpu', default=40, help='number of cpus to use')

    args = parser.parse_args()
    # if a config file is provided, use those plot parameters
    if args.config != None:
        sys.path.append(args.config)
        import plot_params
    # otherwise, use the provided plot parameters
    else:
        plot_params = args

    comm = MPI.COMM_WORLD

    # set the number of cpus to use
    ncpus = plot_params.ncpu

    # Get the source name
    source = plot_params.source

    # define the velocity range to plot
    v_width = plot_params.v_width

    # define model to plot
    models = plot_params.models 

    # path to where models are stored
    base_path = plot_params.base_path 
    source_path = base_path + source + '/'

    # loop through the models and plot them
    for model in models:
        print(model)
        # define the path to the model
        model_path = source_path + model + '/'

        # load in the config file
        config = utils.load_config(path=model_path)

        # ensure that all parameters are present
        config.parameters = modeling.check_parameters(config.parameters)

        # update the visibility dictionary
        # update the file path
        rel_path = config.visibilities['file'][0]
        abs_path = rel_path.replace('../', source_path)
        config.visibilities['file'] = [abs_path]

        # update the image path
        rel_path = config.visibilities['image_file'][0]
        abs_path = rel_path.replace("../", source_path)
        config.visibilities['image_file'] = [abs_path]

        # change the image ticks
        config.visibilities['image_ticks'] = [np.array([-1.5, 0, 1.5])]

        # update nrows and ncols
        config.visibilities['nrows'] = [1,1,1]
        if plot_params.plot_type == 'seperate':
            config.visibilities['ncols'] = [7, 7, 7]
        elif plot_params.plot_type == 'full':
            config.visibilities['ncols'] = [11, 11, 11]

        # not sure what ind0 is 
        config.visibilities['ind0'] = [2,2,2]

        # set the format (not sure what is being formatted)
        config.visibilities['fmt'] = ['5.1f', '5.1f', '5.1f']

        ## read in mm visibilites
        visibilities, images, spectra = utils.load_data(config, model="flared")

        # load in dynesty resulsts
        keys, params, sigma, samples = utils.load_results(config, model_path=model_path, code='dynesty',
                                                    best='peak', unc='std', percentile=68.)

        # adjust the parameters
        params_adj = params.copy()
        params_adj['x0'] *= -1.0
        params_adj['y0'] *= -1.0
        params_adj['pa'] *= -1.0

        # define start and end velocities for the red and blue figures
        v_start_b = params["v_sys"] - (v_width / 2.0)
        v_end_b = params["v_sys"]
        v_start_r = params["v_sys"]
        v_end_r = params["v_sys"] + (v_width / 2.0)

        # load in models and residuals if they exists
        # if not, generate them
        if os.path.exists(model_path + 'm.hdf5'):
            m = modeling.YSOModel()
            m.read(model_path + 'm.hdf5')
        else:
            m = modeling.run_flared_model(visibilities=config.visibilities, params=params, parameters=config.parameters, 
                                        plot=True, ncpus=ncpus, source=source, plot_vis=False, 
                                        ftcode='galario-unstructured')
            m.write(model_path + 'm.hdf5')

        if os.path.exists(model_path + 'm_adj.hdf5'):
            m_adj = modeling.YSOModel()
            m_adj.read("../{}/m_adj.hdf5".format(model))
        else:
            m_adj = modeling.run_flared_model(visibilities=config.visibilities, params=params_adj, parameters=config.parameters, 
                                        plot=True, ncpus=ncpus, source=source, plot_vis=False)
            m_adj.write(model_path + 'm_adj.hdf5')

        if os.path.exists(model_path + "residual_image.hdf5"):
            residual = imaging.Image()
            residual.read("../{}/residual_image.hdf5".format(model))
        else:
            residual = create_residual_image(visibilities, m)
            residual.write(model_path + "residual_image.hdf5")


        # create the plot
        if params.plot_type == 'seperate':
            nplots = 1
        elif params.plot_type == 'full':
            nplots=2
        elif param.plot_type == 'both':
            nplots=3

        for i in range(nplots):
            plt.close()
            if params.plot_type == 'seperate' or params.plot_type == 'both':
                fig, ax = plt.subplots(nrows=3, ncols=7, sharex=True, sharey=True, dpi=300)
                if i == 0:
                    print('making blue figure')
                    v_start = v_start_b
                    v_end = v_end_b
                    color = 'blue'
                    print(v_start, v_end)
                elif i == 1:
                    print('making red figure')
                    v_start = v_start_r
                    v_end = v_end_r
                    color = 'red'
                    print(v_start, v_end)
            elif params.plot_type == 'full' or (params.plot_type == 'both' and i == 2):
                fig, ax = plt.subplots(nrows=3, ncols=11, sharex=True, sharey=True, dpi=300)
                v_start = v_start_b
                v_end = v_end_r
                color= 'full'

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
                    levels = np.linspace(6.0, 31.0, 10) * sigma
                    negative_levels = np.linspace(-31.0, -6.0, 10) * sigma

                    # determine how much to clean
                    maxiter = 600
                    threshold = 0.5

                    # label plot as an observation
                    ax[n, 0].text(0.5, 0.2, 'Observations')

                    # get name of molecule
                    molecule = str(visibilities['lam'][0])
                    print(molecule + " " + str(visibilities['freq'][0]) + ' GHz') # print molecule and frequency
                    
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
                    levels = np.linspace(4.0, 31.0, 10) * sigma
                    negative_levels = np.linspace(-31.0, -4.0, 10) * sigma

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
                    levels = np.linspace(7.0, 31.0, 10) * sigma
                    negative_levels = np.linspace(-31.0, -7.0, 10) * sigma

                    # show the x label for the bottom row
                    show_xlabel = True

                # create the plot
                print('plotting row')
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

            plt.savefig(source_path + "plots/{0}_{1}.png".format(model, color))
        
if __name__ == '__main__':
    main()
