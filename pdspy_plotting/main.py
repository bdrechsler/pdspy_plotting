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
from .plot_grid import plot_grid

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
        if os.path.exists('models/{}_m.hdf5'.format(model)):
            print("reading model")
            m = modeling.YSOModel()
            m.read('models/{}_m.hdf5'.format(model))
        else:
            print("generating model")
            m = modeling.run_flared_model(visibilities=config.visibilities, params=params, parameters=config.parameters, 
                                        plot=True, ncpus=ncpus, source=source, plot_vis=False, 
                                        ftcode='galario-unstructured')
            m.write('models/{}_m.hdf5'.format(model))

        if os.path.exists('models/{}_m_adj.hdf5'.format(model)):
            print("reading adjusted model")
            m_adj = modeling.YSOModel()
            m_adj.read('models/{}_m_adj.hdf5'.format(model))
        else:
            print("generating adjusted model")
            m_adj = modeling.run_flared_model(visibilities=config.visibilities, params=params_adj, parameters=config.parameters, 
                                        plot=True, ncpus=ncpus, source=source, plot_vis=False)
            m_adj.write('models/{}_m_adj.hdf5'.format(model))

        if os.path.exists("res_imgs/{}_res_img.hdf5".format(model)):
            print("reading residual image")
            residual = imaging.Image()
            residual.read("res_imgs/{}_res_img.hdf5".format(model))
        else:
            print("generating residual image")
            residual = create_residual_image(visibilities, m)
            residual.write("res_imgs/{}_res_img.hdf5".format(model))
        
        if plot_params.plot_type == 'seperate' or plot_params.plot_type == 'both':
            print('making blue plot')
            plot_grid(model, visibilities, m_adj, params, params_adj, residual,
                      v_start_b, v_end_b, 0, 'BlueToRed', 'k', 7, 1,
                      v_width, 'blue', 3, 7)
            print('making red plot')
            plot_grid(model, visibilities, m_adj, params, params_adj, residual,
                      v_start_r, v_end_r, 0, 'BlueToRed', 'k', 7, 1,
                      v_width, 'red', 3, 7)
        if plot_params.plot_type == 'full' or plot_params.plot_type == 'both':
            print('making full plot')
            plot_grid(model, visibilities, m_adj, params, params_adj, residual,
                      v_start_b, v_end_r, 0, 'BlueToRed', 'k', 7, 1,
                      v_width, 'full', 3, 11)


        

if __name__ == '__main__':
    main()
