#!/usr/bin/env python

import pdspy.plotting as plotting
import pdspy.interferometry as uv
import pdspy.modeling as modeling
import pdspy.utils as utils
import pdspy.imaging as imaging
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import importlib.util
import sys
import os
from mpi4py import MPI
import argparse
from .residuals import create_residual_image
from .plot_grid import plot_grid

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default=None, help='path to plot_params.py')


    args = parser.parse_args()

    param_path = os.path.dirname(args.config)
    # import plot parameters
    if os.path.exists(args.config):
        spec = importlib.util.spec_from_file_location("parameters", args.config)
        plot_params = importlib.util.module_from_spec(spec)
        sys.modules["parameters"] = plot_params
        spec.loader.exec_module(plot_params)
    else:
        print("Parameter file not found")

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
        config.visibilities['image_ticks'] = [np.array(plot_params.ticks)]

        # update nrows and ncols
        config.visibilities['nrows'] = [1,1,1]
        config.visibilities['ncols'] = [plot_params.ncol, plot_params.ncol, plot_params.ncol]

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
        if os.path.exists(param_path + '/models/{}_m.hdf5'.format(model)):
            print("reading model")
            m = modeling.YSOModel()
            m.read(param_path + '/models/{}_m.hdf5'.format(model))
        else:
            print("generating model")
            os.makedirs(param_path + '/models/', exist_ok=True)
            m = modeling.run_flared_model(visibilities=config.visibilities, params=params, parameters=config.parameters, 
                                        plot=True, ncpus=ncpus, source=source, plot_vis=False, 
                                        ftcode='galario-unstructured')
            m.write(param_path + '/models/{}_m.hdf5'.format(model))

        # make plots directory if it doesn't exist
        os.makedirs(param_path + "plots/", exist_ok=True)
        if os.path.exists(param_path + '/models/{}_m_adj.hdf5'.format(model)):
            print("reading adjusted model")
            m_adj = modeling.YSOModel()
            m_adj.read(param_path + '/models/{}_m_adj.hdf5'.format(model))
        else:
            print("generating adjusted model")
            m_adj = modeling.run_flared_model(visibilities=config.visibilities, params=params_adj, parameters=config.parameters, 
                                        plot=True, ncpus=ncpus, source=source, plot_vis=False)
            m_adj.write(param_path + '/models/{}_m_adj.hdf5'.format(model))

        if os.path.exists(param_path + "/res_imgs/{}_res_img.hdf5".format(model)):
            print("reading residual image")
            residual = imaging.Image()
            residual.read(param_path + "/res_imgs/{}_res_img.hdf5".format(model))
        else:
            print("generating residual image")
            os.makedirs(param_path + '/res_imgs/', exist_ok=True)
            residual = create_residual_image(visibilities, m)
            residual.write(param_path + "/res_imgs/{}_res_img.hdf5".format(model))
        
        if plot_params.plot_type == 'seperate':
            print('making blue plot')
            plot_grid(model, visibilities, m_adj, params, params_adj, residual,
                      v_start_b, v_end_b, 0, 'BlueToRed', 'k', 7, 1,
                      v_width, 'blue', 3, plot_params.ncol, plot_params.contour_levels,
                      plot_size=plot_params.plot_size, outdir=param_path)
            print('making red plot')
            plot_grid(model, visibilities, m_adj, params, params_adj, residual,
                      v_start_r, v_end_r, 0, 'BlueToRed', 'k', 7, 1,
                      v_width, 'red', 3, plot_params.ncol, plot_params.contour_levels,
                      plot_size=plot_params.plot_size, outdir=param_path)
        if plot_params.plot_type == 'full':
            print('making full plot')
            plot_grid(model, visibilities, m_adj, params, params_adj, residual,
                      v_start_b, v_end_r, 0, 'BlueToRed', 'k', 7, 1,
                      v_width, 'full', 3, plot_params.ncol, plot_params.contour_levels,
                      plot_size=plot_params.plot_size, outdir=param_path + "/plots/")

if __name__ == '__main__':
    main()
