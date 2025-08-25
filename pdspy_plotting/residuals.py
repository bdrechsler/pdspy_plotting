from pdspy.interferometry import Visibilities, clean, average

def create_residual_image(visibilities, m):
    # initialize with the data visibilities
    residuals = Visibilities(visibilities['data'][0].u,
                             visibilities['data'][0].v,
                             visibilities['data'][0].freq,
                             visibilities['data'][0].real.copy(),
                             visibilities['data'][0].imag.copy(),
                             visibilities['data'][0].weights)
    
    # subtract the model visibilities
    residuals.real -= m.visibilities['C18O'].real
    residuals.imag -= m.visibilities['C18O'].imag

    # clean to create an image to plot
    print('cleaning ', model)
    res_img = clean(residuals,
                    imsize=visibilities['image_npix'][0],
                    pixel_size=visibilities['image_pixelsize'][0],
                    weighting='natural',
                    convolution='expsinc',
                    mfs=False,
                    mode='spectralline',
                    maxiter=0,
                    uvtaper=None)[0]
    # save the residual
    return res_img


