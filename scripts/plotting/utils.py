"""Utility funtions to be used in both fitting and plotting. 

Contains function for loss, chi^2 calculation, standardization, 
global mean and stddev calculation, and copying plots to cernbox. 

Authors : Miles Cochran-Branson, Quentin Buat
Date : Summer 2022
"""

import numpy as np

# Functions for response on resolution curves 

def make_bins(bmin, bmax, nbins):
    """Make tuples to extract data between bins. """

    returnBins = []
    stepsize = (bmax - bmin) / nbins
    for i in range(nbins):
        returnBins.append((bmin + i*stepsize, bmin + (i+1)*stepsize))
    return returnBins

def get_quantile_width(arr, cl=0.68):
    """Get width of `arr` at `cl`%. Default is 68% CL"""

    q1 = (1. - cl) / 2.
    q2 = 1. - q1
    y = np.quantile(arr, [q1, q2])
    width = (y[1] - y[0]) / 2.
    return width

def response_curve(res, var, bins, cl=0.68):
    """Prepare data fot plotting the response and resolution curve"""

    _bin_centers = []
    _bin_errors = []
    _means = []
    _mean_stat_err = []
    _resol = []
    for _bin in bins:
        a = res[(var > _bin[0]) & (var < _bin[1])]
        if len(a) < 2:
            if len(a) == 0:
                print('Bin was empty! Moving on to next bin')
            else:
                print('Bin has only 1 entry, skipping to avoid std ddof error')
            continue
        _means += [np.mean(a)]
        _mean_stat_err += [np.std(a, ddof=1) / np.sqrt(np.size(a))]
        _resol += [get_quantile_width(a, cl=cl)]
        _bin_centers += [_bin[0] + (_bin[1] - _bin[0]) / 2]
        _bin_errors += [(_bin[1] - _bin[0]) / 2]
    return (np.array(_bin_centers), np.array(_bin_errors), np.array(_means), 
            np.array(_mean_stat_err), np.array(_resol))