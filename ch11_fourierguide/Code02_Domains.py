#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20220714



################################################################################
### Libraries                                                                ###
################################################################################
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import re as RE             # regular expressions, to extract patterns from text strings
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.signal as SIG  # signal processing (e.g. smoothing)
import scipy.interpolate as INTP # interpolation

import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API

# load self-made toolboxes
SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import ConfigToolbox as CONF # project configuration
import IOToolbox as IOT     # common IO functions
import FourierToolbox as FT

#_______________________________________________________________________________
# helper variables
xyz = ['x', 'y', 'z'] # 3d coordinate shorthand
xy = ['x', 'y'] # 2d coordinate shorthand

################################################################################
### Hips in Domains                                                          ###
################################################################################


def FrequencyDomainPlot(ax, fsd, **scatter_kwargs):
    ax.plot( fsd[1:, 're'] \
             , fsd[1:, 'im'] \
             , color = scatter_kwargs.get('edgecolor', 'k')
             , **{kw: val \
                  for kw, val \
                  in scatter_kwargs.items() \
                  if kw not in ['s', 'marker', 'facecolor', 'edgecolor'] \
                  } \
              )
    ax.scatter( fsd[1:, 're'] \
              , fsd[1:, 'im'] \
             , **{kw: val \
                  for kw, val \
                  in scatter_kwargs.items() \
                  if kw not in ['color', 'ls', 'lw'] \
                  } \
              )


dpi = 300
def MakeSignalFigure(show_coef0 = False, figure_kwargs = None):
    if figure_kwargs is None:
        fig = PLT.figure(figsize = (18/2.54, 18/2.54), dpi=dpi)
    else:
        fig = PLT.figure(**figure_kwargs)
    fig.subplots_adjust( \
                              top    = 0.92 \
                            , right  = 0.90 \
                            , bottom = 0.02 \
                            , left   = 0.12 \
                            , wspace = 0.02 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )
    rows = [10,1,2,1,6]
    cols = [4,2]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )
    time_domain = fig.add_subplot(gs[:-2, 0]) # , aspect = 1/4
    time_domain.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    time_domain.set_xlabel(r'stride cycle')
    time_domain.set_ylabel(r'angle (rad)')
    time_domain.set_title('time domain', weight = 'bold')
    time_domain.set_xlim([0.,1.])


    if show_coef0:
        frequency_domain = fig.add_subplot(gs[0,1], aspect = 'equal')
    else:
        frequency_domain = fig.add_subplot(gs[:-2,1], aspect = 'equal')
    frequency_domain.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    frequency_domain.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    frequency_domain.set_xlabel(r'$\Re(c_n)$')
    frequency_domain.set_ylabel(r'$\Im(c_n)$')
    frequency_domain.set_title('frequency domain', weight = 'bold')

    frequency_domain.yaxis.tick_right()
    frequency_domain.yaxis.set_label_position("right")

    td_image = fig.add_subplot(gs[-1,0], aspect = 'equal')
    fd_image = fig.add_subplot(gs[-1,1], aspect = 'equal', sharex = td_image, sharey = td_image)

    if not show_coef0:
        return fig, time_domain, frequency_domain, td_image, fd_image

    coef0_bar = fig.add_subplot(gs[2,1])
    coef0_bar.spines[:].set_visible(False)
    coef0_bar.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    xy = (0.0, 0.7)
    PLT.annotate(r'visual', xy = xy, xytext = xy \
                 , xycoords = 'figure fraction' \
                 , ha = 'left', va = 'center', rotation = 90 \
                 , weight = 'bold' \
                 )

    xy = (0.0, 0.15)
    PLT.annotate(r'numeric', xy = xy, xytext = xy \
                 , xycoords = 'figure fraction' \
                 , ha = 'left', va = 'center', rotation = 90 \
                 , weight = 'bold' \
                 )


    return fig, time_domain, frequency_domain, coef0_bar, td_image, fd_image


def PlotFullImage(ax, imfi):
    img = PLT.imread(imfi)
    ax.imshow(img, origin = 'upper')
    ax.spines[:].set_visible(False)

    for coord in xy:
        eval(f'ax.get_{coord}axis().set_visible(False)')


def HipsInDomains(limbs):
    hips = [lmb['ihip'] for lmb in limbs.values()]
    hip_fsd = FT.ProcrustesAverage(hips, n_iterations = 3)
    x_reco = NP.linspace(0.,1.,101, endpoint = True)
    y_reco = hip_fsd.Reconstruct(x_reco, period = 1.)

    fig_kw = dict(figsize = (18/2.54, 12/2.54), dpi = dpi)
    fig, time_domain, frequency_domain, coef0_bar, td_image, fd_image \
        = MakeSignalFigure( show_coef0 = True \
                       , figure_kwargs = fig_kw \
                        )

    time_domain.plot(x_reco, y_reco \
                      , ls = '-' \
                      , lw = 1. \
                      , alpha = 1. \
                      , color = 'k' \
                      , zorder = 20 \
                      )

    PD.DataFrame.from_dict({'time (s)': x_reco, 'hip angle (rad)': y_reco}).to_csv('figures/data_timedomain.csv')

    for hip in hips:
        time_domain.plot( x_reco \
                        , hip.Reconstruct(x_reco, period = 1.) \
                        , ls = '-' \
                        , lw = 0.5 \
                        , alpha = 0.2 \
                        , color = 'k' \
                        , zorder = 10 \
                      )
    time_domain.set_ylim([0.6, 2.6])
    time_domain.axhline(hip_fsd[0, 're'], color = '0.2', ls = '--', lw = 0.5, alpha = 0.8)
    FrequencyDomainPlot( frequency_domain \
                        , hip_fsd \
                        , s = 50 \
                        , marker = 'o' \
                        , facecolor = 'w' \
                        , edgecolor = 'k' \
                        , alpha = 0.6 \
                        , lw = 1.0 \
                        , ls = '-' \
                        , zorder = 20 \
                        )

    hip_fsd._c.to_csv('figures/data_frequencydomain.csv')

    for hip in hips:
        frequency_domain.scatter(  hip[1:, 're'] \
                                 , hip[1:, 'im'] \
                                 , s = 30 \
                                 , marker = 'x' \
                                 , color = 'k' \
                                 , alpha = 0.2
                                 , zorder = 10 \
                                 )
    FT.EqualLimits(frequency_domain)
    c0s = [hip[0, 're'] for hip in hips]
    xerr = NP.std(c0s)
    lim = NP.abs(hip_fsd[0, 're']) + xerr

    coef0_bar.barh(0.5, hip_fsd[0, 're'] \
                   , height = 0.75 \
                   , facecolor = 'none' \
                   , edgecolor = 'k' \
                   , xerr = xerr \
                   , zorder = 0 \
                   )

    NP.random.seed(42)
    offset = NP.random.uniform(0.,0.6,size = len(c0s)) - 0.3 + 0.5
    coef0_bar.scatter(c0s, offset, s = 2, marker = 'o', facecolor = 'w', edgecolor = 'k', alpha = 0.7, zorder = 10)

    coef0_bar.text(lim*1.2, 0.5, f"${hip_fsd[0, 're']:.1f}\pm{xerr:.1f}$", color = 'k', ha = 'left', va = 'center')
    coef0_bar.set_xlim([-1.4*lim, +1.4*lim])
    coef0_bar.text(0, 0., r"$c_0$", color = 'k', ha = 'center', va = 'top')
    coef0_bar.set_ylim([0., 1.])
    coef0_bar.get_xaxis().set_visible(False)
    coef0_bar.get_yaxis().set_visible(False)


    PlotFullImage(fd_image, './archive/f2_frequency_spreadsheet.png')
    PlotFullImage(td_image, './archive/f2_time_spreadsheet.png')


    fig.savefig('figures/f2_domains.pdf', transparent = False, dpi = dpi)
    PLT.show()



################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    # load default config
    config = IOT.LoadDefaultConfig()

    # load joint fsd, combined as "limbs"
    limbs = IOT.ReLoadLimbs(config)

    # plot hip angles in the domains
    HipsInDomains(limbs)
