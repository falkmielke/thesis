################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import time as TI           # have a break...
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.stats as STATS # for kernel density estimate (distribution plots)

import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API
import seaborn as SB        # even higher level plotting API
# MP.rcParams['text.usetex'] = True


# stats packages
import pymc as PM

# load self-made toolboxes
SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import ConfigToolbox as CONF # project configuration
import IOToolbox as IOT      # common IO functions
import ModelToolbox as MT    # probabilistic modeling
import EigenToolbox as ET    # stride PCA
import PlotToolbox as PT     # plotting helpers

PT.the_font = {  \
        'family': 'sans-serif'
        , 'sans-serif': 'DejaVu Sans'
        , 'size': 10.0 \
    }




##############################################################################
### Data Assembly                                                          ###
##############################################################################
def LoadYoungData():
    # load the data for only a subset of piglets

    # load data
    data = PD.read_csv(f'models{OS.sep}05_age{OS.sep}data.csv', sep = ";").set_index('cycle_idx', inplace = False)

    # only young were in
    data = data.loc[data['age'].values < 25, :]

    # data = data.loc[NP.logical_not(data['is_lbw'].values), :]
    # data['intercept'] = 1.

    data['sampling_group'] = data.loc[:, 'is_validation'].values.astype(int)
    data.loc[data['is_lbw'].values, 'sampling_group'] = 2

    return data


##############################################################################
### Visualization                                                          ###
##############################################################################

def PlotObservations(data, columns):

    PT.PreparePlot()

    dpi = 300
    cm = 1/2.54
    fig = PLT.figure(figsize = (16*cm, 8*cm), dpi = dpi)

    n_cols = len(columns)

    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.98 \
                            , bottom = 0.08 \
                            , left   = 0.06 \
                            , wspace = 0.10 # column spacing \
                            , hspace = 0.40 # row spacing \
                            )

    n = {True: NP.sum(data['is_lbw']) \
         }
    n[False] = data.shape[0] - n[True]

    labels = { \
               'age_log': 'log(age) \n \ ' \
               , 'age': 'age \n (h)' \
               , 'morpho1': 'size PC1 \n (arb. units)' \
               , 'weight': 'mass \n (kg)' \
              }
    n_lbw = NP.nansum(data['is_lbw'].values)
    n_val = NP.nansum(data['is_validation'].values)
    n_nbw = data.shape[0] - NP.sum([n_lbw, n_val])

    for nr, param in enumerate(columns):
        # print (nr, param)


        ### plot
        ax = fig.add_subplot(n_cols, 1, nr+1)

        # color per birth weight group
        colors = { \
                     0: [f'NBW training ({n_nbw})', (0.4, 0.4, 0.4)] \
                   , 1: [f'NBW validation ({n_val})', (0.3, 0.4, 0.7)] \
                   , 2: [f'LBW test group ({n_lbw})', (0.9, 0.5, 0.3)] \
                  }
        for sg, (label, color) in colors.items():
            y = data.loc[data['sampling_group'].values == sg, param].values
            ax.hist(  y \
                    # , bins = 16 \
                    , color = color \
                    , density = True \
                    , histtype = 'stepfilled' \
                    , edgecolor = 'k' \
                    , alpha = 0.8 \
                    , ls = '-' \
                    , lw = 0.5 \
                    , label = f'{label}'# (\(n={len(y)}\))' \
                    )

        ax.set_ylabel(labels.get(param, param).replace('_', ' '))
        # ax.axvline(0., color = 'k', ls = '-', lw = 1., alpha = 0.5)
        # lim = NP.max(NP.abs(ax.get_xlim()))
        # ax.set_xlim(-lim, +lim)
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top']].set_visible(False)

        if nr == 0:
            ax.legend(loc = 1, fontsize = 8, ncol = 1)

    # ax.set_xlabel(r'underestimation \(\leftarrow \quad \mid \quad \rightarrow\) overestimation\quad\ ')

    # fig.suptitle('Histograms of Actual Stride Observations')
    # PLT.show()

    fig.savefig(f'figures{OS.sep}histograms_observed.png', dpi = 300, transparent = False)
    fig.savefig(f'figures{OS.sep}histograms_observed.pdf', dpi = 300, transparent = False)

    return fig

##############################################################################
### Mission Control                                                        ###
##############################################################################
if __name__ == "__main__":

    data = LoadYoungData()

    fig = PlotObservations(data, [col for col in ['weight', 'morpho1', 'age'] \
                             if col not in ['age_log'] \
                             ])
    PLT.show()
    #print (predictions.sample(5).T)
