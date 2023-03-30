################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import pickle as PCK        # storing/re-loading results
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

dpi = 300



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

def LoadPredictions(prediction_file, exclude = []):

    # load what was predicted
    predictions = PD.read_csv(prediction_file, sep = ';').set_index('Unnamed: 0', inplace = False)
    predictions.index.name = 'prediction_idx'

    predictions = predictions.loc[:, [ \
                                       col for col in predictions.columns[::-1] \
                                       if col not in exclude \
                                      ]]

    predictions.set_index(['cycle_idx', 'sample_nr'], inplace = True)
    # print (predictions.head())


    return predictions


def ReCalculatePredictionDifferences(predictions, data):

    n = {grp: NP.sum(data['sampling_group'].values == grp) \
         for grp in range(3) \
         }
    # n[False] = data.shape[0] - n[True]

    all_diffs = {}

    for nr, param in enumerate(predictions.columns):
        # print (nr, param)
        prediction_differences = {grp: [] for grp in n.keys()}

        ### collect data
        for idx, row in data.iterrows():
            actual = row[param]
            try:
                prediction = predictions.loc[idx, param].values
            except KeyError as ke:
                # pass
                # print (ke)
                continue

            difference = prediction - actual
            prediction_differences[row['sampling_group']].append(difference)


        all_diffs[param] = {grp: NP.concatenate(prediction_differences[grp]) \
                            for grp in n.keys()}

    return all_diffs


##############################################################################
### Visualization                                                          ###
##############################################################################

def PlotPredictions(data, all_diffs):

    PT.PreparePlot()

    cm = 1/2.54
    fig = PLT.figure(figsize = (16*cm, 10*cm), dpi = dpi)
    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.98 \
                            , bottom = 0.12 \
                            , left   = 0.06 \
                            , wspace = 0.28 # column spacing \
                            , hspace = 0.40 # row spacing \
                            )


    n_cols = len(all_diffs)


    labels = { \
                 'age': '\(\Delta\) age \n (h)' \
               , 'morpho1': '\(\Delta\) size PC1 \n (arb. units)' \
               , 'weight': '\(\Delta\) mass \n (kg)' \
               # 'age_log': '\(\Delta\) log(age) \n \ ' \
              }


    for nr, param in enumerate(all_diffs.keys()):
        ### plot
        ax = fig.add_subplot(n_cols, 1, nr+1)

        # color per birth weight group
        colors = { \
                     0: ['NBW training', (0.4, 0.4, 0.4)] \
                   , 1: ['NBW validation', (0.3, 0.4, 0.7)] \
                   , 2: ['LBW test group', (0.9, 0.5, 0.3)] \
                  }
        for grp, y in all_diffs[param].items():
            # print (nr, param, grp, len(y))
            ax.hist(  y \
                    , bins = 100 \
                    , color = colors[grp][1] \
                    , density = True \
                    , histtype = 'stepfilled' \
                    , edgecolor = 'k' \
                    , alpha = 0.67 \
                    , ls = '-' \
                    , lw = 0.5 \
                    , label = colors[grp][0] # (\(n={n[sg]}\))' \
                    )

        ax.set_ylabel(labels.get(param, param).replace('_', ' '))
        ax.axvline(0., color = 'k', ls = '-', lw = 1., alpha = 0.5)
        lim = NP.max(NP.abs(ax.get_xlim()))*0.6
        ax.set_xlim(-lim, +lim)
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top']].set_visible(False)


    ax.set_xlabel(r'underestimation \(\leftarrow \quad \mid \quad \rightarrow\) \  overestimation')
    ax.legend(loc = 1, fontsize = 8)

    # fig.suptitle('Histograms of Actual Stride Observations')
    # PLT.show()
    return fig, all_diffs


def StoreOrReload(data, force_reloading, model, n_samples):
    storage_file = f'predictions/{model}_predictions_{n_samples}_diffs.pck'
    if OS.path.exists(storage_file) and not force_reloading:

        print ('loading pre-stored prediction diffs....')
        with open(storage_file, 'rb') as storage:
            all_diffs = PCK.load(storage)
        print (list(all_diffs.keys()))
        print (list(all_diffs['age'].keys()))
        print (all_diffs['age'][0])

    else:
        print ('re-loading predictions')
        predictions = LoadPredictions(f'predictions/{model}_predictions_{n_samples}.csv', exclude = ['weight_cac', 'age_log'])
        #print (predictions.sample(5).T)

        print ('calculating differences...')
        all_diffs = ReCalculatePredictionDifferences(predictions, data)

        print ('storing data...')
        with open(storage_file, 'wb') as storage:
            PCK.dump(all_diffs, storage)

    return all_diffs

##############################################################################
### Mission Control                                                        ###
##############################################################################
if __name__ == "__main__":

    print ('THIS IS JUST A PREVIEW! \n to update the figures, execute the code in the org file for "results".')
    force_reloading = False
    # model_label = 'reference'
    model = 'age'
    n_samples = 16384 # 1024

    data = LoadYoungData()
    all_diffs = StoreOrReload(data, force_reloading, model, n_samples)
    fig, _ = PlotPredictions(data, all_diffs)

    # fig.savefig(f'figures{OS.sep}histograms_prediction_comparison.png', dpi = dpi, transparent = False)
    ## this one is stored from the org file instead!
    PLT.show()

