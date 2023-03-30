#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20211011

"""
plot predicted joint angle profiles

this is a pruned clone of master procedure 23
"""


################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.stats as STATS # for kernel density estimate (distribution plots)
from tqdm import tqdm       # loop progress

import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API
import seaborn as SB        # even higher level plotting API

# load self-made toolboxes
SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import ConfigToolbox as CONF # project configuration
import IOToolbox as IOT      # common IO functions
import EigenToolbox as ET    # stride PCA
import FourierToolbox as FT # Fourier Series toolbox
import PlotToolbox as PT     # plotting helpers





################################################################################
### I/O                                                                      ###
################################################################################
def LoadSettings(n_iterations = 1024):
    ###  settings
    settings = PD.read_csv(f'../predictions{OS.sep}prediction_settings_{n_iterations}.csv', sep = ';').set_index('setting_idx', inplace = False)

    settings.drop(columns = ['intercept_data'], inplace = True)
    settings.columns = [col.replace('_data', '') for col in settings.columns]
    settings = settings.astype(bool)

    # sex
    settings['sex'] = 'female'
    settings.loc[settings['sex_is_male'].values, 'sex'] = 'male'

    # ageclass
    settings['ageclass'] = 'adult'
    for ac in ['adolescent', 'infant']:
        settings.loc[settings[f'ageclass_is_{ac}'].values, 'ageclass'] = ac


    return settings


def LoadPredictions(n_iterations = 1024):
    # load predictions from files

    ###  settings
    settings = LoadSettings(n_iterations)

    ### Body Proportions
    bodyproportions = PD.read_csv(f'../predictions{OS.sep}bodyproportions_predictions_{n_iterations}.csv', sep = ';')

    #bodyproportions.rename(columns = {'setting': 'setting_idx'}, inplace = True)
    bodyproportions.set_index('setting', inplace = True)
    bodyproportions.index.name = 'setting_idx'

    predictions = settings.join(bodyproportions, how = 'right', on = 'setting_idx')

    predictions.reset_index(drop = False, inplace = True)
    predictions.set_index(['setting_idx', 'sample_nr'], inplace = True)


    for block in ['stride', 'posture', 'coordination']:
        pred = PD.read_csv(f'../predictions{OS.sep}{block}_predictions_{n_iterations}.csv', sep = ';')

        pred.drop(columns = [col for col in pred.columns if col.split('_')[-1] in ['data', 'group']], inplace = True)
        pred.set_index(['setting', 'sample_nr'], inplace = True)
        pred.index.names = ['setting_idx', 'sample_nr']


        if block == 'stride':

            # apply stride PCA
            filename = f'data{OS.sep}stride_parameters.pca'
            stride_pca = ET.PrincipalComponentAnalysis.Load(filename)

            transformed = stride_pca.TraFo(pred.loc[:, stride_pca._features])
            transformed.columns = [col.replace('PC', 'stride') for col in transformed.columns]
            pred = pred.join(transformed, how = 'left')

        predictions = predictions.join(pred, how = 'right')

    # print (predictions.sample(5).T)
    return predictions


def PCAReTraFo(pca, component_scores):
    # undo the nonaffine Fourier coefficient pca

    values = component_scores.loc[:, pca.transformed.columns].values

    nonaffines = PD.DataFrame( \
                               pca.ReTraFo(values, unshift = True, unscale = True) \
                               , columns = pca._features \
                               , index = component_scores.index \
                               )


    # append zero'th coeff
    joints = NP.unique([col.split('|')[0] for col in nonaffines.columns])
    for jnt in joints:
        for ri in ['re', 'im']:
            nonaffines.loc[:, f'{jnt}|0|{ri}'] = 0

    # sort columns
    nonaffines = nonaffines.reindex(sorted(nonaffines.columns), axis=1)

    # print (nonaffines.sample(5).T)
    return nonaffines




################################################################################
### FSD                                                                      ###
################################################################################

def FourierSeriesReconstruction(coeffs: PD.Series, time: NP.array = None, label = None) -> PD.DataFrame:
    # Fourier Series transformation from frequency domain to time domain

    # find the time
    if time is None:
        time = NP.round(NP.linspace(0., 1., 100, endpoint = False), 4)

    # re-order index levels
    coeffs = coeffs.reorder_levels(['joint', 'affine', 'coeff', 're_im'])
    # print (coeffs)

    # separate joints
    joints = coeffs.index.levels[0].values
    # print (joints)

    # sort (lexsort issue) # obsolete/done before
    # coeffs.sort_index(inplace = True)

    # reconstruct joint-wise
    joint_angles = PD.DataFrame(index = time, columns = joints)

    fsds = {}
    for jnt in joints:
        # Fourier Series
        fsd = FT.FourierSignal.FromComponents(coeffs.loc[jnt], label = label)
        fsds[jnt] = fsd

        # Reconstruction
        joint_angles.loc[:, jnt] = fsd.Reconstruct( time )

    return joint_angles, fsds



def FSDReconstruction(predictions, nonaffine_df):

    affines = []
    nonaffines = []
    for joint in ['ihip', 'iknee', 'iankle']:
        # affine
        predictions[f'φ_{joint}'] = 0.
        affine = predictions.loc[:, [f'{comp}_{joint}' for comp in ['α', 'μ', 'φ']]].copy()
        affine.columns = [f'a|{joint}|{col}|re' for col in ['α', 'μ', 'φ']]

        affines.append(affine)


    affines = PD.concat(affines, axis = 1)
    # print (affines)

    nonaffines = nonaffine_df.copy()
    nonaffines.columns = [f'n|{col}' for col in nonaffines.columns]
    # print (nonaffines)

    coefficients = affines.join(nonaffines, how = 'left')
    coefficients.drop(columns = [col for col in coefficients.columns if ('ishoulder' in col)], inplace = True)
    coefficients.columns = PD.MultiIndex.from_tuples([col.split('|') for col in coefficients.columns])
    coefficients.columns.names = [ 'affine', 'joint', 'coeff', 're_im']
    # print (coefficients)

    reco_angles = []
    reco_fsds = {}
    for idx, coeffs in tqdm(coefficients.iterrows()):
        angles, fsds = FourierSeriesReconstruction(coeffs, label = '{}_{}'.format(*idx))
        reco_fsds[idx] = fsds
        angles.columns = PD.MultiIndex.from_tuples([(idx[0], col, idx[1]) for col in angles.columns])
        reco_angles.append( angles )

    reco_angles = PD.concat(reco_angles, axis = 1).T
    reco_angles.index.names = ['setting_idx', 'joint', 'sample_nr']
    return reco_angles, reco_fsds

##############################################################################
### prediction visualization                                               ###
##############################################################################


def PlotPredictions(n_iterations, subsample = None):

    ### Load All Info
    # load default config
    config = IOT.LoadDefaultConfig()

    # load analysis data
    data = IOT.LoadAnalysisData(config)

    # load joint fsd, combined as "limbs"
    limbs = IOT.ReLoadLimbs(config)

    settings = LoadSettings(n_iterations)
    print (settings)

    predictions = PD.read_csv(f'../predictions{OS.sep}all_predictions_{n_iterations}.csv', sep = ';')

    predicted_traces = PD.read_csv(f'../predictions{OS.sep}joint_profiles_{n_iterations}.csv', sep = ';')
    predicted_traces.set_index(['setting_idx', 'joint', 'sample_nr'], inplace = True)

    ### Plot

    params = { \
                 'sex': ['female', 'male'] \
               , 'ageclass': ['infant', 'adolescent', 'adult'] \
              }
    n = {}
    for pred in ['sex', 'ageclass']:
        # params[pred] = list(NP.unique(settings[pred].values))
        n[pred] = len(params[pred])

    # figure preparation
    dpi = 300
    fig, gridspec = PT.MakeFigure(  rows = [1]*n['sex'] \
                                  , cols = [1]*n['ageclass'] \
                                  , dimensions = [18, 12] \
                                  , dpi = dpi \
                                  )
    fig.subplots_adjust( \
                              top    = 0.92 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.05 # column spacing \
                            , hspace = 0.05 # row spacing \
                         )

    colors = { \
                 'ihip': (0.4, 0.4, 0.8) \
               , 'iknee': (0.4, 0.8, 0.4) \
               , 'iankle': (0.8, 0.4, 0.4) \
              }
    Darken = lambda color: (color[0]*0.6, color[1]*0.6, color[2]*0.6)

    legend = []
    ref_ax = None
    for setting_idx, setting in settings.iterrows():

        sx = params['sex'].index(setting['sex'])
        ac = params['ageclass'].index(setting['ageclass'])

        if ref_ax is None:
            ax = fig.add_subplot(gridspec[sx, ac])
            ref_ax = ax
        else:
            ax = fig.add_subplot(gridspec[sx, ac], sharex = ref_ax, sharey = ref_ax)


        ### predictions
        means = {}
        traces = predicted_traces.loc[setting_idx, :]
        for joint in ['ihip', 'iknee', 'iankle']:
            joint_traces = traces.loc[joint, :]
            if subsample is not None:
                joint_traces = joint_traces.iloc[NP.random.choice(range(joint_traces.shape[0]), subsample), :]

            joint_traces = joint_traces.T

            if joint in ['ihip', 'iankle']:
                jnt_means = NP.mean(joint_traces, axis = 0).values.reshape((1, -1))

                print (joint_traces.shape, jnt_means.shape)

                joint_traces.loc[:, :] = ((joint_traces.values - jnt_means) * (-1) + jnt_means)

            means[joint] = NP.mean(joint_traces.values, axis = (0, 1) )

        for joint, offset in [['ihip', 2.], ['iknee', 0.], ['iankle', -2.]]:
            joint_traces = traces.loc[joint, :]
            if subsample is not None:
                joint_traces = joint_traces.iloc[NP.random.choice(range(joint_traces.shape[0]), subsample), :]
            time = NP.array(joint_traces.columns, dtype = float)
            joint_traces = joint_traces.T
            joint_traces.loc[:, :] -= means[joint]
            # print (joint_traces.shape)

            if joint in ['ihip', 'iankle']:
                joint_traces *= -1


            joint_traces.loc[:, :] -= NP.mean(joint_traces, axis = 0)

            ax.plot(time, offset + joint_traces.values, color = colors[joint], lw = 0.5, ls = '-', alpha = 0.05)


        ### real data
        data_selection = NP.ones((data.shape[0]))
        for pred in ['sex', 'ageclass']:
            data_selection = NP.logical_and(  data_selection \
                                            , data[pred].values == setting[pred])

        cycles = data.loc[data_selection, :].index.values
        get_legend = True
        for cyc in cycles:
            limb = limbs[cyc]
            time = limb.time

            for joint, offset in [['ihip', 2.], ['iknee', 0.], ['iankle', -2.]]:
                joint_fsd = limb[joint]
                trace = joint_fsd.Reconstruct(x_reco = time)
                if joint in ['iknee']:
                    trace += means[joint]
                else:
                    trace -= means[joint]

                trace -= NP.mean(trace)
                ax.plot(time, offset + trace \
                        , color = '0.2' #Darken(colors[joint]) \
                        , lw = 1., ls = '-', alpha = 0.5 \
                        , label = joint[1:] if get_legend else None \
                        )

            get_legend = False


        if ac == 0:
            ax.set_ylabel(f"{setting['sex']}s\n angle (rad)")
            ax.set_yticks([-2., 0., 2.])
            ax.set_yticklabels(['hip', 'knee', 'ankle'], rotation = 60)
        else:
            ax.get_yaxis().set_visible(False)

        if sx == 0:
            ax.set_title(setting['ageclass'])
            ax.get_xaxis().set_visible(False)
        else:
            if ac == 1:
                ax.set_xlabel('cycle progress')

        if ac == 0 and sx == 1:
            ax.plot([0.25, 0.25], [-3.8, -3.8+NP.pi/3] \
                    , lw = 2.5, color = 'k')

            ax.text( \
                     0.26, -3.8 \
                     , r'$\frac{\pi}{3}$' \
                     , ha = 'left', va = 'bottom' \
                     )
        # if ac == 0 and sx == 0:
        #     ax.legend(loc = 'best', fontsize = 6, ncol = 3)

        # ax.set_ylabel(param_labels.get(param, param))
        # if param == parameters[0]:
        #     ax.legend(loc = 2, ncol = 2)
        # if not(param == parameters[-1]):
        #     ax.get_xaxis().set_visible(False)
        # else:
        #     ax.set_xticks(list(x_ageclass.values()))
        #     ax.set_xticklabels(list(x_ageclass.keys()))
        #     ax.set_xlabel('ageclass')

    ref_ax.set_xlim([0., 0.99])
    ref_ax.set_ylim([-4, 4])

    fig.savefig(f'figures/f7_trace_predictions.png', dpi = dpi, transparent = False)
    PLT.close()






##############################################################################
### Mission Control                                                        ###
##############################################################################
if __name__ == "__main__":
    n_iterations = 16384

    # TODO: redo prediction collection on new sampling

    PlotPredictions(n_iterations, subsample = None)
