#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20220302

"""
compare normal- and low birth weight observations
for all joints
(figure for publication)

NOTE on averaging:
There are multiple ways to show averages. Here some considerations.
Data comes in aligned wrt forelimb (FL) angle, so take the hindlimb as an example.
Three options to average joint angle profiles (JAP):
   (i) take JAP as is (FL-aligned)
       -> produces too flat average JAPs because peaks are not aligned.
  (ii) align wrt whole hindlimb (withers-croup-hindhoof angle)
       -> peak retention, good alignment (because the hindlimb JAP is reproducible and distinct)
 (iii) align wrt each joint
       -> phase calculation (i.e. temporal alignment) is not reliable on some rather flat JAPs
          so alignment is erratic resulting in flat peaks for some joints

Option (ii) was chosen after testing all of them.



"""

################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import pandas as PD         # data management
import numpy as NP          # numerical analysis
import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API



SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import ConfigToolbox as CONF # project configuration
import IOToolbox as IOT     # common IO functions
import FourierToolbox as FT # Fourier Series toolbox
import PlotToolbox as PT    # plot cosmetics


def GetFCASMean(recos: PD.DataFrame) -> NP.array:
    # calculate the joint profile average
    # after temporal superimposition
    # (i.e. frequency domain average)

    # return NP.nanmean(recos, axis = 0)

    # print (recos)
    # print (recos.shape)

    signals = []
    for nr in range(recos.shape[0]):
        trace = recos[nr, :]
        fsd = FT.FourierSignal.FromSignal(time = NP.linspace(0., 1., 101, endpoint = True) \
                                       , signal = trace \
                                       , order = 8 \
                                       , label = str(nr) \
                                       )
        signals.append(fsd)

    mean = FT.ProcrustesAverage(signals \
                                , n_iterations = 3 \
                                , skip_scale = True \
                                , post_align = False \
                                )
    # print (mean)
    return mean.Reconstruct(x_reco = NP.linspace(0., 1., 101, endpoint = True))



def Alignment(  some_limbs \
              , reference_joint \
              , n_iterations = 5 \
              , superimposition_kwargs = dict(skip_shift = False \
                                              , skip_scale = False \
                                              , skip_rotation = False \
                                              ) \
              ):
    # GPA-like alignment of joint angle profiles

    # first, get the average of the reference joint...
    print (f'averaging {reference_joint} angle profile...', ' '*16, end = '\r', flush = True)
    average = FT.ProcrustesAverage( \
                            [lmb[reference_joint] for lmb in some_limbs.values() \
                             if reference_joint in lmb.keys()] \
                            , n_iterations = n_iterations, skip_scale = True, post_align = False \
                            )
    print ('reference averaging done.', ' '*32)

    # ... then align all the limbs to it
    skipped = []
    for label, lmb in some_limbs.items():
        # # standardize the shift and amplitude of the focal joint to give relative values
        # lmb.PrepareRelativeAlignment(focal_joint, skip_rotate = True, skip_center = skip_precenter, skip_normalize = skip_prenorm)
        print (f'aligning {label} ...', ' '*16, end = '\r', flush = True)

        if reference_joint not in lmb.keys():
            skipped.append(label)
            continue

        # align all joints, based on the reference
        lmb.AlignToReference(  reference_joint \
                             , average \
                             , superimposition_kwargs = superimposition_kwargs \
                             )

    print ('skipped', skipped, ' '*32)
    print (f'FCAS on {reference_joint} done!', ' '*16)

    return average





################################################################################
### Plotting                                                                 ###
################################################################################
def PlotLimbs(joint_selection, limbs, n_limit = 100):

    PT.PreparePlot()
    dpi = 300
    cm = 1/2.54
    fig = PLT.figure(dpi = dpi, figsize = (16*cm, 12*cm))
    fig.subplots_adjust( \
                              top    = 0.92 \
                            , right  = 0.88 \
                            , bottom = 0.09 \
                            , left   = 0.08 \
                            , wspace = 0.10 # column spacing \
                            , hspace = 0.10 # row spacing \
                            )

    # panel rows and columns
    bw_categories = [k for k in limbs.keys()]
    n_cat = len(bw_categories)
    jnt_groups = [k for k in joint_selection.keys()]
    n_jgrp = len(jnt_groups)

    # axis grid
    gs = MP.gridspec.GridSpec(n_jgrp, n_cat)

    joint_colors = { \
                    'shoulder': 'darkblue' \
                  , 'elbow': 'darkorange' \
                  , 'wrist': 'darkgreen' \
                  ,  'hip': 'darkblue' \
                  , 'knee': 'darkorange' \
                  , 'ankle': 'darkgreen' \
                    }

    axes = {} # store panels
    ref_ax = None
    for row, jgrp in enumerate(jnt_groups):
        joints = joint_selection[jgrp]
        for col, cat in enumerate(bw_categories):
            if ref_ax is None:
                ax = fig.add_subplot(gs[row, col])
                ref_ax = ax
            else:
                ax = fig.add_subplot(gs[row, col], sharex = ref_ax, sharey = ref_ax)
            axes[(cat, jgrp)] = ax

            PT.PolishAx(ax)

            if row == 0:
                ax.set_title(cat)
                ax.get_xaxis().set_visible(False)
            if col == 0:
                ax.set_ylabel(jgrp)
            else:
                ax.get_yaxis().set_visible(False)

            for xgrid in NP.linspace(0.,1.,5,endpoint=True):
                ax.axvline(xgrid, color = '0.8', lw = 0.5, ls = '-', alpha = 0.5, zorder = -5)
            for ygrid in NP.linspace(-1.,1.,5,endpoint=True):
                ax.axhline(ygrid*NP.pi, color = '0.8', lw = 0.5, ls = '-', alpha = 0.5, zorder = -5)

    means = {} # store means
    means2 = {} # store means
    for row, jgrp in enumerate(jnt_groups):
        joints = joint_selection[jgrp]

        ### superimpose based on whole limb
        superimposition_settings = dict( skip_shift = True \
                                            , skip_scale = True \
                                            , skip_rotation = False \
                                          )
        # store default alignment
        limbs_raw = { cat: {idx: lmb.Copy() for idx, lmb in limbs[cat].items()}\
                      for cat in bw_categories }
        if not (jgrp == 'forelimb'):
            Alignment( \
                {**limbs['LBW'], **limbs['NBW']} \
              , reference_joint = jgrp \
              , n_iterations = 3 \
              , superimposition_kwargs = superimposition_settings \
              )


        # then plot joints
        for col, cat in enumerate(bw_categories):
            for nr, jnt in enumerate(joints):

                counter = 0
                recos = []
                recos3 = []
                ax = axes[(cat, jgrp)]
                for idx, lmb in limbs[cat].items():
                    # plot the unaligned limb
                    lmb_unaligned = limbs_raw[cat][idx]
                    recos3.append(\
                        lmb_unaligned.Plot(ax \
                            , subset_joints = [jnt] \
                            , color = joint_colors[jnt] \
                            , ls = '-' \
                            , lw = 0.5 \
                            , alpha = 1/NP.sqrt(len(limbs[cat])) \
                            , label = None \
                            , zorder = 10 \
                            )[jnt])

                    # ... but store the aligned one
                    y = lmb[jnt].Reconstruct(x_reco = lmb.time)
                    recos.append(y)

                    counter += 1
                    if counter > n_limit:
                        break

                time = lmb.time # store one time (arb. the last limb)

                # store mean joint profile
                recos = NP.stack(recos, axis = 0)
                means[(cat, jgrp, jnt)] = [time, NP.nanmean(recos, axis = 0)]
                recos3 = NP.stack(recos3, axis = 0)
                means2[(cat, jgrp, jnt)] = [time, NP.nanmean(recos3, axis = 0)]
                # print ('averaging', cat, jgrp, jnt)
                # get FCAS means per joint; briefly compare plain mean (dotted?)
                # NOT NECESSARY with total limb alignment
                # means2[(cat, jgrp, jnt)] = [time, GetFCASMean(recos)]


    for row, jgrp in enumerate(jnt_groups):
        joints = joint_selection[jgrp]
        for col, cat in enumerate(bw_categories):
            for nr, jnt in enumerate(joints):
                # plot means

                ax = axes[(cat, jgrp)]
                if False:
                    time, mean = means2[(cat, jgrp, jnt)]

                    ax.plot(  time\
                            , mean \
                            , ls = ':' \
                            , lw = 1 \
                            , color = '0.0'#joint_colors[jnt] \
                            , alpha = 1.0 \
                            , zorder = 30 \
                            )

                time, mean = means[(cat, jgrp, jnt)]
                ax.plot(  time\
                        , mean \
                        , ls = '-' \
                        , lw = 1 \
                        , color = '0.0'#joint_colors[jnt] \
                        , alpha = 1.0 \
                        , zorder = 30 \
                        )


                for cat2 in [c for c in bw_categories if not (c == cat)]:
                    # print (cat, jgrp, jnt, cat2)
                    ax2 = axes[(cat2, jgrp)]
                    ax2.plot(time \
                        , mean \
                        , ls = '--' \
                        , lw = 0.5 \
                        , color = '0.0'# joint_colors[jnt] \
                        , alpha = 1.0 \
                        , zorder = 50 \
                        )


        ref_ax.set_xlim([0., 1.])
        ref_ax.set_ylim([-NP.pi, NP.pi])
        ref_ax.set_xticks([0., 0.5, 1.])
        ref_ax.set_yticks([-NP.pi, 0., NP.pi])
        ref_ax.set_yticklabels([r'\(-\pi\)', r'\(0\)', r'\(+\pi\)'])

        cat = 'LBW'
        translation = {'knee': 'stifle', 'ankle': 'tarsal', 'wrist': 'carpal'}
        Translate = lambda jnt: translation.get(jnt, jnt)
        for nr, jnt in enumerate(joints):
            mean = NP.nanmean(means[(cat, jgrp, jnt)][1])

            ax.text(1.05, mean, Translate(jnt), color = joint_colors[jnt], ha = 'left', va = 'center')


    fig.text(0.5, 0.01, 'cycle progress', ha = 'center', va = 'bottom')
    fig.savefig(f'figures/raw_profile_comparison.png', dpi = dpi, transparent = False)
    fig.savefig(f'figures/raw_profile_comparison.pdf', dpi = dpi, transparent = False)
    PLT.close()
    return fig


################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    # load default config
    config = IOT.LoadDefaultConfig()

    analysis_data = IOT.LoadAnalysisData(config)
    analysis_data = analysis_data.loc[analysis_data['age'].values < 25, :]
    print (analysis_data.shape)



    # load joint fsd, combined as "limbs"
    # limbs_spaghetti = IOT.LoadLimbs(config)
    limbs_aligned = IOT.ReLoadLimbs(config, subset = analysis_data.index.values)

    lbw_indices = analysis_data.loc[analysis_data['is_lbw'].values, :].index.values
    nbw_indices = analysis_data.loc[NP.logical_not(analysis_data['is_lbw'].values), :].index.values

    limbs_lbw = {idx: lmb for idx, lmb in limbs_aligned.items() \
                 if idx in lbw_indices}
    limbs_nbw = {idx: lmb for idx, lmb in limbs_aligned.items() \
                 if idx in nbw_indices}
    # print(limbs_lbw)


    joints = {  'forelimb': [ \
                   'shoulder'
                 , 'elbow' \
                 , 'wrist' \
                 # , 'metacarpal' \
              ] , 'hindlimb': [\
                    'hip' \
                  , 'knee' \
                  , 'ankle' \
                 # , 'metatarsal' \
                 # , 'torso' \
                 # , 'head' \
             ] }
    fig = PlotLimbs(  \
                joint_selection = joints \
              , limbs = {'NBW': limbs_nbw, 'LBW': limbs_lbw} \
              , n_limit = 10000 \
              )
