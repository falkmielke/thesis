#!/usr/bin/env python3


__author__      = "Falk Mielke"
__date__        = 20220818



################################################################################
### Libraries                                                                ###
################################################################################
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import pathlib as PL        # modern path management
import re as RE             # regular expressions, to extract patterns from text strings
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.signal as SIG  # signal processing (e.g. smoothing)
import scipy.interpolate as INTP # interpolation

import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API
import seaborn as SB        # even more plotting

# load self-made toolboxes
SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import ConfigToolbox as CONF # project configuration
import IOToolbox as IOT     # common IO functions
import FourierToolbox as FT
import EigenToolbox as ET

#_______________________________________________________________________________
# helper variables
xyz = ['x', 'y', 'z'] # 3d coordinate shorthand
xy = ['x', 'y'] # 2d coordinate shorthand

################################################################################
### Hips in Domains                                                          ###
################################################################################


def FrequencyDomainPlot(ax, fsd, **scatter_kwargs):
    TraFo = lambda x: x
    # TraFo = lambda x: NP.sign(x)*NP.sqrt(NP.abs(x))
    ax.plot(   TraFo(fsd[1:, 're']) \
             , TraFo(fsd[1:, 'im']) \
             , color = scatter_kwargs.get('edgecolor', 'k')
             , **{kw: val \
                  for kw, val \
                  in scatter_kwargs.items() \
                  if kw not in ['s', 'marker', 'facecolor', 'edgecolor', 'color'] \
                  } \
              )
    ax.scatter( TraFo(fsd[1:, 're']) \
              , TraFo(fsd[1:, 'im']) \
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
                            , bottom = 0.12 \
                            , left   = 0.10 \
                            , wspace = 0.02 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )
    rows = [5,1]
    cols = [4,2]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )
    time_domain = fig.add_subplot(gs[:, 0]) # , aspect = 1/4
    # time_domain.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    time_domain.set_xlabel(r'stride cycle')
    time_domain.set_ylabel(r'angle (rad)')
    time_domain.set_title('time domain', weight = 'bold')
    time_domain.set_xlim([0.,1.])


    if show_coef0:
        frequency_domain = fig.add_subplot(gs[0,1], aspect = 'equal')
    else:
        frequency_domain = fig.add_subplot(gs[0,1], aspect = 'equal')
    frequency_domain.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    frequency_domain.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    frequency_domain.set_xlabel(r'$\Re(c_n)$')
    frequency_domain.set_ylabel(r'$\Im(c_n)$')
    frequency_domain.set_title('frequency domain', weight = 'bold')

    frequency_domain.yaxis.tick_right()
    frequency_domain.yaxis.set_label_position("right")


    if not show_coef0:
        return fig, time_domain, frequency_domain

    coef0_bar = fig.add_subplot(gs[2,1])
    coef0_bar.spines[:].set_visible(False)
    coef0_bar.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    return fig, time_domain, frequency_domain, coef0_bar


################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":


    ### PCA calculation per joint
    # path to data files
    cycle_directory = PL.Path('./data/jointangles')
    joints = ['ihip', 'iknee', 'ishoulder', 'iellbow', 'tail']

    reference_order = 9

    UnwrapFSD = lambda fsd: fsd.GetArray()
    WrapToFSD = lambda vec: FT.FourierSignal.FromDataFrame( \
                                                       PD.DataFrame( \
                                                           vec.reshape((-1, 2))\
                                                           , columns = FT.re_im \
                                                        )
                                                      )

    # load and store joint angle profiles
    data = {}
    for fi in cycle_directory.iterdir():
        joint_angles = PD.read_csv(fi, sep = ';').set_index('time', inplace = False)
        data[fi] = joint_angles

    # store fsd's for each joint
    pcas = {}
    for joint in joints:
        fsds = []
        for fi, joint_angles in data.items():
            angle = joint_angles.loc[:, joint]

            time = angle.index.values
            time -= time[0]
            time /= NP.max(time)
            period = time[-1] - time[0]


            # nonnan
            nonnans = NP.logical_not(NP.isnan(angle.values))
            if NP.sum(nonnans) < 32:
                continue

            # fourier series decomposition
            fsd = FT.FourierSignal.FromSignal(time = time[nonnans] \
                                              , signal = angle.values[nonnans] \
                                              , order = reference_order \
                                              , period = period \
                                              , label = joint \
                                              )

            fsds.append( UnwrapFSD(fsd) )

        if len(fsds) < 10:
            print (joint)
            continue

        # print (fsds)
        fsds = PD.DataFrame(NP.stack(fsds, axis = 0))
        fsds.columns = [f'col{i:02.0f}' for i in fsds.columns]
        fsds.dropna(inplace = True)

        pca = ET.PrincipalComponentAnalysis(fsds.iloc[:, 2:], features = fsds.columns[2:], standardize = False)
        pcas[joint] = pca







    # path to data file
    cycles_file = f"./data/jointangles/10_Elfira.csv"

    # load joint angle profiles
    joint_angles = PD.read_csv(cycles_file, sep = ';').set_index('time', inplace = False)
    # print (joint_angles)

    ## Fourier Series
    jnt = 'hip'
    selected_joint = 'i'+jnt
    angle = joint_angles.loc[:, selected_joint]
    angle -= NP.mean(angle)
    # order = 5 # looped below

    dpi = 300
    fig_kw = dict(figsize = (18/2.54, 10/2.54), dpi = dpi)
    fig, time_domain, frequency_domain \
        = MakeSignalFigure( show_coef0 = False \
                       , figure_kwargs = fig_kw \
                        )

    # time
    time = joint_angles.index.values
    time -= time[0]
    time /= NP.max(time)
    period = time[-1] - time[0]



    time_domain.plot(time, angle \
                      , ls = '-' \
                      , lw = 0.5 \
                      , alpha = 1. \
                      , color = 'k' \
                      , zorder = 20 \
                      )

    time_domain.axhline(NP.mean(angle), ls = '-', color = '0.5', lw = 1, zorder = 0)

    # nonnan
    nonnans = NP.logical_not(NP.isnan(angle.values))

    x_reco = time.copy()
    # x_reco = NP.linspace(0.,1.,101, endpoint = True)

    # orders = NP.arange(3,15, 2)
    # cycles = 0.2+0.7*(orders)/(NP.max(orders))
    # # print (PLT.cycler('color',PLT.cm.Greens([0.9]))[0])
    # time_domain.set_prop_cycle(PLT.cycler('color',PLT.cm.Oranges(cycles)))
    # frequency_domain.set_prop_cycle(PLT.cycler('color',PLT.cm.Oranges([0.9])))
    colors = {  'green': '#006428' \
              , 'orange': '#9e3303' \
              }
    Residual = lambda y, angle: NP.nansum(NP.abs(NP.subtract(y, angle.values)))/NP.sum(NP.logical_not(NP.isnan(angle)))


    ### I. Fourier Series, limited method
    order = reference_order
    # fourier series decomposition
    fsd = FT.FourierSignal.FromSignal(time = time[nonnans] \
                                      , signal = angle.values[nonnans] \
                                      , order = order \
                                      , period = period \
                                      , label = jnt \
                                      )
    # print (fsd)
    FrequencyDomainPlot( frequency_domain \
                        , fsd \
                        , s = 30 \
                        , marker = 'o' \
                        , facecolor = colors['green'] \
                        , edgecolor = colors['green'] \
                        # , alpha = 0.6 \
                        , alpha = 0.8 \
                        , lw = 1.0 \
                        , ls = '-' \
                        , zorder = order \
                        )



    # get coefficients
    coefficients = fsd.GetCoeffDataFrame()

    y_reco = fsd.Reconstruct(x_reco, period = period)

    time_domain.plot(x_reco, y_reco \
                  , ls = '-' \
                  , lw = 1. \
                  , color = colors['green'] \
                  , alpha = 1.0 \
                  , zorder = order \
                  , label = f"FSD ({order} coeff.), $\epsilon = {Residual(y_reco, angle):.3f}$" \
                  )


    ### II. PCA, limited
    # get the PCA
    pca = pcas[selected_joint].Copy()

    # reduce dimensionality
    pca_red = pca.Copy()
    n_components = NP.argmax(NP.cumsum(pca.weights)>.98)
    cumvar = NP.sum(pca.weights[:n_components])
    pca_red.ReduceDimensionality(n_components)

    # get values and transform
    vals = UnwrapFSD(fsd)[2:].reshape((1, -1))
    pc_vals = pca_red.Transform(vals)

    # re-transform
    retransformed = pca_red.ReTraFo(pc_vals, unshift = False, unscale = False)
    fsd_pcad = WrapToFSD(NP.concatenate([[0., 0.], retransformed.ravel()]))

    # plot
    y_reco = fsd_pcad.Reconstruct(x_reco, period = period)


    FrequencyDomainPlot( frequency_domain \
                        , fsd_pcad \
                        , s = 30 \
                        , marker = 'o' \
                        , facecolor = colors['orange'] \
                        , edgecolor = colors['orange'] \
                        # , alpha = 0.6 \
                        , alpha = 0.8 \
                        , lw = 1.0 \
                        , ls = '-' \
                        , zorder = order \
                        )


    time_domain.plot(x_reco, y_reco \
                  , ls = '-' \
                  , lw = 1. \
                  , color = colors['orange'] \
                  , alpha = 0.8 \
                  , zorder = 2*order \
                  , label = f"FSD + PCA ({n_components:.0f} PCs, {100*cumvar:.1f}% var.)\n$\epsilon = {Residual(y_reco, angle):.3f}$" \
                  )



    # frequency_domain.set_xscale('function', functions=(lambda x: NP.sign(x)*NP.sqrt(NP.abs(x)), lambda x: x**2))
    # frequency_domain.set_yscale('function', functions=(lambda x: NP.sign(x)*NP.sqrt(NP.abs(x)), lambda x: x**2))

    FT.EqualLimits(frequency_domain)
    time_domain.legend(fontsize = 8, loc = 'lower left', title = 'transformation reversal', bbox_to_anchor=(1.05, -0.08), fancybox = True)



    fig.savefig('figures/f5_reversibility.pdf', transparent = False, dpi = dpi)
    # PLT.show()
    PLT.close()



    ### residuals: loop all observations
    # path to data files
    cycle_directory = PL.Path('./data/jointangles')
    joints = ['ihip', 'iknee', 'ishoulder', 'iellbow', 'tail']
    observations = []
    # load and store joint angle profiles
    for fi in cycle_directory.iterdir():
        joint_angles = PD.read_csv(fi, sep = ';').set_index('time', inplace = False)
        for joint in joints:

            observations.append([joint, joint_angles.loc[:, joint]])

    # print (observations)


    Residual = lambda y, angle: NP.nansum(NP.abs(NP.subtract(y, angle.values)))/NP.sum(NP.logical_not(NP.isnan(angle)))
    residuals = {}
    counter = 0
    translate = {'iellbow': 'elbow', **{jnt: jnt[1:] for jnt in joints if (jnt[0] == 'i')}}
    for order in NP.arange(3, 25, 1):
        for joint, angle in observations:

            time = angle.index.values
            time -= time[0]
            time /= NP.max(time)
            period = time[-1] - time[0]


            # nonnan
            nonnans = NP.logical_not(NP.isnan(angle.values))
            if NP.sum(nonnans) < 32:
                continue

            # fourier series decomposition
            fsd = FT.FourierSignal.FromSignal(time = time[nonnans] \
                                              , signal = angle.values[nonnans] \
                                              , order = order \
                                              , period = period \
                                              , label = joint \
                                              )


            y_reco = fsd.Reconstruct(time, period = period)

            residuals[counter] = {'order': order, 'joint': translate.get(joint, joint), 'residual': Residual(y_reco, angle)}
            counter += 1


    residuals= PD.DataFrame.from_dict(residuals).T
    residuals.dropna(inplace = True)
    print (residuals)
    data_types = {  'order': int \
                  , 'residual': float \
                }
    residuals= residuals.astype(data_types)

    # print (NP.unique(residuals['pca']))

    # ax = SB.violinplot(x="order", y="residual", col = 'joint', split = True, color = '#99999999', data=residuals)
    g = SB.catplot(  x = "order" \
                    , y = "residual" \
                    , col = "joint" \
                    , data = residuals \
                    , kind = "violin" \
                    , width = 1.8 \
                    , split = True \
                    , color = '#99999999' \
                    , linecolor = 'k' \
                    , linewidth = 0.5 \
                    , alpha = 0.8 \
                     # , inner = None \
                     , inner = 'quartile' \
                    )
    g.set_axis_labels("", "residual")
    g.set_titles("{col_name}")
    # g.set_xticklabels(['3']+['']*6+['10']+['']*6+['17']+['']*6+['24'])
    g.set(ylim=(0, 0.05))
    g.set(xticks=[0,7,14,21])
    g.fig.set_size_inches(18/2.54, 6/2.54)

    PLT.gcf()
    PLT.tight_layout()
    xy = (0.5, 0.01)
    PLT.annotate(r'order', xy = xy, xytext = xy \
                 , xycoords = 'figure fraction' \
                 , ha = 'center', va = 'bottom', rotation = 0 \
                 )
    xy = (0.0, 1.0)
    PLT.annotate(r'A', xy = xy, xytext = xy \
                 , xycoords = 'figure fraction' \
                 , ha = 'left', va = 'top', rotation = 0 \
                 , weight = 'bold' \
                 )

    PLT.gcf().savefig('figures/f6a_residual_fsd.pdf', transparent = False, dpi = dpi)
    PLT.close()


    ### now for the PCA
    residuals = {}
    counter = 0
    for reduce_dim_to in NP.concatenate([NP.arange(3, 15, 1), [0]]):
        for joint, angle in observations:

            time = angle.index.values
            time -= time[0]
            time /= NP.max(time)
            period = time[-1] - time[0]

            # nonnan
            nonnans = NP.logical_not(NP.isnan(angle.values))
            if NP.sum(nonnans) < 32:
                continue

            # fourier series decomposition
            fsd = FT.FourierSignal.FromSignal(time = time[nonnans] \
                                              , signal = angle.values[nonnans] \
                                              , order = reference_order \
                                              , period = period \
                                              , label = joint \
                                              )

            # get the PCA
            pca = pcas[joint].Copy()

            # reduce dimensionality
            if reduce_dim_to > 0:
                pca.ReduceDimensionality(reduce_dim_to)

            # get values and transform
            vals = UnwrapFSD(fsd)[2:].reshape((1, -1))
            pc_vals = pca.Transform(vals)

            # re-transform
            retransformed = pca.ReTraFo(pc_vals, unshift = False, unscale = False)
            fsd_pcad = WrapToFSD(NP.concatenate([[0., 0.], retransformed.ravel()]))

            # plot
            y_reco = fsd_pcad.Reconstruct(time, period = period)


            residuals[counter] = {'dim': 'full' if reduce_dim_to == 0 else str(reduce_dim_to) \
                                  , 'joint': translate.get(joint, joint) \
                                  , 'residual': Residual(y_reco, angle - NP.mean(angle)) \
                                  }
            counter += 1


    residuals= PD.DataFrame.from_dict(residuals).T
    residuals.dropna(inplace = True)
    data_types = {  'dim': str \
                  , 'residual': float \
                }
    residuals = residuals.astype(data_types)
    print (residuals)

    g = SB.catplot(  x = "dim" \
                    , y = "residual" \
                    , col = "joint" \
                    , data = residuals \
                    , kind = "violin" \
                    , width = 1.8 \
                    , split = True \
                    , color = '#99999999' \
                    , linecolor = 'k' \
                    , linewidth = 0.5 \
                    , alpha = 0.8 \
                     # , inner = None \
                     , inner = 'quartile' \
                    )
    g.set_axis_labels("", "residual")
    # g.set_titles("{col_name}")
    g.set_titles("")
    g.set(ylim=(0, 0.05))
    g.set(xticks=[0,3,6,9,12])
    g.fig.set_size_inches(18/2.54, 6/2.54)

    PLT.gcf()
    PLT.tight_layout()
    xy = (0.5, 0.01)
    PLT.annotate(r'PCA dimension', xy = xy, xytext = xy \
                 , xycoords = 'figure fraction' \
                 , ha = 'center', va = 'bottom', rotation = 0 \
                 )
    xy = (0.0, 1.0)
    PLT.annotate(r'B', xy = xy, xytext = xy \
                 , xycoords = 'figure fraction' \
                 , ha = 'left', va = 'top', rotation = 0 \
                 , weight = 'bold' \
                 )

    PLT.gcf().savefig('figures/f6b_residual_pca.pdf', transparent = False, dpi = dpi)
    PLT.close()

    ### merge PDFs
    from PyPDF3 import PdfFileWriter, PdfFileReader
    from PyPDF3.pdf import PageObject

    pdf_filenames = ["figures/f6a_residual_fsd.pdf", "figures/f6b_residual_pca.pdf"]

    input1 = PdfFileReader(open(pdf_filenames[1], "rb"), strict=False)
    input2 = PdfFileReader(open(pdf_filenames[0], "rb"), strict=False)

    page1 = input1.getPage(0)
    page2 = input2.getPage(0)

    total_height = page1.mediaBox.upperRight[1] + page2.mediaBox.upperRight[1]
    total_width = max([page1.mediaBox.upperRight[0], page2.mediaBox.upperRight[0]])

    new_page = PageObject.createBlankPage(None, total_width, total_height)

    # Add first page at the 0,0 position
    new_page.mergePage(page1)
    # Add second page with moving along the axis x
    new_page.mergeTranslatedPage(page2, 0, page1.mediaBox.upperRight[1])

    output = PdfFileWriter()
    output.addPage(new_page)
    output.write(open("figures/f6_residuals.pdf", "wb"))
