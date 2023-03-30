#!/usr/bin/env python3

import sys as SYS
import numpy as NP
import pandas as PD
import matplotlib as MP
import matplotlib.pyplot as PLT
import seaborn as SB

SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import QuaternionGeometryToolbox as QGT # point superimposition tools (e.g. Procrustes)
import SuperimpositionToolbox as ST # superimposition of landmarks

import Code03_EndStartMatching as ESM

xy = ['x', 'y']
dpi = 300



labels = { \
           1: 'head' \
        ,  2: 'nose' \
        ,  3: 'tailbase' \
        ,  4: 'tailtip' \
        ,  5: 'hip' \
        ,  6: 'knee' \
        ,  8: 'heel' \
        ,  9: 'foot' \
        , 14: 'acromion' \
        , 15: 'elbow' \
        , 16: 'wrist' \
        , 17: 'finger' \
        }
landmarks = [k for k in labels.keys()]


def MakeFigure():

    fig = PLT.figure(figsize = (18/2.54, 4.0/2.54), dpi=dpi)
    fig.tight_layout()
    crunch = .0
    fig.subplots_adjust( \
                              top    = 0.98-crunch \
                            , right  = 0.98 \
                            , bottom = 0.02+crunch \
                            , left   = 0.02 \
                            , wspace = 0.05 # column spacing \
                            , hspace = 0.00 # row spacing \
                            )
    rows = [1]
    cols = [4.475, 1]
    # set spines visible to get good scale
    # ... unless we zoom in on the second plot
    cols = [2.5, 1]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )

    return fig, gs



def LoadKinematics(rel = True):
    kinematics = PD.read_csv("../data/all_strides.csv", sep = ';', header = 0)

    # print (kinematics.groupby(['subject','sheet_nr','stride_idx']).agg(len).reset_index())

    # select a bout of strides by Espiegle
    kinematics = kinematics.loc[[val in [17, 18, 19, 20] for val in kinematics['stride_idx'].values], :]

    # ... or Desastre (actually my favorite name, but otherwise a desaster
    # kinematics = kinematics.loc[[val in [7, 8, 9] for val in kinematics['stride_idx'].values], :]

    kinematics.reset_index(inplace = True)
    kinematics.index.name = "frame_nr"

    # shift to the animals reference frame
    reference_landmarks = [1, 3, 5]

    scale = NP.linalg.norm(kinematics.loc[:, [f'pt01_{coord}' for coord in xy]].values[0, :] - kinematics.loc[:, [f'pt05_{coord}' for coord in xy]].values[0, :])
    print (scale)

    for coord in xy:
        kinematics.loc[:, f'ref_{coord}'] = NP.mean(kinematics.loc[:, [f'pt{lm:02.0f}_{coord}' for lm in reference_landmarks]].values, axis = 1)

        shift = kinematics.loc[:, f'ref_{coord}'].values[0]
        for point in [f'pt{lm:02.0f}' for lm in landmarks] + ['ref']:
            kinematics.loc[:, f'{point}_{coord}'] -= shift
            kinematics.loc[:, f'{point}_{coord}'] /= scale



    for lm in landmarks:
        if rel:
            kinematics.loc[:, [f'rel{lm:02.0f}_{coord}' for coord in xy]] \
                = kinematics.loc[:, [f'pt{lm:02.0f}_{coord}' for coord in xy]].values \
                - kinematics.loc[:, [f'ref_{coord}' for coord in xy]].values
        else:
            kinematics.loc[:, [f'rel{lm:02.0f}_{coord}' for coord in xy]] \
                = kinematics.loc[:, [f'pt{lm:02.0f}_{coord}' for coord in xy]].values


    kinematics = kinematics.loc[:, ['time', 'stride_idx']+[f'rel{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]]


    return kinematics


def InspectionPlot(ax, kinematics, strides = None, limits = None):


    columns_all = [f'rel{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]
    points_selection = [f'rel{lm:02.0f}' for lm in landmarks]
    reference_coords = kinematics.loc[kinematics.index.values[-1], columns_all]
    reference_config = ESM.CoordsToConfiguration(reference_coords)



    # connect scatter points
    line_kwargs = dict(color = 'k', lw = 0.5, ls = '-', zorder = 10, alpha = 0.8)
    for pt1, pt2 in ESM.connections:
        if (pt1 not in landmarks) or (pt2 not in landmarks):
            continue

        p1 = reference_config.loc[f'rel{pt1:02.0f}', xy].values
        p2 = reference_config.loc[f'rel{pt2:02.0f}', xy].values
        ESM.PlotLongLine(ax, p1, p2, extend = 0.0, **line_kwargs)



    ### plot consecutive strides
    if strides is None:
        strides = NP.unique(kinematics['stride_idx'].values)
        cycles = [1.-0.4*(sc)/(len(strides)-1) for sc in range(len(strides))]
    else:
        cycles = [0.9]

    # stride_colors = [(1.0 - 0.9*(sc+1)/len(strides), 0.2, 0.2+0.7*(sc+1)/len(strides)) for sc in range(len(strides))]
    stride_colors = PLT.cm.Blues(cycles)
    # stride_colors = PLT.cm.Set1(cycles)
    # stride_colors = PLT.cm.Dark2(cycles)
    # stride_colors = PLT.cm.Accent(cycles)
    # print (stride_colors)

    for stride_idx, color in zip(strides, stride_colors):
        data = kinematics.loc[kinematics['stride_idx'].values == stride_idx, :]

        for lm in landmarks:
            ax.scatter(data.loc[:, f'rel{lm:02.0f}_x'].values \
                       , data.loc[:, f'rel{lm:02.0f}_y'].values \
                       , s = 3 \
                       , marker = 'o' \
                       , color = color \
                       , alpha = 0.4 \
                       )

    for coord in xy:
        eval(f'ax.get_{coord}axis().set_visible(False)')
    ax.spines[:].set_visible(True)
    ax.spines[:].set_visible(False)


    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])


if __name__ == "__main__":
    fig, gs = MakeFigure()

    kinematics = LoadKinematics(rel = False)
    # print (kinematics)

    ax = fig.add_subplot(gs[0], aspect = 'equal')
    InspectionPlot(ax, kinematics)

    kinematics = LoadKinematics(rel = True)

    ax = fig.add_subplot(gs[1], aspect = 'equal')#, sharey = ax)
    InspectionPlot(ax, kinematics, strides = [20], limits = [[-.65, .75], [-1.2, -.28]])

    # panel letters
    for letter, xy in zip(['A', 'B'], [(0., 1.), (0.72, 1.0)]):
        PLT.annotate(r'%s' % (letter), xy = xy, xytext = xy \
                     , xycoords = 'figure fraction' \
                     , ha = 'left', va = 'top' \
                     , weight = 'bold' \
                     )



    fig.savefig('figures/f4_raw_inspection.pdf', dpi = dpi, transparent = False)
    PLT.show()
