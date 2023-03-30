#!/usr/bin/env python3

import sys as SYS
import numpy as NP
import pandas as PD
import matplotlib as MP
import matplotlib.pyplot as PLT

SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import QuaternionGeometryToolbox as QGT # point superimposition tools (e.g. Procrustes)

keyframes = [2,25,51,75,99]
keyframes_relative = [kf - keyframes[0] for kf in keyframes ]
n_keyframes = len(keyframes)

landmarks = [1,5,6]
labels = {1: 'head', 5: 'hip', 6: 'knee'}
line_color = '#AAFFAA'# '#F3E6BB'

xy = ['x', 'y']
dpi = 300

def MakeFigure():

    fig = PLT.figure(figsize = (18/2.54, 10/2.54), dpi=dpi)
    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.97 \
                            , bottom = 0.02 \
                            , left   = 0.12 \
                            , wspace = 0.02 # column spacing \
                            , hspace = 0.12 # row spacing \
                            )
    rows = [2,3,1,3,1]
    cols = [1]*n_keyframes
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )

    return fig, gs


def LoadKinematics():
    kinematics = PD.read_csv("../data/all_strides.csv", sep = ';', header = 0)
    kinematics = kinematics.loc[kinematics['stride_idx'].values == 10, :]

    kinematics.reset_index(inplace = True)
    kinematics.index = kinematics.index.values + 1 - 2
    kinematics.index.name = "frame_nr"

    kinematics['cycle_progress'] = (kinematics.index.values - 2)/97

    kinematics = kinematics.loc[:, ['time', 'cycle_progress']+[f'pt{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]]

    for lm in landmarks:
        if lm == 5:
            continue

        kinematics.loc[:, [f'rel{lm:02.0f}_{coord}' for coord in xy]] \
            = kinematics.loc[:, [f'pt{lm:02.0f}_{coord}' for coord in xy]].values \
            - kinematics.loc[:, [f'pt05_{coord}' for coord in xy]].values

    return kinematics


def PlotLongLine(ax, pt1, pt2, extend = 0.1, **kwargs):
    vec = pt2 - pt1
    start = pt1 - extend*vec
    end = pt2 + extend*vec

    line = PLT.Line2D([start[0], end[0]], [start[1], end[1]], **kwargs)
    ax.add_line(line)
    return line



def PlotFrames(ax, imfi, points):
    img = PLT.imread(imfi)

    ax.imshow(img, origin = 'upper', cmap = 'gray', zorder = 0)

    plot_data = NP.stack([points[[f'pt{lm:02.0f}_{coord}' for coord in xy]] for lm in landmarks])
    plot_data[:, 1] = 800 - plot_data[:, 1]
    # print (plot_data)

    ax.scatter(plot_data[:, 0] \
               , plot_data[:, 1] \
               , s = 5 \
               , marker = 'o' \
               , edgecolor = line_color \
               , facecolor = 'none' \
               , linewidth = 0.5 \
               , alpha = 0.8 \
               , zorder = 10 \
               )

    p1 = plot_data[0, :]
    p2 = plot_data[1, :]
    p3 = plot_data[2, :]
    line_kwargs = dict(color = line_color, lw = 0.5, ls = '-', zorder = 10, alpha = 0.8)
    l1 = PlotLongLine(ax, p1, p2, extend = -0.0, **line_kwargs)
    l2 = PlotLongLine(ax, p2, p3, extend = -0.0, **line_kwargs)

    p2x = p2 + 80 * (p2-p1) / NP.linalg.norm(p2-p1)
    ax.plot([p2[0], p2x[0]], [p2[1], p2x[1]], ls = '--', **{k:v for k, v in line_kwargs.items() if not k == 'ls'})

    v0 = NP.array([1, 0])
    v1 = p2 - p1
    v2 = p3 - p2


    # plot angle
    # https://stackoverflow.com/a/25228427
    r = 150
    theta1 = NP.degrees(NP.math.atan2(NP.linalg.det([v0,v1]),NP.dot(v0,v1)))
    theta2 = NP.degrees(NP.math.atan2(NP.linalg.det([v0,v2]),NP.dot(v0,v2)))
    # print (theta1, theta2)
    arc = MP.patches.Arc(p2, r, r, 0, theta2, theta1, **line_kwargs)
    ax.add_patch(arc)


    for coord in xy:
        eval(f'ax.get_{coord}axis().set_visible(False)')

    ax.spines[:].set_visible(False)

    ax.set_ylim([740, 260])
    ax.set_xlim([460,1240])



def PlotRawKinematics(ax, kinematics):
    # print (kinematics)

    plot_kwargs = dict(ls = '-', lw = 1, alpha = 1., zorder = 20)
    PositionFromStart = lambda vec: vec - vec[0]

    t = kinematics.index.values
    label_position = 45

    for lm in landmarks:
        if lm not in [6]:
            continue

        for coord in xy:
            trace = PositionFromStart(kinematics.loc[:, f'rel{lm:02.0f}_{coord}'])

            ax.plot( t\
                   , trace \
                   , color = 'k' \
                   , **plot_kwargs \
                   )

            ax.text(label_position, 2.+trace[t == label_position] \
                    , f"{coord}" \
                    # , f"{labels[lm]} {coord}" \
                    , ha = 'left', va = 'bottom' \
                    , fontsize = 10 \
                    , alpha = 0.8 \
                    , zorder = 20 \
                    )

    for frame in keyframes:
        ax.axvline(frame, color = (0.1, 0.1, 0.5), lw = 0.5, ls = ':', alpha = 0.6, zorder = 50)

    ax.axhline(0, color = (0.1, 0.1, 0.1), lw = 0.5, ls = '-', alpha = 0.6, zorder = 0)

    ax.set_ylabel('relative knee \npoint (px)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('video frame')
    ax.set_xlim([kinematics.index.values[0], kinematics.index.values[-1]])


def PlotJointAngleProfile(ax, kinematics):

    positions = [ kinematics.loc[:, [f'pt{lm:02.0f}_{coord}' for coord in xy]].values \
                  for lm in landmarks]
    # calculate vectors = differences between two points, pointing distally
    proximal_vector = positions[1] - positions[0]
    distal_vector = positions[2] - positions[1]

    # get joint angle:
    #  - zero at straight joint | +/-π at fully folded configuration
    #  - EXCEPT head and shoulder: zero is hanging down
    #  - positive at CCW rotation, negative at CW rotation
    #  - remember: all movements rightwards
    joint_ang = -QGT.WrapAnglePiPi( \
                 [QGT.AngleBetweenCCW(pvec, dvec) \
                  for pvec, dvec in zip(proximal_vector, distal_vector)] \
                )

    t = kinematics['cycle_progress'].values
    ax.plot(t, joint_ang \
            , lw = 1, ls = '-', color = (0.2, 0.2, 0.2) \
            , zorder = 20 \
            )

    for frame in keyframes:
        ax.axvline(kinematics.loc[frame, 'cycle_progress'] \
                   , color = (0.1, 0.1, 0.5), lw = 0.5, ls = ':', alpha = 0.6, zorder = 50)


    ax.axhline(joint_ang[kinematics.index.values == keyframes_relative[0]] \
                , color = (0.1, 0.1, 0.1), lw = 0.5, ls = '-', alpha = 0.6, zorder = 0)

    ### Extension/Flexion labels
    textpos = (0.2, 0.6)
    ax.annotate(r'' \
                , xy = (0.3, 0.4) \
                , xytext = textpos \
                , xycoords = 'axes fraction' \
                , arrowprops = dict(  arrowstyle="->" \
                                    , linestyle="-" \
                                    , linewidth=0.5 \
                                    , shrinkA=0 \
                                    , shrinkB=0 \
                                      , color = '0.2' \
                                      , alpha = 0.8 \
                                    ) \
                )
    ax.annotate(r'extension' \
                , xy = textpos \
                , xytext = textpos \
                , xycoords = 'axes fraction' \
                , ha = 'left', va = 'bottom' \
                                      , color = '0.2' \
                                      , alpha = 0.8 \
                )

    textpos = (0.55, 0.2)
    ax.annotate(r'' \
                , xy = textpos \
                , xytext = (0.65, 0.70) \
                , xycoords = 'axes fraction' \
                , arrowprops = dict(  arrowstyle="<-" \
                                    , linestyle="-" \
                                    , linewidth=0.5 \
                                    , shrinkA=0 \
                                    , shrinkB=0 \
                                      , color = '0.2' \
                                      , alpha = 0.8 \
                                    ) \
                )
    ax.annotate(r'flexion' \
                , xy = textpos \
                , xytext = textpos \
                , xycoords = 'axes fraction' \
                , ha = 'left', va = 'top' \
                                      , color = '0.2' \
                                      , alpha = 0.8 \
                )




    ax.set_ylabel('joint angle \nprofile (rad)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('cycle progress')
    ax.set_xlim([t[0], t[-1]])


if __name__ == "__main__":
    fig, gs = MakeFigure()

    kinematics = LoadKinematics()
    # print (kinematics)

    for ax_idx, (img_nr, frame) in enumerate(zip(keyframes, keyframes_relative)):
        ax = fig.add_subplot(gs[0, ax_idx], aspect = 'equal') # , aspect = 1/4
        imfi = f'../archive/Elfira/{img_nr:06.0f}.png'

        PlotFrames(ax, imfi, kinematics.loc[frame, :])


    ax = fig.add_subplot(gs[1, :])
    PlotRawKinematics(ax, kinematics)

    ax = fig.add_subplot(gs[3, :])
    PlotJointAngleProfile(ax, kinematics)

    for letter, pos in zip(['A', 'B', 'C'], [1.00, 0.80, 0.40]):
        xy = (0.99, pos-0.01)
        PLT.annotate(r'%s' % (letter), xy = xy, xytext = xy \
                     , xycoords = 'figure fraction' \
                     , ha = 'right', va = 'top' \
                     , weight = 'bold' \
                     )
    for y in [0.8, 0.4]:
        PLT.annotate('', xy = (0.0, y), xytext = (1.0, y) \
                     , xycoords = 'figure fraction' \
                     , arrowprops = dict(  arrowstyle="-" \
                                         , linestyle="-" \
                                         , linewidth=0.5 \
                                         , shrinkA=0 \
                                         , shrinkB=0 \
                                           , color = '0.8' \
                                           , alpha = 0.5 \
                                         ) \
                     )


    fig.savefig('figures/f1_jointangle.pdf', dpi = dpi, transparent = False)
    PLT.show()
