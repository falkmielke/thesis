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


reference_frame = 3 # the limb touchdown, presumed start
test_frames = [93,96,99,102,105] # the candidate end frames
shift_frames = list(range(7)) # start shifting


labels = { \
           1: 'head' \
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
excluded_landmarks = [3,4,15,16,17]
connections = [[1,5] \
               , [3,4] \
               , [5,6] \
               , [6,8] \
               , [8,9] \
               , [14,15] \
               , [15,16] \
               , [16,17] \
               ]


line_color = '#AAFFAA'# '#F3E6BB'
xy = ['x', 'y']
dpi = 300

def MakeFigure():

    fig = PLT.figure(figsize = (18/2.54, 12/2.54), dpi=dpi)
    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.96 \
                            , bottom = 0.12 \
                            , left   = 0.12 \
                            , wspace = 0.20 # column spacing \
                            , hspace = 0.12 # row spacing \
                            )
    rows = [1,1]
    cols = [6,4]
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
    kinematics.index.name = "frame_nr"

    kinematics = kinematics.loc[:, ['time']+[f'pt{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]]

    return kinematics


def PlotLongLine(ax, pt1, pt2, extend = 0.1, **kwargs):
    vec = pt2 - pt1
    start = pt1 - extend*vec
    end = pt2 + extend*vec

    line = PLT.Line2D([start[0], end[0]], [start[1], end[1]], **kwargs)
    ax.add_line(line)
    return line


def CoordsToConfiguration(coords, y_invert = False):
    cols = list(sorted(set([pt[:-2] for pt in coords.index.values])))
    configuration = PD.DataFrame( \
                        NP.stack([coords.loc[[f'{col}_{crd}' for col in cols]]
                                  for crd in xy], axis = 1) \
                         , index = cols \
                         , columns = xy \
                        )
    # mirror y axis to match image coords
    if y_invert:
        configuration.loc[:, 'y'] = 800-configuration.loc[:, 'y']

    configuration.loc[:, 'z'] = 0.
    return configuration



def PlotRawPositions(ax, kinematics):

    columns_all = [f'pt{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]
    points_selection = [f'pt{lm:02.0f}' for lm in landmarks \
                    if not lm in excluded_landmarks]
    reference_coords = kinematics.loc[reference_frame, columns_all]
    reference_config = CoordsToConfiguration(reference_coords, y_invert = True)
    target_frame = test_frames[len(test_frames)//2]
    target_config = CoordsToConfiguration(kinematics.loc[target_frame, columns_all], y_invert = True)


    # split_y = ( ST.Centroid(reference_config.loc[points_selection, :])['x'] \
    #           + ST.Centroid(target_config.loc[points_selection, :])['x'] \
    #           ) //2
    split_y = 810


    img1 = PLT.imread(f'../archive/Elfira/{reference_frame:06.0f}.png')
    img2 = PLT.imread(f'../archive/Elfira/{target_frame:06.0f}.png')

    img = img1.copy()
    img[:, split_y:, :] = img2[:, split_y:, :]
    ax.imshow(img, origin = 'upper', cmap = 'gray', zorder = 0)

    ref_color = '#AAFFAA'
    target_color = '#FFBBFF'
    scatter_kwargs = dict( \
                s = 5 \
               , marker = 'o' \
               , edgecolor = ref_color \
               , facecolor = 'none' \
               , linewidth = 0.5 \
               , alpha = 0.8 \
               , zorder = 10 \
               )
    ax.scatter(reference_config['x'], reference_config['y'] \
               , **scatter_kwargs \
               )

    # connect scatter points
    line_kwargs = dict(color = ref_color, lw = 0.5, ls = '-', zorder = 10, alpha = 0.8)
    for pt1, pt2 in connections:

        p1 = reference_config.loc[f'pt{pt1:02.0f}', xy].values
        p2 = reference_config.loc[f'pt{pt2:02.0f}', xy].values
        PlotLongLine(ax, p1, p2, extend = 0.0, **line_kwargs)


    line_kwargs['color'] = target_color

    for pt1, pt2 in connections:
        p1 = target_config.loc[f'pt{pt1:02.0f}', xy].values
        p2 = target_config.loc[f'pt{pt2:02.0f}', xy].values
        PlotLongLine(ax, p1, p2, extend = 0.0, **line_kwargs)

    scatter_kwargs = dict( \
                s = 5 \
               , marker = 'x' \
               , facecolor = target_color \
               , linewidth = 0.5 \
               , alpha = 0.8 \
               , zorder = 10 \
               )
    for target_frame in test_frames:
        target_config = CoordsToConfiguration(kinematics.loc[target_frame, columns_all], y_invert = True)
        ax.scatter(target_config['x'], target_config['y'] \
                   , **scatter_kwargs \
                   )




    for coord in xy:
        eval(f'ax.get_{coord}axis().set_visible(False)')

    ax.spines[:].set_visible(False)

    ax.set_ylim([740, 260])
    ax.set_xlim([460,1240])


def PlotSuperimposition(ax, kinematics):

    columns_all = [f'pt{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]
    points_selection = [f'pt{lm:02.0f}' for lm in landmarks \
                    if not lm in excluded_landmarks]
    reference_coords = kinematics.loc[reference_frame, columns_all]
    reference_config = CoordsToConfiguration(reference_coords)

    reference_shape = reference_config.loc[points_selection, :].values

    target_configs_si = []
    min_pd = None
    best_cfg = None
    for target_frame in test_frames:
        target_config = CoordsToConfiguration(kinematics.loc[target_frame, columns_all])
        target_shape = target_config.loc[points_selection, :].values

        _, pd, hist = ST.Procrustes(target_shape, reference_shape)
        si_config = target_config.copy()

        si_config.loc[:, :] = ST.ApplyTransformations(si_config.values, hist)
        target_configs_si.append(si_config)

        # store best SI
        if (min_pd is None) or (pd < min_pd):
            best_cfg = si_config


    ### Plot
    ref_color = '#227722' # '#AAFFAA'
    target_color = '#AA44AA' # '#736633' # '#F3E6BB'
    scatter_kwargs = dict( \
                s = 5 \
               , marker = 'o' \
               , edgecolor = ref_color \
               , facecolor = 'none' \
               , linewidth = 0.5 \
               , alpha = 0.8 \
               , zorder = 10 \
               )
    ax.scatter(reference_config['x'], reference_config['y'] \
               , **scatter_kwargs \
               )

    # connect scatter points
    line_kwargs = dict(color = ref_color, lw = 0.5, ls = '-', zorder = 10, alpha = 0.8)
    for pt1, pt2 in connections:

        p1 = reference_config.loc[f'pt{pt1:02.0f}', xy].values
        p2 = reference_config.loc[f'pt{pt2:02.0f}', xy].values
        PlotLongLine(ax, p1, p2, extend = 0.0, **line_kwargs)


    line_kwargs['color'] = target_color

    for pt1, pt2 in connections:
        p1 = best_cfg.loc[f'pt{pt1:02.0f}', xy].values
        p2 = best_cfg.loc[f'pt{pt2:02.0f}', xy].values
        PlotLongLine(ax, p1, p2, extend = 0.0, **line_kwargs)



    scatter_kwargs = dict( \
                s = 5 \
               , marker = 'x' \
               , facecolor = target_color \
               , linewidth = 0.5 \
               , alpha = 0.8 \
               , zorder = 10 \
               )
    for si_confg in target_configs_si:
        ax.scatter(si_confg['x'], si_confg['y'] \
                   , **scatter_kwargs \
                   )


    for coord in xy:
        eval(f'ax.get_{coord}axis().set_visible(False)')

    ax.spines[:].set_visible(False)


def PlotProcrustesDistances(ax, kinematics):

    columns_all = [f'pt{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]
    points_selection = [f'pt{lm:02.0f}' for lm in landmarks \
                    if not lm in excluded_landmarks]

    cycles = 0.9 - (NP.abs(NP.array(shift_frames)-reference_frame)/(len(shift_frames)-reference_frame))
    ax.set_prop_cycle(PLT.cycler('color',PLT.cm.Blues(cycles)))

    for ref_frame in shift_frames:
        reference_coords = kinematics.loc[ref_frame, columns_all]
        reference_config = CoordsToConfiguration(reference_coords)

        reference_shape = reference_config.loc[points_selection, :].values

        procrustes_distances = []
        min_pd = None
        best_frame = None
        for target_frame in kinematics.index.values:
            target_config = CoordsToConfiguration(kinematics.loc[target_frame, columns_all])
            target_shape = target_config.loc[points_selection, :].values

            _, pd, hist = ST.Procrustes(target_shape, reference_shape)

            procrustes_distances.append(1000 * pd)

            # store best SI
            if target_frame-ref_frame > 25:
                if (min_pd is None) or (pd < min_pd):
                    best_frame = target_frame

        procrustes_distances = NP.array(procrustes_distances)
        procrustes_distances = NP.log(2. + procrustes_distances)
        ax.plot(kinematics.index.values \
                , procrustes_distances \
                , ls = '-' \
                , lw = 1. \
                , alpha = 0.8 \
                )

    # ax.set_yscale('log', base = 10)

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('frame number')
    ax.set_ylabel(r'log Procrustes Distance $\left(\times 10^3\right)$')
    ax.set_xlim([0, NP.max(kinematics.index.values)])

def PlotProcrustesHeatmap(ax, kinematics):

    columns_all = [f'pt{lm:02.0f}_{coord}' for lm in landmarks for coord in xy]
    points_selection = [f'pt{lm:02.0f}' for lm in landmarks \
                    if not lm in excluded_landmarks]

    procrustes_distances = {}
    for ref_frame in shift_frames:
        procrustes_distances[ref_frame] = {}

        reference_coords = kinematics.loc[ref_frame, columns_all]
        reference_config = CoordsToConfiguration(reference_coords)

        reference_shape = reference_config.loc[points_selection, :].values

        for test_frame in kinematics.index.values[test_frames[0]:]:
            test_config = CoordsToConfiguration(kinematics.loc[test_frame, columns_all])
            test_shape = test_config.loc[points_selection, :].values

            _, pd, _ = ST.Procrustes(test_shape, reference_shape)

            procrustes_distances[ref_frame][test_frame] = NP.log(2+1000 * pd)


    procrustes_distances = PD.DataFrame.from_dict(procrustes_distances).T
    # print (procrustes_distances)
    # print (NP.argmin(procrustes_distances.values))

    SB.heatmap(procrustes_distances \
               , annot=False \
               , fmt="0.1e" \
               , linewidths=.5 \
               , ax=ax \
               , cmap = 'Blues' \
               , cbar_kws={'label': r'$\times 10^3$'} \
               )

    ### crosshairs
    line_kwargs = dict( \
                        lw = 1. \
                        , ls = '--' \
                        , color = '0.2' \
                        , alpha = 0.8 \
                       )
    ax.axhline(2+0.5, **line_kwargs)
    ax.axvline(99+0.5-test_frames[0], **line_kwargs)

    line_kwargs['ls'] = '-'
    line_kwargs['color'] = PLT.cm.Blues([0.9])
    # only count inner part of the matrix
    vals = procrustes_distances.values[1:-1, 1:-1]
    best = NP.unravel_index(NP.argmin(vals, axis=None) \
                            , vals.shape)
    ax.axhline(best[0]+1.5, **line_kwargs)
    ax.axvline(best[1]+1.5, **line_kwargs)

    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::2])

    ax.set_ylabel('start frame')
    ax.set_xlabel('end frame')

if __name__ == "__main__":
    fig, gs = MakeFigure()

    kinematics = LoadKinematics()
    # print (kinematics)

    ax = fig.add_subplot(gs[0, 0], aspect = 'equal')
    PlotRawPositions(ax, kinematics)

    ax = fig.add_subplot(gs[0, 1], aspect = 'equal')
    PlotSuperimposition(ax, kinematics)

    ax = fig.add_subplot(gs[1, 0])
    PlotProcrustesDistances(ax, kinematics)

    ax = fig.add_subplot(gs[1, 1])
    PlotProcrustesHeatmap(ax, kinematics)

    # panel letters
    for letter, xy in zip(['A', 'B', 'C', 'D'], [(0., 1.), (0.58, 1.0), (0., 0.55), (0.58, 0.55)]):
        PLT.annotate(r'%s' % (letter), xy = xy, xytext = xy \
                     , xycoords = 'figure fraction' \
                     , ha = 'left', va = 'top' \
                     , weight = 'bold' \
                     )

    fig.savefig('figures/f3_endstart_Procrustes.pdf', dpi = dpi, transparent = False)
    PLT.show()
