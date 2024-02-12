#!/usr/bin/env python3
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import pandas as PD         # data management
import numpy as NP          # numerical analysis
import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API
from tqdm import tqdm as tqdm

SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import FourierToolbox as FT # Fourier Series toolbox
import PlotToolbox as PT

PT.PreparePlot()

dpi = 300


def EqualLimits(ax):
    limits = NP.concatenate([ax.get_xlim(), ax.get_ylim()])
    max_lim = NP.max(NP.abs(limits))

    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])

def SetAxesLimits(axes, limits):
    axes['s'].set_xlim([0., 1.])
    axes['s'].set_ylim(NP.array([-1., 1.])*limits['s'])

    axes['f'].set_xlim(NP.array([-1., 1.])*limits['f'])
    EqualLimits(axes['f'])

    axes['d'].set_xlim(NP.array([-1., 1.])*limits['d'])


def FullDespine(ax):
    ax.spines[:].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def MakeSignalFigure(labels = True, label_show = 3):
    fig = PLT.figure(figsize = (800/dpi, 200/dpi), dpi=dpi)
    fig.subplots_adjust( \
                              top    = 0.99 \
                            , right  = 0.99 \
                            , bottom = 0.01 \
                            , left   = 0.01 \
                            , wspace = 0.08 # column spacing \
                            , hspace = 0.00 # row spacing \
                            )
    rows = [1]
    cols = [4,2,1]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )
    signal_domain = fig.add_subplot(gs[0]) # , aspect = 1/4

    if 0 < label_show:
        signal_domain.axhline(0, ls = '-', color = '0.7', lw = 0.5, zorder = 0)
    # if labels:
    #     # signal_domain.set_xlabel(r'stride cycle')
    #     # signal_domain.set_ylabel(r'angle')
    #     # signal_domain.set_title('time domain', fontsize = 8)
    # else:
    #     if 0 < label_show:
    #         fig.text(0.05, 0.95, 'signal', ha = 'left', va = 'top', fontsize = 10)
    signal_domain.set_xlim([0.,1.])
    # signal_domain.set_title('signal', fontsize = 10)


    frequency_domain = fig.add_subplot(gs[1], aspect = 'equal')
    if 1 < label_show:
        frequency_domain.axhline(0, ls = '-', color = '0.7', lw = 0.5, zorder = 0)
        frequency_domain.axvline(0, ls = '-', color = '0.7', lw = 0.5, zorder = 0)

    # if labels:
    #     frequency_domain.set_xlabel(r'$\Re(c_n)$')
    #     frequency_domain.set_ylabel(r'$\Im(c_n)$')
    #     frequency_domain.set_title('frequency domain', fontsize = 8)
    # else:
    #     if 1 < label_show:
    #         fig.text(0.95, 0.95, 'frequency', ha = 'right', va = 'top', fontsize = 10)

    frequency_domain.yaxis.tick_right()
    frequency_domain.yaxis.set_label_position("right")


    bar_domain = fig.add_subplot(gs[2])
    if 2 < label_show:
        # bar_domain.axvline(0, ls = '-', color = 'k', lw = 0.5, zorder = 0)
        bar_domain.plot([0, 0], [-1, 1], ls = '-', color = '0.7', lw = 0.5, zorder = 0)
    # if labels:
    #     pass
    # else:
    #     if 2 < label_show:
    #         fig.text(0.95, 0.23, 'mean', ha = 'right', va = 'top', fontsize = 10)
    bar_domain.set_ylim([-2,2])
    axes = {'f': frequency_domain, 's': signal_domain, 'd': bar_domain}
    for ax in axes.values():
        FullDespine(ax)

    return fig, axes


def PlotFSD(fsd, axes, limits = None, plot_show = 3, center = True):

    ### signal domain
    time = NP.linspace(0., 1., 101, endpoint = True)
    mean = fsd._c.loc[0, 're']
    signal = fsd.Reconstruct(x_reco = time, period = 1.)

    if 0 < plot_show:
        axes['s'].plot(time, signal-(mean if center else 0.), color = '0.7', ls = '-', lw = 1.0, alpha = 1., zorder = 50)

        if center:
            y = -fsd._c.loc[0, 're']
            # axes['s'].plot([0.2, 0.4], [y, y], lw = 0.5, ls = '-', color = 'k')
        else:
            y = fsd._c.loc[0, 're']
            axes['s'].plot([0.2, 0.4], [y, y], lw = 0.5, ls = '--', color = '0.7')

    ### frequency domain
    coeffs = fsd._c.iloc[1:, :]
    re = coeffs['re'].values
    im = coeffs['im'].values

    if 1 < plot_show:
        axes['f'].scatter(re, im, s = 3, marker = 'o', color = '0.7', alpha = 0.9, zorder = 50)
        axes['f'].plot(re, im, ls = '-', lw = 1.0, color = '0.7', alpha = 0.9, zorder = 50)
    EqualLimits(axes['f'])

    ### mean
    if 2 < plot_show:
        axes['d'].barh(0, mean, align = 'center', facecolor = '0.9', edgecolor = '0.7', lw = 0.5)
    axes['d'].set_ylim(-2.0, 2.0)
    axes['d'].set_xlim(NP.array([-1,1])*mean/NP.sum(fsd.GetAmplitudes()))

    if limits is None:
        limits = { 'f': NP.max(fsd._c.iloc[1:, :].values.ravel()) * 1.33 \
                 , 's': NP.max(NP.abs(axes['s'].get_ylim())) * 1.1 \
                 , 'd': NP.max(NP.abs(axes['d'].get_xlim())) * 1.1 \
                 }
    SetAxesLimits(axes, limits)
    return limits


def LoadAverage(joint):
    filename = f'averages/{joint}.csv'

    coeffs = PD.read_csv(filename, sep = ';').set_index('n', inplace = False)

    fsd = FT.FourierSignal.FromDataFrame(coeffs)
    # print (fsd)
    return fsd


def PlotAllAverages():
    for joint in ['head', 'shoulder', 'elbow', 'wrist', 'metacarpal', 'forelimb']:
        avg = LoadAverage(joint)

        fig, axes = MakeSignalFigure(labels = False)

        limits = PlotFSD(avg, axes)
        fig.savefig(f'averages/{joint}.svg', dpi = dpi, transparent = False)
        PLT.close()


def StepwisePlotCarpal():
    avg = LoadAverage('wrist')

    for step in range(4):
        fig, axes = MakeSignalFigure(labels = False, label_show = step)
        limits = PlotFSD(avg, axes, plot_show = step, center = False)

        color = 'darkblue'

        if step == 1:
            ax = axes['s']

            # x annotation
            mean = avg._c.loc[0, 're']
            amp = NP.sum(avg.GetAmplitudes())
            y = -0.1*amp
            ax.text(0.40, y-0.02, 'time', ha = 'right', va = 'top', fontsize = 8, color = color)
            ax.annotate("", xy=(0.4, y), xytext=(0.0, y) \
                        , arrowprops=dict(arrowstyle="->", color = color, lw = 0.5) \
                        , color = color \
                        )

            # mean annotation
            ax.text(0.3, mean, 'mean', ha = 'center', va = 'bottom', fontsize = 8, color = '0.3')

            # y annotation
            x = 0.01
            y = ax.get_ylim()[1]*0.6
            ax.text(x, y + 0.1, 'angle', ha = 'left', va = 'bottom', fontsize = 8, color = color)
            ax.annotate("", xy=(x, y + 0.1), xytext=(x, 0.1) \
                        , arrowprops=dict(arrowstyle="->", color = color, lw = 0.5) \
                        , color = color \
                        )
        if step == 2:
            ax = axes['f']
            ax.text(ax.get_xlim()[1], -0.01, 're', ha = 'right', va = 'top', fontsize = 8, color = color)
            ax.text(-0.01, ax.get_ylim()[1], 'im', ha = 'right', va = 'top', fontsize = 8, color = color)

        if step == 3:
            x = 0.3
            y = avg._c.loc[0, 're']
            ax = axes['s'] # !
            # ax.plot([0., 0.1], [y, y], lw = 1, ls = '--', color = color)
            ax.annotate("", xy=(x, y), xytext=(x, 0.0) \
                        , arrowprops=dict(arrowstyle="<->", color = color, lw = 0.5) \
                        , color = color \
                        )

            fig.text(0.95, 0.23, 'mean', ha = 'right', va = 'top', fontsize = 10, color = color)

        fig.savefig(f'../figures/fsd_zebracarpus_{step}.svg', dpi = dpi, transparent = False)
        if step == 5:
            PLT.show()
        PLT.close()


def PlotCarpalModified(fsd, dc0 = 0., damp = 1., dphi = 0., limits = None):

    fig, axes = MakeSignalFigure(labels = False)

    FT.Shift(fsd, dc0)
    FT.Scale(fsd, damp)
    FT.Rotate(fsd, dphi)

    limits = PlotFSD(fsd, axes, limits = limits, center = False)
    # fig.text(0.05, 0.05, f'mean += {dc0:.2f}, amp *= {damp:.2f}, phase += {dphi:.2f}' \
             # , ha = 'left', va = 'bottom', fontsize = 8, color = '0.7')

    return fig, limits



def WiggleMean(firstpage):
    #OS.system('rm -rf ./frames/*.png')
    avg = LoadAverage('wrist')
    max_shift = 0.2
    max_scale = 1.32

    limits = { 'f': NP.max(avg._c.iloc[1:, :].values.ravel()) * (max_scale + 0.01) \
                 , 's': NP.max(NP.abs(avg.Reconstruct(NP.linspace(0., 1., 101, endpoint = True)))) * (max_scale + 0.01) + max_shift \
                 , 'd': avg._c.iloc[0, 0] *1.05 + max_shift \
                 }

    # duration = 8 # s
    # fps = 25 # Hz
    n_frames = 49
    centers = 0.4*NP.sin(NP.linspace(0., 2*NP.pi, n_frames, endpoint = True))-0.2

    # # plot original signal
    # fig, _ = PlotCarpalModified(fsd = avg.Copy(), limits = limits)
    # fig.savefig(f'carpal_default.svg', dpi = dpi, transparent = False)
    # PLT.close()

    # loop plot
    for frame_nr, dc0 in tqdm(enumerate(centers)):
        fig, _ = PlotCarpalModified(fsd = avg.Copy(), dc0 = dc0, limits = limits)
        fig.savefig(f'../frames/p{firstpage + frame_nr:.0f}.pdf', dpi = dpi, transparent = True)
        # fig.savefig(f'frames/{frame_nr:04.0f}.png', dpi = dpi, transparent = False)
        PLT.close()

    # create video
    # OS.system(f'ffmpeg -y -r {fps:.0f} -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 carpal_d_mean.webm')
    return firstpage + n_frames


def WiggleAmplitude(firstpage):
    # OS.system('rm -rf ./frames/*.png')
    avg = LoadAverage('wrist')
    max_shift = 0.2
    max_scale = 1.32

    limits = { 'f': NP.max(avg._c.iloc[1:, :].values.ravel()) * (max_scale + 0.01) \
                 , 's': NP.max(NP.abs(avg.Reconstruct(NP.linspace(0., 1., 101, endpoint = True)))) * (max_scale + 0.01) + max_shift \
                 , 'd': avg._c.iloc[0, 0] *1.05 + max_shift \
                 }

    # duration = 8 # s
    # fps = 25 # Hz
    n_frames = 49
    amps = 0.4*NP.sin(NP.linspace(0., 2*NP.pi, n_frames, endpoint = True))+ 0.8

    for frame_nr, damp in tqdm(enumerate(amps)):
        fig, _ = PlotCarpalModified(fsd = avg.Copy(), damp = damp, limits = limits)
        fig.savefig(f'../frames/p{firstpage + frame_nr:.0f}.pdf', dpi = dpi, transparent = True)
        # fig.savefig(f'frames/{frame_nr:04.0f}.png', dpi = dpi, transparent = False)
        PLT.close()

    #OS.system(f'ffmpeg -y -r {fps:.0f} -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 carpal_d_amp.webm')
    return firstpage + n_frames


def WigglePhase(firstpage):
    avg = LoadAverage('wrist')
    max_shift = 0.2
    max_scale = 1.32

    limits = { 'f': NP.max(avg._c.iloc[1:, :].values.ravel()) * (max_scale + 0.01) \
                 , 's': NP.max(NP.abs(avg.Reconstruct(NP.linspace(0., 1., 101, endpoint = True)))) * (max_scale + 0.01) + max_shift \
                 , 'd': avg._c.iloc[0, 0] *1.05 + max_shift \
                 }

    # duration = 8 # s
    # fps = 25 # Hz
    n_frames = 49
    phases = 0.25*NP.sin(NP.linspace(0., 2*NP.pi, n_frames, endpoint = True))

    for frame_nr, dphi in tqdm(enumerate(phases)):
        fig, _ = PlotCarpalModified(fsd = avg.Copy(), dphi = dphi, limits = limits)
        fig.savefig(f'../frames/p{firstpage + frame_nr:.0f}.pdf', dpi = dpi, transparent = True)
        PLT.close()

    return firstpage + n_frames
    # OS.system(f'ffmpeg -y -r {fps:.0f} -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 carpal_d_phase.webm')




if __name__ == "__main__":
    # PlotAllAverages()
    # StepwisePlotCarpal()

    OS.system('rm -rf ../frames/*.pdf')
    endframe = WigglePhase(1)
    endframe = WiggleMean(endframe )
    endframe = WiggleAmplitude(endframe )
