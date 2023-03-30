#!/usr/bin/env python3

"""
Demo Program: Fourier Series Decomposition
... as used first on the DZG meeting 2019.

Try it!
"""

__author__      = "Falk Mielke"
__date__        = 20190906



import tkinter as TK
import matplotlib as MP
import matplotlib.pyplot as MPP

import FourierToolbox as FKT



# https://matplotlib.org/gallery/user_interfaces/embedding_in_tk_sgskip.html
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
# from matplotlib.figure import Figure

# https://matplotlib.org/3.1.0/gallery/style_sheets/dark_background.html


import numpy as NP


fsd_order = 5
slid_coefficients = range(2*(fsd_order)+1)
coefficient_labels = ['Re0'] + ['%s%i' % ('Re' if ((c-1) % 2 == 0) else 'Im', (c+1)//2) for c in slid_coefficients[1:] ]
# print (coefficient_labels)


references = { \
            'carpus': [0.2471, 0.0, -0.0299, 0.3206, -0.1871, -0.0623, 0.0547, -0.0749, 0.0156, 0.0282, -0.0074, -0.0045] \
            , 'elbow': [-1.0629, 0.0, -0.1629, -0.1399, 0.0874, -0.0365, 0.0012, 0.0338, -0.0025, -0.0072, 0.0026, 0.006] \
            , 'real': [1.5359, 0.0000, -0.2260, -0.1591, 0.1003, -0.0477, 0.0005, 0.0410, 0.0096, 0.0022, 0.0143, -0.0008] \
            , '-PC1': [0.0, 0.0, -0.1043, -0.0747, 0.0426, -0.0363, 0.0109, 0.0136, 0.005, 0.0008, 0.0013, 0.0071] \
            , 'mean': [0.0, 0.0, -0.0607, -0.0763, 0.051, -0.023, 0.0074, 0.0232, -0.0036, 0.0024, -0.0007, 0.0034] \
            , '+PC1': [0.0, 0.0, -0.0268, -0.0774, 0.0575, -0.0127, 0.0046, 0.0307, -0.0104, 0.0036, -0.0023, 0.0004] \
            # , 'dzg': [-0.4, 0.0, 0.000, -0.433, -0.262, 0.015, -0.029, 0.133, 0.042, 0.042, 0.016, -0.012] \
            }
# reference_fsds = {ref: \
#                 FKT.FourierSignal.FromComplexVector( FKT.MakeComplex(NP.array(coeffs).reshape((-1,2))) ) \
#             for ref, coeffs in references.items() if ref in ['dzg']}

# for rfsd in reference_fsds.values():
#     FKT.Scale(rfsd, 3.)

MPP.style.use('dark_background')

the_font = {  \
        # It's really sans-serif, but using it doesn't override \sffamily, so we tell Matplotlib
        # to use the "serif" font because then Matplotlib won't ask for any special families.
         # 'family': 'serif' \
        # , 'serif': 'Iwona' \
        'family': 'sans-serif'
        , 'sans-serif': 'DejaVu Sans'
        , 'size': 14#*1.27 \
    }
MP.rcParams['text.usetex'] = False

MPP.rc('font',**the_font)
# Tell Matplotlib how to ask TeX for this font.
# MP.texmanager.TexManager.font_info['iwona'] = ('iwona', r'\usepackage[light,math]{iwona}')

MP.rcParams['text.latex.preamble'] = ",".join([\
              r'\usepackage{upgreek}'
            , r'\usepackage{cmbright}'
            , r'\usepackage{sansmath}'
            ])

MP.rcParams['pdf.fonttype'] = 42 # will make output TrueType (whatever that means)

tk_config = dict( \
                  bg = 'black' \
                , fg = 'white' \
                , font = ("DejaVu Sans", 16) \
                )


affine_shortcuts = { \
                      'left': [FKT.Rotate, -0.05] \
                    , 'right': [FKT.Rotate, 0.05] \
                    , 'up': [FKT.Shift, 0.1] \
                    , 'down': [FKT.Shift, -0.1] \
                    , '+': [FKT.Scale, 1.05] \
                    , '-': [FKT.Scale, 0.95] \
                    }




class MuFuSlider(TK.Frame):
    # MUlti FUnctional TK SLIDER
    def __init__(self, master, command, label):
        self.master = master
        self.command = command
        TK.Frame.__init__(self, self.master, bg = tk_config['bg'], bd = 2)
        # TK.Frame.__init__(self, master = self.master)


        self.label = TK.Label(master = self, text = label \
                        , **tk_config \
                        )
        self.label.pack(side = TK.TOP, expand = False)


        self.slider = TK.Scale( \
                          master = self \
                        , from_ = +NP.pi \
                        , to = -NP.pi \
                        , showvalue = True \
                        , resolution = 1/1000. \
                        , tickinterval = None \
                        , orient = TK.VERTICAL \
                        , command = self.command \
                        , **tk_config \
                        ) # 

        self.slider.pack(side = TK.BOTTOM, fill = TK.Y, expand = True)


        # TK.Separator(self, orient="horizontal").pack(side = TK.BOTTOM)

    def Set(self, *args, **kwargs):
        self.slider.set(*args, **kwargs)

    def Get(self, *args, **kwargs):
        return self.slider.get(*args, **kwargs)




indicators_color = '0.7'

def FullDespine(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(bottom = False)
    ax.tick_params(left = False)
    ax.set_xticks([])
    ax.set_yticks([])


    
def AddBaseline(ax, arrow_width = 0.16):
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axhline( 0 \
    #         , color = indicators_color \
    #         , ls = '-' \
    #         , lw = 1 \
    #     )
    arrow_kwargs = dict(   linewidth = 0.5 \
                         , length_includes_head = True \
                         , head_width = arrow_width \
                         , head_length = 0.01 \
                         , color = indicators_color \
                         , zorder = -1 \
                         , clip_on = False \
                       )
    ax.arrow(  0. \
             , 0. \
             , 1. \
             , 0. \
             , **arrow_kwargs \
            )
    
    
def AddCrosses(ax, limit = None):

    FullDespine(ax)
    
    if limit is None:
        limit = NP.max(NP.abs(NP.concatenate([ax.get_xlim(), ax.get_ylim()])))

    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    
    arrow_kwargs = dict(   linewidth = 0.5 \
                         , length_includes_head = False \
                         , head_width = 0.03 \
                         , head_length = 0.05 \
                         , color = indicators_color \
                         , zorder = -1 \
                         , clip_on = False \
                       )
    ax.arrow(  -limit \
             , 0 \
             , 2*limit \
             , 0 \
             , **arrow_kwargs \
            )
    # tick marks
    tickmark_kwargs = dict( ls = '-' \
                           , lw = 0.5 \
                           , color = indicators_color \
                           , zorder = -1 \
                          )
    ticksep = 0.2
    ticklength = .01
    for tick in NP.arange(0,limit,ticksep/2):
        ax.plot([tick, tick], [-ticklength, +ticklength] \
               , **tickmark_kwargs)
        ax.plot([-tick, -tick], [-ticklength, +ticklength] \
               , **tickmark_kwargs)
        ax.plot([-ticklength, +ticklength], [tick, tick] \
               , **tickmark_kwargs)
        ax.plot([-ticklength, +ticklength], [-tick, -tick] \
               , **tickmark_kwargs)
        
    
    
    
    ax.text(  limit, -.05*limit, r'$\Re$' \
            , va = 'top', ha = 'left' \
            , fontsize = 14 \
            , color = indicators_color \
           )
    
    ax.arrow(  0 \
             , -limit \
             , 0 \
             , 2*limit \
             , **arrow_kwargs \
            )
    ax.text(  .05*limit, limit, r'$\Im$' \
            , va = 'bottom', ha = 'left' \
            , fontsize = 14 \
            , color = indicators_color \
           )



# https://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively
class DraggablePoint:
    lock = None #only one can be animated at a time
    def __init__(self, point, master, fsd, coeff):
        self.point = point
        self.press = None
        self.master = master
        self.fsd = fsd
        self.coeff = coeff
        self.background = self.master.canvas.copy_from_bbox(self.master.frequency_domain.bbox)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        # self.background = canvas.copy_from_bbox(self.point.axes.bbox)
        # self.background = self.master.canvas.copy_from_bbox(self.master.frequency_domain.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        press_center, xpress, ypress = self.press
        # dx = event.xdata - xpress
        # dy = event.ydata - ypress
        # self.point.center = (press_center[0]+dx, press_center[1]+dy)
        self.point.center = (event.xdata, event.ydata)

        self.fsd[self.coeff, :] = self.point.center

        # self.master.SliderValuesFromFSD()
        self.master.Update(quick = True)
        # self.redraw()
        # self.master.canvas.blit(self.master.frequency_domain.bbox)


    def AdjustFromFSD(self):
        self.point.center = self.fsd[self.coeff, :]
        self.redraw()


    def redraw(self):
        # canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        # canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        self.master.canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        # self.background = None

        self.master.SliderValuesFromFSD()
        self.master.Update()

        # redraw the full figure
        # self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


def MakeSignalFigure():
    # a standard FSD figure that will be repeatedly used below
    fig = MPP.figure(figsize = (24/2.54, 8/2.54), dpi=150)
    fig.subplots_adjust( \
                              top    = 0.90 \
                            , right  = 0.92 \
                            , bottom = 0.16 \
                            , left   = 0.08 \
                            , wspace = 0.08 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )
    rows = [12]
    cols = [5,2]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )
    time_domain = fig.add_subplot(gs[0]) # , aspect = 1/4
    time_domain.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    time_domain.set_xlabel(r'stride cycle')
    time_domain.set_ylabel(r'angle (rad)')
    time_domain.set_title('time domain')
    time_domain.set_xlim([0.,1.])


    frequency_domain = fig.add_subplot(gs[1], aspect = 'equal')
    frequency_domain.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    frequency_domain.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    # frequency_domain.set_xlabel(r'$\Re(c_n)$')
    # frequency_domain.set_ylabel(r'$\Im(c_n)$')
    # frequency_domain.set_title('frequency domain')

    frequency_domain.yaxis.tick_right()
    frequency_domain.yaxis.set_label_position("right")

    return fig, time_domain, frequency_domain



class FourierDemo(object):
    def __init__(self):

        ### master and figure
        self.master = TK.Tk()
        # self.master.attributes("-fullscreen", True)
        self.master.wm_title("Fourier Series Decomposition")

        self.fig, self.time_domain, self.frequency_domain = MakeSignalFigure()
        self.frequency_domain.set_xlim([-.7, .7])
        self.frequency_domain.set_ylim([-.7, .7])
        MPP.annotate(    r'time domain' \
                            , fontsize = 12 \
                            , xy = (0.35, 0.95) \
                            , xytext = (0.35, 0.95) \
                            , xycoords = 'figure fraction' \
                            , textcoords = 'figure fraction' \
                            , va = 'bottom', ha = 'center' \
                            )

        MPP.annotate(    r'frequency domain' \
                            , fontsize = 12 \
                            , xy = (0.8, 0.95) \
                            , xytext = (0.8, 0.95) \
                            , xycoords = 'figure fraction' \
                            , textcoords = 'figure fraction' \
                            , va = 'bottom', ha = 'center' \
                            )

        AddCrosses(self.frequency_domain)

        
        ### plot area
        self.upper_frame = TK.Frame(master = self.master)

        self.spacing = 20

        # ylim slider
        ylim_frame = TK.Frame(master = self.upper_frame, bg = tk_config['bg'], bd = self.spacing)
        self.ylim = 2.
        self.ylim_slider = TK.Scale( \
                          master = ylim_frame \
                        , from_ = +10. \
                        , to = 0.1 \
                        , showvalue = False \
                        , resolution = 1/10. \
                        , tickinterval = None \
                        , orient = TK.VERTICAL \
                        , command = self.SetYLim \
                        , **tk_config \
                        # , bd = self.spacing
                        ) # 
        self.ylim_slider.set(self.ylim)
        self.ylim_slider.pack( side = TK.LEFT, fill = TK.Y, expand = False )

        ## canvas
        self.canvas_frame = TK.Frame(master = self.upper_frame, bg = tk_config['bg'])
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.canvas_frame)  # A tk.DrawingArea.
        self.canvas.draw()

        ## affine components
        phaseshift_frame = TK.Frame(master = self.upper_frame, bg = tk_config['bg'], bd = self.spacing)

        TK.Label(         master = phaseshift_frame \
                        , text = 'phase' \
                        , **tk_config \
                        ).pack(side = TK.LEFT, expand = False)

        self.phase_slider = TK.Scale( \
                          master = phaseshift_frame \
                        , from_ = 0. \
                        , to = 1. \
                        , showvalue = False \
                        , resolution = 1/101. \
                        , tickinterval = None \
                        , orient = TK.HORIZONTAL \
                        , command = self.AdjustPhase \
                        # , length = 1200 \
                        , **tk_config \
                        ) # 
        self.phase_slider.bind("<ButtonRelease-1>", self.SyncSliders)
        self.phase_slider.pack(side = TK.BOTTOM, fill = TK.X, expand = False )

        amp_frame = TK.Frame(master = self.canvas_frame, bg = tk_config['bg'], bd = self.spacing)

        TK.Label(         master = amp_frame \
                        , text = 'amplitude' \
                        , **tk_config \
                        ).pack(side = TK.TOP, expand = False)

        self.amp_slider = TK.Scale( \
                          master = amp_frame \
                        , from_ = 2. \
                        , to = 0.01 \
                        , showvalue = False \
                        , resolution = 1/101. \
                        , tickinterval = None \
                        , orient = TK.VERTICAL \
                        , command = self.AdjustAmplitude \
                        # , length = 1200 \
                        , **tk_config \
                        ) # 
        self.amp_slider.bind("<ButtonRelease-1>", self.SyncSliders)

        self.amp_slider.pack(side = TK.TOP, fill = TK.Y, expand = True )

        TK.Button( \
                        master = amp_frame \
                        , text = 'flip' \
                        , command = self.FlipAmplitude \
                        , **tk_config \
                        ).pack(side = TK.BOTTOM, fill = TK.X, expand = False )


        ## packing upper part
        phaseshift_frame.pack(side = TK.BOTTOM, fill = TK.X, expand = False )
        amp_frame.pack(side = TK.RIGHT, fill = TK.Y, expand = False)
        ylim_frame.pack( side = TK.LEFT, fill = TK.Y, expand = False )
        self.canvas_frame.pack( side = TK.LEFT, fill = TK.BOTH, expand = True )
        self.upper_frame.pack( side = TK.TOP, fill = TK.BOTH, expand = True )
        canvas_widget = self.canvas.get_tk_widget()
        # print(dir(canvas_widget))#.set(bg = tk_config['bg'])
        canvas_widget.pack( side = TK.RIGHT, fill = TK.BOTH, expand = True)
        amp_frame.pack( side = TK.RIGHT, fill = TK.Y, expand = False )


        ### control area
        self.coeffs_frame = TK.Frame(master = self.master, bg = tk_config['bg'])
        self.coeffs_frame.pack( side = TK.TOP, fill = TK.X, expand = False )        # for slid in slid_coefficients:
        self.slider = {}
        for slid in slid_coefficients:
            self.slider[slid] = MuFuSlider(master = self.coeffs_frame, command = self.UpdateSliders, label = coefficient_labels[slid])

            self.slider[slid].slider.bind("<ButtonRelease-1>", self.SyncSliders)
            self.slider[slid].pack( side = TK.LEFT, fill = TK.Y, expand = False )
            self.slider[slid].Set(0.)

        self.center_button = TK.Button( \
                            master = self.slider[0] \
                            , text = 'center' \
                            , command = self.Center \
                            , bg = tk_config['bg'] \
                            , fg = tk_config['fg'] \
                            , font = tk_config['font'] \
                            )
        self.center_button.pack(side = TK.TOP)
       


        self.fsd = FKT.FourierSignal.FromComplexVector(self.GetCoefficients())
        self.draggables = []
        coords = self.fsd[1:, ['re', 'im']].values
        self.circles = [ MP.patches.Circle((coords[c, 0], coords[c, 1]), 0.01 + 0.03*((coords.shape[0]-c)/coords.shape[0]), fc='w', alpha=0.75)
                         for c in range(coords.shape[0]) ]

        for c_nr, circ in enumerate(self.circles):
            self.frequency_domain.add_patch(circ)
            dr = DraggablePoint(circ, master = self, fsd = self.fsd, coeff = c_nr + 1)
            # dr.background = self.canvas.copy_from_bbox(self.frequency_domain.bbox)
            dr.connect()
            self.draggables.append(dr)



        self.frq_connector = self.frequency_domain.plot( \
                                  self.fsd[1:, 're'] \
                                , self.fsd[1:, 'im'] \
                                , lw = 1 \
                                , ls = ':' \
                                , color = 'w' \
                                , alpha = 1. \
                                , zorder = 50 \
                                # , **common_plotargs \
                                )[0]

        # toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        # toolbar.update()
        # canvas_widget.pack(side = TK.BOTTOM, fill = TK.BOTH, expand = True)


        def on_key_press(event):
            if event.key == '0':
                self.Zero()
            elif event.key == '1':
                self.RefReal()
            elif event.key == '2':
                self.RefMean()
            elif event.key == '3':
                self.RefElbow()
            elif event.key == '4':
                self.RefCarpus()
            # elif event.key == '5':
            #     self.RefDZG()

            elif event.key == 'r':
                self.MakeRef()
            elif event.key == 's':
                self.Superimpose()

            elif event.key == 'd':
                self.SaveFigure()


            elif event.key == '/':
                self.FlipAmplitude()

            elif event.key == 'i':
                self.imgcheck.set(not self.imgcheck.get())
                self.Update()


            elif event.key in affine_shortcuts.keys():
                self.AffineChange(event.key)


            # print("you pressed {}".format(event.key))
            key_press_handler(event, self.canvas)#, toolbar)


        self.canvas.mpl_connect("key_press_event", on_key_press)



        self.menu_frame = TK.Frame(master = self.master, bg = tk_config['bg'])
        self.menu_frame.pack( side = TK.BOTTOM, fill = TK.X, expand = False )

        self.buttons = [
                      ['quit', self.Quit] \
                    , ['zero', self.Zero] \
                    , ['real', self.RefReal] \
                    # , ['dzg', self.RefDZG] \
                    , ['carpus', self.RefCarpus] \
                    , ['elbow', self.RefElbow] \
                    , ['mean', self.RefMean] \
                     ]
        self.menu = [ TK.Button( \
                            master = self.menu_frame \
                            , text = button \
                            , command = command \
                            , bg = tk_config['bg'] \
                            , fg = tk_config['fg'] \
                            , font = tk_config['font'] \
                            ) \
                      for button, command in self.buttons ]

        self.reference_fsd = None
        self.refcheck = TK.IntVar()
        self.refplot_handle = self.frequency_domain.plot( \
                                  NP.zeros((fsd_order+1,)) \
                                , NP.zeros((fsd_order+1,)) \
                                , lw = 1 \
                                , ls = '--' \
                                , color = (0.5,0.9,0.5) \
                                , alpha = 1. \
                                , zorder = 60 \
                                # , **common_plotargs \
                                )[0] 


        self.stringed = TK.IntVar()
        # img = MP.image.imread('dzg_logo.png')
        # self.imgplot_handle = self.frequency_domain.imshow(img, cmap="gray", extent = (-0.30, 0.70, -0.48, 0.27))
        # self.imgplot_handle.set_visible(self.imgcheck.get())

        self.menu = self.menu + \
                    [ TK.Checkbutton( \
                            master = self.menu_frame \
                            , variable = self.refcheck \
                            , text = 'show ref' \
                            , command = self.Update \
                            , bg = 'grey' \
                            , fg = 'black' \
                            , highlightcolor = 'black' \
                            ) \
                    , TK.Checkbutton( \
                            master = self.menu_frame \
                            , variable = self.stringed \
                            , text = 'string' \
                            , command = self.Update \
                            , bg = 'grey' \
                            , fg = 'black' \
                            , highlightcolor = 'black' \
                            ) \
                      ]

        self.superimposition_label = TK.StringVar()
        self.menu = self.menu + [ \
                          TK.Button( \
                              master = self.menu_frame \
                            , text = 'set ref' \
                            , command = self.MakeRef \
                            , bg = tk_config['bg'] \
                            , fg = tk_config['fg'] \
                            , font = tk_config['font'] \
                            )
                        , TK.Button( \
                              master = self.menu_frame \
                            , text = 'superimpose' \
                            , command = self.Superimpose \
                            , bg = tk_config['bg'] \
                            , fg = tk_config['fg'] \
                            , font = tk_config['font'] \
                            ) \
                        , TK.Label( master = self.menu_frame \
                            , textvariable = self.superimposition_label \
                            , bg = tk_config['bg'] \
                            , fg = tk_config['fg'] \
                            , font = tk_config['font'] \
                            )
                    ]


        for element in self.menu:
            element.pack(side = TK.RIGHT)

        # If you put self.master.destroy() here, it will cause an error if the window is
        # closed with the window manager.

        self.Update()


    def GetCoefficients(self):
        coefficients = [self.slider[slid].Get() for slid in slid_coefficients]
        coefficients = NP.concatenate([NP.array([[coefficients[0], 0]]), NP.array(coefficients[1:]).reshape((-1,2))], axis = 0)
        return FKT.MakeComplex(coefficients)


    def SetCoefficients(self, vector):
        self.fsd._c.loc[:, :] = vector

        self.SliderValuesFromFSD()
        self.DraggablesFromFSD()


    def RefCarpus(self):
        self.SetReference('carpus')
    # def RefDZG(self):
    #     self.SetReference('dzg')
    def RefElbow(self):
        self.SetReference('elbow')
    def RefMean(self):
        self.SetReference('mean')
    def RefReal(self):
        self.SetReference('real')


    def SetReference(self, ref):
        self.SetCoefficients(NP.array(references[ref]).reshape((-1,2)))
        self.SyncSliders(None)


    def Zero(self):
        self.SetCoefficients(NP.zeros(self.fsd._c.shape))
        self.Update()

    def GetPhase(self):
        try:
            phase = self.fsd.GetMainPhase()
        except RecursionError as e:
            return 0

        if NP.isnan(phase):
            return 0
        else:
            return phase

    def AdjustPhase(self, _):
        phaseshift = self.phase_slider.get() - self.GetPhase()
        FKT.Rotate(self.fsd, phaseshift)

        self.Update()

    def GetAmp(self):
        amp = NP.sum(self.fsd.GetAmplitudes())
        if NP.isnan(amp):
            return 0
        else:
            return amp

    def AdjustAmplitude(self, _):
        slider = self.amp_slider.get()
        amp = NP.abs(self.GetAmp())
        if amp == 0:
            return
        FKT.Scale(self.fsd, NP.sign(slider) * NP.abs(slider) / amp)

        self.Update()

    def AffineChange(self, key):
        fcn, param = affine_shortcuts[key]
        fcn(self.fsd, param)

        self.Update()





    def UpdateSliders(self, val):
        # get values from all sliders and adjust FSD & draggables accordingly
        # print (val)
        self.fsd[0, 're'] = self.slider[0].Get()

        for slid in slid_coefficients[1:]:
            self.fsd._c.iloc[(slid-1)//2+1, (slid-1)%2] = self.slider[slid].Get()

        # self.SliderValuesFromFSD()
        # self.DraggablesFromFSD() # !!! circular
        self.Update()


    def SliderValuesFromFSD(self):
        # set all sliders to values from FSD

        self.slider[0].Set(self.fsd[0, 're'])
        for slid in slid_coefficients[1:]:
            # print (slid, (slid-1)//2+1, (slid-1)%2)
            self.slider[slid].Set(self.fsd._c.iloc[(slid-1)//2+1, (slid-1)%2])



    def DraggablesFromFSD(self):
        for drp in self.draggables:
            drp.AdjustFromFSD()

        # self.canvas.blit(self.frequency_domain.bbox)

    def SyncSliders(self, _):

        self.SliderValuesFromFSD()
        self.DraggablesFromFSD()
        self.Update()


    def Update(self, quick = False):
        # 'getdouble', 'getint', 'getvar'

        if quick:
            self.frq_connector.set_data( self.fsd[1:, 're'] \
                                   , self.fsd[1:, 'im'])
            self.canvas.draw()
            return

        self.phase_slider.set(self.GetPhase())
        self.amp_slider.set(self.GetAmp())


        self.time_domain.cla()
        # self.frequency_domain.cla()

        AddBaseline(self.time_domain, arrow_width = 0.03*self.ylim)

        self.time_domain.set_ylim([-self.ylim, self.ylim])
        self.time_domain.set_xlim([0., 1.])

        self.time_domain.set_xlabel('time (period)', color = 'w')


        time = NP.linspace(0., 1., 101, endpoint = True)
        # print (fsd.Reconstruct(time))
        signal = self.fsd.Reconstruct(x_reco = time, period = 1.)
        y = signal - signal[0] * (1. if self.stringed.get() else 0.)
        self.time_domain.plot(time, y, 'w-', linewidth = 2)

        # self.fsd.GetCentroid()
        self.time_domain.axhline(0, color = '0.7', ls = '--', lw = 1)

        # for drp in self.draggables:
        #     drp.redraw()
        self.frq_connector.set_data( self.fsd[1:, 're'] \
                                   , self.fsd[1:, 'im'])
        # self.frequency_domain.set_xlim([-self.ylim/7, self.ylim/7])
        # self.frequency_domain.set_ylim([-self.ylim/7, self.ylim/7])

        # self.canvas.blit(self.frequency_domain.bbox)

        self.PlotReferences()

        if self.stringed.get():
            lim = NP.max(NP.abs([NP.max(y), NP.min(y), 0.01]))
            self.time_domain.set_ylim([-lim, lim])

        self.canvas.draw()


    def Center(self):
        self.fsd[0,'re'] = 0
        self.SyncSliders(None)

    def FlipAmplitude(self):
        FKT.Scale(self.fsd, -1)
        self.SyncSliders(None)



    def SetYLim(self, _):
        self.ylim = self.ylim_slider.get()
        self.Update()

    def PlotReferences(self):

        # self.imgplot_handle.set_visible(self.imgcheck.get())

        if self.reference_fsd is None:
            return

        if self.refcheck.get():
            t = NP.linspace(0., 1., 101, endpoint = True)
            y = self.reference_fsd.Reconstruct(x_reco = t, period = 1.)
            c = self.reference_fsd[1:, ['re','im']].values


            self.time_domain.plot(t, y, color = (0.5,0.9,0.5), lw = 1, ls = '--')

            self.refplot_handle.set_data(c[:,0], c[:,1])
            self.refplot_handle.set_visible(True)
        else:
            self.refplot_handle.set_visible(False)



    def MakeRef(self):
        self.reference_fsd = self.fsd.Copy()
        self.refcheck.set(True)
        self.PlotReferences()
        self.Update()

    def Superimpose(self):
        if self.reference_fsd is None:
            return
        if NP.all(self.reference_fsd._c.values.ravel() == 0):
            return

        if NP.sum(self.fsd.GetAmplitudes()) == 0:
            return
            
        procrustes = FKT.Procrustes(self.fsd, self.reference_fsd, compute_distance = True, label = None)
        procrustes.ApplyTo(self.fsd)
        
        indicator_text = d_y = "d_c0 = {translation:.2f}\td_A = {scaling:.2f}\td_phi = {rotation:.2f} \t residual {residual_euclid:.3f} \t".format( \
                                 **procrustes \
                             )

        self.superimposition_label.set(indicator_text)

        self.SyncSliders(None)


    def SaveFigure(self):
        self.fig.savefig('export.svg', dpi = 300, transparent = True)


    def Quit(self):
        self.master.quit()     # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate



if __name__ == "__main__":

    demo = FourierDemo()
    tkinter.mainloop()
