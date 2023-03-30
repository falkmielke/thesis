#!/usr/bin/env python3

"""
common plotting functions
"""

__author__      = "Falk Mielke"
__date__        = 20170508


#________________________________________________________________________________
### Prerequisites
import numpy as NP
import matplotlib as MP
import matplotlib.pyplot as PLT




"""
#######################################################################
### Plot Preparation                                                ###
#######################################################################
"""
# cp -R /usr/share/texmf-dist/fonts/opentype /usr/lib/python3.6/site-packages/matplotlib/mpl-data/fonts
# rm ~/.cache/matplotlib/fontList.py3k.cache

the_font = {  \
        # It's really sans-serif, but using it doesn't override \sffamily, so we tell Matplotlib
        # to use the "serif" font because then Matplotlib won't ask for any special families.
         # 'family': 'serif' \
        # , 'serif': 'Iwona' \
        'family': 'sans-serif'
        , 'sans-serif': 'DejaVu Sans'
        , 'size': 10#*1.27 \
    }

### http://blog.olgabotvinnik.com/blog/2012/11/15/2012-11-15-how-to-set-helvetica-as-the-default-sans-serif-font-in/
# /usr/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/


# MP.rcParams['font.sans-serif'] = 'Iwona'
# the_font_name = 'Iwona'#'Avant Garde'#'DejaVu Sans'#'Iwona'
# http://matplotlib.org/users/usetex.html

# https://stackoverflow.com/questions/29187618/matplotlib-and-latex-beamer-correct-size
# not useful since I don't know the column pixel width in the journal


def MakeFigure(rows = [1], cols = [1], dimensions = [11,7], style = None, dpi = 300):

    PreparePlot()

    if style is not None:
        # print(PLT.style.available)
        PLT.style.use(style) #'seaborn-paper'
    # custom styles can go to ~/.config/matplotlib/stylelib
    # originals in /usr/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib
    



    
# set figure size with correct font size
    # to get to centimeters, the value is converted to inch (/2.54) 
    #                        and multiplied with fudge factor (*1.25).
    # The image then has to be scaled down from 125% to 100% to remove the fudge scaling.
    cm = 1./2.54
    figwidth  = dimensions[0] * cm
    figheight = dimensions[1] * cm

# define figure
    # columnwidth = 455.24411 # from latex \showthe\columnwidth
    fig = PLT.figure( \
                              figsize = (figwidth, figheight) \
                            , facecolor = None \
                            , dpi = dpi \
                            )
    # PLT.ion() # "interactive mode". Might e useful here, but i don't know. Try to turn it off later.

# define axis spacing
    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.28 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )

# # a supertitle for the figure; relevant if multiple subplots
    # fig.suptitle( r"Falk's Reaktionszeiten, 17.-22. Oktober 2015" )

# define subplots
    gs = MP.gridspec.GridSpec( \
                                  len(rows) \
                                , len(cols) \
                                , height_ratios = rows \
                                , width_ratios = cols \
                                )


    # axarr = []
    # axarr.append( fig.add_subplot(gs[0,0]) )
    # axarr.append( fig.add_subplot(gs[2,0], sharex = axarr[0], sharey = axarr[0]) )


    return fig, gs



def PolishAx(ax):
# axis cosmetics
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.tick_params(top = False)
    ax.tick_params(right = False)
    # ax.tick_params(left=False)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)


def FullDespine(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(bottom = False)
    ax.tick_params(left = False)
    ax.set_xticks([])
    ax.set_yticks([])



def PolishPlot():
    pass

def PreparePlot():
    # select some default rc parameters
    MP.rcParams['text.usetex'] = True
    PLT.rc('font',**the_font)
    # Tell Matplotlib how to ask TeX for this font.
    # MP.texmanager.TexManager.font_info['iwona'] = ('iwona', r'\usepackage[light,math]{iwona}')

    MP.rcParams['text.latex.preamble'] = " ".join([\
                  r'\usepackage{upgreek}'
                , r'\usepackage{cmbright}'
                , r'\usepackage{sansmath}'
                ])

    MP.rcParams['pdf.fonttype'] = 42 # will make output TrueType (whatever that means)



def AddPanelLetter(letter):
    PLT.annotate(r'\textbf{%s}' % (letter) \
            , xy = (0,0) \
            , xytext = (0.005, 0.99) \
            , textcoords = 'figure fraction' \
            , va = 'top', ha = 'left' \
            , fontsize = 12 \
            , arrowprops = dict(facecolor = 'none', edgecolor = 'none') \
            )






def InvertColor(fig, axarr):

    ColorInvert = lambda c: tuple(1.0-NP.array(MP.colors.ColorConverter().to_rgb(c)))
    ColorInvertAlpha = lambda c: tuple(NP.abs( NP.array((1.0, 1.0, 1.0, 0.0))-NP.array(MP.colors.ColorConverter().to_rgba(c)) ))

    # texts, lines and annotation text
    for plotobj in fig.findobj(MP.text.Text) \
                + fig.findobj(MP.lines.Line2D) \
                + fig.findobj(MP.text.Annotation) \
                :
        PLT.setp(plotobj, color = ColorInvert(PLT.getp(plotobj, 'color')))

    for plotobj in fig.findobj(MP.lines.Line2D):
        plotobj.set_markeredgecolor( ColorInvert(plotobj.get_markeredgecolor()))

    # rectangles and spines
    for plotobj in fig.findobj(MP.patches.Rectangle) \
                + fig.findobj(MP.spines.Spine) \
                :
        PLT.setp(plotobj, facecolor = ColorInvert(PLT.getp(plotobj, 'facecolor')) \
                        , edgecolor = ColorInvert(PLT.getp(plotobj, 'edgecolor')))

    # no idea
    for plotobj in fig.findobj(MP.collections.LineCollection) \
        :
        PLT.setp(plotobj, color = ColorInvertAlpha(PLT.getp(plotobj, 'color')[0]))

    # the scatter dots
    for plotobj in fig.findobj(MP.collections.PathCollection) \
                :
        PLT.setp(plotobj, facecolor = ColorInvertAlpha(PLT.getp(plotobj, 'facecolor')[0]) \
                        , edgecolor = ColorInvertAlpha(PLT.getp(plotobj, 'edgecolor')[0]))



    for plotobj in fig.findobj(MP.collections.PolyCollection) \
                :
        PLT.setp(plotobj, facecolor = ColorInvertAlpha(PLT.getp(plotobj, 'facecolor')[0]) \
                        , edgecolor = ColorInvertAlpha(PLT.getp(plotobj, 'edgecolor')[0]))

    for ax in axarr:
        # does not work?! 
        for plotobj in \
                  ax.get_xticklines() \
                + ax.get_yticklines() \
                :
            PLT.setp(plotobj, color = ColorInvert(PLT.getp(plotobj, 'color')))

        ax.get_xaxis().set_tick_params(which='major', reset = True, direction='out', colors = '1', top = False)
        ax.get_yaxis().set_tick_params(which='major', reset = True, direction='out', colors = '1', right = False)
        # for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        #         # tick.label.set_fontsize(14) 
        #         # specify integer or one of preset strings, e.g.
        #         #tick.label.set_fontsize('x-small') 
        #         # tick.label.set_rotation('vertical')
        #         tick.label.set_colors('1')
                # tick.label.set_color('1')




################################
### Iwona installation:

## copy Iwona from latex path:
# /usr/share/texmf-dist/fonts/opentype/nowacki/iwona/

# run the following script with fontforge:
"""
#!/usr/local/bin/fontforge
# Quick and dirty hack: converts a font to truetype (.ttf)
# http://www.stuermer.ch/blog/convert-otf-to-ttf-font-on-ubuntu.html

# install fontforge
# for i in *.otf; do fontforge -script otf2ttf.sh $i; done

Print("Opening "+$1);
Open($1);
Print("Saving "+$1:r+".ttf");
Generate($1:r+".ttf");
Quit(0); 
"""

## ttf version needs to be copied to
# /usr/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf

## delete matplotlib cache:
# rm ~/.cache/matplotlib/fontList.py3k.cache
#################################


#################################
# Three Ways to use Iwona!
#################################
"""
(in order of execution priority)
(i)   matplotlibrc file
(ii)  custom styles
(iii) manual RC parameter configuration

"""

#######################
### (i) matplotlibrc
# find the default local matplotlibrc file:
# (root)# find / -name "*matplotlibrc*" 2> /dev/null
# for example here: /usr/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
# copy it here:
# ~/.config/matplotlib/matplotlibrc
#
# open it in text editor and manipulate the very default configuration.


def FontTesting():
    # Example data
    t = NP.arange(0.0, 1.0 + 0.01, 0.01)
    s = NP.cos(4 * NP.pi * t) + 2


    #######################
    ### (ii) Styles!!
    # custom styles can go to ~/.config/matplotlib/stylelib
    # originals in /usr/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib

    # print(PLT.style.available)
    # PLT.style.use('seaborn-paper')
    # PLT.style.use(['dark_background', 'seaborn-paper']) # combine styles

    # PLT.style.use('thesis')
    ## ~/.config/matplotlib/stylelib/thesis.mplstyle contains:
    """
    font.family : 'Iwona'
    font.size : 10
    """


    ## maybe useful:
    # print(MP.get_configdir())

    #######################
    ### (iii) Manual rc configuration
    MP.rcParams['text.usetex'] = False ## !!! DO NOT DO TEX !!! otherwise the font will be irrevertibly reset.
    MP.rcParams['font.size'] = 10
    MP.rcParams['font.family'] = 'Iwona'
    MP.rcParams['pdf.fonttype'] = 42 # will make output TrueType (whatever that means)
    ## OR:
    PLT.rc('font',**{'family': 'Iwona', 'size': 10})


    #######################
    ### check final parameters:
    # print (MP.rcParams)


    ## find fonts:
    # (A) by file name
    fp = MP.font_manager.FontProperties( \
                                          fname = '/usr/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/IwonaLight.ttf'
                                        )
    # (B) by a selection like rc parameter config
    # fp = MP.font_manager.FontProperties( \
    #                                       family = 'Iwona'  \
    #                                     , weight = 'normal' \
    #                                     , size   = 10 \
    #                                     )
    print(fp)
    print( MP.font_manager.findfont(fp) )

    ##############################
    ### The Plot ###
# set figure size
    # to get to centimeters, the value is converted to inch (/2.54) 
    #                        and multiplied with empirical factor (*1.25).
    cm = 1.25/2.54 # inches and mysterious scaling.
    figwidth  = 12 * cm
    figheight = 8 * cm

# define figure
    fig = PLT.figure( \
                              figsize = (figwidth, figheight) \
                            , facecolor = None \
                            , dpi = 150 \
                            )
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(t, s)


    ## some LaTeX commands work, others not!
    # see http://matplotlib.org/users/mathtext.html#mathtext-tutorial
    ax.set_xlabel(r'time (s)', fontweight = 'bold')
    ax.set_ylabel(r'il voltage (mV) $\mu$')
    ax.set_title(r"mathtext is Number "
              r"$\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
              , color='gray')

    fig.subplots_adjust(top=0.9)


    # save and show
    # fig.savefig('test_latexplots.pdf', dpi = 150, transparent = 'true')
    PLT.show()




if __name__ == "__main__":
    # example procedure:
    fig, gs = MakeFigure()
    ax = fig.add_subplot(1,1,1)

    # from http://matplotlib.org/users/usetex.html#usetex-tutorial
    t = NP.arange(0.0, 1.0 + 0.01, 0.01)
    s = NP.cos(4 * NP.pi * t) + 2

    ax.plot(t, s)

    ax.set_xlabel(r'time (s)', fontweight = 'bold')
    ax.set_ylabel(r'il \textbf{voltage} (mV) $\mathbf{\upmu}$')
    ax.set_title(r"mathtext is Number "
              r"$\sum\limits_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
              , color='gray')

    fig.subplots_adjust(top=0.85)


    PLT.show()
