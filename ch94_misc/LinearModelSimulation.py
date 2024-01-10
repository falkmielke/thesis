#!/usr/bin/env python3

import numpy as NP
import scipy.stats as STATS
import matplotlib as MP
import matplotlib.pyplot as PLT
import seaborn as SB
import pymc as PM
import pytensor as PP
import pytensor.tensor as PT

dpi = 300

class Settings(dict):
    def __init__(self, **kwargs):
        self.SetDefaults()

        for key, value in kwargs.items():
            self[key] = value

    def SetDefaults(self):
        self['n_slopes'] = 1
        self['intercept'] = 0.
        self['noise_level'] = 0.01
        self['n_observations'] = 2**10
        self['x_range'] = [-0.4, 0.4]
        #self['slopes'] = [0.5, -0.3, 0.1]



### (i) simulate data
def SimulateData(settings = None):
    NP.random.seed(42)
    if settings is None:
        settings = Settings() # use default settings

    # if there were no slopes...
    if settings['slopes'] is None:
        raise IOError("no slopes!")

    # number of slopes
    settings['n_slopes'] = len(settings['slopes'])

    # dimensionality required for matrix operation
    settings['slopes'] = NP.array(settings['slopes']).reshape(-1,1)

    # set number of observations
    settings['n_observations'] = int(2**9)
    settings['n_samples'] = settings['n_observations']
    # settings['n_samples'] = int(NP.power(settings['n_observations'], 1/settings['n_slopes'])//1)
    # settings['n_observations'] = int(settings['n_samples']**settings['n_slopes'])

    # empty output
    y = NP.zeros((settings['n_observations'], 1))

    # start with the intercept
    y += settings['intercept']

    # the input space
    x = NP.stack([NP.random.uniform(*settings['x_range'], settings['n_observations']) for i in range(settings['n_slopes'])], axis = 1)
    # x = NP.array(NP.meshgrid(*[NP.linspace(settings['x_range'][0], settings['x_range'][1], settings['n_samples'], endpoint = True) \
    #                            for _ in range(settings['n_slopes'])])).T.reshape(-1,settings['n_slopes'])

    # add up the slope effects
    y += NP.dot(x, settings['slopes'])

    # add noise
    x += NP.random.normal(0., settings['noise_level'], x.shape)
    y += NP.random.normal(0., settings['noise_level'], (settings['n_observations'], 1))

    return x, y, settings


class Simulation(object):

    def __init__(self, **kwargs):
        self.x, self.y, self.settings = SimulateData(Settings(**kwargs))

        self.CreateModel()

        self.trace = None
        self.x_prediction_raw = None
        self.prediction = None


    def CreateModel(self):
        with PM.Model() as self.model:

            idata = PM.Data(f'idata', NP.ones((self.settings['n_observations'], 1)), mutable = True)
            a = PM.Normal('intercept', shape = (1, 1))

            data = PM.Data(f'xdata', self.x, mutable = True)
            b = PM.Normal(f'slopes', shape = (self.settings['n_slopes'], 1))

            estimator = PT.dot(idata, a) + PT.dot(data, b)
            # estimator = a + PT.dot(data, b)

            edata = PM.Data(f'edata', NP.ones((self.settings['n_observations'], 1)), mutable = True)
            e = PM.HalfNormal('noise', shape = (1, 1))
            residual = PT.dot(edata, e)

            posterior = PM.Normal('y', mu = estimator, sigma = residual, observed = self.y)


    def FitModel(self):
        with self.model:
            self.trace = PM.sample(2**10 \
                            , progressbar = False \
                              )

    def PreparePredictionData(self):

        n_predictions = self.settings['n_observations'] # must be! (pymc limitation)
        # print (n_samples, n_samples**n_slopes, n_predictions)

        if False:
             grid = NP.array(NP.meshgrid(*[NP.linspace(*self.settings['x_range'], self.settings['n_samples'], endpoint = True) for _ in range(self.settings['n_slopes'])])).T.reshape(-1,self.settings['n_slopes'])
             x_predict = grid
        else:
             x_predict = NP.stack([NP.random.uniform(*self.settings['x_range'], n_predictions) for i in range(self.settings['n_slopes'])], axis = 1)
        # x_predict = NP.stack([NP.linspace(0., 1., n_predictions) for i in range(n_slopes)], axis = 1)
        return x_predict


    def SetData(self, x_pred = None):

        if x_pred is None:
            x_pred = self.PreparePredictionData()

        self.x_prediction_raw = x_pred.copy()

        i_predict = NP.ones((self.settings['n_observations'], 1))
        e_predict = NP.ones((self.settings['n_observations'], 1))
        # print (x_predict.shape, i_predict.shape, e_predict.shape)
        data_predict = {  'xdata': x_pred \
                        , 'idata': i_predict \
                        , 'edata': e_predict \
                        }

        with self.model:
            PM.set_data(data_predict)


    def CleanUpSampling(self, sampling):
        prediction = sampling['y']
        # print (x_pred.shape, prediction.shape)

        # average over "chain" and "sample" dimension
        remove_noise = False
        if remove_noise:
            prediction = prediction.mean(axis = (0, 1))
        else:
            x_pred= NP.concatenate([self.x_prediction_raw.copy()]*prediction.shape[1], axis = 0)
            x_pred= NP.concatenate([x_pred]*prediction.shape[0], axis = 0)
            prediction = NP.concatenate(prediction, axis = 0)
            prediction = NP.concatenate(prediction, axis = 0)

            #x_pred= NP.stack([x_pred]*prediction.shape[1], axis = 0)
            #x_pred= NP.stack([x_pred]*prediction.shape[0], axis = 0)

        #print (x_pred.shape, prediction.shape)
        return x_pred, prediction.ravel()


    def PredictiveSampling(self):

        if self.x_prediction_raw is None:
            self.SetData()

        with self.model:
            sampling = PM.sample_posterior_predictive( self.trace \
                                                  , progressbar = False \
                                                  , return_inferencedata = False \
                                                  , random_seed = 42 \
                                                 )
        print() # progressbar bug


        self.x_pred, self.prediction = self.CleanUpSampling(sampling)

    def LinearRegression(self, col = 0):
        x, y = self.x[:, col].ravel(), self.y.ravel()
        reg = STATS.linregress(x, y)
        return reg


def MakePlot():
    cm = 1./2.54
    figwidth  = 16 * cm
    figheight = 16 * cm

    fig = PLT.figure( \
                              figsize = (figwidth, figheight) \
                            , facecolor = None \
                            , dpi = dpi \
                            )
    # PLT.ion() # "interactive mode". Might e useful here, but i don't know. Try to turn it off later.

    fig.subplots_adjust( \
                              top    = 0.98 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.0 # column spacing \
                            , hspace = 0.0 # row spacing \
                            )

    gs = MP.gridspec.GridSpec(2, 2 \
                            , height_ratios = [1, 4] \
                            , width_ratios = [4,1] \
                              )

    axx = fig.add_subplot(gs[0,0])
    ax  = fig.add_subplot(gs[1,0], sharex = axx)#, aspect = 'equal')
    axy = fig.add_subplot(gs[1,1], sharey = ax)
    axl = fig.add_subplot(gs[0,1])

    axx.spines[:].set_visible(False)
    axx.spines['bottom'].set_visible(True)
    axx.get_yaxis().set_visible(False)
    axx.get_xaxis().set_visible(False)
    axy.spines[:].set_visible(False)
    axy.spines['left'].set_visible(True)
    axy.get_xaxis().set_visible(False)
    axy.get_yaxis().set_visible(False)

    axl.spines[:].set_visible(False)
    axl.get_xaxis().set_visible(False)
    axl.get_yaxis().set_visible(False)

    return fig, {'ax': ax, 'x': axx, 'y': axy, 'l': axl}


def EdgeHeightPlot(edges, heights):
    e = []
    h = []
    e.append(edges[0])
    h.append(0)
    for i in range(len(edges)-1):
        e.append(edges[i])
        h.append(heights[i])
        e.append(edges[i+1])
        h.append(heights[i])

    e.append(edges[-1])
    h.append(0)

    return e, h


n_bins = 32


def PlotData(ax_dict, sim, color = '0.5', label = None, zorder = 0):
    ax = ax_dict['ax']
    axx = ax_dict['x']
    axy = ax_dict['y']
    axl = ax_dict['l']

    heights, edges = NP.histogram(sim.x[:, 0].ravel(), bins = n_bins, density = True)
    edgesx = edges
    bins = NP.add(edges[:-1], edges[1:])/2
    step = NP.mean(NP.diff(edges))
    axx.bar(bins, heights, width = step \
            , facecolor = color \
            , edgecolor = 'none' \
            , alpha = 0.4 \
            , zorder = zorder \
            )
    e, h = EdgeHeightPlot(edges, heights)
    axx.plot(e, h \
            , color = 'k' \
            , linewidth = 0.5, ls = '-', alpha = 0.8 \
            , zorder = zorder + 100 \
            )

    scatter_kwargs = dict(marker = 'o', facecolor = color, edgecolor = 'k' \
            , linewidth = 0.5 \
            , zorder = zorder + 60 \
            , label = label \
            )
    ax.scatter(sim.x[:, 0], sim.y, s = 4, alpha = 0.8, **scatter_kwargs)
    if label is not None:
        axl.scatter([0], [0], s = 10, alpha = 1.0, **scatter_kwargs)


    heights, edges = NP.histogram(sim.y.ravel(), bins = n_bins, density = True)
    edgesy = edges
    bins = NP.add(edges[:-1], edges[1:])/2
    step = NP.mean(NP.diff(edges))
    axy.barh(bins, heights, height = step \
            , facecolor = color \
            , edgecolor = 'none' \
            , alpha = 0.4 \
            , zorder = zorder \
            )
    e, h = EdgeHeightPlot(edges, heights)
    axy.plot(h, e \
            , color = 'k' \
            , linewidth = 0.5, ls = '-', alpha = 0.8 \
            , zorder = zorder + 100 \
            )





def PlotPrediction(ax_dict, sim, color = 'darkblue', label = None, zorder = 0):
    ax = ax_dict['ax']
    axx = ax_dict['x']
    axy = ax_dict['y']
    axl = ax_dict['l']

    heights, edges = NP.histogram(sim.x_pred[:, 0].ravel(), bins = n_bins, density = True)
    edgesx = edges
    bins = NP.add(edges[:-1], edges[1:])/2
    axx.bar(bins, heights, width = NP.mean(NP.diff(edges)) \
            , facecolor = color \
            , edgecolor = 'none' \
            , alpha = 0.4 \
            , zorder = zorder \
            )
    e, h = EdgeHeightPlot(edges, heights)
    axx.plot(e, h \
            , color = color \
            , zorder = zorder + 100 \
            , linewidth = 0.5, ls = '-', alpha = 0.8 \
            )


    scatter_kwargs = dict(marker = '.', color = color \
            , zorder = zorder \
            , label = label \
            )
    ax.scatter(sim.x_pred[:, 0], sim.prediction, s = 1, alpha = 0.01, **scatter_kwargs)

    if label is not None:
        axl.scatter([0], [0], s = 10, alpha = 1.0, **scatter_kwargs)


    heights, edges = NP.histogram(sim.prediction.ravel(), bins = n_bins, density = True)
    edgesy = edges
    bins = NP.add(edges[:-1], edges[1:])/2
    axy.barh(bins, heights, height = NP.mean(NP.diff(edges)) \
            , facecolor = color \
            , edgecolor = 'none' \
            , alpha = 0.4 \
            , zorder = zorder \
            )
    e, h = EdgeHeightPlot(edges, heights)
    axy.plot(h, e \
            , color = color \
            , linewidth = 0.5, ls = '-', alpha = 0.8 \
            , zorder = zorder + 100 \
            )


def UniformLimits(ax_dict):
    ax = ax_dict['ax']
    lims = list(ax.get_xlim()) + list(ax.get_ylim())
    lims = [min(lims), max(lims)]
    ax.set_xlim(lims)
    ax.set_ylim(lims)


    ax.set_xlabel('x')
    ax.set_ylabel('y')



def RunProcedure(slopes, shift = False):
    sim = Simulation(slopes = slopes \
                            , intercept = 0. \
                            , x_range = [-0.5, 0.5] \
               )
    sim.FitModel()
    sim.PredictiveSampling()

    fig, ax_dict = MakePlot()

    PlotData(ax_dict, sim, color = '0.5', label = 'observation', zorder = 0)

    PlotPrediction(ax_dict, sim, color = (0.3, 0.4, 0.7), label = 'in-sample', zorder = 10)


    x_pred = sim.PreparePredictionData()
    if shift:
        x_pred[:, 0] -= 0.1*NP.diff(sim.settings['x_range'])
    else:
        mu = 0.75*sim.settings['x_range'][0] + 0.25*sim.settings['x_range'][1]
        sigma = 0.1*NP.diff(sim.settings['x_range'])
        x_pred[:, 0] = NP.random.normal(mu, sigma, sim.settings['n_observations'])
    sim.SetData(x_pred)
    sim.PredictiveSampling()

    PlotPrediction(ax_dict, sim, color = (0.9, 0.5, 0.3), label = 'out-of-sample', zorder = 20)


    ax_dict['l'].set_xlim([0.95, 1.1])
    slope0 = sim.settings['slopes'][0]
    n_slopes = sim.settings['n_slopes'] - 1
    ax_dict['l'].legend(loc = 0, fontsize = 8 \
                        , title = f"""slope{'s' if n_slopes > 1 else ''}: {slope0} (+{n_slopes})""")

    UniformLimits(ax_dict)

    fig.savefig(f"""./show/prediction_{sim.settings['slopes'][0, 0]:0.2f}_slopes{sim.settings['n_slopes']}.png""", dpi = dpi)
    # PLT.show()
    PLT.close()


if __name__ == "__main__":

    other_slopes = [0.2, -0.1, 0.3, -0.3, 0.1, -0.2]
    for slope0 in [0.2, 0.4, 0.0]:
        slopes = [slope0] + other_slopes
        for i in range(len(slopes)):
            if i == 0:
                continue
            print ('#'*5, f'{i} slopes for slope[0] = {slope0}', '#'*5)
            RunProcedure(slopes[:i], shift = False)
