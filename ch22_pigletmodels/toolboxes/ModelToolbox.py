#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20210828

"""
Gathering functions which are useful for statistical model creation, sampling, visualization and comparison.
"""

# TODO: flexible distribution and priors
# TODO: "show": mark only relevant variables to enter visualization
# DONE: pymc3.model.set_data instead of aesara magic! see example: https://docs.pymc.io/api/model.html#pymc3.model.set_data
# DONE: predictive sampling: restoring shared values does not work.

"""
"""

################################################################################
### Libraries                                                                ###
################################################################################
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import re as RE             # regular expression (substring extraction)
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import cloudpickle as PI         # storage of models and results
from datetime import datetime as DT # sampling timestamp

import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API
import seaborn as SB        # even higher level plotting API


# simple, quick stats
import scipy.stats as STATS # quick statistical analysis
import statsmodels.api as SM # statistical models
import statsmodels.formula.api as SMF # formula api for models

# probabilistic stats
import pymc as PM          # basic modeling
import arviz as AZ         # model stats/ analysis
import patsy as PA         # formula notation

from packaging import version as VER
# print (VER.parse(PM.__version__))
import aesara as AE
import aesara.tensor as AT



##############################################################################
### Helper functions                                                       ###
##############################################################################
# extract coordinates from a summary row
ExtractIdx = lambda txt: tuple(map(int, RE.findall('\[(\d+)\,(\d+)\]', txt)[0]))

# data matrix from data and formula
DataMatrix = lambda formula, data: \
                        NP.asarray( \
                                  PA.dmatrix( formula \
                                , data = data \
                                , return_type = 'dataframe' \
                                ))

# print the shape of a aesara tensor
# https://aholzner.wordpress.com/2016/01/03/printing-a-tensors-shape-in-theano/
PrintShape = lambda label, tensor: AE.printing.Print(label, attrs = [ 'shape' ])(tensor)



##############################################################################
### Component Management                                                   ###
##############################################################################

#_______________________________________________________________________________
def AddComponent(  model: PM.Model \
                 , data: PD.DataFrame \
                 , params: list \
                 , label: str \
                 , observables: list \
                 , prior_kwargs: dict = {'mu': 0., 'sigma': 1.} \
                 , verbose: bool = False \
                 ):
    # generate a single, non-hierarchical component

    # tensor dimensions
    n_observables = len(observables)
    n_params = len(params)

    if verbose:
        print (f"### assembling {label} ###")
        print ("\tpriors:", prior_kwargs)

    formula = " + ".join(params)
    if not "1" in params:
        formula = "0 + " + formula
    if verbose:
        print (f"\tformula: {formula}")

    data_matrix = DataMatrix(formula, data)
    if verbose:
        print ("\tdmat:\t", str(data_matrix[:5, :]).replace('\n', '\n\t\t') \
               , "\n\t\t", data_matrix.shape \
               )

    data_shape = data_matrix.shape[1]

    if not(data_shape == n_params):
        raise Warning('Note that you probably included a categorical parameter. Be aware that this might cause trouble when having an intercept in the model (redundant sampling).')
        n_params = data_shape


    with model:
        shared = PM.Data(f'{label}_data', data_matrix, mutable = True)

        population = PM.Normal( label \
                                , shape = (n_params, n_observables ) \
                                # , mu = 0., sigma = 1. \
                                , **prior_kwargs \
                               )


    component = AT.dot(shared, population)

    if verbose:
        PrintShape(f'component: {population}', component)
        # print (f"\tcomponent: {population}", "{}x{}".format(n_params, len(params)))
        print ("\tdone!")

    # PrintShape('shared', shared)
    # PrintShape('population', population)

    return component


#_______________________________________________________________________________
def AddMultiLevelComponent( \
                            model: PM.Model \
                          , data: PD.DataFrame \
                          , params: list \
                          , label: str \
                          , level: str \
                          , observables: list \
                          , prior_kwargs: dict = {'mu': 0., 'sigma': 1.} \
                          , verbose: bool = False \
                          ):
    # generate a single, multilevel component of a model
    # label: a name for this part of the model
    # params: list of parameters ("predictors") which are multiplied with slopes; e.g. ['x1', 'x2']
    # level must be a PD.Categorical in data PD.DataFrame
    # observables: the dependent model variables (as a list), e.g. ['y1', 'y2']
    # both params and observables are columns in the data
    # returns: (n_observations x n_observables) model component

    # I label counts as "n"
    n_observables = len(observables)
    n_params = len(params)

    if verbose:
        print (f"### assembling {label} on {level} ###")
        print ("\tpriors:", prior_kwargs)

    # data matrix
    # to be honest, I haven't tried using multiple "params", but it /should/ work.
    # You can also have ["1"] in the params list to get a multilevel intercept
    # using patsy dmatrix is a bit overkill, you can just grab values from the data frame instead.
    data_matrix = NP.asarray(PA.dmatrix( f" + ".join(['0'] + params) \
                                               , data = data \
                                               , return_type = 'dataframe' \
                                              ))

    # design matrix for group level
    # "level" is a categorical column in the "data"
    # here, the Patsy dmatrix is actually useful because it generates a boolean matrix from the categorical
    group_matrix = NP.asarray( \
                        PA.dmatrix( f'0 + {level}' \
                                    , data = data \
                                    , return_type = 'dataframe' \
                                   ) \
                           )
    n_groups = group_matrix.shape[1]

    if verbose:
        print ("\tdmat:\t", str(group_matrix[:5, :]).replace('\n', '\n\t\t') \
               , "\n\t\t", group_matrix.shape \
               )


    # convert data to a "theano.shared" via the fabulous PM.Data function
    with model:
        shared_dat = PM.Data(f'{label}_data', data_matrix, mutable = True)
        shared_grp = PM.Data(f'{label}_group', group_matrix, mutable = True)

    # loop observables (stacked afterwards) because their parameter blocks are independent
    obslwise = [] # will aggregate the slopes per observable (obsl); stacked below
    for obsl, observable in enumerate(observables):
        with model:
            if label == 'intercept':
                hp_mean = PM.Normal( f'{label}_ml_population_{observable}' \
                                        , shape = n_params \
                                        , **{k: v[obsl] for k, v in prior_kwargs.items()} \
                                       )

                population = PM.Normal( f'{label}_ml_{level}_{observable}' \
                                        , shape = (n_groups, n_params ) \
                                        , mu = hp_mean \
                                        , sigma = prior_kwargs.get('sigma', [1.]*n_observables)[obsl] \
                                       )

                # # attempt to leave out the hyperprior
                # population = PM.Normal( f'{label}_ml_{level}_{observable}' \
                #                         , shape = (n_groups, n_params ) \
                #                         , **{k: v[obsl] for k, v in prior_kwargs.items()} \
                #                        )

            else:
                # prepare n_groups x n_params slopes
                population = PM.Normal( f'{label}_ml_{level}_{observable}' \
                                        , shape = (n_groups, n_params ) \
                                        , **{k: v[obsl] for k, v in prior_kwargs.items()} \
                                       )

            # only use the group-specific slope per observation
            # groupwise = PM.Deterministic(f'{label}_ml_{level}_{observable}_groupwise' \
            #                              , AT.dot(shared_grp, population) \
            #                              ) # n_observations x n_params
            groupwise = AT.dot(shared_grp, population)
            # print(f'{label} {level} {observable}')
            # PrintShape('groupwise', groupwise)

            # multiply data and slopes; sum over parameters (components are added)
            grpdat_matrix = AT.mul(shared_dat, groupwise)
            # PrintShape('grpdat_M', grpdat_matrix)
            grpdat = AT.sum(grpdat_matrix, axis = 1)
            # PrintShape('grpdat', grpdat)

            # grpdat_matrix = PM.Deterministic(f'{label}_ml_{level}_{observable}_grpdatmat' \
            #                              , AT.mul(shared_dat, groupwise) \
            #                              ) # n_observations x n_params
            # grpdat = PM.Deterministic(f'{label}_ml_{level}_{observable}_grpdat' \
            #                              , AT.sum(grpdat_matrix, axis = 1) \
            #                              ) # n_observations x 1

        # append observable-wise list
        obslwise.append(grpdat)

    # stack observables
    with model:
        component = AT.stack(obslwise, axis = 1)
        # PrintShape('component', component)

    if verbose:
        PrintShape(f'component: {component}', component)
        print ("\tdone!")

    return component


#_______________________________________________________________________________
def AddMultiVariateComponent( \
                   model: PM.Model \
                 , data: PD.DataFrame \
                 , params: list \
                 , label: str \
                 , observables: list \
                 , prior_kwargs: dict = {} \
                 , verbose: bool = False \
                 ):
    # generate a multivariate, non-hierarchical component

    # relevant dimensions
    n_observables = len(observables)
    n_params = len(params)

    # print (f'adding mv: {label}')

    if verbose:
        print (f"### assembling mv {label} ###")
        print ("\tpriors:", prior_kwargs)

    formula = " + ".join(params)
    if not "1" in params:
        formula = "0 + " + formula
    if verbose:
        print (f"\tformula: {formula}")

    data_matrix = DataMatrix(formula, data)
    if verbose:
        print ("\tdmat:\t", str(data_matrix[:5, :]).replace('\n', '\n\t\t') \
               , "\n\t\t", data_matrix.shape \
               )


    with model:

        shared = PM.Data(f'{label}_data', data_matrix, mutable = True)


        # predictor variable SLOPES can be correlated,
        #   which needs to be incorporated in the model

        ### estimate correlation of predictor SLOPES in the model

        covariates = []
        for obsl in range(n_observables):
            # prior for covariates covariance
            cholesky_matrix, _, _ = PM.LKJCholeskyCov( f'{label}_slope_cholesky_{observables[obsl]}' \
                                                       , n = n_params \
                                                       , eta = 1. #prior_kwargs.get('eta', 1.) \
                                                       , sd_dist = PM.HalfCauchy.dist(1.) \
                                                       , compute_corr = True \
                                                      )

            # # compute the covariates covariance and correlation matrix
            # cholesky_matrix = PM.expand_packed_triangular(n_params, packed_cholesky, lower = True)
            # # (additional covariance diagnostics are possible, see below)
            # covariance_matrix = PM.Deterministic(f'{label}_slope_covariance' \
                #                                      , AT.dot(cholesky_matrix, cholesky_matrix.T))


            # print (label, observables[obsl])
            # population effect, LKJ prior
            population = PM.MvNormal(  f'{label}_population_{observables[obsl]}' \
                                            , mu = 0. #prior_kwargs.get('mu', NP.zeros((n_observables,))).ravel()[obsl] \
                                            , chol = cholesky_matrix \
                                            , shape = n_params \
                                            )
            covariates.append(population)

        block = AT.stack(covariates, axis = 1)
        # PrintShape('covariates', block)

        component = AT.dot(shared, block)

    if verbose:
        PrintShape(f'component: {population}', component)
        print ("\tdone!")

    return component





#_______________________________________________________________________________
def AddMultiVariateMultiLevelComponent( \
                   model: PM.Model \
                 , data: PD.DataFrame \
                 , params: list \
                 , level: str \
                 , label: str \
                 , observables: list \
                 , prior_kwargs: dict = {'mu': 0.} \
                 , verbose: bool = False \
                 ):
    # generate a multivariate, hierarchical model component

    # dimensions are critical!
    n_observables = len(observables)
    n_params = len(params)

    # print (f'adding mv: {label}')

    if verbose:
        print (f"### assembling mv: {label} | {level} ###")
        print ("\tpriors:", prior_kwargs)

    formula = " + ".join(params)
    if not "1" in params:
        formula = "0 + " + formula
    if verbose:
        print (f"\tformula: {formula}")

    data_matrix = DataMatrix(formula, data)

    # design matrix for group level
    group_matrix = NP.asarray( \
                        PA.dmatrix( f'0 + {level}' \
                                    , data = data \
                                    , return_type = 'dataframe' \
                                   ) \
                           )
    group_indices = data[level].cat.codes.values.reshape((-1,1))
    n_groups = len(NP.unique(group_indices))
    # print('grp indices', group_indices.shape)

    data_shape = data_matrix.shape
    n_observations = data_shape[0]

    with model:

        shared_dat = PM.Data(f'{label}_data', data_matrix, mutable = True)
        shared_grp = PM.Data(f'{label}_group', group_matrix, mutable = True)


        # predictor variable SLOPES can be correlated,
        #   which needs NOT to be incorporated in the model
        #   as fixed ('empirical') covariance
        #   but ALWAYS as sampled covariance
        # datamat = data.loc[:, params].values
        # data_centralized = NP.zeros_like(datamat)


        # one for each observable
        obslwise = []
        for obsl in range(n_observables):
            ### estimate correlation of SLOPES in the model
            # prior for covariates covariance
            cholesky_matrix, _, _ = PM.LKJCholeskyCov( f'{label}_slope_cholesky_{observables[obsl]}' \
                                                 , n = n_params \
                                                 , eta = prior_kwargs.get('eta', 1.) \
                                                 , sd_dist = PM.HalfCauchy.dist(1.) \
                                                 , compute_corr = True \
                                                )

            # # compute the covariates covariance and correlation matrix
            # cholesky_matrix = PM.expand_packed_triangular(n_params, packed_cholesky, lower = True)
            # # (additional covariance diagnostics are possible, see below)
            # covariance_matrix = PM.Deterministic(f'{label}_slope_covariance' \
            #                                          , AT.dot(cholesky_matrix, cholesky_matrix.T))

            if False: # population only useful on high number of groups
                # population effect, LKJ prior
                population = PM.MvNormal( f'{label}_population_{observables[obsl]}' \
                                          , mu = prior_kwargs.get('mu', 0.) \
                                          , chol = cholesky_matrix \
                                          , shape = n_params \
                                         )

                # sampling parametrization
                groupwise = [ \
                              PM.MvNormal( f'{label}_grp{grp}_{observables[obsl]}' \
                                           , mu = population \
                                           , chol = cholesky_matrix \
                                           , shape = n_params \
                                          ) \
                              for grp in range(n_groups) \
                             ]
            else:
                # plain prior
                mu_prior = prior_kwargs.get('mu', NP.zeros((n_groups*n_params,)))
                groupwise = [ \
                              PM.MvNormal( f'{label}_grp{grp}_{observables[obsl]}' \
                                           , mu = mu_prior[grp*n_params:(grp+1)*n_params] \
                                           , chol = cholesky_matrix \
                                           , shape = n_params \
                                          ) \
                              for grp in range(n_groups) \
                             ]
            allgroups = AT.stack(groupwise, axis=0)

            # add up slopes for each of the groups
            obsn_params = AT.dot(shared_grp, allgroups) # take the right group for each row

            block = shared_dat * obsn_params # multiply data and slopes

            obslwise.append(AT.sum(block, axis = 1))

            if verbose:
                PrintShape('\tallgroups', allgroups) # grp x param
                PrintShape('\tobsn_params', obsn_params) # obsn x params
                PrintShape('\tblock', block) # obsn x params
        # NOTE: all the intermediates can be wrapped into a "PM.deterministic" for display
        # obslwise = PM.Deterministic(f'{label}_all' \
        #                                  , AT.stack(obslwise, axis = 1) \
        #                                  )


        obslwise = AT.stack(obslwise, axis = 1)

    if verbose:
        PrintShape(f'\t{label}: {obslwise}', obslwise)
        print ("\tdone!")

    return obslwise


def AddPosterior( \
                   model: PM.Model \
                 , data: PD.DataFrame \
                 , estimator \
                 , observables: list \
                 , robust: bool = False \
                 , prior_kwargs = {} \
                 , verbose: bool = False \
                 ):
    ### Likelihood/Posterior/Observables
    #   "level" is only relevant for data centering if fixed_covariance

    if verbose:
        print (f"### Posterior ###")
        print (f'adding observables: {", ".join(observables)}')
        print ("\tpriors:", prior_kwargs)

    n_observables = len(observables)

    with model:

        # residual = PM.HalfCauchy('residual', prior_kwargs.get('residual', 1.), shape = (1, n_observables))
        shared = PM.Data(f'residual_data', NP.ones((data.shape[0],1)), mutable = True)
        residual = AT.dot(shared, PM.HalfCauchy('residual', prior_kwargs.get('residual', 1.), shape = (1, n_observables)))

        # posterior, i.e. left hand side
        if robust:
            # Student T
            dof = PM.Gamma('dof', prior_kwargs.get('alpha', 5.), prior_kwargs.get('beta', 0.1))#, shape = n_observables )
            posterior = PM.StudentT(  'posterior' \
                                  , mu = estimator \
                                  , sigma = residual \
                                  , nu = dof \
                                  , observed = data.loc[:, observables].values \
                                  , shape = residual.shape \
                                    )


        else:
            posterior = PM.Normal(  'posterior' \
                                  , mu = estimator \
                                  , sigma = residual \
                                  , observed = data.loc[:, observables].values \
                                  , shape = residual.shape \
                                )

    # nothing to return because this happens in model context


def AddMultiVariatePosterior( \
                   model: PM.Model \
                 , data: PD.DataFrame \
                 , estimator \
                 , observables: list \
                 , level: str \
                 , robust: bool = False \
                 , fixed_covariance: bool = False \
                 , prior_kwargs: dict = {} \
                 , verbose: bool = False \
                 , ref_shape: tuple = None \
                 ):
    ### Likelihood/Posterior/Observables
    #   "level" is only relevant for data centering if fixed_covariance

    if verbose:
        print (f"### Posterior ###")
        print (f'adding mv observables: {", ".join(observables)}')
        print ("\tpriors:", prior_kwargs)
    n_observables = len(observables)

    with model:


        # prior for observables covariance
        if fixed_covariance:
            data_matrix = data.loc[:, observables].values

            # centralize
            if level == 'population':
                data_matrix -= NP.mean(data_matrix, axis = 0)
            else:
                # groupwise centralization
                data_centralized = NP.zeros_like(data_matrix)
                grouping = data[level].cat.codes.values
                for ig in NP.unique(grouping):
                    data_centralized[grouping==ig, :] = data_matrix[grouping==ig, :] \
                                                    - data_matrix[grouping==ig, :].mean(axis=0)

                data_matrix = data_centralized


            # covariance
            covariance_matrix = NP.cov(data_matrix.T, ddof = 1)

            # cholesky
            cholesky_matrix = NP.linalg.cholesky(covariance_matrix)


        else:
            # sd_dist = PM.HalfNormal.dist(1.)
            # sd_dist = PM.HalfCauchy.dist(1.)
            if False:
                # testing the two possible return settings depending on `compute_corr`
                packed_cholesky = PM.LKJCholeskyCov(  'cholesky' \
                                                      , n = n_observables \
                                                      , eta = prior_kwargs.get('eta', 1.) \
                                                      , sd_dist = PM.HalfCauchy.dist(1.) \
                                                      , compute_corr = False \
                                                    )

                # compute the observables covariance and correlation matrix
                cholesky_matrix = PM.expand_packed_triangular(n_observables, packed_cholesky, lower = True)

            else:
                cholesky_matrix, _, _ = PM.LKJCholeskyCov(  'cholesky' \
                                                      , n = n_observables \
                                                      , eta = 1. #prior_kwargs.get('eta', 1.) \
                                                      , sd_dist = PM.HalfCauchy.dist(1.) \
                                                      , compute_corr = True \
                                                    )

            # # additional covariance diagnostics
            # # could alternatively be recovered from the Cholesky matrix posterior
            # # thus, remove for complex models
            # covariance_matrix = PM.Deterministic('covariance_matrix' \
            #                                      , AT.dot(cholesky_matrix, cholesky_matrix.T) \
            #                                      )
            # standard_deviations = PM.Deterministic('standard_deviations' \
            #                                        , AT.sqrt(AT.diag(covariance_matrix)) \
            #                                        )
            # correlation_matrix = PM.Deterministic('correlation_matrix' \
            #                                       , AT.diag(standard_deviations**-1) \
            #                                       .dot(covariance_matrix.dot(AT.diag(standard_deviations**-1))) \
            #                                       )
            # cross_correlation = PM.Deterministic('cross_correlation' \
            #                                      , correlation_matrix[NP.triu_indices(n_observables, k=1)] \
            #                                      )

        # posterior, i.e. left hand side
        if robust:
            # Student T
            # degrees of freedom: for weird reason, need to be transposed to enter MvStudentT
            dof = PM.Gamma('dof' \
                           , 5. #prior_kwargs.get('alpha', NP.ones((n_observables,1))*5.) \
                           , 0.1 #prior_kwargs.get('beta', NP.ones((n_observables,1))*0.1) \
                           # , shape = (1, n_observables) \
                           ) # dof shape: must be scalar (otherwise ValueError)
            # shared = PM.Data(f'dof_data', NP.ones((data.shape[0],1)), mutable = True)
            # print (data.loc[:, observables].values.shape)

            # posterior Student block
            posterior = PM.MvStudentT(  'posterior' \
                                      , mu = estimator \
                                      , chol = cholesky_matrix \
                                      , nu = dof \
                                      , observed = data.loc[:, observables].values \
                                      , shape = ref_shape #TODO
                                     )


        else:
            posterior = PM.MvNormal(  'posterior' \
                                  , mu = estimator \
                                  , chol = cholesky_matrix \
                                  , observed = data.loc[:, observables].values \
                                  , shape = ref_shape #TODO
                                )

    # nothing to return



##############################################################################
### Settings                                                               ###
##############################################################################
class Settings(dict):
    """
    setting storage by subclassing dict
    with default settings
    from http://stackoverflow.com/a/9550596
    """

    def __init__(self, settings = None, basic_settings = None):
        # first initialize all mandatory values.
        self.SetDefaults()

        # then overwrite everything by the basic settings.
        if basic_settings is not None:
            for key in basic_settings.keys():
                self[key] = basic_settings[key]

        # finally overwrite everything the user provides.
        if settings is not None:
            for key in settings.keys():
                self[key] = settings[key]

    def __str__(self):
        predictor_string = "\n\t\t" + str(self.GetPredictors()).replace('\n', '\n\t\t')
        return "\n\t".join([f"""{k}: {predictor_string if k == 'predictors' else str(v)}""" \
                            for k,v in self.items()])


# predictor block is handled as a data frame
# but stored as a dict
    def GetPredictors(self) -> PD.DataFrame:
        # convert preDICTor into a data frame
        predictor_df = PD.DataFrame.from_dict(self['predictors']).T
        predictor_df.index.name = 'label'
        predictor_df.columns = ['level', 'group']

        return predictor_df

    def SetPredictors(self, predictor_df: PD.DataFrame):
        # store predictors as dict
        self['predictors'] = predictor_df.T.to_dict()


    def SetDefaults(self):
        # set the default settings
        # should work for simulation and real data
        # set to be a quick test

        ### observables as list
        self['observables'          ] = ['PC1'] # list of observables
        self['center_observables'   ] = False # center observables on their mean
        self['obs_multivariate'     ] = False # model observables as a multivariate block
        self['obscov_fixed'         ] = True # ... optionally: fix mv block covariance

        # use Normal (robust = False) or StudentT (robust = True) likelihood distribution
        self['robust'               ] = False

        ### intercept
        self['intercept_level'      ] = 'population' # {None, population, litter, subject}

        # # nesting method if hierarchical intercept
        # self['intercept_offset'     ] = False # "offset" or "sampling" parametrization; ignored if "population" TODO ignored by now; only "sampling"

        ### predictors
        self['predictors'] = { \
                              'age': ['population', None] \
                            , 'speed': ['subject', 'stride'] \
                            , 'duty': ['subject', 'stride'] \
                              } # this is an example! make sure to overwrite!
        # for better structure
        self.SetPredictors(self.GetPredictors())

        # priors to pass to the component constructors
        self['priors'               ] = {'age': {'mu': 0., 'sigma': 1.} \
                                         , 'observables': {'eta': 1., 'mu': NP.zeros(len(self['observables']))} \
                                         }


        ### sampling
        self['n_steps'              ] = 2**8
        self['n_cores'              ] = 1
        self['n_chains'             ] = 4

        ### visualization
        self['show'                 ] = []



    @classmethod
    def DeepCopy(cls, settings_dict):
        # prepare a deep copy
        # avoiding to make it a "dict"
        return cls(settings = settings_dict)

    def Copy(self):
        # prepare a deep copy
        # avoiding to make it a "dict"

        copy = Settings.DeepCopy(self.copy())
        # deep copy of the settings object
        copy['predictors'] = self['predictors'].copy()
        return copy


    def ToDict(self):
        # return this as a conventional dict
        return {k:v for k, v in self.items()}


##############################################################################
### Model                                                                  ###
##############################################################################
class Model(object):
    # a wrapper for
    #   - data
    #   - settings
    #   - a pymc model
    #   - sampling outcome ("trace")
    #   - convenience functions (storage, loading)

    def __init__(  self \
                 , data: PD.DataFrame \
                 , settings: Settings \
                 , label: str = None \
                 , verbose: bool = False \
                 ):
        # prepare the model

        # link data and settings
        self.data = data.copy()
        self.settings = settings.Copy()

        # initialize empty results
        self.model = PM.Model()
        self.trace = None
        self.samplingtime = None

        # a label used as model name
        self.label = label
        if label is not None:
            self.model.name = label
        else:
            self.model.name = 'model'

        # set verbosity
        self.SetVerbose(verbose)

        # dimensions
        self.n = {}
        self.n['observations'] = self.data.shape[0]
        self.n['observables'] = len(self['observables'])

        # to store shared variables (ie. PM.Data labels)
        self.shared = []

        # posterior predictive sampling
        self.predictive_settings = None # data frame with sampling combinatorics
        self.predictive_samples = None # data frame with sampling outcome

        ### modeling process
        self.Print (f'\n### generating model {self.label} ###' )

        # prepare model generation
        self.Preparation()

        # construct the model
        self.ModelConstruction()

        ### convenience
        # forwarding relevant dictionary functions
        self.keys = self.settings.keys
        self.values = self.settings.values
        self.items = self.settings.items
        self.get = self.settings.get


    def SetVerbose(self, verbose: bool = True):
        # set the print function if the user chooses so
        self.verbose = verbose
        if self.verbose:
            self.Print = print
        else:
            self.Print = lambda *args, **kwargs: None


#______________________________________________________________________
# direct access to settings
    def __getitem__(self, item):
        # getting an item of the settings
        if item == 'predictors':
            return self.settings.GetPredictors()
        return self.settings[item]

    def __setitem__(self, item, value):
        # setting an item of the settings
        if item == 'predictors':
            # predictor block is handled as a data frame
            # but stored as a dict
            self.settings.SetPredictors(value)
        else:
            self.settings[item] = value


#______________________________________________________________________
    def Preparation(self):
        # several heuristics and data manipulations
        # which happen prior to model construction


        # center observables
        self.mean_storage = {'mu': [], 'sigma': [], 'dof': []} #
        for obs in self['observables']:
            values = self.data.loc[:, obs].values
            if obs[-3:] == 'phi':
                # special treatment of "phase":
                # zero-center circular values
                mu0 = STATS.circmean(values, low = -NP.pi, high = +NP.pi)
                sd0 = STATS.circstd(values, low = -NP.pi, high = +NP.pi)
                nu0 = 10.
                # # TODO: this should be improved for beta distributed values etc. as well
                # values -= circular_mean
                # nu0, mu0, sd0 = STATS.t.fit(values)
                # mu0 += circular_mean

            else:
                nu0, mu0, sd0 = STATS.t.fit(values)

            self.mean_storage['mu'].append(mu0)
            self.mean_storage['sigma'].append(sd0)
            self.mean_storage['dof'].append(nu0)

            if self['center_observables']:
                self.data.loc[:, obs] -= mu0

        # heuristic priors
        self.start_values = {'mu': [], 'sigma': [], 'dof': []}
        for obs in self['observables']:
            nu0, mu0, sd0 = STATS.t.fit(self.data.loc[:, obs].values)
            self.start_values['mu'].append(mu0)
            self.start_values['sigma'].append(sd0)
            self.start_values['dof'].append(nu0)

        self.start_values = {val: NP.array(start) \
                             for val, start in self.start_values.items() \
                             }

        self.Print('start values:')
        self.Print(self.start_values)


#______________________________________________________________________
    def ModelConstruction(self, verbose: bool = False):
        ## construct a model, piecewise, from components



        self.estimator = 0

        ## (1) Intercept
        if self['intercept_level'] is None:
            self.Print ('no intercept')
        elif self['intercept_level'] == 'population':
            self.AddComponent(["1"] \
                              , "intercept" \
                              , prior_kwargs = self['priors'].get('intercept' \
                                                    , {'mu': self.start_values['mu'] \
                                                       , 'sigma': 2*self.start_values['sigma'] \
                                                       }) \
                              )
        else:
            self.AddMultiLevelComponent(["1"], "intercept", self['intercept_level'] \
                              , prior_kwargs = self['priors'].get('intercept' \
                                                    , {'mu': self.start_values['mu'] \
                                                       , 'sigma': 2*self.start_values['sigma'] \
                                                       }) \
                                        )


        ## (2) Predictors
        #  groupwise iteration and adding components
        predictor_df = self.settings.GetPredictors()
        predictor_df.loc[:, 'group'] = [idx if rw['group'] is None else rw['group'] \
                                        for idx, rw in predictor_df.iterrows()]
        self.Print(predictor_df)
        groups = NP.unique(predictor_df.loc[:, 'group'].values)
        for grp in groups:
            # work on one parameter block
            group_df = predictor_df.loc[ \
                                         predictor_df['group'].values == grp \
                                         , :]
            # filter out the ones which are not on a level
            group_df = group_df.loc[[lvl is not None for lvl in group_df['level'].values], :]

            if group_df.shape[0] == 0:
                self.Print(f'no {grp} block')
                continue

            group_predictors = list(group_df.index.values)
            group_level = group_df['level'].values[0]

            # again, switch for level
            if len(group_predictors) == 1:
                if group_level == 'population':
                    # single variate predictor
                    self.AddComponent(params = group_predictors \
                                      , label = grp \
                              , prior_kwargs = self['priors'].get(grp \
                                                    , {'mu': NP.zeros((self.n['observables'],)) \
                                                       , 'sigma': 2*self.start_values['sigma'] \
                                                       }) \
                                      )
                else:
                    # multi level
                    self.AddMultiLevelComponent(params = group_predictors \
                                                , label = grp \
                                                , level = group_level \
                              , prior_kwargs = self['priors'].get(grp \
                                                    , {'mu': NP.zeros((self.n['observables'],)) \
                                                       , 'sigma': 2*self.start_values['sigma'] \
                                                       }) \
                                                )
            else:
                # multivariate blocks
                if group_level == 'population':
                    self.AddMultiVariateComponent( params = group_predictors \
                                                   , label = grp \
                              , prior_kwargs = self['priors'].get(grp \
                                                    , {'mu': NP.zeros((self.n['observables'],)) \
                                                       }) \
                                                  )
                else:
                    self.AddMultiVariateMultiLevelComponent(params = group_predictors \
                                                            , label = grp \
                                                            , level = group_level \
                              , prior_kwargs = self['priors'].get(grp \
                                                    , {'mu': NP.zeros((self.n['observables'],)) \
                                                       }) \
                                                            )

        ## (3) Likelihood
        self.AddPosterior()


    def AddPosterior(self):
        # add a posterior|likelihood|observables (-block)
        if (len(self['observables']) == 1) or (not self['obs_multivariate']):
            # simple, independent posteriors
            AddPosterior( \
                          model = self.model \
                        , data = self.data \
                        , estimator = self.estimator \
                        , observables = self['observables'] \
                        , robust = self['robust'] \
                        , prior_kwargs = self['priors'].get('observables', {})
                        , verbose = self.verbose \
                        )
            self.shared.append(f'residual_data')

        else:
            # multivariate posterior
            AddMultiVariatePosterior( \
                          model = self.model \
                        , data = self.data \
                        , estimator = self.estimator \
                        , observables = self['observables'] \
                        , level = self['intercept_level'] \
                        , robust = self['robust'] \
                        , fixed_covariance = self['obscov_fixed'] \
                        , prior_kwargs = self['priors'].get('observables', {})
                        , verbose = self.verbose \
                        , ref_shape = self.estimator.shape \
                        )
            self.shared.append(f'residual_data')




    def AddComponent(self, params: list, label: str, prior_kwargs: dict):
        # wrapper for accessing the more general AddComponent function
        # with settings of this model
        component = AddComponent( \
                                            model = self.model \
                                          , data = self.data \
                                          , params = params \
                                          , label = label \
                                          , observables = self['observables'] \
                                          , prior_kwargs = prior_kwargs \
                                          , verbose = self.verbose \
                                         )
        self.estimator = self.estimator + component
        self.shared.append(f'{label}_data')

    def AddMultiVariateComponent(self, params: list, label: str, prior_kwargs: dict):
        # wrapper for accessing the more general AddMultiVariateComponent function
        # with settings of this model
        component = AddMultiVariateComponent( \
                                            model = self.model \
                                          , data = self.data \
                                          , params = params \
                                          , label = label \
                                          , observables = self['observables'] \
                                          , prior_kwargs = prior_kwargs \
                                          , verbose = self.verbose \
                                         )
        self.estimator = self.estimator + component
        self.shared.append(f'{label}_data')

    def AddMultiLevelComponent(self, params: list, label: str, level: str, prior_kwargs: dict):
        # wrapper for accessing the more general AddMultiLevelComponent function
        # with settings of this model
        # level must be the name of a categorical data variable
        component = AddMultiLevelComponent( \
                                            model = self.model \
                                          , data = self.data \
                                          , params = params \
                                          , label = label \
                                          , level = level \
                                          , observables = self['observables'] \
                                          , prior_kwargs = prior_kwargs \
                                          , verbose = self.verbose \
                                         )
        self.estimator = self.estimator + component
        self.shared.append(f'{label}_data')
        self.shared.append(f'{label}_group')

    def AddMultiVariateMultiLevelComponent(self, params: list, label: str, level: str, prior_kwargs: dict):
        # wrapper for accessing the more general AddMultiVariateMultiLevelComponent function
        # with settings of this model
        # level must be the name of a categorical data variable
        component = AddMultiVariateMultiLevelComponent( \
                                            model = self.model \
                                          , data = self.data \
                                          , params = params \
                                          , label = label \
                                          , level = level \
                                          , observables = self['observables'] \
                                          , prior_kwargs = prior_kwargs \
                                          , verbose = self.verbose \
                                         )
        self.estimator = self.estimator + component
        self.shared.append(f'{label}_data')
        self.shared.append(f'{label}_group')

#______________________________________________________________________
    def Sample(self \
                 , n_steps: int = 1024, n_tune: int = None \
                 , cores: int = 1, chains: int = 4 \
                 , progressbar: bool = None \
                 ):

        self.Print (f'\n### sampling {self.label} ###' )

        # user control: progress bar
        if progressbar is None:
            progressbar = self.verbose

        # the sampling procedure
        with self.model:
            self.trace = PM.sample( \
                              draws = n_steps \
                            , tune = (n_steps if n_tune is None else n_tune)*1 \
                            , progressbar = progressbar \
                            , cores = cores \
                            , chains = chains \
                            , target_accept = 0.99 \
                            , return_inferencedata = True \
                            , random_seed = 42 \
                            )

        self.samplingtime = DT.now().strftime("%Y/%m/%d %H:%M:%S")

#______________________________________________________________________
# model results and diagnostics
    def Hygiene(self, title = None):
        # model diagnostics to assure sampling was satisfactory

        if title is None:
            title = self.label

        print ('_'*(18+len(title)), '\n')
        print (f'######   {title.upper()}   ######')
        # give the summary data frame
        if self.trace is None:
            print ('not sampled yet!')
            return None
        else:
            print (f'\tsampling of {self.samplingtime}')


        var_names = ['intercept'] \
                    + [pred for pred in self.settings.GetPredictors().index.values]
        if self['robust']:
            var_names += ['dof']
        var_names = [f'{self.label}::{vn}' for vn in var_names]

        # traceplot
        if True:
            print ('### trace ###')
            # should be hairy caterpillars
            PM.plot_trace(self.trace, var_names = var_names, combined = True)
            PLT.gcf().suptitle(title)
            PLT.show()


        if True:
            print ('### energy ###')
            # marginal energy and energy transition should roughly overlap
            PM.plot_energy(self.trace)
            PLT.gcf().suptitle(title)
            PLT.show()


        # Diagnostic stats
        if True:
            print ('### bfmi per chain ###')
            # should be greater than 0.3
            bmfis = PM.bfmi(self.trace)
            print("\n".join([ f'\t{b}\t{g}' \
                                for b, g in zip( \
                                     list(map(lambda bfmi: f'{bfmi:.2f}', bmfis)) \
                                   , [':)' if bfmi > 0.3 else 'NO GOOD!!' \
                                      for bfmi in bmfis] \
                                  )] \
                              ))
            # print(PM.ess(self.trace))

            print ('### rhat ###')
            # should be close to one
            rhat = PM.rhat(self.trace)
            rhat = {vn[len(self.label)+1:]: rhat[vn].values.ravel() \
                    for vn in var_names}
            Disp = lambda rh: f'{rh-1.0:.1e}'

            print('\n'.join( \
                             [f'\t{vn}\t1.0 + \t[{", ".join(map(Disp, rh))}]\t' \
                              + (':)' if NP.all(NP.abs(rh - 1.0) < 1e-2) else 'NO GOOD!')
                              for vn, rh in rhat.items()] \
                            )
                  )

            # print(PM.mcse(self.trace))


        if True:
            print ('### effective sample size ###')
            # When the model is converging properly, both lines in this plot should be roughly linear.
            m = 3
            subsets = list(range(len(var_names)//m+1))
            for n in subsets:
                sub_vars = var_names[n*m:(n+1)*m]
                if len(sub_vars) == 0:
                    continue
                PM.plot_ess( \
                             self.trace \
                             , kind = "evolution" \
                             , var_names = sub_vars \
                             #, coords=coords \
                            )
                PLT.gcf().suptitle(f'{title} ({n+1}/{len(subsets)})')
                PLT.show()

        # hygiene done! hope everything is clean?
        print ('_'*(18+len(title)), '\n')


    def Summary(self):
        # give the summary data frame
        if self.trace is None:
            print ('not sampled yet!')
            return None

        # order to the model outcome
        observable_translation = { \
                                    \
                                  }

        # stride params as predictors
        predictors = self.settings.GetPredictors()
        predictors = predictors.loc[predictors['group'].values == 'stride', :]
        preds = predictors.index.values

        # loop observables
        for obsl_nr, observable in enumerate(self['observables']):
            for parameter in [v for v in self.settings.GetPredictors().index.values] + ['intercept']:

                # group the parameter slopes
                observable_translation[f'{self.label}::{parameter}[0,{obsl_nr:.0f}]'] \
                                = [observable, parameter]
                observable_translation[f'{self.label}::{parameter}[0, {obsl_nr:.0f}]'] \
                                = [observable, parameter]

            # stride params
            for pred_nr, pred in enumerate(preds):
                observable_translation[f'{self.label}::stride_population_{observable}[{pred_nr:.0f}]'] \
                            = [observable, pred]


            # ml clbodymass
            for ac in range(3):
                for pred in ['lbodymass', 'lleg_m', 'lbmi']:
                    observable_translation[f'{self.label}::{pred}_ml_ageclass_{observable}[{ac},0]'] \
                        = [observable, pred]
                    observable_translation[f'{self.label}::c{pred}_ml_ageclass_{observable}[{ac},0]'] \
                        = [observable, f'c{pred}']
                    observable_translation[f'{self.label}::{pred}_ml_ageclass_{observable}[{ac}, 0]'] \
                        = [observable, pred]
                    observable_translation[f'{self.label}::c{pred}_ml_ageclass_{observable}[{ac}, 0]'] \
                        = [observable, f'c{pred}']

            # for the ml intercept case
            observable_translation[f'{self.label}::intercept_ml_population_{observable}[0]'] \
                            = [observable, 'intercept']

            # residual
            observable_translation[f'{self.label}::residual[0, {obsl_nr:.0f}]'] = [observable, 'residual']
            observable_translation[f'{self.label}::residual[0,{obsl_nr:.0f}]'] = [observable, 'residual']
            observable_translation[f'{self.label}::residual'] = ['misc', 'residual']
            # dof
            observable_translation[f'{self.label}::dof[0, {obsl_nr:.0f}]'] = [observable, 'dof']
            observable_translation[f'{self.label}::dof[0,{ obsl_nr:.0f}]'] = [observable, 'dof']
            observable_translation[f'{self.label}::dof'] = ['misc', 'dof']

            # cross correlation
            for obsl_nr2, observable2 in enumerate(self['observables']):
                observable_translation[f'{self.label}::cholesky_corr[{obsl_nr},{obsl_nr2}]'] = ['misc', f'{observable}|{observable2}']
                observable_translation[f'{self.label}::cholesky_corr[{obsl_nr}, {obsl_nr2}]'] = ['misc', f'{observable}|{observable2}']

            # stds
            observable_translation[f'{self.label}::cholesky_stds[{obsl_nr}]'] = [observable, 'std']


        # summary
        summary = PM.summary(self.trace)
        print (summary)
        # print ('\n'.join([f'{k}:\t{v}' for k,v in observable_translation.items()]))

        # improve index
        summary.index = PD.MultiIndex.from_tuples( \
                                               ( observable_translation.get(idx, ['misc', None])[0] \
                                                 , idx.replace(f'{self.label}::','') \
                                                 , observable_translation.get(idx, ['misc', None])[1] \
                                                )
                                               for idx in summary.index.values \
                                              )

        summary.sort_index(inplace = True)
        summary.index.names = ['observable', 'effect', 'predictor']

        return summary



#______________________________________________________________________
# Predictive Sampling
    def PredictiveSampling(  self \
                           , sampling_settings: PD.DataFrame \
                           , n_samples: int = None \
                           , lower_level_predictions: dict = None \
                           , prefix: str = 'store' \
                           , backwards: bool = False \
                           , thinning: int = None \
                           ):
        # posterior predictive sampling
        self.predictive_settings = sampling_settings

        # store shared values
        # shared_storage = {param: shared.get_value() \
        #                   for param, shared in self.shared.items() \
        #                   }
        # print ([k for k in self.shared.keys()])
        # print ('pre')
        # self.PrintAllShapes()
        if not OS.path.exists(f'predictions{OS.sep}{prefix}'):
            OS.system(f'mkdir predictions{OS.sep}{prefix}')

        # gather predictions for each setting
        predictions = []
        if backwards:
            sampling_settings = sampling_settings.loc[sampling_settings.index.values[::-1], :]
        be_warned = False

        prediction_data = {}
        lower_level_storage = {}
        for label in self.shared: # + ['residual_data']: # TODO remove "residual" once it is in the model
            # print (label)

            param = "_".join(label.split("_")[:-1])

            # check if param is in the settings
            if label not in sampling_settings.columns:
                # print (setting_values)
                # print (label, setting_values.index)

                if lower_level_predictions is None:
                    raise IOError(f'please set a value for {param}.')

                # TODO
                # sampled from a second level model
                prediction_data[label] = lower_level_predictions[(setting_idx, label)]

                if not ('_group' == label[-6:]):
                    lower_level_storage[label] = lower_level_predictions[(setting_idx, label)].ravel()
                else:
                    # turn group matrix into categorical
                    grpmat = lower_level_predictions[(setting_idx, label)]
                    lower_level_storage[label] = NP.array(NP.sum( grpmat \
                                                       @ (NP.arange(grpmat.shape[1]).reshape((-1,1)) + 1) \
                                                        , axis = 1).ravel(), dtype = int)



            # single-step modeling
            else:
                # can we set arbitrary n_samples? yes!
                prediction_data[label] = sampling_settings[label].values.reshape((-1,1))

        # print (prediction_data)

        with self.model:
            # 20221006 this seems not to be fixed in the actual pymc4 release?
            # -> 20221020 it is fixed! https://github.com/pymc-devs/pymc/issues/6230

            # simplified procedure in pymc4+
            # print (prediction_data['μ_wrist_data'].shape)

            # remove vars which are not in the model data (e.g. "residual" in multivariate models)
            named_vars = [k for k in self.model.named_vars.keys()]
            prediction_data = {k: v for k, v in prediction_data.items() if f'{self.label}::{k}' in named_vars}
            PM.set_data(prediction_data)

            # this is how "samples" works nowadays (https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html#Examples)
            trace_thinned = self.trace.sel(draw=slice(None, (None if thinning is None else thinning-1), None))
            sampling = PM.sample_posterior_predictive(  trace_thinned \
                                                      #, keep_size = False \
                                                      #, samples = n_samples \
                                                      #, samples = 1 \
                                                      #, samples = self.n['observations'] \
                                                      #, progressbar = True \
                                                        , return_inferencedata = False \
                                                      )[f'{self.label}::posterior']
            print() # progressbar bug

            if thinning is not None:
                # when there are more samples requested than available in the trace
                if thinning > sampling.shape[0]:
                    print ('only {posterior.shape[0]} samples available; thinning ({thinning}) requested but not applied.')

        # print (sampling.shape) # (thinning*n_chains, n_observed, n_observables)

        # format/index; to data frame
        predictions = []
        for nr, setting_idx in enumerate(sampling_settings.index.values):
            pred = PD.DataFrame(NP.concatenate([sampling[chain, :, nr, :] for chain in range(sampling.shape[0])], axis = 0) \
                                , columns = self['observables'] \
                                )

            pred['cycle_idx'] = setting_idx
            predictions.append(pred)

        predictions = PD.concat(predictions)
        predictions.index.name = 'sample_nr'
        predictions.reset_index(drop = False, inplace = True)
        print (predictions)

        return predictions

        for setting_idx, setting_values in sampling_settings.iterrows():
            # set all the "shared" data coefficients to specific setting_values according to the sampling settings

            # print ('\n')
            # print (prefix, self.label)
            # print (setting_idx, setting_values)

            # load previous prediction
            prediction_storage = f'predictions{OS.sep}{prefix}{OS.sep}{self.label}_predictions_{setting_idx}.csv'
            if OS.path.exists(prediction_storage):

                if not be_warned:
                    raise Warning('Skipping an existing prediction. Make sure to empty your "predictions" folder if you intend to re-sample.')
                    be_warned = True
                # skip this setting if it was stored before
                pred = PD.read_csv(prediction_storage, sep = ';')
                pred.set_index('sample_nr', inplace = True)
                predictions.append(pred)
                continue


            prediction_data = {}
            lower_level_storage = {}
            for label in self.shared:
                # print (label)

                param = "_".join(label.split("_")[:-1])

                # check if param is in the settings
                if label not in setting_values.index:
                    # print (setting_values)
                    # print (label, setting_values.index)
                    raise IOError(f'missing values for {setting_idx} {param}. Add it to sampling_settings and optionally lower_level_predictions.')

                # multi-step modeling: some values were sampled before
                if setting_values[label] is None:
                    if lower_level_predictions is None:
                        raise IOError(f'please set a value for {setting_idx} {param}.')

                    # sampled from a second level model
                    prediction_data[label] = lower_level_predictions[(setting_idx, label)]

                    if not ('_group' == label[-6:]):
                        lower_level_storage[label] = lower_level_predictions[(setting_idx, label)].ravel()[:n_samples]
                    else:
                        # turn group matrix into categorical
                        grpmat = lower_level_predictions[(setting_idx, label)]
                        lower_level_storage[label] = NP.array(NP.sum( grpmat \
                                                           @ (NP.arange(grpmat.shape[1]).reshape((-1,1)) + 1) \
                                                            , axis = 1).ravel()[:n_samples], dtype = int)



                # single-step modeling
                else:
                    # self.shared[param].set_value( setting_values[label] * NP.ones((5*self.n['observations'],1)) )
                    # can we set arbitrary n_samples? yes!
                    # prediction_data[label] = setting_values[label] * NP.ones((self.n['observations'], 1)) # works with MvNormal but not MvStudentT out of the box
                    # prediction_data[label] = setting_values[label] * NP.ones((n_samples, 1)) # works with MvNormal after mu.shape trick
                    prediction_data[label] = setting_values[label] * NP.ones((n_samples, 1)) # works with MvNormal after mu.shape trick

            # # print ({lab: dat.shape for lab, dat in prediction_data.items()})
            # with self.model:

            # print ('setting', setting_idx)
            # self.PrintAllShapes()

            # for k, v in self.shared.items():
            #     PrintShape(k, v)

            # print ('now, actually sampling')
            # predictive sampling
            with self.model:
                if self.verbose:
                    self.Print(f'sampling {setting_idx} {setting_values.to_dict()}')
                if True:#  not pymc_v3:
                    # 20221006 this seems not to be fixed in the actual pymc4 release?
                    # -> 20221020 it is fixed! https://github.com/pymc-devs/pymc/issues/6230

                    # simplified procedure in pymc4+
                    # print (prediction_data['μ_wrist_data'].shape)
                    PM.set_data(prediction_data)

                    # this is how "samples" works nowadays (https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html#Examples)
                    trace_thinned = self.trace.sel(draw=slice(None, (None if thinning is None else thinning-1), None))
                    sampling = PM.sample_posterior_predictive(  trace_thinned \
                                                              #, keep_size = False \
                                                              #, samples = n_samples \
                                                              #, samples = 1 \
                                                              #, samples = self.n['observations'] \
                                                              #, progressbar = True \
                                                                , return_inferencedata = False \
                                                              )[f'{self.label}::posterior']

                    if thinning is not None:
                        # when there are more samples requested than available in the trace
                        if thinning > sampling.shape[0]:
                            print ('only {posterior.shape[0]} samples available; thinning ({thinning}) requested but not applied.')

                    print (sampling.shape)
                    # print (sampling)

                else:
                    # DEPRECATED
                    count = 0
                    sampling = []
                    while count < n_samples:
                        prediction_cp = {label: data[count:count+self.n['observations'], :] \
                                         for label, data in prediction_data.items() \
                                         }
                        # print ([(v.shape[0], self.n['observations']) for v in prediction_cp.values()])
                        # print ([v.shape[0] == self.n['observations'] for v in prediction_cp.values()])
                        if not all( [v.shape[0] == self.n['observations'] for v in prediction_cp.values()]):
                            raise IOError('observable shape mismatch; have you sampled sufficient lower level predictions?')
                            break

                        PM.set_data(prediction_cp)
                        sampling_outcome = PM.sample_posterior_predictive(  self.trace \
                                                                  #, samples = 1 \
                                                                  , progressbar = False \
                                                                  , return_inferencedata = False \
                                                                  )
                        # print (sampling_outcome)
                        sampling.append( \
                                    sampling_outcome[f'{self.label}::posterior'] \
                                         )
                        count += self.n['observations']
                        print (f'sampling setting {setting_idx}: {count}/{n_samples}' + ' '*10, end = '\r', flush = True)

                    sampling = NP.concatenate(sampling, axis = 0)

                    sampling = sampling[:n_samples, :]
                # print ('\n', sampling.shape)
                # print ('\n', sampling[f'lbodymass_posterior'].shape)
                # mental note: the outcome.shape is always n_samples x shared.shape
                # print (sampling)

            # store prediction
            pred = PD.DataFrame.from_dict(dict( \
                                setting = setting_idx \
                                # , 'observable': observable \
                                # , 'samples': sampling[f'{self.label}_posterior'][:, :, obs_nr].ravel() \
                                , ** lower_level_storage \
                                , ** { \
                                     observable: sampling[:, :, obs_nr].ravel() \
                                     for obs_nr, observable in enumerate(self['observables']) \
                                    }
                                ))

            pred.index.name = 'sample_nr'

            pred.to_csv(prediction_storage, sep = ';')
            predictions.append(pred)



        # putting all together
        predictions = PD.concat(predictions, axis = 0)

        # # restore actual data values in shared variables
        # print ('pre')
        # for param, shared in self.shared.items():
        #     print(shared_storage[param])
        #     print(self.shared[param].get_value())
        #     self.shared[param].set_value(shared_storage[param])
        # print ('post')
        # # TODO: restoring shared values does not work.

        # keep samples
        self.predictive_samples = predictions

        return predictions



    def PrintAllShapes(self):
        # print the shape of all model parts (for debugging)
        print (self.shared)
        return

        print (dir(self.model))
        print('cont', self.model.cont_vars)
        print('disc', self.model.disc_vars)
        print('named', self.model.named_vars)
        print('vars', self.model.vars)
        # print(self.model.coords)
        # print(self.model.deterministics)

        # print(self.model.RV_dims)
        # print(self.model.basic_RVs)
        # print(self.model.free_RVs)
        # print(self.model.observed_RVs)
        # for var in self.model.vars:
        #     PrintShape(str(var), var)
        # for var, shared in self.shared.items():
        #     PrintShape(var, shared)
        # for var in self.model.free_RVs:
        #     PrintShape(str(var), var)
        for name, var in self.model.named_vars.items():
            PrintShape(name, var)


#______________________________________________________________________
    def Save(self, filename: str):
        # print (PM.summary(self.trace))
        # pickle everything associated with the model.
        with open(filename, 'wb') as storefile:
            PI.dump({ \
                      'label': self.label \
                    , 'data': self.data.to_dict() \
                    , 'model': self.model \
                    , 'trace': self.trace \
                    , 'samplingtime': self.samplingtime \
                    , 'settings': self.settings.ToDict() \
                    , 'shared': self.shared \
                    , 'start_values': self.start_values \
                    , 'predictive_settings': None if self.predictive_settings is None else self.predictive_settings.to_dict() \
                    , 'predictive_samples': None if self.predictive_samples is None else self.predictive_samples.to_dict() \
                    , 'verbose': self.verbose \
                    }, storefile)


    @classmethod
    def Load(cls, filename: str):
        # open a model from storage.
        # print (filename)
        with open(filename, 'rb') as storefile:
            loaded = PI.load(storefile)

        data = PD.DataFrame.from_dict(loaded['data'])
        # restore categoricals
        if 'ageclass' in data.columns:
            data['ageclass'] = PD.Categorical(data['ageclass'].values \
                                     , categories=["adult", "adolescent", "infant"]
                                     , ordered = True \
                                     )

        for param in ['subject', 'sex']:
            if param in data.columns:
                data[param] = PD.Categorical(data[param].values \
                                     , ordered = True \
                                     )


        settings = Settings(settings = loaded['settings'])
        mdl = cls(data, settings)
        mdl.start_values = loaded['start_values']
        mdl.predictive_settings = None if loaded.get('predictive_settings', None) is None \
                                 else PD.DataFrame.from_dict(loaded['predictive_settings'])
        mdl.predictive_samples = None if loaded.get('predictive_samples', None) is None \
                                 else PD.DataFrame.from_dict(loaded['predictive_samples'])
        if 'label' in loaded.keys():
            mdl.label = loaded['label']
        else:
            # forgot to save this in old models
            mdl.label = filename
        mdl.model = loaded['model']
        mdl.trace = loaded['trace']
        mdl.samplingtime = loaded['samplingtime']
        mdl.shared = loaded['shared']
        mdl.verbose = loaded.get('verbose', False)

        print (f'loaded {filename}!')

        return mdl


##############################################################################
### Model Diagnostics                                                      ###
##############################################################################
predictor_translator = { \
                         'sex_is_male': 'female \(\\rightarrow\) male' \
                       , 'ageclass_is_adolescent': 'adlt \(\\rightarrow\) adol' \
                       , 'ageclass_is_infant': 'adlt \(\\rightarrow\) inft' \
                       , 'lbodymass': 'log.bodymass' \
                       , 'clbodymass': 'c.log.bodymass' \
                       , 'lleg_m': 'log.leg length' \
                       , 'clleg_m': 'c.log.leg length' \
                       , 'lbmi': 'log.bmi' \
                       , 'clbmi': 'c.log.bmi' \
                       , 'μ_trunk': 'trunk angle'
                       , 'stride1': 'stride PC1'
                       , 'stride2': 'stride PC2'
                       , 'std': '\\(\epsilon\\)' \
                         , 'sex_is_M': 'female \(\\rightarrow\) male' \
                         , 'clr_fl_log': 'log. FL clearance' \
                         , 'clr_hl_log': 'log. HL clearance' \
                         , 'duty_fl_log': 'FL duty factor' \
                         , 'duty_hl_log': 'HL duty factor' \
                         , 'distance_diml': 'd.s. distance' \
                         , 'frequency_diml': 'd.s. frequency' \
                         , 'speed_diml': 'diml. speed' \
                         , 'head_angle': 'head angle' \
                         , 'phase_hl': 'hindlimb phase' \
                         , 'μ_hip': 'mean hip angle' \
                         , 'α_hip': 'hip eROM' \
                         , 'μ_knee': 'mean stifle angle' \
                         , 'α_knee': 'stifle eROM' \
                         , 'μ_ankle': 'mean tarsal angle' \
                         , 'α_ankle': 'tarsal eROM' \
                         , 'μ_shoulder': 'mean shoulder angle' \
                         , 'α_shoulder': 'shoulder eROM' \
                         , 'μ_elbow': 'mean elbow angle' \
                         , 'α_elbow': 'elbow eROM' \
                         , 'μ_wrist': 'mean carpal angle' \
                         , 'α_wrist': 'carpal eROM' \
                         ,  'PC1': 'CC1' \
                         ,  'PC2': 'CC2' \
                         ,  'PC3': 'CC3' \
                         ,  'PC4': 'CC4' \
                         ,  'PC5': 'CC5' \
                         ,  'PC6': 'CC6' \
                         ,  'PC7': 'CC7' \
                         ,  'PC8': 'CC8' \
                         ,  'PC9': 'CC9' \
                         , 'PC10': 'CC10' \
                         , 'PC11': 'CC11' \
                         , 'PC12': 'CC12' \
                        }
multilevel_translator = { \
                          '': 'only on multilevel; to be refined' \
                        #   'clbodymass': 'c.log.bodymass/ageclass (a/a/i)' \
                        # , 'lbodymass': 'log.bodymass/ageclass (a/a/i)' \
                         }

observable_translator = { \
                           'leg m': 'leg length (m)' \
                         , 'bodymass': 'body mass (kg)' \
                         , 'bmi': 'bmi (kg/m)' \
                          , 'lleg m': 'log leg length (m)' \
                         , 'lbodymass': 'log body mass (kg)' \
                         , 'lbmi': 'log bmi (kg/m)' \
                          , 'distance diml': 'd.s. distance'
                          , 'frequency diml': 'd.s. frequency'
                          , 'speed diml': 'd.s. speed'
                          , 'μ trunk': 'trunk angle' \
                          , 'μ ihip': 'hip angle' \
                          , 'α ihip': 'hip amplitude' \
                          , 'μ iknee': 'stifle angle' \
                          , 'α iknee': 'stifle amplitude' \
                          , 'μ iankle': 'tarsal angle' \
                          , 'α iankle': 'tarsal amplitude' \
                          , 'μ iwrist': 'carpal angle' \
                          , 'α iwrist': 'carpal amplitude' \
                          , 'age': 'age (h)' \
                          , 'age_log': 'age (log)' \
                          , 'log_age': 'age (log)' \
                          , 'age log': 'age (log)' \
                          , 'morpho1': 'size PC1' \
                          , 'weight': 'mass (kg)' \
                          }


def CalculateCorrelation(data, observables, level):
    # calculate all the relevant correlation measures

    data_matrix = data.loc[:, observables].values

    # centralize
    if level == 'population':
        data_matrix -= NP.mean(data_matrix, axis = 0)
    else:
        # groupwise centralization
        data_centralized = NP.zeros_like(data_matrix)
        grouping = data[level].cat.codes.values
        for ig in NP.unique(grouping):
            data_centralized[grouping==ig, :] = data_matrix[grouping==ig, :] \
                                            - data_matrix[grouping==ig, :].mean(axis=0)

        data_matrix = data_centralized


    # covariance
    covariance_matrix = NP.cov(data_matrix.T, ddof = 1)

    # cholesky
    cholesky_matrix = NP.linalg.cholesky(covariance_matrix)

    # covariance_matrix = PM.Deterministic('covariance_matrix' \
    #                                      , AT.dot(cholesky_matrix, cholesky_matrix.T) \
    #                                      )
    standard_deviations = NP.sqrt(NP.diag(covariance_matrix))
    correlation_matrix = NP.diag(standard_deviations**-1) \
                         .dot(covariance_matrix.dot(NP.diag(standard_deviations**-1)))
    # print (correlation_matrix)
    # print (NP.corrcoef(data_matrix.T, ddof = 1))

    return correlation_matrix, covariance_matrix, cholesky_matrix, standard_deviations, data_matrix



def GetFixedCorrelation(model):
    data = model.data
    observables = model['observables']
    level = model['intercept_level']

    correlation_matrix, _, _, _, data_matrix = CalculateCorrelation(data, observables, level)

    ReplaceUnderscore = lambda txt: txt.replace('_', ' ')
    results_table = PD.DataFrame( \
                                   NP.round(correlation_matrix, 2) \
                                 , index = map(ReplaceUnderscore, observables) \
                                 , columns = map(ReplaceUnderscore, observables) \
                                 )
    for col in results_table.columns:
        results_table.loc[:, col] = results_table[col].map('\({:+.2f}\)'.format)

    for idx in range(model.n['observables']):
        for col in range(model.n['observables']):
            if idx == col:
                results_table.iloc[idx, col] = '\(1\)'
                continue

            # significance?
            r, p = STATS.pearsonr(data_matrix[:, idx], data_matrix[:, col])
            # print (idx, col, r, p)
            if p < 0.05:
                results_table.iloc[idx, col] = results_table.iloc[idx, col] + ' *'



    results_table.index = [observable_translator.get(idx, idx) for idx in results_table.index.values]
    results_table.columns = [observable_translator.get(idx, idx) for idx in results_table.columns.values]



    output = results_table.to_markdown() \
                          .replace(':', '-') \
                          .replace('-|-', '-+-')

    return output



def GetPosteriorCorrelation(model):
    summary = model.Summary()
    # print (summary)

    row_selection = [idx for idx in summary.index.values \
                     if 'cholesky_corr' in idx[1]]

    result_corr = {}
    for measure in ['mean', 'hdi_3%', 'hdi_97%']:
        correlation_matrix = summary.loc[row_selection, measure]

        correlation_matrix.index = PD.MultiIndex.from_tuples([ \
                                                               ExtractIdx(idx[1]) \
                                                               for idx in correlation_matrix.index.values \
                                                              ])


        correlation_matrix = NP.stack( [ \
                                         correlation_matrix.loc[i, :].values \
                                         for i in range(model.n['observables']) \
                                        ], axis = 1)

        result_corr[measure] = correlation_matrix


    ReplaceUnderscore = lambda txt: txt.replace('_', ' ')
    results_table = PD.DataFrame( \
                            NP.array([f'\({num:+0.2f}\)' for num in result_corr['mean'].ravel()] \
                                     ).reshape((model.n['observables'], -1))\
                                 , index = map(ReplaceUnderscore, model['observables']) \
                                 , columns = map(ReplaceUnderscore, model['observables']) \
                                 )

    for idx in range(model.n['observables']):
        for col in range(model.n['observables']):
            if idx == col:
                results_table.iloc[idx, col] = '\(1\)'
                continue

            if result_corr['hdi_3%'][idx, col] * result_corr['hdi_97%'][idx, col] > 0:
                results_table.iloc[idx, col] = results_table.iloc[idx, col] + ' *'


    results_table.index = [observable_translator.get(idx, idx) for idx in results_table.index.values]
    results_table.columns = [observable_translator.get(idx, idx) for idx in results_table.columns.values]

    output = results_table.to_markdown() \
                          .replace(':', '-') \
                          .replace('-|-', '-+-')

    return output


# effects list
def GetPredictorEffects(model, transpose = False, categorize_rows = False, exclude_obs = []):
    summary = model.Summary()
    # print (summary)
    observables = [obs for obs in model['observables'] if not obs in exclude_obs]

    predictor_columns = ['intercept'] + list(model.settings.GetPredictors().index.values) + ['std']
    # print (predictor_columns)

    # swap two elements for categorical order
    try:
        a, b = predictor_columns.index('head_angle'), predictor_columns.index('phase_hl')
        predictor_columns[b], predictor_columns[a] = predictor_columns[a], predictor_columns[b]
    except ValueError as ve:
        print (ve)
        print (predictor_columns)


    # manual calc of stds
    if model['obscov_fixed'] or not model['obs_multivariate']:
        corr, cov, chol, std, dmat = CalculateCorrelation(model.data, observables, model['intercept_level'])
        print (std, observables)

    results = PD.DataFrame(index = observables \
                           , columns = predictor_columns)

    for obsl_nr, observable in enumerate(observables):
        # print ('\n####', observable)
        sub_summary = summary.loc[observable, :]
        sub_summary.index = sub_summary.index.swaplevel(0,1)
        if model['obscov_fixed']:
            sub_summary.loc[('std', f'std[{obsl_nr}]'), 'mean'] = std[obsl_nr]
        if not model['obs_multivariate']:
            # print (summary.index.values)
            # print (sub_summary.index.values)
            sub_summary.loc[('std', f'std[{obsl_nr}]'), 'mean'] = sub_summary.loc[('residual', f'residual[0, {obsl_nr}]'), 'mean'] # rename?

        multilevels = []
        for predictor in predictor_columns:
            effect = sub_summary.loc[predictor, :]

            if effect.shape[0] > 1:
                multilevels.append(predictor)
                effect.index = ["{}[{},{}]".format(predictor, *ExtractIdx(idx)) \
                                for idx in effect.index.values]

                values = []
                for idx, eff in effect.iterrows():
                    if predictor in ['std']:
                        val = f"\(\pm {eff['mean']:.2f}\)"
                    else:
                        val = f"\({eff['mean']:+.2f}\)"

                    if (eff['hdi_3%'] * eff['hdi_97%'] > 0) and not (predictor in ['intercept', 'std']):
                        val = val + ' *'

                    values.append(val)

                results.loc[observable, predictor] = " / ".join(values)

            else:
                if predictor in ['std']:
                    results.loc[observable, predictor] = f"\(\pm {effect['mean'].iloc[0]:.2f}\)"
                else:
                    results.loc[observable, predictor] = f"\({effect['mean'].iloc[0]:+.2f}\)"

                if (effect['hdi_3%'].iloc[0] * effect['hdi_97%'].iloc[0] > 0) and not (predictor in ['intercept', 'std']):
                    results.loc[observable, predictor] = results.loc[observable, predictor] + ' *'

    # print (results)

    ReplaceUnderscore = lambda txt: txt.replace('_', ' ')

    # Predictor Effects Table
    results.index = [ReplaceUnderscore(idx) for idx in results.index.values]
    results.index = [observable_translator.get(idx, idx) for idx in results.index.values]
    results.columns = [multilevel_translator.get(col, col) for col in results.columns.values \
                       ]
    results.columns = [predictor_translator.get(col, col) for col in results.columns.values]

    if transpose:
        results = results.T


    if categorize_rows:
        rows_dict = {\
                       r'female \(\rightarrow\) male': 'subject' \
                     , r'log. FL clearance': 'gait' \
                     , r'log. HL clearance': 'gait' \
                     , r'FL duty factor': 'gait' \
                     , r'HL duty factor': 'gait' \
                     , r'd.s. distance': 'gait' \
                     , r'd.s. frequency': 'gait' \
                     , r'diml. speed': 'gait' \
                     , r'head angle': 'gait' \
                     , r'hindlimb phase': 'gait' \
                     , r'mean hip angle': 'dyn.p.' \
                     , r'hip eROM': 'dyn.p.' \
                     , r'mean stifle angle': 'dyn.p.' \
                     , r'stifle eROM': 'dyn.p.' \
                     , r'mean tarsal angle': 'dyn.p.' \
                     , r'tarsal eROM': 'dyn.p.' \
                     , r'mean shoulder angle': 'dyn.p.' \
                     , r'shoulder eROM': 'dyn.p.' \
                     , r'mean elbow angle': 'dyn.p.' \
                     , r'elbow eROM': 'dyn.p.' \
                     , r'mean carpal angle': 'dyn.p.' \
                     , r'carpal eROM': 'dyn.p.' \
                     , r'CC1': 'coord.' \
                     , r'CC2': 'coord.' \
                     , r'CC3': 'coord.' \
                     , r'CC4': 'coord.' \
                     , r'CC5': 'coord.' \
                     , r'CC6': 'coord.' \
                     , r'CC7': 'coord.' \
                     , r'CC8': 'coord.' \
                     , r'CC9': 'coord.' \
                     , r'CC10': 'coord.' \
                     , r'CC11': 'coord.' \
                     , r'CC12': 'coord.' \
                     }
        # print (results.index)
        results.index = PD.MultiIndex.from_tuples([(rows_dict.get(row, 'model'), row) for row in results.index.values])
        results.index.names = ['category', 'parameter']
        results.reset_index(inplace = True)
        print (results)

    output = results.to_markdown(floatfmt = '+.2f', tablefmt = "orgtbl", headers = results.columns) \

    return output



##############################################################################
### Model Comparison                                                       ###
##############################################################################
def ModelComparison(models, plot = False):

    modelstore = {model.label: model.trace for model in models}
    #modelstore = {model.model: model.trace for model in models}
    comp_waic = AZ.compare(modelstore, ic = "loo")
    # comp_waic.index = [model.name for model in comp_waic.index.values]

    if plot:
        AZ.plot_compare(comp_waic)
        PLT.show()


    return comp_waic



##############################################################################
### Simulation                                                             ###
##############################################################################
def SimulateData(settings: dict = None) -> PD.DataFrame:
    # quick generation of simulated data for testing

    if settings is None:
        ### simulation settings
        settings = {}

        # isolated effects
        settings['sex_pc2'] = True # categorical effect
        settings['clearance_pc1'] = True # slope effect
        settings['age_pc2'] = True # "random intercept"
        settings['age_speedslopes_pc1'] = True # "random slope"

        # overlapping effects
        settings['sex_speed'] = False # speed is different for the sexes # no problem
        settings['corr_c_s'] = True # a correlation between clearance and speed

        # correlated Posterior
        settings['corr_post'] = False # correlated PCs



    ref = PD.DataFrame.from_dict( {  \
                                'speed_rel': [ 6.5, 2.0 , 3.4 ] \
                              , 'clearance': [ 0.36 , 0.05 , 0.27 ] \
                              , 'PC1': [ 0. , 0.4 , -0.554 ] \
                              , 'PC2': [ 0. , 0.31 , -0.52 ] \
                               } \
                             ).T
    ref.columns = ['mean', 'std', 'min']


    NP.random.seed(42)

    n_subjects = 99
    subjects = {'subject_idx': NP.arange(n_subjects)}
    subjects['subject'] = [f's{nr:02.0f}' for nr in subjects['subject_idx']]
    subjects['sex'] = NP.random.choice(['m', 'f'], size = n_subjects)
    subjects['ageclass'] = NP.random.choice(['adult', 'infant'], size = n_subjects)

    subjects = PD.DataFrame.from_dict(subjects).set_index('subject_idx', inplace = False)

    n = 990
    data = {'idx': NP.arange(n)}


    data['subject_idx'] = NP.random.choice(subjects.index.values, size = n)

    for col in ['speed_rel', 'clearance', 'PC1', 'PC2']:
        data[col] = NP.random.normal( \
                                        ref.loc[col, 'mean'] \
                                      , ref.loc[col, 'std'] \
                                      , size = n \
                                     )

    Standardized = lambda vec: (vec - NP.nanmean(vec)) / (2*NP.nanstd(vec))

    # clearance and speed are correlated (~ -0.4)
    data['speed_rel'] = data['speed_rel'] \
        - float(settings['corr_c_s']) * 1.9*Standardized(data['clearance'])

    data = PD.DataFrame.from_dict(data).set_index('idx', inplace = False)

    data = data.join(subjects, how = 'left', on = 'subject_idx')

    # sex effect on relative speed
    data.loc[data['sex'] == 'f', 'speed_rel'] -= 0.8 * float(settings['sex_speed'])
    data.loc[data['sex'] == 'm', 'speed_rel'] += 1.2 * float(settings['sex_speed'])

    # sex effect on PC2
    data.loc[data['sex'] == 'f', 'PC2'] += 0.35 * float(settings['sex_pc2'])
    data.loc[data['sex'] == 'm', 'PC2'] -= 0.15 * float(settings['sex_pc2'])

    # age effect on PC2
    data.loc[data['ageclass'] == 'infant', 'PC2'] += 0.2 * float(settings['age_pc2'])

    # age-class depd speed slope of PC1
    # and clearance effect on PC1
    age_slopes = {'adult': 0.8, 'infant': -0.2}
    data['PC1'] = [ \
                    data.loc[idx, 'PC1'] \
                    - float(settings['clearance_pc1']) * 0.5 * Standardized(data.loc[:, 'clearance'])[idx] \
                    + float(settings['age_speedslopes_pc1']) * age_slopes[data.loc[idx, 'ageclass']] * Standardized(data.loc[:, 'speed_rel'])[idx] \
                    for idx in range(n) \
                   ]

    # plausible value ranges
    data.loc[:, 'speed_rel'] -= NP.nanmin(data['speed_rel'].values) - ref.loc['speed_rel', 'min']

    # PC2 and PC1 are correlated
    data['PC2'] = data['PC2'] \
                - float(settings['corr_post']) * 0.5*Standardized(data['PC1'].values)


    # categoricals
    for param in ['ageclass', 'sex', 'subject_idx']:
        data[param] = PD.Categorical(data[param].values \
                                 , ordered = True \
                                 )

    for cat, reference_value in { \
                                  'sex': 'f' \
                                , 'ageclass': 'adult' \
                                 }.items():
        for val in NP.unique(data[cat].values):
            # skip the reference value
            if val == reference_value:
                continue

            # create a boolean
            data[f'{cat}_is_{val}'] = NP.array(data[cat].values == val, dtype = float)

    for col in ['speed_rel', 'clearance']:
        data.loc[:, col] -= NP.mean(data[col].values)

    # print (STATS.pearsonr(data['speed_rel'].values, data['clearance'].values))
    # print (STATS.pearsonr(data['PC1'].values, data['PC2'].values))

    # SUMMARY:
    # - correlation of clearance and speed (-0.4)
    # - sex-depd speed (+2.)
    # - PC1 ~ clearance (-0.5)
    # - PC1 ~ ageclass*speed (infant: -0.2, adult: 0.8)
    # - PC2 ~ sex (f = m + 0.5)
    # - PC2 ~ ageclass (-0.5)
    # - PC1 ~ PC2
    return data




##############################################################################
### Testing                                                                ###
##############################################################################

def QuickTesting():
    # test with simulated data
    data = SimulateData()

    # choose settings
    settings = Settings(dict( \
                            observables = ['PC1'] \
                            # observables = ['PC1', 'PC2'] \
                          , robust = False \
                          , predictors = { \
                                           'ageclass': ['population', None] \
                                           # , 'sex': ['population', None] \
                                           # , 'clearance': ['population', 'stride'] \
                                           # , 'speed_rel': ['population', 'stride'] \
                                           }
                          , priors = {'ageclass': {'mu': 0., 'sigma': 1.} \
                                         , 'observables': {'eta': 1., 'mu': NP.zeros((len(self['observables']), ))} \
                                         } \
                          ))
                        # fun fact: this reminds me of the clunky R implementation I always hated :/

    model = Model( \
                   data = data \
                 , settings = settings \
                 , label = 'test' \
                 , verbose = True \
                 )

    model.Sample(n_steps = 1024, cores = 4, chains = 4)

    print (model.Summary())

    # try saving
    model.Save('test.mdl')

    # try loading
    model = Model.Load('test.mdl')

    print (model.Summary())

    PM.plot_trace(model.trace)
    PLT.show()


def TestModelComparison():
    # test with simulated data
    data = SimulateData()

    # choose settings
    settings = Settings(dict( \
                            observables = ['PC1', 'PC2'] \
                          , robust = False \
                          , predictors = { \
                                           'ageclass_is_infant': ['population', None] \
                                           , 'sex_is_m': ['population', None] \
                                           , 'clearance': ['population', 'stride'] \
                                           , 'speed_rel': ['population', 'stride'] \
                                           }
                          , priors = {  'ageclass': {'mu': 0., 'sigma': 1.} \
                                      , 'sex': {'mu': 0., 'sigma': 1.} \
                                      , 'clearance': {'mu': 0., 'sigma': 1.} \
                                      , 'speed_rel': {'mu': 0., 'sigma': 1.} \
                                      , 'indercept': {'mu': 0., 'sigma': 1.} \
                                      , 'observables': {'eta': 1., 'mu': NP.zeros((1, 2))} \
                                      } \
                          ))


    all_settings = {}

    # single
    settings1 = settings.Copy()
    settings1['predictors']['clearance'] = ['population', None]
    settings1['predictors']['speed_rel'] = ['population', None]
    all_settings['single'] = settings1

    # muva
    settings1 = settings.Copy()
    settings1['predictors']['clearance'] = ['population', 'stride']
    settings1['predictors']['speed_rel'] = ['population', 'stride']
    all_settings['muva'] = settings1

    # mule
    settings1 = settings.Copy()
    settings1['predictors']['clearance'] = ['population', None]
    settings1['predictors']['speed_rel'] = ['ageclass', None]
    all_settings['mule'] = settings1

    # vale
    settings1 = settings.Copy()
    settings1['predictors']['clearance'] = ['ageclass', 'stride']
    settings1['predictors']['speed_rel'] = ['ageclass', 'stride']
    all_settings['vale'] = settings1


    comparison = []
    for label, setting in all_settings.items():
        model = Model( \
                   data = data \
                 , settings = setting \
                 , label = label \
                 , verbose = False \
                 )

        model.Sample(n_steps = 1024, cores = 4, chains = 4, progressbar = True)

        print (model.Summary())
        comparison.append(model)

    ModelComparison(comparison)



if __name__ == "__main__":
    # QuickTesting()

    TestModelComparison()
