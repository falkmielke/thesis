#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20210920

"""
take a stored model and extract tables for inclusion in a manuscript
"""


################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import numpy as NP          # numerical analysis
import pandas as PD         # data management

# load self-made toolboxes
SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import ConfigToolbox as CONF # project configuration
import IOToolbox as IOT      # common IO functions
import ModelToolbox as MT    # probabilistic modeling
import EigenToolbox as ET    # stride parameter pca



## observable correlations
def ExtractCorrelations(label, model):

    if model['obscov_fixed']:
        print ('\textract correlation fixed')
        correlation_table = MT.GetFixedCorrelation(model)
    else:
        print ('\textract correlation sampled')
        correlation_table = MT.GetPosteriorCorrelation(model)

    print (correlation_table)
    with open(f'results/{label}_correlation.org', 'w') as fi:
        print(correlation_table, file = fi)

    if label == 'bodyproportions':
        print ('\textract correlation fixed')
        correlation_table = MT.GetFixedCorrelation(model)
        print (correlation_table)
        with open(f'results/{label}_empiricalcorrelation.org', 'w') as fi:
            print(correlation_table, file = fi)



## predictor effects
def ExtractPredictorEffects(label, model):

    print ('\textract predictor effects')
    effects_table = MT.GetPredictorEffects(model, transpose = True, categorize_rows = True, exclude_obs = ['weight_cac']) # , 'age_log'

    print (effects_table)
    effects_table = '\\footnotesize\n'+effects_table
    with open(f'results/{label}_predictors.org', 'w') as fi:
        print(effects_table, file = fi)

### stride PCA

def LoadStridePCA():
    filename = 'data/stride_parameters.pca'
    pca = ET.PrincipalComponentAnalysis.Load(filename)
    # print (pca)

    table = pca.ToORG()
    table.columns = [MT.observable_translator.get(col.replace('_', ' '), col) for col in table.columns]
    output = table.to_markdown(floatfmt = '+.2f', tablefmt = "orgtbl", headers = table.columns)
    # output = output.replace(':', '-') \
    #                       .replace('-|-', '-+-')

    print (output)
    with open(f'results/stride_pca.org', 'w') as fi:
        print(output, file = fi)


##############################################################################
### Mission Control                                                        ###
##############################################################################
if __name__ == "__main__":

    LoadStridePCA()

    n_iterations = 16384 # 1024 # 32768
    models = { \
                 'age': f'models/05_age/reference_{n_iterations}.mdl' \
               # , 'bodyproportions': f'models/proportions/bodyproportions_{n_iterations}.mdl' \
               # , 'stride': f'models/stride/reference_{n_iterations}.mdl' \
               # , 'posture': f'models/posture/reference_{n_iterations}.mdl' \
               # , 'coordination': f'models/coordination/reference_{n_iterations}.mdl' \
              }

    for label, model_storefile in models.items():
        print ('#'*16)
        print (label)

        model = MT.Model.Load(model_storefile)

        if model.settings['obs_multivariate']:
            ExtractCorrelations(label, model)
        ExtractPredictorEffects(label, model)
