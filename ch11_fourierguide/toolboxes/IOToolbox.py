#!/usr/bin/env python3

"""
common I/O functions
and little helpers
"""

################################################################################
### Libraries                                                                ###
################################################################################
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.interpolate as INTP # interpolation

import ConfigToolbox as CONF # project configuration
import FourierToolbox as FT # Fourier Series toolbox


################################################################################
### Data I/O                                                                 ###
################################################################################

def LoadStrideCycles(data_file: str) -> PD.DataFrame:
    # load and prepare the main kinematics data file

    # load the csv
    data = PD.read_csv(data_file, sep = ';')

    # adjust index
    data.set_index(['subject', 'sheet_nr', 'stride_idx', 'time'], inplace = True)

    return data


def LoadJoints(config) -> dict:
    # convert joint dictionary from a config (loaded as string)
    # to the desired joint definitions
    return {int(key): eval(value) for key, value in config['joints'].items()}


# create the usual file label from a data frame row label
LabelRectify = lambda label: f"{label[2]:02.0f}_{label[0]}{'' if label[1] == 0 else label[1]}"



################################################################################
### Limbs                                                                    ###
################################################################################

def LoadLimbs(config) -> dict:
    # load stride cycle data as "limb", i.e. joints associated

    # loop cycles
    cycle_list = PD.read_csv(config['datafiles']['stridetimes_file'], sep = ';').set_index('stride_idx', inplace = False).loc[:, ['spreadsheet']]

    # check excluded cycles
    excluded = list(map(int, config['exclude'].keys()))

    # ... and store limbs on the way
    limbs = {}
    skipped = 0
    for idx, stride in zip(cycle_list.index.values, cycle_list.values.ravel()):

        # skip excluded cycles
        if idx in excluded:
            skipped += 1
            continue

        # data labeling
        label = f'{idx:02.0f}_{stride}'
        print (f'({idx}/{len(cycle_list)})', label, ' '*16, end = '\r', flush = True)

        # store the results
        fsd_file = f"{config['folders']['jointfsd']}{OS.sep}{label}.csv"
        coefficient_df = PD.read_csv(fsd_file, sep = ';').set_index(['re_im', 'coeff'], inplace = False)

        # create a limb
        lmb = FT.NewLimb(coefficient_df, label = label, coupled = True)

        # lmb.PrepareRelativeAlignment(test_joint, skip_rotate = True, skip_center = False, skip_normalize = False)
        limbs[label] = lmb

    print (f"Limbs loaded ({len(limbs)} cycles, {skipped} excluded)", " "*16)

    return limbs


def ReLoadLimbs(config) -> dict:
    # load limbs from storage files

    # loop cycles
    cycle_list = PD.read_csv(config['datafiles']['stridetimes_file'], sep = ';').set_index('stride_idx', inplace = False).loc[:, ['spreadsheet']]

    # check excluded cycles
    excluded = list(map(int, config['exclude'].keys()))

    # ... and store limbs on the way
    limbs = {}
    for idx, stride in zip(cycle_list.index.values, cycle_list.values.ravel()):
        label = f'{idx:02.0f}_{stride}'
        print (f'({idx}/{len(cycle_list)})', label, ' '*16, end = '\r', flush = True)

        # skip excluded cycles
        if idx in excluded:
            continue

        # load limb
        limb_file = f"{config['folders']['limbs']}{OS.sep}{label}_{config['analysis']['reference_joint']}.lmb"
        limbs[label] = FT.NewLimb.Load(limb_file)

    print (f"Limbs loaded ({len(limbs)} cycles)", " "*16)

    return limbs


def StoreLimbs(config, limbs: dict) -> None:
    # store limbs in a shelf
    for label, lmb in limbs.items():
        limb_file = f"{config['folders']['limbs']}{OS.sep}{label}_{config['analysis']['reference_joint']}.lmb"

        lmb.Save(limb_file)


################################################################################
### Analysis Data                                                            ###
################################################################################
# transformations
LogTraFo = lambda vec: NP.log(vec)
UnLogTraFo = lambda lvec: NP.log(lvec)
Center = lambda vec: vec - NP.mean(vec)

# data i/o
def LoadAnalysisData(config: CONF.Config) -> PD.DataFrame:

    # load data
    data = PD.read_csv(config['datafiles']['analysis_data'], sep = ';') \
             .set_index('cycle_idx', inplace = False)

    # make categoricals
    # (i) explicit categories
    data['ageclass'] = PD.Categorical(data['ageclass'].values \
                                 , categories=["adult", "adolescent", "infant"]
                                 , ordered = True \
                                 )

    # (ii) default categories
    for param in ['subject', 'sex']:
        data[param] = PD.Categorical(data[param].values \
                                 , ordered = True \
                                 )

    # (iii) booleans
    for cat, reference_value in { \
                                  'sex': 'female' \
                                , 'ageclass': 'adult' \
                                 }.items():
        for val in NP.unique(data[cat].values):
            # skip the reference value
            if val == reference_value:
                continue

            # create a boolean
            data[f'{cat}_is_{val}'] = NP.array(data[cat].values == val, dtype = float)

    ## logarithmize body proportions
    for param in [  'bodymass' \
                  , 'leg_m' \
                  , 'bmi' \
                  ]:
        data[f'l{param}'] = LogTraFo(data[param].values)


    ## centered body proportions
    for param in [  'bodymass' \
                  , 'leg_m' \
                  , 'bmi' \
                  , 'lbodymass' \
                  , 'lleg_m' \
                  , 'lbmi'
                  ]:
        data[f'c{param}'] = NP.nan # initialize empty

        # ... should be groupwise
        for agegrp in NP.unique(data['ageclass'].values):
            selection = data['ageclass'].values == agegrp

            # center
            data.loc[selection, f'c{param}'] = Center(data.loc[selection, param].values)


    return data

################################################################################
### Helpers                                                                  ###
################################################################################

## Ordered unique elements of a nested list
# https://stackoverflow.com/a/716489
# https://twitter.com/raymondh/status/944125570534621185
CombineNestedList = lambda original: list(dict.fromkeys(sum(original, [])))

## list of cycles
def ListCycles(data: PD.DataFrame) -> list:
    # extract all subject/stride combinations from the data frame
    # (i.e. list of recordings)

    return data.index.droplevel('time').unique()


## add a zero column to allow 3D magic
Fake3D = lambda points: NP.c_[ points, NP.zeros((points.shape[0], 1)) ]

## joint angle profile rectification
def CyclicWrapInterpolateAngle(angle: NP.array, skip_wrap: bool = False) -> NP.array:
    # three steps in one:
    #  - repeat cyclic trace
    #  - wrap angular data
    #  - interpolate NAN gaps

    # replication of the trace
    time = NP.linspace(0., 3., 3*len(angle), endpoint = False)

    signal = NP.concatenate([angle]*3)

    # exclude nans
    nonnans = NP.logical_not(NP.isnan(signal))

    # wrapping: some signals can jump the 2π border
    if NP.any(NP.abs(NP.diff(signal[nonnans])) > NP.pi):
        wrapped_signal = signal.copy()
        wrapped_signal[wrapped_signal < 0] += 2*NP.pi
    else:
        wrapped_signal = signal

    # rbf interpolation of NANs
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    intp_signal = INTP.Rbf(time[nonnans] \
                             , wrapped_signal[nonnans] \
                             , function = 'thin_plate' \
                             )(time)

    # cut middle part
    angle_intp = intp_signal[len(angle):2*len(angle)]

    return angle_intp



def SexFromName(name):
    # query gender of a name from a web database
    # https://instructobit.com/tutorial/110/Python-3-urllib%3A-making-requests-with-GET-or-POST-parameters
    import urllib.request

    with urllib.request.urlopen( f"https://api.genderize.io?name={name}" ) as response:
        response_text = response.read()

        result = eval(response_text)

    return result['gender']

################################################################################
### Data Access Convenience                                                  ###
################################################################################
def LoadDefaultConfig() -> CONF.Config:
    # load the default config file

    # give config path
    config_file = f'data{OS.sep}papio.conf'

    # load config file
    config = CONF.Config.Load(config_file)

    return config
