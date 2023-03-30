#!/usr/bin/env python3


################################################################################
### Config Toolbox                                                           ###
################################################################################
__author__      = "Falk Mielke"
__date__        = 20210727

"""
This toolbox contains everything to manage a config file
in the context of the Papio project.

Questions are welcome (falkmielke.biology@mailbox.org)
"""


################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import os as OS             # operating system control and file operations
import re as RE             # regular expressions, to extract patterns from text strings
import configparser as CONF # configuration


#_______________________________________________________________________________
# optionally print progress

StatusText = lambda *txt, **kwargs: None

def SetVerbose():
    # a function to print output of this script from another control script
    global StatusText
    StatusText = lambda *txt, **kwargs: print (*txt, **kwargs)

def SetSilent():
    # a function to mute output of this script from another control script
    global StatusText
    StatusText = lambda *txt, **kwargs: None

# per default: silent operation
SetSilent()


################################################################################
### Settings                                                                 ###
################################################################################
class Config(CONF.ConfigParser):
    # configuration and organization of the kinematics analysis.

    @classmethod
    def Load(cls, load_file: str):
        # read the content of an existing config file
        config = cls()

        # load file
        StatusText(f'loading config from {load_file}.')
        config.read(load_file)

        # remember filename
        config.store_file = load_file

        return config

    @classmethod
    def GenerateDefault(cls, config_filename: str):
        # prepares a settings file using confparser.
        # content are:
        #   - folder structure
        #   - video settings
        #   - analysis parameters


        StatusText('generating default config file.')
        config = cls()

        config.store_file = config_filename

        # folder structure
        folders = [   'kinematics' \
                    , 'jointangles' \
                    , 'jointfsd' \
                   ]


        # folders to config
        config['folders'] = { \
                              fld: f'data{OS.sep}{fld}' \
                              for fld in [  'kinematics' \
                                            , 'jointangles' \
                                            , 'jointfsd' \
                                            , 'limbs' \
                                          ] \
                             }

        # video settings
        config['video'] = { \
                              'resolution_x': 1280.0 \
                            , 'resolution_y': 800.0 \
                            , 'hflip': "[ 6 , 8 , 15 , 16 , 24 , 25 , 30 , 31 , 36 , 37 , 38 , 39 ]" \
                           }

        config['framerates'] = { \
                                   'Baby2': '200/1' #'10000000/50029' \
                                 , 'Baby1': '200/1' #'10000000/50029' \
                                 , 'Chantal': '200/1' #'1000000/4999' \
                                 , 'Derive': '200/1' #'10000000/50043' \
                                 , 'Desastre1': '200/1' #'10000000/386847' \
                                 , 'Desastre2': '200/1' #'10000000/50033' \
                                 , 'Desastre3': '200/1' #'10000000/50033' \
                                 , 'Elfira': '200/1' #'10000000/50059' \
                                 , 'Emeraude': '200/1' #'10000000/50017' \
                                 , 'Epine': '200/1' #'250000/1251' \
                                 , 'Espiegle2': '200/1' #'10000000/100269' \
                                 , 'Espiegle1': '200/1' #'10000000/49991' \
                                 , 'Falbala2': '200/1' #'2500000/12547' \
                                 , 'Falbala1': '200/1' #'10000000/50051' \
                                 , 'Filosophie': '200/1' #'10000000/100269' \
                                 , 'Fleur4': '200/1' #'2500000/12519' \
                                 , 'Fleur3': '200/1' #'400000/2003' \
                                 , 'Fleur1': '200/1' #'78125/614' \
                                 , 'Ford': '200/1' #'10000000/77491' \
                                 , 'Goloum': '200/1' #'5000000/25007' \
                                 , 'Grant': '200/1' #'1000000/5013' \
                                 , 'Hilna': '200/1' #'10000000/131163' \
                                 , 'Urbaine': '200/1' #'10000000/77491' \
                                 , 'Ursuline': '200/1' #'200000/1589' \
                                 , 'Vanessa': '200/1' #'1250000/18741' \
                                 , 'zChris': '200/1' \
                                 , 'zBabar': '200/1' \
                                 , 'zVictoire': '200/1' \
                                 , 'zVolga': '200/1' \
                                 , 'zVinci': '200/1' \
                                 , 'zVoltarelle': '200/1' \
                                 , 'zTassadite': '200/1' \
                                }


        # files
        config['datafiles'] = { \
                                 'raw_folder': f'data{OS.sep}xl' \
                               , 'stride_infos': f'data{OS.sep}Bipedal_strides_TD_ForFalkV2.xlsx' \
                               , 'subject_rawinfos': f'data{OS.sep}BaboonAge_ForFalk.csv' \
                               , 'subject_info': f'data{OS.sep}subject_info.txt' \
                               , 'kinematics_file': f'data{OS.sep}all_strides.csv' \
                               , 'stridetimes_file': f'data{OS.sep}stride_timing.csv' \
                               , 'jointangles_file': f'data{OS.sep}stride_cyclediffs.csv' \
                               , 'cyclediffs_file': f'data{OS.sep}stride_cyclediffs.csv' \
                               , 'cyclemissing_file': f'data{OS.sep}stride_missing.csv' \
                               , 'coordination_data': f'data{OS.sep}coordination_data.csv'  \
                               , 'coordination_pca': f'data{OS.sep}coordination.pca'  \
                               , 'metadata': f'data{OS.sep}analysis_metadata.csv'  \
                               , 'pose': f'data{OS.sep}analysis_pose.csv' \
                               , 'coordination': f'data{OS.sep}analysis_coordination.csv'  \
                               , 'analysis_data': f'data{OS.sep}analysis_all.csv' \
                              }

        # analysis parameters
        config['analysis'] = { \
                                 'fourier_order': 8 \
                               , 'flip_leftwards': False \
                               , 'reference_joint': 'ilimb' \
                               , 'superimposition_choice': dict(skip_shift = True \
                                              , skip_scale = True \
                                              , skip_rotation = False \
                                              ) \
                               , 'pca_joints': ['ihip', 'iknee', 'iankle'] \
                               , 'toe_landmark': 9 \
                               , 'toemove_threshold': 1.0 \
                               , 'trunk_landmarks': [3,1] \
                               , 'thigh_landmarks': [5,6] \
                               , 'shank_landmarks': [6,7] \
                               , 'clearance_landmarks': [5,9] \
                               , 'speedref_landmarks': [5,14,1,2,3] \
                              }

        # landmarks
        config['landmarks'] = { \
                                0: 'fake landmark for segment angles' \
                              , 99: 'fake landmark for segment angles' \
                              , 1: 'Back of the head' \
                              , 2: 'Tip of the nose' \
                              , 3: 'Base of the tail' \
                              , 4: 'Tip of the tail' \
                              , 5: 'great trochanter' \
                              , 6: 'ipsilateral knee (patella)' \
                              , 7: 'ipsilateral malleolus' \
                              , 8: 'ipsilateral heel' \
                              , 9: 'Tip of the ipsilateral foot' \
                              , 10: 'contralateral knee (patella)' \
                              , 11: 'contralateral malleolus' \
                              , 12: 'contralateral heel' \
                              , 13: 'Tip of the contralateral foot' \
                              , 14: 'Acromion' \
                              , 15: 'ipsilateral elbow' \
                              , 16: 'ipsilateral wrist' \
                              , 17: 'ipsilateral tip of finger' \
                              , 18: 'contralateral wrist' \
                              , 19: 'contralateral tip of finger' \
                              }

        # joints
        config['joints'] = { \
                              0: ([14,5,1,2], 'head') \
                            , 1: ([14,5,3,4], 'tail') \
                            , 2: ([14,5,5,6], 'ihip') \
                            , 3: ([5,6,6,7], 'iknee') \
                            , 4: ([6,7,8,9], 'iankle') \
                            , 5: ([14,5,5,10], 'chip') \
                            , 6: ([5,10,10,11], 'cknee') \
                            , 7: ([10,11,12,13], 'cankle') \
                            , 8: ([5,14,14,15], 'ishoulder') \
                            , 9: ([14,15,15,16], 'iellbow') \
                            , 10: ([15,16,16,17], 'iwrist') \
                            , 11: ([16,17,18,19], 'cwrist') \
                            , 12: ([5,1,5,9], 'ilimb') \
                            , 13: ([5,1,5,13], 'climb') \
                            , 14: ([99,0,5,14], 'trunk') \
                            }

        # subset of joints that enter the analysis
        config['selection'] = {nr: jnt for nr, jnt in enumerate([ \
                                  'ihip' \
                                , 'iknee' \
                                , 'iankle' \
                                , 'ilimb' \
                                , 'ishoulder' \
                                , 'climb' \
                                , 'trunk' \
                               ]) }

        # excluded stride IDs
        config['exclude'] = { \
                                36: 'exclude Hilna (mail by François / 20210801)' \
                              , 37: 'exclude Hilna (mail by François / 20210801)' \
                              , 38: 'exclude Hilna (mail by François / 20210801)' \
                              , 39: 'exclude Hilna (mail by François / 20210801)' \
                              # , 11: 'too much missing data' -> restored \
                              # , 12: 'too much missing data' -> restored \
                              # , 20: 'too much missing data' -> restored \
                              , 44: 'Berillon data excluded (inconsistent; missing landmarks; framerate unclear; sampling issues/outliers)'
                              , 45: 'unplausibly high speed (Chris)' \
                              , 46: 'Berillon data excluded (inconsistent; missing landmarks; framerate unclear; sampling issues/outliers)'
                              , 47: 'Victoire is only a limb (no upper body ref)' \
                              , 48: 'completely off data (Vinci)' \
                              , 49: 'Berillon data excluded (inconsistent; missing landmarks; framerate unclear; sampling issues/outliers)'
                              , 50: 'Berillon data excluded (inconsistent; missing landmarks; framerate unclear; sampling issues/outliers)'
                             }

        # store config
        config.Store()

        return config


    def Store(self, store_file: str = None, overwrite: bool = False):
        # store config, optionally to a different file.

        if store_file is None:
            store_file = self.store_file

        # check if file exists
        if OS.path.exists(store_file) and not overwrite:

            while ( res:=input(f"Overwrite {store_file}? (y/n)").lower() ) not in {"y", "n"}: pass
            if not res == 'y':
                StatusText('not saved!')
                return

        # write the file
        with open(store_file, 'w') as configfile:
            self.write(configfile)
            StatusText(f'\tstored to {store_file}.')


    def CreateFolders(self):
        # create empty folders
        for fld in self['folders'].values():

            if not OS.path.exists(fld):
                StatusText(f'\tcreating folder: {fld}')
                OS.mkdir(fld)


################################################################################
### Running Example                                                          ###
################################################################################
def ExampleConfig():
    # display output
    SetVerbose()

    # give a path
    config_file = OS.sep.join(['..', 'data','papio.conf'])

    # generate a project file
    config = Config.GenerateDefault(config_file)

    # load a project file
    config = Config.Load(config_file)

    print(config)
