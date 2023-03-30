#!/usr/bin/env python3

################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________


import sys as SYS           # system control


SYS.path.append('toolboxes') # makes the folder where the toolbox files are located accessible to python
import EigenToolbox as ET    # coordination PCA


### coordination PCA
def LoadCoordinationPCA():
    filename = 'data/coordination.pca'
    cpca = ET.PrincipalComponentAnalysis.Load(filename)

    return cpca
