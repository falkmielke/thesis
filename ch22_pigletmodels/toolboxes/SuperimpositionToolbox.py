#!/usr/bin/env python3

################################################################################
### Superimposition Toolbox                                                  ###
################################################################################
"""
It is hard to say whether there are procedures more central to my work than Procrustes Superimposition (maybe Fourier Series or Principal Component Analysis). 
I've had [two](https://doi.org/10.1007/s11692-018-9456-9?target=_blank) [publications](https://doi.org/10.1093/zoolinnean/zlz135?target=_blank) which 
introduce slight tweaks of the general procedure, and I'm currently working on a more subtle one that will use plain vanilla Procrustes as part of my own inverse dynamics workflow.

It all started, as so often, with code copied from a [stack overflow response](https://stackoverflow.com/a/18927641) and which I've adapted for my purpose. 
Initially, I understood close to nothing. 
But over time, I (wrongly) adjusted the code, found mistakes, and understood more.
Only today, by coincidence, I fully understood the last enigmatic component (the "reflection" issue with SVD). 

This anecdote serves as a motivation to students to go out using (and thereby gradually understanding) the code they find online. 
I'll also take the occasion to briefly explain my Procrustes toolbox.


All data are points, organized in an n×m matrix, with n being the number of points, and m the dimension (always 3 with the code below).


Enjoy!

"""

__author__ = "Falk Mielke"
__date__ = 20200907


# the only library required herein is numpy.
import numpy as NP




################################################################################
### Data Properties                                                          ###
################################################################################
# The following functions extract features of the data which are relevant.

def Centroid(pointset):
    # the centroid of the data.
    return NP.mean(pointset, axis = 0)


def RCS(pointset):
    # root centroid size, i.e. the scale of the data, defined as the root sum of 
    # euclidean distances of all points from their centroid.
    return NP.sqrt(NP.sum(NP.power(pointset-Centroid(pointset), 2) ))



################################################################################
### Transformation Operations                                                ###
################################################################################
def Center(pointset, trafo_history = None):
    # move all points so that the centroid is at the origin.
    centroid = Centroid(pointset)

    if trafo_history is not None:
        MakeHistory(trafo_history, translate = -centroid, log = 'center')

    return pointset - centroid


def UnitScale(pointset, trafo_history = None):
    # Scale the points so that RCS is one.
    rcs = RCS(pointset)

    if trafo_history is not None:        
        MakeHistory(trafo_history, scale = 1/rcs, log = 'unitscale')

    return pointset / rcs


def RotateCentered(pointset, rotation_matrix, trafo_history = None):
    # NOT USED
    # rotate a set of points around its centroid
    centroid = Centroid(pointset)

    if trafo_history is not None:
        MakeHistory(trafo_history, rotate = rotation_matrix, log = 'rotate')

    return ((pointset.copy() - centroid) @ rotation_matrix) + centroid


def Rotate(pointset, rotation_matrix, trafo_history = None):
    # rotate a set of points
    # Note: usually (but not generally) it makes sense to center the points.
    centroid = Centroid(pointset)

    if trafo_history is not None:
        MakeHistory(trafo_history, rotate = rotation_matrix, log = 'rotate')

    return pointset.copy() @ rotation_matrix


def Standardize(pointset, trafo_history = None):
    # Standardize a point set, i.e. center and scale to unity.
    pointset = Center(pointset.copy(), trafo_history)
    pointset = UnitScale(pointset, trafo_history)

    return pointset




################################################################################
### Optimal Rotation                                                         ###
################################################################################
def KabschRotation(focal_points, ref_points):
    # optimum rotation matrix to rotate focal points to reference points
    # points must be Nx3 numpy arrays.
    # following https://en.wikipedia.org/wiki/Kabsch_algorithm
    # cf. 
    #   http://nghiaho.com/?page_id=671
    #   https://github.com/charnley/rmsd
    #   http://www.kwon3d.com/theory/jkinem/rotmat.html

    if not (focal_points.shape == ref_points.shape):
        raise IOError('point sets must have the same shape. {} / {}'.format(focal_points.shape, ref_points.shape))

    # deep copy
    focal_points = focal_points.copy()
    ref_points = ref_points.copy()


    # standardize 
    focal_points = Standardize(focal_points)
    ref_points = Standardize(ref_points)


    # calculate cross-dispersion (correlation) matrix
    cross_dispersion = focal_points.T @ ref_points
    
    # singular value decomposition of cross-dispersion matrix
    U,_,V_T = NP.linalg.svd(cross_dispersion, full_matrices = False)
    V = V_T.T

    # check reflection case
    D = NP.eye(3)*NP.array([1., 1., NP.linalg.det(V @ U.T)])

    # return the rotation matrix 
    return V @ D @ U.T

################################################################################
### NAN helper                                                               ###
################################################################################
def CommonNonNaNs(forms: list) -> NP.array:
    # returns the index of points which are not nan for all forms in the list

    NaNs = lambda form: NP.any(NP.isnan(form), axis = 1)

    nan_indices = NaNs(forms[0])
    for frm in forms[1:]:
        nan_indices = NP.logical_or(nan_indices, NaNs(frm))

    return NP.logical_not(nan_indices)



################################################################################
### Procrustes Superimpdotosition                                               ###
################################################################################
def Procrustes(focalform_raw: NP.array, referenceform_raw: NP.array, trafo_history: dict = None) \
                -> (NP.array, float, dict):
    # http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    # input forms must be numpy arrays of the shape [points, dimensions]

    ## work on copies, to be safe
    referenceform = referenceform_raw.copy()
    focalform = focalform_raw.copy()

    ## match nr of points and nr of dimensions
    if (not referenceform.shape[0] == focalform.shape[0]) or (not referenceform.shape[1] == focalform.shape[1]):
        raise IOError('forms must be of the same "shape" (Procrustes/numpy joke :)')

    ## store transformations
    if trafo_history is None:
        trafo_history = GetEmptyHistory()


    ## standardize
    referenceform = Standardize(referenceform)
    focalform = Standardize(focalform, trafo_history)


    ## rotation
    # optimum rotation matrix of focal form
    # cf. http://nghiaho.com/?page_id=671
    # https://en.wikipedia.org/wiki/Kabsch_algorithm

    # calculate cross-dispersion (correlation) matrix (cf. http://www.kwon3d.com/theory/jkinem/rotmat.html)
    cross_dispersion = referenceform.T @ focalform
    
    # singular value decomposition of cross-dispersion matrix
    U, singular_values, V = NP.linalg.svd(cross_dispersion, full_matrices = False)

    # ... to get the rotation matrix R as follows:
    rotation_matrix = V.T @ U.T

    # if R has negative determinant, it is a reflection matrix and must be modified.
    if NP.linalg.det(rotation_matrix) < 0:
        rotation_matrix = ( V.T @ (NP.eye(3) * NP.array([1., 1., -1.])) ) @ U.T
        singular_values[-1] *= -1


    ## further Procrustes measures
    singular_value_trace = singular_values.sum()

    # standarised distance 
    procrustes_distance = 1 - singular_value_trace**2


    ## scaling
    # optimum scaling of data
    scaling_factor = singular_value_trace * RCS(referenceform_raw) / RCS(focalform)


    ## transformation
    # ref_rcs = RCS(referenceform_raw)
    ref_centroid = Centroid(referenceform_raw)
    transformed_data =  Rotate(focalform, rotation_matrix) \
                        * scaling_factor \
                      + ref_centroid


    ## log
    MakeHistory(trafo_history \
                , rotate = rotation_matrix \
                , scale = scaling_factor \
                , translate = ref_centroid \
                , log = 'procrustes' \
                )


    # return transformed data, residual distance, and log
    return transformed_data, procrustes_distance, trafo_history



################################################################################
### Making History                                                           ###
################################################################################
# While modifying objects, we'd like to keep track of the things that happen.
# Obviously, this facilitates bug fixing.
# But what's more important, it allows to exapt a series of transformations to
#   different point sets.

# all starts with an empty history.
def GetEmptyHistory():
    # as history, I use dicts with defined keywords.
    return {key: [] for key in ['rotate', 'scale', 'translate', 'log']}

def MakeHistory(hist, scale = 1., translate = NP.zeros((3,)), rotate = NP.eye(3), log = ''):
    # This convenience function is used to appended actions to a history dict.  
    hist['scale'].append(scale)
    hist['translate'].append(translate)
    hist['rotate'].append(rotate)
    hist['log'].append(log)



def InvertHistory(trafo_history):
    # this inverts the course of actions to exactly undo a series of transformations.
    # (not fully tested)
    inverted_history = GetEmptyHistory()
    for shift, scale, rotation_matrix, log in zip(trafo_history['translate'], trafo_history['scale'], trafo_history['rotate'], trafo_history['log']):
        MakeHistory(inverted_history, scale = 1/scale, translate = -shift, rotate = rotation_matrix.T, log = f'{log} (inverted)')

    return inverted_history



def ApplyTransformations(points, trafo_history):
    # This will apply a previously stored series of transformations to data.
    points = points.copy()
    for shift, scale, rotation_matrix in zip(trafo_history['translate'], trafo_history['scale'], trafo_history['rotate']):
        points = scale * Rotate(points, rotation_matrix) + shift

    return points


def ApplyRotations(points, trafo_history):
    # This will apply a previously stored series of ROTATIONS to data, ignoring the other aspects.
    points = points.copy()
    for rotation_matrix in trafo_history['rotate']:
        points = Rotate(points, rotation_matrix)

    return points





################################################################################
# eof.
