#!/usr/bin/env python3

################################################################################
### Quaternion and Geometry Toolbox                                          ###
################################################################################
__author__      = "Falk Mielke"
__date__        = 20210115

"""
A few helper functions for the work with quaternions and anatomical limb segments.

further reading: http://mielke-bio.info/falk/quaternions
Questions please: falkmielke.biology@mailbox.org

Falk Mielke
2021/01/15
"""



#######################################################################
### Libraries                                                       ###
#######################################################################
# system
import warnings as WRN

# numbers
import numpy as NP
with WRN.catch_warnings():
    WRN.simplefilter("ignore")
    import quaternion as Q

from scipy.spatial.transform import Rotation as ROT
import scipy.spatial.distance as DIST

# data
import pandas as PD

# plots and graphics
import matplotlib as MP
MP.use("TkAgg")
import matplotlib.pyplot as MPP

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

from mpl_toolkits.mplot3d import art3d


# import FilterToolbox as FilT
import SuperimpositionToolbox as ST




#######################################################################
### Quaternions                                                     ###
#######################################################################
QuatFromArray = lambda arr: Q.as_quat_array(arr)# NP.array([NP.quaternion(*q) for q in arr])
ArrayFromQuat = lambda quat: Q.as_float_array(quat) # NP.stack([NP.append(q.w, q.vec) for q in quat])

### numeric quaternions
# rotate a Q by a Q
QRotate = lambda q, rot: rot*q*rot.conj()


# turn 3D position vectors (positions in rows) into quaternions
# https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array/8505658#8505658
PrependZeroColumn = lambda nx3_array: NP.c_[ NP.zeros((nx3_array.shape[0], 1)), nx3_array ]
QuaternionFrom3D = lambda nx3_array: QuatFromArray(PrependZeroColumn(nx3_array))

# AsQuatArray = QuatFromArray


# rotate vectors by a Q
RotateVectors = lambda vectors, rot_q: ArrayFromQuat(QRotate(QuaternionFrom3D(vectors), rot_q))[:, 1:]  

# find ONE Q_? would rotate a Q to a reference Q_r
FindQuaternionRotation = lambda reference, q_to_rotate: \
                NP.quaternion( (NP.linalg.norm(reference.vec) * NP.linalg.norm(q_to_rotate.vec)) + NP.dot(reference.vec, q_to_rotate.vec) \
                            , *NP.cross(reference.vec, q_to_rotate.vec) \
                            ).normalized()
# NOTE: in 3D cases, better use the Kabsch algorithm



# check for consistency with https://github.com/sympy/sympy/blob/master/sympy/algebras/quaternion.py
CombineAngleAxis = lambda vec, theta: (NP.array([v/NP.sin(theta/2) for v in vec]), theta)
ToAxisAngle = lambda q: CombineAngleAxis(q.vec, 2*NP.arccos(q.w))

SingleQuatToArray = lambda q: NP.append(q.w, q.vec)

# q = NP.quaternion(0.8,0.,0.,0.)
# print (q.w)
# print(dir(q))



# angle between two vectors, counter-clockwise from second to first vector
AngleBetweenCCW = lambda v1, v2: (NP.arctan2(*v1[::-1]) - NP.arctan2(*v2[::-1])) % (2 * NP.pi)

# wrap to [-π/π]
# https://stackoverflow.com/a/29237626
WrapAnglePiPi = lambda ang: NP.arctan2(NP.sin(ang), NP.cos(ang))




def UnitVector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / NP.linalg.norm(vector)

## angle between two vectors, classical method
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def AngleBetween(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> AngleBetween((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> AngleBetween((1, 0, 0), (1, 0, 0))
            0.0
            >>> AngleBetween((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    NOTE: always smallest angle;

    """
    v1_u = UnitVector(v1)
    v2_u = UnitVector(v2)
    return NP.arccos(NP.clip(NP.dot(v1_u, v2_u), -1.0, 1.0))




# Reference Rotation
def RotateToReference(refseg, seg1):
    # rotate a quaternion to stand on the z axis
    # and also rotate a second quaternion attached to it
   
    q0 = NP.quaternion(0., *refseg)
    q1 = NP.quaternion(0., *seg1)
    # print (q0, q1)
    rotated_refseg = NP.quaternion(0.,0., 0., NP.linalg.norm(refseg))
    rot0 = FindQuaternionRotation(reference = q0, q_to_rotate = rotated_refseg)


    return QRotate(q1, rot0).vec




def GetRotationQuaternion(points, refpoints = NP.concatenate([NP.zeros((1,3)), NP.eye(3)], axis = 0)):
    # get the rotation quaternion for a segment
    # by finding optimal rotation between points. 
    # IMPORTANT: The output is a numpy array, and NOT a numpy quaternion.
    #            To apply rotation later, convert it to quaternion with NP.quaternion(*quat)

    rotmat = ST.KabschRotation(points, refpoints)
    quat = NP.roll(ROT.from_matrix(rotmat).as_quat().ravel(), 1) # weird component order in ROT
    quat[0] *= -1 # store the conjugate to get direction right

    return quat


def FlipToMinimalAngleQuaternion(quat):
    # works on a numpy float array, not on a quaternion

    angle = 2*NP.arccos(quat[:, 0].ravel())
    inverse_angle = 2*NP.arccos(-quat[:, 0].ravel())

    if NP.sum(NP.abs(inverse_angle)) < NP.sum(NP.abs(angle)):
        return quat * -1
    else:
        return quat
        



#######################################################################
### Quaternion Time Derivatives                                     ###
#######################################################################
# ### Time Derivatives
# def AngularVelocityDiff(Rd, t):
#     Rd = ArrayFromQuat(Rd) # convert to float array
#     Rdot = NP.divide(NP.diff(Rd, axis = 0), NP.stack([NP.diff(t)] * Rd.shape[1], axis = 1)) # initialize Rdot (i.e. the time derivative dq)


#     # make quaternion
#     Rd = QuatFromArray(Rd[:-1, :])
#     Rdot = QuatFromArray(Rdot)

#     # apply the ang vel formula consistent with https://stackoverflow.com/questions/56790186/how-to-compute-angular-velocity-using-numpy-quaternion
#     return ArrayFromQuat(2*Rdot/Rd)[:, 1:]


### Time Derivatives
def QuaternionDiff(quat, times, lowpass_filter = None):
    dt = NP.mean(NP.diff(times))
    # quat = PrependZeroColumn(data_array.copy()) # convert to nx4 float array 

    
    if lowpass_filter is None:
        quat = quat.copy()
    else: 
        raise(Exception("Filtering not included in this toolbox."))
        # quat = NP.stack([FilT.Smooth(times, quat[:,q].ravel(), lowpass_filter) for q in range(quat.shape[1])], axis = 1)

    from scipy.interpolate import CubicSpline


    ## assure continuity of quaternions in cases where there are sign flips (q ~ -q)
    # print (NP.sum(NP.abs(NP.diff(NP.sign(quat), axis = 0)) > 0, axis = 1)[390:400] )
    for i in range(quat.shape[0]):
        q = quat[i, :]
        if NP.sum(NP.abs(NP.sign(q) - NP.sign(quat[i-1, :])) > 0) >= 3:
            quat[i:, :] *= -1

    ## initialize the time derivative of quaternion array
    dquat_dt = NP.gradient(quat, dt, axis = 0, edge_order = 2)
    # dquat_dt_qs = CubicSpline(times, quat).derivative()(times)


    # initialize the second time derivative of quaternion array
    d2quat_dt2 = NP.gradient(dquat_dt, dt, axis = 0, edge_order = 2)
    # d2quat_dt2_qs = CubicSpline(times, quat).derivative(2)(times)


    ## ==> the cubic spline variant looks a bit noisier with Dumas' example.



    # convert all numeric derivatives to quaternion arrays
    quat = QuatFromArray(quat)
    dquat_dt = QuatFromArray(dquat_dt)
    d2quat_dt2 = QuatFromArray(d2quat_dt2)


    ## first derivative:
    # three options:
    # (1) the ang vel formula consistent with https://stackoverflow.com/questions/56790186/how-to-compute-angular-velocity-using-numpy-quaternion
    # also consistent with eqn (3) in http://www.euclideanspace.com/physics/kinematics/angularvelocity/QuaternionDifferentiation2.pdf
    # diff1 = 2* (dquat_dt / quat)

    # # (2) the quaternion differential as in Dumas paper and toolbox
    diff1 = 2 * NP.multiply(dquat_dt, quat.conj())
    # print(diff1[50:55, :])

    # # (3) the numpy ang vel formula.
    # diff1 = NP.quaternion_time_series.angular_velocity(quat, t = times)
    # print(diff1[50:55, :])

    
    ## second derivative:
    # https://physics.stackexchange.com/questions/460311/derivation-for-angular-acceleration-from-quaternion-profile

    # consistent with the formula referenced by Dumas et al.
    diff2 = 2*( NP.multiply(d2quat_dt2, quat.conj()) + NP.multiply(dquat_dt, dquat_dt.conj()) )
    # vector values of the second summand are zero

    # cf. 
    # https://jamey.thesharps.us/2016/05/16/angular-velocity-and-quaternion
    # http://www.euclideanspace.com/physics/kinematics/angularvelocity/QuaternionDifferentiation2.pdf

    return ArrayFromQuat(diff1)[:, 1:],  ArrayFromQuat(diff2)[:, 1:]





#######################################################################
### Line Intersection                                               ###
#######################################################################
# https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
def LineIntersect(lines):
    """
    calculate intersection point of lines (arbitrary number of spatial dimensions)
        input parameter: 
            lines ... list of lines, defined by start and end point tuples
                      e.g. [ [(0.,0.,0.), (0.,0.,1.)], [(1.,0.,0.), (0.,1.,0.)] ]

    start_points and end_points are NxD arrays defining N lines.
    D is the dimension of the space. This function returns the least squares intersection 
    of the N lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf

    see also
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_more_than_two_dimensions
    """
    # generate all line direction vectors 

    stacked_lines = NP.stack(lines, axis = -1)
    start_points = stacked_lines[0,:,:].T
    end_points = stacked_lines[1,:,:].T

    vectors_normed = (end_points-start_points)/NP.linalg.norm(end_points-start_points,axis=1)[:,NP.newaxis] # normalized

    # generate the array of all projectors 
    projectors = NP.eye(vectors_normed.shape[1]) - vectors_normed[:,:,NP.newaxis]*vectors_normed[:,NP.newaxis] 

    # generate R matrix and q vector
    R = projectors.sum(axis=0)
    q = (projectors @ start_points[:,:,NP.newaxis]).sum(axis=0) # @ = matrix multiplication

    # solve the least squares problem for the intersection point p, 
    # using Rp = q
    try:
        # exact solver
        intersect_point = NP.linalg.solve(R,q)

    except NP.linalg.LinAlgError as la_err:
        # in case of parallel lines, the matrix gets singular, there would be infinitely many solutions.
        WRN.warn(str(la_err))
        return NP.mean(start_points, axis = 0)#NP.linalg.lstsq(R,q,rcond=None)[0].ravel()


    return intersect_point.ravel()








def TestLineIntersect():

    # 3D, trivial
    line1 = [(0.,0.,0.), (0.,0.,1.)]
    line2 = [(1.,0.,0.), (0.,1.,0.)]
    print (LineIntersect([line1, line2]))

    # 3D, parallel lines
    line1 = [(0.,0.,0.), (0.,0.,1.)]
    line2 = [(1.,1.,1.), (1.,1.,2.)]
    print (LineIntersect([line1, line2]))

    # 3D, many lines
    line1 = [(0.,0.,0.), (0.,0.,1.)]
    line2 = [(1.,0.,0.), (0.,1.,0.)]
    line3 = [(3.,3.,0.), (3.,2.,1.)]
    line4 = [(-1.,-3.,0.), (0.,-4.,-1.)]
    print (LineIntersect([line1, line2, line3, line4]))


    # 2D
    line1 = [(0.,0.), (0.,1.)]
    line2 = [(1.,0.), (0.5,1.)]
    print (LineIntersect([line1, line2]))

    # 4D
    line1 = [(0.,0.,0., 0), (0.,0.,1., 1)]
    line2 = [(1.,0.,0., 0), (0.,1.,0., 1)]
    print (LineIntersect([line1, line2]))




#######################################################################
### Joint Detection                                                 ###
#######################################################################
def FindJoint(lines):
    # find the joint between two segments if only two points along the segment are known.

    v1 = NP.diff(lines[0], axis = 0).ravel()
    v2 = NP.diff(lines[1], axis = 0).ravel()

    angle = AngleBetween(v1, v2)

    intersect = LineIntersect(lines)

    distmat = DIST.cdist(lines[0], lines[1])
    mindist_points = NP.unravel_index(NP.argmin(distmat), distmat.shape)
    p1 = lines[0][mindist_points[0], :]
    p2 = lines[1][mindist_points[1], :]

    meanpoint = (p1 + p2) / 2.


    # using the Pythagorean Trigonometric Identity to weigh the intersect and the meanpoint
    joint = intersect * NP.sin(angle)**2 + meanpoint * NP.cos(angle)**2

    # print (v1, v2, angle, intersect, meanpoint, joint)

    return joint



def TestFindJoint():

    p1 = NP.array([0.5, 0., 0.])
    p2 = NP.array([1.5, 0., 0.])


    # fig = MPP.figure()
    # ax = fig.add_subplot(1,1,1,aspect = 'equal')
    for ang in NP.arange(0, 360, 5):
        rotation = ROT.from_euler('z', ang, degrees=True)
        vec = rotation.apply(NP.array([1., 0., 0.]))
        # print (ang, vec)

        p3 = 0.5*vec
        p4 = 1.5*vec

        l1 = NP.stack([p1, p2], axis = 0)
        l2 = NP.stack([p3, p4], axis = 0)

        joint = FindJoint([l1, l2])
        print (ang, joint)

        fig = MPP.figure()
        ax = fig.add_subplot(1,1,1,aspect = 'equal')

        ax.plot(l1[:, 0], l1[:, 1], 'k-')
        ax.plot(l2[:, 0], l2[:, 1], 'k-')
        ax.scatter(joint[0], joint[1])

        ax.set_xlim([-1.6,1.6])
        ax.set_ylim([-1.6,1.6])
        MPP.show()

#######################################################################
### Visualization                                                   ###
#######################################################################
# MPP.style.use('dark_background')
the_font = {  \
        # It's really sans-serif, but using it doesn't override \sffamily, so we tell Matplotlib
        # to use the "serif" font because then Matplotlib won't ask for any special families.
         # 'family': 'serif' \
        # , 'serif': 'Iwona' \
        'family': 'sans-serif'
        , 'sans-serif': 'DejaVu Sans'
        , 'size': 10#*1.27 \
    }
MP.rcParams['text.usetex'] = False

MPP.rc('font',**the_font)
# Tell Matplotlib how to ask TeX for this font.
# MP.texmanager.TexManager.font_info['iwona'] = ('iwona', r'\usepackage[light,math]{iwona}')

MP.rcParams['text.latex.preamble'] = ",".join([ \
              r'\usepackage{upgreek}' \
            , r'\usepackage{cmbright}' \
            , r'\usepackage{sansmath}' \
            ])

MP.rcParams['pdf.fonttype'] = 42 # will make output TrueType (whatever that means)



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


# https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
def AxisEqual3D(ax):
    extents = NP.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = NP.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


class Arrow3D(FancyArrowPatch):
    # http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)






def TestQuaternionMultiply():
    q1 = NP.quaternion(0.2, 0.3, 0.4, 0.5)
    q2 = NP.quaternion(0.1, 0.7, 0.1, 0.6)

    print (q1*q2)

    def QuatMult(q1, q2): 
        qs1 = q1.w
        qs2 = q2.w 

        qv1 = q1.vec
        qv2 = q2.vec

        qs = qs1*qs2 - NP.dot(qv1.T, qv2)
        qv = qs1*qv2 + qs2*qv1 + NP.cross(qv1, qv2)

        print (qs, qv)

        return NP.quaternion(qs, qv[0], qv[1], qv[2])

    print (QuatMult(q1, q2))



#######################################################################
### Testing                                                         ###
#######################################################################

if __name__ == "__main__":
    # TestLineIntersect()
    # TestQuaternionMultiply()
    # TestFindJoint()

    print (dir(Q.quaternion(0.5, 0.5, 0.5, 0.5)))






