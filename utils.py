# Utility classes and functions for Coursera SDC Course 2.
#
# Authors: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies

import numpy as np
from numpy import sin, cos, arctan2, sqrt

class StampedData():
    def __init__(self):
        self.data = []
        self.t = []

    def convert_lists_to_numpy(self):
        self.data = np.array(self.data)
        self.t = np.array(self.t)

def to_rot(r):
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    
    Input:
        r : array-like [roll, pitch, yaw] in radians
        
    Output:
        R : 3x3 rotation matrix
             Applies rotations in order: X (roll), Y (pitch), Z (yaw)
    
    Application:
        - Use to rotate vectors or coordinate frames
        - Used in building transformation matrices (like to_mat)
    """
     
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx

def to_mat(p, r):
    "Given a position [m] and orientation as RPY [rad], create homogenous transformation matrix."
    """
    Create a 4x4 homogeneous transformation matrix from position and Euler angles.
    
    Input:
        p : array-like [x, y, z] position in meters
        r : array-like [roll, pitch, yaw] in radians
        
    Output:
        T : 4x4 homogeneous transformation matrix
            Combines rotation (from to_rot) and translation (position)
    
    Application:
        - Transform points from the object's local frame to world frame
        - Used in robotics, autonomous vehicles, or 3D motion tasks
    """
    # 
    R = to_rot(r)
    return np.mat(np.r_[np.c_[R, p], [np.array([0, 0, 0, 1])]])

def from_mat(T):
    "Get position [m] and orientation as RPY [rad] from homogenous transformation matrix."
    """
    Extract position and Euler angles from a 4x4 homogeneous transformation matrix.
    
    Input:
        T : 4x4 homogeneous transformation matrix
        
    Output:
        p : [x, y, z] position in meters
        r : [roll, pitch, yaw] in radians
        
    Application:
        - Retrieve the object's position and orientation from a transformation matrix
        - Useful in robotics, autonomous vehicles, or 3D motion analysis
    """
    p = [T[0,3], T[1,3], T[2,3]]
    r = [arctan2(T[2,1],T[2,2]) , arctan2(-T[2,0],sqrt(T[2,1] ** 2 + T[2,2] ** 2)), arctan2(T[1,0],T[0,0]) ]
    return p, r

def transform_data_right(p, r, T_frame):
    "Transform data to a different frame."
    """
    Transform a list of positions and Euler angles to a different frame (post-multiplication).
    
    Input:
        p       : Nx3 array of positions [x, y, z] in meters
        r       : Nx3 array of Euler angles [roll, pitch, yaw] in radians
        T_frame : 4x4 transformation matrix representing the new frame
    
    Output:
        p_new : Nx3 array of transformed positions
        r_new : Nx3 array of transformed Euler angles
    
    Application:
        - Converts points and orientations from a local frame to another frame
        - Post-multiplies: T_new = T_original * T_frame
        - Useful for transforming sensor or robot data to a reference frame
    """
    p_new = [0]*len(p)
    r_new = [0]*len(p)

    for i in (range(len(p))):
        T_i = to_mat(p[i, :], r[i, :])
        T_new = T_i.dot(T_frame)
        p_new[i], r_new[i] = from_mat(T_new)

    return np.array(p_new), np.array(r_new)

def transform_data_left(p, r, T_frame):
    "Transform data to different frame."
    """
    Transform a list of positions and Euler angles to a different frame (pre-multiplication).
    
    Input:
        p       : Nx3 array of positions [x, y, z] in meters
        r       : Nx3 array of Euler angles [roll, pitch, yaw] in radians
        T_frame : 4x4 transformation matrix representing the new frame
    
    Output:
        p_new : Nx3 array of transformed positions
        r_new : Nx3 array of transformed Euler angles
    
    Application:
        - Converts points and orientations from a local frame to another frame
        - Pre-multiplies: T_new = T_frame * T_original
        - Useful for transforming sensor or robot data into a global or reference frame
    """
    p_new = [0]*len(p)
    r_new = [0]*len(p)

    for i in (range(len(p))):
        T_i = to_mat(p[i, :], r[i, :])
        T_new = T_frame.dot(T_i)
        p_new[i], r_new[i] = from_mat(T_new)

    return np.array(p_new), np.array(r_new)

def to_own_frame(r, x):
    """
    Rotate vectors into their local (own) frame using corresponding Euler angles.
    
    Input:
        r : Nx3 array of Euler angles [roll, pitch, yaw] in radians
        x : Nx3 array of vectors/points in world frame
    
    Output:
        x_new : Nx3 array of vectors transformed into the local frame
    
    Application:
        - Converts world-frame vectors to the object's local frame
        - Useful for analyzing velocities, forces, or measurements relative to the object
        - Each vector x[i] is rotated using the rotation matrix from r[i]
    """
    x_new = np.zeros(x.shape)

    for i in (range(len(x))):
        x_new[i] = x[i].dot(to_rot(r[i]))

    return x_new

def to_angular_rates(r, r_dot):
    """
    Compute the inverse of the Euler kinematical matrix for the given roll,
    pitch, and yaw angles - the kinematical matrix is used to compute the
    rate of change of the Euler angles as a function of the angular velocity.
    """
    """
    Convert Euler angle rates to angular velocity in the body frame.
    
    Input:
        r     : [roll, pitch, yaw] Euler angles in radians
        r_dot : [roll_dot, pitch_dot, yaw_dot] time derivatives of Euler angles (rad/s)
    
    Output:
        omega : [wx, wy, wz] angular velocity vector in the body frame (rad/s)
    
    Application:
        - Maps the rate of change of Euler angles to the angular velocity of the object
        - Useful in robotics, drones, and vehicle dynamics
        - Uses the inverse of the Euler kinematic matrix
    """
    # Still need to differentiate the Euler angles first wrt time.
    cr = np.cos(r[0])
    sr = np.sin(r[0])
    sp = np.sin(r[1])
    cp = np.cos(r[1])

    # Generate inverse of kinematical matrix.
    # M = np.array([[1, tp*sr, tp*cr], [0, cr, -sr], [0, sr/cp, cr/cp]])
    G = np.array([[1, 0, -sp], [0, cr, sr*cp], [0, -sr, cr*cp]])

    return G @ r_dot

def integ(x, t):
    out = [None] * (len(x) + 1)
    for i in range(len(out)):
        dt = t[i + 1] - t[i]
        out[i+1,:] = out[i,:] + x[i,:]*dt

    return out

def diff(x, t):
    out = [None] * (len(x) - 1)
    for i in (range(len(out))):
        # Forward finite difference scheme.
        dt = t[i + 1] - t[i]
        dx = x[i + 1, :] - x[i, :]
        out[i] = dx / dt

    return out