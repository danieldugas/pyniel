import numpy as np
import tf.transformations as se3

def se3_from_transformstamped(trans):
    """
    trans : TransformStamped
    M : 4x4 se3 matrix
    """
    transl = np.array([
        trans.transform.translation.x,
        trans.transform.translation.y,
        trans.transform.translation.z,
    ])
    quat = np.array([
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w,
    ])
    return se3_from_transl_quat(transl, quat)

def se3_from_transl_quat(transl, quat):
    """
    transl : [x, y, z]
    quat : [x, y, z, w]
    M : 4x4 se3 matrix
    """
    return np.dot(se3.translation_matrix(transl), se3.quaternion_matrix(quat))

def rot_mat_from_basis_vectors(x, y):
    """ right hand basis Z = X x Y
    x : [x y z]
    y : [x y z]
    M : 3x3 rot matrix"""
    xnorm = x / np.linalg.norm(x)
    ynorm = y / np.linalg.norm(y)
    # what if x and y are not perfectly orthogonal?
    if np.dot(xnorm, ynorm) > 0.01:
        print("Warning: basis vectors are not orthogonal")
    znorm = np.cross(xnorm, ynorm)
    rotmat = np.stack([xnorm, ynorm, znorm]).T
    return rotmat

def se3_from_pos_rot3(pos, rot):
    """
    pos : [x, y, z]
    rot : 3x3 rot matrix
    M : 4x4 se3 matrix
    """
    rot4 = se3.identity_matrix()
    rot4[:3, :3] = rot
    M = np.dot(se3.translation_matrix(pos), rot4)
    return M
