import numba as nb
import numpy as np

@nb.njit
def quaternion_multiply(q1, q2):
    """Hamilton product q1 * q2"""
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2

    return np.array([a, b, c, d], dtype=np.float64)

@nb.njit
def q_normalize(q):
    return q / np.sqrt((q*q).sum())

@nb.njit
def axis_angle_to_quaternion(axis, angle_deg):
    """Create quaternion from axis and angle."""
    angle_rad = angle_deg * np.pi / 180.0
    h = 0.5 * angle_rad
    s = np.sin(h)
    return q_normalize(np.array([np.cos(h), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64))