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

# these functions are copied from Kikuchipy v0.11.2 to make my scripts compatible with the new dev version and venv needed for pattern processing
@nb.njit(
    "Tuple((float64, float64, float64, float64))(float64, float64, float64)",
    cache=True,
    nogil=True,
    fastmath=True,
)
def get_cosine_sine_of_alpha_and_azimuthal_local(sample_tilt, tilt, azimuthal):
    alpha = (np.pi / 2) - np.deg2rad(sample_tilt) + np.deg2rad(tilt)
    azimuthal = np.deg2rad(azimuthal)
    return np.cos(alpha), np.sin(alpha), np.cos(azimuthal), np.sin(azimuthal)

@nb.njit(
    (
        "float64[:, :, :]"
        "(float64[:], float64[:], float64[:], int64, int64, float64, float64, float64, bool_[:])"
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def get_direction_cosines_for_varying_pc_local(
    pcx,
    pcy,
    pcz,
    nrows,
    ncols,
    tilt,
    azimuthal,
    sample_tilt,
    signal_mask,
):
    nrows_array = np.arange(nrows - 1, -1, -1)
    ncols_array = np.arange(ncols)

    ca, sa, cw, sw = get_cosine_sine_of_alpha_and_azimuthal_local(
        sample_tilt=sample_tilt,
        tilt=tilt,
        azimuthal=azimuthal,
    )

    det_x_factor = (1 - ncols) * 0.5
    det_y_factor = (1 - nrows) * 0.5

    idx_1d = np.arange(nrows * ncols)[signal_mask]
    rows = idx_1d // ncols
    cols = np.mod(idx_1d, ncols)

    n_pcs = pcx.size
    n_pixels = idx_1d.size
    r_g_array = np.zeros((n_pcs, n_pixels, 3), dtype=np.float64)

    for i in nb.prange(n_pcs):
        xpc = ncols * (0.5 - pcx[i])
        ypc = nrows * (0.5 - pcy[i])
        zpc = nrows * pcz[i]

        det_x = xpc + det_x_factor + ncols_array
        det_y = ypc - det_y_factor - nrows_array

        Ls = -sw * det_x + zpc * cw
        Lc = cw * det_x + zpc * sw

        for j in nb.prange(n_pixels):
            r_g_array[i, j, 0] = det_y[rows[j]] * ca + sa * Ls[cols[j]]
            r_g_array[i, j, 1] = Lc[cols[j]]
            r_g_array[i, j, 2] = -sa * det_y[rows[j]] + ca * Ls[cols[j]]

    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = np.true_divide(r_g_array, norm)

    return r_g_array
