import itertools
import numpy as np
import rotations


def xyt2h_partial(measurements: np.ndarray):
    """Convert a translation and rotation to a homography.

    Args:
        x (float | np.ndarray): The x-translation. Can be a scalar or an array.
        y (float | np.ndarray): The y-translation. Can be a scalar or an array.
        theta (float | np.ndarray): The rotation angle in radians. Can be a scalar or an array.

    Returns:
        np.ndarray: The homography parameters. Shape is (8,) if scalars provided. Shape is (..., 8) if arrays provided."""
    x, y, theta = measurements[..., 0], measurements[..., 1], measurements[..., 2]
    if all(isinstance(i, (int, float)) for i in (x, y, theta)):
        return np.array([np.cos(theta) - 1, -np.sin(theta), x*np.cos(theta) - y*np.sin(theta),
                         np.sin(theta), np.cos(theta) - 1, x*np.sin(theta) + y*np.cos(theta),
                         0, 0])
    else:
        _0 = np.zeros(x.shape[0])
        _c = np.cos(theta)
        _s = np.sin(theta)
        return np.array([_c - 1, -_s, x*_c-y*_s, _s, _c - 1, x*_s+y*_c, _0, _0]).T


def xyt2h(shifts: np.ndarray, PC: tuple | list | np.ndarray) -> np.ndarray:
    """Convert a translation and rotation to a homography.

    Args:
        shifts (np.ndarray): The shifts. Shape is (N, 3).  First entry is x-shift,
                             second entry is y-shift, third entry is rotation angle.
        PC (tuple | list | np.ndarray): The pattern center.
        tilt (float | int): The tilt angle of the sample in degrees.

    Returns:
        np.ndarray: The homography parameters. Shape is (8,) if scalars provided. Shape is (..., 8) if arrays provided."""
    # Check the shape of the inputs
    if shifts.ndim == 1:
        shifts = shifts[None]
    ## First convert the shifts to a global rotation matrix in the detector frame
    # Decompose inputs
    x01, x02, DD = PC
    x, y, theta = shifts[..., 0], shifts[..., 1], shifts[..., 2]
    # Get xy hats (we rotate w.r.t. the PC so we don't need to change the x and y)
    # x_hat = x + x01 * (np.cos(theta) - 1) + x02 * np.sin(theta)
    # y_hat = y - x01 * np.sin(theta) + x02 * (np.cos(theta) - 1)
    x_hat = x
    y_hat = y
    # Get omegas
    w1 = np.around(np.arctan(-y_hat / DD), 3)  # w32
    w2 = np.around(np.arctan(x_hat / DD), 3)  # W13
    w3 = np.around(theta, 3)  # w21
    # Get the global rotation matrix in the sample frame
    c1, c2, c3 = np.cos(w1), np.cos(w2), np.cos(w3)
    s1, s2, s3 = np.sin(w1), np.sin(w2), np.sin(w3)
    # Rs = np.array([[c2*c3, s1*s2*c3 - c1*s3, c1*s2*c3 + s1*s3],
    #                [c2*s3, s1*s2*s3 + c1*c3, c1*s2*s3 - s1*c3],
    #                [-s2, s1*c2, c1*c2]])
    Rs = np.array([[c2*c3, -c2*s3, s2],
                   [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
                   [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2]])
    Rs = Rs.transpose(2, 0, 1)
    # Get the global rotation matrix in the detector frame
    # Psr = rotations.eu2om(np.array([0.0, tilt * np.pi / 180, 0.0], dtype=float))
    # Rr = np.matmul(Psr.T, np.matmul(Rs, Psr))
    # Rr = np.matmul(Psr, np.matmul(Rs, Psr.T))
    Rr = Rs
    Rr = Rr / Rr[..., 2, 2][:, None, None]

    ## Now get the homography
    # Decompose inputs
    # m11, m12, m13 = Rr[..., 0, 0], Rr[..., 0, 1], Rr[..., 0, 2]
    # m21, m22, m23 = Rr[..., 1, 0], Rr[..., 1, 1], Rr[..., 1, 2]
    m31, m32      = Rr[..., 2, 0], Rr[..., 2, 1]
    # Compose shape function matrix values
    g0 = DD + m31 * x01 + m32 * x02
    # g11 = DD * m11 - m31 * x01 - g0
    # g22 = DD * m22 - m32 * x02 - g0
    # g13 = DD * ((m11 - 1) * x01 + m12 * x02 + m13 * DD) + x01 * (DD - g0)
    # g23 = DD * (m21 * x01 + (m22 - 1) * x02 + m23 * DD) + x02 * (DD - g0)
    # Compose homography
    # h11 = g11 / g0
    # h12 = (DD * m12 - m32 * x01) / g0
    # h13 = g13 / g0
    # h21 = (DD * m21 - m31 * x02) / g0
    # h22 = g22 / g0
    # h23 = g23 / g0
    h31 = m31 / g0
    h32 = m32 / g0
    homographies = xyt2h_partial(shifts)
    homographies[..., 6] = h31
    homographies[..., 7] = h32
    # homographies = np.squeeze(np.array([h11, h12, h13, h21, h22, h23, h31, h32]))  # shape is (8) for a 2D input, (8, N) for a 3D input, (8, H, W) for a 4D input
    return np.squeeze(homographies)


def h2F(H, PC):
    """Calculate the deviatoric deformation gradient from a homography using the projection geometry (pattern center).
    Note that the deformation gradient is insensitive to hydrostatic dilation.
    Within the PC, the detector distance, DD (PC[2]), must be positive. The calculation requires the distance to be negative,
    as the homography is calculated from the detector to the sample. This function will negate the provided DD distance.

    Args:
        H (np.ndarray): The homography matrix.
        PC (np.ndarray): The pattern center.

    Returns:
        np.ndarray: The deviatoric deformation gradient."""
    # Reshape the homography if necessary
    if H.ndim == 1:
        H.reshape(1, 8)

    # Extract the data from the inputs
    if np.asarray(PC).ndim == 1:
        input_pc = np.array(PC)
        PC = np.ones(H.shape[:-1] + (3,))
        PC[..., :3] = input_pc[:3]
    x01, x02, DD = PC[..., 0], PC[..., 1], PC[..., 2]
    h11, h12, h13, h21, h22, h23, h31, h32 = H[..., 0], H[..., 1], H[..., 2], H[..., 3], H[..., 4], H[..., 5], H[..., 6], H[..., 7]

    # Negate the detector distance becase our coordinates have +z pointing from the sample towards the detector
    # The calculation is the opposite, so we need to negate the distance
    # DD = -DD

    # Calculate the deformation gradient
    beta0 = 1 - h31 * x01 - h32 * x02
    Fe11 = 1 + h11 + h31 * x01
    Fe12 = h12 + h32 * x01
    Fe13 = (h13 - h11*x01 - h12*x02 + x01*(beta0 - 1))/DD
    Fe21 = h21 + h31 * x02
    Fe22 = 1 + h22 + h32 * x02
    Fe23 = (h23 - h21*x01 - h22*x02 + x02*(beta0 - 1))/DD
    Fe31 = DD * h31
    Fe32 = DD * h32
    Fe33 = beta0
    Fe = np.array([[Fe11, Fe12, Fe13], [Fe21, Fe22, Fe23], [Fe31, Fe32, Fe33]]) / beta0

    # Reshape the output if necessary
    if Fe.ndim == 4:
        Fe = np.moveaxis(Fe, (0, 1, 2, 3), (2, 3, 0, 1))
    elif Fe.ndim == 3:
        Fe = np.squeeze(np.moveaxis(Fe, (0, 1, 2), (1, 2, 0)))

    return Fe


def F2h(Fe: np.ndarray, PC: tuple | list | np.ndarray) -> np.ndarray:
    """Calculate the homography from a deformation gradient using the projection geometry (pattern center).

    Args:
        Fe (np.ndarray): The deformation gradient.
        PC (tuple | list | np.ndarray): The pattern center.

    Returns:
        np.ndarray: The homography matrix."""
    # Reshape the deformation gradient if necessary
    if Fe.ndim == 3:
        Fe = Fe[None, ...]
    elif Fe.ndim == 2:
        Fe = Fe[None, None, ...]

    # Extract the data from the inputs
    x01, x02, DD = PC
    F11, F12, F13, F21, F22, F23, F31, F32 = Fe[..., 0, 0], Fe[..., 0, 1], Fe[..., 0, 2], Fe[..., 1, 0], Fe[..., 1, 1], Fe[..., 1, 2], Fe[..., 2, 0], Fe[..., 2, 1]

    # Negate the detector distance becase our coordinates have +z pointing from the sample towards the detector
    # The calculation is the opposite, so we need to negate the distance
    # DD = -DD

    # Calculate the homography
    g0 = DD + F31 * x01 + F32 * x02
    g11 = DD * F11 - F31 * x01 - g0
    g22 = DD * F22 - F32 * x02 - g0
    g13 = DD * ((F11 - 1) * x01 + F12 * x02 + F13 * DD) + x01 * (DD - g0)
    g23 = DD * (F21 * x01 + (F22 - 1) * x02 + F23 * DD) + x02 * (DD - g0)
    h11 = g11 / g0
    h12 = (DD * F12 - F32 * x01) / g0
    h13 = g13 / g0
    h21 = (DD * F21 - F31 * x02) / g0
    h22 = g22 / g0
    h23 = g23 / g0
    h31 = F31 / g0
    h32 = F32 / g0
    H = np.array([h11, h12, h13, h21, h22, h23, h31, h32])

    # Reshape the output if necessary
    if H.ndim == 3:
        H = np.squeeze(np.moveaxis(H, (0, 1, 2), (1, 2, 0)))
    if H.ndim == 2:
        H = np.squeeze(H.T)

    return H


def F2strain(Fe: np.ndarray, C: np.ndarray = None, small_strain: bool = False) -> tuple:
    """Calculate the elastic strain tensor from the deformation gradient.
    Also calculates the lattice rotation matrix.

    Args:
        Fe (np.ndarray): The deformation gradient.
        C (np.ndarray): The stiffness tensor. If not provided, only the strain is returned

    Returns:
        epsilon (np.ndarray): The elastic strain tensor
        omega (np.ndarray): The lattice rotation matrix
        stress (np.ndarray): The stress tensor. Only returned if the stiffness tensor is provided."""
    if Fe.ndim == 4:
        I = np.eye(3)[None, None, ...]
    elif Fe.ndim == 3:
        I = np.squeeze(np.eye(3)[None, ...])
    elif Fe.ndim == 2:
        I = np.eye(3)


    # Calculate the rotation tensor
    # Use the polar decomposition of the deformation gradient to decompose it into the rotation matrix and the stretch tensor
    W, S, V = np.linalg.svd(Fe, full_matrices=True)
    omega_finite = np.matmul(W, V)

    # Calculate the small strain tensor
    # Use small strain theory to decompose the deformation gradient into the elastic strain tensor and the rotation matrix
    if small_strain:
        d = Fe - I
        if Fe.ndim == 2:
            dT = d.T
        elif Fe.ndim == 3:
            dT = d.transpose(0, 2, 1)
        elif Fe.ndim == 4:
            dT = d.transpose(0, 1, 3, 2)
        epsilon = 0.5 * (d + dT)
    else:
        Sigma = np.einsum('...i,ij->...ij', S, np.eye(3))
        if Fe.ndim == 2:
            epsilon = np.matmul(W, np.matmul(Sigma, W.T)) - I
        elif Fe.ndim == 3:
            epsilon = np.matmul(W, np.matmul(Sigma, W.transpose(0, 2, 1))) - I
        elif Fe.ndim == 4:
            epsilon = np.matmul(W, np.matmul(Sigma, W.transpose(0, 1, 3, 2))) - I

    # Convert finite rotation matrix to a lattice rotation matrix
    v = rotations.om2ax(omega_finite)
    rotation_vector = v[..., :3] * v[..., 3][..., None]
    omega = np.zeros_like(omega_finite)
    omega[..., 0, 1] = -rotation_vector[..., 2]
    omega[..., 0, 2] = rotation_vector[..., 1]
    omega[..., 1, 2] = rotation_vector[..., 0]
    omega[..., 1, 0] = -omega[..., 0, 1]
    omega[..., 2, 0] = -omega[..., 0, 2]
    omega[..., 2, 1] = -omega[..., 1, 2]

    # Convert to sample frame rotation
    if Fe.ndim == 3:
        omega = np.transpose(omega, (0, 2, 1))
    elif Fe.ndim == 4:
        omega = np.transpose(omega, (0, 1, 3, 2))

    # Apply a tolerance
    # epsilon[np.abs(epsilon) < 1e-4] = 0
    # omega[np.abs(omega) < 1e-4] = 0

    if C is None:
        return epsilon, omega
    else:
        """Assume the surface normal stress is zero to separate e11, e22, e33
        Solve the following system of equations:
         0 = C3311*e11 + C3322*e22 + C3333*e33
         X =     1*e11 +     0*e22 -     1*e33
         Y =     0*e11 +     1*e22 -     1*e33
        where X the 11 strain from the deformation gradient (technically e11 - e33) and 
        Y is the 22 strain from the deformation gradient (technically e22 - e33)"""
        # Grab elastic constants
        # C3333 = C[..., 2, 2]  # C33 = C11 for cubic
        # C3311 = C[..., 0, 2]  # C13 = C12 for cubic
        # C3322 = C[..., 1, 2]  # C23 = C12 for cubic
        # # Construct the dependent variables vector
        # _0 = np.zeros_like(C3333)
        # _1 = np.ones_like(C3333)
        # b = np.array([_0, epsilon[..., 0, 0], epsilon[..., 1, 1]])
        # b = np.moveaxis(b, 0, -1)
        # # Construct the coefficient matrix and repeat it for broadcasting
        # A = np.array([[C3311, C3322, C3333], [_1, _0, -1*_1], [_0, _1, -1*_1]])
        # A = np.transpose(A, (2, 3, 0, 1))
        # # Solve the system of equations
        # x = np.linalg.solve(A, b)
        # # Store the results
        # epsilon[..., 0, 0] = x[..., 0]
        # epsilon[..., 1, 1] = x[..., 1]
        # epsilon[..., 2, 2] = x[..., 2]

        ### Original method for using zero surface traction to get e33
        C3311Xep11 = C[..., 0, 1] * epsilon[..., 0, 0]  #C12 * epsilon11
        C3322Xep22 = C[..., 1, 2] * epsilon[..., 1, 1]  #C23 * epsilon22
        C3323Xep23 = C[..., 2, 3] * epsilon[..., 1, 2]  #C34 * epsilon23
        C3331Xep31 = C[..., 2, 4] * epsilon[..., 2, 0]  #C35 * epsilon31
        C3312Xep12 = C[..., 0, 5] * epsilon[..., 0, 1]  #C36 * epsilon12
        e33 = - (C3311Xep11 + C3322Xep22 + 2*(C3323Xep23 + C3331Xep31 + C3312Xep12)) / C[..., 2, 2]
        epsilon[..., 2, 2] = e33
        epsilon[..., 0, 0] += e33
        epsilon[..., 1, 1] += e33

        # Calculate the stress tensor using Hooke's law
        epsilon_voigt = np.zeros(Fe.shape[:-2] + (6,))
        epsilon_voigt[..., 0] = epsilon[..., 0, 0]
        epsilon_voigt[..., 1] = epsilon[..., 1, 1]
        epsilon_voigt[..., 2] = epsilon[..., 2, 2]
        epsilon_voigt[..., 3] = epsilon[..., 1, 2] * 2
        epsilon_voigt[..., 4] = epsilon[..., 0, 2] * 2
        epsilon_voigt[..., 5] = epsilon[..., 0, 1] * 2
        stress_voigt = np.einsum('...ij,...j', C, epsilon_voigt)
        stress = np.zeros(Fe.shape[:-2] + (3, 3))
        stress[..., 0, 0] = stress_voigt[..., 0]
        stress[..., 1, 1] = stress_voigt[..., 1]
        stress[..., 2, 2] = stress_voigt[..., 2]
        stress[..., 1, 2] = stress_voigt[..., 3]
        stress[..., 0, 2] = stress_voigt[..., 4]
        stress[..., 0, 1] = stress_voigt[..., 5]

        # Apply a tolerance
        # stress[np.abs(stress) < 1e-4] = 0

        return epsilon, omega, stress
