import numpy as np
from scipy import interpolate, linalg, optimize

import rotations


def HREBSD(numanlges: int, Euler_Angle: np.ndarray, progname: str, nmldeffile: str):
    """Master function for the HREBSD
    INPUT:
        numanlges: 
        Euler_Angle: array of the Euler angles, shape (3)
        progname: 
        nmldeffile: 
        enl: """
    params = read_nml(nmldeffile)
    params = dict(numsx=4, numsy=4, size_ROI=5, exptfile="PATFILE", ipf_wd="IDK")
    binx, biny = params["numsx"], params["numsy"]

    raise NotImplementedError("This is not implemented yet")


def cross_correlation_function(dims: tuple, a: np.ndarray, b: np.ndarray) -> tuple(np.ndarray, tuple):
    """Cross correlation function from EMSoft in the SEM/EMHREBSD.f90 file
    INPUT:
        dims: tuple of dimensions of the array to be x-correlated
        a: reference image array
        b: test image array
    OUTPUT:
        c: x-correlation function of a and b
        max_pos: tuple of the location of the maximum value in the x-correlation function
    """
    cdims = 2 * np.array(dims) - 1
    apad = np.zeros(cdims)
    bpad = np.zeros(cdims)
    apad[:dims[0], :dims[1]] = a
    bpad[:dims[0], :dims[1]] = b
    ffta = np.fft.fft2(apad)
    fftb = np.fft.fft2(bpad)
    ### TODO: Check if this is the correct way to do this
    ### Might not need conj here, might need it on both?
    c = np.fft.ifft2(ffta * np.conj(fftb))
    ### TODO: Check if this is the correct way to do this
    max_pos = np.unravel_index(np.argmax(c), c.shape)
    return (c, max_pos)


def peak_interpolate(interp_size: int, size_interp: int, z_size: int, max_pos: tuple, ngrid: np.ndarray, interp_step: float, interp_ngrid: np.ndarray, z: np.ndarray) -> tuple(np.ndarray, np.ndarray, np.ndarray):
    """Peak interpolation function from EMSoft in the SEM/EMHREBSD.f90 file
    ### TODO: interp_size is not used, need to check if this is correct
    INPUT:
        interp_size: size of the interpolation grid
        size_interp: size of the interpolation grid
        z_size: size of the interpolation grid
        max_pos: tuple of the location of the maximum value in the x-correlation function
        ngrid: array of the interpolation grid, shape (interp_size)
        interp_step: interpolation step
        interp_ngrid: array of the interpolation grid, shape (size_interp)
        z: array of the interpolation grid, shape (interp_size, interp_size)
    OUTPUT:
        q: shift vector
        interp_ngrid: array of the interpolation grid, shape (size_interp)
        z: array of the interpolation grid, shape (interp_size, interp_size)
    """
    f = interpolate.RectBivariateSpline(ngrid, ngrid, z, kx=1, ky=1, s=0)(interp_ngrid[j], interp_ngrid[i])
    f = lambda args: rgbi3p(*args)
    zi = np.zeros((size_interp, size_interp), dtype=float)

    for i in range(size_interp):
        for j in range(size_interp):
            if i == 0 and j == 0:
                md = 1
            else:
                md = 2
            # zi[j, i] = f(md, interp_size, interp_size, ngrid, ngrid, z, 1, interp_ngrid[j], interp_ngrid[i])
            zi[j, i] = f[j, i]

    max_pos_interp = np.unravel_index(np.argmax(zi), zi.shape)
    interp_half = (size_interp - 1) / 2
    q = np.array([(max_pos[1] - (z_size + 1) / 2) + ((max_pos_interp[1] - interp_half) * interp_step),
                  ((z_size + 1) / 2 - max_pos[0]) + ((interp_half - max_pos_interp[0]) * interp_step)])

    return (q, interp_ngrid, zi)


def setROI(L: tuple, PC: tuple, n_roi: int, roi_distance: float) -> tuple(np.ndarray, np.ndarray):
    """Set the region of interest
    INPUT:
        L: tuple of the dimensions of the image
        PC: tuple of the pattern center
        n_roi: number of regions of interest
        roi_distance: distance between the regions of interest
        roi_center: array of the center of the regions of interest, shape (n_roi, 2)
        r: array of the regions of interest, shape (3, n_roi)
    OUTPUT:
        roi_center: array of the center of the regions of interest, shape (n_roi, 2)
        r: array of the regions of interest, shape (3, n_roi)
    """
    Lx, Ly = L
    PCx, PCy, DD = PC
    roi_center = np.zeros((n_roi, 2))
    r = np.zeros((3, n_roi))

    for i in range(1, n_roi + 1):
        if i < n_roi:
            roi_center[i-1] = np.floor((Lx / 2 + roi_distance * np.cos(i * 2 * np.pi / (n_roi - 1)), Ly / 2 + roi_distance * np.sin(i * 2 * np.pi / (n_roi - 1))))
        else:
            roi_center[i-1] = [Lx / 2, Ly / 2]
        r[:, i-1] = [roi_center[i-1][0] - PCx, roi_center[i-1][1] - PCy, DD]
    
    return (roi_center, r)


def main_minf(N: int, r: np.ndarray, q: np.ndarray, Euler_Angle:  np.ndarray, C_c: np.ndarray, R_tilt: np.ndarray) -> tuple(np.ndarray, float):
    """Main function for the minimization of the objective function
    INPUT:
        N: number of rois
        r: array of the regions of interest, shape (3, n_roi)
        q: , shape (3, n_roi)
        Euler_Angle: array of the Euler angles, shape (3)
        C_c: shape (6,6)
        R_tilt: shape (9)
    OUTPUT:
        Ftensor: shape (9)
        minf: minimum value of the objective function"""
    ### TODO: write this function
    f = np.zeros((6, r.shape[1]), dtype=float)
    f[:3] = r
    f[3:] = q

    # Determine the orientation matricies and rotation matricies for stiffness tensor coordinate transformation
    gs2c, gc2s, RM, RN = StiffnessRotation(Euler_Angle)
    RM_inv = inv_ludcmp(RM)

    # Rotate stiffness tensor from crystal to sample frame
    C = np.zeros((9, 9), dtype=float)
    C_s = np.matmul(np.matmul(RM_inv, C_c), RN)
    C[:6, :6] = C_s
    C[7, :] = R_tilt

    # Setup minimization parameters
    step_tol = 1e-6
    func_tol = 1e-6
    max_eval = 800
    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    lb = x - 0.05
    ub = x + 0.05
    
    # Minimize the objective function
    val = optimize.fmin_cobyla(myfunc, x, myconstraint, args=(f,), consargs=(C,), maxfun=max_eval)
    return val


# def myfunc(n: int, x: np.ndarray, need_gradient: bool, f: np.ndarray) -> float:
def myfunc(x: np.ndarray, f: np.ndarray) -> float:
    """Objective function for the minimization
    INPUT:
        x: array of the variables, shape (n)
        f: array of the objective function, shape (6, n)
    OUTPUT:
        fval: value of the objective function"""
    ### TODO: figure out what the actual inputs/outputs are
    ### TODO: write this function
    need_gradient = False
    DD = f[2, 0]
    dims2 = f.shape
    row1, row2, row3 = np.zeros((3, dims2[1])), np.zeros((3, dims2[1])), np.zeros((3, dims2[1]))
    r1 = np.repeat(f[0], 3, axis=0).reshape(3, dims2[1])
    r2 = np.repeat(f[1], 3, axis=0).reshape(3, dims2[1])
    r3 = np.repeat(f[2], 3, axis=0).reshape(3, dims2[1])
    q1 = np.repeat(f[3], 3, axis=0).reshape(3, dims2[1])
    q2 = np.repeat(f[4], 3, axis=0).reshape(3, dims2[1])
    q3 = np.repeat(f[5], 3, axis=0).reshape(3, dims2[1])
    row1[0] = 1.0
    row2[1] = 1.0
    row3[2] = 1.0
    if need_gradient:
        pass
    val = 0.5*np.sum(
        linalg.norm(
            (DD/(x[2]*r1 + x[5]*r2 + x[8]*r3)) * (x[0]*(r1*row1)+x[4]*(r2*row1)+x[6]*(r3*row1)+x[1]*(r1*row2)+x[4]*(r2*row2)+x[7]*(r3*row2) + x[2]*(r1*row3)+x[5]*(r2*row3) + x[8]*(r3*row3)) - ((r1+q1)*row1 + (r2+q2)*row2 + (r3+q3)*row3),
            axis=1)**2)
    return val


# def myconstraint(n: int, x: np.ndarray, need_gradient: bool, C: np.ndarray) -> float:
def myconstraint(x: np.ndarray, C: np.ndarray) -> float:
    """Constraint function for the minimization
    INPUT:
        x: array of the variables, shape (n)
        C: array of the constraint function, shape (9, 9)
    OUTPUT:
        cval: value of the constraint function"""
    ### TODO: figure out what the actual inputs/outputs are
    ### TODO: write this function
    need_gradient = False
    # Get rotation matrix for sample tilt
    R = C[6].reshape(3, 3)
    # Extract stiffness components
    C_s = C[:6, :6]
    if need_gradient:
        grad = np.zeros(n, dtype=float)
        grad[0] = np.abs(C[2, 0])
        grad[1] = np.abs(C[2, 5])
        grad[2] = np.abs(C[2, 4])
        grad[3] = np.abs(C[2, 5])
        grad[4] = np.abs(C[2, 1])
        grad[5] = np.abs(C[2, 3])
        grad[6] = np.abs(C[2, 4])
        grad[7] = np.abs(C[2, 3])
        grad[8] = np.abs(C[2, 2])
    F_s = np.matmul(np.matmul(R, x.reshape(3, 3)), np.transpose(R))
    beta = F_s - np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, 3)
    e = 0.5 * (beta + np.transpose(beta))
    e_s = np.array([e[0,0], e[1,1], e[2,2], 2*e[1,2], 2*e[0,2], 2*e[0,1]]).reshape(6, 1)
    s_s = np.matmul(C_s, e_s)
    val = s_s[2, 0]
    return val


def StiffnessRotation(Euler_Angle: np.ndarray) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Compute the stiffness rotation matrix
    INPUT:
        Euler_Angle: array of the Euler angles, shape (3)
    OUTPUT:
        gs2c: shape (3,3)
        gc2s: shape (3,3)
        RM: shape (3,3)
        RN: shape (3,3)"""
    ### TODO: Get rotations functions together
    gs2c = rotations.eu2om(Euler_Angle)
    gc2s = np.transpose(gs2c)
    R = gs2c
    RM = np.array([R[0,0]**2, R[0,1]**2, R[0,2]**2, 1*R[0,1]*R[0,2], 1*R[0,2]*R[0,0], 1*R[0,0]*R[0,1],
                   R[1,0]**2, R[1,1]**2, R[1,2]**2, 1*R[1,1]*R[1,2], 1*R[1,2]*R[1,0], 1*R[1,0]*R[1,1],
                   R[2,0]**2, R[2,1]**2, R[2,2]**2, 1*R[2,1]*R[2,2], 1*R[2,2]*R[2,0], 1*R[2,0]*R[2,1],
                   R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2], R[1,1]*R[2,2]+R[1,2]*R[2,1], R[1,0]*R[2,2]+R[1,2]*R[2,0],
                   R[1,1]*R[2,0]+R[1,0]*R[2,1],R[0,0]*R[2,0], R[0,1]*R[2,1], R[0,2]*R[2,2], R[0,1]*R[2,2]+R[0,2]*R[2,1],
                   R[0,2]*R[2,0]+R[0,0]*R[2,2], R[0,0]*R[2,1]+R[0,1]*R[2,0],R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2],
                   R[0,1]*R[1,2]+R[0,2]*R[1,1], R[0,2]*R[1,0]+R[0,0]*R[1,2], R[0,0]*R[1,1]+R[0,1]*R[1,0]]).reshape(6,6)
    RM = np.transpose(RM)

    RN = np.array([R[0,0]**2, R[0,1]**2, R[0,2]**2, R[0,1]*R[0,2], R[0,2]*R[0,0], R[0,0]*R[0,1],
                   R[1,0]**2, R[1,1]**2, R[1,2]**2, R[1,1]*R[1,2], R[1,2]*R[1,0], R[1,0]*R[1,1],
                   R[2,0]**2, R[2,1]**2, R[2,2]**2, R[2,1]*R[2,2], R[2,2]*R[2,0], R[2,0]*R[2,1],
                   1*R[1,0]*R[2,0], 1*R[1,1]*R[2,1], 1*R[1,2]*R[2,2], R[1,1]*R[2,2]+R[1,2]*R[2,1],
                   R[1,0]*R[2,2]+R[1,2]*R[2,0], R[1,1]*R[2,0]+R[1,0]*R[2,1], 1*R[0,0]*R[2,0],
                   1*R[0,1]*R[2,1], 1*R[0,2]*R[2,2], R[0,1]*R[2,2]+R[0,2]*R[2,1],
                   R[0,2]*R[2,0]+R[0,0]*R[2,2], R[0,0]*R[2,1]+R[0,1]*R[2,0], 1*R[0,0]*R[1,0], 1*R[0,1]*R[1,1],
                   1*R[0,2]*R[1,2], R[0,1]*R[1,2]+R[0,2]*R[1,1], R[0,2]*R[1,0]+R[0,0]*R[1,2],
                   R[0,0]*R[1,1]+R[0,1]*R[1,0]]).reshape(6,6)
    RN = np.transpose(RN)
    
    return (gs2c, gc2s, RM, RN)


def inv_ludcmp(a: np.ndarray, n: int) -> np.ndarray:
    """Invert a matrix using LU decomposition
    INPUT:
        a: array of the matrix to be inverted, shape (n, n)
        n: size of the matrix
    OUTPUT:
        ainv: inverted matrix, shape (n, n)"""
    p, l, u = linalg.lu(a, permute_l=False)
    l = np.dot(p, l)
    l_inv = np.linalg.inv(l)
    u_inv = np.linalg.inv(u)
    ainv = np.dot(u_inv, l_inv)
    return ainv


def Rot2LatRot(R_finite: np.ndarray) -> np.ndarray:
    """Convert a finite rotation matrix to a lattice rotation matrix
    INPUT:
        R_finite: array of the finite rotation matrix, shape (3, 3)
    OUTPUT:
        R_lat: array of the lattice rotation matrix, shape (3, 3)"""
    v = rotations.om2ax(R_finite)
    rotation_vector = v[:3] * v[3]
    R_lat = np.zeros_like(R_finite)
    R_lat[0, 1] = -rotation_vector[2]
    R_lat[0, 2] = rotation_vector[1]
    R_lat[1, 2] = rotation_vector[0]
    R_lat[1, 0] = -R_lat[0, 1]
    R_lat[2, 0] = -R_lat[0, 2]
    R_lat[2, 1] = -R_lat[1, 2]
    R_lat = np.transpose(R_lat) # convert to sample frame rotation
    return R_lat


def fRemapbicubic(binx: int, biny:int, R: np.ndarray, PC: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Remap an image using bicubic interpolation
    INPUT:
        binx: binning in the x direction
        biny: binning in the y direction
        R: array of the rotation matrix, shape (3, 3)
        PC: array of the pattern center, shape (3)
        image: array of the image, shape (n, n)
    OUTPUT:
        image: array of the remapped image, shape (n, n)"""
    image_rotated = np.zeros_like(image)
    for row2 in range(biny):
        for col2 in range(binx):
            yy = -(row2 - PC[1])
            xx = col2 - PC[0]

            rp = np.array([xx, yy, PC[2]]).reshape(3, 1)
            r_rot = np.matmul((np.matmul(R, rp).reshape(3).dot(np.array([0, 0, 1]))) * R, rp).reshape(3)
            row1 = -r_rot[1] + PC[1]
            col1 = r_rot[0] + PC[0]

            b = row1
            a = col1
            x1 = np.floor(a)
            y1 = np.floor(b)

            ### TODO: Check if this is condition statement and the indexing is correct
            if x1 >= 1 and y1 >= 1 and x1 <= binx-2 and y1 <= biny-2:
                P = image[y1-1:y1+3, x1-1:x1+3]
                dx = a - x1
                dy = b - y1
                intensity = bicubicInterpolate(P, dx, dy)
            else:
                intensity = 0
            image_rotated[row2, col2] = intensity
    return image_rotated

def bicubicInterpolate(p: np.ndarray, x: float, y: float) -> float:
    """Bicubic interpolation function
    INPUT:
        p: array of the interpolation points, shape (4, 4)
        x: x coordinate of the interpolation point
        y: y coordinate of the interpolation point
    OUTPUT:
        q: interpolated value at the interpolation point"""
    q1 = cubicInterpolate(p[0], x)
    q2 = cubicInterpolate(p[1], x)
    q3 = cubicInterpolate(p[2], x)
    q4 = cubicInterpolate(p[3], x)
    q = cubicInterpolate(np.array([q1, q2, q3, q4]), y)
    return q


def cubicInterpolate(p: np.ndarray, x: float) -> float:
    """Cubic interpolation function
    INPUT:
        p: array of the interpolation points, shape (4)
        x: x coordinate of the interpolation point
    OUTPUT:
        q: interpolated value at the interpolation point"""
    q = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
    return q


def rgbi3p(md: int, nxd: int, nyd: int, xd: np.ndarray, yd: np.ndarray, zd: np.ndarray, nip: int, xi: float, yi: float) -> np.ndarray:
    """Rectanguular-grid bivariate interpolation
    INPUT:
        md: mode of computation
            1 for new, 2 for old
        nxd: number of the input-grid data points in the x direction, minimum 2
        nyd: number of the input-grid data points in the y direction, minimum 2
        xd: array of x coordinates of the input grid, shape (nxd)
        yd: array of y coordinates of the input grid, shape (nyd)
        zd: array of the input grid data, shape (nxd, nyd)
        nip: number of interpolation points to be computed, minimum 1
        xi: x coordinate of the interpolation point, shape (nip)
        yi: y coordinate of the interpolation point, shape (nip)
    OUTPUT:
        zi: interpolated value at the interpolation point, shape (nip)"""
    raise NotImplementedError