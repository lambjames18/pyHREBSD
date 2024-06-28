from tqdm.auto import tqdm
import numpy as np
import rotations


# Constants
R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
R3 = 0.8660254037844386467637231707529361834714026269051903140279034897
LAUE_O = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [R2, 0, 0, R2],
        [R2, 0, 0, -R2],
        [0, R2, R2, 0],
        [0, -R2, R2, 0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [R2, R2, 0, 0],
        [R2, -R2, 0, 0],
        [R2, 0, R2, 0],
        [R2, 0, -R2, 0],
        [0, R2, 0, R2],
        [0, -R2, 0, R2],
        [0, 0, R2, R2],
        [0, 0, -R2, R2],
    ],
    dtype=np.float64,
)


def segment_grains(quaternions, threshold):
    # quaternions: 2D grid of quaternions
    # threshold: misorientation threshold for grain boundaries

    rows, cols = quaternions.shape[:2]
    grain_ids = np.zeros((rows, cols), dtype=int)
    average_misorientation = np.zeros((rows, cols), dtype=float)
    current_grain_id = 1
    coords = np.array(np.meshgrid(np.arange(rows), np.arange(cols))).T.reshape(-1, 2)
    progress_bar = tqdm(total=rows * cols, desc="Segmenting grains")
    for i, j in coords:
        if grain_ids[i, j] == 0:
            # Start a new grain
            grain_ids[i, j] = current_grain_id
            queue = [(i, j)]
            while queue:
                x, y = queue.pop(0)
                reference_quaternion = quaternions[x, y]
                neighbors = []
                if x - 1 >= 0:
                    neighbors.append((x - 1, y))
                if x + 1 < rows:
                    neighbors.append((x + 1, y))
                if y - 1 >= 0:
                    neighbors.append((x, y - 1))
                if y + 1 < cols:
                    neighbors.append((x, y + 1))
                neighbors = np.array(neighbors)
                neighbors_quaternions = quaternions[
                    neighbors[:, 0], neighbors[:, 1]
                ]
                angles = misorientation(
                    reference_quaternion,
                    neighbors_quaternions,
                    degrees=True,
                    symmetry=True,
                    both=True,
                )
                angles = np.abs(angles)[..., 3]
                m0 = angles <= threshold
                m1 = grain_ids[neighbors[:, 0], neighbors[:, 1]] == 0
                m2 = m1 | (grain_ids[neighbors[:, 0], neighbors[:, 1]] == current_grain_id)
                average_misorientation[x, y] = angles[m0 & m2].mean()
                for i, (nx, ny) in enumerate(neighbors[m0 & m1]):
                    grain_ids[nx, ny] = current_grain_id
                    queue.append((nx, ny))
                # Update progress bar
                progress_bar.update(1)
            current_grain_id += 1

    return grain_ids, average_misorientation


def segment_grains_3d(quaternions, threshold, mask=None):
    # quaternions: 3D grid of quaternions
    # threshold: misorientation threshold for grain boundaries

    if mask is None:
        mask = np.ones(quaternions.shape[:3], dtype=bool)
    rows, cols, depth = quaternions.shape[:3]
    grain_ids = np.zeros((rows, cols, depth), dtype=int)
    average_misorientation = np.zeros((rows, cols, depth), dtype=float)
    current_grain_id = 1
    coords = np.array(
        np.meshgrid(np.arange(rows), np.arange(cols), np.arange(depth))
    ).T.reshape(-1, 3)
    progress_bar = tqdm(total=rows * cols * depth, desc="Segmenting grains")
    for i, j, k in coords:
        if (grain_ids[i, j, k] == 0) and mask[i, j, k]:
            # Start a new grain
            grain_ids[i, j, k] = current_grain_id
            queue = [(i, j, k)]
            while queue:
                x, y, z = queue.pop(0)
                reference_quaternion = quaternions[x, y, z]
                neighbors = []
                if (x - 1 >= 0) and mask[x - 1, y, z]:
                    neighbors.append((x - 1, y, z))
                if (x + 1 < rows) and mask[x + 1, y, z]:
                    neighbors.append((x + 1, y, z))
                if (y - 1 >= 0) and mask[x, y - 1, z]:
                    neighbors.append((x, y - 1, z))
                if (y + 1 < cols) and mask[x, y + 1, z]:
                    neighbors.append((x, y + 1, z))
                if (z - 1 >= 0) and mask[x, y, z - 1]:
                    neighbors.append((x, y, z - 1))
                if (z + 1 < depth) and mask[x, y, z + 1]:
                    neighbors.append((x, y, z + 1))
                neighbors = np.array(neighbors)
                neighbors_quaternions = quaternions[
                    neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]
                ]
                angles = misorientation(
                    reference_quaternion,
                    neighbors_quaternions,
                    degrees=True,
                    symmetry=True,
                    both=True,
                )
                angles = angles.reshape(neighbors_quaternions.shape)
                angles = np.abs(angles)[..., 3]
                m0 = angles <= threshold
                m1 = grain_ids[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]] == 0
                m2 = m1 | (grain_ids[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]] == current_grain_id)
                average_misorientation[x, y, z] = angles[m0 & m2].mean()
                for i, (nx, ny, nz) in enumerate(neighbors[m0 & m1]):
                    grain_ids[nx, ny, nz] = current_grain_id
                    queue.append((nx, ny, nz))
                # Update progress bar
                progress_bar.update(1)
            current_grain_id += 1
    return grain_ids, average_misorientation


def misorientation(q1, q2, degrees=True, symmetry=True, both=True):
    # N1: number of quaternions in q1
    # N2: number of quaternions in q2
    if symmetry:
        axangle = misorientation_sym(q1, inverse_qu(q2), both=both)
    else:
        mis = quaternion_raw_multiply(q1, inverse_qu(q2))
        if len(mis.shape) == 1:
            mis = mis.reshape(1, 1, *mis.shape)
        axangle = rotations.qu2ax(mis)
        axangle[axangle[..., 2] < 0] = -axangle[axangle[..., 2] < 0] + 0
    # axangle is shape (N1, N2, 24, 24, 4) symboth, (N1, N2, 1, 24, 4) symnoboth, or (N1, N2, 4) nsym
    # combine the two inner dimensions
    axangle = axangle.reshape(axangle.shape[0], axangle.shape[1], -1, axangle.shape[-1])
    argmin = np.abs(axangle[..., 3]).argmin(axis=-1)
    min_axangles = axangle[np.arange(axangle.shape[0]), np.arange(axangle.shape[1]), argmin]
    if degrees:
        min_axangles[..., 3] = np.rad2deg(min_axangles[..., 3])
    return np.squeeze(min_axangles)


def misorientation_sym(q1, q2, both=True):
    if both:
        q1s = quaternion_raw_multiply(q1, LAUE_O)
    else:
        q1s = q1.reshape(1, *q1.shape)
    q2s = quaternion_raw_multiply(q2, LAUE_O)
    mis = quaternion_raw_multiply(q1s, q2s)
    if not both:
        mis = mis.reshape(mis.shape[0], 1, *mis.shape[1:])
    axangle = rotations.qu2ax(mis)
    axangle[axangle[..., 2] < 0] = -axangle[axangle[..., 2] < 0] + 0
    return axangle


def quaternion_raw_multiply(
    a: np.array, b: np.array, verbose: bool = False
) -> np.array:
    a_shape, b_shape = a.shape, b.shape
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("The last dimension of both arrays must be 4.")
    if a.ndim == b.ndim + 1:
        b = b.reshape((1,) + b.shape)
    elif a.ndim + 1 == b.ndim:
        a = a.reshape((1,) + a.shape)
    elif a.ndim != b.ndim:
        raise ValueError(
            "The two arrays must have the same number of dimensions or one more dimension in one of the arrays."
        )
    # Make sure we have an array of quaternions, not just a quaternion
    if a.ndim == 1:
        a = a.reshape((1, -1))
        b = b.reshape((1, -1))
    # Now we setup broadcasting
    a = a.reshape(
        a.shape[:1] + (1,) + a.shape[1:]
    )  # this sets up broadcasting for the number of unique quaternions in a
    b = b.reshape(
        (1,) + b.shape
    )  # this sets up broadcasting for the number of unique quaternions in b
    if a.ndim > 3:
        a = a.reshape(a.shape[:-1] + (1,) + a.shape[-1:])
        b = b.reshape(b.shape[:-2] + (1,) + b.shape[-2:])
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    stack = np.stack((ow, ox, oy, oz), -1)
    if verbose:
        print(
            " . input:",
            a_shape,
            b_shape,
            "convert:",
            a.shape,
            b.shape,
            "output:",
            stack.shape,
        )
    return standardize_qu(stack)


def standardize_qu(q: np.array) -> np.array:
    q_out = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q_out = np.where(q_out[..., :1] < 0, -q_out, q_out)
    q_out += 0
    return q_out


def inverse_qu(qu: np.array):
    qu[..., 1:] = -qu[..., 1:]
    return qu


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import h5py

    import utilities

    # path = "E:/cells/CoNi90-ParallelCells_20240320_27064_scan6.ang"
    # ang_data = utilities.read_ang(path, patshape=(1024, 1024))
    # quaternions = ang_data.quats
    # threshold = 3.0
    # grain_ids, kam = grain_segmentation(quaternions, threshold)
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # ax[0].imshow(grain_ids, cmap="terrain")
    # im = ax[1].imshow(kam, cmap="jet")
    # l = ax[1].get_position()
    # cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    # plt.colorbar(im, cax=cax)
    # plt.show()
    h5 = h5py.File("D:/Research/CoNi_16/Data/3D/new/CoNi16_aligned_corrected.dream3d", "r")
    quats = np.roll(h5["DataContainers/ImageDataContainer/CellData/Quats"][...], 1, axis=-1)
    mask =h5["DataContainers/ImageDataContainer/CellData/Mask"][..., 0]
    h5.close()
    threshold = 2.0
    ids, kam = segment_grains_3d(quats, threshold, mask)
    np.save("D:/Research/CoNi_16/Data/3D/new/CoNi16_aligned_corrected_grain_ids.npy", ids)
    np.save("D:/Research/CoNi_16/Data/3D/new/CoNi16_aligned_corrected_kam.npy", kam)