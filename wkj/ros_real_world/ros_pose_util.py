import numpy as np


def rotvec_to_matrix(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle == 0:
        return np.eye(3)

    axis = rotvec / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


def matrix_to_rotvec(matrix):
    angle = np.arccos((np.trace(matrix) - 1) / 2)
    if angle == 0:
        return np.zeros(3)

    rx = matrix[2, 1] - matrix[1, 2]
    ry = matrix[0, 2] - matrix[2, 0]
    rz = matrix[1, 0] - matrix[0, 1]
    r = np.array([rx, ry, rz])
    return (angle / (2 * np.sin(angle))) * r


def pos_rot_to_mat(pos, rotvec):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rotvec_to_matrix(rotvec)
    mat[..., 3, 3] = 1
    return mat


def mat_to_pos_rot(mat):
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot_matrix = mat[..., :3, :3]
    rotvec = matrix_to_rotvec(rot_matrix)
    return pos, rotvec


def pos_rot_to_pose(pos, rotvec):
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rotvec
    return pose


def pose_to_pos_rot(pose):
    pos = pose[..., :3]
    rotvec = pose[..., 3:]
    return pos, rotvec


def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def main():

    # 示例用法
    pose = np.array([1, 2, 3, 0.1, 0.2, 0.3])
    mat = pose_to_mat(pose)
    pose_back = mat_to_pose(mat)

    print("Original pose:", pose)
    print("Converted matrix:", mat)
    print("Converted back to pose:", pose_back)


if __name__ == "__main__":
    main()
