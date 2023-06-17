import torch
import numpy as np
import smplx
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def create_body_model(model_path, gender, device):
    model_params = dict(model_path=model_path,
                        create_global_orient=False,
                        create_transl=False,
                        create_body_pose=False,
                        create_betas=True,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        flat_hand_mean=True,
                        create_expression=False,
                        create_jaw_pose=False,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        use_face_contour=True,
                        dtype=torch.float32,
                        batch_size=1,
                        gender=gender,
                        use_pca=False,
                        model_type='smplx')
    body_model = smplx.create(**model_params).to(device=device)
    return body_model

def truncate_pose(hand_pose):
    # (-20,20)
    tmp = hand_pose[[0, 1, 3, 6, 9, 10, 12, 15, 19, 27, 28, 30, 33]]
    tmp[tmp > np.pi / 9] = np.pi / 9
    tmp[tmp < -np.pi / 9] = -np.pi / 9
    hand_pose[[0, 1, 3, 6, 9, 10, 12, 15, 19, 27, 28, 30, 33]] = tmp
    # (-10,10)
    tmp = hand_pose[[4, 7, 13, 16, 22, 25, 31, 34]]
    tmp[tmp > np.pi / 18] = np.pi / 18
    tmp[tmp < -np.pi / 18] = -np.pi / 18
    hand_pose[[4, 7, 13, 16, 22, 25, 31, 34]] = tmp
    # (-120,20)
    tmp = hand_pose[list(range(2, 36, 3))]
    tmp[tmp > np.pi / 9] = np.pi / 9
    tmp[tmp < -np.pi / 3 * 2] = -np.pi / 3 * 2
    hand_pose[list(range(2, 36, 3))] = tmp
    # (-60,180)
    tmp = hand_pose[[36, 40]]
    tmp[tmp > np.pi] = np.pi
    tmp[tmp < -np.pi / 3] = -np.pi / 3
    hand_pose[[36, 40]] = tmp
    #  (-60,120)
    tmp = hand_pose[[37, 39, 42]]
    tmp[tmp > np.pi / 3 * 2] = np.pi / 3 * 2
    tmp[tmp < -np.pi / 3] = -np.pi / 3
    hand_pose[[37, 39, 42]] = tmp
    # (-90,60)
    tmp = hand_pose[[38, 41, 44]]
    tmp[tmp > np.pi / 3] = np.pi / 3
    tmp[tmp < -np.pi / 2] = -np.pi / 2
    hand_pose[[38, 41, 44]] = tmp
    # (-60,20)
    tmp = hand_pose[[18, 21, 24]]
    tmp[tmp > np.pi / 9] = np.pi / 9
    tmp[tmp < -np.pi / 3] = -np.pi / 3
    hand_pose[[18, 21, 24]] = tmp
    # (-20,90)
    tmp = hand_pose[[43]]
    tmp[tmp > np.pi / 2] = np.pi / 2
    tmp[tmp < -np.pi / 9] = -np.pi / 9
    hand_pose[[43]] = tmp
    return hand_pose

def filter_abnormal_hand_pose(model_path=None,
                         smplx_left_hand_pose_all=None,
                         smplx_right_hand_pose_all=None):
    smplx_model = np.load(model_path)
    lhand_pose_mean = smplx_model['hands_meanl']
    rhand_pose_mean = smplx_model['hands_meanr']
    
    left_hand_pose_output = []
    right_hand_pose_output = []

    for idx in range(smplx_left_hand_pose_all.shape[0]):
        left_hand_pose = smplx_left_hand_pose_all[idx] + lhand_pose_mean.copy()
        right_hand_pose = smplx_right_hand_pose_all[idx] + rhand_pose_mean.copy()
        left_hand_pose = truncate_pose(left_hand_pose)
        right_hand_pose[list(range(
            0, 45, 3))] = -right_hand_pose[list(range(0, 45, 3))]
        right_hand_pose = -truncate_pose(-right_hand_pose)
        right_hand_pose[list(range(
            0, 45, 3))] = -right_hand_pose[list(range(0, 45, 3))]

        left_hand_pose = left_hand_pose - lhand_pose_mean.copy()
        right_hand_pose = right_hand_pose - rhand_pose_mean.copy()
        
        left_hand_pose_output.append(left_hand_pose)
        right_hand_pose_output.append(right_hand_pose)

    return np.array(left_hand_pose_output), np.array(right_hand_pose_output)

def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4

    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result

def smoothen_poses(poses, global_root_orient=None, window_length=11):
    """Smooth joint angles. Poses and global_root_orient should be given as rotation vectors."""
    n_joints = poses.shape[1] // 3

    # Convert poses to quaternions.
    qs = R.as_quat(R.from_rotvec(np.reshape(poses, (-1, 3))))
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)

    # Smooth the quaternions.
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)

    ps_clean = R.as_rotvec(R.from_quat(np.reshape(qs_clean, (-1, 4))))
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])

    if global_root_orient is not None:
        root_qs = R.as_quat(R.from_rotvec(global_root_orient))
        root_smooth = savgol_filter(root_qs, window_length=window_length, polyorder=2, axis=0)
        root_clean = root_smooth / np.linalg.norm(root_smooth)
        root_clean = R.as_rotvec(R.from_quat(root_clean))

    return ps_clean