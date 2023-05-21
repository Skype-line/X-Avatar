import numpy as np
import glob
import os
import os.path as osp
import trimesh
from tqdm import tqdm
import torch
import kaolin
import argparse
import pickle as pkl


def compute_iou_w_mesh(mesh, gt_mesh):
    mesh_bounds = mesh.bounds
    gt_mesh_bounds = gt_mesh.bounds
    xx1 = np.max([mesh_bounds[0, 0], gt_mesh_bounds[0, 0]])
    yy1 = np.max([mesh_bounds[0, 1], gt_mesh_bounds[0, 1]])
    zz1 = np.max([mesh_bounds[0, 2], gt_mesh_bounds[0, 2]])

    xx2 = np.min([mesh_bounds[1, 0], gt_mesh_bounds[1, 0]])
    yy2 = np.min([mesh_bounds[1, 1], gt_mesh_bounds[1, 1]])
    zz2 = np.min([mesh_bounds[1, 2], gt_mesh_bounds[1, 2]])

    vol1 = (mesh_bounds[1, 0] - mesh_bounds[0, 0]) * (
        mesh_bounds[1, 1] - mesh_bounds[0, 1]) * (mesh_bounds[1, 2] -
                                                  mesh_bounds[0, 2])
    vol2 = (gt_mesh_bounds[1, 0] - gt_mesh_bounds[0, 0]) * (
        gt_mesh_bounds[1, 1] - gt_mesh_bounds[0, 1]) * (gt_mesh_bounds[1, 2] -
                                                        gt_mesh_bounds[0, 2])
    inter_vol = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1]) * np.max(
        [0, zz2 - zz1])

    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-11)
    return iou


def compute_iou_w_points(points, gt_points):
    mesh_bounds = np.concatenate([
        np.min(points, axis=0, keepdims=True),
        np.max(points, axis=0, keepdims=True)
    ],
                                 axis=0)
    gt_mesh_bounds = np.concatenate([
        np.min(gt_points, axis=0, keepdims=True),
        np.max(gt_points, axis=0, keepdims=True)
    ],
                                    axis=0)
    xx1 = np.max([mesh_bounds[0, 0], gt_mesh_bounds[0, 0]])
    yy1 = np.max([mesh_bounds[0, 1], gt_mesh_bounds[0, 1]])
    zz1 = np.max([mesh_bounds[0, 2], gt_mesh_bounds[0, 2]])

    xx2 = np.min([mesh_bounds[1, 0], gt_mesh_bounds[1, 0]])
    yy2 = np.min([mesh_bounds[1, 1], gt_mesh_bounds[1, 1]])
    zz2 = np.min([mesh_bounds[1, 2], gt_mesh_bounds[1, 2]])

    vol1 = (mesh_bounds[1, 0] - mesh_bounds[0, 0]) * (
        mesh_bounds[1, 1] - mesh_bounds[0, 1]) * (mesh_bounds[1, 2] -
                                                  mesh_bounds[0, 2])
    vol2 = (gt_mesh_bounds[1, 0] - gt_mesh_bounds[0, 0]) * (
        gt_mesh_bounds[1, 1] - gt_mesh_bounds[0, 1]) * (gt_mesh_bounds[1, 2] -
                                                        gt_mesh_bounds[0, 2])
    inter_vol = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1]) * np.max(
        [0, zz2 - zz1])

    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-11)
    return iou


def compute_normal_consistency(normals_src, mesh_tgt, src2tgt_idx):
    normals_src = normals_src / np.linalg.norm(
        normals_src, axis=-1, keepdims=True)
    normals_tgt = mesh_tgt.face_normals[src2tgt_idx]
    normals_tgt = normals_tgt / np.linalg.norm(
        normals_tgt, axis=-1, keepdims=True)

    src2tgt_normals_dot_product = (normals_tgt * normals_src).sum(axis=-1)
    src2tgt_normals_dot_product = np.abs(src2tgt_normals_dot_product)
    src2tgt_normals_dot_product[np.isnan(src2tgt_normals_dot_product)] = 1.

    return src2tgt_normals_dot_product


def compute_p2f_distance(points_src, mesh_tgt):
    _, src_tgt_dist, src2tgt_idx = trimesh.proximity.closest_point(
        mesh_tgt, points_src)
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    return src_tgt_dist, src2tgt_idx


def find_hand_face_points(points_src, normals_src, points_smplx,
                          smplx_label_vector):
    # find points belong to each hand and face in scan mesh with registered SMPLX mesh
    points_src = torch.tensor(points_src).cuda().unsqueeze(0)
    normals_src = torch.tensor(normals_src).cuda().unsqueeze(0)
    points_smplx = torch.tensor(points_smplx).cuda().unsqueeze(0)

    _, close_id_src = kaolin.metrics.pointcloud.sided_distance(
        points_src, points_smplx)
    src_label_vector = smplx_label_vector[close_id_src[0]]
    src_lhand_ids = torch.where(src_label_vector == 1)[0]
    src_rhand_ids = torch.where(src_label_vector == 2)[0]
    src_face_ids = torch.where(src_label_vector == 3)[0]
    
    lhand_points_src = torch.gather(points_src[0], 0,
                                    src_lhand_ids.unsqueeze(-1).expand(-1, 3))
    lhand_normals_src = torch.gather(normals_src[0], 0,
                                     src_lhand_ids.unsqueeze(-1).expand(-1, 3))
    rhand_points_src = torch.gather(points_src[0], 0,
                                    src_rhand_ids.unsqueeze(-1).expand(-1, 3))
    rhand_normals_src = torch.gather(normals_src[0], 0,
                                     src_rhand_ids.unsqueeze(-1).expand(-1, 3))
    face_points_src = torch.gather(points_src[0], 0,
                                   src_face_ids.unsqueeze(-1).expand(-1, 3))
    face_normals_src = torch.gather(normals_src[0], 0,
                                    src_face_ids.unsqueeze(-1).expand(-1, 3))
    
    lhand_points_src = lhand_points_src.detach().cpu().numpy()
    lhand_normals_src = lhand_normals_src.detach().cpu().numpy()
    rhand_points_src = rhand_points_src.detach().cpu().numpy()
    rhand_normals_src = rhand_normals_src.detach().cpu().numpy()
    face_points_src = face_points_src.detach().cpu().numpy()
    face_normals_src = face_normals_src.detach().cpu().numpy()
    
    return lhand_points_src, rhand_points_src, lhand_normals_src, rhand_normals_src, \
        face_points_src, face_normals_src


def evaluate_per_frame(mesh_src,
                       mesh_tgt,
                       mesh_smplx,
                       lhand_ids,
                       rhand_ids,
                       face_ids,
                       num_samples=10000):
    # load source mesh, target mesh, and smplx mesh
    points_src, faces_src = trimesh.sample.sample_surface(
        mesh_src, num_samples)
    normals_src = mesh_src.face_normals[faces_src]
    points_tgt, faces_tgt = trimesh.sample.sample_surface(
        mesh_tgt, num_samples)
    normals_tgt = mesh_tgt.face_normals[faces_tgt]
    points_smplx = mesh_smplx.vertices

    # compute global chamfer distance (whole body)
    src_tgt_dist, src2tgt_idx = compute_p2f_distance(points_src, mesh_tgt)
    tgt_src_dist, tgt2src_idx = compute_p2f_distance(points_tgt, mesh_src)
    global_CD_mean = (src_tgt_dist.mean() + tgt_src_dist.mean()) / 2
    global_CD_max = (src_tgt_dist.max() + tgt_src_dist.max()) / 2

    # compute global normal consistency (whole body)
    src_tgt_NC = compute_normal_consistency(normals_src, mesh_tgt, src2tgt_idx)
    tgt_src_NC = compute_normal_consistency(normals_tgt, mesh_src, tgt2src_idx)
    global_NC_mean = (src_tgt_NC.mean() + tgt_src_NC.mean()) / 2

    # compute global IOU (whole body)
    global_iou = compute_iou_w_mesh(mesh_src, mesh_tgt)

    # load SMPLX part label vector
    smplx_label_vector = torch.zeros(points_smplx.shape[0],
                                     dtype=torch.int32).cuda()
    smplx_label_vector[lhand_ids] = 1
    smplx_label_vector[rhand_ids] = 2
    smplx_label_vector[face_ids] = 3

    # find hand points in source mesh with registered SMPLX mesh
    lhand_points_src, rhand_points_src, lhand_normals_src, rhand_normals_src, face_points_src, face_normals_src = \
        find_hand_face_points(points_src, normals_src, points_smplx, smplx_label_vector)

    # find hand points in target mesh with registered SMPLX mesh
    lhand_points_tgt, rhand_points_tgt, lhand_normals_tgt, rhand_normals_tgt, face_points_tgt, face_normals_tgt = \
        find_hand_face_points(points_tgt, normals_tgt, points_smplx, smplx_label_vector)

    # concatenate left and right hand points
    hand_points_src = np.concatenate([lhand_points_src, rhand_points_src],
                                     axis=0)
    hand_normals_src = np.concatenate([lhand_normals_src, rhand_normals_src],
                                      axis=0)
    hand_points_tgt = np.concatenate([lhand_points_tgt, rhand_points_tgt],
                                     axis=0)
    hand_normals_tgt = np.concatenate([lhand_normals_tgt, rhand_normals_tgt],
                                      axis=0)

    # compute local chamfer distance (hands)
    hand_src_tgt_dist, hand_src2tgt_idx = compute_p2f_distance(
        hand_points_src, mesh_tgt)
    hand_tgt_src_dist, hand_tgt2src_idx = compute_p2f_distance(
        hand_points_tgt, mesh_src)
    local_hand_CD_mean = (hand_src_tgt_dist.mean() +
                          hand_tgt_src_dist.mean()) / 2
    local_hand_CD_max = (hand_src_tgt_dist.max() + hand_tgt_src_dist.max()) / 2

    # compute local chamfer distance (face)
    face_src_tgt_dist, face_src2tgt_idx = compute_p2f_distance(
        face_points_src, mesh_tgt)
    face_tgt_src_dist, face_tgt2src_idx = compute_p2f_distance(
        face_points_tgt, mesh_src)
    local_face_CD_mean = (face_src_tgt_dist.mean() +
                          face_tgt_src_dist.mean()) / 2
    local_face_CD_max = (face_src_tgt_dist.max() + face_tgt_src_dist.max()) / 2

    # compute local normal consistency (hands)
    hand_src_tgt_NC = compute_normal_consistency(hand_normals_src, mesh_tgt,
                                                 hand_src2tgt_idx)
    hand_tgt_src_NC = compute_normal_consistency(hand_normals_tgt, mesh_src,
                                                 hand_tgt2src_idx)
    local_hand_NC_mean = (hand_src_tgt_NC.mean() + hand_tgt_src_NC.mean()) / 2

    # compute local normal consistency (face)
    face_src_tgt_NC = compute_normal_consistency(face_normals_src, mesh_tgt,
                                                 face_src2tgt_idx)
    face_tgt_src_NC = compute_normal_consistency(face_normals_tgt, mesh_src,
                                                 face_tgt2src_idx)
    local_face_NC_mean = (face_src_tgt_NC.mean() + face_tgt_src_NC.mean()) / 2

    # compute local IOU (hands)
    lhand_iou = compute_iou_w_points(lhand_points_src, lhand_points_tgt)
    rhand_iou = compute_iou_w_points(rhand_points_src, rhand_points_tgt)
    local_hand_iou = (lhand_iou + rhand_iou) / 2

    # compute local IOU (face)
    local_face_iou = compute_iou_w_points(face_points_src, face_points_tgt)

    return global_CD_mean, global_CD_max, global_NC_mean, global_iou, \
        local_hand_CD_mean, local_hand_CD_max, local_hand_NC_mean, local_hand_iou, \
        local_face_CD_mean, local_face_CD_max, local_face_NC_mean, local_face_iou


def evaluate(pd_dir, gt_dir, gender=None, mode='test'):
    verts_ids = pkl.load(open(
        f'./lib/smplx/smplx_model/non_watertight_{gender}_vertex_labels.pkl', 'rb'), encoding='latin1')
    lhand_ids = torch.tensor(verts_ids['left_hand']).cuda()
    rhand_ids = torch.tensor(verts_ids['right_hand']).cuda()
    face_ids = torch.tensor(verts_ids['face']).cuda()

    pd_data_list = sorted(glob.glob(os.path.join(pd_dir, f'meshes_{mode}', '*.ply')))
    gt_data_list = sorted(glob.glob(os.path.join(gt_dir, mode, '*', 'meshes_ply', '*.ply')))
    smplx_data_list = sorted(glob.glob(os.path.join(gt_dir, mode, '*', 'SMPLX', '*.ply')))
    assert len(pd_data_list) == len(gt_data_list) == len(smplx_data_list)

    print('Evaluating {} meshes'.format(len(pd_data_list)))
    global_CD_mean_list, global_CD_max_list, global_NC_mean_list, global_iou_list, \
        local_hand_CD_mean_list, local_hand_CD_max_list, local_hand_NC_mean_list, local_hand_iou_list, \
        local_face_CD_mean_list, local_face_CD_max_list, local_face_NC_mean_list, local_face_iou_list \
            = [], [], [], [], [], [], [], [], [], [], [], []

    for pd_data, gt_data, smplx_data in tqdm(
            zip(pd_data_list, gt_data_list, smplx_data_list)):
        pd_mesh = trimesh.load(pd_data)
        gt_mesh = trimesh.load(gt_data)
        smplx_mesh = trimesh.load(smplx_data)

        global_CD_mean, global_CD_max, global_NC_mean, global_iou, \
            local_hand_CD_mean, local_hand_CD_max, local_hand_NC_mean, local_hand_iou, \
            local_face_CD_mean, local_face_CD_max, local_face_NC_mean, local_face_iou \
            = evaluate_per_frame(pd_mesh, gt_mesh, smplx_mesh, lhand_ids, rhand_ids, face_ids)

        global_CD_mean_list.append(global_CD_mean)
        global_CD_max_list.append(global_CD_max)
        global_NC_mean_list.append(global_NC_mean)
        global_iou_list.append(global_iou)
        local_hand_CD_mean_list.append(local_hand_CD_mean)
        local_hand_CD_max_list.append(local_hand_CD_max)
        local_hand_NC_mean_list.append(local_hand_NC_mean)
        local_hand_iou_list.append(local_hand_iou)
        local_face_CD_mean_list.append(local_face_CD_mean)
        local_face_CD_max_list.append(local_face_CD_max)
        local_face_NC_mean_list.append(local_face_NC_mean)
        local_face_iou_list.append(local_face_iou)

    global_CD_mean_array = np.array(global_CD_mean_list)
    global_CD_max_array = np.array(global_CD_max_list)
    global_NC_mean_array = np.array(global_NC_mean_list)
    global_iou_array = np.array(global_iou_list)
    local_hand_CD_mean_array = np.array(local_hand_CD_mean_list)
    local_hand_CD_max_array = np.array(local_hand_CD_max_list)
    local_hand_NC_mean_array = np.array(local_hand_NC_mean_list)
    local_hand_iou_array = np.array(local_hand_iou_list)
    local_face_CD_mean_array = np.array(local_face_CD_mean_list)
    local_face_CD_max_array = np.array(local_face_CD_max_list)
    local_face_NC_mean_array = np.array(local_face_NC_mean_list)
    local_face_iou_array = np.array(local_face_iou_list)

    np.savez(os.path.join(pd_dir, f'{mode}_metrics.npz'),
                global_CD_mean=global_CD_mean_array,
                global_CD_max=global_CD_max_array,
                global_NC_mean=global_NC_mean_array,
                global_iou=global_iou_array,
                local_hand_CD_mean=local_hand_CD_mean_array,
                local_hand_CD_max=local_hand_CD_max_array,
                local_hand_NC_mean=local_hand_NC_mean_array,
                local_hand_iou=local_hand_iou_array,
                local_face_CD_mean=local_face_CD_mean_array,
                local_face_CD_max=local_face_CD_max_array,
                local_face_NC_mean=local_face_NC_mean_array,
                local_face_iou=local_face_iou_array)

    print('global_CD_mean: {:.2f}, global_CD_max: {:.2f}, global_NC_mean: {:.3f}, global_iou: {:.3f}, \n \
        local_hand_CD_mean: {:.2f}, local_hand_CD_max: {:.2f}, local_hand_NC_mean: {:.3f}, local_hand_iou: {:.3f}, \n \
        local_face_CD_mean: {:.2f}, local_face_CD_max: {:.2f}, local_face_NC_mean: {:.3f}, local_face_iou: {:.3f}'.format(\
            global_CD_mean_array.mean()*1e3, global_CD_max_array.mean()*1e3, global_NC_mean_array.mean(), global_iou_array.mean(), \
            local_hand_CD_mean_array.mean()*1e3, local_hand_CD_max_array.mean()*1e3, local_hand_NC_mean_array.mean(), local_hand_iou_array.mean(), \
            local_face_CD_mean_array.mean()*1e3, local_face_CD_max_array.mean()*1e3, local_face_NC_mean_array.mean(), local_face_iou_array.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pd_dir',
        type=str,
        default=
        '../outputs/XHumans_smplx/Person_ID_scan'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=
        '/path/to/XHumans/Dataset/Person_ID'
    )
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    gender = open(osp.join(args.gt_dir, 'gender.txt')).readlines()[0].strip()

    if len(os.listdir(args.pd_dir)) != 0:
        evaluate(args.pd_dir, args.gt_dir, gender=gender, mode=args.mode)
    else:
        print('Predicted mesh directory does not exist!')
    print('-' * 20)


if __name__ == '__main__':
    main()