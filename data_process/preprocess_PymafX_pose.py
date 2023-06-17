"""
Smoothen the output of PyMAF and save the results as a pickle file.
"""
import os
import os.path as osp
import numpy as np
import pickle
import joblib
import argparse

import trimesh
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
from smooth_pose_util import filter_abnormal_hand_pose, smoothen_poses, create_body_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, default='tennis')
    parser.add_argument('--gender', type=str, default='MALE', help="Doesn't matter actually.")
    parser.add_argument('--data_dir', type=str, default='/path/to/PymafX/data')
    parser.add_argument('--model_dir', type=str, default='../code/lib/smplx/smplx_model')
    parser.add_argument('--every_n_frames', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--debug', action='store_true', help='whether to save the meshes for debugging')
    args = parser.parse_args()
    
    # load SMPL-X model
    model_path = osp.join(args.model_dir, f"SMPLX_{args.gender}.npz")
    body_model = create_body_model(model_path, args.gender, args.device)

    # load PyMAF output
    pymaf_output = joblib.load(f'{args.data_dir}/{args.seq_name}.pkl')
    motion_files = pymaf_output['smplx_params'][::args.every_n_frames]
    
    save_path = f'{args.data_dir}/{args.seq_name}_smoothed.pkl'

    smplx_body_pose = []
    smplx_left_hand_pose = []
    smplx_right_hand_pose = []
    smplx_jaw_pose = []
    smplx_expression = []

    # Convert the format of poses from Matrix to Axis-Angle
    print('Converting the format of poses from Matrix to Axis-Angle...')
    for f in motion_files:
        batch_num = f['body_pose'].shape[0]

        body_pose = quaternion_to_axis_angle(matrix_to_quaternion(f['body_pose'])).reshape(batch_num, -1)
        left_hand_pose = quaternion_to_axis_angle(matrix_to_quaternion(f['left_hand_pose'])).reshape(batch_num, -1)
        right_hand_pose = quaternion_to_axis_angle(matrix_to_quaternion(f['right_hand_pose'])).reshape(batch_num, -1)
        jaw_pose = quaternion_to_axis_angle(matrix_to_quaternion(f['jaw_pose'])).reshape(batch_num, -1)        
        expression = f['expression']

        smplx_body_pose.append(body_pose)
        smplx_left_hand_pose.append(left_hand_pose)
        smplx_right_hand_pose.append(right_hand_pose)
        smplx_jaw_pose.append(jaw_pose)
        smplx_expression.append(expression)

    smplx_body_pose_all = torch.cat([result for result in smplx_body_pose], dim=0).cpu().numpy()
    smplx_left_hand_pose_all = torch.cat([result for result in smplx_left_hand_pose], dim=0).cpu().numpy()
    smplx_right_hand_pose_all = torch.cat([result for result in smplx_right_hand_pose], dim=0).cpu().numpy()
    smplx_jaw_pose_all = torch.cat([result for result in smplx_jaw_pose], dim=0).cpu().numpy()
    smplx_expression_all = torch.cat([result for result in smplx_expression], dim=0).cpu().numpy()

    # Filter out abnormal hand poses
    print('Filtering out abnormal hand poses...')
    smplx_left_hand_pose_all, smplx_right_hand_pose_all = filter_abnormal_hand_pose(model_path=model_path,
                                                                               smplx_left_hand_pose_all=smplx_left_hand_pose_all,
                                                                               smplx_right_hand_pose_all=smplx_right_hand_pose_all)

    body_pose_dim = smplx_body_pose_all.shape[1]
    left_hand_pose_dim = smplx_left_hand_pose_all.shape[1]
    right_hand_pose_dim = smplx_right_hand_pose_all.shape[1]
    
    smplx_pose_all = np.hstack([smplx_body_pose_all, smplx_left_hand_pose_all, smplx_right_hand_pose_all, smplx_jaw_pose_all])

    # Smoothen poses
    print('Smoothening poses...')
    smplx_pose_all_smoothed = smoothen_poses(smplx_pose_all)

    smplx_body_pose_all_smoothed = smplx_pose_all_smoothed[:, :body_pose_dim]
    smplx_left_hand_pose_all_smoothed = smplx_pose_all_smoothed[:, body_pose_dim:body_pose_dim+left_hand_pose_dim]
    smplx_right_hand_pose_all_smoothed = smplx_pose_all_smoothed[:, body_pose_dim+left_hand_pose_dim:body_pose_dim+left_hand_pose_dim+right_hand_pose_dim]
    smplx_jaw_pose_all_smoothed = smplx_pose_all_smoothed[:, body_pose_dim+left_hand_pose_dim+right_hand_pose_dim:]

    smplx_smoothed_output = {'body_pose': smplx_body_pose_all_smoothed,
                             'left_hand_pose': smplx_left_hand_pose_all_smoothed,
                             'right_hand_pose': smplx_right_hand_pose_all_smoothed,
                             'jaw_pose': smplx_jaw_pose_all_smoothed,
                             'expression': smplx_expression_all}
    
    # Save the smoothed poses
    pickle.dump(smplx_smoothed_output, open(save_path, 'wb'))

    if args.debug:
        mesh_save_dir = f'{args.data_dir}/{args.seq_name}'
        os.makedirs(mesh_save_dir, exist_ok=True)

        for idx in range(smplx_body_pose_all_smoothed.shape[0]):
            model_output = body_model(
                return_verts=True,
                global_orient=torch.from_numpy(smplx_body_pose_all_smoothed[idx, :3]).float().to(
                    args.device).unsqueeze(0),
                transl=None,
                body_pose=torch.from_numpy(smplx_body_pose_all_smoothed[idx, 3:66]).float().to(
                    args.device).unsqueeze(0),
                betas=None,
                expression=torch.from_numpy(smplx_expression_all[idx]).float().to(
                    args.device).unsqueeze(0),
                left_hand_pose=torch.from_numpy(smplx_left_hand_pose_all_smoothed[idx]).float().to(
                        args.device).unsqueeze(0),
                right_hand_pose=torch.from_numpy(smplx_right_hand_pose_all_smoothed[idx]).float().to(
                                args.device).unsqueeze(0),
                jaw_pose=torch.from_numpy(smplx_jaw_pose_all_smoothed[idx]).float().to(
                    args.device).unsqueeze(0),
                leye_pose=None,
                reye_pose=None,
                return_full_pose=False)
            
            vertices = model_output.vertices.detach().cpu()
            smplx_mesh = trimesh.Trimesh(vertices.numpy()[0], body_model.faces, process=False)
            _ = smplx_mesh.export(osp.join(mesh_save_dir, f'{idx:04d}.ply'))
        
        
if __name__ == '__main__':
    main()
    