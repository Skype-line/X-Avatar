"""
Convert GRAB dataset to SMPL[-H/X] meshes and SMPL[-H/X] params
"""
import sys
import argparse
import smplx
import numpy as np
import torch
import glob
import os.path as osp
import pymesh
import os
from tqdm import tqdm
import cv2
import trimesh
from pytorch3d.structures import Meshes
import pickle

num_betas = 10
num_exp = 10

def generate_SMPLX(data_root: str,
                   model_root: str,
                   gender: str,
                   dtype=torch.float32,
                   device='cuda:0'):
    """ generate SMPL-X meshes with downloaded SMPL-X params """
    model_params = dict(model_path=model_root,
                        create_global_orient=False,
                        create_transl=False,
                        create_body_pose=False,
                        create_betas=False,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        flat_hand_mean=False,
                        create_expression=False,
                        create_jaw_pose=False,
                        create_leye_pose=False,
                        create_reye_pose=False,
                        num_betas=num_betas,
                        num_expression_coeffs=num_exp,
                        dtype=dtype,
                        batch_size=1,
                        gender=gender,
                        use_pca=False,
                        model_type='smplx')
    body_model = smplx.create(**model_params).to(device=device)

    watertight_face = np.load('../code/lib/smplx/smplx_model/watertight_smplx_faces.npy')
    from pymeshfix import _meshfix

    data_name_list = sorted(os.listdir(data_root))
    for data_name in data_name_list:
        if not data_name.endswith('.npz'):
            continue
        save_dir = os.path.join(data_root, data_name.split('.')[0], 'SMPLX')
        os.makedirs(save_dir, exist_ok=True)
        data = np.load(os.path.join(data_root, data_name))

        frame_N = data['trans'].shape[0]
        for i in tqdm(range(frame_N)):
            params_data = {}
            global_orient = data['root_orient'][i]
            R_mod = cv2.Rodrigues(np.array([-np.pi / 2, 0, 0]))[0]
            R_root = cv2.Rodrigues(global_orient)[0]
            new_root = R_mod.dot(R_root)
            global_orient = cv2.Rodrigues(new_root)[0].reshape(3)
            
            params_data['global_orient'] = global_orient.astype(np.float32)
            params_data['transl'] = data['trans'][i].astype(np.float32)
            params_data['betas'] = data['betas'][:num_betas].astype(np.float32)
            params_data['body_pose'] = data['pose_body'][i].astype(np.float32)
            params_data['left_hand_pose'] = (data['pose_hand'][i][:45] -
                body_model.left_hand_mean.detach().cpu().numpy()).astype(np.float32)
            params_data['right_hand_pose'] = (data['pose_hand'][i][45:] -
                body_model.right_hand_mean.detach().cpu().numpy()).astype(np.float32)
            params_data['jaw_pose'] = data['pose_jaw'][i].astype(np.float32)
            params_data['leye_pose'] = data['pose_eye'][i][:3].astype(np.float32)
            params_data['reye_pose'] = data['pose_eye'][i][3:].astype(np.float32)
            params_data['expression'] = data['expression'][i][:num_exp].astype(np.float32)

            model_output = body_model(
                return_verts=True,
                global_orient=torch.from_numpy(
                    params_data['global_orient']).to(device).unsqueeze(0),
                transl=torch.from_numpy(
                    params_data['transl']).to(device).unsqueeze(0),
                body_pose=torch.from_numpy(
                    params_data['body_pose']).to(device).unsqueeze(0),
                betas=torch.from_numpy(
                    params_data['betas']).to(device).unsqueeze(0),
                expression=torch.from_numpy(
                    params_data['expression']).to(device).unsqueeze(0),
                left_hand_pose=torch.from_numpy(
                    params_data['left_hand_pose']).to(device).unsqueeze(0),
                right_hand_pose=torch.from_numpy(
                    params_data['right_hand_pose']).to(device).unsqueeze(0),
                jaw_pose=torch.from_numpy(
                    params_data['jaw_pose']).to(device).unsqueeze(0),
                leye_pose=torch.from_numpy(
                    params_data['leye_pose']).to(device).unsqueeze(0),
                reye_pose=torch.from_numpy(
                    params_data['reye_pose']).to(device).unsqueeze(0),
                return_full_pose=False)

            vertices = model_output.vertices.detach().cpu()

            # convert SMPL-X mesh from non-watertight to watertight
            tin = _meshfix.PyTMesh()
            tin.load_array(v=vertices.numpy()[0], f=watertight_face)
            _ = tin.remove_smallest_components()
            
            tin.save_file('{}/mesh-f{}_smplx.ply'.format(save_dir, str(i).zfill(6)))
            
            with open('{}/mesh-f{}_smplx.pkl'.format(save_dir, str(i).zfill(6)), 'wb') as f:
                pickle.dump(params_data, f)

def compute_average_shape(data_root: str):
    """ 
    compute the average shape of all SMPL-H meshes transfered from SMPL-X with 
    code: https://github.com/vchoutas/smplx/tree/master/transfer_model 
    SMPL-H model: https://mano.is.tue.mpg.de/, beta components = 16, but we only use 10
    """
    betas_list = list()
    params_file_list = sorted(glob.glob(os.path.join(data_root, 'train', '*', 'SMPLH', 'smplh.pkl')))

    for params_file in tqdm(params_file_list):
        with open(params_file, 'rb') as f:
            params_data = pickle.load(f, encoding='latin')
            betas = params_data['betas'].squeeze(0).detach().cpu().numpy()
            betas_list.append(betas)
    mean_shape = np.average(np.concatenate(betas_list), axis=0)
    np.save(osp.join(data_root, 'mean_shape_smpl.npy'), mean_shape)

def convert_transfered_SMPLH2SMPL(data_root: str,
                                  model_root: str,
                                  gender: str,
                                  dtype=torch.float32,
                                  device='cuda:0'):
    """ 
    convert transfered SMPL-H meshes further to SMPL meshes 
    SMPL-H params here are generated with original model transfer code here:
    https://github.com/vchoutas/smplx/tree/main/transfer_model
    """
    model_params = dict(model_path=model_root,
                        create_global_orient=False,
                        create_transl=False,
                        create_betas=False,
                        num_betas=num_betas,
                        dtype=dtype,
                        batch_size=1,
                        gender=gender,
                        model_type='smpl')
    body_model = smplx.create(**model_params).to(device=device)

    betas = np.load(osp.join(data_root, 'mean_shape_smpl.npy'))

    sub_folder_list = sorted(os.listdir(data_root))
    for sub_folder in sub_folder_list:
        if not osp.isdir(osp.join(data_root, sub_folder)):
            continue
        save_folder = osp.join(data_root, sub_folder, 'SMPL')
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        
        params_file_list = sorted(
            glob.glob(os.path.join(data_root, sub_folder, 'SMPLH', '*smplh.pkl')))
        
        for params_file in tqdm(params_file_list):
            with open(params_file, 'rb') as f:
                params_data = pickle.load(f, encoding='latin1')
                new_params_data = {}
                new_params_data['global_orient'] = params_data[
                    'global_orient'].squeeze().detach().cpu().numpy()
                new_params_data['transl'] = params_data['transl'].squeeze(
                ).detach().cpu().numpy()
                new_params_data['betas'] = betas.astype(np.float32)
                new_params_data['body_pose'] = np.concatenate([
                    torch.flatten(params_data['body_pose']).detach().cpu().numpy(),
                    np.zeros(6, dtype=np.float32)])

                model_output = body_model(
                    return_verts=True,
                    global_orient=torch.from_numpy(
                        new_params_data['global_orient']).to(device).unsqueeze(0),
                    transl=torch.from_numpy(
                        new_params_data['transl']).to(device).unsqueeze(0),
                    body_pose=torch.from_numpy(
                        new_params_data['body_pose']).to(device).unsqueeze(0),
                    betas=torch.from_numpy(
                        new_params_data['betas']).to(device).unsqueeze(0),
                    return_full_pose=False)

                vertices = model_output.vertices.detach().cpu()

            with open(os.path.join(save_folder, params_file.split('/')[-1].replace(
                            'smplh.pkl', 'smpl.pkl')), 'wb') as f:
                pickle.dump(new_params_data, f)

            th_smpl_meshes = Meshes(
                verts=vertices,
                faces=torch.tensor(np.array([body_model.faces] * len(vertices)).astype(np.int32),
                                   dtype=torch.int32))

            mesh = pymesh.meshio.form_mesh(
                th_smpl_meshes.verts_list()[0].cpu(),
                th_smpl_meshes.faces_list()[0].cpu())
            pymesh.save_mesh(os.path.join(save_folder,
                params_file.split('/')[-1].replace('smplh.pkl', 'smpl.ply')), mesh)

def generate_transfered_SMPLH(data_root: str,
                              model_root: str,
                              gender: str,
                              dtype=torch.float32,
                              device='cuda:0'):
    """ 
    generate SMPL-H meshes with transfered SMPL-H params 
    SMPL-H params here are generated with original model transfer code here:
    https://github.com/vchoutas/smplx/tree/main/transfer_model
    """
    model_params = dict(model_path=model_root,
                        create_global_orient=False,
                        create_transl=False,
                        create_betas=False,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        num_betas=num_betas,
                        num_expression_coeffs=num_exp,
                        dtype=dtype,
                        batch_size=1,
                        use_pca=False,
                        gender=gender,
                        flat_hand_mean=False,
                        model_type='smplh')
    body_model = smplx.create(**model_params).to(device=device)

    betas = np.load(osp.join(data_root, 'mean_shape_smpl.npy'))
    
    sub_folder_list = sorted(os.listdir(data_root))
    
    for sub_folder in sub_folder_list:
        if not osp.isdir(osp.join(data_root, sub_folder)):
            continue
        params_file_list = sorted(
            glob.glob(os.path.join(data_root, sub_folder, 'SMPLH', '*smplh.pkl')))
        for params_file in tqdm(params_file_list):
            with open(params_file, 'rb') as f:
                params_data = pickle.load(f, encoding='latin1')
                new_params_data = {}
                new_params_data['global_orient'] = params_data['global_orient'].squeeze().detach().cpu().numpy()
                new_params_data['transl'] = params_data['transl'].squeeze().detach().cpu().numpy()
                new_params_data['betas'] = betas.astype(np.float32)
                new_params_data['body_pose'] = torch.flatten(
                    params_data['body_pose']).detach().cpu().numpy()
                new_params_data['left_hand_pose'] = (
                    torch.flatten(params_data['left_hand_pose']) -
                    body_model.left_hand_mean).detach().cpu().numpy()
                new_params_data['right_hand_pose'] = (
                    torch.flatten(params_data['right_hand_pose']) -
                    body_model.right_hand_mean).detach().cpu().numpy()

            with open(params_file, 'wb') as f:
                pickle.dump(new_params_data, f)

def convert_transfered_SMPLH2SMPL_in_batch(data_root: str,
                                           model_root: str,
                                           gender: str,
                                           dtype=torch.float32,
                                           device='cuda:0'):
    """ 
    convert transfered SMPL-H meshes further to SMPL meshes 
    SMPL-H params here are generated with modified batched-version of model transfer code here:
    https://github.com/vchoutas/smplx/tree/main/transfer_model
    """
    model_params = dict(model_path=model_root,
                        create_global_orient=False,
                        create_transl=False,
                        create_betas=False,
                        num_betas=num_betas,
                        dtype=dtype,
                        batch_size=1,
                        gender=gender,
                        model_type='smpl')
    body_model = smplx.create(**model_params).to(device=device)

    betas = np.load(osp.join(osp.split(data_root)[0], 'mean_shape_smpl.npy'))

    sub_folder_list = sorted(os.listdir(data_root))
    for sub_folder in sub_folder_list:
        if not osp.isdir(osp.join(data_root, sub_folder)):
            continue
        print('Processing {}'.format(sub_folder))
        params_file = os.path.join(data_root, sub_folder, 'SMPLH', 'smplh.pkl')
        save_folder = osp.join(data_root, sub_folder, 'SMPL')
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        
        smplx_param_list = sorted(
            glob.glob(os.path.join(data_root, sub_folder, 'SMPLX', '*smplx.pkl')))
        
        with open(params_file, 'rb') as f:
            params_data = pickle.load(f, encoding='latin1')
            N = len(params_data['global_orient'])
            assert N == len(smplx_param_list)
            for i in tqdm(range(N)):
                new_params_data = {}
                new_params_data['global_orient'] = params_data[
                    'global_orient'][i].squeeze().detach().cpu().numpy()
                new_params_data['transl'] = params_data['transl'][i].detach(
                ).cpu().numpy()
                new_params_data['betas'] = betas.astype(np.float32)
                new_params_data['body_pose'] = np.concatenate([
                    torch.flatten(
                        params_data['body_pose'][i]).detach().cpu().numpy(),
                    np.zeros(6, dtype=np.float32)])

                model_output = body_model(
                    return_verts=True,
                    global_orient=torch.from_numpy(
                        new_params_data['global_orient']).to(device).unsqueeze(0),
                    transl=torch.from_numpy(
                        new_params_data['transl']).to(device).unsqueeze(0),
                    body_pose=torch.from_numpy(
                        new_params_data['body_pose']).to(device).unsqueeze(0),
                    betas=torch.from_numpy(
                        new_params_data['betas']).to(device).unsqueeze(0),
                    return_full_pose=False)

                vertices = model_output.vertices.detach().cpu().numpy()[0]

                with open(os.path.join(save_folder,
                            smplx_param_list[i].split('/')[-1].replace(
                                'smplx', 'smpl')), 'wb') as f:
                    pickle.dump(new_params_data, f)
                
                mesh = trimesh.Trimesh(vertices=vertices,
                                       faces=body_model.faces)
                _ = mesh.export(
                    os.path.join(save_folder,smplx_param_list[i].split('/')[-1].replace(
                            'smplx', 'smpl').replace('pkl', 'ply')))

def generate_transfered_SMPLH_in_batch(data_root: str,
                                       model_root: str,
                                       gender: str,
                                       dtype=torch.float32,
                                       device='cuda:0'):
    """ 
    generate SMPL-H meshes with transfered SMPL-H params,
    SMPL-H params here are generated with modified batched-version of model transfer code here:
    https://github.com/vchoutas/smplx/tree/main/transfer_model
    """
    model_params = dict(model_path=model_root,
                        create_global_orient=False,
                        create_transl=False,
                        create_betas=False,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        num_betas=num_betas,
                        num_expression_coeffs=num_exp,
                        dtype=dtype,
                        batch_size=1,
                        use_pca=False,
                        gender=gender,
                        flat_hand_mean=False,
                        model_type='smplh')
    body_model = smplx.create(**model_params).to(device=device)

    betas = np.load(osp.join(osp.split(data_root)[0], 'mean_shape_smpl.npy'))
    
    sub_folder_list = sorted(os.listdir(data_root))

    for sub_folder in sub_folder_list:
        if not osp.isdir(osp.join(data_root, sub_folder)):
            continue
        print('Processing {}'.format(sub_folder))
        params_file = os.path.join(data_root, sub_folder, 'SMPLH', 'smplh.pkl')
        smplx_param_list = sorted(
            glob.glob(os.path.join(data_root, sub_folder, 'SMPLX', '*smplx.pkl')))
        with open(params_file, 'rb') as f:
            params_data = pickle.load(f, encoding='latin1')
            N = len(params_data['global_orient'])
            assert N == len(smplx_param_list)
            for i in tqdm(range(N)):
                new_params_data = {}
                new_params_data['global_orient'] = params_data['global_orient'][i].squeeze().detach().cpu().numpy()
                new_params_data['transl'] = params_data['transl'][i].detach().cpu().numpy()
                new_params_data['betas'] = betas.astype(np.float32)
                new_params_data['body_pose'] = torch.flatten(
                    params_data['body_pose'][i]).detach().cpu().numpy()
                new_params_data['left_hand_pose'] = (
                    torch.flatten(params_data['left_hand_pose'][i]) -
                    body_model.left_hand_mean).detach().cpu().numpy()
                new_params_data['right_hand_pose'] = (
                    torch.flatten(params_data['right_hand_pose'][i]) -
                    body_model.right_hand_mean).detach().cpu().numpy()
                
                with open(os.path.join(data_root, sub_folder, 'SMPLH',
                            smplx_param_list[i].split('/')[-1].replace(
                                'smplx', 'smplh')), 'wb') as f:
                    pickle.dump(new_params_data, f)
        os.remove(params_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender',
                        type=str,
                        default='female',
                        help='gender of the subject')
    parser.add_argument(
        '--data_root',
        type=str,
        default=
        '/path/to/GRAB/Dateset/Person_ID',
        help='data root')
    parser.add_argument('--model_root',
                        type=str,
                        default='/path/to/SMPL(-H/X)/models',
                        help='model root')

    args = parser.parse_args()

    gender = args.gender
    data_root = args.data_root
    model_root = args.model_root

    """GRAB dataset processing"""

    generate_SMPLX(data_root='{}/train'.format(data_root), model_root=model_root, gender=gender)
    generate_SMPLX(data_root='{}/test'.format(data_root), model_root=model_root, gender=gender)

    """ After transfering SMPLX to SMPLH and SMPL"""
    compute_average_shape(data_root)

    for mode in ['train', 'test']:
        convert_transfered_SMPLH2SMPL_in_batch(
            data_root='{}/{}'.format(data_root, mode),
            model_root=model_root,
            gender=gender)

        generate_transfered_SMPLH_in_batch(
            data_root='{}/{}'.format(data_root, mode),
            model_root=
            '{}/smplh_amass/SMPLH_{}.pkl'.format(model_root, gender.upper()),
            gender=gender)


if __name__ == '__main__':
    sys.exit(main())