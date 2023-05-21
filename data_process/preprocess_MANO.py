"""
Convert MANO training scans registrations and parameters to our data format.
"""
import sys
import numpy as np
import pickle
import pymesh
import os.path as osp
from tqdm import tqdm
import argparse
import glob

import smplx
import torch
from pytorch3d.structures import Meshes
from pymeshfix import _meshfix

def convert_params2mesh_mano(param_dir: str,
                             model_root: str,
                             dtype=torch.float32,
                             device='cuda:0'):
    model_params = dict(model_path=model_root,
                        create_global_orient=False,
                        create_transl=False,
                        create_betas=False,
                        create_hand_pose=False,
                        dtype=dtype,
                        batch_size=1,
                        is_rhand=True,
                        use_pca=False,
                        flat_hand_mean=True,
                        model_type='mano')
    body_model = smplx.create(**model_params).to(device=device)

    betas = None
    params_file_list = sorted(glob.glob(osp.join(param_dir, '*', '*.pkl')))

    for params_file in tqdm(params_file_list):
        with open(params_file, 'rb') as f:
            params_data = {}
            hands_params = pickle.load(f, encoding='latin1')
            if betas is None:
                betas = hands_params['betas']
                np.save(osp.join(param_dir, 'mean_shape_mano.npy'), betas)
            params_data['global_orient'] = hands_params['pose'][:3].astype(
                np.float32)
            params_data['transl'] = hands_params['trans'].astype(np.float32)
            params_data['betas'] = betas.astype(np.float32)
            params_data['hand_pose'] = hands_params['pose'][3:].astype(
                np.float32)

        model_output = body_model(
            return_verts=True,
            global_orient=torch.from_numpy(
                params_data['global_orient']).to(device).unsqueeze(0),
            transl=torch.from_numpy(
                params_data['transl']).to(device).unsqueeze(0),
            betas=torch.from_numpy(
                params_data['betas']).to(device).unsqueeze(0),
            hand_pose=torch.from_numpy(
                params_data['hand_pose']).to(device).unsqueeze(0),
            return_full_pose=False)

        vertices = model_output.vertices.detach().cpu()[0]

        with open(params_file, 'wb') as f:
            pickle.dump(params_data, f)

        # Patching holes
        tin = _meshfix.PyTMesh()
        tin.load_array(v=vertices, f=body_model.faces)
        if tin.boundaries():
            _ = tin.fill_small_boundaries()
        tin.save_file(params_file.replace('.pkl', '.ply'))
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--param_dir',
        type=str,
        default=
        '/path/to/manoposesv10/mano_poses_v1_0')
    parser.add_argument('--model_root',
                        type=str,
                        default='/path/to/MANO_RIGHT.pkl',
                        help='model root')
    
    args = parser.parse_args()

    convert_params2mesh_mano(
        param_dir=args.param_dir,
        model_root=args.model_root,
    )


if __name__ == '__main__':
    sys.exit(main())