"""
Demo on X-Avatars animated by motions from our X-Humans dataset.
pretrained model in .ckpt format
"""
import hydra
import os
import os.path as osp
import glob
from tqdm import trange
import numpy as np
import pickle as pkl

import torch
import pytorch_lightning as pl

from lib.xavatar_scan_model import XAVATARSCANModel
from lib.xavatar_rgbd_model import XAVATARRGBDModel


@hydra.main(config_path="config", config_name="config")
def main(opt):
    pl.seed_everything(42, workers=True)

    # load meta info
    meta_info = np.load('./meta_info.npz', allow_pickle=True)
    betas = meta_info['betas']
    num_hand_pose = meta_info['num_pca_comps'].item() if meta_info['use_pca'].item() else 45
    model_type = opt.experiments.model.model_type

    # load pretrained model
    model_class = XAVATARSCANModel if meta_info['representation'] == 'occ' else XAVATARRGBDModel
    
    if glob.glob('./checkpoints/*.pth'):
        pretrained_model = torch.load(glob.glob('./checkpoints/*.pth')[-1])
        model = model_class(opt=opt.experiments.model,
                        meta_info=meta_info)
        model.shape_net.load_state_dict(pretrained_model['shapenet_state_dict'])
        model.deformer.load_state_dict(pretrained_model['deformer_state_dict'])
        model.color_net.load_state_dict(pretrained_model['colornet_state_dict'])
        model = model.cuda()
    else:
        if opt.experiments.epoch == 'last':
            pretrained_model = './checkpoints/last.ckpt'
        else:
            pretrained_model = glob.glob('./checkpoints/epoch=%d*.ckpt' %
                                        opt.experiments.epoch)[0]
        model = model_class.load_from_checkpoint(checkpoint_path=pretrained_model,
                                                opt=opt.experiments.model,
                                                meta_info=meta_info).cuda()
    print(model)

    
    # load motion sequence
    motion_path = hydra.utils.to_absolute_path(opt.demo.motion_path)
    if model_type == 'mano':
        motion_files = sorted(glob.glob(osp.join(motion_path, '*.pkl')))
    else:
        motion_files = sorted(glob.glob(osp.join(motion_path, '*', model_type.upper(), '*.pkl')))
    
    smpl_params_all = []
    for f in motion_files[::opt.demo.every_n_frames]:
        f = pkl.load(open(f, 'rb'), encoding='latin1')
        if model_type == 'smplx':
            smpl_params = np.zeros(99+2*num_hand_pose)
            smpl_params[0] = 1
            smpl_params[1:4] = f['transl']
            smpl_params[4:7] = f['global_orient']
            smpl_params[7:70] = f['body_pose']
            smpl_params[70:70+num_hand_pose] = f['left_hand_pose']
            smpl_params[70+num_hand_pose:70+2*num_hand_pose] = f['right_hand_pose']
            smpl_params[70+2*num_hand_pose:73+2*num_hand_pose] = np.zeros(3)
            smpl_params[73+2*num_hand_pose:76+2*num_hand_pose] = np.zeros(3)
            smpl_params[76+2*num_hand_pose:79+2*num_hand_pose] = f['jaw_pose']
            smpl_params[79+2*num_hand_pose:89+2*num_hand_pose] = betas
            smpl_params[89+2*num_hand_pose:99+2*num_hand_pose] = f['expression']
        elif model_type == 'smplh':
            smpl_params = np.zeros(80+2*num_hand_pose)
            smpl_params[0] = 1
            smpl_params[1:4] = f['transl']
            smpl_params[4:7] = f['global_orient']
            smpl_params[7:70] = f['body_pose']
            smpl_params[70:70+num_hand_pose] = f['left_hand_pose']
            smpl_params[70+num_hand_pose:70+2*num_hand_pose] = f['right_hand_pose']
            smpl_params[70+2*num_hand_pose:80+2*num_hand_pose] = betas
        elif model_type == 'smpl':
            smpl_params = np.zeros(86)
            smpl_params[0] = 1
            smpl_params[1:4] = f['transl']
            smpl_params[4:7] = f['global_orient']
            smpl_params[7:76] = f['body_pose']
            smpl_params[76:86] = betas
        elif model_type == 'mano':
            smpl_params = np.zeros(17+num_hand_pose)
            smpl_params[0] = 1
            # smpl_params[1:4] = f['transl']
            smpl_params[4:7] = f['global_orient']
            smpl_params[7:7+num_hand_pose] = f['hand_pose']
            smpl_params[7+num_hand_pose:17+num_hand_pose] = betas
        else:
            raise NotImplementedError

        smpl_params = torch.tensor(smpl_params).float().cuda()
        smpl_params_all.append(smpl_params)
    
    smpl_params_all = torch.stack(smpl_params_all)

    for i in trange(smpl_params_all.shape[0]):
        smpl_params = smpl_params_all[[i]]
        data = model.smpl_server.forward(smpl_params, absolute=False)
        if model_type == 'smplx':
            data['smpl_thetas'] = smpl_params[:, 7:70]
            data['smpl_exps'] = smpl_params[:, -10:]
        elif model_type == 'smplh':
            data['smpl_thetas'] = smpl_params[:, 7:70]
            data['smpl_exps'] = None
        elif model_type == 'smpl':
            data['smpl_thetas'] = smpl_params[:, 7:76]
            data['smpl_exps'] = None
        elif model_type == 'mano':
            data['smpl_thetas'] = smpl_params[:, 7:7+num_hand_pose]
            data['smpl_exps'] = None
        else:
            raise NotImplementedError
        
        results = model.plot(data,
                             res=opt.demo.resolution,
                             verbose=opt.demo.verbose,
                             fast_mode=opt.demo.fast_mode)

        if opt.demo.save_mesh:
            mode = motion_path.split('/')[-1]
            save_folder = f'meshes_{mode}'
            if not osp.exists(save_folder):
                os.makedirs(save_folder)
            results['mesh_def'].export(osp.join(save_folder, '%04d_def.ply' % i))
            if opt.demo.verbose:
                results['mesh_cano'].export(osp.join(save_folder,'%04d_cano.ply' % i))


if __name__ == '__main__':
    main()