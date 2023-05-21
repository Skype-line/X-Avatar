"""
Demo on X-Avatars animated by motions from monocular RGB videos.
Motions are extracted by PyMAF-X https://github.com/HongwenZhang/PyMAF-X and smoothed
"""

import os
import os.path as osp
import glob
import joblib
import hydra
from tqdm import trange
import numpy as np

import cv2
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
    data = joblib.load(opt.demo.motion_path)
    
    smpl_params_all = []
    for i in range(data['body_pose'].shape[0]):
        if i % opt.demo.every_n_frames!=0:
            continue
        # convert global orientation from PyMAF-X's format to ours 
        global_orient = data['body_pose'][i][:3]
        R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        R_root = cv2.Rodrigues(global_orient)[0]
        new_root = R_mod.dot(R_root)
        global_orient = cv2.Rodrigues(new_root)[0].reshape(3)
        
        body_pose = data['body_pose'][i][3:-6]
        left_hand_pose = data['left_hand_pose'][i]
        right_hand_pose = data['right_hand_pose'][i]
        jaw_pose = data['jaw_pose'][i]
        expression = data['expression'][i]

        smpl_params = torch.zeros(99+2*num_hand_pose, dtype=torch.float32)
        smpl_params[0] = 1
        smpl_params[4:7] = torch.tensor(global_orient, dtype=torch.float32)
        smpl_params[7:70] = torch.tensor(body_pose, dtype=torch.float32)
        smpl_params[70:70+num_hand_pose] = torch.tensor(left_hand_pose, dtype=torch.float32) - model.smpl_server.model.left_hand_mean.squeeze().cpu()
        smpl_params[70+num_hand_pose:70+2*num_hand_pose] = torch.tensor(right_hand_pose, dtype=torch.float32) - model.smpl_server.model.right_hand_mean.squeeze().cpu()
        smpl_params[70+2*num_hand_pose:73+2*num_hand_pose] = torch.zeros(3, dtype=torch.float32)
        smpl_params[73+2*num_hand_pose:76+2*num_hand_pose] = torch.zeros(3, dtype=torch.float32)
        smpl_params[76+2*num_hand_pose:79+2*num_hand_pose] = torch.tensor(jaw_pose, dtype=torch.float32)
        smpl_params[79+2*num_hand_pose:89+2*num_hand_pose] = torch.tensor(betas, dtype=torch.float32)
        smpl_params[89+2*num_hand_pose:99+2*num_hand_pose] = torch.tensor(expression, dtype=torch.float32)
        
        smpl_params_all.append(smpl_params.cuda())

    smpl_params_all = torch.stack(smpl_params_all, dim=0)
    
    for i in trange(smpl_params_all.shape[0]):
        smpl_params = smpl_params_all[[i]]
        data = model.smpl_server.forward(smpl_params, absolute=False)
        data['smpl_thetas'] = smpl_params[:, 7:70]
        data['smpl_exps'] = smpl_params[:, -10:]
        
        results = model.plot(data,
                             res=opt.demo.resolution,
                             verbose=opt.demo.verbose,
                             fast_mode=opt.demo.fast_mode)

        if opt.demo.save_mesh:
            save_folder = osp.basename(opt.demo.motion_path).split('.')[0]
            if not osp.exists(save_folder):
                os.makedirs(save_folder)
            results['mesh_def'].export(osp.join(save_folder, '%04d_def.ply' % i))
            if opt.demo.verbose:
                results['mesh_cano'].export(osp.join(save_folder,'%04d_cano.ply' % i))


if __name__ == '__main__':
    main()