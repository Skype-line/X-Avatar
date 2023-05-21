import numpy as np
import os.path as osp
import glob
import hydra
import pickle as pkl

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import kaolin
import trimesh


class MANODataSet(Dataset):

    def __init__(self, dataset_path, model_type, use_pca,
                 num_pca_comps, flat_hand_mean,
                 mode='train'):

        self.model_type = model_type
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps

        dataset_path = hydra.utils.to_absolute_path(dataset_path)
        self.betas = np.load(osp.join(dataset_path, 'mean_shape_{}.npy'.format(self.model_type)))[:10]
        
        self.regstr_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*.ply')))
        self.mano_params_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*.pkl')))
        
        assert len(self.regstr_list) == len(self.mano_params_list)

        if mode == 'test': # only use one sample for testing
            self.regstr_list = self.regstr_list[0:1]
            self.mano_params_list = self.mano_params_list[0:1]
            
        self.meta_info = {
            'betas': self.betas,
            'representation': 'occ',
            'use_pca': use_pca,
            'num_pca_comps': num_pca_comps,
            'flat_hand_mean': flat_hand_mean
        }

    def __getitem__(self, index):

        data = {}

        while True:
            try:
                regstr_mesh = trimesh.load(self.regstr_list[index])
                with open(self.mano_params_list[index], 'rb') as f:
                    params_data = pkl.load(f, encoding='latin')
                    global_orient = params_data[
                        'global_orient'] if 'global_orient' in params_data else None
                    transl = params_data[
                        'transl'] if 'transl' in params_data else None
                    hand_pose = params_data[
                        'hand_pose'] if 'hand_pose' in params_data else None
                    betas = self.betas
                break
            except:
                print('corrupted ply/pkl')
                print(self.mano_params_list[index])
                index = np.random.randint(self.__len__())

        # load 3D scan
        regstr_verts = regstr_mesh.vertices - transl
        regstr_verts = torch.tensor(regstr_verts).float()
        regstr_faces = regstr_mesh.faces.astype(np.int32)
        regstr_faces = torch.tensor(regstr_faces).long()
        data['regstr_verts'] = regstr_verts
        data['regstr_faces'] = regstr_faces

        # load MANO parameters
        num_hand_pose = self.num_pca_comps if self.use_pca else 45

        smpl_params = torch.zeros([17 + num_hand_pose]).float()
        smpl_params[0] = 1
        smpl_params[4:7] = torch.tensor(global_orient).float()
        smpl_params[7:7 + num_hand_pose] = torch.tensor(hand_pose).float()
        smpl_params[7 + num_hand_pose:17 +
                    num_hand_pose] = torch.tensor(betas).float()
        data['smpl_params'] = smpl_params
        data['smpl_thetas'] = smpl_params[7:7 + num_hand_pose]
        data['smpl_exps'] = None
        
        return data

    def __len__(self):
        return len(self.mano_params_list)


class MANODataProcessor():
    ''' 
    Used to generate groud-truth occupancy and bone transformations 
    in batchs during training 
    '''
    def __init__(self, opt, meta_info, model_type, **kwargs):
        from lib.model.smpl import MANOServer

        self.opt = opt
        self.betas = meta_info['betas'] if 'betas' in meta_info else None
        self.use_pca = meta_info['use_pca'] if 'use_pca' in meta_info else True
        self.num_pca_comps = meta_info[
            'num_pca_comps'] if 'num_pca_comps' in meta_info else None
        self.flat_hand_mean = meta_info[
            'flat_hand_mean'] if 'flat_hand_mean' in meta_info else None

        
        self.smpl_server = MANOServer(betas=self.betas,
                                        use_pca=self.use_pca,
                                        num_pca_comps=self.num_pca_comps,
                                        flat_hand_mean=self.flat_hand_mean)

        self.sampler = hydra.utils.instantiate(opt.sampler)

        self.split_numbers = [self.opt.points_per_frame]
        self.part_types = ['hand']


    def process(self, data):

        smpl_output = self.smpl_server(data['smpl_params'], absolute=False)
        data.update(smpl_output)

        pts_d, shape_gt = [], []
             
        for i in range(len(data['regstr_verts'])):
            regstr_verts = data['regstr_verts'][i].expand(1, -1, -1)
            
            
            num_verts = data['regstr_verts'][i].shape[0]
            random_idx = torch.randint(
                0,
                num_verts, [1, self.opt.points_per_frame, 1],
                device=smpl_output['smpl_verts'].device)

            num_dim = data['regstr_verts'][i].shape[1]

            random_pts = torch.gather(regstr_verts, 1,
                                        random_idx.expand(-1, -1, num_dim))
            sampled_pts = self.sampler.get_points(random_pts, sample_size=0)
            
            pts_d.append(sampled_pts)
            shape_gt.append(kaolin.ops.mesh.check_sign(
                    regstr_verts, data['regstr_faces'][i],
                    sampled_pts).float().unsqueeze(-1))

        data['pts_d'] = torch.cat(pts_d, dim=0)
        data['shape_gt'] = torch.cat(shape_gt, dim=0)
        data['split_numbers'] = torch.tensor(self.split_numbers).cuda()
        data['part_types'] = self.part_types
        
        return data


''' Customized collate function to deal with different sizes of vertices/faces '''


def collate_fn(batch):

    data = {}
    regstr_verts, regstr_faces, smpl_params, smpl_thetas, smpl_exps = [],[],[],[],[]

    for item in batch:
        regstr_verts.append(item['regstr_verts'])
        regstr_faces.append(item['regstr_faces'])
        smpl_params.append(item['smpl_params'])
        smpl_thetas.append(item['smpl_thetas'])
        if item['smpl_exps'] is not None:
            smpl_exps.append(item['smpl_exps'])
        
    data['regstr_verts'] = regstr_verts
    data['regstr_faces'] = regstr_faces
    data['smpl_params'] = torch.stack(smpl_params)
    data['smpl_thetas'] = torch.stack(smpl_thetas)
    data['smpl_exps'] = torch.stack(smpl_exps) if len(smpl_exps) > 0 else None
    return data


class MANODataModule(pl.LightningDataModule):

    def __init__(self, opt, model_type, **kwargs):
        super().__init__()
        self.opt = opt
        self.model_type = model_type

    def setup(self, stage=None):

        if stage == 'fit':
            self.dataset_train = MANODataSet(
                dataset_path=self.opt.dataset_path,
                model_type=self.model_type,
                use_pca=self.opt.use_pca,
                num_pca_comps=self.opt.num_pca_comps,
                flat_hand_mean=self.opt.flat_hand_mean,
                mode='train')

        self.dataset_val = MANODataSet(dataset_path=self.opt.dataset_path,
                                        model_type=self.model_type,
                                        use_pca=self.opt.use_pca,
                                        num_pca_comps=self.opt.num_pca_comps,
                                        flat_hand_mean=self.opt.flat_hand_mean,
                                        mode='test')

        self.meta_info = self.dataset_val.meta_info

    def train_dataloader(self):
        dataloader = DataLoader(self.dataset_train,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=collate_fn,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.dataset_val,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=collate_fn,
                                pin_memory=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset_val,
                                batch_size=1,
                                num_workers=self.opt.num_workers,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=collate_fn,
                                pin_memory=True)
        return dataloader