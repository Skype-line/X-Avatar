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

class AMASSDataSet(Dataset):

    def __init__(self, dataset_path, model_type, use_pca,
                 num_pca_comps, flat_hand_mean,
                 mode='train'):

        self.model_type = model_type
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps

        dataset_path = hydra.utils.to_absolute_path(dataset_path)
        self.betas = np.load(osp.join(dataset_path, 'mean_shape_{}.npy'.format(self.model_type)))[:10]
        self.gender = open(osp.join(dataset_path, 'gender.txt')).readlines()[0].strip()

        self.regstr_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*', model_type.upper(), '*.ply')))
        self.smplx_params_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*', model_type.upper(), '*.pkl')))
        
        assert len(self.regstr_list) == len(self.smplx_params_list)

        if mode == 'test': # only use one sample for testing
            self.regstr_list = self.regstr_list[0:1]
            self.smplx_params_list = self.smplx_params_list[0:1]
            
        self.meta_info = {
            'gender': self.gender,
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
                with open(self.smplx_params_list[index], 'rb') as f:
                    params_data = pkl.load(f, encoding='latin')
                    global_orient = params_data[
                        'global_orient'] if 'global_orient' in params_data else None
                    transl = params_data[
                        'transl'] if 'transl' in params_data else None
                    body_pose = params_data[
                        'body_pose'] if 'body_pose' in params_data else None
                    hand_pose = params_data[
                        'hand_pose'] if 'hand_pose' in params_data else None
                    left_hand_pose = params_data[
                        'left_hand_pose'] if 'left_hand_pose' in params_data else None
                    right_hand_pose = params_data[
                        'right_hand_pose'] if 'right_hand_pose' in params_data else None
                    leye_pose = np.zeros(3)
                    reye_pose = np.zeros(3)
                    jaw_pose = params_data[
                        'jaw_pose'] if 'jaw_pose' in params_data else None
                    betas = self.betas
                    expression = params_data[
                        'expression'] if 'expression' in params_data else None
                break
            except:
                print('corrupted ply/pkl')
                print(self.smplx_params_list[index])
                index = np.random.randint(self.__len__())

        # load 3D scan
        regstr_verts = regstr_mesh.vertices - transl
        regstr_verts = torch.tensor(regstr_verts).float()
        regstr_faces = regstr_mesh.faces.astype(np.int32)
        regstr_faces = torch.tensor(regstr_faces).long()
        data['regstr_verts'] = regstr_verts
        data['regstr_faces'] = regstr_faces

        # load SMPL(-H/X) parameters
        num_hand_pose = self.num_pca_comps if self.use_pca else 45

        if self.model_type == 'smplx':
            smpl_params = torch.zeros([99 + 2 * num_hand_pose]).float()
            smpl_params[0] = 1
            smpl_params[4:7] = torch.tensor(global_orient).float()
            smpl_params[7:70] = torch.tensor(body_pose).float()
            smpl_params[70:70 +
                        num_hand_pose] = torch.tensor(left_hand_pose).float()
            smpl_params[70 + num_hand_pose:70 + 2 *
                        num_hand_pose] = torch.tensor(right_hand_pose).float()
            smpl_params[70 + 2 * num_hand_pose:73 +
                        2 * num_hand_pose] = torch.tensor(leye_pose).float()
            smpl_params[73 + 2 * num_hand_pose:76 +
                        2 * num_hand_pose] = torch.tensor(reye_pose).float()
            smpl_params[76 + 2 * num_hand_pose:79 +
                        2 * num_hand_pose] = torch.tensor(jaw_pose).float()
            smpl_params[79 + 2 * num_hand_pose:89 +
                        2 * num_hand_pose] = torch.tensor(betas).float()
            smpl_params[89 + 2 * num_hand_pose:99 +
                        2 * num_hand_pose] = torch.tensor(expression).float()
            data['smpl_thetas'] = smpl_params[7:70]
            data['smpl_exps'] = torch.tensor(expression).float()
        elif self.model_type == 'smplh':
            smpl_params = torch.zeros([80 + 2 * num_hand_pose]).float()
            smpl_params[0] = 1
            smpl_params[4:7] = torch.tensor(global_orient).float()
            smpl_params[7:70] = torch.tensor(body_pose).float()
            smpl_params[70:70 +
                        num_hand_pose] = torch.tensor(left_hand_pose).float()
            smpl_params[70 + num_hand_pose:70 + 2 *
                        num_hand_pose] = torch.tensor(right_hand_pose).float()
            smpl_params[70 + 2 * num_hand_pose:80 +
                        2 * num_hand_pose] = torch.tensor(betas).float()
            data['smpl_thetas'] = smpl_params[7:70]
            data['smpl_exps'] = None
        elif self.model_type == 'smpl':
            smpl_params = torch.zeros([86]).float()
            smpl_params[0] = 1
            smpl_params[4:7] = torch.tensor(global_orient).float()
            smpl_params[7:76] = torch.tensor(body_pose).float()
            smpl_params[76:86] = torch.tensor(betas).float()
            data['smpl_thetas'] = smpl_params[7:76]
            data['smpl_exps'] = None
        elif self.model_type == 'mano':
            smpl_params = torch.zeros([17 + num_hand_pose]).float()
            smpl_params[0] = 1
            smpl_params[4:7] = torch.tensor(global_orient).float()
            smpl_params[7:7 + num_hand_pose] = torch.tensor(hand_pose).float()
            smpl_params[7 + num_hand_pose:17 +
                        num_hand_pose] = torch.tensor(betas).float()
            data['smpl_thetas'] = smpl_params[4:7 + num_hand_pose]
            data['smpl_exps'] = None
        else:
            raise NotImplementedError

        data['smpl_params'] = smpl_params

        return data

    def __len__(self):
        return len(self.smplx_params_list)


class AMASSDataProcessor():
    ''' 
    Used to generate groud-truth occupancy and bone transformations 
    in batchs during training 
    '''
    def __init__(self, opt, meta_info, model_type, **kwargs):
        from lib.model.smpl import SMPLServer, SMPLHServer, SMPLXServer, MANOServer

        self.opt = opt
        self.gender = meta_info['gender'] if 'gender' in meta_info else None
        self.betas = meta_info['betas'] if 'betas' in meta_info else None
        self.use_pca = meta_info['use_pca'] if 'use_pca' in meta_info else True
        self.num_pca_comps = meta_info[
            'num_pca_comps'] if 'num_pca_comps' in meta_info else None
        self.flat_hand_mean = meta_info[
            'flat_hand_mean'] if 'flat_hand_mean' in meta_info else None

        if model_type == 'smplx':
            self.smpl_server = SMPLXServer(gender=self.gender,
                                           betas=self.betas,
                                           use_pca=self.use_pca,
                                           num_pca_comps=self.num_pca_comps,
                                           flat_hand_mean=self.flat_hand_mean)
        elif model_type == 'smplh':
            self.smpl_server = SMPLHServer(gender=self.gender,
                                           betas=self.betas,
                                           use_pca=self.use_pca,
                                           num_pca_comps=self.num_pca_comps,
                                           flat_hand_mean=self.flat_hand_mean)
        elif model_type == 'smpl':
            self.smpl_server = SMPLServer(gender=self.gender,
                                          betas=self.betas)
        elif model_type == 'mano':
            self.smpl_server = MANOServer(betas=self.betas,
                                          use_pca=self.use_pca,
                                          num_pca_comps=self.num_pca_comps,
                                          flat_hand_mean=self.flat_hand_mean)
        else:
            raise NotImplementedError

        self.sampler = hydra.utils.instantiate(opt.sampler)

        if self.opt.category_sample:
            if model_type == 'smplx':
                # load part labels
                verts_ids = pkl.load(open(hydra.utils.to_absolute_path(
                        'lib/smplx/smplx_model/watertight_{}_vertex_labels.pkl'.
                        format(self.gender)), 'rb'), encoding='latin1')
                
                self.body_ids = torch.tensor(
                    verts_ids['body']).cuda() if 'body' in verts_ids else None
                self.lhand_ids = torch.tensor(verts_ids['left_hand']).cuda(
                    ) if 'left_hand' in verts_ids else None
                self.rhand_ids = torch.tensor(verts_ids['right_hand']).cuda(
                    ) if 'right_hand' in verts_ids else None
                self.face_ids = torch.tensor(
                    verts_ids['face']).cuda() if 'face' in verts_ids else None
                self.eye_mouth_ids = torch.tensor(verts_ids['eyes_mouth']).cuda(
                    ) if 'eyes_mouth' in verts_ids else None
                
                # set part splits
                self.split_numbers = [self.opt.points_per_frame // 3, 
                                      self.opt.points_per_frame // 3, 
                                      self.opt.points_per_frame // 6,
                                      self.opt.points_per_frame // 6]
                self.part_types = ['body', 'face', 'lhand', 'rhand']
            else:
                raise NotImplementedError
        else:
            self.split_numbers = [self.opt.points_per_frame]
            self.part_types = ['full'] if model_type == 'smplx' else ['body']


    def process(self, data):

        smpl_output = self.smpl_server(data['smpl_params'], absolute=False)
        data.update(smpl_output)

        pts_d, shape_gt = [], []
             
        for i in range(len(data['regstr_verts'])):
            regstr_verts = data['regstr_verts'][i].expand(1, -1, -1)
            
            if self.opt.category_sample:
                random_idx = list()

                body_ids = self.body_ids[
                    self.body_ids < regstr_verts.
                    shape[1]] if self.body_ids is not None else None
                face_ids = self.face_ids[
                    self.face_ids < regstr_verts.
                    shape[1]] if self.face_ids is not None else None
                eye_mouth_ids = self.eye_mouth_ids[
                    self.eye_mouth_ids < regstr_verts.
                    shape[1]] if self.eye_mouth_ids is not None else None
                
                if body_ids is not None:
                    random_body_idx = body_ids[torch.randint(
                        0,
                        body_ids.shape[0],
                        [1, self.split_numbers[0], 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_body_idx)
                if face_ids is not None:
                    random_face_idx = face_ids[torch.randint(
                        0,
                        face_ids.shape[0],
                        [1, self.split_numbers[1]//2, 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_face_idx)
                if eye_mouth_ids is not None:
                    random_eye_mouth_idx = eye_mouth_ids[torch.randint(
                        0,
                        eye_mouth_ids.shape[0],
                        [1, self.split_numbers[1]//2, 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_eye_mouth_idx)
                if self.lhand_ids is not None:
                    random_lhand_idx = self.lhand_ids[torch.randint(
                        0,
                        self.lhand_ids.shape[0],
                        [1, self.split_numbers[2], 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_lhand_idx)
                if self.rhand_ids is not None:
                    random_rhand_idx = self.rhand_ids[torch.randint(
                        0,
                        self.rhand_ids.shape[0],
                        [1, self.split_numbers[3], 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_rhand_idx)
                random_idx = torch.cat(random_idx, dim=1)
            else:
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


class AMASSDataModule(pl.LightningDataModule):

    def __init__(self, opt, model_type, **kwargs):
        super().__init__()
        self.opt = opt
        self.model_type = model_type

    def setup(self, stage=None):

        if stage == 'fit':
            self.dataset_train = AMASSDataSet(
                dataset_path=self.opt.dataset_path,
                model_type=self.model_type,
                use_pca=self.opt.use_pca,
                num_pca_comps=self.opt.num_pca_comps,
                flat_hand_mean=self.opt.flat_hand_mean,
                mode='train')

        self.dataset_val = AMASSDataSet(dataset_path=self.opt.dataset_path,
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