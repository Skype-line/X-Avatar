import numpy as np
import os.path as osp
import glob
import hydra
import pickle as pkl
import open3d as o3d

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import kaolin
import trimesh


def category_sample_points(part_ids, verts, colors, normals, num_points,
                           num_dim):
    if part_ids is not None and part_ids.shape[0] != 0:
        random_part_idx = part_ids[torch.randint(0,
                                                 part_ids.shape[0],
                                                 [1, num_points, 1],
                                                 device=verts.device)]
        random_pts = torch.gather(verts, 1,
                                  random_part_idx.expand(-1, -1, num_dim))
        random_pts_color = torch.gather(
            colors, 1, random_part_idx.expand(
                -1, -1, num_dim)) if colors is not None else None
        random_pts_normal = torch.gather(
            normals, 1, random_part_idx.expand(-1, -1, num_dim))
        valid = True
    else:
        random_pts = torch.zeros([1, num_points, num_dim],
                                 device=verts.device).float()
        random_pts_color = torch.zeros([1, num_points, num_dim],
                                       device=verts.device).float()
        random_pts_normal = torch.zeros([1, num_points, num_dim],
                                        device=verts.device).float()
        valid = False
    return random_pts, random_pts_color, random_pts_normal, valid


class XHumansDataSet(Dataset):

    def __init__(self, dataset_path, model_type, use_pca,
                 num_pca_comps, flat_hand_mean,
                 mode='train'):

        self.model_type = model_type
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        
        dataset_path = hydra.utils.to_absolute_path(dataset_path)
        self.betas = np.load(osp.join(dataset_path, 'mean_shape_{}.npy'.format(model_type)))
        self.gender = open(osp.join(dataset_path, 'gender.txt')).readlines()[0].strip()

        self.regstr_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*', model_type.upper(), '*.ply')))
        self.smplx_params_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*', model_type.upper(), '*.pkl')))
        self.pcl_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*', 'pcl', '*.ply')))
        
        assert len(self.regstr_list) == len(self.smplx_params_list) == len(self.pcl_list)

        if mode == 'test': # only use one sample for testing
            self.regstr_list = self.regstr_list[0:1]
            self.smplx_params_list = self.smplx_params_list[0:1]
            self.pcl_list = self.pcl_list[0:1]
        
        self.meta_info = {
            'gender': self.gender,
            'betas': self.betas,
            'representation': 'sdf',
            'use_pca': use_pca,
            'num_pca_comps': num_pca_comps,
            'flat_hand_mean': flat_hand_mean
        }

    def __getitem__(self, index):

        data = {}

        while True:
            try:
                regstr_mesh = trimesh.load(self.regstr_list[index])
                pcl = o3d.io.read_point_cloud(self.pcl_list[index])
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

        # load SMPL(-H/X) registration
        regstr_verts = regstr_mesh.vertices - transl
        regstr_verts = torch.tensor(regstr_verts).float()
        regstr_faces = regstr_mesh.faces.astype(np.int32)
        regstr_faces = torch.tensor(regstr_faces).long()
        regstr_normals = torch.tensor(regstr_mesh.vertex_normals.copy()).float()
        data['regstr_verts'] = regstr_verts
        data['regstr_faces'] = regstr_faces
        data['regstr_normals'] = regstr_normals
        
        # load human point cloud
        pcl_verts = np.array(pcl.points) - transl
        pcl_verts = torch.tensor(pcl_verts).float()
        pcl_colors = torch.tensor(pcl.colors).float()
        pcl_normals = torch.tensor(pcl.normals).float()
        data['pcl_verts'] = pcl_verts
        data['pcl_colors'] = pcl_colors
        data['pcl_normals'] = pcl_normals
    
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


class XHumansDataProcessor():
    ''' 
    Used to generate groud-truth sdf and bone transformations 
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
        self.model_type = model_type

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
                        'lib/smplx/smplx_model/non_watertight_{}_vertex_labels.pkl'
                        .format(self.gender)), 'rb'), encoding='latin1')  
                
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
                
                self.regstr_label_vector = torch.zeros(
                    self.smpl_server.verts_c.shape[1], dtype=torch.int32).cuda()
                
                if self.face_ids is not None:
                    self.regstr_label_vector[self.face_ids] = 1
                if self.lhand_ids is not None:
                    self.regstr_label_vector[self.lhand_ids] = 2
                if self.rhand_ids is not None:
                    self.regstr_label_vector[self.rhand_ids] = 3
                if self.eye_mouth_ids is not None:
                    self.regstr_label_vector[self.eye_mouth_ids] = 4
            else:
                raise NotImplementedError

    def process(self, data):

        smpl_output = self.smpl_server(data['smpl_params'], absolute=False)
        data.update(smpl_output)

        pts_surf, color_gt, normal_gt, split_numbers, part_types = [], [], [], [], []

        for i in range(data['regstr_verts'].shape[0]):
            regstr_verts = data['regstr_verts'][i].expand(1, -1, -1)
            regstr_normals = data['regstr_normals'][i].expand(1, -1, -1)
            pcl_verts = data['pcl_verts'][i].expand(1, -1, -1)
            pcl_colors = data['pcl_colors'][i].expand(1, -1, -1)
            pcl_normals = data['pcl_normals'][i].expand(1, -1, -1)
            num_dim = data['regstr_verts'][i].shape[1]
            
            if self.opt.category_sample:
                random_pts_list = []
                random_color_list = []
                random_normal_list = []
                
                # predict the part type of each point
                _, close_id = kaolin.metrics.pointcloud.sided_distance(
                    pcl_verts, regstr_verts)
                pcl_label_vector = self.regstr_label_vector[close_id[0]]
                body_ids = torch.where(
                    pcl_label_vector ==
                    0)[0] if self.body_ids is not None else None
                face_ids = torch.where(
                    pcl_label_vector ==
                    1)[0] if self.face_ids is not None else None
                lhand_ids = torch.where(
                    pcl_label_vector ==
                    2)[0] if self.lhand_ids is not None else None
                rhand_ids = torch.where(
                    pcl_label_vector ==
                    3)[0] if self.rhand_ids is not None else None
                eye_mouth_ids = torch.where(
                        pcl_label_vector == 4
                    )[0] if self.eye_mouth_ids is not None else None
                
                num_body_pts = self.opt.points_per_frame // 3
                random_body_pts, random_body_color, random_body_normal, _ = category_sample_points(
                    body_ids, pcl_verts, pcl_colors, pcl_normals,
                    num_body_pts, num_dim)
                split_numbers.append(num_body_pts)
                part_types.append('body')
                random_pts_list.append(random_body_pts)
                random_color_list.append(random_body_color)
                random_normal_list.append(random_body_normal)

                random_eye_mouth_pts, random_eye_mouth_color, random_eye_mouth_normal, eye_mouth_is_valid = category_sample_points(
                    eye_mouth_ids, pcl_verts, pcl_colors, pcl_normals,
                    self.opt.points_per_frame // 5, num_dim)
                if eye_mouth_is_valid:
                    random_pts_list.append(random_eye_mouth_pts)
                    random_color_list.append(random_eye_mouth_color)
                    random_normal_list.append(random_eye_mouth_normal)

                num_face_pts = self.opt.points_per_frame // 3 if eye_mouth_is_valid else self.opt.points_per_frame // 15 * 8
                random_face_pts, random_face_color, random_face_normal, _ = category_sample_points(
                    face_ids, pcl_verts, pcl_colors, pcl_normals,
                    num_face_pts, num_dim)
                split_numbers.append(self.opt.points_per_frame // 15 * 8)
                part_types.append('face')
                random_pts_list.append(random_face_pts)
                random_color_list.append(random_face_color)
                random_normal_list.append(random_face_normal)
                
                random_lhand_pts, random_lhand_color, random_lhand_normal, lhand_is_valid = category_sample_points(
                    lhand_ids, pcl_verts, pcl_colors, pcl_normals,
                    self.opt.points_per_frame // 15, num_dim)
                
                random_rhand_pts, random_rhand_color, random_rhand_normal, rhand_is_valid = category_sample_points(
                    rhand_ids, pcl_verts, pcl_colors, pcl_normals,
                    self.opt.points_per_frame // 15, num_dim)
                
                if lhand_is_valid:
                    _, close_id = kaolin.metrics.pointcloud.sided_distance(
                        random_lhand_pts, regstr_verts)
                    random_lhand_pts = regstr_verts[:, close_id[0]]
                    random_lhand_normal = regstr_normals[:,
                                                            close_id[0]]
                    split_numbers.append(self.opt.points_per_frame // 15)
                    part_types.append('lhand')
                    random_pts_list.append(random_lhand_pts)
                    random_color_list.append(random_lhand_color)
                    random_normal_list.append(random_lhand_normal)
                
                if rhand_is_valid:
                    _, close_id = kaolin.metrics.pointcloud.sided_distance(
                        random_rhand_pts, regstr_verts)
                    random_rhand_pts = regstr_verts[:, close_id[0]]
                    random_rhand_normal = regstr_normals[:,
                                                            close_id[0]]
                    split_numbers.append(self.opt.points_per_frame // 15)
                    part_types.append('rhand')
                    random_pts_list.append(random_rhand_pts)
                    random_color_list.append(random_rhand_color)
                    random_normal_list.append(random_rhand_normal)

                random_pts = torch.cat(random_pts_list, dim=1)
                random_color = torch.cat(random_color_list, dim=1)
                random_normal = torch.cat(random_normal_list, dim=1)
            else:
                num_verts = pcl_verts.shape[1]
                random_idx = torch.randint(
                    0,
                    num_verts, [1, self.opt.points_per_frame, 1],
                    device=smpl_output['smpl_verts'].device)
                random_pts = torch.gather(
                    pcl_verts, 1, random_idx.expand(-1, -1, num_dim))
                random_normal = torch.gather(
                    pcl_normals, 1, random_idx.expand(-1, -1, num_dim))
                random_color = torch.gather(
                    pcl_colors, 1, random_idx.expand(-1, -1, num_dim))
                part_types = ['full'] if self.model_type == 'smplx' else ['body']
                split_numbers = [self.opt.points_per_frame]

            sampled_pts = self.sampler.get_points(random_pts, 0)
            pts_surf.append(sampled_pts)
            normal_gt.append(random_normal)
            color_gt.append(random_color)

        data['pts_surf'] = torch.cat(pts_surf, dim=0)
        data['color_gt'] = torch.cat(color_gt, dim=0)
        data['normal_gt'] = torch.cat(normal_gt, dim=0)
        data['split_numbers'] = split_numbers
        data['part_types'] = part_types
        return data


''' Customized collate function to deal with different sizes of vertices/faces '''


def collate_fn(batch):

    data = {}
    regstr_verts, regstr_faces, regstr_normals, pcl_verts, pcl_colors, \
        pcl_normals, smpl_params, smpl_thetas, smpl_exps = \
            [],[],[],[],[],[],[],[],[]

    for item in batch:
        regstr_verts.append(item['regstr_verts'])
        regstr_faces.append(item['regstr_faces'])
        regstr_normals.append(item['regstr_normals'])
        smpl_params.append(item['smpl_params'])
        smpl_thetas.append(item['smpl_thetas'])
        if item['smpl_exps'] is not None:
            smpl_exps.append(item['smpl_exps'])
        pcl_verts.append(item['pcl_verts'])
        pcl_colors.append(item['pcl_colors'])
        pcl_normals.append(item['pcl_normals'])

    data['regstr_verts'] = torch.stack(regstr_verts)
    data['regstr_faces'] = torch.stack(regstr_faces)
    data['regstr_normals'] = torch.stack(regstr_normals)
    data['smpl_params'] = torch.stack(smpl_params)
    data['smpl_thetas'] = torch.stack(smpl_thetas)
    data['smpl_exps'] = torch.stack(smpl_exps) if len(smpl_exps) > 0 else None
    data['pcl_verts'] = pcl_verts
    data['pcl_colors'] = pcl_colors
    data['pcl_normals'] = pcl_normals

    return data


class XHumansDataModule(pl.LightningDataModule):

    def __init__(self, opt, model_type, **kwargs):
        super().__init__()
        self.opt = opt
        self.model_type = model_type

    def setup(self, stage=None):

        if stage == 'fit':
            self.dataset_train = XHumansDataSet(
                dataset_path=self.opt.dataset_path,
                model_type=self.model_type,
                use_pca=self.opt.use_pca,
                num_pca_comps=self.opt.num_pca_comps,
                flat_hand_mean=self.opt.flat_hand_mean,
                mode='train')

        self.dataset_val = XHumansDataSet(dataset_path=self.opt.dataset_path,
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