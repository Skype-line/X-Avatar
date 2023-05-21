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


def sample_surface_wnormalcolor(mesh, count, id_mask=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    area = mesh.area_faces.copy()
    
    if id_mask is not None:
        mask = np.ones(area.shape[0], dtype=bool)
        mask[id_mask] = False
        area[mask] = 0
        
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # do the same for normal
    normals = mesh.vertex_normals.view(np.ndarray)[mesh.faces]
    nml_origins = normals[:, 0]
    nml_vectors = normals[:, 1:]#.copy()
    nml_vectors -= np.tile(nml_origins, (1, 2)).reshape((-1, 2, 3))

    colors = mesh.visual.vertex_colors[:,:3].astype(np.float32)
    colors = colors / 255.0
    colors = colors.view(np.ndarray)[mesh.faces]
    clr_origins = colors[:, 0]
    clr_vectors = colors[:, 1:]#.copy()
    clr_vectors -= np.tile(clr_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # pull the vectors for the faces we are going to sample from
    nml_origins = nml_origins[face_index]
    nml_vectors = nml_vectors[face_index]

    clr_origins = clr_origins[face_index]
    clr_vectors = clr_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    sample_normal = (nml_vectors * random_lengths).sum(axis=1)
    sample_color = (clr_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = sample_normal + nml_origins

    colors = sample_color + clr_origins

    return samples, normals, colors, face_index


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
        self.scan_list = sorted(
            glob.glob(osp.join(dataset_path, mode, '*', 'meshes_ply', '*.ply')))
        
        assert len(self.regstr_list) == len(self.smplx_params_list) == len(self.scan_list)
        
        if mode == 'test': # only use one sample for testing
            self.regstr_list = self.regstr_list[0:1]
            self.smplx_params_list = self.smplx_params_list[0:1]
            self.scan_list = self.scan_list[0:1]
        
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
                scan_mesh = trimesh.load(self.scan_list[index])
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
        data['regstr_verts'] = regstr_verts
        data['regstr_faces'] = regstr_faces
        
        # load 3D human scan
        scan_verts = scan_mesh.vertices - transl
        scan_verts = torch.tensor(scan_verts).float()
        scan_faces = scan_mesh.faces.astype(np.int32)
        scan_faces = torch.tensor(scan_faces).long()
        scan_colors = torch.tensor(
            scan_mesh.visual.vertex_colors[:, :3]).float() / 255.0
        scan_normals = torch.tensor(
            scan_mesh.vertex_normals.copy()).float()
        data['scan_mesh'] = scan_mesh
        data['scan_verts'] = scan_verts
        data['scan_faces'] = scan_faces
        data['scan_colors'] = scan_colors
        data['scan_normals'] = scan_normals

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
        data['smpl_transl'] = torch.tensor(transl).float()
        return data

    def __len__(self):
        return len(self.smplx_params_list)


class XHumansDataProcessor():
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
                
                # set part splits
                self.split_numbers = [self.opt.points_per_frame // 3 
                                      + self.opt.points_per_frame // 12, # in space + body
                                    self.opt.points_per_frame // 3, # face
                                    self.opt.points_per_frame // 8, # left hand
                                    self.opt.points_per_frame // 8, # right hand
                                    ]
                self.part_types = ['body', 'face', 'lhand', 'rhand']
            else:
                raise NotImplementedError
        else:
            self.split_numbers = [self.opt.points_per_frame]
            self.part_types = ['full'] if model_type == 'smplx' else ['body']

    def process(self, data):

        smpl_output = self.smpl_server(data['smpl_params'], absolute=False)
        data.update(smpl_output)

        pts_d_batch, shape_gt_batch, pts_surf_batch, color_gt_batch, normal_gt_batch = [], [], [], [], []

        for i in range(len(data['regstr_verts'])):
            regstr_verts = data['regstr_verts'][i].expand(1, -1, -1)
            regstr_faces = data['regstr_faces'][i]
            scan_mesh = data['scan_mesh'][i]
            scan_verts = data['scan_verts'][i].expand(1, -1, -1)
            scan_faces = data['scan_faces'][i]
            scan_colors = data['scan_colors'][i].expand(1, -1, -1)
            scan_normals = data['scan_normals'][i].expand(1, -1, -1)
            transl = data['smpl_transl'][i].expand(1, -1)
            num_dim = data['regstr_verts'][i].shape[1]

            if self.opt.category_sample:
                random_idx = list()
                
                # predict the part type of each point
                _, close_id = kaolin.metrics.pointcloud.sided_distance(
                    scan_verts, regstr_verts)
                scan_label_vector = self.regstr_label_vector[close_id[0]]
                body_ids = torch.where(
                    scan_label_vector ==
                    0)[0] if self.body_ids is not None else None
                face_ids = torch.where(
                    scan_label_vector ==
                    1)[0] if self.face_ids is not None else None
                if body_ids is not None:
                    random_body_idx = body_ids[torch.randint(
                        0,
                        body_ids.shape[0],
                        [1, self.split_numbers[0] * 4 // 5, 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_body_idx)
                if face_ids is not None:
                    random_face_idx = face_ids[torch.randint(
                        0,
                        face_ids.shape[0],
                        [1, self.split_numbers[1] // 2, 1],
                        device=smpl_output['smpl_verts'].device)]
                    random_idx.append(random_face_idx)
                random_idx = torch.cat(random_idx, dim=1)
            else:
                num_verts = data['scan_verts'][i].shape[0]
                random_idx = torch.randint(
                    0,
                    num_verts, [1, self.opt.points_per_frame, 1],
                    device=smpl_output['smpl_verts'].device)

            pts_d_data = list()
            pts_surf_data = list()
            color_gt_data = list()
            normal_gt_data = list()

            random_pts = torch.gather(scan_verts, 1, random_idx.expand(-1, -1, num_dim))
            sampled_pts = self.sampler.get_points(random_pts, 200)
            
            pts_surf_data.append(random_pts)
            color_gt_data.append(torch.gather(scan_colors, 1, random_idx.expand(-1, -1, num_dim)))
            normal_gt_data.append(torch.gather(scan_normals, 1, random_idx.expand(-1, -1, num_dim)))
            pts_d_data.append(sampled_pts)
            
            if self.opt.category_sample:
                # sample face details from the surface
                face_vertices = kaolin.ops.mesh.index_vertices_by_faces(scan_verts, scan_faces)
                _, eye_mouth_f_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                    regstr_verts[:, self.eye_mouth_ids], face_vertices)
                eye_mouth_pts, eye_mouth_normals, eye_mouth_colors, _ = sample_surface_wnormalcolor(
                    scan_mesh, self.split_numbers[1] // 2,
                    eye_mouth_f_idx.detach().cpu().numpy()[0])
                eye_mouth_pts = (
                    torch.tensor(eye_mouth_pts).float().cuda() -
                    transl).unsqueeze(0)
                eye_mouth_normals = torch.tensor(
                    eye_mouth_normals).float().cuda().unsqueeze(0)
                eye_mouth_normals = eye_mouth_normals / torch.norm(
                    eye_mouth_normals, dim=-1, keepdim=True)
                eye_mouth_colors = torch.tensor(
                    eye_mouth_colors).float().cuda().unsqueeze(0)
                sampled_eye_mouth_pts = eye_mouth_pts + (
                    torch.randn_like(eye_mouth_pts) *
                    self.opt.sampler.local_sigma)
                pts_d_data.append(sampled_eye_mouth_pts)
                pts_surf_data.append(eye_mouth_pts)
                color_gt_data.append(eye_mouth_colors)
                normal_gt_data.append(eye_mouth_normals)

                pts_d_data = torch.cat(pts_d_data, dim=1)
                
                smpl_lhand_idx = self.lhand_ids[torch.randint(
                    0,
                    self.lhand_ids.shape[0],
                    [1, self.split_numbers[2], 1],
                    device=smpl_output['smpl_verts'].device)]
                smpl_rhand_idx = self.rhand_ids[torch.randint(
                    0,
                    self.rhand_ids.shape[0],
                    [1, self.split_numbers[3], 1],
                    device=smpl_output['smpl_verts'].device)]
                smpl_hand_idx = torch.cat(
                    [smpl_lhand_idx, smpl_rhand_idx], dim=1)

                random_hand_pts = torch.gather(
                    regstr_verts, 1,
                    smpl_hand_idx.expand(-1, -1, num_dim))
                sampled_hand_pts = random_hand_pts + (
                    torch.randn_like(random_hand_pts) *
                    self.opt.sampler.local_sigma)

                pts_d_batch.append(
                    torch.cat([pts_d_data, sampled_hand_pts], dim=1))
                
                shape_gt_batch.append(
                    torch.cat([
                        kaolin.ops.mesh.check_sign(
                            scan_verts, scan_faces,
                            pts_d_data).float().unsqueeze(-1),
                        kaolin.ops.mesh.check_sign(
                            regstr_verts, regstr_faces,
                            sampled_hand_pts).float().unsqueeze(-1)
                    ], dim=1))
                
                pts_surf_data.append(random_hand_pts)

                _, close_id = kaolin.metrics.pointcloud.sided_distance(
                    random_hand_pts, scan_verts)
                color_gt_data.append(scan_colors[:, close_id[0]])            
            else:
                pts_d_data = torch.cat(pts_d_data, dim=1)
                pts_d_batch.append(pts_d_data)
                shape_gt_batch.append(
                    kaolin.ops.mesh.check_sign(
                        scan_verts, scan_faces,
                        pts_d_data).float().unsqueeze(-1))

            pts_surf_data = torch.cat(pts_surf_data, dim=1)
            color_gt_data = torch.cat(color_gt_data, dim=1)
            normal_gt_data = torch.cat(normal_gt_data, dim=1)
            pts_surf_batch.append(pts_surf_data)
            color_gt_batch.append(color_gt_data)
            normal_gt_batch.append(normal_gt_data)

        data['pts_d'] = torch.cat(pts_d_batch, dim=0)
        data['shape_gt'] = torch.cat(shape_gt_batch, dim=0)
        data['pts_surf'] = torch.cat(pts_surf_batch, dim=0)
        data['color_gt'] = torch.cat(color_gt_batch, dim=0)
        data['normal_gt'] = torch.cat(normal_gt_batch, dim=0)
        data['split_numbers'] = torch.tensor(self.split_numbers).cuda()
        data['part_types'] = self.part_types
        return data


''' Customized collate function to deal with different sizes of vertices/faces '''
def collate_fn(batch):

    data = {}
    regstr_verts, regstr_faces, smpl_params, smpl_thetas, smpl_transl, smpl_exps, \
        scan_mesh, scan_verts, scan_faces, scan_colors, scan_normals= \
            [],[],[],[],[],[],[],[],[],[],[]

    for item in batch:
        regstr_verts.append(item['regstr_verts'])
        regstr_faces.append(item['regstr_faces'])
        smpl_params.append(item['smpl_params'])
        smpl_thetas.append(item['smpl_thetas'])
        smpl_transl.append(item['smpl_transl'])
        if item['smpl_exps'] is not None:
            smpl_exps.append(item['smpl_exps'])
        scan_mesh.append(item['scan_mesh'])
        scan_verts.append(item['scan_verts'])
        scan_faces.append(item['scan_faces'])
        scan_colors.append(item['scan_colors'])
        scan_normals.append(item['scan_normals'])
        
    data['regstr_verts'] = regstr_verts
    data['regstr_faces'] = regstr_faces
    data['smpl_params'] = torch.stack(smpl_params)
    data['smpl_thetas'] = torch.stack(smpl_thetas)
    data['smpl_transl'] = torch.stack(smpl_transl)
    data['smpl_exps'] = torch.stack(smpl_exps) if len(smpl_exps) > 0 else None
    data['scan_mesh'] = scan_mesh
    data['scan_verts'] = scan_verts
    data['scan_faces'] = scan_faces
    data['scan_colors'] = scan_colors
    data['scan_normals'] = scan_normals
    
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