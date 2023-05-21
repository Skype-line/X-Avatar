import wandb
import hydra
import os
import os.path as osp
from pathlib import Path
import numpy as np
import pickle as pkl
import torch
import pytorch_lightning as pl

from lib.model.smpl import SMPLServer, SMPLHServer, SMPLXServer, MANOServer
from lib.model.sample import PointOnBones, PointInSpace
from lib.model.network import ImplicitNetwork
from lib.utils.meshing import generate_mesh
from lib.model.helpers import masked_softmax
from lib.model.deformer import ForwardDeformer, skinning
from lib.utils.render import render_trimesh, render_joint, weights2colors


class XAVATARModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()
        self.opt = opt
        self.model_type = opt.model_type
        self.train_color = opt.colornet.train_color

        gender = str(meta_info['gender']) if 'gender' in meta_info else None
        betas = meta_info['betas'] if 'betas' in meta_info else None
        use_pca = meta_info['use_pca'] if 'use_pca' in meta_info else None
        num_pca_comps = meta_info['num_pca_comps'] if 'num_pca_comps' in meta_info else None
        flat_hand_mean = meta_info['flat_hand_mean'] if 'flat_hand_mean' in meta_info else None
        
        if self.model_type == 'smplx':
            verts_label_id = pkl.load(open(hydra.utils.to_absolute_path(
                    'lib/smplx/smplx_model/non_watertight_{}_vertex_labels.pkl'
                    .format(gender)), 'rb'), encoding='latin1')
            self.face_hand_id = torch.Tensor(
                np.concatenate([verts_label_id['left_hand'], verts_label_id['right_hand'],
                    verts_label_id['face']])).long()
            self.body_id = torch.Tensor(verts_label_id['body']).long()
        else:
            self.face_hand_id = None
            self.body_id = None

        # load SMPL(-H/X) server, get corresponding conditional input dimension 
        # for the shapenet and output dimension for the lbs weight network
        if self.model_type == 'smplx':
            self.smpl_server = SMPLXServer(gender=gender,
                                           betas=betas,
                                           use_pca=use_pca,
                                           num_pca_comps=num_pca_comps,
                                           flat_hand_mean=flat_hand_mean)
            cond_dim = 73  # body pose + expression
            d_out = 59 if opt.deformer.softmax_mode == 'hierarchical' else 55
        elif self.model_type == 'smplh':
            self.smpl_server = SMPLHServer(gender=gender,
                                           betas=betas,
                                           use_pca=use_pca,
                                           num_pca_comps=num_pca_comps,
                                           flat_hand_mean=flat_hand_mean)
            cond_dim = 63  # body pose
            d_out = 55 if opt.deformer.softmax_mode == 'hierarchical' else 52
        elif self.model_type == 'smpl':
            self.smpl_server = SMPLServer(gender=gender, betas=betas)
            cond_dim = 69  # body pose
            d_out = 25 if opt.deformer.softmax_mode == 'hierarchical' else 24
        elif self.model_type == 'mano':
            self.smpl_server = MANOServer(betas=betas,
                                          use_pca=use_pca,
                                          num_pca_comps=num_pca_comps,
                                          flat_hand_mean=flat_hand_mean)
            cond_dim = num_pca_comps if use_pca else 45
            d_out = 17 if opt.deformer.softmax_mode == 'hierarchical' else 16
        else:
            raise NotImplementedError

        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)
        self.sampler = PointInSpace(global_sigma=0.5, local_sigma=0.05)

        self.data_processor = data_processor
        
        # load shape network, color network and deformer
        self.shape_net = ImplicitNetwork(**opt.network, cond_dim=cond_dim)
        self.color_net = ImplicitNetwork(**opt.colornet)
        self.deformer = ForwardDeformer(opt.deformer,
                                        model_type=self.model_type,
                                        d_out=d_out)

    def configure_optimizers(self):
        params = [
            {
                'params': self.shape_net.parameters(),
            },
            {
                'params': self.deformer.parameters()
            },
        ]
        if self.train_color:
            params.append({
                'params': self.color_net.parameters()
            })
        optimizer = torch.optim.Adam(params, lr=self.opt.optim.lr)
        return optimizer

    def forward(self,
                pts_d,
                smpl_tfs,
                cond,
                eval_mode=True,
                with_normal=False,
                with_color=False,
                is_training=False,
                split_numbers=None,
                part_type_list=None):

        accum_shape, pts_c_list, mask_list, split_num_list, index_list = [], [], [], [], []

        num_points = pts_d.shape[1]
        if is_training:
            if self.opt.network.representation == 'occ':
                split_numbers = [
                    num_points - torch.sum(split_numbers[1:]),
                    split_numbers[1], split_numbers[2], split_numbers[3]
                ] if self.data_processor.opt.category_sample else [num_points]
              # in other cases, split_numbers comes from the dataloader
        else:
            # split to prevent out of memory
            batch_points = 6000
            if self.model_type == 'smplx':
                part_type_list = ['full'] * (num_points // batch_points + 1)
            elif self.model_type == 'mano':
                part_type_list = ['hand'] * (num_points // batch_points + 1)
            else:
                part_type_list = ['body'] * (num_points // batch_points + 1)
            
            split_numbers = batch_points

        for part_type, pts_d_split in zip(
                part_type_list, torch.split(pts_d, split_numbers, dim=1)):
            # compute canonical correspondences
            pts_c, intermediates = self.deformer(pts_d_split,
                                                None,
                                                smpl_tfs,
                                                part_type,
                                                eval_mode=eval_mode)

            mask_list.append(intermediates['valid_ids'])
            num_batch, num_point, num_init, num_dim = pts_c.shape
            pts_c = pts_c.reshape(num_batch, num_point * num_init, num_dim)
            pts_c_list.append(pts_c)
            split_num_list.append(pts_c.shape[1])
        pts_c = torch.cat(pts_c_list, dim=1)
        # query shape in canonical space
        shape_pd = self.shape_net(pts_c, cond, return_feature=False)
        shape_pd_list = torch.split(shape_pd, split_num_list, dim=1)

        for shape_pd, mask in zip(shape_pd_list, mask_list):
            num_batch, num_point, num_init = mask.shape
            shape_pd = shape_pd.reshape(num_batch, num_point, num_init)
            # select points with the highest occupancy probablities
            if eval_mode:
                mode = 'max' if self.opt.network.representation == 'occ' else 'min'
                shape_pd, index = masked_softmax(shape_pd,
                                                mask,
                                                dim=-1,
                                                mode=mode)
            else:
                shape_pd, index = masked_softmax(
                    shape_pd,
                    mask,
                    dim=-1,
                    mode=self.opt.network.softmax_mode,
                    soft_blend=self.opt.soft_blend)

            accum_shape.append(shape_pd)
            index_list.append(index)

        shape_pd = torch.cat(accum_shape, 1)

        # query normal in deformed space and color in canonical space
        if with_normal or with_color:
            new_pts_c_list = []
            torch.set_grad_enabled(True)
            for pts_c, index, mask in zip(pts_c_list, index_list, mask_list):
                num_batch, num_point, num_init = mask.shape
                pts_c = pts_c.reshape(num_batch, num_point, num_init, num_dim)
                pts_c = torch.gather(
                    pts_c, 2,
                    index.unsqueeze(-1).expand(num_batch, num_point, 1,
                                                num_dim))[:, :, 0, :]
                new_pts_c_list.append(pts_c)
            pts_c = torch.cat(new_pts_c_list, dim=1)

            normal_pd = self.extract_normal(pts_c,
                                            cond,
                                            smpl_tfs,
                                            is_training=is_training)
            if self.opt.network.representation == 'sdf':
                normal_pd *= -1

            if with_color:
                _, pose_cond, feature = self.shape_net(pts_c,
                                                        cond,
                                                        return_feature=True)

                color_input = torch.cat([pts_c, normal_pd], dim=-1)
                color_cond = torch.cat([pose_cond, feature], dim=-1)
                color_pd = self.color_net(color_input, color_cond)

                color_pd = torch.sigmoid(color_pd)
                return pts_c, shape_pd, color_pd, normal_pd
            else:
                return pts_c, shape_pd, None, normal_pd
        else:
            return pts_c, shape_pd, None, None

    def training_step(self, data, data_idx):
        """
        Training step. Differs depending on the data type (Scan/RGB-D).
        Implemented in the subclasses: xavatar_scan_model, xavatar_rgbd_model.
        """
        pass

    def validation_step(self, data, data_idx):
        if self.data_processor is not None:
            data = self.data_processor.process(data)

        with torch.no_grad():
            if data_idx == 0:
                plot_res = self.plot(data, res=128)
                img_all = plot_res['img_all']
                self.logger.experiment.log({"vis": [wandb.Image(img_all)]})

                smpl_verts = self.smpl_server.verts_c

                _, mesh_cano = self.extract_mesh(
                    smpl_verts,
                    data['smpl_tfs'][[0]],
                    data['smpl_thetas'][[0]],
                    data['smpl_exps'][[0]]
                    if data['smpl_exps'] is not None else None,
                    res_up=3,
                    canonical=True,
                    with_weights=True,
                    fast_mode=False)

                mesh_def, _ = self.extract_mesh(
                    smpl_verts,
                    data['smpl_tfs'][[0]],
                    data['smpl_thetas'][[0]],
                    data['smpl_exps'][[0]]
                    if data['smpl_exps'] is not None else None,
                    res_up=4,
                    canonical=False,
                    with_weights=False,
                    fast_mode=True)

                # save one canonical mesh and one deformed mesh for evaluation
                save_dir = osp.join(
                    str(Path(self.logger.save_dir).parent.parent.parent),
                    'meshes_val')
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)
                _ = mesh_cano.export(
                    osp.join(save_dir, 'epoch{}_cano.ply'.format(
                        self.current_epoch)))
                _ = mesh_def.export(
                    osp.join(save_dir,
                                 'epoch{}_def.ply'.format(self.current_epoch)))
        return None
    
    def plot(self, data, res=128, verbose=True, fast_mode=False):

        res_up = np.log2(res // 32)

        if verbose:
            # get canonical meshes visualized in RGB color / lbs weights. 
            surf_pred_cano, surf_vis_lbs = self.extract_mesh(
                self.smpl_server.verts_c,
                data['smpl_tfs'][[0]],
                data['smpl_thetas'][[0]],
                data['smpl_exps'][[0]]
                if data['smpl_exps'] is not None else None,
                res_up=res_up,
                canonical=True,
                with_weights=True)

            # get deformed meshes visualized in RGB color
            surf_pred_def, _ = self.extract_mesh(
                data['smpl_verts'][[0]],
                data['smpl_tfs'][[0]],
                data['smpl_thetas'][[0]],
                data['smpl_exps'][[0]]
                if data['smpl_exps'] is not None else None,
                res_up=res_up,
                canonical=False,
                with_weights=False)

            img_pred_cano = render_trimesh(surf_pred_cano)
            img_vis_lbs = render_trimesh(surf_vis_lbs)
            img_pred_def = render_trimesh(surf_pred_def)

            img_pred_cano[-512:, :, :] = img_vis_lbs[1024:-512, :, :]
            img_joint = render_joint(data['smpl_jnts'].data.cpu().numpy()[0],
                                     self.smpl_server.bone_ids)

            img_pred_def[1024:-512, :, :3] = img_joint
            img_pred_def[1024:-512, :, -1] = 255

            results = {
                'img_all': np.concatenate([img_pred_cano, img_pred_def],
                                          axis=1),
                'mesh_cano': surf_pred_cano,
                'mesh_lbs': surf_vis_lbs,
                'mesh_def': surf_pred_def
            }
        else:
            smpl_verts = self.smpl_server.verts_c if fast_mode else data[
                'smpl_verts'][[0]]

            surf_pred_def, _ = self.extract_mesh(
                smpl_verts,
                data['smpl_tfs'][[0]],
                data['smpl_thetas'][[0]],
                data['smpl_exps'][[0]]
                if data['smpl_exps'] is not None else None,
                res_up=res_up,
                canonical=False,
                with_weights=False,
                fast_mode=fast_mode)

            img_pred_def = render_trimesh(surf_pred_def, mode='t')
            results = {'img_all': img_pred_def, 'mesh_def': surf_pred_def}

        return results

    def extract_mesh(self,
                     smpl_verts,
                     smpl_tfs,
                     smpl_thetas,
                     smpl_exps,
                     canonical=False,
                     with_weights=False,
                     res_up=2,
                     fast_mode=False):
        '''
        In fast mode, we extract canonical mesh and then forward skin it to posed space.
        This is faster as it bypasses root finding.
        However, it's not deforming the continuous field, but the discrete mesh.
        '''
        cond = torch.cat(
            [smpl_thetas / np.pi, smpl_exps],
            dim=-1) if smpl_exps is not None else smpl_thetas / np.pi

        if canonical or fast_mode:
            def cano_forward(xc, cond, shape_net, color_net, with_color=False):
                torch.set_grad_enabled(True)
                batch_points = 6000

                accum_shape, accum_color, accum_normal = [], [], []
                for xc_split in torch.split(xc, batch_points, dim=1):
                    shape_pd, pose_cond, feature = shape_net(
                        xc_split, cond, return_feature=True)
                    accum_shape.append(shape_pd)
                    if with_color:
                        normal_pd = self.extract_normal(xc_split,
                                                        cond,
                                                        smpl_tfs,
                                                        is_training=False)
                        if self.opt.network.representation == 'sdf':
                            normal_pd *= -1
                        
                        color_input = torch.cat([xc_split, normal_pd], dim=-1)
                        color_cond = torch.cat([pose_cond, feature], dim=-1)
                        color_pd = color_net(color_input, color_cond)
                        color_pd = torch.sigmoid(color_pd)
                        
                        accum_color.append(color_pd)
                        accum_normal.append(normal_pd)

                shape_pd = torch.cat(accum_shape, 1)
                color_pd = torch.cat(accum_color, 1) if len(
                    accum_color) > 0 else None  #(num_batch, num_point, 3)
                normal_pd = torch.cat(accum_normal,
                                      1) if len(accum_normal) > 0 else None
                return xc, shape_pd, color_pd, normal_pd

            func = lambda x: cano_forward(
                x, cond, self.shape_net, self.color_net, with_color=False)
            color_func = lambda x: cano_forward(
                x, cond, self.shape_net, self.color_net, with_color=True)
        else:
            func = lambda x: self.forward(x,
                                          smpl_tfs,
                                          cond,
                                          eval_mode=True,
                                          with_normal=False,
                                          with_color=False,
                                          is_training=False)
            color_func = lambda x: self.forward(x,
                                                smpl_tfs,
                                                cond,
                                                eval_mode=True,
                                                with_normal=False,
                                                with_color=True,
                                                is_training=False)
       
        mesh = generate_mesh(
            func,
            smpl_verts.squeeze(0),
            res_up=res_up,
            shape_represenation=self.opt.network.representation)

        if self.train_color:
            _, _, vertex_colors, _ = color_func(
                torch.tensor(mesh.vertices).float().unsqueeze(0).cuda())
            vertex_colors = vertex_colors.squeeze(0).detach().cpu().numpy() * 255
            mesh.visual.vertex_colors = vertex_colors

        if fast_mode:
            verts = torch.tensor(mesh.vertices).type_as(smpl_verts)
            weights = self.deformer.query_weights(verts[None],
                                                  None).clamp(0, 1)[0]

            verts_mesh_deformed = skinning(verts.unsqueeze(0),
                                           weights.unsqueeze(0),
                                           smpl_tfs).data.cpu().numpy()[0]
            mesh.vertices = verts_mesh_deformed

        if with_weights:
            verts = torch.tensor(mesh.vertices).cuda().float()
            weights = self.deformer.query_weights(verts[None],
                                                  None).clamp(0, 1)[0]
            mesh_vis_lbs = mesh.copy()
            mesh_vis_lbs.visual.vertex_colors = weights2colors(
                weights.data.cpu().numpy(), model_type=self.model_type)
            return mesh, mesh_vis_lbs
        else:
            return mesh, None

    def extract_normal(self, x_c, cond, tfs, is_training):
        x_c.requires_grad_(True)
        x_d = self.deformer.forward_skinning(x_c, None, tfs)
        num_dim = x_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(x_d,
                                     requires_grad=False,
                                     device=x_d.device)
            d_out[..., i] = 1
            grad = torch.autograd.grad(
                outputs=x_d,
                inputs=x_c,
                grad_outputs=d_out,
                create_graph=is_training,
                retain_graph=True if i < num_dim - 1 else is_training,
                only_inputs=True)[0]
            grads.append(grad)

        grads = torch.stack(grads, dim=-2)
        num_batch, num_points, _, _ = grads.shape
        grads_inv = grads.reshape(-1, num_dim, num_dim).inverse().reshape(
            num_batch, num_points, num_dim, num_dim)

        output = self.shape_net(x_c, cond)
        gradient_c = gradient(x_c, output, is_training)

        gradient_d = torch.einsum('bki,bkij->bkj', gradient_c, grads_inv)

        gradient_d = -torch.nn.functional.normalize(
            gradient_d, dim=2, eps=1e-6)

        return gradient_d


def gradient(inputs, outputs, is_training=True):
    d_points = torch.ones_like(outputs,
                               requires_grad=False,
                               device=outputs.device)
    points_grad = torch.autograd.grad(outputs=outputs,
                                      inputs=inputs,
                                      grad_outputs=d_points,
                                      create_graph=is_training,
                                      retain_graph=is_training,
                                      only_inputs=True)[0]
    return points_grad