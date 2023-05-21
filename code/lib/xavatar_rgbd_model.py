import torch
import numpy as np
from lib.xavatar_basic_model import XAVATARModel, gradient


class XAVATARRGBDModel(XAVATARModel):

    def training_step(self, data, data_idx):
        # Data prep
        if self.data_processor is not None:
            data = self.data_processor.process(data)

        cond = torch.cat(
            [data['smpl_thetas'] / np.pi, data['smpl_exps']], dim=-1
        ) if data['smpl_exps'] is not None else data['smpl_thetas'] / np.pi

        # shape loss
        _, shape_pd, color_pd, normal_pd = self.forward(data['pts_surf'],
                                                            data['smpl_tfs'],
                                                            cond,
                                                            eval_mode=True,
                                                            with_normal=self.opt.lambda_vert_normal>0,
                                                            with_color=self.train_color,
                                                            is_training=True,
                                                            split_numbers=data['split_numbers'],
                                                            part_type_list=data['part_types'])

        loss = 0.0
        loss_shape = shape_pd.abs().mean()

        self.log('shape_loss', loss_shape)
        loss += loss_shape
        
        # normal loss
        loss_normal = ((normal_pd - data['normal_gt']).abs()).norm(
                        2, dim=-1).mean()
        
        self.log('normal_loss', loss_normal)
        loss += self.opt.lambda_vert_normal * loss_normal

        # color loss
        if self.current_epoch >= self.opt.nepochs_pretrain and self.train_color:
            loss_color = torch.nn.functional.l1_loss(color_pd, data['color_gt'])
            
            self.log('color_loss', loss_color)
            loss += self.opt.lambda_vert_color * loss_color
        

        num_batch = data['pts_surf'].shape[0]

        if self.opt.lambda_lbs_w > 0:
            # Joint weight loss
            pts_c, w_gt = self.sampler_bone.get_joints(
                self.smpl_server.joints_c.expand(num_batch, -1, -1))

            w_pd = self.deformer.query_weights(pts_c, None)
            loss_bone_w = torch.nn.functional.mse_loss(w_pd, w_gt)
            self.log('bone_w_loss', loss_bone_w)
            loss += self.opt.lambda_lbs_w * loss_bone_w

            # Surface weight loss
            if self.face_hand_id is not None:
                w_surf_gt = self.smpl_server.model.lbs_weights
                w_surf_pd = self.deformer.query_weights(
                    self.smpl_server.verts_c, None).squeeze(0)
                loss_surf_w = torch.nn.functional.mse_loss(
                    w_surf_pd[self.face_hand_id],
                    w_surf_gt[self.face_hand_id])
                self.log('surf_w_loss', loss_surf_w)
                loss += self.opt.lambda_lbs_w * loss_surf_w
        
        # Eikonal loss, ref: https://arxiv.org/pdf/2002.10099.pdf
        if self.opt.lambda_eikonal > 0:
            if self.model_type == 'smplx':
                smpl_verts_c = self.smpl_server.verts_c[:, self.body_id]
            else:
                smpl_verts_c = self.smpl_server.verts_c
            indices = torch.randperm(smpl_verts_c.shape[1])[:6000].cuda()
            verts_c = torch.index_select(smpl_verts_c, 1, indices)
            sample = self.sampler.get_points(verts_c, sample_size=0)
            sample.requires_grad_(True)
            output = self.shape_net(sample.repeat(num_batch, 1, 1),
                                    cond)[:1]
            grad_theta = gradient(sample, output, is_training=True)

            eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
            loss = loss + self.opt.lambda_eikonal * eikonal_loss
            self.log('eikonal_loss', eikonal_loss)

            
        return loss



