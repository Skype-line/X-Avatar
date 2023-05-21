import torch
import numpy as np
from lib.xavatar_basic_model import XAVATARModel


class XAVATARSCANModel(XAVATARModel):

    def training_step(self, data, data_idx):
        # Data prep
        if self.data_processor is not None:
            data = self.data_processor.process(data)

        cond = torch.cat(
            [data['smpl_thetas'] / np.pi, data['smpl_exps']], dim=-1
        ) if data['smpl_exps'] is not None else data['smpl_thetas'] / np.pi

        loss = 0.0

        # shape loss
        _, shape_pd, _, _ = self.forward(data['pts_d'],
                                         data['smpl_tfs'],
                                         cond,
                                         eval_mode=False,
                                         with_normal=False,
                                         with_color=False,
                                         is_training=True,
                                         split_numbers=data['split_numbers'],
                                         part_type_list=data['part_types'])
        
        loss_shape = torch.nn.functional.binary_cross_entropy_with_logits(
                shape_pd, data['shape_gt'])

        self.log('shape_loss', loss_shape)
        loss += loss_shape

        if 'pts_surf' in data:
            _, _, color_pd, normal_pd = self.forward(data['pts_surf'],
                                                    data['smpl_tfs'],
                                                    cond,
                                                    eval_mode=True,
                                                    with_normal=self.opt.lambda_vert_normal>0,
                                                    with_color=self.train_color,
                                                    is_training=True,
                                                    split_numbers=data['split_numbers'],
                                                    part_type_list=data['part_types'])
            
            # normal loss
            if self.model_type == 'smplx' and self.opt.lambda_vert_normal > 0:
                loss_normal = torch.nn.functional.mse_loss(
                    normal_pd[:, data['split_numbers'][0]:data['normal_gt'].shape[1], :],
                    data['normal_gt'][:, data['split_numbers'][0]:, :])

                self.log('normal_loss', loss_normal)
                loss += self.opt.lambda_vert_normal * loss_normal

            # color loss
            if self.current_epoch >= self.opt.nepochs_pretrain and self.train_color:
                loss_color = torch.nn.functional.l1_loss(color_pd, data['color_gt'])
                
                self.log('color_loss', loss_color)
                loss += self.opt.lambda_vert_color * loss_color
            
        num_batch = data['pts_d'].shape[0]
        
        # Bone occupancy loss
        if self.opt.lambda_bone_shape > 0:
            pts_c, shape_gt = self.sampler_bone.get_points(
                self.smpl_server.joints_c.expand(num_batch, -1, -1))

            shape_pd = self.shape_net(pts_c, cond)

            loss_bone_shape = torch.nn.functional.binary_cross_entropy_with_logits(
                shape_pd, shape_gt.unsqueeze(-1))

            loss = loss + self.opt.lambda_bone_shape * loss_bone_shape
            self.log('bone_shape_loss', loss_bone_shape)

        
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
        
        return loss



