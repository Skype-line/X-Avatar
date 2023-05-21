import torch
import hydra
import numpy as np
from lib.smplx.body_models import SMPL, SMPLH, SMPLX, MANO


class SMPLServer(torch.nn.Module):

    def __init__(self, gender='male', betas=None, v_template=None):
        super().__init__()

        self.model = SMPL(
            model_path=hydra.utils.to_absolute_path('lib/smplx/smpl_model'),
            gender=gender,
            batch_size=1,
            use_hands=False,
            use_feet_keypoints=False,
            dtype=torch.float32).cuda()

        self.bone_parents = self.model.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(len(self.bone_parents)):
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template != None:
            self.v_template = torch.tensor(v_template).float().cuda()
        else:
            self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 86), dtype=torch.float32).cuda()
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if self.betas is not None and self.v_template is None:
            param_canonical[0, -10:] = self.betas

        output = self.forward(param_canonical, absolute=True)
        
        self.verts_c = output['smpl_verts']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()

    def forward(self, model_params, absolute=False):
        """return SMPL output from params

        Args:
            model_params : smpl parameters. shape: [B, 86].
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 

        Returns:
            smpl_verts: vertices. shape: [B, 6890. 3]
            smpl_tfs: bone transformations. shape: [B, 24, 4, 4]
            smpl_jnts: joint positions. shape: [B, 25, 3]
        """

        output = {}

        scale, transl, global_orient, body_pose, betas = torch.split(model_params,
                                                   [1, 3, 3, 69, 10],
                                                   dim=1)

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        smpl_output = self.model.forward(betas=betas,
                                         transl=torch.zeros_like(transl),
                                         body_pose=body_pose,
                                         global_orient=global_orient,
                                         return_verts=True,
                                         return_full_pose=True,
                                         v_template=self.v_template)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1)
        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)

        output['smpl_tfs'] = tf_mats

        return output


class SMPLHServer(torch.nn.Module):

    def __init__(self,
                 gender='male',
                 betas=None,
                 v_template=None,
                 use_pca=False,
                 num_pca_comps=12,
                 flat_hand_mean=False):
        super().__init__()

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps if use_pca else 45
        self.flat_hand_mean = flat_hand_mean
        self.model = SMPLH(
            model_path=hydra.utils.to_absolute_path('lib/smplx/smplh_model'),
            gender=gender,
            batch_size=1,
            num_pca_comps=num_pca_comps,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            dtype=torch.float32).cuda()

        self.bone_parents = self.model.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(len(self.bone_parents)):
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template != None:
            self.v_template = torch.tensor(v_template).float().cuda()
        else:
            self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 80 + 2 * self.num_pca_comps),
                                          dtype=torch.float32).cuda()
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if flat_hand_mean == False:
            param_canonical[
                0, 70:115] = -self.model.left_hand_mean.unsqueeze(0)
            param_canonical[
                0, 115:160] = -self.model.right_hand_mean.unsqueeze(0)
        if self.betas is not None and self.v_template is None:
            param_canonical[0, -10:] = self.betas

        output = self.forward(param_canonical, absolute=True)
        
        self.verts_c = output['smpl_verts']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()

    def forward(self, model_params, absolute=False):
        """return SMPLH output from params

        Args:
            model_params : smplx parameters. shape: [B, 80 + 2 * num_pca_comps].
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 

        Returns:
            smpl_verts: vertices. shape: [B, 6890. 3]
            smpl_tfs: bone transformations. shape: [B, 52, 4, 4]
            smpl_jnts: joint positions. shape: [B, 53, 3]
        """

        output = {}

        scale, transl, global_orient, body_pose, left_hand_pose, right_hand_pose, betas = torch.split(
                model_params, [1, 3, 3, 63, self.num_pca_comps, self.num_pca_comps, 10], dim=1)

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        smpl_output = self.model.forward(betas=betas,
                                         transl=torch.zeros_like(transl),
                                         global_orient=global_orient,
                                         body_pose=body_pose,
                                         left_hand_pose=left_hand_pose,
                                         right_hand_pose=right_hand_pose,
                                         return_verts=True,
                                         return_full_pose=True,
                                         v_template=self.v_template)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1)

        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)

        output['smpl_tfs'] = tf_mats

        return output


class SMPLXServer(torch.nn.Module):

    def __init__(
            self,
            gender='male',
            betas=None,
            v_template=None,
            use_pca=False,
            num_pca_comps=12,
            flat_hand_mean=False):
        super().__init__()

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps if use_pca else 45
        self.flat_hand_mean = flat_hand_mean
        self.model = SMPLX(
            model_path=hydra.utils.to_absolute_path('lib/smplx/smplx_model'),
            gender=gender,
            batch_size=1,
            num_pca_comps=num_pca_comps,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            dtype=torch.float32).cuda()

        self.bone_parents = self.model.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(len(self.bone_parents)):
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template != None:
            self.v_template = torch.tensor(v_template).float().cuda()
        else:
            self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 99 + 2 * self.num_pca_comps),
                                          dtype=torch.float32).cuda()
        
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        param_canonical[0, 76 + 2 * self.num_pca_comps] = 0.2
        
        if flat_hand_mean == False:
            param_canonical[
                0, 70:115] = -self.model.left_hand_mean.unsqueeze(0)
            param_canonical[
                0, 115:160] = -self.model.right_hand_mean.unsqueeze(0)
        if self.betas is not None and self.v_template is None:
            param_canonical[0, -20:-10] = self.betas

        output = self.forward(param_canonical, absolute=True)

        self.verts_c = output['smpl_verts']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()
        self.pose_feature_c = output['smpl_pose_feature']
        self.expr_c = output['smpl_expr']


    def forward(self, model_params, absolute=False):
        """return SMPLX output from params

        Args:
            model_params : smplx parameters. shape: [B, 99+2*num_pca_comps].
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 

        Returns:
            smpl_verts: vertices. shape: [B, 10475. 3]
            smpl_tfs: bone transformations. shape: [B, 55, 4, 4]
            smpl_jnts: joint positions. shape: [B, 56, 3]
        """

        output = {}
        scale, transl, global_orient, body_pose, left_hand_pose, right_hand_pose, leye_pose, reye_pose,\
            jaw_pose, betas, expression = torch.split(
            model_params, [1, 3, 3, 63, self.num_pca_comps, self.num_pca_comps, 3, 3, 3, 10, 10], dim=1)
        
        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)
            expression = torch.zeros_like(expression)

        smpl_output = self.model.forward(betas=betas,
                                         expression=expression,
                                         transl=torch.zeros_like(transl),
                                         global_orient=global_orient,
                                         body_pose=body_pose,
                                         left_hand_pose=left_hand_pose,
                                         right_hand_pose=right_hand_pose,
                                         leye_pose=leye_pose,
                                         reye_pose=reye_pose,
                                         jaw_pose=jaw_pose,
                                         return_verts=True,
                                         return_full_pose=True,
                                         v_template=self.v_template)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1)

        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)

        output['smpl_tfs'] = tf_mats
        output['smpl_pose_feature'] = smpl_output.pose_feature.clone()
        output['smpl_expr'] = smpl_output.expression.clone()

        return output


class MANOServer(torch.nn.Module):

    def __init__(
            self,
            betas=None,
            v_template=None,
            use_pca=False,
            num_pca_comps=6,
            flat_hand_mean=False):
        super().__init__()

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps if use_pca else 45
        self.flat_hand_mean = flat_hand_mean
        self.model = MANO(
            model_path=hydra.utils.to_absolute_path('lib/smplx/mano_model'),
            batch_size=1,
            num_pca_comps=num_pca_comps,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            dtype=torch.float32).cuda()

        self.bone_parents = self.model.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(len(self.bone_parents)):
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template != None:
            self.v_template = torch.tensor(v_template).float().cuda()
        else:
            self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 17 + self.num_pca_comps),
                                          dtype=torch.float32).cuda()
        param_canonical[0, 0] = 1
        if flat_hand_mean == False:
            param_canonical[0, 7:52] = -self.model.hand_mean.unsqueeze(0)

        if self.betas is not None and self.v_template is None:
            param_canonical[0, -10:] = self.betas
            
        output = self.forward(param_canonical, absolute=True)
        self.verts_c = output['smpl_verts']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()

    def forward(self, model_params, absolute=False):
        """return SMPL output from params

        Args:
            model_params : mano parameters. shape: [B, 17 + num_pca_comps].
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 

        Returns:
            smpl_verts: vertices. shape: [B, 778. 3]
            smpl_tfs: bone transformations. shape: [B, 16, 4, 4]
            smpl_jnts: joint positions. shape: [B, 17, 3]
        """

        output = {}

        scale, transl, global_orient, hand_pose, betas = torch.split(
            model_params, [1, 3, 3, self.num_pca_comps, 10], dim=1)
        
        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        smpl_output = self.model.forward(betas=betas,
                                         transl=torch.zeros_like(transl),
                                         global_orient=global_orient,
                                         hand_pose=hand_pose,
                                         return_verts=True,
                                         return_full_pose=True,
                                         v_template=self.v_template)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1)

        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)

        output['smpl_tfs'] = tf_mats

        return output