import torch
from torch import einsum
import torch.nn.functional as F

from lib.model.broyden import broyden
from lib.model.network import ImplicitNetwork
from lib.model.helpers import hierarchical_softmax


class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, opt, model_type, d_out, **kwargs):
        super().__init__()
        
        self.opt = opt
        self.softmax_mode = self.opt.softmax_mode
        self.model_type = model_type
        
        self.lbs_network = ImplicitNetwork(**self.opt.network, d_out=d_out)

        self.soft_blend = 20

    def forward(self, xd, cond, tfs, part_type='full', eval_mode=False):
        """Given deformed point return its caonical correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            part_type (str): part type, one of ['body', 'face', 'lhand', 'rhand', 'full']
            eval_mode (bool): whether to compute correction term for implicit differentiation
        
        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
         
        if part_type == 'body':
            init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19]
        elif part_type == 'face':
            init_bones = [12, 15, 22]
        elif part_type == 'lhand':
            init_bones = [20] + list(range(25, 40))
        elif part_type == 'rhand':
            init_bones = [21] + list(range(40, 55))
        elif part_type == 'full':
            init_bones = [0, 1, 2, 4, 5, 15, 16, 17, 18, 19] + list(range(20,55))
        elif part_type == 'hand':
            init_bones = list(range(16))
            
        xc_init = self.init(xd, tfs, init_bones)

        xc_opt, others = self.search(xd,
                                     xc_init,
                                     cond,
                                     tfs,
                                     eval_mode=eval_mode)

        if eval_mode:
            return xc_opt, others

        # compute correction term for implicit differentiation during training

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()

        # reshape to [B,?,D] for network query
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_opt = xc_opt.reshape((n_batch, n_point * n_init, n_dim))

        xd_opt = self.forward_skinning(xc_opt, cond, tfs)

        grad_inv = self.gradient(xc_opt, cond, tfs).inverse()

        correction = xd_opt - xd_opt.detach()
        correction = einsum("bnij,bnj->bni", -grad_inv.detach(), correction)

        # trick for implicit diff with autodiff:
        # xc = xc_opt + 0 and xc' = correction'
        xc = xc_opt + correction

        # reshape back to [B,N,I,D]
        xc = xc.reshape(xc_init.shape)

        return xc, others

    def init(self, xd, tfs, init_bones):
        """Transform xd to canonical space for initialization

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            init_bones (list): list of bone indices for initialization
        
        Returns:
            xc_init (tensor): gradients. shape: [B, N, I, D]
        """
        n_batch, n_point, _ = xd.shape
        _, n_joint, _, _ = tfs.shape

        xc_init = []
        for i in init_bones:
            w = torch.zeros((n_batch, n_point, n_joint), device=xd.device)
            w[:, :, i] = 1
            xc_init.append(skinning(xd, w, tfs, inverse=True))

        xc_init = torch.stack(xc_init, dim=2)

        return xc_init

    def search(self, xd, xc_init, cond, tfs, eval_mode=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            
        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        # reshape to [B,?,D] for other functions
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_init = xc_init.reshape(n_batch, n_point * n_init, n_dim)
        xd_tgt = xd.repeat_interleave(n_init, dim=1)

        # compute init jacobians
        if not eval_mode:
            J_inv_init = self.gradient(xc_init, cond, tfs).inverse()
        else:
            w = self.query_weights(xc_init, cond, mask=None)
            J_inv_init = einsum("bpn,bnij->bpij", w,
                                tfs)[:, :, :3, :3].inverse()

        # reshape init to [?,D,...] for boryden
        xc_init = xc_init.reshape(-1, n_dim, 1)
        J_inv_init = J_inv_init.flatten(0, 1)

        # construct function for root finding
        def _func(xc_opt, mask=None):
            # reshape to [B,?,D] for other functions
            xc_opt = xc_opt.reshape(n_batch, n_point * n_init, n_dim)

            # import ipdb
            # ipdb.set_trace()

            xd_opt = self.forward_skinning(xc_opt, cond, tfs, mask=mask)
            error = xd_opt - xd_tgt
            # reshape to [?,D,1] for boryden
            error = error.flatten(0, 1)[mask].unsqueeze(-1)
            return error

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, xc_init, J_inv_init)

        # reshape back to [B,N,I,D]
        xc_opt = result["result"].reshape(n_batch, n_point, n_init, n_dim)
        result["valid_ids"] = result["valid_ids"].reshape(
            n_batch, n_point, n_init)

        return xc_opt, result

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            mask (tensor, optional): valid indices. shape: [B, N]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        w = self.query_weights(xc, cond, mask=mask)
        xd = skinning(xc, w, tfs, inverse=False)
        return xd

    def query_weights(self, xc, cond, mask=None):
        """Get skinning weights in canonical space

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): valid indices. shape: [B, N]

        Returns:
            w (tensor): skinning weights. shape: [B, N, J]
        """
        w = self.lbs_network(xc, cond, mask)
        w = self.soft_blend * w

        if self.softmax_mode == "hierarchical":
            w = hierarchical_softmax(w, self.model_type)
        else:
            w = F.softmax(w, dim=-1)
        
        return w

    def gradient(self, xc, cond, tfs):
        """Get gradients df/dx

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            grad (tensor): gradients. shape: [B, N, D, D]
        """
        xc.requires_grad_(True)

        xd = self.forward_skinning(xc, cond, tfs)

        grads = []
        for i in range(xd.shape[-1]):
            d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
            d_out[:, :, i] = 1
            grad = torch.autograd.grad(
                outputs=xd,
                inputs=xc,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)

        return torch.stack(grads, dim=-2)


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = einsum("bpn,bnij->bpij", w, tfs)
        x_h = einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    return x_h[:, :, :3]
