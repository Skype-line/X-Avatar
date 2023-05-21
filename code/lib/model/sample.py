import torch


class PointOnBones:
    def __init__(self, bone_ids):
        self.bone_ids = bone_ids

    def get_points(self, joints, num_per_bone=5):
        """Sample points on bones in canonical space.

        Args:
            joints (tensor): joint positions to define the bone positions. shape: [B, J, D]
            num_per_bone (int, optional): number of sample points on each bone. Defaults to 5.

        Returns:
            samples (tensor): sampled points in canoncial space. shape: [B, ?, 3]
            probs (tensor): ground truth occupancy for samples (all 1). shape: [B, ?]
        """

        num_batch, _, _ = joints.shape

        samples = []

        for bone_id in self.bone_ids:

            if bone_id[0] < 0 or bone_id[1] < 0:
                continue

            bone_dir = joints[:, bone_id[1]] - joints[:, bone_id[0]]

            scalars = (
                torch.linspace(0, 1, steps=num_per_bone, device=joints.device)
                .unsqueeze(0)
                .expand(num_batch, -1)
            )
            scalars = (
                scalars + torch.randn((num_batch, num_per_bone), device=joints.device) * 0.1
            ).clamp_(0, 1)

            samples.append(
                joints[:, bone_id[0]].unsqueeze(1).expand(-1, scalars.shape[-1], -1)
                + torch.einsum("bn,bi->bni", scalars, bone_dir) # b: num_batch, n: num_per_bone, i: 3-dim
            )

        samples = torch.cat(samples, dim=1)

        probs = torch.ones((num_batch, samples.shape[1]), device=joints.device)

        return samples, probs

    def get_joints(self, joints):
        """Sample joints in canonical space.

        Args:
            joints (tensor): joint positions to define the bone positions. shape: [B, J, D]

        Returns:
            samples (tensor): sampled points in canoncial space. shape: [B, ?, 3]
            weights (tensor): ground truth skinning weights for samples (all 1). shape: [B, ?, J]
        """
        num_batch, num_joints, _ = joints.shape

        samples = []
        weights = []

        for k in range(num_joints):
            samples.append(joints[:, k])
            weight = torch.zeros((num_batch, num_joints), device=joints.device)
            weight[:, k] = 1
            weights.append(weight)

        for bone_id in self.bone_ids:

            if bone_id[0] < 0 or bone_id[1] < 0:
                continue

            samples.append(joints[:, bone_id[1]])

            weight = torch.zeros((num_batch, num_joints), device=joints.device)
            weight[:, bone_id[0]] = 1
            weights.append(weight)

        samples = torch.stack(samples, dim=1)
        weights = torch.stack(weights, dim=1)

        return samples, weights


class PointInSpace:
    def __init__(self, global_sigma=1.8, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, sample_size=None):
        """Sample one point near each of the given point + 1/8 uniformly. 

        Args:
            pc_input (tensor): sampling centers. shape: [B, N, D]

        Returns:
            samples (tensor): sampled points. shape: [B, N + sample_size, D]
        """

        batch_size, pts_num, dim = pc_input.shape
        
        if sample_size is None:
            sample_size = pts_num

        sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        sample_global = (torch.rand(batch_size, sample_size, dim, device=pc_input.device)
            * (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_global, sample_local], dim=1)

        return sample
