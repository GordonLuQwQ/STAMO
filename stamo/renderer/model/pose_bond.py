# version 1
# from __future__ import annotations

# import torch
# import torch.nn as nn


# NUM_HANDS = 2
# JOINTS_PER_HAND = 21
# NUM_JOINTS = NUM_HANDS * JOINTS_PER_HAND


# class JointGaussianSplatEncoder(nn.Module):
#     """Build 42 joint-local 2.5D tokens on the DINO 14x14 patch grid.

#     As in Wan JointControl, normalized UV is first scaled to feature-grid
#     coordinates and a Gaussian is evaluated directly against ``arange(14)``.
#     The normalized Gaussian pools the full-image DINO memory into one local
#     feature. Direct camera depth and a learned joint ID are then added before
#     the final LayerNorm:

#         H_joint = LN(R_local + E_depth(z_camera) + E_joint_id).
#     """

#     def __init__(
#         self,
#         output_dim: int,
#         input_size: int = 224,
#         patch_size: int = 16,
#         patch_sigma: float = 0.75,
#         depth_mean: float = 0.0,
#         depth_scale: float = 1.0,
#         depth_hidden_dim: int = 64,
#     ) -> None:
#         super().__init__()
#         self.output_dim = int(output_dim)
#         self.num_joints = NUM_JOINTS
#         self.input_size = int(input_size)
#         self.patch_size = int(patch_size)
#         self.patch_sigma = float(patch_sigma)
#         self.depth_mean = float(depth_mean)
#         self.depth_scale = float(depth_scale)
#         self.depth_hidden_dim = int(depth_hidden_dim)
#         if self.input_size != 224:
#             raise ValueError(
#                 "The joint-only bond path is defined on one 224x224 canvas; "
#                 f"got input_size={self.input_size}"
#             )
#         if self.patch_size <= 0 or self.input_size % self.patch_size:
#             raise ValueError("patch_size must divide the 224px input size")
#         if self.patch_sigma <= 0:
#             raise ValueError("joint patch_sigma must be positive")
#         if self.depth_scale <= 0:
#             raise ValueError("joint depth_scale must be positive")
#         if self.depth_hidden_dim <= 0:
#             raise ValueError("joint depth_hidden_dim must be positive")

#         self.grid_size = self.input_size // self.patch_size
#         self.num_patches = self.grid_size * self.grid_size
#         self.joint_id_embed = nn.Embedding(self.num_joints, self.output_dim)
#         nn.init.normal_(self.joint_id_embed.weight, std=0.02)
#         self.depth_encoder = nn.Sequential(
#             nn.Linear(1, self.depth_hidden_dim),
#             nn.SiLU(),
#             nn.Linear(self.depth_hidden_dim, self.output_dim, bias=False),
#         )
#         self.norm = nn.LayerNorm(self.output_dim)

#     def generate_heatmaps(
#         self,
#         uv_coords: torch.Tensor,
#         sigma=None,
#     ) -> torch.Tensor:

#         if uv_coords.ndim != 3 or uv_coords.shape[1:] != (self.num_joints, 2):
#             raise ValueError(
#                 "uv_coords must have shape [B,42,2], got "
#                 f"{tuple(uv_coords.shape)}"
#             )
#         batch_size, num_joints, _ = uv_coords.shape
#         u, v = uv_coords[..., 0], uv_coords[..., 1]
#         y_range = torch.arange(
#             self.grid_size,
#             device=uv_coords.device,
#             dtype=uv_coords.dtype,
#         )
#         x_range = torch.arange(
#             self.grid_size,
#             device=uv_coords.device,
#             dtype=uv_coords.dtype,
#         )
#         grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
#         distance_sq = (
#             (
#                 grid_x.view(1, 1, self.grid_size, self.grid_size)
#                 - u.view(batch_size, num_joints, 1, 1)
#             ).square()
#             + (
#                 grid_y.view(1, 1, self.grid_size, self.grid_size)
#                 - v.view(batch_size, num_joints, 1, 1)
#             ).square()
#         )
#         active_sigma = self.patch_sigma if sigma is None else float(sigma)
#         if active_sigma <= 0:
#             raise ValueError("heatmap sigma must be positive")
#         return torch.exp(-distance_sq / (2.0 * active_sigma**2))

#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         pose_uvz: torch.Tensor,
#     ) -> torch.Tensor:
#         expected_pose_shape = (
#             NUM_HANDS,
#             JOINTS_PER_HAND,
#             3,
#         )
#         if pose_uvz.ndim != 4 or tuple(pose_uvz.shape[1:]) != expected_pose_shape:
#             raise ValueError(
#                 "pose_uvz must have shape [B,2,21,3], got "
#                 f"{tuple(pose_uvz.shape)}"
#             )
#         expected_image_shape = (
#             pose_uvz.shape[0],
#             self.num_patches,
#             self.output_dim,
#         )
#         if tuple(image_embeddings.shape) != expected_image_shape:
#             raise ValueError(
#                 "image_embeddings must be the 196 non-CLS DINO tokens: "
#                 f"got={tuple(image_embeddings.shape)}, "
#                 f"expected={expected_image_shape}"
#             )

#         id_feat = self.joint_id_embed(
#             torch.arange(self.num_joints, device=pose_uvz.device)
#         )
#         joint_features = id_feat.unsqueeze(0).expand(
#             pose_uvz.shape[0], -1, -1
#         )
#         normalized_uv = pose_uvz.reshape(
#             pose_uvz.shape[0], self.num_joints, 3
#         )[..., :2].float()
#         uv_feature_grid = normalized_uv * float(self.grid_size)
#         patch_gaussians_fp32 = self.generate_heatmaps(uv_feature_grid)
#         expected_joint_shape = (
#             pose_uvz.shape[0],
#             self.num_joints,
#             self.output_dim,
#         )
#         if tuple(joint_features.shape) != expected_joint_shape:
#             raise RuntimeError(
#                 "Unexpected joint-ID feature shape: "
#                 f"got={tuple(joint_features.shape)}, expected={expected_joint_shape}"
#             )
#         expected_patch_shape = (
#             pose_uvz.shape[0],
#             self.num_joints,
#             self.grid_size,
#             self.grid_size,
#         )
#         if tuple(patch_gaussians_fp32.shape) != expected_patch_shape:
#             raise RuntimeError(
#                 "Unexpected patch-centre heatmap shape: "
#                 f"got={tuple(patch_gaussians_fp32.shape)}, "
#                 f"expected={expected_patch_shape}"
#             )

#         flat_heatmaps = patch_gaussians_fp32.flatten(2)
#         normalized_heatmaps = flat_heatmaps / (
#             flat_heatmaps.sum(dim=-1, keepdim=True) + 1e-6
#         )
#         local_features = torch.bmm(
#             normalized_heatmaps.to(dtype=image_embeddings.dtype),
#             image_embeddings,
#         )

#         direct_depth = pose_uvz.reshape(
#             pose_uvz.shape[0], self.num_joints, 3
#         )[..., 2:3].float()
#         depth_input = (direct_depth - self.depth_mean) / self.depth_scale
#         depth_features = self.depth_encoder(
#             depth_input.to(dtype=joint_features.dtype)
#         )

#         token_dtype = local_features.dtype
#         return self.norm(
#             local_features
#             + depth_features.to(dtype=token_dtype)
#             + joint_features.to(dtype=token_dtype)
#         )


# class PoseConditionProjector(nn.Module):
#     """Concatenate 196 RGB and 42 joint tokens for one Q-Former memory."""

#     def __init__(
#         self,
#         projector: nn.Module,
#         token_dim: int,
#         image_size: int,
#         joint_patch_sigma: float = 0.75,
#         joint_depth_mean: float = 0.0,
#         joint_depth_scale: float = 1.0,
#         joint_depth_hidden_dim: int = 64,
#     ) -> None:
#         super().__init__()
#         self.projector = projector
#         # ``token_dim`` is the DINO/pose-memory width.  The learned queries may
#         # use a different hidden width, so keep the two contracts separate.
#         self.token_dim = int(token_dim)
#         self.pose_encoder = JointGaussianSplatEncoder(
#             output_dim=self.token_dim,
#             input_size=image_size,
#             patch_sigma=joint_patch_sigma,
#             depth_mean=joint_depth_mean,
#             depth_scale=joint_depth_scale,
#             depth_hidden_dim=joint_depth_hidden_dim,
#         )

#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         pose_uvz: torch.Tensor,
#     ) -> torch.Tensor:
#         if image_embeddings.ndim != 3:
#             raise ValueError(
#                 "DINO embeddings must have shape [B,N,C], got "
#                 f"{tuple(image_embeddings.shape)}"
#             )
#         if image_embeddings.shape[-1] != self.token_dim:
#             raise ValueError(
#                 f"DINO token dim is {image_embeddings.shape[-1]}, expected "
#                 f"{self.token_dim}"
#             )

#         pose_uvz = pose_uvz.to(
#             device=image_embeddings.device,
#             dtype=next(self.pose_encoder.parameters()).dtype,
#         )
#         pose_tokens = self.pose_encoder(
#             image_embeddings,
#             pose_uvz,
#         ).to(image_embeddings.dtype)
#         if pose_tokens.shape[1:] != (NUM_JOINTS, self.token_dim):
#             raise RuntimeError(
#                 "Unexpected pose memory shape: "
#                 f"got={tuple(pose_tokens.shape)}, "
#                 f"expected=[B,{NUM_JOINTS},{self.token_dim}]"
#             )

#         memory = torch.cat((image_embeddings, pose_tokens), dim=1)
#         expected_memory_shape = (
#             image_embeddings.shape[0],
#             image_embeddings.shape[1] + NUM_JOINTS,
#             self.token_dim,
#         )
#         if tuple(memory.shape) != expected_memory_shape:
#             raise RuntimeError(
#                 "Unexpected concatenated Q-Former memory shape: "
#                 f"got={tuple(memory.shape)}, expected={expected_memory_shape}"
#             )
#         return self.projector(memory)


# version 2
from __future__ import annotations

import torch
import torch.nn as nn


NUM_HANDS = 2
JOINTS_PER_HAND = 21
NUM_JOINTS = NUM_HANDS * JOINTS_PER_HAND


class JointGaussianSplatEncoder(nn.Module):
    """Build a JointControl-style spatial hand map on the FLUX 14x14 grid.

    As in Wan JointControl, normalized UV is first scaled to feature-grid
    coordinates and a Gaussian is evaluated directly against ``arange(14)``.
    The normalized Gaussian pools the full-image DINO memory into one local
    feature. Direct camera depth and a learned joint ID are then added before
    the final LayerNorm:

        H_joint = LN(R_local + E_depth(z_camera) + E_joint_id).

    The existing joint features are projected to ``hand_output_channels`` and
    splatted with the same Gaussian maps.  The result is a dense
    ``[B,hand_output_channels,14,14]`` map aligned one-to-one with FLUX image
    tokens.
    """

    def __init__(
        self,
        output_dim: int,
        input_size: int = 224,
        patch_size: int = 16,
        patch_sigma: float = 0.75,
        depth_mean: float = 0.0,
        depth_scale: float = 1.0,
        depth_hidden_dim: int = 64,
        hand_output_channels: int = 16,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_joints = NUM_JOINTS
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.patch_sigma = float(patch_sigma)
        self.depth_mean = float(depth_mean)
        self.depth_scale = float(depth_scale)
        self.depth_hidden_dim = int(depth_hidden_dim)
        self.hand_output_channels = int(hand_output_channels)
        if self.input_size != 224:
            raise ValueError(
                "The joint-only bond path is defined on one 224x224 canvas; "
                f"got input_size={self.input_size}"
            )
        if self.patch_size <= 0 or self.input_size % self.patch_size:
            raise ValueError("patch_size must divide the 224px input size")
        if self.patch_sigma <= 0:
            raise ValueError("joint patch_sigma must be positive")
        if self.depth_scale <= 0:
            raise ValueError("joint depth_scale must be positive")
        if self.depth_hidden_dim <= 0:
            raise ValueError("joint depth_hidden_dim must be positive")
        if self.hand_output_channels <= 0:
            raise ValueError("hand_output_channels must be positive")

        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.joint_id_embed = nn.Embedding(self.num_joints, self.output_dim)
        nn.init.normal_(self.joint_id_embed.weight, std=0.02)
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, self.depth_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.depth_hidden_dim, self.output_dim, bias=False),
        )
        self.norm = nn.LayerNorm(self.output_dim)
        self.hand_output_projection = nn.Linear(
            self.output_dim,
            self.hand_output_channels,
        )

    def generate_heatmaps(
        self,
        uv_coords: torch.Tensor,
        sigma=None,
    ) -> torch.Tensor:

        if uv_coords.ndim != 3 or uv_coords.shape[1:] != (self.num_joints, 2):
            raise ValueError(
                "uv_coords must have shape [B,42,2], got "
                f"{tuple(uv_coords.shape)}"
            )
        batch_size, num_joints, _ = uv_coords.shape
        u, v = uv_coords[..., 0], uv_coords[..., 1]
        y_range = torch.arange(
            self.grid_size,
            device=uv_coords.device,
            dtype=uv_coords.dtype,
        )
        x_range = torch.arange(
            self.grid_size,
            device=uv_coords.device,
            dtype=uv_coords.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
        distance_sq = (
            (
                grid_x.view(1, 1, self.grid_size, self.grid_size)
                - u.view(batch_size, num_joints, 1, 1)
            ).square()
            + (
                grid_y.view(1, 1, self.grid_size, self.grid_size)
                - v.view(batch_size, num_joints, 1, 1)
            ).square()
        )
        active_sigma = self.patch_sigma if sigma is None else float(sigma)
        if active_sigma <= 0:
            raise ValueError("heatmap sigma must be positive")
        return torch.exp(-distance_sq / (2.0 * active_sigma**2))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        pose_uvz: torch.Tensor,
    ) -> torch.Tensor:
        expected_pose_shape = (
            NUM_HANDS,
            JOINTS_PER_HAND,
            3,
        )
        if pose_uvz.ndim != 4 or tuple(pose_uvz.shape[1:]) != expected_pose_shape:
            raise ValueError(
                "pose_uvz must have shape [B,2,21,3], got "
                f"{tuple(pose_uvz.shape)}"
            )
        expected_image_shape = (
            pose_uvz.shape[0],
            self.num_patches,
            self.output_dim,
        )
        if tuple(image_embeddings.shape) != expected_image_shape:
            raise ValueError(
                "image_embeddings must be the 196 non-CLS DINO tokens: "
                f"got={tuple(image_embeddings.shape)}, "
                f"expected={expected_image_shape}"
            )

        id_feat = self.joint_id_embed(
            torch.arange(self.num_joints, device=pose_uvz.device)
        )
        joint_features = id_feat.unsqueeze(0).expand(
            pose_uvz.shape[0], -1, -1
        )
        normalized_uv = pose_uvz.reshape(
            pose_uvz.shape[0], self.num_joints, 3
        )[..., :2].float()
        uv_feature_grid = normalized_uv * float(self.grid_size)
        patch_gaussians_fp32 = self.generate_heatmaps(uv_feature_grid)
        expected_joint_shape = (
            pose_uvz.shape[0],
            self.num_joints,
            self.output_dim,
        )
        if tuple(joint_features.shape) != expected_joint_shape:
            raise RuntimeError(
                "Unexpected joint-ID feature shape: "
                f"got={tuple(joint_features.shape)}, expected={expected_joint_shape}"
            )
        expected_patch_shape = (
            pose_uvz.shape[0],
            self.num_joints,
            self.grid_size,
            self.grid_size,
        )
        if tuple(patch_gaussians_fp32.shape) != expected_patch_shape:
            raise RuntimeError(
                "Unexpected patch-centre heatmap shape: "
                f"got={tuple(patch_gaussians_fp32.shape)}, "
                f"expected={expected_patch_shape}"
            )

        flat_heatmaps = patch_gaussians_fp32.flatten(2)
        normalized_heatmaps = flat_heatmaps / (
            flat_heatmaps.sum(dim=-1, keepdim=True) + 1e-6
        )
        local_features = torch.bmm(
            normalized_heatmaps.to(dtype=image_embeddings.dtype),
            image_embeddings,
        )

        direct_depth = pose_uvz.reshape(
            pose_uvz.shape[0], self.num_joints, 3
        )[..., 2:3].float()
        depth_input = (direct_depth - self.depth_mean) / self.depth_scale
        depth_features = self.depth_encoder(
            depth_input.to(dtype=joint_features.dtype)
        )

        token_dtype = local_features.dtype
        joint_tokens = self.norm(
            local_features
            + depth_features.to(dtype=token_dtype)
            + joint_features.to(dtype=token_dtype)
        )
        joint_channels = self.hand_output_projection(joint_tokens)
        hand_map = torch.einsum(
            "bjhw,bjc->bchw",
            patch_gaussians_fp32.to(dtype=joint_channels.dtype),
            joint_channels,
        )
        expected_hand_shape = (
            pose_uvz.shape[0],
            self.hand_output_channels,
            self.grid_size,
            self.grid_size,
        )
        if tuple(hand_map.shape) != expected_hand_shape:
            raise RuntimeError(
                "Unexpected spatial hand-map shape: "
                f"got={tuple(hand_map.shape)}, expected={expected_hand_shape}"
            )
        return hand_map


class PoseConditionProjector(nn.Module):
    """Keep RGB Q-Former context and spatial hand conditioning separate."""

    def __init__(
        self,
        projector: nn.Module,
        token_dim: int,
        image_size: int,
        joint_patch_sigma: float = 0.75,
        joint_depth_mean: float = 0.0,
        joint_depth_scale: float = 1.0,
        joint_depth_hidden_dim: int = 64,
        hand_output_channels: int = 16,
    ) -> None:
        super().__init__()
        self.projector = projector
        # ``token_dim`` is the DINO/pose-memory width.  The learned queries may
        # use a different hidden width, so keep the two contracts separate.
        self.token_dim = int(token_dim)
        self.hand_output_channels = int(hand_output_channels)
        self.pose_encoder = JointGaussianSplatEncoder(
            output_dim=self.token_dim,
            input_size=image_size,
            patch_sigma=joint_patch_sigma,
            depth_mean=joint_depth_mean,
            depth_scale=joint_depth_scale,
            depth_hidden_dim=joint_depth_hidden_dim,
            hand_output_channels=self.hand_output_channels,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        pose_uvz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_embeddings.ndim != 3:
            raise ValueError(
                "DINO embeddings must have shape [B,N,C], got "
                f"{tuple(image_embeddings.shape)}"
            )
        if image_embeddings.shape[-1] != self.token_dim:
            raise ValueError(
                f"DINO token dim is {image_embeddings.shape[-1]}, expected "
                f"{self.token_dim}"
            )

        pose_uvz = pose_uvz.to(
            device=image_embeddings.device,
            dtype=next(self.pose_encoder.parameters()).dtype,
        )
        hand_map = self.pose_encoder(
            image_embeddings,
            pose_uvz,
        ).to(image_embeddings.dtype)
        expected_hand_shape = (
            image_embeddings.shape[0],
            self.hand_output_channels,
            self.pose_encoder.grid_size,
            self.pose_encoder.grid_size,
        )
        if tuple(hand_map.shape) != expected_hand_shape:
            raise RuntimeError(
                "Unexpected spatial hand-map shape: "
                f"got={tuple(hand_map.shape)}, expected={expected_hand_shape}"
            )
        image_condition = self.projector(image_embeddings)
        return image_condition, hand_map
