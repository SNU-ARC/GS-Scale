import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from torch.autograd import Function

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import threading
import psutil
PHYSICAL_CORES = psutil.cpu_count(logical=False)
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from datetime import datetime, timedelta
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from intel_extension_for_pytorch.optim._functional import adam_step, adam
from cpu_adam import adam_for_next, update_counter, adam_deferred_update, adam_for_next_with_counter, calculate_update_ids
from gsplat.rendering import rasterization
from gsplat.rendering import frustum_culling_gpu
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.utils import save_ply


class FusedCPUAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, maximize=False, foreach=None, fused=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=foreach)
        super(FusedCPUAdam, self).__init__(params, defaults)
        self.fused = fused
        self.params_attr = {}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # Call the original AdamW step function
        return adam_step(
            self=self,
            closure=closure
        )



@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = -1
    test_after: int = -1
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Init from ply
    init_ply: bool = False

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = True
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Learning rate for position (xyz)
    position_lr: float = 1.6e-4
    # Learning rate for scaling
    scaling_lr: float = 5e-3

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def expand_bits(v):
    """Expand bits for 10-bit values: abcdefghij -> a0b0c0...j0"""
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v

def morton_encode(xyz, scale=1024):
    """
    xyz: (N, 3) tensor, each coordinate ∈ [0, 1]
    Returns: (N,) Morton code
    """
    xyz_scaled = torch.clamp((xyz * scale).long(), 0, scale - 1)  # scale to integers
    x, y, z = xyz_scaled.unbind(-1)
    x = expand_bits(x)
    y = expand_bits(y)
    z = expand_bits(z)
    morton = (x << 2) | (y << 1) | z
    return morton

def morton_sort_indices(xyz, scale=1024):
    """
    Returns indices that would sort the input by Morton order.
    """
    morton = morton_encode(xyz, scale)
    return torch.argsort(morton)

def normalize_xyz(xyz):
    min_xyz = xyz.min(dim=0)[0]
    max_xyz = xyz.max(dim=0)[0]
    return (xyz - min_xyz) / (max_xyz - min_xyz + 1e-6)

def create_splats_with_optimizers_in_hostmem(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    position_lr: float = 1.6e-4,
    scaling_lr: float = 5e-3,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    points_norm = normalize_xyz(points)
    sorted_idx = morton_sort_indices(points_norm)
    points = points[sorted_idx].contiguous()
    rgbs = rgbs[sorted_idx].contiguous()
    scales = scales[sorted_idx].contiguous()

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    quats = quats[sorted_idx].contiguous()
    opacities = opacities[sorted_idx].contiguous()

    params_gpu = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), position_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scaling_lr),
        ("quats", torch.nn.Parameter(quats), 1e-3),
    ]

    params_cpu = [
        # name, value, lr
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colorsN = torch.zeros((N, (sh_degree + 1) ** 2 - 1, 3))  # [N, K, 3]
        colors0 = rgb_to_sh(rgbs).unsqueeze(1)
        params_cpu.append(("sh0", torch.nn.Parameter(colors0), 2.5e-3))
        params_cpu.append(("shN", torch.nn.Parameter(colorsN), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params_cpu.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params_cpu.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats_gpu = torch.nn.ParameterDict({n: v for n, v, _ in params_gpu}).to(device) 
    splats_cpu = torch.nn.ParameterDict({n: v for n, v, _ in params_cpu})   # Parameters except opacities and sh are in the host mem
    splats = torch.nn.ParameterDict({**splats_gpu, **splats_cpu})
    
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    else:
        optimizer_class_cpu = FusedCPUAdam
        optimizer_class_gpu = torch.optim.Adam
    optimizers_gpu = {
        name: optimizer_class_gpu(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(0.9**BS, 0.999**BS), # Grendel (ICLR'26)
        )
        for name, _, lr in params_gpu
    }
    optimizers_cpu = {
        name: optimizer_class_cpu(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(0.9**BS, 0.999**BS),
            fused=True,
#            adamw_mode=False,
        )
        for name, _, lr in params_cpu
    }

    return splats, optimizers_gpu, optimizers_cpu


def pipeline_bubble():
    print("Pipeline bubble due to densification!")
    return 1


def cpu_optimizer_parallel(
    valid_ids_cpu: Tensor,
    update_ids: Tensor,
    counter: Tensor,
    weight1: Tensor,
    weight2: Tensor, 
    weight3: Tensor,
    grad1: Tensor,
    grad2: Tensor,
    grad3: Tensor,
    opt1_1: Tensor,
    opt2_1: Tensor,
    opt3_1: Tensor,
    opt1_2: Tensor,
    opt2_2: Tensor,
    opt3_2: Tensor,
    lr1: float,
    lr2: float,
    lr3: float,
    step: float,
    beta1: float,
    beta2: float,
    eps: float
):
    with torch.no_grad():
        # CPU Optimizer with deferred update
        adam_deferred_update(weight1, grad1, opt1_1, opt1_2, update_ids, counter, step, lr1, beta1, beta2, eps) 
        adam_deferred_update(weight2, grad2, opt2_1, opt2_2, update_ids, counter, step, lr2, beta1, beta2, eps) 
        adam_deferred_update(weight3, grad3, opt3_1, opt3_2, update_ids, counter, step, lr3, beta1, beta2, eps) 

        # Update counter
        update_counter(counter, update_ids)

        # Only perform zero_grad for the valid gaussians. The rest are already zero.
        grad1[valid_ids_cpu] = 0
        grad2[valid_ids_cpu] = 0
        grad3[valid_ids_cpu] = 0

    return 1


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            test_after=cfg.test_after,
            init_ply=cfg.init_ply,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers_gpu, self.optimizers_cpu = create_splats_with_optimizers_in_hostmem(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            position_lr=cfg.position_lr,
            scaling_lr=cfg.scaling_lr,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
 
        # Densification Strategy
        #self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")


    def rasterize_splats_only(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors0: Tensor,
        colorsN: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        device = self.device
        image_ids = kwargs.pop("image_ids", None)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors0=colors0,
            colors=colorsN,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )

        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        device = self.device
        means = self.splats["means"]  # [N, 3], gpu
        quats = self.splats["quats"]  # [N, 4], gpu
        scales = torch.exp(self.splats["scales"])  # [N, 3], gpu
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,], cpu

        image_ids = kwargs.pop("image_ids", None)
        colors0 = self.splats["sh0"]    # cpu
        colorsN = self.splats["shN"]    # cpu

        # Perform frustum culling on GPU
        valid_ids = frustum_culling_gpu(means, quats, scales, torch.linalg.inv(camtoworlds), Ks, width, height) # custom kernel
        # valid_ids, _ = torch.sort(valid_ids)
        if cfg.batch_size > 1:
            valid_ids = torch.unique(valid_ids)
        valid_ids_cpu = valid_ids.to("cpu", non_blocking=False)

        means_gpu = means[valid_ids]
        quats_gpu = quats[valid_ids]
        scales_gpu = scales[valid_ids]

        # Transfer with 32MB block size
        count = valid_ids_cpu.shape[0]
        with torch.no_grad():
            colorsN_gpu = torch.empty((count, colorsN.shape[1], colorsN.shape[2]), dtype=colorsN.dtype, device=device, requires_grad=False)
            if count < 180000:
                temp = torch.empty(colorsN_gpu.shape, dtype=colorsN.dtype, pin_memory=True)
                torch.index_select(colorsN, 0, valid_ids_cpu, out=temp)
                colorsN_gpu.copy_(temp, non_blocking=True)
            else:
                nstage = count // 180000     # 180000*45*4 = 32MB
                offset = torch.arange(nstage) * (count // nstage)
                for i in range(nstage-1):
                    temp = torch.empty((count // nstage, colorsN.shape[1], colorsN.shape[2]), dtype=colorsN.dtype, pin_memory=True)
                    torch.index_select(colorsN, 0, valid_ids_cpu[offset[i]:offset[i+1]], out=temp)
                    colorsN_gpu[offset[i]:offset[i+1]].copy_(temp, non_blocking=True)
                temp = torch.empty((count - offset[nstage-1], colorsN.shape[1], colorsN.shape[2]), dtype=colorsN.dtype, pin_memory=True)
                torch.index_select(colorsN, 0, valid_ids_cpu[offset[nstage-1]:], out=temp)
                colorsN_gpu[offset[nstage-1]:].copy_(temp, non_blocking=True)
            colors0_gpu = colors0[valid_ids_cpu].detach().pin_memory().to(device, non_blocking=True)
            opacities_gpu = opacities[valid_ids_cpu].detach().pin_memory().to(device, non_blocking=True)

        opacities_gpu = opacities_gpu.requires_grad_()
        colors0_gpu = colors0_gpu.requires_grad_()
        colorsN_gpu = colorsN_gpu.requires_grad_()

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means_gpu,
            quats=quats_gpu,
            scales=scales_gpu,
            opacities=opacities_gpu,
            colors0=colors0_gpu,
            colors=colorsN_gpu,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )

        param_list = [opacities_gpu, colors0_gpu, colorsN_gpu]
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info, param_list, valid_ids_cpu


    def gpu_forward_backward(self, step, camtoworlds, Ks, valid_ids, opacities, colors0, colorsN, width, height, image_ids, masks, pixels, results):
        cfg = self.cfg
        # sh schedule
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # forward
        renders, alphas, info = self.rasterize_splats_only(
            camtoworlds=camtoworlds,
            Ks=Ks,
            means=self.splats["means"][valid_ids],
            quats=self.splats["quats"][valid_ids],
            scales=torch.exp(self.splats["scales"][valid_ids]),
            opacities=opacities,
            colors0=colors0,
            colorsN=colorsN,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB",
            masks=masks,
        )
        
        param_list = [opacities, colors0, colorsN]
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        self.cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers_gpu,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # loss
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

        loss.backward()

        # GPU Optimizer (M.S.Q Update)
        for optimizer in self.optimizers_gpu.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        results["info"] = info
        results["sh_degree_to_use"] = sh_degree_to_use
        results["l1loss"] = l1loss
        results["ssimloss"] = ssimloss
        results["loss"] = loss
        results["param_list"] = param_list

        torch.cuda.synchronize()


    def prologue(self, trainloader_iter, schedulers):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        is_densify = False
        data = next(trainloader_iter)
        camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
        height, width = pixels.shape[1:3]

        # sh schedule
        sh_degree_to_use = 0

        # forward
        renders, alphas, info, param_list, valid_ids_cpu = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB",
            masks=masks,
        )
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        self.cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers_gpu,
            state=self.strategy_state,
            step=0,
            info=info,
        )

        # loss
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

        # Move info from gpu to cpu for densification
        info["gaussian_ids"] = valid_ids_cpu[info["gaussian_ids"].to("cpu", non_blocking=False)]
        info["radii"] = info["radii"].to("cpu", non_blocking=False)

        loss.backward()

        desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
        #pbar.set_description(desc)

        # GPU Optimizer (M.S.Q Update)
        for optimizer in self.optimizers_gpu.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Only "means" apply lr schedule. "means" is on GPU.
        for scheduler in schedulers:
            scheduler.step()

        # Move info from gpu to cpu for densification
        info["means2d"].data = info["means2d"].data.to("cpu", non_blocking=False)
        info["means2d"].grad = info["means2d"].grad.to("cpu", non_blocking=False)

        # Update grad2d state for later densification
        if isinstance(self.cfg.strategy, DefaultStrategy):
            is_densify = self.cfg.strategy.step_post_backward_only_update(
                params=self.splats,
                state=self.strategy_state,
                step=0,
                info=info,
                packed=cfg.packed,
            )
        else:
            assert_never(self.cfg.strategy)

        # Device to Host
        # Manually backpropagate grads from gpu to cpu. Backprop for means, quats, and scales are done via autograd.
        opacities_cpu_grad = torch.empty_like(param_list[0].data, device="cpu", pin_memory=True)
        opacities_grad = param_list[0].grad * param_list[0].data * (1 - param_list[0].data)     # derivative of sigmoid
        opacities_cpu_grad.copy_(opacities_grad, non_blocking=False)

        colors0_cpu_grad = torch.empty_like(param_list[1].data, device="cpu", pin_memory=True)
        colors0_grad = param_list[1].grad
        colors0_cpu_grad.copy_(colors0_grad, non_blocking=False)

        colorsN_cpu_grad = torch.empty_like(param_list[2].data, device="cpu", pin_memory=True)
        colorsN_grad = param_list[2].grad
        colorsN_cpu_grad.copy_(colorsN_grad, non_blocking=False)

        # Frustum culling & H2D communication for the next iteration
        device = self.device
        means = self.splats["means"]  # [N, 3], gpu
        quats = self.splats["quats"]  # [N, 4], gpu
        scales = torch.exp(self.splats["scales"])  # [N, 3], gpu

        # Data pre-load for frustum culling
        data = next(trainloader_iter)   # To identify valid gaussians for the next iteration
        camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
        height, width = pixels.shape[1:3]

        # Perform frustum culling on GPU
        valid_ids = frustum_culling_gpu(means, quats, scales, torch.linalg.inv(camtoworlds), Ks, width, height)
        if cfg.batch_size > 1:
            valid_ids = torch.unique(valid_ids)
        # valid_ids, _ = torch.sort(valid_ids)

        # Prepare for CPU Optimizer update (Older version of valid_ids_cpu must be used here!)
        self.splats["opacities"].grad[valid_ids_cpu] = opacities_cpu_grad
        self.splats["sh0"].grad[valid_ids_cpu] = colors0_cpu_grad
        self.splats["shN"].grad[valid_ids_cpu] = colorsN_cpu_grad

        # Lazy update is not allowed when densification
        valid_ids_cpu_prev = None
        if is_densify:
            # CPU Optimizer update
            # CPU Optimizer
            for optimizer in self.optimizers_cpu.values():
                optimizer.step() 

            # Only perform zero_grad for the valid gaussians. The rest are already zero.
            self.splats["opacities"].grad[valid_ids_cpu] = 0
            self.splats["sh0"].grad[valid_ids_cpu] = 0
            self.splats["shN"].grad[valid_ids_cpu] = 0

            # Run post-backward steps after backward and optimizer
            optimizers_all = {**self.optimizers_gpu, **self.optimizers_cpu}
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward_wo_update(
                    params=self.splats,
                    optimizers=optimizers_all,
                    state=self.strategy_state,
                    step=0,
                    info=info,
                    packed=cfg.packed,
                )
                self.splats["opacities"].grad = torch.zeros_like(self.splats["opacities"].data, device="cpu")
                self.splats["sh0"].grad = torch.zeros_like(self.splats["sh0"].data, device="cpu")
                self.splats["shN"].grad = torch.zeros_like(self.splats["shN"].data, device="cpu")
            else:
                assert_never(self.cfg.strategy)

            # Frustum culling again with new parameters
            means = self.splats["means"]  # [N, 3], gpu
            quats = self.splats["quats"]  # [N, 4], gpu
            scales = torch.exp(self.splats["scales"])  # [N, 3], gpu
            valid_ids = frustum_culling_gpu(means, quats, scales, torch.linalg.inv(camtoworlds), Ks, width, height)
            if cfg.batch_size > 1:
                valid_ids = torch.unique(valid_ids)
            # valid_ids, _ = torch.sort(valid_ids)
            valid_ids_cpu = valid_ids.to("cpu", non_blocking=False)

            # Host to Device
            opacities = torch.sigmoid(self.splats["opacities"][valid_ids_cpu].detach()).pin_memory().to(device, non_blocking=True).requires_grad_()
            colors0 = self.splats["sh0"][valid_ids_cpu].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
            colorsN = self.splats["shN"][valid_ids_cpu].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
        else:
            valid_ids_cpu_prev = valid_ids_cpu
            valid_ids_cpu = valid_ids.to("cpu", non_blocking=False)
    
            # Host to Device & CPU Update for next iter
            # Transfer with 32MB block size
            count = valid_ids_cpu.shape[0]
            with torch.no_grad():
                colorsN = torch.empty((count, self.splats["shN"].shape[1], self.splats["shN"].shape[2]), dtype=self.splats["shN"].dtype, device=device, requires_grad=False)
                optimizer = list(self.optimizers_cpu.values())[2]
                lr = optimizer.param_groups[0]['lr']
                b1 = optimizer.param_groups[0]['betas'][0]
                b2 = optimizer.param_groups[0]['betas'][1]
                eps = optimizer.param_groups[0]['eps']
                s = 0
                opt1 = torch.zeros_like(self.splats["shN"].data)
                opt2 = torch.zeros_like(self.splats["shN"].data)
                if count < 180000:
                    weight = torch.empty(colorsN.shape, dtype=self.splats["shN"].dtype, pin_memory=True)
                    adam_for_next(self.splats["shN"].data, self.splats["shN"].grad, opt1, opt2, valid_ids_cpu, weight, s, lr, b1, b2, eps)
                    colorsN.copy_(weight, non_blocking=True)
                else:
                    nstage = count // 180000     # 180000*45*4 = 32MB
                    offset = torch.arange(nstage) * (count // nstage)
                    for i in range(nstage-1):
                        weight = torch.empty((count // nstage, self.splats["shN"].shape[1], self.splats["shN"].shape[2]), dtype=self.splats["shN"].dtype, pin_memory=True)
                        adam_for_next(self.splats["shN"].data, self.splats["shN"].grad, opt1, opt2, valid_ids_cpu[offset[i]:offset[i+1]], weight, s, lr, b1, b2, eps)
                        colorsN[offset[i]:offset[i+1]].copy_(weight, non_blocking=True)
                    weight = torch.empty((count - offset[nstage-1], self.splats["shN"].shape[1], self.splats["shN"].shape[2]), dtype=self.splats["shN"].dtype, pin_memory=True)
                    adam_for_next(self.splats["shN"].data, self.splats["shN"].grad, opt1, opt2, valid_ids_cpu[offset[nstage-1]:], weight, s, lr, b1, b2, eps)
                    colorsN[offset[nstage-1]:].copy_(weight, non_blocking=True)
    
                colors0 = torch.empty((count, self.splats["sh0"].shape[1], self.splats["sh0"].shape[2]), dtype=self.splats["sh0"].dtype, device=device, requires_grad=False)
                optimizer = list(self.optimizers_cpu.values())[1]
                lr = optimizer.param_groups[0]['lr']
                opt1 = torch.zeros_like(self.splats["sh0"].data)
                opt2 = torch.zeros_like(self.splats["sh0"].data)
                weight = torch.empty(colors0.shape, dtype=self.splats["sh0"].dtype, pin_memory=True)
                adam_for_next(self.splats["sh0"].data, self.splats["sh0"].grad, opt1, opt2, valid_ids_cpu, weight, s, lr, b1, b2, eps)
                colors0.copy_(weight, non_blocking=True)
    
                opacities = torch.empty((count), dtype=self.splats["opacities"].dtype, device=device, requires_grad=False)
                optimizer = list(self.optimizers_cpu.values())[0]
                lr = optimizer.param_groups[0]['lr']
                opt1 = torch.zeros_like(self.splats["opacities"].data)
                opt2 = torch.zeros_like(self.splats["opacities"].data)
                weight = torch.empty(opacities.shape, dtype=self.splats["opacities"].dtype, pin_memory=True)
                adam_for_next(self.splats["opacities"].data, self.splats["opacities"].grad, opt1, opt2, valid_ids_cpu, weight, s, lr, b1, b2, eps)
                opacities.copy_(torch.sigmoid(weight), non_blocking=True)
    
            opacities = opacities.detach().requires_grad_()
            colors0 = colors0.detach().requires_grad_()
            colorsN = colorsN.detach().requires_grad_()

        data_list = [camtoworlds, Ks, pixels, image_ids, masks, height, width]
        param_list = [opacities, colors0, colorsN]

        return is_densify, valid_ids, valid_ids_cpu, valid_ids_cpu_prev, data_list, param_list


    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers_gpu["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Initialize gradient tensors to zero. We only want to perform zero_grad for the valid gaussians.
        self.splats["opacities"].grad = torch.zeros_like(self.splats["opacities"].data, device="cpu")
        self.splats["sh0"].grad = torch.zeros_like(self.splats["sh0"].data, device="cpu")
        self.splats["shN"].grad = torch.zeros_like(self.splats["shN"].data, device="cpu")

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))


        ########## Prologue (First iteration outside the loop for pipelining) ##########
        is_densify, valid_ids, valid_ids_cpu, valid_ids_cpu_prev, data_list, param_list = self.prologue(trainloader_iter, schedulers)
        camtoworlds, Ks, pixels, image_ids, masks, height, width = data_list
        opacities, colors0, colorsN = param_list
        counter = torch.zeros(self.splats["opacities"].shape[0], dtype=torch.int8, device="cpu")  # Initial counter

        ########## Main Loop ##########
        for step in pbar:
            # First iteration is performed before entering the loop.
            if step == 0:
                continue

            # Free reserved memory only when the portion of used gaussians exceeds 0.1
            # Freeing reserved memory every iteration is inefficient.
            if valid_ids.shape[0] / self.splats["means"].shape[0] > 0.1:
                torch.cuda.empty_cache()

            ############# Start #############
            torch.cuda.synchronize()
            results = {}

            subthread = threading.Thread(
                target=self.gpu_forward_backward, 
                args=(step, camtoworlds, Ks, valid_ids, opacities, colors0, colorsN, width, height, image_ids, masks, pixels, results, ), 
                daemon=False
            )
            subthread.start()

            if is_densify:
                pipeline_bubble()
            else:
                w1 = self.splats["opacities"].data
                g1 = self.splats["opacities"].grad
                optimizer = list(self.optimizers_cpu.values())[0]
                lr1 = optimizer.param_groups[0]['lr']
                b1 = optimizer.param_groups[0]['betas'][0]
                b2 = optimizer.param_groups[0]['betas'][1]
                eps = optimizer.param_groups[0]['eps']
                param = list(optimizer.state.keys())[0]
                optimizer.state[param]['step'] += 1
                s = int(optimizer.state[param]['step'])
                opt1_1 = optimizer.state[param]['exp_avg']
                opt1_2 = optimizer.state[param]['exp_avg_sq']

                w2 = self.splats["sh0"].data
                g2 = self.splats["sh0"].grad
                optimizer = list(self.optimizers_cpu.values())[1]
                lr2 = optimizer.param_groups[0]['lr']
                param = list(optimizer.state.keys())[0]
                optimizer.state[param]['step'] += 1
                opt2_1 = optimizer.state[param]['exp_avg']
                opt2_2 = optimizer.state[param]['exp_avg_sq']

                w3 = self.splats["shN"].data
                g3 = self.splats["shN"].grad
                optimizer = list(self.optimizers_cpu.values())[2]
                lr3 = optimizer.param_groups[0]['lr']
                param = list(optimizer.state.keys())[0]
                optimizer.state[param]['step'] += 1
                opt3_1 = optimizer.state[param]['exp_avg']
                opt3_2 = optimizer.state[param]['exp_avg_sq']

                update_ids = calculate_update_ids(valid_ids_cpu_prev.int(), counter, int(self.splats["opacities"].shape[0]))

                cpu_optimizer_parallel(valid_ids_cpu_prev, update_ids, counter, w1, w2, w3, g1, g2, g3, opt1_1, opt2_1, opt3_1, opt1_2, opt2_2, opt3_2, lr1, lr2, lr3, s, b1, b2, eps)
            
            #################################    

            ############# Join #############
            torch.cuda.synchronize()
            subthread.join()
            ################################

            # Only "means" apply lr schedule. "means" is on GPU.
            for scheduler in schedulers:
                scheduler.step()

            info = results["info"]
            sh_degree_to_use = results["sh_degree_to_use"]
            l1loss = results["l1loss"]
            ssimloss = results["ssimloss"]
            loss = results["loss"]
            param_list = results["param_list"]

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            pbar.set_description(desc)

            # Move info from gpu to cpu for densification
            info["gaussian_ids"] = valid_ids_cpu[info["gaussian_ids"].to("cpu", non_blocking=False)]
            info["radii"] = info["radii"].to("cpu", non_blocking=False)
            info["means2d"].data = info["means2d"].data.to("cpu", non_blocking=False)
            info["means2d"].grad = info["means2d"].grad.to("cpu", non_blocking=False)

            # Update grad2d state for later densification
            if isinstance(self.cfg.strategy, DefaultStrategy):
                is_densify = self.cfg.strategy.step_post_backward_only_update(
                    params=self.splats,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
                # In last iter, must execute remaining pipeline
                if step == max_steps - 1:
                    is_densify = True
            else:
                assert_never(self.cfg.strategy)

            # Device to Host
            # Manually backpropagate grads from gpu to cpu. Backprop for means, quats, and scales are done via autograd.
            opacities_cpu_grad = torch.empty_like(param_list[0].data, device="cpu", pin_memory=True)
            opacities_grad = param_list[0].grad * param_list[0].data * (1 - param_list[0].data)     # derivative of sigmoid
            opacities_cpu_grad.copy_(opacities_grad, non_blocking=False) 
    
            colors0_cpu_grad = torch.empty_like(param_list[1].data, device="cpu", pin_memory=True)
            colors0_grad = param_list[1].grad
            colors0_cpu_grad.copy_(colors0_grad, non_blocking=False)
    
            colorsN_cpu_grad = torch.empty_like(param_list[2].data, device="cpu", pin_memory=True)
            colorsN_grad = param_list[2].grad
            colorsN_cpu_grad.copy_(colorsN_grad, non_blocking=False)

            # Frustum culling & H2D communication for the next iteration
            device = self.device
            means = self.splats["means"]  # [N, 3], gpu
            quats = self.splats["quats"]  # [N, 4], gpu
            scales = torch.exp(self.splats["scales"])  # [N, 3], gpu

            # Data pre-load for frustum culling
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            height, width = pixels.shape[1:3]

            # Perform frustum culling on GPU
            valid_ids = frustum_culling_gpu(means, quats, scales, torch.linalg.inv(camtoworlds), Ks, width, height)
            if cfg.batch_size > 1:
                valid_ids = torch.unique(valid_ids)
            # valid_ids, _ = torch.sort(valid_ids)

            # Prepare for CPU Optimizer update (Older version of valid_ids_cpu must be used here!)
            # This must be done after join. If not, it will disrupt cpu adam optimizer update.
            self.splats["opacities"].grad[valid_ids_cpu] = opacities_cpu_grad
            self.splats["sh0"].grad[valid_ids_cpu] = colors0_cpu_grad
            self.splats["shN"].grad[valid_ids_cpu] = colorsN_cpu_grad

            # Lazy update is not allowed when densification
            if is_densify:
                # CPU Optimizer update (all parameters must be updated here + counter is reset)
                update_ids = torch.arange(self.splats["opacities"].shape[0], dtype=torch.int32, device="cpu")
                w1 = self.splats["opacities"].data
                g1 = self.splats["opacities"].grad
                optimizer = list(self.optimizers_cpu.values())[0]
                lr1 = optimizer.param_groups[0]['lr']
                b1 = optimizer.param_groups[0]['betas'][0]
                b2 = optimizer.param_groups[0]['betas'][1]
                eps = optimizer.param_groups[0]['eps']
                param = list(optimizer.state.keys())[0]
                optimizer.state[param]['step'] += 1
                s = int(optimizer.state[param]['step'])
                opt1_1 = optimizer.state[param]['exp_avg']
                opt1_2 = optimizer.state[param]['exp_avg_sq']

                w2 = self.splats["sh0"].data
                g2 = self.splats["sh0"].grad
                optimizer = list(self.optimizers_cpu.values())[1]
                lr2 = optimizer.param_groups[0]['lr']
                param = list(optimizer.state.keys())[0]
                optimizer.state[param]['step'] += 1
                opt2_1 = optimizer.state[param]['exp_avg']
                opt2_2 = optimizer.state[param]['exp_avg_sq']

                w3 = self.splats["shN"].data
                g3 = self.splats["shN"].grad
                optimizer = list(self.optimizers_cpu.values())[2]
                lr3 = optimizer.param_groups[0]['lr']
                param = list(optimizer.state.keys())[0]
                optimizer.state[param]['step'] += 1
                opt3_1 = optimizer.state[param]['exp_avg']
                opt3_2 = optimizer.state[param]['exp_avg_sq']

                with torch.no_grad():
                    adam_deferred_update(w1, g1, opt1_1, opt1_2, update_ids, counter, s, lr1, b1, b2, eps) 
                    adam_deferred_update(w2, g2, opt2_1, opt2_2, update_ids, counter, s, lr2, b1, b2, eps) 
                    adam_deferred_update(w3, g3, opt3_1, opt3_2, update_ids, counter, s, lr3, b1, b2, eps) 

                # Only perform zero_grad for the valid gaussians. The rest are already zero.
                self.splats["opacities"].grad[valid_ids_cpu] = 0
                self.splats["sh0"].grad[valid_ids_cpu] = 0
                self.splats["shN"].grad[valid_ids_cpu] = 0

                # Run post-backward steps after backward and optimizer
                optimizers_all = {**self.optimizers_gpu, **self.optimizers_cpu}
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward_wo_update(
                        params=self.splats,
                        optimizers=optimizers_all,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
                    self.splats["opacities"].grad = torch.zeros_like(self.splats["opacities"].data, device="cpu")
                    self.splats["sh0"].grad = torch.zeros_like(self.splats["sh0"].data, device="cpu")
                    self.splats["shN"].grad = torch.zeros_like(self.splats["shN"].data, device="cpu")
                else:
                    assert_never(self.cfg.strategy)

                # Frustum culling again with new parameters
                means = self.splats["means"]  # [N, 3], gpu
                quats = self.splats["quats"]  # [N, 4], gpu
                scales = torch.exp(self.splats["scales"])  # [N, 3], gpu
                valid_ids = frustum_culling_gpu(means, quats, scales, torch.linalg.inv(camtoworlds), Ks, width, height)
                if cfg.batch_size > 1:
                    valid_ids = torch.unique(valid_ids)
                # valid_ids, _ = torch.sort(valid_ids)
                valid_ids_cpu = valid_ids.to("cpu", non_blocking=False)

                # Host to Device
                opacities = torch.sigmoid(self.splats["opacities"][valid_ids_cpu].detach()).pin_memory().to(device, non_blocking=True).requires_grad_()
                colors0 = self.splats["sh0"][valid_ids_cpu].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
                colorsN = self.splats["shN"][valid_ids_cpu].detach().pin_memory().to(device, non_blocking=True).requires_grad_()

                # New counter is set
                counter = torch.zeros(self.splats["opacities"].shape[0], dtype=torch.int8, device="cpu")  # Initial counter
            else:
                # Host to Device & CPU Update for next iter
                # Transfer with 32MB block size
                valid_ids_cpu_prev = valid_ids_cpu
                valid_ids_cpu = valid_ids.to("cpu", non_blocking=False)
                count = valid_ids_cpu.shape[0]
                with torch.no_grad():
                    colorsN = torch.empty((count, self.splats["shN"].shape[1], self.splats["shN"].shape[2]), dtype=self.splats["shN"].dtype, device=device, requires_grad=False)
                    optimizer = list(self.optimizers_cpu.values())[2]
                    lr = optimizer.param_groups[0]['lr']
                    b1 = optimizer.param_groups[0]['betas'][0]
                    b2 = optimizer.param_groups[0]['betas'][1]
                    eps = optimizer.param_groups[0]['eps']
                    param = list(optimizer.state.keys())[0]
                    s = int(optimizer.state[param]['step'])
                    opt1 = optimizer.state[param]['exp_avg']
                    opt2 = optimizer.state[param]['exp_avg_sq']
                    if count < 180000:
                        weight = torch.empty(colorsN.shape, dtype=self.splats["shN"].dtype, pin_memory=True)
                        adam_for_next_with_counter(self.splats["shN"].data, self.splats["shN"].grad, opt1, opt2, valid_ids_cpu, weight, counter, s, lr, b1, b2, eps)
                        colorsN.copy_(weight, non_blocking=True)
                    else:
                        nstage = count // 180000     # 180000*45*4 = 32MB
                        offset = torch.arange(nstage) * (count // nstage)
                        for i in range(nstage-1):
                            weight = torch.empty((count // nstage, self.splats["shN"].shape[1], self.splats["shN"].shape[2]), dtype=self.splats["shN"].dtype, pin_memory=True)
                            adam_for_next_with_counter(self.splats["shN"].data, self.splats["shN"].grad, opt1, opt2, valid_ids_cpu[offset[i]:offset[i+1]], weight, counter, s, lr, b1, b2, eps)
                            colorsN[offset[i]:offset[i+1]].copy_(weight, non_blocking=True)
                        weight = torch.empty((count - offset[nstage-1], self.splats["shN"].shape[1], self.splats["shN"].shape[2]), dtype=self.splats["shN"].dtype, pin_memory=True)
                        adam_for_next_with_counter(self.splats["shN"].data, self.splats["shN"].grad, opt1, opt2, valid_ids_cpu[offset[nstage-1]:], weight, counter, s, lr, b1, b2, eps)
                        colorsN[offset[nstage-1]:].copy_(weight, non_blocking=True)
        
                    colors0 = torch.empty((count, self.splats["sh0"].shape[1], self.splats["sh0"].shape[2]), dtype=self.splats["sh0"].dtype, device=device, requires_grad=False)
                    optimizer = list(self.optimizers_cpu.values())[1]
                    lr = optimizer.param_groups[0]['lr']
                    param = list(optimizer.state.keys())[0]
                    opt1 = optimizer.state[param]['exp_avg']
                    opt2 = optimizer.state[param]['exp_avg_sq']
                    weight = torch.empty(colors0.shape, dtype=self.splats["sh0"].dtype, pin_memory=True)
                    adam_for_next_with_counter(self.splats["sh0"].data, self.splats["sh0"].grad, opt1, opt2, valid_ids_cpu, weight, counter, s, lr, b1, b2, eps)
                    colors0.copy_(weight, non_blocking=True)
        
                    opacities = torch.empty((count), dtype=self.splats["opacities"].dtype, device=device, requires_grad=False)
                    optimizer = list(self.optimizers_cpu.values())[0]
                    lr = optimizer.param_groups[0]['lr']
                    param = list(optimizer.state.keys())[0]
                    opt1 = optimizer.state[param]['exp_avg']
                    opt2 = optimizer.state[param]['exp_avg_sq']
                    weight = torch.empty(opacities.shape, dtype=self.splats["opacities"].dtype, pin_memory=True)
                    adam_for_next_with_counter(self.splats["opacities"].data, self.splats["opacities"].grad, opt1, opt2, valid_ids_cpu, weight, counter, s, lr, b1, b2, eps)
                    opacities.copy_(torch.sigmoid(weight), non_blocking=True)
        
                opacities = opacities.detach().requires_grad_()
                colors0 = colors0.detach().requires_grad_()
                colorsN = colorsN.detach().requires_grad_()

            # Misc. (logging, save checkpoint, eval)
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                alloc_mem = torch.cuda.max_memory_allocated() / 1024**3
                res_mem = torch.cuda.max_memory_reserved() / 1024**3
                stats = {
                    "allocated_mem": alloc_mem,
                    "reserved_mem": res_mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                rgb = None
                save_ply(self.splats, f"{self.ply_dir}/point_cloud_{step}.ply", rgb)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # self.render_traj(step)


    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            
            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "sztu": (
            "GauU Scene SZTU dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=4000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/GauU_Scene/SZTU",
                data_factor=3.4175,
                test_every=10,
                test_after=-1,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=1.6e-5,
                scaling_lr=5e-3,
                result_dir="results/sztu",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "sziit": (
            "GauU Scene SZIIT dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=5000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/GauU_Scene/SZIIT",
                data_factor=3.4175,
                test_every=10,
                test_after=-1,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=1.6e-5,
                scaling_lr=5e-3,
                result_dir="results/sziit",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "lfls": (
            "GauU Scene LFLS dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=4000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/GauU_Scene/LFLS",
                data_factor=3.4175,
                test_every=10,
                test_after=-1,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=1.6e-5,
                scaling_lr=5e-3,
                result_dir="results/lfls",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "rubble": (
            "Mill19 Rubble dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=5000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/mill19/rubble-pixsfm",
                data_factor=4,
                test_every=83,
                test_after=-1,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=1.6e-5,
                scaling_lr=5e-3,
                result_dir="results/rubble",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "building": (
            "Mill19 Building dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=5000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/mill19/building-pixsfm",
                data_factor=4,
                test_every=97,
                test_after=-1,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=8e-6,
                scaling_lr=2.5e-3,
                result_dir="results/building",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "aerial": (
            "MatrixCity Aerial dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=5000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/MatrixCity/aerial",
                data_factor=1.2,
                test_every=-1,
                test_after=5620,
                init_ply=True,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=8e-6,
                scaling_lr=2.5e-3,
                result_dir="results/aerial",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "residence": (
            "UrbanScene3D Residence dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=5000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/urban_scene_3d/residence-pixsfm",
                data_factor=4,
                test_every=-1,
                test_after=2561,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=1.6e-5,
                scaling_lr=5e-3,
                result_dir="results/residence",
                packed=True,
                sparse_grad=False,
            ),
        ),
        "sciart": (
            "UrbanScene3D Sci-Art dataset",
            Config(
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=5000),
                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0002, grow_scale3d=0.01, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00015, grow_scale3d=0.005, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00013, grow_scale3d=0.003, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.0001, grow_scale3d=0.002, refine_stop_iter=15000),
#                strategy=DefaultStrategy(verbose=True, grow_grad2d=0.00008, grow_scale3d=0.0016, refine_stop_iter=15000),
                data_dir="data/urban_scene_3d/sci-art-pixsfm",
                data_factor=4,
                test_every=-1,
                test_after=2998,
                max_steps=200_000,
                eval_steps=[200_000],
                save_steps=[200_000],
                ply_steps=[200_000],
                position_lr=1.6e-5,
                scaling_lr=5e-3,
                result_dir="results/sciart",
                packed=True,
                sparse_grad=False,
            ),
        ),
    }
   
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
