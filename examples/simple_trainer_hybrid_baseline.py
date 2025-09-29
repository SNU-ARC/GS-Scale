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
from intel_extension_for_pytorch.optim._functional import adam_step, adamw_step
from cpu_adam import world_to_cam, persp_proj, quat_to_rotmat
from gsplat.rendering import rasterization
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
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


# Custom operator for fast D2H transfer in forward, and fast H2D transfer in backward
class Fast_D2H_Operation(Function):
    @staticmethod
    def forward(ctx, tensor_cpu, target_device):
        tensor_gpu = tensor_cpu.pin_memory().to(target_device, non_blocking=True)
        return tensor_gpu

    @staticmethod
    def backward(ctx, grad_gpu):
        grad_cpu = torch.empty_like(grad_gpu, device="cpu", pin_memory=True)
        grad_cpu.copy_(grad_gpu, non_blocking=False)
        return grad_cpu, None

fast_d2h = Fast_D2H_Operation.apply


def frustum_culling(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor, # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> Tensor:
    R = quat_to_rotmat(quats)  # (..., 3, 3)
    M = R * scales[..., None, :]  # (..., 3, 3)
    covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)

    means_c, covars_c = world_to_cam(means, covars, viewmats)
    
    # Only support pinhole camera model 
    means2d, covars2d = persp_proj(means_c, covars_c, Ks, width, height)

    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    depths = means_c[..., 2]  # [C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    tmp = torch.sqrt(torch.clamp(b**2 - det, min=0.01))
    v1 = b + tmp  # (...,)
    r1 = 3.33 * torch.sqrt(v1)
    radius_x = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 0, 0]), r1))
    radius_y = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 1, 1]), r1))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # (..., 2)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )

    valid_ids = (valid.squeeze(0) & inside.squeeze(0)).nonzero().squeeze(-1)
 
    return valid_ids


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
    device: str = "cpu",
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

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), position_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scaling_lr),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colorsN = torch.zeros((N, (sh_degree + 1) ** 2 - 1, 3))  # [N, K, 3]
        colors0 = rgb_to_sh(rgbs).unsqueeze(1)
        params.append(("sh0", torch.nn.Parameter(colors0), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colorsN), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params})   # paramters in the host mem
    
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = FusedCPUAdam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
#            adamw_mode=False,
        )
        for name, _, lr in params
    }
    return splats, optimizers


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
        self.splats, self.optimizers = create_splats_with_optimizers_in_hostmem(
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
            device="cpu",
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
 
        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

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

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            ) 
       
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        camtoworlds_cpu: Optional[Tensor] = None,
        Ks_cpu: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        device = self.device
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            #colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
            colors0 = self.splats["sh0"]
            colorsN = self.splats["shN"]

        # Perform frustum culling on CPU
        valid_ids = frustum_culling(means, quats, scales, torch.linalg.inv(camtoworlds_cpu), Ks_cpu, width, height)

        colorsN_gpu = colorsN[valid_ids].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
        quats_gpu = quats[valid_ids].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
        means_gpu = means[valid_ids].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
        scales_gpu = scales[valid_ids].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
        colors0_gpu = colors0[valid_ids].detach().pin_memory().to(device, non_blocking=True).requires_grad_()
        opacities_gpu = opacities[valid_ids].detach().pin_memory().to(device, non_blocking=True).requires_grad_()

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

        param_list = [means_gpu, quats_gpu, scales_gpu, opacities_gpu, colors0_gpu, colorsN_gpu]
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info, param_list, valid_ids

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
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

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
        self.splats["means"].grad = torch.zeros_like(self.splats["means"].data, device="cpu")
        self.splats["quats"].grad = torch.zeros_like(self.splats["quats"].data, device="cpu")
        self.splats["scales"].grad = torch.zeros_like(self.splats["scales"].data, device="cpu")
        self.splats["opacities"].grad = torch.zeros_like(self.splats["opacities"].data, device="cpu")
        self.splats["sh0"].grad = torch.zeros_like(self.splats["sh0"].data, device="cpu")
        self.splats["shN"].grad = torch.zeros_like(self.splats["shN"].data, device="cpu")

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

#        from torch.profiler import profile, record_function, ProfilerActivity
#        with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/rubble_hybrid_baseline')) as p:
        for step in pbar:
#            if step == 29950:
#                torch.cuda.memory._record_memory_history(max_entries=100000)
#            if step == 15000:
#                exit(0)
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            camtoworlds_cpu = data["camtoworld"]
            Ks = data["K"].to(device)  # [1, 3, 3]
            Ks_cpu = data["K"]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]
            
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            torch.cuda.empty_cache()

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info, param_list, valid_ids = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                camtoworlds_cpu=camtoworlds_cpu,
                Ks_cpu=Ks_cpu,
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
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
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            info["gaussian_ids"] = valid_ids[info["gaussian_ids"].to("cpu", non_blocking=False)]
            info["radii"] = info["radii"].to("cpu", non_blocking=False)

            loss.backward()

            # Move info from gpu to cpu for densification
            info["means2d"].data = info["means2d"].data.to("cpu", non_blocking=False)
            info["means2d"].grad = info["means2d"].grad.to("cpu", non_blocking=False)

            # Manually backpropagate grads from gpu to cpu
            opacities_cpu_grad = torch.empty_like(param_list[3].data, device="cpu", pin_memory=True)
            opacities_grad = param_list[3].grad * param_list[3].data * (1 - param_list[3].data)
            opacities_cpu_grad.copy_(opacities_grad, non_blocking=False)    # derivative of sigmoid
            self.splats["opacities"].grad[valid_ids] = opacities_cpu_grad

            means_cpu_grad = torch.empty_like(param_list[0].data, device="cpu", pin_memory=True)
            means_cpu_grad.copy_(param_list[0].grad, non_blocking=False)
            self.splats["means"].grad[valid_ids] = means_cpu_grad

            scales_cpu_grad = torch.empty_like(param_list[2].data, device="cpu", pin_memory=True)
            scales_grad = param_list[2].grad * param_list[2].data
            scales_cpu_grad.copy_(scales_grad, non_blocking=False)    # derivative of exp
            self.splats["scales"].grad[valid_ids] = scales_cpu_grad

            colors0_cpu_grad = torch.empty_like(param_list[4].data, device="cpu", pin_memory=True)
            colors0_cpu_grad.copy_(param_list[4].grad, non_blocking=False)
            self.splats["sh0"].grad[valid_ids] = colors0_cpu_grad

            quats_cpu_grad = torch.empty_like(param_list[1].data, device="cpu", pin_memory=True)
            quats_cpu_grad.copy_(param_list[1].grad, non_blocking=False)
            self.splats["quats"].grad[valid_ids] = quats_cpu_grad

            colorsN_cpu_grad = torch.empty_like(param_list[5].data, device="cpu", pin_memory=True)
            colorsN_cpu_grad.copy_(param_list[5].grad, non_blocking=False)
            self.splats["shN"].grad[valid_ids] = colorsN_cpu_grad

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

#            if step > 29950:
#                timestamp = datetime.now().strftime(TIME_FORMAT_STR)
#                torch.cuda.memory._dump_snapshot(f"gsplat_{timestamp}.html")

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
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
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                rgb = None
                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0)

                save_ply(self.splats, f"{self.ply_dir}/point_cloud_{step}.ply", rgb)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                #optimizer.zero_grad(set_to_none=True)
            # Only perform zero_grad for the valid gaussians. The rest are already zero.
            self.splats["means"].grad[valid_ids] = 0
            self.splats["quats"].grad[valid_ids] = 0
            self.splats["scales"].grad[valid_ids] = 0
            self.splats["opacities"].grad[valid_ids] = 0
            self.splats["sh0"].grad[valid_ids] = 0
            self.splats["shN"].grad[valid_ids] = 0

            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()


            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                is_densify = self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
                # We must re-initialize gradient tensors to zero after densification
                if is_densify:
                    self.splats["means"].grad = torch.zeros_like(self.splats["means"].data, device="cpu")
                    self.splats["quats"].grad = torch.zeros_like(self.splats["quats"].data, device="cpu")
                    self.splats["scales"].grad = torch.zeros_like(self.splats["scales"].data, device="cpu")
                    self.splats["opacities"].grad = torch.zeros_like(self.splats["opacities"].data, device="cpu")
                    self.splats["sh0"].grad = torch.zeros_like(self.splats["sh0"].data, device="cpu")
                    self.splats["shN"].grad = torch.zeros_like(self.splats["shN"].data, device="cpu")

            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

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
            camtoworlds_cpu = data["camtoworld"]
            Ks_cpu = data["K"]
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            
            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                camtoworlds_cpu=camtoworlds_cpu,
                Ks_cpu=Ks_cpu,
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
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

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
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

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

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


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
