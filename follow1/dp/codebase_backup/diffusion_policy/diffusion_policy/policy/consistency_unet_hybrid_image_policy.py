from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
# from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from diffusion_policy.model.consistency.karras_diffusion import KarrasDenoiser, karras_sample
from diffusion_policy.model.consistency.sampler import create_named_schedule_sampler, LossAwareSampler
from diffusion_policy.model.consistency.scripts_util import create_ema_and_scales_fn
from diffusion_policy.model.consistency.nn import update_ema
from diffusion_policy.model.consistency.fp16_utils import master_params_to_model_params, make_master_params, get_param_groups_and_shapes

import functools
import copy

class ConsistencyUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDIMScheduler,
            ema_scale,
            sample,
            # obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        ''' create ema '''
        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=ema_scale.target_ema_mode,
            start_ema=ema_scale.start_ema,
            scale_mode=ema_scale.scale_mode,
            start_scales=ema_scale.start_scales,
            end_scales=ema_scale.end_scales,
            total_steps=ema_scale.total_training_steps,
            distill_steps_per_iter=ema_scale.distill_steps_per_iter,
        )

        '''-- model follow DP --'''
        self.model = model
        self.param_groups_and_shapes = get_param_groups_and_shapes(
            self.model.named_parameters()
        )
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )

        '''-- target model --'''
        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)
        self.target_model.train()

        self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
            self.target_model.named_parameters()
        )
        self.target_model_master_params = make_master_params(
            self.target_model_param_groups_and_shapes
        )

        '''-- scheduler follow CM --'''
        self.diffusion = KarrasDenoiser(
            sigma_data=noise_scheduler.sigma_data,
            sigma_max=noise_scheduler.sigma_max,
            sigma_min=noise_scheduler.sigma_min,
            distillation=noise_scheduler.distillation,
            weight_schedule=noise_scheduler.weight_schedule,
        )
        self.schedule_sampler = create_named_schedule_sampler(noise_scheduler.schedule_sampler, self.diffusion)

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0 if obs_as_global_cond else obs_feature_dim,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.training_mode = ema_scale.training_mode ## new
        #
        self.sampler = sample.sampler
        self.generator = sample.generator
        self.ts = sample.ts
        self.clip_denoised = sample.clip_denoised
        self.sigma_min = noise_scheduler.sigma_min
        self.sigma_max = noise_scheduler.sigma_max

        self.s_churn = sample.s_churn
        self.s_tmin = sample.s_tmin
        self.s_tmax = float(sample.s_tmax)
        self.s_noise = sample.s_noise
        self.steps = sample.steps
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = karras_sample(
            self.diffusion,
            self.model,
            (B, T, Da),
            cond_data, 
            cond_mask,
            steps=self.steps,
            clip_denoised=self.clip_denoised,
            local_cond=None,
            global_cond=global_cond,
            device=self.device,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            sampler=self.sampler,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
            generator=None,
            ts=self.ts,
        )
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, global_step):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        ## -- Importance-sample timesteps for a batch
        t, weights = self.schedule_sampler.sample(trajectory.shape[0], self.device)

        ema, num_scales = self.ema_scale_fn(global_step) ## 注意：参数设置 is different between CD and CT

        ## -- 声明 compute loss 模式 -- ##
        ## -- -- 定义于 karra_diffusion.py / consistency_loss()
        if self.training_mode == "consistency_training":
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.model,
                trajectory,
                num_scales,
                target_model=self.target_model,
                local_cond = local_cond,
                global_cond = global_cond,
            )
        else:
            raise ValueError(f"Warning training mode {self.training_mode}")

        ## -- 计算 loss -- ##
        losses = compute_losses() ## 重点

        ## 当 sampler = LossSecondMomentResampler 才会启动
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        ## -- loss 加权取平均 -- ##
        loss = (losses["loss"] * weights).mean()

        return loss


    def update_target_ema(self, global_step):

        ## Note: 此处针对原代码（consistency model）进行改动：
        ## 为防止 master_params 和 target_model_master_params 没有随着模型更新
        ## 此处 显性地实时地同步一遍 master_params 和 target_model_master_params
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )
        self.target_model_master_params = make_master_params(
            self.target_model_param_groups_and_shapes
        )

        target_ema, scales = self.ema_scale_fn(global_step)
        with torch.no_grad():
            update_ema(
                self.target_model_master_params,
                self.master_params,
                rate = target_ema,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )
        # print(self.target_model_master_params)
