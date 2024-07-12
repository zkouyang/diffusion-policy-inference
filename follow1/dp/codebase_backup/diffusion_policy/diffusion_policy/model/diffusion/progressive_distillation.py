"""
Implementation of the Progressive Distillation Transformer Policy
"""

import math
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class GuassianDiffusion:

    def __init__(
        self,
        policy,
        p_sampler: DDPMScheduler,
        time_scale=1,
    ):
        self.policy = policy
        self.time_scale = time_scale
        self.p_sampler = p_sampler

        betas = self.p_sampler.betas
        alphas = self.p_sampler.alphas
        alphas_cumprod = self.p_sampler.alphas_cumprod
        alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1], dtype=torch.float64, device=betas.device),
                alphas_cumprod[:-1],
            ),
            0,
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
        )

    def inference(
        self,
        x,
        t,
        cond=None,
    ):
        return self.policy.model(x, t * self.time_scale, cond)

    def get_alpha_sigma(self, x, t):
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma

    def to_device(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(
            device
        )
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)


class GuassianDiffusionDefault(GuassianDiffusion):

    def __init__(
        self,
        policy,
        p_sampler: DDPMScheduler,
        time_scale=1.0,
        gamma=0.3,
    ):
        super().__init__(policy, p_sampler, time_scale)
        self.gamma = gamma

    def distill_loss(self, student_diffusion, x, t, cond, eps=None):
        if eps is None:
            eps = torch.randn_like(x, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = self.p_sampler.add_noise(x, eps, t + 1)
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.inference(z.float(), t.float() + 1, cond).double()
            rec = (alpha * z - sigma * v).clip(-1, 1)
            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
            v_1 = self.inference(z_1.float(), t.float(), cond).double()
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
            eps_2 = (z - alpha_s * x_2) / sigma_s
            v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)

        v = student_diffusion.policy.model(z.float(), t.float() * self.time_scale, cond)
        my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
        return F.mse_loss(w * v.float(), w * v_2.float())


def E_(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def make_diffusion(policy, p_sampler, time_scale=1, gamma=0.3):
    return GuassianDiffusionDefault(policy, p_sampler, time_scale, gamma)

@torch.no_grad()
def transfer_techer_params_to_student(student, teacher):
    for student_param, teacher_param in zip(
        student.parameters(), teacher.parameters()
    ):
        student_param.data.copy_(teacher_param.data)
