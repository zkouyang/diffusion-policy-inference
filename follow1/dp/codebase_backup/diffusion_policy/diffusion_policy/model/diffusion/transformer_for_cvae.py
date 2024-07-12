from typing import Union, Optional, Tuple
import logging
import torch
from einops import rearrange, repeat, reduce
import torch.nn as nn
from torch.autograd import Variable
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class TransformerForCVAE(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 256,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.latent_dim = 32
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T + 2 * T_cond, n_emb))
        self.encd_pos_emb = nn.Parameter(torch.zeros(1, T + T_cond + 1, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.latent_proj = nn.Linear(n_emb, self.latent_dim * 2)
        self.latent_out_proj = nn.Linear(self.latent_dim, n_emb)
        self.query_embed = nn.Embedding(T, n_emb)
        self.cls_embed = nn.Embedding(1, n_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.trans_encoder = None
        self.trans_decoder = None
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="relu",
                    batch_first=True,
                    norm_first=False,
                )
                self.trans_encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_cond_layers
                )
            else:
                self.trans_encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="relu",
                batch_first=True,
                norm_first=False,  # important for stability
            )
            self.trans_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.trans_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layer
            )

        # attention mask
        # if causal_attn:
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
        #     # therefore, the upper triangle should be -inf and others (including diag) should be 0.
        #     sz = T
        #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #     self.register_buffer("mask", mask)

        #     if time_as_cond and obs_as_cond:
        #         S = T_cond
        #         t, s = torch.meshgrid(
        #             torch.arange(T),
        #             torch.arange(S),
        #             indexing='ij'
        #         )
        #         mask = t >= (s-1) # add one dimension since time is the first token in cond
        #         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #         self.register_buffer('memory_mask', mask)

        #     else:
        #         self.memory_mask = None
        # else:
        #     self.mask = None
        #     self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.n_emb = n_emb
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForCVAE):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("encd_pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def _generate_masks(self, seq_length, device):
        mask = (
            torch.triu(torch.ones(seq_length, seq_length, device=device)) == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

        memory_mask = None
        if self.T_cond is not None and self.T_cond != seq_length:
            S = self.T_cond - 1
            t, _ = torch.meshgrid(
                torch.arange(seq_length, device=device),
                torch.arange(S, device=device),
                indexing="ij",
            )
            memory_mask = t >= S  # (seq_length, T_cond)
            memory_mask = (
                memory_mask.float()
                .masked_fill(memory_mask == 0, float("-inf"))
                .masked_fill(memory_mask == 1, float(0.0))
            )

        return mask, memory_mask

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(
        self, sample: torch.Tensor, cond: Optional[torch.Tensor] = None, **kwargs
    ):
        """
        sample: (B,T,input_dim)
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        bs = cond.shape[0]

        input_emb = self.input_emb(sample)  # (B, T, n_emb)
        cond_obs_emb = self.cond_obs_emb(cond)  # (B, To, n_emb)

        # Transformer (encoder)
        trans_encoder_input = torch.cat([input_emb, cond_obs_emb], dim=1)
        position_embeddings = self.pos_emb[:, : trans_encoder_input.shape[1], :]
        trans_encoder_input = self.drop(trans_encoder_input + position_embeddings)
        seq_length = trans_encoder_input.shape[1]
        mask, _ = self._generate_masks(seq_length, trans_encoder_input.device)
        # TODO: bug while using MLP mode
        memory = self.trans_encoder(src=trans_encoder_input, mask=mask)

        # Transformer (decoder)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        position_embeddings = self.pos_emb[:, : memory.shape[1], :]
        memory = self.drop(memory + position_embeddings)

        tgt = torch.zeros_like(query_embed, device=cond.device)
        tgt = self.drop(tgt + query_embed)
        trans_decoder_output = self.trans_decoder(tgt, memory)

        # head
        x = self.ln_f(trans_decoder_output)
        x = self.head(x)
        return x