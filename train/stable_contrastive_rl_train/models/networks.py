
import torch
import torch.nn as nn
from torch.distributions import (
    Distribution,
    Normal,
    Independent
)

from typing import List, Dict, Optional, Tuple, Iterator
from itertools import chain

from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder


class ContrastiveImgEncoder(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        """
        TODO
        """
        super(ContrastiveImgEncoder, self).__init__()

        self.context_size = context_size
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = goal_encoding_size

        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )

        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO
        """
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        return obs_encoding, goal_encoding


class ContrastiveQNetwork(nn.Module):
    def __init__(
        self,
        img_encoder: ContrastiveImgEncoder,
        action_size: int,
        twin_q: bool = True,
    ) -> None:
        """
        TODO
        """
        super(ContrastiveQNetwork, self).__init__()
        # self.context_size = context_size
        # self.learn_angle = learn_angle
        # self.len_trajectory_pred = len_traj_pred
        # if self.learn_angle:
        #     self.num_action_params = 4  # last two dims are the cos and sin of the angle
        # else:
        #     self.num_action_params = 2

        self.img_encoder = img_encoder
        # self.context_size = context_size
        self.action_size = action_size
        # self.obs_encoding_size = obs_encoding_size
        # self.goal_encoding_size = goal_encoding_size
        self.twin_q = twin_q

        self.obs_a_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.obs_encoding_size + self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )

        self.goal_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

        if self.twin_q:
            self.obs_a_linear_layers2 = nn.Sequential(
                nn.Linear(self.img_encoder.obs_encoding_size + self.action_size, 256),
                nn.ReLU(),
                nn.Linear(256, 32)
            )

            self.goal_linear_layers2 = nn.Sequential(
                nn.Linear(self.img_encoder.goal_encoding_size, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
            )

    def forward(
        self, obs_img: torch.tensor, action: torch.tensor, goal_img: torch.tensor
    ) -> torch.Tensor:
        """
        TODO
        """
        # obs_encoding = self.obs_mobilenet(obs_img)
        # obs_encoding = self.flatten(obs_encoding)
        # obs_encoding = self.compress_observation(obs_encoding)
        # obs_a_encoding = self.obs_a_linear_layers(
        #     torch.cat([obs_encoding, action], dim=-1))
        #
        # obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        # goal_encoding = self.goal_mobilenet(obs_goal_input)
        # goal_encoding = self.flatten(goal_encoding)
        # goal_encoding = self.compress_goal(goal_encoding)
        # goal_encoding = self.goal_linear_layers(goal_encoding)
        obs_encoding, goal_encoding = self.img_encoder(obs_img, goal_img)
        obs_a_encoding = self.obs_a_linear_layers(
            torch.cat([obs_encoding, action], dim=-1))
        goal_encoding = self.goal_linear_layers(goal_encoding)

        outer = torch.einsum('ik,jk->ij', obs_a_encoding, goal_encoding)

        if self.twin_q:
            obs_encoding2, goal_encoding2 = self.img_encoder(obs_img, goal_img)
            obs_a_encoding2 = self.obs_a_linear_layers2(
                torch.cat([obs_encoding2, action], dim=-1))
            goal_encoding2 = self.goal_linear_layers2(goal_encoding2)

            outer2 = torch.einsum('ik,jk->ij', obs_a_encoding2, goal_encoding2)
            outer = torch.stack([outer, outer2], dim=-1)

        return outer


class ContrastivePolicy(nn.Module):
    def __init__(
        self,
        img_encoder: ContrastiveImgEncoder,
        action_size: int,
        min_log_std: Optional[float] = -13,
        max_log_std: Optional[float] = -2,
    ) -> None:
        """
        TODO
        """
        pass
        super(ContrastivePolicy, self).__init__()

        self.img_encoder = img_encoder
        self.action_size = action_size
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.obs_encoding_size + self.img_encoder.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.mu_layers = nn.Sequential(
            nn.Linear(32, self.action_size),
        )
        self.log_std_layers = nn.Sequential(
            nn.Linear(32, self.action_size)
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, detach_img_encode: bool = True,
    ) -> Distribution:
        obs_encoding, goal_encoding = self.img_encoder(obs_img, goal_img)

        if detach_img_encode:
            obs_encoding = obs_encoding.detach()
            goal_encoding = goal_encoding.detach()

        obs_goal_encoding = self.linear_layers(
            torch.cat([obs_encoding, goal_encoding], dim=-1))

        mu = self.mu_layers(obs_goal_encoding)
        log_std = self.log_std_layers(obs_goal_encoding)
        log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)

        dist = Independent(Normal(mu, std, validate_args=False),
                           reinterpreted_batch_ndims=1)

        return dist

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for name, param in chain(self.linear_layers.named_parameters(recurse=recurse),
                                 self.mu_layers.named_parameters(recurse=recurse),
                                 self.log_std_layers.named_parameters(recurse=recurse)):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        gen = chain(
            self.linear_layers._named_members(
                lambda module: module._parameters.items(),
                prefix=prefix, recurse=recurse),
            self.mu_layers._named_members(
                lambda module: module._parameters.items(),
                prefix=prefix, recurse=recurse),
            self.log_std_layers._named_members(
                lambda module: module._parameters.items(),
                prefix=prefix, recurse=recurse),
        )
        for elem in gen:
            yield elem

# class ContrastiveQf(nn.Module):
#     def __init__(self,
#                  hidden_sizes,
#                  representation_dim,
#                  action_dim,
#                  obs_dim=None,
#                  use_image_obs=False,
#                  imsize=None,
#                  # img_encoder_type="shared",
#                  img_encoder_arch='cnn',
#                  repr_norm=False,
#                  repr_norm_temp=True,
#                  repr_log_scale=None,
#                  twin_q=False,
#                  **kwargs,
#                  ):
#         super().__init__()
#
#         # self._state_dim = state_dim
#         self._use_image_obs = use_image_obs
#         self._imsize = imsize
#         # self._img_encoder_type = img_encoder_type
#         self._img_encoder_arch = img_encoder_arch
#         self._obs_dim = obs_dim
#         self._action_dim = action_dim
#         self._representation_dim = representation_dim
#         self._repr_norm = repr_norm
#         self._repr_norm_temp = repr_norm_temp
#         self._twin_q = twin_q
#
#         self._img_encoder = None
#         if self._use_image_obs:
#             # TODO (chongyiz): check convolution layer output
#             assert isinstance(imsize, int)
#             cnn_kwargs = kwargs.copy()
#             layer_norm = cnn_kwargs.pop('layer_norm')
#
#             if self._img_encoder_arch == 'cnn':
#                 cnn_kwargs['input_width'] = imsize
#                 cnn_kwargs['input_height'] = imsize
#                 cnn_kwargs['input_channels'] = 3
#                 cnn_kwargs['output_size'] = None
#                 cnn_kwargs['kernel_sizes'] = [8, 4, 3]
#                 cnn_kwargs['n_channels'] = [32, 64, 64]
#                 cnn_kwargs['strides'] = [4, 2, 1]
#                 cnn_kwargs['paddings'] = [2, 1, 1]
#                 cnn_kwargs['conv_normalization_type'] = 'layer' if layer_norm else 'none'
#                 cnn_kwargs['fc_normalization_type'] = 'layer' if layer_norm else 'none'
#                 cnn_kwargs['output_conv_channels'] = True
#                 self._img_encoder = CNN(**cnn_kwargs)
#             elif self._img_encoder_arch == 'res_cnn':
#                 cnn_kwargs['input_width'] = imsize
#                 cnn_kwargs['input_height'] = imsize
#                 cnn_kwargs['input_channels'] = 3
#                 cnn_kwargs['output_size'] = None
#                 cnn_kwargs['kernel_sizes'] = [8, 4, 3]
#                 cnn_kwargs['n_channels'] = [32, 64, 64]
#                 cnn_kwargs['strides'] = [4, 2, 1]
#                 cnn_kwargs['paddings'] = [2, 1, 1]
#                 cnn_kwargs['conv_normalization_type'] = 'layer' if layer_norm else 'none'
#                 cnn_kwargs['fc_normalization_type'] = 'layer' if layer_norm else 'none'
#                 cnn_kwargs['output_conv_channels'] = True
#                 self._img_encoder = ResCNN(**cnn_kwargs)
#             elif self._img_encoder_arch == 'impala_cnn':
#                 raise NotImplementedError
#             else:
#                 raise RuntimeError("Unknown image encoder architecture: {}".format(
#                     self._img_encoder_arch))
#         #     if self._img_encoder_type == 'shared':
#         #         cnn_kwargs['output_size'] = self._state_dim * 2
#         #         self._img_encoder = TwoChannelCNN(**cnn_kwargs)
#         #     elif self._img_encoder_type == 'separate':
#         #         cnn_kwargs['output_size'] = self._state_dim
#         #         self._obs_img_encoder = CNN(**cnn_kwargs)
#         #         self._goal_img_encoder = CNN(**cnn_kwargs)
#         #     else:
#         #         raise RuntimeError("Unknown image encoder type: {}".format(self._img_encoder_type))
#
#         state_dim = self._img_encoder.conv_output_flat_size if self._use_image_obs else self._obs_dim
#
#         self._sa_encoder = Mlp(
#             hidden_sizes, representation_dim, state_dim + self._action_dim,
#             **kwargs,
#         )
#         self._g_encoder = Mlp(
#             hidden_sizes, representation_dim, state_dim,
#             **kwargs,
#         )
#         self._sa_encoder2 = Mlp(
#             hidden_sizes, representation_dim, state_dim + self._action_dim,
#             **kwargs,
#         )
#         self._g_encoder2 = Mlp(
#             hidden_sizes, representation_dim, state_dim,
#             **kwargs,
#         )
#
#         if self._repr_norm_temp:
#             if repr_log_scale is None:
#                 self._repr_log_scale = nn.Parameter(
#                     ptu.zeros(1, requires_grad=True))
#             else:
#                 assert isinstance(repr_log_scale, float)
#                 self._repr_log_scale = repr_log_scale
#
#     # def _unflatten_conv(self, conv_hidden):
#     #     """Normalize observation and goal"""
#     #     imlen = self._imsize * self._imsize * 3
#     #     # img_shape = (-1, self._imsize, self._imsize, 3)
#     #
#     #     # state = torch.reshape(
#     #     #     obs[:, :imlen], img_shape)
#     #     # goal = torch.reshape(
#     #     #     obs[:, imlen:], img_shape)
#     #     state = obs[:, :imlen]
#     #     goal = obs[:, imlen:]
#     #
#     #     return state, goal
#
#     @property
#     def repr_norm(self):
#         return self._repr_norm
#
#     @property
#     def repr_log_scale(self):
#         return self._repr_log_scale
#
#     def _compute_representation(self, obs, action, hidden=None):
#         # The optional input hidden is the image representations. We include this
#         # as an input for the second Q value when twin_q = True, so that the two Q
#         # values use the same underlying image representation.
#         if hidden is None:
#             if self._use_image_obs:
#                 # TODO (chongyiz): check the image shape = (B, C, H, W)?
#                 imlen = self._imsize * self._imsize * 3
#                 state = self._img_encoder(obs[:, :imlen]).flatten(1)
#                 goal = self._img_encoder(obs[:, imlen:]).flatten(1)
#                 # obs = torch.cat([
#                 #     self._img_encoder(obs[:, :imlen]),
#                 #     self._img_encoder(obs[:, imlen:])], dim=-1)
#             else:
#                 state = obs[:, :self._obs_dim]
#                 goal = obs[:, self._obs_dim:]
#         else:
#             state, goal = hidden
#
#         if hidden is None:
#             sa_repr = self._sa_encoder(torch.cat([state, action], dim=-1))
#             g_repr = self._g_encoder(goal)
#         else:
#             sa_repr = self._sa_encoder2(torch.cat([state, action], dim=-1))
#             g_repr = self._g_encoder2(goal)
#
#         if self._repr_norm:
#             sa_repr = sa_repr / torch.norm(sa_repr, dim=1, keepdim=True)
#             g_repr = g_repr / torch.norm(g_repr, dim=1, keepdim=True)
#
#             if self._repr_norm_temp:
#                 sa_repr = sa_repr / torch.exp(self._repr_log_scale)
#
#         return sa_repr, g_repr, (state, goal)
#
#     def forward(self, obs, action, repr=False):
#         # state = obs[:, :self._obs_dim]
#         # goal = obs[:, self._obs_dim:]
#         #
#         # sa_repr = self._sa_encoder(torch.cat([state, action], dim=-1))
#         # g_repr = self._g_encoder(goal)
#         #
#         # if self._repr_norm:
#         #     sa_repr = sa_repr / torch.norm(sa_repr, dim=1, keepdim=True)
#         #     g_repr = g_repr / torch.norm(g_repr, dim=1, keepdim=True)
#         #
#         #     if self._repr_norm_temp:
#         #         sa_repr = sa_repr / torch.exp(self._repr_log_scale)
#
#         sa_repr, g_repr, hidden = self._compute_representation(
#             obs, action)
#         # sa_repr_norm = torch.norm(sa_repr, dim=-1)
#         # g_repr_norm = torch.norm(g_repr, dim=-1)
#         # outer = torch.einsum('ik,jk->ij', sa_repr, g_repr)
#         outer = torch.bmm(sa_repr.unsqueeze(0), g_repr.permute(1, 0).unsqueeze(0))[0]
#
#         if self._twin_q:
#             sa_repr2, g_repr2, _ = self._compute_representation(
#                 obs, action, hidden)
#             # outer2 = torch.einsum('ik,jk->ij', sa_repr2, g_repr2)
#             outer2 = torch.bmm(sa_repr2.unsqueeze(0), g_repr2.permute(1, 0).unsqueeze(0))[0]
#             # assert torch.all(outer2 == tmp_outer2)
#
#             outer = torch.stack([outer, outer2], dim=-1)
#
#         if repr:
#             sa_repr_norm = torch.norm(sa_repr, dim=-1)
#             g_repr_norm = torch.norm(g_repr, dim=-1)
#
#             sa_repr_norm2 = torch.norm(sa_repr2, dim=-1)
#             g_repr_norm2 = torch.norm(g_repr2, dim=-1)
#
#             sa_repr = torch.stack([sa_repr, sa_repr2], dim=-1)
#             g_repr = torch.stack([g_repr, g_repr2], dim=-1)
#             sa_repr_norm = torch.stack([sa_repr_norm, sa_repr_norm2], dim=-1)
#             g_repr_norm = torch.stack([g_repr_norm, g_repr_norm2], dim=-1)
#
#             return outer, sa_repr, g_repr, sa_repr_norm, g_repr_norm
#         else:
#             return outer
