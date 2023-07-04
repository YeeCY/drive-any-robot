from typing import Optional, Tuple, Callable
from functools import partial
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder
from stable_contrastive_rl_train.models.utils import (
    identity,
    fanin_init,
)


class CNN(nn.Module):
    def __init__(
            self,
            # num_images: int = 1,
            # num_classes: int = 1000,
            # width_mult: float = 1.0,
            # inverted_residual_setting: Optional[List[List[int]]] = None,
            # round_nearest: int = 8,
            # block: Optional[Callable[..., nn.Module]] = None,
            # norm_layer: Optional[Callable[..., nn.Module]] = None,
            # dropout: = 0.2,
            input_width,
            input_height,
            num_images,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            added_fc_input_size=0,
            norm_type='none',
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
    ) -> None:
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert norm_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = num_images * 3
        self.hidden_activation = hidden_activation
        self.norm_type = norm_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.pool_type = pool_type

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        input_channels = self.input_channels
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                self.pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_sizes[i],
                        stride=pool_strides[i],
                        padding=pool_paddings[i],
                    )
                )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_height,
            self.input_width,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.norm_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.norm_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.norm_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def forward(self, x):
        conv_input = x.narrow(start=0,
                              length=self.conv_input_length,
                              dim=1).contiguous()
        # reshape from batch of flattened images into (channels, h, w)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)
        h = h.view(h.size(0), -1)

        return h


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.,
            layer_norm=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_last_activations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_last_activations:
            return output, h
        else:
            return output


class ContrastiveImgEncoder(nn.Module):
    def __init__(
        self,
        context_size,
        **kwargs,
    ) -> None:
        """
        TODO
        """
        super(ContrastiveImgEncoder, self).__init__()

        self.context_size = context_size

        kwargs["num_images"] = self.context_size + 1
        self.obs_encoder = CNN(
            **kwargs
        )
        kwargs["num_images"] = 1
        self.goal_encoder = CNN(
            **kwargs
        )

        self.obs_encoding_dim = self.obs_encoder.conv_output_flat_size
        self.goal_encoding_dim = self.goal_encoder.conv_output_flat_size

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding = self.obs_encoder(obs_img.flatten(1))
        goal_encoding = self.goal_encoder(goal_img.flatten(1))

        return obs_encoding, goal_encoding


class MobileNetImgEncoder(nn.Module):
    def __init__(self,
                 context_size,
                 **kwargs):
        super(MobileNetImgEncoder, self).__init__()

        mobilenet = MobileNetEncoder(num_images=1)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_dim = 1000
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_dim),
            nn.ReLU(),
        )
        stacked_mobilenet = MobileNetEncoder(num_images=1)  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_dim = 1000
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, self.goal_encoding_dim),
            nn.ReLU(),
        )

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        goal_encoding = self.goal_mobilenet(goal_img)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        return obs_encoding, goal_encoding


class ContrastiveQNetwork(nn.Module):
    def __init__(
        self,
        img_encoder: [ContrastiveImgEncoder, CNN, MobileNetImgEncoder],
        hidden_sizes: list,
        action_size: int,
        representation_dim: int = 16,
        twin_q: bool = True,
        init_w: float = 1e-4,
        hidden_init: str = "xavier_uniform",
        hidden_activation: str = "ReLU",
        layer_norm: bool = False,
    ) -> None:
        """
        TODO
        """
        super(ContrastiveQNetwork, self).__init__()

        self.img_encoder = img_encoder
        self.action_size = action_size
        self.twin_q = twin_q

        # self.waypoint_obs_linear_layers = nn.Sequential(
        #     nn.Linear(self.img_encoder.obs_encoding_size + self.action_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 16)
        # )
        # self.dist_obs_linear_layers = nn.Sequential(
        #     nn.Linear(self.img_encoder.obs_encoding_size + 1, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 16)
        # )
        #
        # self.waypoint_goal_linear_layers = nn.Sequential(
        #     nn.Linear(self.img_encoder.goal_encoding_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 16),
        # )
        # self.dist_goal_linear_layers = nn.Sequential(
        #     nn.Linear(self.img_encoder.goal_encoding_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 16),
        # )
        if hidden_init == "xavier_uniform":
            hidden_init = partial(nn.init.xavier_uniform_,
                                  gain=nn.init.calculate_gain(hidden_activation.lower()))
        elif hidden_init == "fanin":
            hidden_init = fanin_init
        else:
            raise NotImplementedError
        hidden_activation = getattr(nn, hidden_activation)()
        if isinstance(init_w, str):
            init_w = float(init_w)

        if isinstance(self.img_encoder, CNN):
            self.img_encoder.obs_encoding_dim = self.img_encoder.conv_output_flat_size
            self.img_encoder.goal_encoding_dim = self.img_encoder.conv_output_flat_size

        # self.sa_net = Mlp(
        #     hidden_sizes, representation_dim, self.img_encoder.obs_encoding_dim + self.action_size,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )
        # self.sa_net = Mlp(
        #     hidden_sizes, representation_dim, self.img_encoder.obs_encoding_dim,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )
        # self.g_net = Mlp(
        #     hidden_sizes, representation_dim, self.img_encoder.goal_encoding_dim,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )
        # self.sa_net = Mlp(
        #     hidden_sizes, representation_dim, 4,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )
        # self.g_net = Mlp(
        #     hidden_sizes, representation_dim, 2,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )
        self.logit_net = Mlp(
            hidden_sizes, 1,
            self.img_encoder.obs_encoding_dim + self.action_size + self.img_encoder.goal_encoding_dim,
            init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            layer_norm=layer_norm,
        )
        # self.logit_net = Mlp(
        #     hidden_sizes, 1, 6,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )

        if self.twin_q:
            # self.sa_net2 = Mlp(
            #     hidden_sizes, representation_dim, self.img_encoder.obs_encoding_dim + self.action_size,
            #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            #     layer_norm=layer_norm,
            # )
            # self.sa_net2 = Mlp(
            #     hidden_sizes, representation_dim, self.img_encoder.obs_encoding_dim,
            #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            #     layer_norm=layer_norm,
            # )
            # self.g_net2 = Mlp(
            #     hidden_sizes, representation_dim, self.img_encoder.goal_encoding_dim,
            #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            #     layer_norm=layer_norm,
            # )
            # self.sa_net2 = Mlp(
            #     hidden_sizes, representation_dim, 4,
            #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            #     layer_norm=layer_norm,
            # )
            # self.g_net2 = Mlp(
            #     hidden_sizes, representation_dim, 2,
            #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            #     layer_norm=layer_norm,
            # )
            self.logit_net2 = Mlp(
                hidden_sizes, 1,
                self.img_encoder.obs_encoding_dim + self.action_size + self.img_encoder.goal_encoding_dim,
                init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
                layer_norm=layer_norm,
            )
            # self.logit_net2 = Mlp(
            #     hidden_sizes, 1, 6,
            #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            #     layer_norm=layer_norm,
            # )

    def forward(
        self, obs_img: torch.tensor, waypoint: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # augment inputs to match labels size-wise
        waypoint = waypoint.clone().reshape(
            (waypoint.shape[0], 5, 4)
        )  # don't modify the original waypoint
        waypoint[:, 1:, :2] = waypoint[:, 1:, :2] - waypoint[:, :-1, :2].clone()

        waypoint = waypoint.reshape(
            (waypoint.shape[0], self.action_size))

        # dummy_waypoint = torch.zeros_like(waypoint)

        # image shape = (B, C, H, W)
        if isinstance(self.img_encoder, CNN):
            obs_encoding = self.img_encoder(obs_img.reshape(obs_img.shape[0], -1))
            goal_encoding = self.img_encoder(goal_img.reshape(goal_img.shape[0], -1))
        else:
            obs_encoding, goal_encoding = self.img_encoder(obs_img, goal_img)
        # obs_encoding, goal_encoding = obs_img, goal_img

        # sa_repr = self.sa_net(torch.cat([obs_encoding, dummy_waypoint], dim=-1))
        # sa_repr = self.sa_net(torch.cat([obs_encoding, waypoint], dim=-1))
        # g_repr = self.g_net(goal_encoding)
        # outer = torch.bmm(sa_repr.unsqueeze(0), g_repr.permute(1, 0).unsqueeze(0))[0]
        # outer = torch.einsum('ik,jk->ij', sa_repr, g_repr)
        logits = self.logit_net(torch.cat([obs_encoding, waypoint, goal_encoding], dim=-1))

        if self.twin_q:
            # sa_repr2 = self.sa_net2(torch.cat([obs_encoding, dummy_waypoint], dim=-1))
            # sa_repr2 = self.sa_net2(torch.cat([obs_encoding, waypoint], dim=-1))
            # g_repr2 = self.g_net2(goal_encoding)
            logits2 = self.logit_net2(torch.cat([obs_encoding, waypoint, goal_encoding], dim=-1))

            # outer2 = torch.bmm(sa_repr2.unsqueeze(0), g_repr2.permute(1, 0).unsqueeze(0))[0]
            # outer2 = torch.einsum('ik,jk->ij', sa_repr2, g_repr2)
            # outer = torch.stack([outer, outer2], dim=-1)
            logits = torch.cat([logits, logits2], dim=-1)

            # sa_repr = torch.stack([sa_repr, sa_repr2], dim=-1)
            # g_repr = torch.stack([g_repr, g_repr2], dim=-1)

        # return outer, sa_repr, g_repr
        return logits

    # def critic_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
    #     if self.twin_q:
    #         params = chain(self.waypoint_obs_linear_layers.named_parameters(recurse=recurse),
    #                        self.dist_obs_linear_layers.named_parameters(recurse=recurse),
    #                        self.waypoint_goal_linear_layers.named_parameters(recurse=recurse),
    #                        self.dist_goal_linear_layers.named_parameters(recurse=recurse),
    #                        self.waypoint_obs_linear_layers2.named_parameters(recurse=recurse),
    #                        self.dist_obs_linear_layers2.named_parameters(recurse=recurse),
    #                        self.waypoint_goal_linear_layers2.named_parameters(recurse=recurse),
    #                        self.dist_goal_linear_layers2.named_parameters(recurse=recurse))
    #     else:
    #         params = chain(self.waypoint_obs_linear_layers.named_parameters(recurse=recurse),
    #                        self.dist_obs_linear_layers.named_parameters(recurse=recurse),
    #                        self.waypoint_goal_linear_layers.named_parameters(recurse=recurse),
    #                        self.dist_goal_linear_layers.named_parameters(recurse=recurse))
    #
    #     for name, param in params:
    #         yield param


class ContrastivePolicy(nn.Module):
    def __init__(
        self,
        img_encoder: [ContrastiveImgEncoder, CNN],
        hidden_sizes: list,
        action_size: int,
        learn_angle: bool = True,
        std: float = None,
        init_w: float = 1e-4,
        hidden_init: str = "xavier_uniform",
        hidden_activation: str = "ReLU",
        layer_norm: bool = False,
        min_log_std: float = None,
        max_log_std: float = None,
        std_architecture: float = "shared",
    ) -> None:
        """
        TODO
        """
        super(ContrastivePolicy, self).__init__()

        self.img_encoder = img_encoder
        self.action_size = action_size
        self.learn_angle = learn_angle

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.std = std
        self.std_architecture = std_architecture

        if hidden_init == "xavier_uniform":
            hidden_init = partial(nn.init.xavier_uniform_,
                                  gain=nn.init.calculate_gain(hidden_activation.lower()))
        elif hidden_init == "fanin":
            hidden_init = fanin_init
        else:
            raise NotImplementedError
        hidden_activation = getattr(nn, hidden_activation)()
        if isinstance(init_w, str):
            init_w = float(init_w)

        if isinstance(self.img_encoder, CNN):
            self.img_encoder.obs_encoding_dim = self.img_encoder.conv_output_flat_size
            self.img_encoder.goal_encoding_dim = self.img_encoder.conv_output_flat_size

        self.policy_net = Mlp(
            hidden_sizes, self.action_size,
            self.img_encoder.obs_encoding_dim + self.img_encoder.goal_encoding_dim,
            init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
            layer_norm=layer_norm,
        )
        # self.policy_net = Mlp(
        #     hidden_sizes, self.action_size, 4,
        #     init_w=init_w, hidden_activation=hidden_activation, hidden_init=hidden_init,
        #     layer_norm=layer_norm,
        # )

        if std is None:
            if self.std_architecture == "shared":
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, self.action_size)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    torch.zeros(self.action_size, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            log_std = np.log(std)
            assert self.min_log_std <= log_std <= self.max_log_std

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.img_encoder, CNN):
            obs_encoding = self.img_encoder(obs_img.reshape(obs_img.shape[0], -1))
            goal_encoding = self.img_encoder(goal_img.reshape(goal_img.shape[0], -1))
        else:
            obs_encoding, goal_encoding = self.img_encoder(obs_img, goal_img)
        # obs_encoding, goal_encoding = obs_img, goal_img

        waypoint_mu, h = self.policy_net(
            torch.cat([obs_encoding, goal_encoding], dim=-1),
            return_last_activations=True
        )

        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
            waypoint_std = torch.exp(log_std)
        else:
            waypoint_std = torch.from_numpy(
                np.array([self.std, ])).float().to(obs_encoding.device)
            waypoint_std = waypoint_std[None].repeat([obs_encoding.shape[0], self.action_size])

        # augment outputs to match labels size-wise
        waypoint_mu = waypoint_mu.reshape(
            (waypoint_mu.shape[0], 5, 4)  # TODO: use dynamic action size
        )
        waypoint_mu[:, :, :2] = torch.cumsum(
            waypoint_mu[:, :, :2], dim=1
        )
        if self.learn_angle:
            waypoint_mu[:, :, 2:] = F.normalize(
                waypoint_mu[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        waypoint_mu = waypoint_mu.reshape(
            (waypoint_mu.shape[0], self.action_size))

        return waypoint_mu, waypoint_std

    # def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
    #     for name, param in chain(self.linear_layers.named_parameters(recurse=recurse),
    #                              self.waypoint_mu_layers.named_parameters(recurse=recurse),
    #                              self.waypoint_std_layers.named_parameters(recurse=recurse),
    #                              self.dist_mu_layers.named_parameters(recurse=recurse),
    #                              self.dist_std_layers.named_parameters(recurse=recurse)):
    #         yield param
    #
    # def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
    #     gen = chain(
    #         self.linear_layers._named_members(
    #             lambda module: module._parameters.items(),
    #             prefix=prefix, recurse=recurse),
    #         self.waypoint_mu_layers._named_members(
    #             lambda module: module._parameters.items(),
    #             prefix=prefix, recurse=recurse),
    #         self.waypoint_std_layers._named_members(
    #             lambda module: module._parameters.items(),
    #             prefix=prefix, recurse=recurse),
    #         self.dist_mu_layers._named_members(
    #             lambda module: module._parameters.items(),
    #             prefix=prefix, recurse=recurse),
    #         self.dist_std_layers._named_members(
    #             lambda module: module._parameters.items(),
    #             prefix=prefix, recurse=recurse),
    #     )
    #     for elem in gen:
    #         yield elem
