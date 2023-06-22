import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from functools import partial
# from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder
# from gnm_train.models.base_model import BaseModel

from stable_contrastive_rl_train.models.base_model import BaseRLModel
from stable_contrastive_rl_train.models.networks import (
    CNN,
    ContrastiveImgEncoder,
    ContrastiveQNetwork,
    ContrastivePolicy
)
from stable_contrastive_rl_train.models.utils import fanin_init


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class StableContrastiveRL(BaseRLModel):
    def __init__(
        self,
        context_size: int = 1,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        soft_target_tau: Optional[float] = 0.005,
        img_encoder_kwargs=None,
        contrastive_critic_kwargs=None,
        policy_kwargs=None,
    ) -> None:
        """
        TODO
        """
        super(StableContrastiveRL, self).__init__(context_size, len_traj_pred, learn_angle)

        self.soft_target_tau = soft_target_tau

        # action size = waypoint sizes + distance size
        self.action_size = self.len_trajectory_pred * self.num_action_params

        assert img_encoder_kwargs is not None
        if img_encoder_kwargs["hidden_init"] == "xavier_uniform":
            hidden_init = partial(nn.init.xavier_uniform_,
                                  gain=nn.init.calculate_gain(
                                      img_encoder_kwargs["hidden_activation"].lower()))
        elif img_encoder_kwargs["hidden_init"] == "fanin":
            hidden_init = fanin_init
        else:
            raise NotImplementedError
        img_encoder_kwargs["hidden_init"] = hidden_init

        hidden_activation = getattr(nn, img_encoder_kwargs["hidden_activation"])()
        img_encoder_kwargs["hidden_activation"] = hidden_activation

        # TODO (chongyi): delete if statement for context_size == 0.
        if self.context_size == 0:
            img_encoder_kwargs["num_images"] = 1
            self.img_encoder = CNN(
                **img_encoder_kwargs
            )
            self.target_img_encoder = CNN(
                **img_encoder_kwargs
            )
            self.policy_image_encoder = CNN(
                **img_encoder_kwargs
            )
        else:
            self.img_encoder = ContrastiveImgEncoder(
                self.context_size,
                **img_encoder_kwargs
            )
            self.target_img_encoder = ContrastiveImgEncoder(
                self.context_size,
                **img_encoder_kwargs
            )
            self.policy_img_encoder = ContrastiveImgEncoder(
                self.context_size,
                **img_encoder_kwargs
            )

        assert contrastive_critic_kwargs is not None
        contrastive_critic_kwargs["action_size"] = self.action_size
        self.q_network = ContrastiveQNetwork(
            self.img_encoder, **contrastive_critic_kwargs)
        self.target_q_network = ContrastiveQNetwork(
            self.target_img_encoder, **contrastive_critic_kwargs)

        assert policy_kwargs is not None
        policy_kwargs["action_size"] = self.action_size
        policy_kwargs["learn_angle"] = self.learn_angle
        if self.context_size == 0:
            self.policy_network = ContrastivePolicy(
                self.policy_image_encoder, **policy_kwargs)
        else:
            self.policy_network = ContrastivePolicy(
                self.policy_img_encoder, **policy_kwargs)

        # copy_model_params_from_to(self.q_network.critic_parameters(),
        #                           self.target_q_network.critic_parameters())
        copy_model_params_from_to(self.q_network, self.target_q_network)

        # self.soft_target_tau = soft_target_tau
        # self.target_update_period = target_update_period
        # self.gradient_clipping = gradient_clipping

        # self.use_td = use_td
        # self.multiply_batch_size_scale
        # self.entropy_coefficient = entropy_coefficient
        # self.adaptive_entropy_coefficient = entropy_coefficient is None
        # self.target_entropy = target_entropy
        # self.discount = discount

    # @property
    # def q_network_parameters(self):
    #     return self.q_network.parameters()

    # @property
    # def policy_network_parameters(self):
    #     return self.policy_network.parameters()

    def soft_update_target_q_network(self):
        soft_update_from_to(
            self.q_network, self.target_q_network, self.soft_target_tau
        )

    # def forward(
    #     self, *x
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # obs_encoding = self.obs_mobilenet(obs_img)
    #     # obs_encoding = self.flatten(obs_encoding)
    #     # obs_encoding = self.compress_observation(obs_encoding)
    #     # obs_a_encoder = self.obs_a_linear_layers(
    #     #     torch.cat([obs_encoding, action], dim=-1))
    #     #
    #     # obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
    #     # goal_encoding = self.goal_mobilenet(obs_goal_input)
    #     # goal_encoding = self.flatten(goal_encoding)
    #     # goal_encoding = self.compress_goal(goal_encoding)
    #     # goal_encoding = self.goal_linear_layers(goal_encoding)
    #     #
    #     # z = torch.cat([obs_encoding, goal_encoding], dim=1)
    #     # z = self.linear_layers(z)
    #     # # dist_pred = self.dist_predictor(z)
    #     # # action_pred = self.action_predictor(z)
    #     #
    #     # # augment outputs to match labels size-wise
    #     # action_pred = action_pred.reshape(
    #     #     (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
    #     # )
    #     # action_pred[:, :, :2] = torch.cumsum(
    #     #     action_pred[:, :, :2], dim=1
    #     # )  # convert position deltas into waypoints
    #     # if self.learn_angle:
    #     #     action_pred[:, :, 2:] = F.normalize(
    #     #         action_pred[:, :, 2:].clone(), dim=-1
    #     #     )  # normalize the angle prediction
    #     #
    #     # return dist_pred, action_pred

    def forward(
        self, obs_img: torch.tensor, waypoint: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO docstring
        """
        obs_a_repr, goal_repr = self.q_network(
            obs_img, waypoint, goal_img)
        target_obs_a_repr, target_goal_repr = self.target_q_network(
            obs_img, waypoint, goal_img)
        waypoint_mean, waypoint_std = self.policy_network(
            obs_img, goal_img)

        return obs_a_repr, goal_repr, target_obs_a_repr, target_goal_repr, waypoint_mean, waypoint_std
