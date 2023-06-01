import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
# from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder
# from gnm_train.models.base_model import BaseModel

from stable_contrastive_rl_train.models.base_model import BaseRLModel
from stable_contrastive_rl_train.models.networks import (
    ContrastiveImgEncoder,
    ContrastiveQNetwork,
    ContrastivePolicy
)


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
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
        twin_q: Optional[bool] = True,
        min_log_std: Optional[float] = -13,
        max_log_std: Optional[float] = -2,
        fixed_std: Optional[list] = None,
        soft_target_tau: Optional[float] = 0.005,
    ) -> None:
        """
        TODO
        """
        super(StableContrastiveRL, self).__init__(context_size, len_traj_pred, learn_angle)

        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = goal_encoding_size
        self.twin_q = twin_q
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.fixed_std = fixed_std
        self.soft_target_tau = soft_target_tau

        # action size = waypoint sizes + distance size
        self.action_size = self.len_trajectory_pred * self.num_action_params + 1

        # mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        # self.obs_mobilenet = mobilenet.features
        # self.compress_observation = nn.Sequential(
        #     nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
        #     nn.ReLU(),
        # )
        # self.obs_a_linear_layers = nn.Sequential(
        #     nn.Linear(self.obs_encoding_size + self.action_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 32)
        # )
        #
        # stacked_mobilenet = MobileNetEncoder(
        #     num_images=2 + self.context_size
        # )  # stack the goal and the current observation
        # self.goal_mobilenet = stacked_mobilenet.features
        # self.compress_goal = nn.Sequential(
        #     nn.Linear(stacked_mobilenet.last_channel, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, self.goal_encoding_size),
        #     nn.ReLU(),
        # )
        # self.goal_linear_layers = nn.Sequential(
        #     nn.Linear(self.goal_encoding_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 32),
        # )

        # self.dist_predictor = nn.Sequential(
        #     nn.Linear(32, 1),
        # )
        # self.action_predictor = nn.Sequential(
        #     nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        # )

        # TODO (chongyiz): trying to remove some of the image encoders and increase batch size
        # self.qf = qf
        self.img_encoder = ContrastiveImgEncoder(
            self.context_size,
            self.obs_encoding_size,
            self.goal_encoding_size
        )
        self.target_img_encoder = ContrastiveImgEncoder(
            self.context_size,
            self.obs_encoding_size,
            self.goal_encoding_size
        )
        self.policy_image_encoder = ContrastiveImgEncoder(
            self.context_size,
            self.obs_encoding_size,
            self.goal_encoding_size
        )

        self.q_network = ContrastiveQNetwork(
            self.img_encoder, self.action_size, self.twin_q)
        self.target_q_network = ContrastiveQNetwork(
            self.target_img_encoder, self.action_size, self.twin_q)
        self.policy_network = ContrastivePolicy(
            self.policy_image_encoder, self.action_size, fixed_std=self.fixed_std)

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
        self, waypoint_obs_img: torch.tensor, dist_obs_img: torch.tensor,
        waypoint: torch.tensor, dist: torch.tensor,
        waypoint_goal_img: torch.tensor, dist_goal_img: torch.tensor,
        stop_grad_actor_img_encoder: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        We want to use DataParallel feature of pytorch
        Args:
            obs_img (torch.Tensor): batch of observations
            action (torch.Tensor): batch of actions
            goal_img (torch.Tensor): batch of goals
        Returns:
            logits (torch.Tensor): predicted logits
            mu (torch.Tensor): predicted policy mean
            std (torch.Tensor): predicted policy std
        """
        waypoint_obs_repr, dist_obs_repr, waypoint_goal_repr, dist_goal_repr = self.q_network(
            waypoint_obs_img, dist_obs_img, waypoint, dist,
            waypoint_goal_img, dist_goal_img)
        target_waypoint_obs_repr, target_dist_obs_repr, target_waypoint_goal_repr, target_dist_goal_repr = \
            self.target_q_network(waypoint_obs_img, dist_obs_img, waypoint, dist,
                                  waypoint_goal_img, dist_goal_img)
        waypoint_mean, dist_mean, waypoint_std, dist_std = self.policy_network(
            waypoint_obs_img, dist_obs_img,
            waypoint_goal_img, dist_goal_img,
            detach_img_encode=stop_grad_actor_img_encoder)

        return waypoint_obs_repr, dist_obs_repr, waypoint_goal_repr, dist_goal_repr, \
               target_waypoint_obs_repr, target_dist_obs_repr, target_waypoint_goal_repr, target_dist_goal_repr, \
               waypoint_mean, dist_mean, waypoint_std, dist_std
