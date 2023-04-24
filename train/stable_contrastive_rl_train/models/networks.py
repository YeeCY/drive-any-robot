import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple, Iterator, Sequence
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

        # mobilenet = MobileNetEncoder(num_images=1 + self.context_size,
        #                              norm_layer=nn.InstanceNorm2d)
        # self.obs_mobilenet = mobilenet.features
        # self.compress_observation = nn.Sequential(
        #     nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
        #     nn.ReLU(),
        # )
        # stacked_mobilenet = MobileNetEncoder(
        #     num_images=2 + self.context_size,
        #     norm_layer=nn.InstanceNorm2d
        # )  # stack the goal and the current observation
        # self.goal_mobilenet = stacked_mobilenet.features
        # self.compress_goal = nn.Sequential(
        #     nn.Linear(stacked_mobilenet.last_channel, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, self.goal_encoding_size),
        #     nn.ReLU(),
        # )
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

        self.img_encoder = img_encoder
        self.action_size = action_size
        self.twin_q = twin_q

        self.waypoint_obs_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.obs_encoding_size + self.action_size - 1, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )
        self.dist_obs_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.obs_encoding_size + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )

        self.waypoint_goal_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
        )
        self.dist_goal_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
        )

        if self.twin_q:
            self.waypoint_obs_linear_layers2 = nn.Sequential(
                nn.Linear(self.img_encoder.obs_encoding_size + self.action_size - 1, 256),
                nn.ReLU(),
                nn.Linear(256, 16)
            )
            self.dist_obs_linear_layers2 = nn.Sequential(
                nn.Linear(self.img_encoder.obs_encoding_size + 1, 256),
                nn.ReLU(),
                nn.Linear(256, 16)
            )

            self.waypoint_goal_linear_layers2 = nn.Sequential(
                nn.Linear(self.img_encoder.goal_encoding_size, 256),
                nn.ReLU(),
                nn.Linear(256, 16),
            )
            self.dist_goal_linear_layers2 = nn.Sequential(
                nn.Linear(self.img_encoder.goal_encoding_size, 256),
                nn.ReLU(),
                nn.Linear(256, 16),
            )

    def forward(
        self, waypoint_obs_img: torch.tensor, dist_obs_img: torch.tensor,
        waypoint: torch.tensor, dist: torch.tensor,
        waypoint_goal_img: torch.tensor, dist_goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO
        """
        waypoint_obs_encoding, waypoint_goal_encoding = self.img_encoder(
            waypoint_obs_img, waypoint_goal_img)
        dist_obs_encoding, dist_goal_encoding = self.img_encoder(
            dist_obs_img, dist_goal_img)

        waypoint_obs_repr = self.obs_waypoint_linear_layers(
            torch.cat([waypoint_obs_encoding, waypoint], dim=-1))
        dist_obs_repr = self.obs_dist_linear_layers(
            torch.cat([dist_obs_encoding, dist], dim=-1))
        waypoint_goal_repr = self.waypoint_goal_linear_layers(waypoint_goal_encoding)
        dist_goal_repr = self.dist_goal_linear_layers(dist_goal_encoding)

        if self.twin_q:
            # obs_encoding2, goal_encoding2 = self.img_encoder(obs_img, goal_img)
            # reuse encoding from the image encoder
            waypoint_obs_repr2 = self.obs_waypoint_linear_layers2(
                torch.cat([waypoint_obs_encoding, waypoint], dim=-1))
            dist_obs_repr2 = self.obs_dist_linear_layers2(
                torch.cat([dist_obs_encoding, dist], dim=-1))
            waypoint_goal_repr2 = self.waypoint_goal_linear_layers2(waypoint_goal_encoding)
            dist_goal_repr2 = self.dist_goal_linear_layers2(dist_goal_encoding)

            waypoint_obs_repr = torch.stack(
                [waypoint_obs_repr, waypoint_obs_repr2], dim=-1)
            dist_obs_repr = torch.stack(
                [dist_obs_repr, dist_obs_repr2], dim=-1)
            waypoint_goal_repr = torch.stack([waypoint_goal_repr, waypoint_goal_repr2], dim=-1)
            dist_goal_repr = torch.stack([dist_goal_repr, dist_goal_repr2], dim=-1)

        return waypoint_obs_repr, dist_obs_repr, waypoint_goal_repr, dist_goal_repr

    def critic_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if self.twin_q:
            params = chain(self.waypoint_obs_linear_layers.named_parameters(recurse=recurse),
                           self.dist_obs_linear_layers.named_parameters(recurse=recurse),
                           self.waypoint_goal_linear_layers.named_parameters(recurse=recurse),
                           self.dist_goal_linear_layers.named_parameters(recurse=recurse),
                           self.waypoint_obs_linear_layers2.named_parameters(recurse=recurse),
                           self.dist_obs_linear_layers2.named_parameters(recurse=recurse),
                           self.waypoint_goal_linear_layers2.named_parameters(recurse=recurse),
                           self.dist_goal_linear_layers2.named_parameters(recurse=recurse))
        else:
            params = chain(self.waypoint_obs_linear_layers.named_parameters(recurse=recurse),
                           self.dist_obs_linear_layers.named_parameters(recurse=recurse),
                           self.waypoint_goal_linear_layers.named_parameters(recurse=recurse),
                           self.dist_goal_linear_layers.named_parameters(recurse=recurse))

        for name, param in params:
            yield param


class ContrastivePolicy(nn.Module):
    def __init__(
        self,
        img_encoder: ContrastiveImgEncoder,
        action_size: int,
        min_std: Optional[float] = 1e-6,
        fixed_std: Optional[list] = None,
    ) -> None:
        """
        TODO
        """
        super(ContrastivePolicy, self).__init__()

        self.img_encoder = img_encoder
        self.action_size = action_size
        self.min_std = min_std
        self.fixed_std = fixed_std
        self.learn_angle = True

        self.waypoint_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.obs_encoding_size + self.img_encoder.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.dist_linear_layers = nn.Sequential(
            nn.Linear(self.img_encoder.obs_encoding_size + self.img_encoder.goal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.waypoint_mu_layers = nn.Sequential(
            nn.Linear(32, self.action_size - 1),
        )
        self.dist_mu_layers = nn.Sequential(
            nn.Linear(32, 1),
        )

        if self.fixed_std is None:
            # (chongyiz): learnable std doesn't work now.
            self.waypoint_std_layers = nn.Sequential(
                nn.Linear(32, self.action_size - 1),
                nn.Softplus(),
            )
            self.dist_std_layers = nn.Sequential(
                nn.Linear(32, 1),
                nn.Softplus(),
            )

    def forward(
        self, waypoint_obs_img: torch.tensor, dist_obs_img: torch.tensor,
        waypoint_goal_img: torch.tensor, dist_goal_img: torch.tensor,
        detach_img_encode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        waypoint_obs_encoding, waypoint_goal_encoding = self.img_encoder(waypoint_obs_img, waypoint_goal_img)
        dist_obs_encoding, dist_goal_encoding = self.img_encoder(dist_obs_img, dist_goal_img)
        device = waypoint_obs_encoding.device
        batch_size = waypoint_obs_encoding.shape[0]

        # DELETEME (chongyiz)
        # if detach_img_encode:
        #     obs_encoding = obs_encoding.detach()
        #     goal_encoding = goal_encoding.detach()

        # TODO (start from here)

        waypoint_obs_goal_encoding = self.waypoint_linear_layers(
            torch.cat([waypoint_obs_encoding, waypoint_goal_encoding], dim=-1))
        dist_obs_goal_encoding = self.dist_linear_layers(
            torch.cat([dist_obs_encoding, dist_goal_encoding], dim=-1))

        waypoint_mu = self.waypoint_mu_layers(waypoint_obs_goal_encoding)
        dist_mu = self.dist_mu_layers(dist_obs_goal_encoding)
        if self.fixed_std is None:
            waypoint_std = self.waypoint_std_layers(waypoint_obs_goal_encoding) + self.min_std
            dist_std = self.dist_std_layers(dist_obs_goal_encoding) + self.min_std

            std = torch.cat([waypoint_std, dist_std], dim=-1)
        else:
            std = torch.from_numpy(np.array([self.fixed_std, ])).float().to(
                device)
            std = std.repeat(batch_size, 1)

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
            (waypoint_mu.shape[0], self.action_size - 1))

        mu = torch.cat([waypoint_mu, dist_mu], dim=-1)

        return mu, std

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
