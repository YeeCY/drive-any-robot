
import torch
import torch.nn as nn

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

        mobilenet = MobileNetEncoder(num_images=1 + self.context_size,
                                     norm_layer=nn.InstanceNorm2d)
        self.obs_mobilenet = mobilenet.features
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )

        stacked_mobilenet = MobileNetEncoder(
            num_images=1,
            norm_layer=nn.InstanceNorm2d
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            # nn.Linear(1024, self.goal_encoding_size),
            # nn.ReLU(),
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

        # obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(goal_img)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # use torch.bmm to prevent numerical error
        # outer = torch.einsum('ik,jk->ij', obs_a_encoding, goal_encoding)
        # outer = torch.bmm(obs_a_encoding.unsqueeze(0), goal_encoding.permute(1, 0).unsqueeze(0))[0]

        if self.twin_q:
            obs_encoding2, goal_encoding2 = self.img_encoder(obs_img, goal_img)
            obs_a_encoding2 = self.obs_a_linear_layers2(
                torch.cat([obs_encoding2, action], dim=-1))
            goal_encoding2 = self.goal_linear_layers2(goal_encoding2)

            # outer2 = torch.einsum('ik,jk->ij', obs_a_encoding2, goal_encoding2)
            # outer2 = torch.bmm(obs_a_encoding2.unsqueeze(0), goal_encoding2.permute(1, 0).unsqueeze(0))[0]
            # outer = torch.stack([outer, outer2], dim=-1)

            obs_a_encoding = torch.stack([obs_a_encoding, obs_a_encoding2], dim=-1)
            goal_encoding = torch.stack([goal_encoding, goal_encoding2], dim=-1)

        return obs_a_encoding, goal_encoding

    # def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
    #     if self.twin_q:
    #         chain_named_parameters = chain(
    #             self.obs_a_linear_layers.named_parameters(recurse=recurse),
    #             self.goal_linear_layers.named_parameters(recurse=recurse),
    #             self.obs_a_linear_layers2.named_parameters(recurse=recurse),
    #             self.goal_linear_layers2.named_parameters(recurse=recurse)
    #         )
    #     else:
    #         chain_named_parameters = chain(
    #             self.obs_a_linear_layers.named_parameters(recurse=recurse),
    #             self.goal_linear_layers.named_parameters(recurse=recurse),
    #         )
    #
    #     for name, param in chain_named_parameters:
    #         yield param
    #
    # def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
    #     if self.twin_q:
    #         gen = chain(
    #             self.obs_a_linear_layers._named_members(
    #                 lambda module: module._parameters.items(),
    #                 prefix=prefix, recurse=recurse),
    #             self.goal_linear_layers._named_members(
    #                 lambda module: module._parameters.items(),
    #                 prefix=prefix, recurse=recurse),
    #             self.obs_a_linear_layers2._named_members(
    #                 lambda module: module._parameters.items(),
    #                 prefix=prefix, recurse=recurse),
    #             self.goal_linear_layers2._named_members(
    #                 lambda module: module._parameters.items(),
    #                 prefix=prefix, recurse=recurse),
    #         )
    #     else:
    #         gen = chain(
    #             self.obs_a_linear_layers._named_members(
    #                 lambda module: module._parameters.items(),
    #                 prefix=prefix, recurse=recurse),
    #             self.goal_linear_layers._named_members(
    #                 lambda module: module._parameters.items(),
    #                 prefix=prefix, recurse=recurse),
    #         )
    #
    #     for elem in gen:
    #         yield elem


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return mu, std

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
