import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from gnm_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    RandomizedClassBalancer,
    get_image_path,
    to_local_coords,
)

from stable_contrastive_rl_train.data.data_utils import GeometricClassBalancer


class RLDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        is_action: bool,
        transform: transforms,
        aspect_ratio: float,
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        discount: float,
        len_traj_pred: int,
        learn_angle: bool,
        oracle_angles: float,
        num_oracle_trajs: int,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
    ):
        """
        Main GNM dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            is_action (bool): Whether to use the action dataset or the distance dataset
            transform (transforms): Transforms to apply to the image data
            aspect_ratio (float): Aspect ratio of the images (w/h)
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.is_action = is_action
        self.dataset_name = dataset_name
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.transform = transform
        self.aspect_ratio = aspect_ratio
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.discount = discount
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.oracle_angles = oracle_angles
        self.num_oracle_trajs = num_oracle_trajs

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self._gen_index_to_data()

    def _gen_index_to_data(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """

        self.index_to_data = []
        if self.discount >= 0:
            self.label_balancer = GeometricClassBalancer(self.distance_categories, self.discount)
        else:
            self.label_balancer = RandomizedClassBalancer(self.distance_categories)

        dataset_type = "action" if self.is_action else "distance"
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"rl_dataset_type_{dataset_type}_waypoint_spacing_{self.waypoint_spacing}_min_dist_cat_{self.min_dist_cat}_max_dist_cat_{self.max_dist_cat}_discount_{self.discount}_len_traj_pred_{self.len_traj_pred}_learn_angle_{self.learn_angle}_context_size_{self.context_size}_context_type_{self.context_type}_end_slack_{self.end_slack}_goals_per_obs_{self.goals_per_obs}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f1:
                self.index_to_data = pickle.load(f1)
        except:
            # if the index_to_data file doesn't exist, create it
            print(
                f"Sampling subgoals for each observation in the {self.dataset_name} {dataset_type} rl dataset..."
            )
            print(
                "This will take a while, but it will only be done once for each configuration per dataset."
            )
            for i in tqdm.tqdm(range(len(self.traj_names))):
                f_curr = self.traj_names[i]
                with open(
                    os.path.join(
                        os.path.join(self.data_folder, f_curr), "traj_data.pkl"
                    ),
                    "rb",
                ) as f3:
                    traj_data = pickle.load(f3)
                traj_len = len(traj_data["position"])
                # start sampling a little bit into the trajectory to give enought time to generate context
                for curr_time in range(
                    self.context_size * self.waypoint_spacing,
                    traj_len - self.end_slack,
                ):
                    max_len = min(
                        int(self.max_dist_cat * self.waypoint_spacing),
                        traj_len - curr_time - 1,
                    )
                    # sampled_dists = []

                    # sample self.goals_per_obs goals per observation
                    for _ in range(self.goals_per_obs):
                        # sample a distance from the distance categories as long as it is less than the trajectory length
                        filter_func = (
                            lambda dist: int(dist * self.waypoint_spacing) <= max_len
                        )
                        len_to_goal = self.label_balancer.sample(filter_func)
                        # sampled_dists.append(len_to_goal)

                        # break the loop if there are no more valid distances to sample
                        if len_to_goal is None:
                            break

                        # if the length to the goal is negative, then we are using negative mining (sample an goal from another trajectory)
                        if len_to_goal == -1:
                            new = np.random.randint(1, len(self.traj_names))
                            f_rand = self.traj_names[(i + new) % len(self.traj_names)]
                            with open(
                                os.path.join(self.data_folder, f_rand, "traj_data.pkl"),
                                "rb",
                            ) as f4:
                                rand_traj_data = pickle.load(f4)
                            rand_traj_len = len(rand_traj_data["position"])
                            goal_time = np.random.randint(rand_traj_len)
                            f_goal = f_rand
                        else:
                            goal_time = curr_time + int(
                                len_to_goal * self.waypoint_spacing
                            )
                            f_goal = f_curr
                        self.index_to_data += [(f_curr, f_goal, curr_time, goal_time)]
            with open(index_to_data_path, "wb") as f2:
                pickle.dump(self.index_to_data, f2)

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation for visualization
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image for visualization
                transf_obs_image (torch.Tensor): tensor of shape [(context_size + 1) * 3, H, W] containing the images of the context and the observation after transformation for training
                transf_goal_image (torch.Tensor): tensor of shape [(context_size + 1) * 3, H, W] containing the images of the goals after transformation for training
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                dataset_index (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        # f_curr, f_goal, curr_time, goal_time = self.index_to_data[i]
        f_curr, _, curr_time, _ = self.index_to_data[i]
        # We need to resample goal for each data
        with open(os.path.join(self.data_folder, f_curr, "traj_data.pkl"), "rb") as f:
            curr_traj_data = pickle.load(f)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        max_len = min(
            int(self.max_dist_cat * self.waypoint_spacing),
            curr_traj_len - curr_time - 1,
        )

        # sample a distance from the distance categories as long as it is less than the trajectory length
        filter_func = (
            lambda dist: int(dist * self.waypoint_spacing) <= max_len
        )
        len_to_goal = self.label_balancer.sample(filter_func)

        # if the length to the goal is negative, then we are using negative mining (sample an goal from another trajectory)
        if len_to_goal == -1:
            new = np.random.randint(1, len(self.traj_names))
            f_rand = self.traj_names[(i + new) % len(self.traj_names)]
            with open(
                os.path.join(self.data_folder, f_rand, "traj_data.pkl"),
                "rb",
            ) as f4:
                rand_traj_data = pickle.load(f4)
            rand_traj_len = len(rand_traj_data["position"])
            goal_time = np.random.randint(rand_traj_len)
            f_goal = f_rand
        else:
            goal_time = curr_time + int(
                len_to_goal * self.waypoint_spacing
            )
            f_goal = f_curr

        transf_obs_images = []
        transf_next_obs_images = []
        if self.context_type == "randomized":
            # sample self.context_size random times from interval [0, curr_time) with no replacement
            context_times = np.random.choice(
                list(range(curr_time)), self.context_size, replace=False
            )
            context_times.append(curr_time)
            context = [(f_curr, t) for t in context_times]

            # TODO (chongyiz): add next obs context
        elif self.context_type == "randomized_temporal":
            f_rand_curr, _, rand_curr_time, _ = self.index_to_data[
                np.random.randint(0, len(self))
            ]
            context_times = list(
                range(
                    rand_curr_time + -self.context_size * self.waypoint_spacing,
                    rand_curr_time,
                    self.waypoint_spacing,
                )
            )
            context = [(f_rand_curr, t) for t in context_times]
            context.append((f_curr, curr_time))

            # TODO (chongyiz): add next obs context
        elif self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
            next_context = [(f_curr, t + 1) for t in context_times]
        else:
            raise ValueError(f"Invalid type {self.context_type}")
        for (f, t), (next_f, next_t) in zip(context, next_context):
            obs_image_path = get_image_path(self.data_folder, f, t)
            obs_image, transf_obs_image = img_path_to_data(
                obs_image_path,
                self.transform,
                self.aspect_ratio,
            )
            transf_obs_images.append(transf_obs_image)

            next_obs_image_path = get_image_path(self.data_folder, next_f, next_t)
            next_obs_image, transf_next_obs_image = img_path_to_data(
                next_obs_image_path,
                self.transform,
                self.aspect_ratio,
            )
            transf_next_obs_images.append(transf_next_obs_image)

        transf_obs_image = torch.cat(transf_obs_images, dim=0)
        transf_next_obs_image = torch.cat(transf_next_obs_images, dim=0)

        with open(os.path.join(self.data_folder, f_goal, "traj_data.pkl"), "rb") as f:
            goal_traj_data = pickle.load(f)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"
        goal_image_path = get_image_path(self.data_folder, f_goal, goal_time)
        goal_image, transf_goal_image = img_path_to_data(
            goal_image_path,
            self.transform,
            self.aspect_ratio,
        )

        data = [
            obs_image,
            next_obs_image,
            goal_image,
            transf_obs_image,
            transf_next_obs_image,
            transf_goal_image,
        ]

        if self.is_action:
            spacing = self.waypoint_spacing
            traj_len = min(self.len_traj_pred, (goal_time - curr_time) // spacing)
            pos_goal = goal_traj_data["position"][goal_time, :2]
            pos_list = curr_traj_data["position"][
                curr_time : curr_time + (traj_len + 1) * spacing : spacing,
                :2,
            ]
            if self.learn_angle:
                pos_goal = np.concatenate(
                    (
                        pos_goal,
                        np.array(goal_traj_data["yaw"][goal_time]).reshape(1),
                    ),
                    axis=0,
                )
                pos_list_angle = curr_traj_data["yaw"][
                    curr_time : curr_time + (traj_len + 1) * spacing : spacing
                ]
                pos_list = np.concatenate(
                    (pos_list, pos_list_angle.reshape(len(pos_list_angle), 1)),
                    axis=1,
                )
                param_dim = 3
            else:
                param_dim = 2

            goals_appendage = pos_goal * np.ones(
                (self.len_traj_pred - traj_len, param_dim)
            )
            pos_list = np.concatenate((pos_list, goals_appendage), axis=0)  # (x, y, angle)
            pos_nplist = np.array(pos_list[1:])
            yaw = curr_traj_data["yaw"][curr_time]
            waypoints = to_local_coords(pos_nplist, pos_list[0], yaw)
            waypoints = torch.Tensor(waypoints.astype(float))
            goal = to_local_coords(pos_goal, pos_list[0], yaw)
            goal = torch.Tensor(goal.astype(float))
            if self.learn_angle:  # localize the waypoint angles
                waypoints[1:, 2] -= waypoints[0, 2]
                waypoints = calculate_sin_cos(waypoints)
            if self.normalize:
                waypoints[:, :2] /= (
                    self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )  # only divide the dx and dy
                goal[:2] /= (
                    self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )
            data.extend(
                [
                    goal,
                    waypoints,
                ]
            )

            # TODO (chongyi): we get sin/cos bugs here! Fix it.
            # construct oracle actions spanning -oracle_angles to oracle_angles
            # traj_len = torch.sum(
            #     torch.linalg.norm(waypoints[1:, :2] - waypoints[:-1, :2], dim=-1)
            # )
            # oracle_angles = torch.linspace(
            #     -np.deg2rad(self.oracle_angles), np.deg2rad(self.oracle_angles),
            #     self.num_oracle_trajs)
            # if traj_len > 0:
            #     oracle_end_wpts = torch.stack([
            #         traj_len * torch.cos(oracle_angles),
            #         traj_len * torch.sin(oracle_angles),
            #     ], dim=-1)
            # else:
            #     oracle_end_wpts = torch.stack([
            #         torch.zeros_like(oracle_angles),
            #         torch.zeros_like(oracle_angles)
            #     ], dim=-1)
            # oracle_waypoints = oracle_end_wpts[:, None] * \
            #                    torch.linspace(0, 1, self.len_traj_pred)[None, :, None]
            # first_waypoint = waypoints[0]
            # # first_waypoint_dist = torch.linalg.norm(first_waypoint[:2])
            # # waypoint_offset = torch.stack([
            # #     first_waypoint_dist * torch.cos(oracle_angles),
            # #     first_waypoint_dist * torch.sin(oracle_angles),
            # # ], dim=-1)
            # oracle_waypoints += first_waypoint[:2]
            # if self.learn_angle:
            #     oracle_waypoints = torch.cat([
            #         oracle_waypoints,
            #         torch.cos(oracle_angles)[:, None, None].repeat_interleave(self.len_traj_pred, axis=1),
            #         torch.sin(oracle_angles)[:, None, None].repeat_interleave(self.len_traj_pred, axis=1),
            #     ], dim=-1)
            # else:
            #     oracle_waypoints = torch.cat([
            #         oracle_waypoints,
            #         oracle_angles[:, None, None].repeat_interleave(self.len_traj_pred, dim=1),
            #     ], dim=-1)
            # data.append(oracle_waypoints)

            # pos_list = np.concatenate((pos_list, goals_appendage), axis=0)  # (x, y, angle)
            # pos_nplist = np.array(pos_list[1:])
            oracle_angles = np.linspace(
                -np.deg2rad(self.oracle_angles), np.deg2rad(self.oracle_angles),
                self.num_oracle_trajs)
            # yaw = curr_traj_data["yaw"][curr_time]

            oracle_waypoints = []
            for oracle_angle in oracle_angles:
                pos_np = np.array(pos_list[1:])
                # pos_np[:, 2] -= oracle_angle
                oracle_waypoint = to_local_coords(pos_np, pos_list[0], yaw - oracle_angle)

                oracle_waypoints.append(oracle_waypoint)
            oracle_waypoints = np.asarray(oracle_waypoints)
            oracle_waypoints = torch.Tensor(oracle_waypoints.astype(float))

            if self.learn_angle:  # localize the waypoint angles
                oracle_waypoints[:, 1:, 2] -= oracle_waypoints[:, [0], 2]
                oracle_waypoints = calculate_sin_cos(oracle_waypoints)
            if self.normalize:
                oracle_waypoints[..., :2] /= (
                    self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )  # only divide the dx and dy
            data.append(oracle_waypoints)

            data.append(torch.Tensor(pos_list[0].astype(float)))
            data.append(torch.Tensor(np.array(yaw).astype(float)))
            # data.append(torch.Tensor(pos_list.astype(float)))
        else:
            # temporal distance
            if f_curr == f_goal:
                dist_label = torch.FloatTensor(
                    [(goal_time - curr_time) / self.waypoint_spacing]
                )
            else:  # negative mining
                dist_label = torch.FloatTensor([self.max_dist_cat])
            data.append(dist_label)

        data.append(torch.LongTensor([self.dataset_index]))
        return tuple(data)


class RLEvalDataset(RLDataset):
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        # f_curr, f_goal, curr_time, goal_time = self.index_to_data[i]
        f_curr, _, curr_time, _ = self.index_to_data[i]
        # We need to resample goal for each data
        with open(os.path.join(self.data_folder, f_curr, "traj_data.pkl"), "rb") as f:
            curr_traj_data = pickle.load(f)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        max_len = min(
            int(self.max_dist_cat * self.waypoint_spacing),
            curr_traj_len - curr_time - 1,
        )

        # sample a distance from the distance categories as long as it is less than the trajectory length
        filter_func = (
            lambda dist: int(dist * self.waypoint_spacing) <= max_len
        )
        len_to_goal = self.label_balancer.sample(filter_func)

        # if the length to the goal is negative, then we are using negative mining (sample an goal from another trajectory)
        if len_to_goal == -1:
            new = np.random.randint(1, len(self.traj_names))
            f_rand = self.traj_names[(i + new) % len(self.traj_names)]
            with open(
                    os.path.join(self.data_folder, f_rand, "traj_data.pkl"),
                    "rb",
            ) as f4:
                rand_traj_data = pickle.load(f4)
            rand_traj_len = len(rand_traj_data["position"])
            goal_time = np.random.randint(rand_traj_len)
            f_goal = f_rand
        else:
            goal_time = curr_time + int(
                len_to_goal * self.waypoint_spacing
            )
            f_goal = f_curr

        index_to_data = {
            "f_curr": f_curr,
            "curr_time": curr_time,
            "f_goal": f_goal,
            "goal_time": goal_time,
        }

        transf_obs_images = []
        transf_next_obs_images = []
        if self.context_type == "randomized":
            # sample self.context_size random times from interval [0, curr_time) with no replacement
            context_times = np.random.choice(
                list(range(curr_time)), self.context_size, replace=False
            )
            context_times.append(curr_time)
            context = [(f_curr, t) for t in context_times]

            # TODO (chongyiz): add next obs context
        elif self.context_type == "randomized_temporal":
            f_rand_curr, _, rand_curr_time, _ = self.index_to_data[
                np.random.randint(0, len(self))
            ]
            context_times = list(
                range(
                    rand_curr_time + -self.context_size * self.waypoint_spacing,
                    rand_curr_time,
                    self.waypoint_spacing,
                )
            )
            context = [(f_rand_curr, t) for t in context_times]
            context.append((f_curr, curr_time))

            # TODO (chongyiz): add next obs context
        elif self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
            next_context = [(f_curr, t + 1) for t in context_times]
        else:
            raise ValueError(f"Invalid type {self.context_type}")
        for (f, t), (next_f, next_t) in zip(context, next_context):
            obs_image_path = get_image_path(self.data_folder, f, t)
            obs_image, transf_obs_image = img_path_to_data(
                obs_image_path,
                self.transform,
                self.aspect_ratio,
            )
            transf_obs_images.append(transf_obs_image)

            next_obs_image_path = get_image_path(self.data_folder, next_f, next_t)
            next_obs_image, transf_next_obs_image = img_path_to_data(
                next_obs_image_path,
                self.transform,
                self.aspect_ratio,
            )
            transf_next_obs_images.append(transf_next_obs_image)

        transf_obs_image = torch.cat(transf_obs_images, dim=0)
        transf_next_obs_image = torch.cat(transf_next_obs_images, dim=0)

        with open(os.path.join(self.data_folder, f_goal, "traj_data.pkl"), "rb") as f:
            goal_traj_data = pickle.load(f)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"
        goal_image_path = get_image_path(self.data_folder, f_goal, goal_time)
        goal_image, transf_goal_image = img_path_to_data(
            goal_image_path,
            self.transform,
            self.aspect_ratio,
        )

        data = [
            obs_image,
            next_obs_image,
            goal_image,
            transf_obs_image,
            transf_next_obs_image,
            transf_goal_image,
        ]

        if self.is_action:
            spacing = self.waypoint_spacing
            traj_len = min(self.len_traj_pred, (goal_time - curr_time) // spacing)
            pos_goal = goal_traj_data["position"][goal_time, :2]
            pos_list = curr_traj_data["position"][
                       curr_time: curr_time + (traj_len + 1) * spacing: spacing,
                       :2,
                       ]
            if self.learn_angle:
                pos_goal = np.concatenate(
                    (
                        pos_goal,
                        np.array(goal_traj_data["yaw"][goal_time]).reshape(1),
                    ),
                    axis=0,
                )
                pos_list_angle = curr_traj_data["yaw"][
                                 curr_time: curr_time + (traj_len + 1) * spacing: spacing
                                 ]
                pos_list = np.concatenate(
                    (pos_list, pos_list_angle.reshape(len(pos_list_angle), 1)),
                    axis=1,
                )
                param_dim = 3
            else:
                param_dim = 2

            goals_appendage = pos_goal * np.ones(
                (self.len_traj_pred - traj_len, param_dim)
            )
            pos_list = np.concatenate((pos_list, goals_appendage), axis=0)  # (x, y, angle)
            pos_nplist = np.array(pos_list[1:])
            yaw = curr_traj_data["yaw"][curr_time]
            waypoints = to_local_coords(pos_nplist, pos_list[0], yaw)
            waypoints = torch.Tensor(waypoints.astype(float))
            goal = to_local_coords(pos_goal, pos_list[0], yaw)
            goal = torch.Tensor(goal.astype(float))
            if self.learn_angle:  # localize the waypoint angles
                waypoints[1:, 2] -= waypoints[0, 2]
                waypoints = calculate_sin_cos(waypoints)
            if self.normalize:
                waypoints[:, :2] /= (
                        self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )  # only divide the dx and dy
                goal[:2] /= (
                        self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )
            data.extend(
                [
                    goal,
                    waypoints,
                ]
            )

            # pos_list = np.concatenate((pos_list, goals_appendage), axis=0)  # (x, y, angle)
            # pos_nplist = np.array(pos_list[1:])
            oracle_angles = np.linspace(
                -np.deg2rad(self.oracle_angles), np.deg2rad(self.oracle_angles),
                self.num_oracle_trajs)
            # yaw = curr_traj_data["yaw"][curr_time]

            oracle_waypoints = []
            for oracle_angle in oracle_angles:
                pos_np = np.array(pos_list[1:])
                # pos_np[:, 2] -= oracle_angle
                oracle_waypoint = to_local_coords(pos_np, pos_list[0], yaw - oracle_angle)

                oracle_waypoints.append(oracle_waypoint)
            oracle_waypoints = np.asarray(oracle_waypoints)
            oracle_waypoints = torch.Tensor(oracle_waypoints.astype(float))

            if self.learn_angle:  # localize the waypoint angles
                oracle_waypoints[:, 1:, 2] -= oracle_waypoints[:, [0], 2]
                oracle_waypoints = calculate_sin_cos(oracle_waypoints)
            if self.normalize:
                oracle_waypoints[..., :2] /= (
                        self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )  # only divide the dx and dy
            data.append(oracle_waypoints)

            data.append(torch.Tensor(pos_list[0].astype(float)))
            data.append(torch.Tensor(np.array(yaw).astype(float)))
            # data.append(torch.Tensor(pos_list.astype(float)))
        else:
            # temporal distance
            if f_curr == f_goal:
                dist_label = torch.FloatTensor(
                    [(goal_time - curr_time) / self.waypoint_spacing]
                )
            else:  # negative mining
                dist_label = torch.FloatTensor([self.max_dist_cat])
            data.append(dist_label)

        data.append(torch.LongTensor([self.dataset_index]))
        data.append(index_to_data)

        return tuple(data)

