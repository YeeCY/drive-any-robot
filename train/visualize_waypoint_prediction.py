import os
import wandb
import argparse
import numpy as np
import yaml
import glob
import pickle as pkl
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as TF
# import time
#
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.backends.cudnn as cudnn
#
# from gnm_train.models.gnm import GNM
# from gnm_train.models.siamese import SiameseModel
# from gnm_train.models.stacked import StackedModel
# from gnm_train.data.gnm_dataset import GNM_Dataset
# # from gnm_train.data.pairwise_distance_dataset import PairwiseDistanceDataset
# from gnm_train.data.pairwise_distance_dataset import (
#     PairwiseDistanceEvalDataset,
#     PairwiseDistanceFailureDataset
# )
# from gnm_train.training.train_utils import load_model
# from gnm_train.evaluation.eval_utils import eval_loop
#
# from stable_contrastive_rl_train.data.rl_dataset import RLDataset
# from stable_contrastive_rl_train.models.base_model import DataParallel
# from stable_contrastive_rl_train.models.stable_contrastive_rl import StableContrastiveRL
# from stable_contrastive_rl_train.evaluation.eval_utils import eval_rl_loop
from gnm_train.data.data_utils import (
    VISUALIZATION_IMAGE_SIZE,
    get_image_path
)

from gnm_train.visualizing.visualize_utils import (
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)
from gnm_train.visualizing.action_utils import (
    plot_trajs_and_points,
    plot_trajs_and_points_on_image,
)


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    goal_pos,
    gnm_pred_waypoints,
    rl_mc_pred_waypoints,
    label_waypoints,
    save_path=None,
    display=False,
    dataset_name="recon",
):
    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])
    if len(gnm_pred_waypoints.shape) > 2:
        trajs = [*gnm_pred_waypoints, *rl_mc_pred_waypoints, label_waypoints]
    else:
        trajs = [gnm_pred_waypoints, rl_mc_pred_waypoints, label_waypoints]
    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=[BLUE, YELLOW, MAGENTA],
        traj_labels=["gnm prediction", "stable contrastive rl mc prediction", "ground truth"],
        point_colors=[GREEN, RED],
        bearing_headlength=[5, 5, 5],
        bearing_headaxislength=[4.5, 4.5, 4.5],
        bearing_headwidth=[3, 3, 3],
    )
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[BLUE, YELLOW, MAGENTA],
        point_colors=[GREEN, RED],
    )
    ax[2].imshow(goal_img)
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)


def get_image(path, aspect_ratio):
    img = Image.open(path)
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * aspect_ratio))
    )  # crop to the right ratio
    viz_img = TF.resize(img, VISUALIZATION_IMAGE_SIZE)

    viz_img = Image.fromarray(np.array(viz_img))
    viz_img = viz_img.resize(VIZ_IMAGE_SIZE)

    return viz_img


def main(config):
    # read results
    gnm_dir = config["result_dirs"]["gnm"]
    rl_mc_dir = config["result_dirs"]["stable_contrastive_rl_mc"]
    # rl_td_dir = config["result_dirs"]["stable_contrastive_rl_td"]
    gnm_filename = os.path.join(gnm_dir, "results.pkl")
    rl_mc_filename = os.path.join(rl_mc_dir, "results.pkl")
    # rl_td_filename = os.path.join(rl_td_dir, "results.pkl")
    data_folder = config["data_folder"]
    aspect_ratio = config["image_size"][0] / config["image_size"][1]
    os.makedirs(config["save_dir"], exist_ok=True)

    with open(gnm_filename, "rb") as f:
        gnm_results = pkl.load(f)
    with open(rl_mc_filename, "rb") as f:
        rl_mc_results = pkl.load(f)
    # with open(rl_td_filename, "rb") as f:
    #     rl_td_results = pkl.load(f)
    # FIXME (chongyi): why the two datasets are different?
    # assert set(gnm_results.keys()) == set(rl_mc_results.keys())

    for label, gnm_result in gnm_results.items():
        if label not in rl_mc_results:
            continue

        save_path = os.path.join(config["save_dir"], label + ".png")

        f_curr = gnm_result["f_curr"]  # f_close is exactly the same as f_curr
        f_goal = gnm_result["f_goal"]
        curr_time = gnm_result["curr_time"]
        goal_time = gnm_result["goal_time"]
        gnm_goal_pos = gnm_result["goal_pos"]
        gnm_pred_waypoints = gnm_result["pred_waypoints"]
        gnm_label_waypoints = gnm_result["label_waypoints"]

        rl_mc_result = rl_mc_results[label]
        rl_mc_goal_pos = rl_mc_result["goal_pos"]
        rl_mc_pred_waypoints = rl_mc_result["pred_waypoints"]
        rl_mc_label_waypoints = rl_mc_result["label_waypoints"]

        assert np.all(gnm_goal_pos == rl_mc_goal_pos)
        assert np.all(rl_mc_label_waypoints == rl_mc_label_waypoints)

        if np.linalg.norm(rl_mc_pred_waypoints - gnm_label_waypoints) >= 0:
            obs_image_path = get_image_path(data_folder, f_curr, curr_time)
            obs_image = get_image(
                obs_image_path,
                aspect_ratio,
            )
            goal_image_path = get_image_path(data_folder, f_goal, goal_time)
            goal_image = get_image(
                goal_image_path,
                aspect_ratio,
            )

            compare_waypoints_pred_to_label(
                obs_image,
                goal_image,
                gnm_goal_pos,
                gnm_pred_waypoints,
                rl_mc_pred_waypoints,
                gnm_label_waypoints,
                save_path=save_path,
            )

    print("FINISH VISUALIZATION")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Robot Agnostic Learning")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/gnm/gnm_public.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    print(config)
    main(config)
