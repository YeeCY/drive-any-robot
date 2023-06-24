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
from stable_contrastive_rl_train.evaluation.visualization_utils import (
    plot_trajs
)


def display_traj_dist_pred(
    global_curr_pos, global_goal_pos,
    gnm_path_idxs,
    rl_mc_logit_sum_path_idxs, rl_td_logit_sum_path_idxs,
    rl_mc_close_logit_diff_path_idxs, rl_td_close_logit_diff_path_idxs,
    rl_mc_far_logit_diff_path_idxs, rl_td_far_logit_diff_path_idxs,
    rl_mc_mi_diff_path_idxs, rl_td_mi_diff_path_idxs,
    text_color="black", save_path=None, display=False
):
    plt.figure()
    fig, ax = plt.subplots(1, 1)

    plt.suptitle(f"gnm w/o context w/ long horizon path: {gnm_path_idxs}\n"
                 + f"scrl mc w/o context w/ long horizon logit-sum path: {rl_mc_close_logit_diff_path_idxs}\n"
                 + f"scrl td w/o context w/ long horizon logit-sum path: {rl_td_close_logit_diff_path_idxs}\n"
                 + f"scrl mc w/o context w/ long horizon close-logit-diff path: {rl_mc_close_logit_diff_path_idxs}\n"
                 + f"scrl td w/o context w/ long horizon close-logit-diff path: {rl_td_close_logit_diff_path_idxs}\n"
                 + f"scrl mc w/o context w/ long horizon far-logit-diff path: {rl_mc_far_logit_diff_path_idxs}\n"
                 + f"scrl td w/o context w/ long horizon far-logit-diff path: {rl_td_far_logit_diff_path_idxs}\n"
                 + f"scrl mc w/o context w/ long horizon mi-diff path: {rl_mc_mi_diff_path_idxs}\n"
                 + f"scrl td w/o context w/ long horizon mi-diff path: {rl_td_mi_diff_path_idxs}",
                 y=1.2,
                 color=text_color)

    traj_len = len(global_curr_pos)
    assert len(np.unique(global_goal_pos)) == 3, "Multiple goal positions found!"
    global_goal_pos = global_goal_pos[0]

    plot_trajs(
        ax,
        [*global_curr_pos, global_goal_pos],
        point_colors=[BLUE] + [GREEN] * (traj_len - 1) + [RED],
        point_labels=["start"] + ["obs"] * (traj_len - 1) + ["goal"],
    )

    fig.set_size_inches(6.5, 6.5)
    ax.set_title(f"Trajectory Visualization")
    ax.set_aspect("equal", "box")

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
    rl_mc_logit_sum_dir = config["result_dirs"]["scrl_mc_logit_sum"]
    rl_td_logit_sum_dir = config["result_dirs"]["scrl_td_logit_sum"]
    rl_mc_close_logit_diff_dir = config["result_dirs"]["scrl_mc_close_logit_diff"]
    rl_td_close_logit_diff_dir = config["result_dirs"]["scrl_td_close_logit_diff"]
    rl_mc_far_logit_diff_dir = config["result_dirs"]["scrl_mc_far_logit_diff"]
    rl_td_far_logit_diff_dir = config["result_dirs"]["scrl_td_far_logit_diff"]
    rl_mc_mi_diff_dir = config["result_dirs"]["scrl_mc_mi_diff"]
    rl_td_mi_diff_dir = config["result_dirs"]["scrl_td_mi_diff"]
    gnm_filename = os.path.join(gnm_dir, "results.pkl")
    rl_mc_logit_sum_filename = os.path.join(rl_mc_logit_sum_dir, "results.pkl")
    rl_td_logit_sum_filename = os.path.join(rl_td_logit_sum_dir, "results.pkl")
    rl_mc_close_logit_diff_filename = os.path.join(rl_mc_close_logit_diff_dir, "results.pkl")
    rl_td_close_logit_diff_filename = os.path.join(rl_td_close_logit_diff_dir, "results.pkl")
    rl_mc_far_logit_diff_filename = os.path.join(rl_mc_far_logit_diff_dir, "results.pkl")
    rl_td_far_logit_diff_filename = os.path.join(rl_td_far_logit_diff_dir, "results.pkl")
    rl_mc_mi_diff_filename = os.path.join(rl_mc_mi_diff_dir, "results.pkl")
    rl_td_mi_diff_filename = os.path.join(rl_td_mi_diff_dir, "results.pkl")
    data_folder = config["data_folder"]
    aspect_ratio = config["image_size"][0] / config["image_size"][1]
    os.makedirs(config["save_dir"], exist_ok=True)

    with open(gnm_filename, "rb") as f:
        gnm_results = pkl.load(f)
    with open(rl_mc_logit_sum_filename, "rb") as f:
        rl_mc_logit_sum_results = pkl.load(f)
    with open(rl_td_logit_sum_filename, "rb") as f:
        rl_td_logit_sum_results = pkl.load(f)
    with open(rl_mc_close_logit_diff_filename, "rb") as f:
        rl_mc_close_logit_diff_results = pkl.load(f)
    with open(rl_td_close_logit_diff_filename, "rb") as f:
        rl_td_close_logit_diff_results = pkl.load(f)
    with open(rl_mc_far_logit_diff_filename, "rb") as f:
        rl_mc_far_logit_diff_results = pkl.load(f)
    with open(rl_td_far_logit_diff_filename, "rb") as f:
        rl_td_far_logit_diff_results = pkl.load(f)
    with open(rl_mc_mi_diff_filename, "rb") as f:
        rl_mc_mi_diff_results = pkl.load(f)
    with open(rl_td_mi_diff_filename, "rb") as f:
        rl_td_mi_diff_results = pkl.load(f)
    assert (
        set(gnm_results.keys())
        == set(rl_mc_logit_sum_results.keys())
        == set(rl_td_logit_sum_results.keys())
        == set(rl_mc_close_logit_diff_results.keys())
        == set(rl_td_close_logit_diff_results.keys())
        == set(rl_mc_far_logit_diff_results.keys())
        == set(rl_td_far_logit_diff_results.keys())
        == set(rl_mc_mi_diff_results.keys())
        == set(rl_td_mi_diff_results.keys())
    )

    for label, gnm_result in gnm_results.items():
        assert label in rl_mc_logit_sum_results
        assert label in rl_td_logit_sum_results
        assert label in rl_mc_close_logit_diff_results
        assert label in rl_td_close_logit_diff_results
        assert label in rl_mc_far_logit_diff_results
        assert label in rl_td_far_logit_diff_results
        assert label in rl_mc_mi_diff_results
        assert label in rl_td_mi_diff_results

        save_path = os.path.join(config["save_dir"], label + ".png")

        f_traj = gnm_result["f_traj"]
        context_size = gnm_result["context_size"]
        end_slack = gnm_result["end_slack"]
        subsampling_spacing = gnm_result["subsampling_spacing"]
        goal_time = gnm_result["goal_time"]
        global_curr_pos = gnm_result["global_curr_pos"]
        global_goal_pos = gnm_result["global_goal_pos"]
        gnm_path_idxs = gnm_result["path_idxs"]

        # logit-sum
        rl_mc_logit_sum_result = rl_mc_logit_sum_results[label]
        assert f_traj == rl_mc_logit_sum_result["f_traj"]
        assert context_size == rl_mc_logit_sum_result["context_size"]
        assert end_slack == rl_mc_logit_sum_result["end_slack"]
        assert subsampling_spacing == rl_mc_logit_sum_result["subsampling_spacing"]
        assert goal_time == rl_mc_logit_sum_result["goal_time"]
        assert np.all(global_curr_pos == rl_mc_logit_sum_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_mc_logit_sum_result["global_goal_pos"])
        rl_mc_logit_sum_path_idxs = rl_mc_logit_sum_result["path_idxs"]

        rl_td_logit_sum_result = rl_td_logit_sum_results[label]
        assert f_traj == rl_td_logit_sum_result["f_traj"]
        assert context_size == rl_td_logit_sum_result["context_size"]
        assert end_slack == rl_td_logit_sum_result["end_slack"]
        assert subsampling_spacing == rl_td_logit_sum_result["subsampling_spacing"]
        assert goal_time == rl_td_logit_sum_result["goal_time"]
        assert np.all(global_curr_pos == rl_td_logit_sum_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_td_logit_sum_result["global_goal_pos"])
        rl_td_logit_sum_path_idxs = rl_td_logit_sum_result["path_idxs"]

        # close-logit-diff
        rl_mc_close_logit_diff_result = rl_mc_close_logit_diff_results[label]
        assert f_traj == rl_mc_close_logit_diff_result["f_traj"]
        assert context_size == rl_mc_close_logit_diff_result["context_size"]
        assert end_slack == rl_mc_close_logit_diff_result["end_slack"]
        assert subsampling_spacing == rl_mc_close_logit_diff_result["subsampling_spacing"]
        assert goal_time == rl_mc_close_logit_diff_result["goal_time"]
        assert np.all(global_curr_pos == rl_mc_close_logit_diff_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_mc_close_logit_diff_result["global_goal_pos"])
        rl_mc_close_logit_diff_path_idxs = rl_mc_close_logit_diff_result["path_idxs"]

        rl_td_close_logit_diff_result = rl_td_close_logit_diff_results[label]
        assert f_traj == rl_td_close_logit_diff_result["f_traj"]
        assert context_size == rl_td_close_logit_diff_result["context_size"]
        assert end_slack == rl_td_close_logit_diff_result["end_slack"]
        assert subsampling_spacing == rl_td_close_logit_diff_result["subsampling_spacing"]
        assert goal_time == rl_td_close_logit_diff_result["goal_time"]
        assert np.all(global_curr_pos == rl_td_close_logit_diff_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_td_close_logit_diff_result["global_goal_pos"])
        rl_td_close_logit_diff_path_idxs = rl_td_close_logit_diff_result["path_idxs"]

        # far-logit-diff
        rl_mc_far_logit_diff_result = rl_mc_far_logit_diff_results[label]
        assert f_traj == rl_mc_far_logit_diff_result["f_traj"]
        assert context_size == rl_mc_far_logit_diff_result["context_size"]
        assert end_slack == rl_mc_far_logit_diff_result["end_slack"]
        assert subsampling_spacing == rl_mc_far_logit_diff_result["subsampling_spacing"]
        assert goal_time == rl_mc_far_logit_diff_result["goal_time"]
        assert np.all(global_curr_pos == rl_mc_far_logit_diff_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_mc_far_logit_diff_result["global_goal_pos"])
        rl_mc_far_logit_diff_path_idxs = rl_mc_far_logit_diff_result["path_idxs"]

        rl_td_far_logit_diff_result = rl_td_far_logit_diff_results[label]
        assert f_traj == rl_td_far_logit_diff_result["f_traj"]
        assert context_size == rl_td_far_logit_diff_result["context_size"]
        assert end_slack == rl_td_far_logit_diff_result["end_slack"]
        assert subsampling_spacing == rl_td_far_logit_diff_result["subsampling_spacing"]
        assert goal_time == rl_td_far_logit_diff_result["goal_time"]
        assert np.all(global_curr_pos == rl_td_far_logit_diff_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_td_far_logit_diff_result["global_goal_pos"])
        rl_td_far_logit_diff_path_idxs = rl_td_far_logit_diff_result["path_idxs"]

        # mi-diff
        rl_mc_mi_diff_result = rl_mc_mi_diff_results[label]
        assert f_traj == rl_mc_mi_diff_result["f_traj"]
        assert context_size == rl_mc_mi_diff_result["context_size"]
        assert end_slack == rl_mc_mi_diff_result["end_slack"]
        assert subsampling_spacing == rl_mc_mi_diff_result["subsampling_spacing"]
        assert goal_time == rl_mc_mi_diff_result["goal_time"]
        assert np.all(global_curr_pos == rl_mc_mi_diff_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_mc_mi_diff_result["global_goal_pos"])
        rl_mc_mi_diff_path_idxs = rl_mc_mi_diff_result["path_idxs"]

        rl_td_mi_diff_result = rl_td_mi_diff_results[label]
        assert f_traj == rl_td_mi_diff_result["f_traj"]
        assert context_size == rl_td_mi_diff_result["context_size"]
        assert end_slack == rl_td_mi_diff_result["end_slack"]
        assert subsampling_spacing == rl_td_mi_diff_result["subsampling_spacing"]
        assert goal_time == rl_td_mi_diff_result["goal_time"]
        assert np.all(global_curr_pos == rl_td_mi_diff_result["global_curr_pos"])
        assert np.all(global_goal_pos == rl_td_mi_diff_result["global_goal_pos"])
        rl_td_mi_diff_path_idxs = rl_td_far_logit_diff_result["path_idxs"]

        display_traj_dist_pred(
            global_curr_pos,
            global_goal_pos,
            gnm_path_idxs.tolist(),
            rl_mc_logit_sum_path_idxs.tolist(),
            rl_td_logit_sum_path_idxs.tolist(),
            rl_mc_close_logit_diff_path_idxs.tolist(),
            rl_td_close_logit_diff_path_idxs.tolist(),
            rl_mc_far_logit_diff_path_idxs.tolist(),
            rl_td_far_logit_diff_path_idxs.tolist(),
            rl_mc_mi_diff_path_idxs.tolist(),
            rl_td_mi_diff_path_idxs.tolist(),
            "black",
            save_path,
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
