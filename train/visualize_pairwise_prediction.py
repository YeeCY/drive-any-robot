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


def display_pairwise_distance_pred(
    imgs, titles,
    dist_label, gnm_dist_pred, rl_mc_reg_dist_pred, rl_td_reg_dist_pred,
    text_color="black", save_path=None, display=False
):
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))

    plt.suptitle(f"label: {dist_label}\ngnm prediction: {gnm_dist_pred}\nstable contrastive rl mc regression: {rl_mc_reg_dist_pred}\n stable contrastive rl td regression: {rl_td_reg_dist_pred}",
                 color=text_color)

    for axis, img, title in zip(ax, imgs, titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # make the plot large
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

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
    viz_img = np.array(viz_img)

    return viz_img


def main(config):
    # read results
    gnm_dir = config["result_dirs"]["gnm"]
    rl_mc_reg_dir = config["result_dirs"]["stable_contrastive_rl_mc_regression"]
    rl_td_reg_dir = config["result_dirs"]["stable_contrastive_rl_td_regression"]
    rl_mc_critic_dir = config["result_dirs"]["stable_contrastive_rl_mc_critic"]
    rl_td_critic_dir = config["result_dirs"]["stable_contrastive_rl_td_critic"]
    gnm_filename = os.path.join(gnm_dir, "results.pkl")
    rl_mc_reg_filename = os.path.join(rl_mc_reg_dir, "results.pkl")
    rl_td_reg_filename = os.path.join(rl_td_reg_dir, "results.pkl")
    data_folder = config["data_folder"]
    aspect_ratio = config["image_size"][0] / config["image_size"][1]
    os.makedirs(config["save_dir"], exist_ok=True)

    with open(gnm_filename, "rb") as f:
        gnm_results = pkl.load(f)
    with open(rl_mc_reg_filename, "rb") as f:
        rl_mc_reg_results = pkl.load(f)
    with open(rl_td_reg_filename, "rb") as f:
        rl_td_reg_results = pkl.load(f)
    assert set(gnm_results.keys()) == set(rl_mc_reg_results.keys())

    for label, gnm_result in gnm_results.items():
        assert label in rl_mc_reg_results

        save_path = os.path.join(config["save_dir"], label + ".png")

        f_close = gnm_result["f_close"]  # f_close is exactly the same as f_curr
        f_far = gnm_result["f_far"]
        curr_time = gnm_result["curr_time"]
        close_time = gnm_result["close_time"]
        far_time = gnm_result["far_time"]
        context_times = gnm_result["context_times"]
        far_dist_label = gnm_result["far_dist_label"]
        close_dist_label = gnm_result["close_dist_label"]
        gnm_far_dist_pred = gnm_result["far_dist_pred"]
        gnm_close_dist_pred = gnm_result["close_dist_pred"]

        rl_mc_reg_result = rl_mc_reg_results[label]
        rl_mc_reg_far_dist_pred = rl_mc_reg_result["far_dist_pred"]
        rl_mc_reg_far_dist_label = rl_mc_reg_result["far_dist_label"]
        rl_mc_reg_close_dist_pred = rl_mc_reg_result["close_dist_pred"]
        rl_mc_reg_close_dist_label = rl_mc_reg_result["close_dist_label"]

        assert far_dist_label == rl_mc_reg_far_dist_label
        assert close_dist_label == rl_mc_reg_close_dist_label

        rl_td_reg_result = rl_td_reg_results[label]
        rl_td_reg_far_dist_pred = rl_td_reg_result["far_dist_pred"]
        rl_td_reg_far_dist_label = rl_td_reg_result["far_dist_label"]
        rl_td_reg_close_dist_pred = rl_td_reg_result["close_dist_pred"]
        rl_td_reg_close_dist_label = rl_td_reg_result["close_dist_label"]

        assert far_dist_label == rl_td_reg_far_dist_label
        assert close_dist_label == rl_td_reg_close_dist_label

        # with open(rl_mc_reg_filename, "rb") as f:
        #     rl_mc_reg_result = pkl.load(f)
        #
        # assert f_close == rl_mc_reg_result["f_close"]
        # assert f_far == rl_mc_reg_result["f_far"]
        # assert curr_time == rl_mc_reg_result["curr_time"]
        # assert close_time == rl_mc_reg_result["close_time"]
        # assert far_time == rl_mc_reg_result["far_time"]
        # assert context_times == rl_mc_reg_result["context_times"]
        # # gnm_far_dist_pred = gnm_result["far_dist_pred"]
        # # gnm_far_dist_label = gnm_result["far_dist_label"]
        # # gnm_close_dist_pred = gnm_result["close_dist_pred"]
        # # gnm_close_dist_label = gnm_result["close_dist_label"]

        # with open(os.path.join(config["data_folder"], f_close, "traj_data.pkl"), "rb") as close_f:
        #     curr_traj_data = pkl.load(f)
        # curr_traj_len = len(curr_traj_data["position"])
        # assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        obs_image_path = get_image_path(data_folder, f_close, curr_time)
        obs_image = get_image(
            obs_image_path,
            aspect_ratio,
        )
        close_image_path = get_image_path(data_folder, f_close, close_time)
        close_image = get_image(
            close_image_path,
            aspect_ratio,
        )
        far_image_path = get_image_path(data_folder, f_far, far_time)
        far_image = get_image(
            far_image_path,
            aspect_ratio,
        )

        display_pairwise_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            f"close_pred = {gnm_close_dist_pred}, far_pred = {gnm_far_dist_pred}",
            f"close_pred = {rl_mc_reg_close_dist_pred}, far_pred = {rl_mc_reg_far_dist_pred}",
            f"close_pred = {rl_td_reg_close_dist_pred}, far_pred = {rl_td_reg_far_dist_pred}",
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
