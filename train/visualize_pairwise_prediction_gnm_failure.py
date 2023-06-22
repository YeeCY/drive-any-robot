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

from gnm_train.data.data_utils import (
    VISUALIZATION_IMAGE_SIZE,
    get_image_path
)
from gnm_train.visualizing.visualize_utils import (
    VIZ_IMAGE_SIZE
)


def display_pairwise_distance_pred(
    imgs, titles,
    dist_label,
    gnm_dist_pred,
    rl_mc_logit_diff_pred, rl_td_logit_diff_pred,
    rl_mc_mi_diff_pred, rl_td_mi_diff_pred,
    text_color="black", save_path=None, display=False
):
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))

    plt.suptitle(f"label: {dist_label}\ngnm regression: {gnm_dist_pred}\n"
                 + f"contrastive rl mc close-logit-diff: {rl_mc_logit_diff_pred}\n"
                 + f"contrastive rl td close-logit-diff: {rl_td_logit_diff_pred}\n"
                 + f"contrastive rl mc mi-diff: {rl_mc_mi_diff_pred}\n"
                 + f"contrastive rl td mi-diff: {rl_td_mi_diff_pred}\n",
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

    viz_img = Image.fromarray(np.array(viz_img))
    viz_img = viz_img.resize(VIZ_IMAGE_SIZE)

    return viz_img


def main(config):
    # read results
    gnm_dir = config["result_dirs"]["gnm"]
    rl_mc_close_logit_diff_dir = config["result_dirs"]["crl_mc_logit_diff"]
    rl_td_close_logit_diff_dir = config["result_dirs"]["crl_td_logit_diff"]
    rl_mc_mi_diff_dir = config["result_dirs"]["crl_mc_mi_diff"]
    rl_td_mi_diff_dir = config["result_dirs"]["crl_td_mi_diff"]
    gnm_filename = os.path.join(gnm_dir, "results.pkl")
    rl_mc_close_logit_diff_filename = os.path.join(rl_mc_close_logit_diff_dir, "results.pkl")
    rl_td_close_logit_diff_filename = os.path.join(rl_mc_close_logit_diff_dir, "results.pkl")
    rl_mc_mi_diff_filename = os.path.join(rl_mc_mi_diff_dir, "results.pkl")
    rl_td_mi_diff_filename = os.path.join(rl_td_mi_diff_dir, "results.pkl")
    data_folder = config["data_folder"]
    aspect_ratio = config["image_size"][0] / config["image_size"][1]
    os.makedirs(config["save_dir"], exist_ok=True)

    with open(gnm_filename, "rb") as f:
        gnm_results = pkl.load(f)
    with open(rl_mc_logit_diff_filename, "rb") as f:
        rl_mc_logit_diff_results = pkl.load(f)
    with open(rl_td_logit_diff_filename, "rb") as f:
        rl_td_logit_diff_results = pkl.load(f)
    with open(rl_mc_mi_diff_filename, "rb") as f:
        rl_mc_mi_diff_results = pkl.load(f)
    with open(rl_td_mi_diff_filename, "rb") as f:
        rl_td_mi_diff_results = pkl.load(f)
    assert set(gnm_results.keys()) \
           == set(rl_mc_logit_diff_results.keys()) \
           == set(rl_td_logit_diff_results.keys()) \
           == set(rl_mc_mi_diff_results.keys()) \
           == set(rl_td_mi_diff_results.keys())

    for label, gnm_result in gnm_results.items():
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

        # MC close_logit_diff
        rl_mc_logit_diff_result = rl_mc_logit_diff_results[label]
        rl_mc_logit_diff_obs_a_close_g_logit = rl_mc_logit_diff_result["close_dist_pred"]
        rl_mc_logit_diff_close_dist_label = rl_mc_logit_diff_result["close_dist_label"]
        rl_mc_logit_diff_far_g_a_close_g_logit = rl_mc_logit_diff_result["far_dist_pred"]
        rl_mc_logit_diff_far_dist_label = rl_mc_logit_diff_result["far_dist_label"]

        assert close_dist_label == rl_mc_logit_diff_close_dist_label
        assert far_dist_label == rl_mc_logit_diff_far_dist_label

        # TD close_logit_diff
        rl_td_logit_diff_result = rl_td_logit_diff_results[label]
        rl_td_logit_diff_obs_a_close_g_logit = rl_td_logit_diff_result["close_dist_pred"]
        rl_td_logit_diff_close_dist_label = rl_td_logit_diff_result["close_dist_label"]
        rl_td_logit_diff_far_g_a_close_g_logit = rl_td_logit_diff_result["far_dist_pred"]
        rl_td_logit_diff_far_dist_label = rl_td_logit_diff_result["far_dist_label"]

        assert close_dist_label == rl_td_logit_diff_close_dist_label
        assert far_dist_label == rl_td_logit_diff_far_dist_label

        # MC close_far_mi_diff
        rl_mc_mi_diff_result = rl_mc_mi_diff_results[label]
        rl_mc_mi_diff_close_g_logit_diff = rl_mc_mi_diff_result["close_dist_pred"]
        rl_mc_mi_diff_close_dist_label = rl_mc_mi_diff_result["close_dist_label"]
        rl_mc_mi_diff_far_g_logit_diff = rl_mc_mi_diff_result["far_dist_pred"]
        rl_mc_mi_diff_far_dist_label = rl_mc_mi_diff_result["far_dist_label"]

        assert close_dist_label == rl_mc_mi_diff_close_dist_label
        assert far_dist_label == rl_mc_mi_diff_far_dist_label

        # TD close_far_mi_diff
        rl_td_mi_diff_result = rl_td_mi_diff_results[label]
        rl_td_mi_diff_close_g_logit_diff = rl_td_mi_diff_result["close_dist_pred"]
        rl_td_mi_diff_close_dist_label = rl_td_mi_diff_result["close_dist_label"]
        rl_td_mi_diff_far_g_logit_diff = rl_td_mi_diff_result["far_dist_pred"]
        rl_td_mi_diff_far_dist_label = rl_td_mi_diff_result["far_dist_label"]

        assert close_dist_label == rl_td_mi_diff_close_dist_label
        assert far_dist_label == rl_td_mi_diff_far_dist_label

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

        gnm_marker = u"\u2713" if gnm_far_dist_pred > gnm_close_dist_pred else u"\u2717"
        rl_mc_logit_diff_marker = u"\u2713" if rl_mc_logit_diff_obs_a_close_g_logit > rl_mc_logit_diff_far_g_a_close_g_logit else u"\u2717"
        rl_td_logit_diff_marker = u"\u2713" if rl_td_logit_diff_obs_a_close_g_logit > rl_td_logit_diff_far_g_a_close_g_logit else u"\u2717"
        rl_mc_mi_diff_marker = u"\u2713" if rl_mc_mi_diff_close_g_logit_diff > rl_mc_mi_diff_far_g_logit_diff else u"\u2717"
        rl_td_mi_diff_marker = u"\u2713" if rl_td_mi_diff_close_g_logit_diff > rl_td_mi_diff_far_g_logit_diff else u"\u2717"

        display_pairwise_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            f"close_pred = {gnm_close_dist_pred}, far_pred = {gnm_far_dist_pred} [{gnm_marker}]",
            fr"$f(s, a, g_c) = {rl_mc_logit_diff_obs_a_close_g_logit}$, $f(g_f, a, g_c) = {rl_mc_logit_diff_far_g_a_close_g_logit}$ [{rl_mc_logit_diff_marker}]",
            fr"$f(s, a, g_c) = {rl_td_logit_diff_obs_a_close_g_logit}$, $f(g_f, a, g_c) = {rl_td_logit_diff_far_g_a_close_g_logit}$ [{rl_td_logit_diff_marker}]",
            # rf"f(s, a, g_c) - f(g_f, a, g_c) = {}, far_pred = {rl_td_reg_far_dist_pred}",
            fr"$f(s, a, g_c) - f(g_f, a, g_c) = {rl_mc_mi_diff_close_g_logit_diff}$, $f(s, a, g_f) - f(g_c, a, g_f) = {rl_mc_mi_diff_far_g_logit_diff}$ [{rl_mc_mi_diff_marker}]",
            fr"$f(s, a, g_c) - f(g_f, a, g_c) = {rl_td_mi_diff_close_g_logit_diff}$, $f(s, a, g_f) - f(g_c, a, g_f) = {rl_td_mi_diff_far_g_logit_diff}$ [{rl_td_mi_diff_marker}]",
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
