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
    ORANGE,
)
from stable_contrastive_rl_train.evaluation.visualization_utils import (
    plot_trajs
)

from gps.plotter import GPSPlotter


def display_traj_dist_pred(
    gps_plotter,
    obs_image, gnm_path_image, rl_mc_sorting_path_image, rl_td_sorting_path_image,
    obs_latlong, goal_latlong, cand_latlong,
    global_obs_pos, global_goal_pos, global_cand_pos,
    gnm_path_idxs, gnm_success,
    rl_mc_sorting_path_idxs, rl_mc_sorting_success,
    rl_td_sorting_path_idxs, rl_td_sorting_success,
    text_color="black", save_path=None, display=False
):
    plt.figure()
    fig, ax = plt.subplots(1, 1)

    # if not np.isfinite(obs_latlong).all():
    #     obs_latlong = np.array(gps_plotter.se_latlong)
    # if not np.isfinite(goal_latlong).all():
    #     goal_latlong = np.array(gps_plotter.se_latlong)

    gnm_marker = u"\u2713" if gnm_success else u"\u2717"
    rl_mc_sorting_marker = u"\u2713" if rl_mc_sorting_success else u"\u2717"
    rl_td_sorting_marker = u"\u2713" if rl_td_sorting_success else u"\u2717"

    plt.suptitle(f"gnm: {gnm_path_idxs} [{gnm_marker}]\n"
                 + f"scrl mc sorting: {rl_mc_sorting_path_idxs} [{rl_mc_sorting_marker}]\n"
                 + f"scrl td sorting: {rl_td_sorting_path_idxs} [{rl_td_sorting_marker}]\n",
                 y=1.4,
                 color=text_color)

    # traj_len = len(global_curr_pos)
    # assert len(np.unique(global_goal_pos)) == 3, "Multiple goal positions found!"
    # global_goal_pos = global_goal_pos[0]
    #
    # plot_trajs(
    #     ax,
    #     [*global_curr_pos, global_goal_pos],
    #     point_colors=[BLUE] + [GREEN] * (traj_len - 1) + [RED],
    #     point_labels=["start"] + ["obs"] * (traj_len - 1) + ["goal"],
    # )

    traj_len = len(obs_latlong)
    assert len(np.unique(goal_latlong)) == 2, "Multiple goal positions found!"
    goal_latlong = goal_latlong[0][None]

    num_candidates = len(cand_latlong)

    gps_plotter.plot_latlong(
        ax,
        np.concatenate([cand_latlong, obs_latlong, goal_latlong, cand_latlong[gnm_path_idxs], cand_latlong[rl_mc_sorting_path_idxs], cand_latlong[rl_td_sorting_path_idxs]]),
        colors=[YELLOW] * num_candidates + [BLUE] + [GREEN] * (traj_len - 1) + [RED] + [CYAN] * len(gnm_path_idxs) + [MAGENTA] * len(rl_mc_sorting_path_idxs) + [ORANGE] * len(rl_td_sorting_path_idxs),
        labels=["candidate"] * num_candidates + ["start"] + ["obs"] * (traj_len - 1) + ["goal"] + ["gnm path"] * len(gnm_path_idxs) + ["scrl mc sorting path"] * len(rl_mc_sorting_path_idxs) + ["scrl td sorting path"] * len(rl_td_sorting_path_idxs),
        adaptive_satellite_img=True,
        font_size=8,
        text_len=len(cand_latlong),
    )
    # gps_plotter.plot_latlong(
    #     ax,
    #     cand_latlong[gnm_path_idxs],
    #     colors=[CYAN] * len(gnm_path_idxs),
    #     labels=["gnm path"] * len(gnm_path_idxs),
    #     adaptive_satellite_img=False,
    #     font_size=8,
    #     add_text=False
    # )
    # gps_plotter.plot_latlong(
    #     ax,
    #     cand_latlong[rl_mc_sorting_path_idxs],
    #     colors=[MAGENTA] * len(rl_mc_sorting_path_idxs),
    #     labels=["scrl mc path"] * len(rl_mc_sorting_path_idxs),
    #     adaptive_satellite_img=False,
    #     font_size=8,
    #     add_text=False,
    # )
    # gps_plotter.plot_latlong(
    #     ax,
    #     np.concatenate([obs_latlong, goal_latlong]),
    #     colors=[BLUE] + [GREEN] * (traj_len - 1) + [RED],
    #     labels=["start"] + ["obs"] * (traj_len - 1) + ["goal"],
    #     adaptive_satellite_img=False,
    #     font_size=8,
    #     add_text=False,
    # )
    # latlong = np.concatenate([obs_latlong, goal_latlong], axis=0)
    # gps_plotter.plot_latlong_and_compass_bearing(ax, latlong, np.zeros(latlong.shape[0]))

    # remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # put the legend below the plot
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(0.0, -0.05), loc="upper left", ncol=2)

    fig.set_size_inches(6.5, 6.5)
    ax.set_title(f"Trajectory Visualization")
    ax.set_aspect("equal", "box")

    obs_fig, obs_axes = plt.subplots(1, len(obs_image))
    for i in range(len(obs_image)):
        obs_axes[i].imshow(obs_image[i])
        obs_axes[i].set_title("obs{}".format(i))
        obs_axes[i].xaxis.set_visible(False)
        obs_axes[i].yaxis.set_visible(False)
    obs_fig.set_size_inches(6.5 * len(obs_image), 6.5)

    gnm_path_fig, gnm_path_axes = plt.subplots(1, len(gnm_path_image))
    for i in range(len(gnm_path_image)):
        gnm_path_axes[i].imshow(gnm_path_image[i])
        gnm_path_axes[i].set_title("gnm_path{}".format(i))
        gnm_path_axes[i].xaxis.set_visible(False)
        gnm_path_axes[i].yaxis.set_visible(False)
    gnm_path_fig.set_size_inches(6.5 * len(gnm_path_image), 6.5)

    rl_mc_sorting_path_fig, rl_mc_sorting_path_axes = plt.subplots(1, len(rl_mc_sorting_path_image))
    for i in range(len(rl_mc_sorting_path_image)):
        rl_mc_sorting_path_axes[i].imshow(rl_mc_sorting_path_image[i])
        rl_mc_sorting_path_axes[i].set_title("rl_mc_sorting_path{}".format(i))
        rl_mc_sorting_path_axes[i].xaxis.set_visible(False)
        rl_mc_sorting_path_axes[i].yaxis.set_visible(False)
    rl_mc_sorting_path_fig.set_size_inches(6.5 * len(rl_mc_sorting_path_image), 6.5)

    rl_td_sorting_path_fig, rl_td_sorting_path_axes = plt.subplots(1, len(rl_td_sorting_path_image))
    for i in range(len(rl_td_sorting_path_image)):
        rl_td_sorting_path_axes[i].imshow(rl_td_sorting_path_image[i])
        rl_td_sorting_path_axes[i].set_title("rl_td_sorting_path{}".format(i))
        rl_td_sorting_path_axes[i].xaxis.set_visible(False)
        rl_td_sorting_path_axes[i].yaxis.set_visible(False)
    rl_td_sorting_path_fig.set_size_inches(6.5 * len(rl_td_sorting_path_image), 6.5)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(os.path.join(save_dir, "gps"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "obs"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "gnm"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "rl_mc_sorting"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "rl_td_sorting"), exist_ok=True)

        fig.savefig(
            os.path.join(save_dir, "gps", save_path.split("/")[-1]),
            bbox_inches="tight",
        )
        obs_fig.savefig(
            os.path.join(save_dir, "obs", save_path.split("/")[-1]),
            bbox_inches="tight",
        )
        gnm_path_fig.savefig(
            os.path.join(save_dir, "gnm", save_path.split("/")[-1]),
            bbox_inches="tight",
        )
        rl_mc_sorting_path_fig.savefig(
            os.path.join(save_dir, "rl_mc_sorting", save_path.split("/")[-1]),
            bbox_inches="tight",
        )
        rl_td_sorting_path_fig.savefig(
            os.path.join(save_dir, "rl_td_sorting", save_path.split("/")[-1]),
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
    rl_mc_sorting_dir = config["result_dirs"]["scrl_mc_sorting"]
    rl_td_sorting_dir = config["result_dirs"]["scrl_td_sorting"]
    gnm_filename = os.path.join(gnm_dir, "results.pkl")
    rl_mc_sorting_filename = os.path.join(rl_mc_sorting_dir, "results.pkl")
    rl_td_sorting_filename = os.path.join(rl_td_sorting_dir, "results.pkl")
    data_folder = config["data_folder"]
    aspect_ratio = config["image_size"][0] / config["image_size"][1]
    os.makedirs(config["save_dir"], exist_ok=True)

    # new nw and se latlongs for visualization
    gps_plotter = GPSPlotter(
        # nw_latlong=(37.915185, -122.334651),
        # se_latlong=(37.914884, -122.334064),
        zoom=22,
    )

    with open(gnm_filename, "rb") as f:
        gnm_results = pkl.load(f)
    with open(rl_mc_sorting_filename, "rb") as f:
        rl_mc_sorting_results = pkl.load(f)
    with open(rl_td_sorting_filename, "rb") as f:
        rl_td_sorting_results = pkl.load(f)
    assert (
        set(gnm_results.keys())
        == set(rl_mc_sorting_results.keys())
        == set(rl_td_sorting_results.keys())
    )

    for label, gnm_result in gnm_results.items():
        assert label in rl_mc_sorting_results
        assert label in rl_td_sorting_results

        save_path = os.path.join(config["save_dir"], label + ".png")

        f_traj = gnm_result["f_traj"][0]
        context_size = gnm_result["context_size"]
        end_slack = gnm_result["end_slack"]
        subsampling_spacing = gnm_result["subsampling_spacing"]
        goal_time = gnm_result["goal_time"]
        obs_latlong = gnm_result["obs_latlong"]
        goal_latlong = gnm_result["goal_latlong"]
        cand_latlong = gnm_result["cand_latlong"]
        global_obs_pos = gnm_result["global_obs_pos"]
        global_goal_pos = gnm_result["global_goal_pos"]
        global_cand_pos = gnm_result["global_cand_pos"]
        cand_infos = gnm_result["cand_infos"]
        gnm_path_idxs = gnm_result["path_idxs"]
        traj_len = len(global_obs_pos) - 1
        gnm_success = np.any(np.abs(gnm_path_idxs - traj_len) <= 3)

        # get observatiojn images
        with open(os.path.join(data_folder, f_traj, "traj_data.pkl"), "rb") as f:
            traj_data = pkl.load(f)
        traj_len = len(traj_data["position"])
        obs_images = []
        for curr_time in range(
            context_size,
            traj_len - end_slack,
            subsampling_spacing,
        ):
            obs_image_path = get_image_path(data_folder, f_traj, curr_time)
            obs_image = get_image(
                obs_image_path,
                aspect_ratio,
            )

            obs_images.append(obs_image)
        obs_image = np.stack(obs_images)

        gnm_path_images = []
        for idx in gnm_path_idxs:
            f_path, path_time = cand_infos[idx]
            path_image_path = get_image_path(data_folder, f_path, path_time)
            path_image = get_image(
                path_image_path,
                aspect_ratio,
            )
            gnm_path_images.append(path_image)
        gnm_path_image = np.stack(gnm_path_images)


        # sorting
        rl_mc_sorting_result = rl_mc_sorting_results[label]
        assert f_traj == rl_mc_sorting_result["f_traj"][0]
        assert context_size == rl_mc_sorting_result["context_size"]
        assert end_slack == rl_mc_sorting_result["end_slack"]
        assert subsampling_spacing == rl_mc_sorting_result["subsampling_spacing"]
        assert goal_time == rl_mc_sorting_result["goal_time"]
        assert np.all(obs_latlong == rl_mc_sorting_result["obs_latlong"])
        assert np.all(goal_latlong == rl_mc_sorting_result["goal_latlong"])
        assert np.all(cand_latlong == rl_mc_sorting_result["cand_latlong"])
        assert np.all(global_obs_pos == rl_mc_sorting_result["global_obs_pos"])
        assert np.all(global_goal_pos == rl_mc_sorting_result["global_goal_pos"])
        assert np.all(global_cand_pos == rl_mc_sorting_result["global_cand_pos"])
        rl_mc_sorting_path_idxs = rl_mc_sorting_result["path_idxs"]
        rl_mc_sorting_success = np.any(np.abs(rl_mc_sorting_path_idxs - traj_len) <= 2)

        rl_mc_path_images = []
        for idx in rl_mc_sorting_path_idxs:
            f_path, path_time = cand_infos[idx]
            path_image_path = get_image_path(data_folder, f_path, path_time)
            path_image = get_image(
                path_image_path,
                aspect_ratio,
            )
            rl_mc_path_images.append(path_image)
        rl_mc_path_image = np.stack(rl_mc_path_images)

        rl_td_sorting_result = rl_td_sorting_results[label]
        assert f_traj == rl_td_sorting_result["f_traj"][0]
        assert context_size == rl_td_sorting_result["context_size"]
        assert end_slack == rl_td_sorting_result["end_slack"]
        assert subsampling_spacing == rl_td_sorting_result["subsampling_spacing"]
        assert goal_time == rl_td_sorting_result["goal_time"]
        assert np.all(obs_latlong == rl_td_sorting_result["obs_latlong"])
        assert np.all(goal_latlong == rl_td_sorting_result["goal_latlong"])
        assert np.all(cand_latlong == rl_mc_sorting_result["cand_latlong"])
        assert np.all(global_obs_pos == rl_td_sorting_result["global_obs_pos"])
        assert np.all(global_goal_pos == rl_td_sorting_result["global_goal_pos"])
        assert np.all(global_cand_pos == rl_td_sorting_result["global_cand_pos"])
        rl_td_sorting_path_idxs = rl_td_sorting_result["path_idxs"]
        rl_td_sorting_success = np.any(np.abs(rl_td_sorting_path_idxs - traj_len) <= 2)

        rl_td_path_images = []
        for idx in rl_td_sorting_path_idxs:
            f_path, path_time = cand_infos[idx]
            path_image_path = get_image_path(data_folder, f_path, path_time)
            path_image = get_image(
                path_image_path,
                aspect_ratio,
            )
            rl_td_path_images.append(path_image)
        rl_td_path_image = np.stack(rl_td_path_images)

        display_traj_dist_pred(
            gps_plotter,
            obs_image,
            gnm_path_image,
            rl_mc_path_image,
            rl_td_path_image,
            obs_latlong,
            goal_latlong,
            cand_latlong,
            global_obs_pos,
            global_goal_pos,
            global_cand_pos,
            gnm_path_idxs.tolist(), gnm_success,
            rl_mc_sorting_path_idxs.tolist(), rl_mc_sorting_success,
            rl_td_sorting_path_idxs.tolist(), rl_td_sorting_success,
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
