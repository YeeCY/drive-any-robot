import os
import yaml
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import pickle as pkl

from stable_contrastive_rl_train.visualizing.critic_utils import plot_oracle_critic_pred
from gnm_train.visualizing.visualize_utils import (
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

EPS = 1e-6

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def visualize_critic_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_oracle_waypoints: np.ndarray,
    batch_oracle_waypoints_critic: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_pred_waypoints_critic: np.ndarray,
    batch_label_waypoints: np.ndarray,
    batch_label_waypoints_critic: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    index_to_data: dict,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "critic_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_goals)
        == len(batch_oracle_waypoints)
        == len(batch_oracle_waypoints_critic)
        == len(batch_pred_waypoints)
        == len(batch_pred_waypoints_critic)
        == len(batch_label_waypoints)
        == len(batch_label_waypoints_critic)
    )

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    if save_folder is not None:
        result_save_path = os.path.join(visualize_path, f"results.pkl")
    for i in range(min(batch_size, num_images_preds)):
        obs_img = numpy_to_img(batch_obs_images[i])
        goal_img = numpy_to_img(batch_goal_images[i])
        dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        oracle_waypoints = batch_oracle_waypoints[i]
        oracle_waypoints_critic = batch_oracle_waypoints_critic[i]
        pred_waypoints = batch_pred_waypoints[i]
        pred_waypoints_critic = batch_pred_waypoints_critic[i]
        label_waypoints = batch_label_waypoints[i]
        label_waypoints_critic = batch_label_waypoints_critic[i]

        if normalized:
            pred_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
            oracle_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos[:2] *= data_config[dataset_name]["metric_waypoint_spacing"]

        result_label = "f_curr={}_f_goal={}_curr_time={}_goal_time={}".format(
            index_to_data["f_curr"][i],
            index_to_data["f_goal"][i],
            int(index_to_data["curr_time"][i]),
            int(index_to_data["goal_time"][i]),
        )
        result = {
            "f_curr": index_to_data["f_curr"][i],
            "f_goal": index_to_data["f_goal"][i],
            "curr_time": int(index_to_data["curr_time"][i]),
            "goal_time": int(index_to_data["goal_time"][i]),
            "goal_pos": goal_pos,
            "oracle_waypoints": oracle_waypoints,
            "pred_waypoints": pred_waypoints,
            "label_waypoints": label_waypoints,
            "oracle_critic": oracle_waypoints_critic,
            "pred_critic": pred_waypoints_critic,
            "label_critic": label_waypoints_critic,
        }

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        plot_oracle_critic_pred(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            oracle_waypoints,
            oracle_waypoints_critic,
            pred_waypoints,
            pred_waypoints_critic,
            label_waypoints,
            label_waypoints_critic,
            save_path,
            display
        )

        if os.path.exists(result_save_path):
            with open(result_save_path, "rb") as f:
                results = pkl.load(f)
            results[result_label] = result
            with open(result_save_path, "wb") as f:
                pkl.dump(results, f)
        else:
            results = {result_label: result}
            with open(result_save_path, "wb+") as f:
                pkl.dump(results, f)

        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_critic_prediction": wandb_list})


def plot_traj_dist_pred(
    dataset_name: str,
    local_goal_pos: np.ndarray,
    label_waypoints: np.ndarray,
    global_curr_pos: np.ndarray,
    global_goal_pos: np.ndarray,
    path_idxs: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    traj_len = len(label_waypoints)

    fig, ax = plt.subplots(1, 1)
    if len(label_waypoints.shape) > 2:
        trajs = [*label_waypoints]
    else:
        trajs = [label_waypoints]

    plot_trajs(
        ax,
        # np.array(oracle_trajs)[..., :2],  # don't plot yaws
        [*global_curr_pos, global_goal_pos],
        # traj_colors=sm.to_rgba(normalized_oracle_waypoints_critic)[:, :3],
        # traj_labels=["oracle"] * len(oracle_waypoints_critic),
        point_colors=[BLUE] + [GREEN] * (traj_len - 1) + [RED],
        point_labels=["start"] + ["obs"] * (traj_len - 1) + ["goal"],
    )

    plt.suptitle(f"path indices: {path_idxs.tolist()}",
                 y=0.95,
                 color="black")

    # ax[2].imshow(goal_img)
    # ax[2].xaxis.set_visible(False)
    # ax[2].yaxis.set_visible(False)

    # fig.set_size_inches(18.5, 10.5)
    fig.set_size_inches(6.5, 6.5)
    ax.set_title(f"Trajectory Visualization")
    # ax[1].set_title(f"Observation")
    # ax[2].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)


def plot_trajs(
    ax: plt.Axes,
    # list_trajs: list,
    list_points: list,
    # traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    # traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    # default_coloring: bool = True,
    point_size: float = 7.0,
):
    # assert (
    #     len(list_trajs) <= len(traj_colors) or default_coloring
    # ), "Not enough colors for trajectories"
    # assert len(list_points) <= len(point_colors), "Not enough colors for points"
    # assert (
    #     len(list_trajs) == len(traj_labels) or default_coloring
    # ), "Not enough labels for trajectories"
    # assert len(list_points) == len(point_labels), "Not enough labels for points"

    # for i, traj in enumerate(list_trajs):
    #     ax.plot(
    #         traj[:, 0],
    #         traj[:, 1],
    #         color=traj_colors[i],
    #         label=traj_labels[i],
    #         linewidth=traj_width,
    #     )
    #     if traj.shape[1] > 2:  # traj data also includes yaw of the robot
    #         bearings = gen_bearings_from_waypoints(traj)
    #         ax.quiver(
    #             traj[::quiver_freq, 0],
    #             traj[::quiver_freq, 1],
    #             bearings[::quiver_freq, 0],
    #             bearings[::quiver_freq, 1],
    #             color=traj_colors[i] * 0.5,
    #             scale=1.0,
    #             headlength=bearing_headlength[i],
    #             headaxislength=bearing_headaxislength[i],
    #             headwidth=bearing_headwidth[i],
    #         )

    for i, pt in enumerate(list_points):
        ax.plot(
            pt[0],
            pt[1],
            color=point_colors[i],
            marker="o",
            markersize=point_size,
            label=point_labels[i],
        )
        ax.text(
            pt[0] + 0.10,
            pt[1] - 0.10,
            str(i)
        )

    # remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # put the legend below the plot
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(0.0, -0.05), loc="upper left", ncol=2)
    # ax.set_aspect("equal", "box")


def visualize_traj_dist_pred(
    traj_obs_images: np.ndarray,
    traj_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    traj_local_goal_positions: np.ndarray,
    traj_label_waypoints: np.ndarray,
    traj_global_current_positions: np.ndarray,
    traj_global_goal_positions: np.ndarray,
    path_idxs: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    display: bool = False,
):
    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "trajectory_distance_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    assert (
        len(traj_obs_images)
        == len(traj_goal_images)
        == len(traj_local_goal_positions)
        == len(traj_label_waypoints)
        == len(traj_global_current_positions)
        == len(traj_global_goal_positions)
    )

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    assert len(np.unique(dataset_indices)) == 1, "Multiple dataset indices found!"
    dataset_name = dataset_names[int(dataset_indices[0])]

    # traj_len = traj_obs_images.shape[0]
    # for i in range(traj_len):
    #     obs_img = numpy_to_img(traj_obs_images[i])
    #     goal_img = numpy_to_img(traj_goal_images[i])
    #     dataset_name = dataset_names[int(dataset_indices[i])]
    #     local_goal_pos = traj_local_goal_positions[i]
    #     label_waypoints = traj_label_waypoints[i]
    #     global_curr_pos = traj_global_current_positions[i]
    #     global_goal_pos = traj_global_goal_positions[i]
    #
    #     if normalized:
    #         local_goal_pos[:2] *= data_config[dataset_name]["metric_waypoint_spacing"]
    #         label_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
    #         global_curr_pos[:2] *= data_config[dataset_name]["metric_waypoint_spacing"]
    #         global_goal_pos[:2] *= data_config[dataset_name]["metric_waypoint_spacing"]
    #
    #     save_path = None
    #     if visualize_path is not None:
    #         save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")
    #
    #     plot_traj_dist_pred(
    #         obs_img,
    #         goal_img,
    #         dataset_name,
    #         local_goal_pos,
    #         label_waypoints,
    #         global_curr_pos,
    #         global_goal_pos,
    #         save_path,
    #         display
    #     )

    # local_goal_pos = traj_local_goal_positions[i]
    # label_waypoints = traj_label_waypoints[i]
    # global_curr_pos = traj_global_current_positions[i]
    # global_goal_pos = traj_global_goal_positions[i]

    if normalized:
        traj_local_goal_positions[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
        traj_label_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
        traj_global_current_positions[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
        traj_global_goal_positions[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]

    save_path = None
    if visualize_path is not None:
        save_path = os.path.join(visualize_path, f"trajectory.png")

    assert len(np.unique(traj_global_goal_positions)) == 3, "Multiple goal positions found!"
    global_goal_positions = traj_global_goal_positions[0]

    plot_traj_dist_pred(
        dataset_name,
        traj_local_goal_positions,
        traj_label_waypoints,
        traj_global_current_positions,
        global_goal_positions,
        path_idxs,
        save_path,
        display
    )
