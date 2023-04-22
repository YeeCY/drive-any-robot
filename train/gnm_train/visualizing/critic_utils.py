import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional, List
import wandb
import yaml
from gnm_train.visualizing.visualize_utils import (
    numpy_to_img,
    ceil,
    floor,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)
from gnm_train.visualizing.action_utils import (
    gen_bearings_from_waypoints,
    # plot_trajs_and_points,
    plot_trajs_and_points_on_image,
)

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def visualize_critic_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_oracle_waypoints: np.ndarray,
    batch_oracle_critics: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    """
    Compare critic predictions for trajectories spanning -x degrees to x degrees in front of the robot using egocentric visualization. This visualization is for the last batch in the dataset.

    Args:
        TODO
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
        dataset_names: indices corresponding to the dataset name
        batch_goals (np.ndarray): batch of goal positions [batch_size, 2]
        batch_pred_waypoints (np.ndarray): batch of predicted waypoints [batch_size, horizon, 4] or [batch_size, horizon, 2] or [batch_size, num_trajs_sampled horizon, {2 or 4}]
        batch_label_waypoints (np.ndarray): batch of label waypoints [batch_size, T, 4] or [batch_size, horizon, 2]
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        normalized (bool): whether the waypoints are normalized
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
    """
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
        == len(batch_label_waypoints)
    )

    assert len(batch_oracle_waypoints) == num_images_preds

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        obs_img = numpy_to_img(batch_obs_images[i])
        goal_img = numpy_to_img(batch_goal_images[i])
        dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        oracle_waypoints = batch_oracle_waypoints[i]
        oracle_critics = batch_oracle_critics[i]
        label_waypoints = batch_label_waypoints[i]

        if normalized:
            # pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
            oracle_waypoints[..., :2] *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        plot_oracle_critic_pred(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            oracle_waypoints,
            oracle_critics,
            label_waypoints,
            save_path,
            display
        )

        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_critic_prediction": wandb_list})


def plot_oracle_critic_pred(
    obs_img,
    goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    oracle_waypoints: np.ndarray,
    oracle_critics: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    Visualize critic prediction for oracle waypoints using egocentric visualization.

    Args:
        obs_img: image of the observation
        goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        goal_pos: goal position in the image
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """

    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])
    if len(oracle_waypoints.shape) > 2:
        trajs = [*oracle_waypoints, label_waypoints]
    else:
        trajs = [oracle_waypoints, label_waypoints]

    # create line with color bar
    # reference: https://stackoverflow.com/a/49184882
    vmin, vmax = floor(oracle_critics.min(), 1), ceil(oracle_critics.max(), 1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=[np.array(cmap.to_rgba(oracle_critic))
                     for oracle_critic in oracle_critics[:, 0]] + [MAGENTA],
        point_colors=[GREEN, RED],
    )
    plt.colorbar(cmap, ticks=np.linspace(vmin, vmax, 11), ax=ax[0],
                 fraction=0.046, pad=0.04)
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[np.array(cmap.to_rgba(oracle_critic))
                     for oracle_critic in oracle_critics[:, 0]] + [MAGENTA],
        point_colors=[GREEN, RED],
    )
    plt.colorbar(cmap, ticks=np.linspace(vmin, vmax, 11), ax=ax[1],
                 fraction=0.046, pad=0.04)

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


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    # traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        point_labels: list of labels for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    # assert (
    #     len(list_trajs) == len(traj_labels) or default_coloring
    # ), "Not enough labels for trajectories"
    assert len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=traj_colors[i],
            # label=traj_labels[i],
        )
        if traj.shape[1] > 2:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        ax.plot(
            pt[0],
            pt[1],
            color=point_colors[i],
            marker="o",
            markersize=7.0,
            label=point_labels[i],
        )
    ax.legend()
    # put the legend below the plot
    ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")
