import os
import wandb
import pickle as pkl
import numpy as np
import yaml
import torch
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

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
from gnm_train.visualizing.action_utils import (
    plot_trajs_and_points,
    plot_trajs_and_points_on_image
)

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def visualize_traj_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
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
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_goals)
        == len(batch_pred_waypoints)
        == len(batch_label_waypoints)
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
        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]

        if normalized:
            pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

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
            "pred_waypoints": pred_waypoints,
            "label_waypoints": label_waypoints,
        }

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        compare_waypoints_pred_to_label(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            pred_waypoints,
            label_waypoints,
            save_path,
            display,
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
        wandb.log({f"{eval_type}_action_prediction": wandb_list})


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.

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
    if len(pred_waypoints.shape) > 2:
        trajs = [*pred_waypoints, label_waypoints]
    else:
        trajs = [pred_waypoints, label_waypoints]
    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
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


def visualize_dist_pairwise_pred(
    batch_obs_images: np.ndarray,
    batch_close_images: np.ndarray,
    batch_far_images: np.ndarray,
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    index_to_data: dict,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
):
    """
    Visualize the distance classification predictions and labels for an observation-goal image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_close_images (np.ndarray): batch of close goal images [batch_size, height, width, channels]
        batch_far_images (np.ndarray): batch of far goal images [batch_size, height, width, channels]
        batch_close_preds (np.ndarray): batch of close predictions [batch_size]
        batch_far_preds (np.ndarray): batch of far predictions [batch_size]
        batch_close_labels (np.ndarray): batch of close labels [batch_size]
        batch_far_labels (np.ndarray): batch of far labels [batch_size]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_close_images)
        == len(batch_far_images)
        == len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    if save_folder is not None:
        result_save_path = os.path.join(visualize_path, f"results.pkl")
    for i in range(min(batch_size, num_images_preds)):
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)
        obs_image = numpy_to_img(batch_obs_images[i])
        close_image = numpy_to_img(batch_close_images[i])
        far_image = numpy_to_img(batch_far_images[i])

        result_label = "f_close={}_f_far={}_curr_time={}_close_time={}_far_time={}_context_times={}".format(
            index_to_data["f_close"][i],
            index_to_data["f_far"][i],
            int(index_to_data["curr_time"][i]),
            int(index_to_data["close_time"][i]),
            int(index_to_data["far_time"][i]),
            list(torch.stack(index_to_data["context_times"]).numpy()[:, i]),
        )
        result = {
            "f_close": index_to_data["f_close"][i],
            "f_far": index_to_data["f_far"][i],
            "curr_time": int(index_to_data["curr_time"][i]),
            "close_time": int(index_to_data["close_time"][i]),
            "far_time": int(index_to_data["far_time"][i]),
            "context_times": list(torch.stack(index_to_data["context_times"]).numpy()[:, i]),
            "far_dist_pred": far_dist_pred,
            "far_dist_label": far_dist_label,
            "close_dist_pred": close_dist_pred,
            "close_dist_label": close_dist_label,
        }

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")

        if close_dist_pred < far_dist_pred:
            text_color = "black"
        else:
            text_color = "red"

        display_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_pred = {close_dist_pred}, far_pred = {far_dist_pred}",
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            text_color,
            save_path,
            display,
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
        wandb.log({f"{eval_type}_pairwise_classification": wandb_list})


def display_distance_pred(
    imgs: list,
    titles: list,
    dist_pred: float,
    dist_label: float,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))

    plt.suptitle(f"prediction: {dist_pred}\nlabel: {dist_label}", color=text_color)

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
