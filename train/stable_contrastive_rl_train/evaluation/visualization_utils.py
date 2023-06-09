import os
import yaml
import wandb
import numpy as np
import pickle as pkl

from gnm_train.visualizing.visualize_utils import numpy_to_img
from stable_contrastive_rl_train.visualizing.critic_utils import plot_oracle_critic_pred

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
