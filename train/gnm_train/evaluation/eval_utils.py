import wandb
import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Optional, Dict

# from gnm_train.visualizing.action_utils import visualize_traj_pred
# from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
from gnm_train.evaluation.visualization_utils import visualize_traj_pred
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.training.logger import Logger
from gnm_train.training.train_utils import get_total_loss
from stable_contrastive_rl_train.evaluation.eval_utils import (
    save_dist_pairwise_pred,
    save_traj_dist_pred,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam


def eval_loop(
    model: nn.Module,
    # optimizer: Adam,
    # train_dist_loader: DataLoader,
    # train_action_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    # pairwise_test_freq: int = 5,
    save_pairwise_dist_pred_freq: int = 5,
    save_traj_dist_pred_freq: int = 5,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    save_failure_index_to_data: bool = False,
    eval_waypoint: bool = True,
    eval_pairwise_dist_pred: bool = True,
    eval_traj_dist_pred: bool = True,
):
    """
    Train and evaluate the model for several epochs.

    Args:
        model: model to train
        optimizer: optimizer to use
        train_dist_loader: dataloader for training distance predictions
        train_action_loader: dataloader for training action predictions
        test_dataloaders: dict of dataloaders for testing
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        log_freq: frequency of logging to wandb
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        pairwise_test_freq: frequency of testing pairwise distance accuracy
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        load_best: whether to load the best model or not
    """
    # assert 0 <= alpha <= 1
    # latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if eval_waypoint:
            eval_total_losses = []
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} GNM Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                dist_loader = test_dataloaders[dataset_type]["distance"]
                action_loader = test_dataloaders[dataset_type]["action"]
                test_dist_loss, test_action_loss = evaluate(
                    dataset_type,
                    model,
                    dist_loader,
                    action_loader,
                    device,
                    project_folder,
                    normalized,
                    epoch,
                    alpha,
                    learn_angle,
                    print_log_freq,
                    image_log_freq,
                    num_images_log,
                    use_wandb,
                )

                total_eval_loss = get_total_loss(test_dist_loss, test_action_loss, alpha)
                eval_total_losses.append(total_eval_loss)
                print(f"{dataset_type}_total_loss: {total_eval_loss}")
                print(f"{dataset_type}_dist_loss: {test_dist_loss}")
                print(f"{dataset_type}_action_loss: {test_action_loss}")

                if use_wandb:
                    wandb.log({f"{dataset_type}_total_loss": total_eval_loss})
                    wandb.log({f"{dataset_type}_dist_loss": test_dist_loss})
                    wandb.log({f"{dataset_type}_action_loss": test_action_loss})

        if eval_pairwise_dist_pred:
            print(f"Start Pairwise Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            for dataset_type in test_dataloaders:
                if "pairwise" in test_dataloaders[dataset_type]:
                    pairwise_dist_loader = test_dataloaders[dataset_type]["pairwise"]
                    pairwise_accuracy, pairwise_auc, failure_index_to_data = pairwise_acc(
                        model,
                        pairwise_dist_loader,
                        device,
                        project_folder,
                        epoch,
                        dataset_type,
                        print_log_freq,
                        save_result_freq=save_pairwise_dist_pred_freq,
                        save_failure_index_to_data=save_failure_index_to_data,
                    )

                    if use_wandb:
                        wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
                        wandb.log({f"{dataset_type}_pairwise_auc": pairwise_auc})

                    print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
                    print(f"{dataset_type}_pairwise_auc: {pairwise_auc}")

        if eval_traj_dist_pred:
            print(f"Start Trajectory Distance Prediction Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            for dataset_type in test_dataloaders:
                if "traj_dist_pred" in test_dataloaders[dataset_type]:
                    traj_loader = test_dataloaders[dataset_type]["traj_dist_pred"]
                    traj_dist_pred(
                        model,
                        traj_loader,
                        device,
                        project_folder,
                        epoch,
                        dataset_type,
                        print_log_freq,
                        save_traj_dist_pred_freq,
                    )

    if save_failure_index_to_data:
        # save failure_idxs_to_data
        failure_index_to_data_path = os.path.join(
            project_folder,
            "pairwise_dist_prediction_failure_index_to_data.pkl",
        )
        with open(failure_index_to_data_path, "wb") as f:
            pickle.dump(failure_index_to_data, f)
        print(f"Distance pairwise prediction failure index saved to: {os.path.abspath(failure_index_to_data_path)}")

    print()


def evaluate(
    eval_type: str,
    model: nn.Module,
    eval_dist_loader: DataLoader,
    eval_action_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        eval_dist_loader (DataLoader): dataloader for distance prediction
        eval_action_loader (DataLoader): dataloader for action prediction
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        print_log_freq (int): frequency of printing loss
        image_log_freq (int): frequency of logging images
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type, window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", eval_type, window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    total_loss_logger = Logger(
        "total_loss_logger", eval_type, window_size=print_log_freq
    )

    variables = [
        dist_loss_logger,
        action_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
        total_loss_logger,
    ]
    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(eval_dist_loader), len(eval_action_loader))

    with torch.no_grad():
        for i, val in enumerate(zip(eval_dist_loader, eval_action_loader)):
            dist_vals, action_vals = val
            (
                dist_obs_image,
                dist_goal_image,
                dist_trans_obs_image,
                dist_trans_goal_image,
                dist_label,
                dist_dataset_index,
                dist_index_to_data,
            ) = dist_vals
            (
                action_obs_image,
                action_goal_image,
                action_trans_obs_image,
                action_trans_goal_image,
                action_goal_pos,
                action_label,
                action_dataset_index,
                action_index_to_data,
            ) = action_vals
            dist_obs_data = dist_trans_obs_image.to(device)
            dist_goal_data = dist_trans_goal_image.to(device)
            dist_label = dist_label.to(device)

            dist_pred, _ = model(dist_obs_data, dist_goal_data)
            dist_loss = F.mse_loss(dist_pred, dist_label)

            action_obs_data = action_trans_obs_image.to(device)
            action_goal_data = action_trans_goal_image.to(device)
            action_label = action_label.to(device)

            _, action_pred = model(action_obs_data, action_goal_data)
            action_loss = F.mse_loss(action_pred, action_label)
            action_waypts_cos_sim = F.cosine_similarity(
                action_pred[:2], action_label[:2], dim=-1
            ).mean()
            multi_action_waypts_cos_sim = F.cosine_similarity(
                torch.flatten(action_pred[:2], start_dim=1),
                torch.flatten(action_label[:2], start_dim=1),
                dim=-1,
            ).mean()
            if learn_angle:
                action_orien_cos_sim = F.cosine_similarity(
                    action_pred[2:], action_label[2:], dim=-1
                ).mean()
                multi_action_orien_cos_sim = F.cosine_similarity(
                    torch.flatten(action_pred[2:], start_dim=1),
                    torch.flatten(action_label[2:], start_dim=1),
                    dim=-1,
                ).mean()
                action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
                multi_action_orien_cos_sim_logger.log_data(
                    multi_action_orien_cos_sim.item()
                )

            total_loss = alpha * (1e-3 * dist_loss) + (1 - alpha) * action_loss

            dist_loss_logger.log_data(dist_loss.item())
            action_loss_logger.log_data(action_loss.item())
            action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            multi_action_waypts_cos_sim_logger.log_data(
                multi_action_waypts_cos_sim.item()
            )
            total_loss_logger.log_data(total_loss.item())

            if i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            if i % image_log_freq == 0:
                visualize_traj_pred(
                    to_numpy(action_obs_image),
                    to_numpy(action_goal_image),
                    to_numpy(action_dataset_index),
                    to_numpy(action_goal_pos),
                    to_numpy(action_pred),
                    to_numpy(action_label),
                    eval_type,
                    normalized,
                    project_folder,
                    epoch,
                    action_index_to_data,
                    num_images_log,
                    use_wandb=use_wandb,
                )
    data_log = {}
    for var in variables:
        log_display = f"(epoch {epoch}) "
        data_log[var.full_name()] = var.average()
        print(log_display + var.display())
    print()
    if use_wandb:
        wandb.log(data_log)
    return dist_loss_logger.average(), action_loss_logger.average()


def pairwise_acc(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    save_folder: str,
    epoch: int,
    eval_type: str,
    print_log_freq: int = 100,
    save_result_freq: int = 1000,
    save_failure_index_to_data: bool = False,
):
    """
    Evaluate the model on the pairwise distance accuracy metric. Given 1 observation and 2 subgoals, the model should determine which goal is closer.

    Args:
        model (nn.Module): The model to evaluate.
        eval_loader (DataLoader): The dataloader for the evaluation dataset.
        device (torch.device): The device to use for evaluation.
        save_folder (str): The folder to save the evaluation results.
        epoch (int): The current epoch.
        eval_type (str): The type of evaluation. Can be "train" or "val".
        print_log_freq (int, optional): The frequency at which to print the evaluation results. Defaults to 100.
        image_log_freq (int, optional): The frequency at which to log the evaluation results. Defaults to 1000.
        num_images_log (int, optional): The number of images to log. Defaults to 32.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to True.
        display (bool, optional): Whether to display the evaluation results. Defaults to False.
    """
    correct_list = []
    auc_list = []
    failure_index_to_data = dict(
        f_close=[], f_far=[], curr_time=[], close_time=[], far_time=[], context_times=[])

    model.eval()
    num_batches = len(eval_loader)

    with torch.no_grad():
        for i, vals in enumerate(eval_loader):
            (
                obs_image,
                close_image,
                far_image,
                transf_obs_image,
                transf_close_image,
                transf_far_image,
                close_dist_label,
                far_dist_label,
                index_to_data,
            ) = vals
            batch_size = transf_obs_image.shape[0]
            transf_obs_image = transf_obs_image.to(device)
            transf_close_image = transf_close_image[:, -3:].to(device)
            transf_far_image = transf_far_image.to(device)

            close_pred, _ = model(transf_obs_image, transf_close_image[:, -3:])
            far_pred, _ = model(transf_obs_image, transf_far_image[:, -3:])

            close_pred_flat = close_pred.reshape(close_pred.shape[0])
            far_pred_flat = far_pred.reshape(far_pred.shape[0])

            close_pred_flat = to_numpy(close_pred_flat)
            far_pred_flat = to_numpy(far_pred_flat)

            correct = np.where(far_pred_flat > close_pred_flat, 1, 0)
            if save_failure_index_to_data:
                failure_idx = np.arange(batch_size)[np.logical_not(correct)]
            correct_list.append(correct.copy())
            correct[batch_size // 2:] = np.logical_not(correct[batch_size // 2:]).astype(np.int)

            if save_failure_index_to_data:
                failure_index_to_data["f_close"].extend(
                    np.array(index_to_data["f_close"])[failure_idx].tolist())
                failure_index_to_data["f_far"].extend(
                    np.array(index_to_data["f_far"])[failure_idx].tolist())
                failure_index_to_data["curr_time"].extend(
                    index_to_data["curr_time"][failure_idx].numpy().tolist())
                failure_index_to_data["close_time"].extend(
                    index_to_data["close_time"][failure_idx].numpy().tolist())
                failure_index_to_data["far_time"].extend(
                    index_to_data["far_time"][failure_idx].numpy().tolist())
                failure_index_to_data["context_times"].extend(
                    index_to_data["context_times"][failure_idx].numpy().tolist())

            # compute AUC here: does this make sense ewith binary classifier predicting 0/1 only?
            # use the difference between regression numbers as score here.
            auc = roc_auc_score(
                np.concatenate([np.ones_like(correct[:batch_size // 2]),
                                np.zeros_like(correct[batch_size // 2:])]),
                np.concatenate([(far_pred_flat - close_pred_flat)[:batch_size // 2],
                                (close_pred_flat - far_pred_flat)[batch_size // 2:]])
            )
            auc_list.append(auc)

            if i % print_log_freq == 0:
                print(f"({i}/{num_batches}) batch of points processed")

            if i % save_result_freq == 0:
                save_dist_pairwise_pred(
                    to_numpy(close_pred),
                    to_numpy(far_pred),
                    to_numpy(close_dist_label),
                    to_numpy(far_dist_label),
                    eval_type,
                    save_folder,
                    epoch,
                    index_to_data,
                )
        if len(correct_list) == 0:
            return 0
        return np.concatenate(correct_list).mean(), np.asarray(auc_list).mean(), failure_index_to_data


def traj_dist_pred(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    save_folder: str,
    epoch: int,
    eval_type: str,
    print_log_freq: int = 100,
    save_result_freq: int = 1000,
):
    model.eval()
    num_trajs = len(eval_loader)

    with torch.no_grad():
        for i, vals in enumerate(eval_loader):
            (
                obs_image,
                goal_image,
                transf_obs_image,
                transf_goal_image,
                obs_latlong,
                goal_latlong,
                local_goal_pos,
                waypoint_label,
                global_obs_pos,
                global_goal_pos,
                dist_label,
                dataset_index,
                index_to_traj,
            ) = tuple([v[0] for v in vals[:-1]] + [vals[-1]])
            traj_len = transf_obs_image.shape[0]
            transf_obs_image = transf_obs_image.to(device)
            transf_goal_image = transf_goal_image.to(device)

            # planning with GNM
            current_obs_idx = 0
            path_obs_idxs = [current_obs_idx]

            while current_obs_idx != traj_len - 1 and len(path_obs_idxs) < traj_len:
                mask = torch.zeros(transf_obs_image.shape[0], dtype=torch.bool, device=device)
                sg_indices = torch.arange(traj_len, dtype=torch.int, device=device)
                mask[current_obs_idx] = True
                transf_curr_obs_image = transf_obs_image[mask]
                transf_sg_image = transf_obs_image[~mask]
                sg_indices = sg_indices[~mask]

                obs_sg_dist = model(
                    transf_curr_obs_image.repeat_interleave(transf_sg_image.shape[0], dim=0),
                    transf_sg_image
                )[0]
                sg_g_dist = model(
                    transf_sg_image,
                    transf_goal_image[:transf_sg_image.shape[0]]  # transf_goal_images are the same for the entire trajectory
                )[0]

                sg_idx = int(sg_indices[torch.argmin(obs_sg_dist + sg_g_dist)])

                # assume we can move to the subgoal exactly
                path_obs_idxs.append(sg_idx)
                current_obs_idx = sg_idx

            if i % print_log_freq == 0:
                print(f"({i}/{num_trajs}) trajectories processed")

            if i % save_result_freq == 0:
                save_traj_dist_pred(
                    to_numpy(obs_latlong),
                    to_numpy(goal_latlong),
                    to_numpy(global_obs_pos),
                    to_numpy(global_goal_pos),
                    np.array(path_obs_idxs),
                    eval_type,
                    save_folder,
                    epoch,
                    index_to_traj,
                )
