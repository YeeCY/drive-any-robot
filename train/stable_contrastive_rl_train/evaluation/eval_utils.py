import wandb
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict

from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
from stable_contrastive_rl_train.visualizing.critic_utils import visualize_critic_pred
from stable_contrastive_rl_train.training.train_utils import (
    get_critic_loss,
    get_actor_loss,
)
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.training.logger import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.optim import Adam
from torch.optim import Optimizer
from torch.distributions import (
    Normal,
    Independent
)


def eval_rl_loop(
    model: nn.Module,
    # optimizer: Dict[str, Optimizer],
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
    pairwise_test_freq: int = 5,
    current_epoch: int = 0,
    # target_update_freq: int = 1,
    discount: float = 0.99,
    use_td: bool = True,
    bc_coef: float = 0.05,
    mle_gcbc_loss: bool = False,
    # stop_grad_actor_img_encoder: bool = True,
    use_actor_waypoint_q_loss: bool = False,
    use_actor_dist_q_loss: bool = False,
    learn_angle: bool = True,
    use_wandb: bool = True,
):
    """
    Train and evaluate the model for several epochs.

    Args:
        model: model to train
        optimizer: optimizer to use
        train_rl_loader: dataloader for training RL algorithm
        test_dataloaders: dict of dataloaders for testing
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        log_freq: frequency of logging to wandb
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        pairwise_test_freq: frequency of testing pairwise distance accuracy
        current_epoch: epoch to start training from
        discount: discount factor
        use_td: whether to use C-Learning (TD) or Contrastive NCE (MC)
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        load_best: whether to load the best model or not
    """
    assert 0 <= discount <= 1

    for epoch in range(current_epoch, current_epoch + epochs):
        # print(
        #     f"Start Stable Contrastive RL Training Epoch {epoch}/{current_epoch + epochs - 1}"
        # )
        # train(
        #     model,
        #     optimizer,
        #     train_dist_loader,
        #     train_action_loader,
        #     device,
        #     project_folder,
        #     normalized,
        #     epoch,
        #     target_update_freq,
        #     discount,
        #     use_td,
        #     bc_coef,
        #     mle_gcbc_loss,
        #     stop_grad_actor_img_encoder,
        #     use_actor_waypoint_q_loss,
        #     use_actor_dist_q_loss,
        #     learn_angle,
        #     print_log_freq,
        #     image_log_freq,
        #     num_images_log,
        #     use_wandb,
        # )

        # eval_total_losses = []
        eval_critic_losses, eval_actor_losses = [], []
        for dataset_type in test_dataloaders:
            # print(
            #     f"Start {dataset_type} Stable Contrastive RL Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            # )
            # dist_loader = test_dataloaders[dataset_type]["distance"]
            # action_loader = test_dataloaders[dataset_type]["action"]
            # test_critic_loss, test_actor_loss = evaluate(
            #     dataset_type,
            #     model,
            #     dist_loader,
            #     action_loader,
            #     device,
            #     project_folder,
            #     normalized,
            #     epoch,
            #     discount,
            #     use_td,
            #     bc_coef,
            #     mle_gcbc_loss,
            #     use_actor_waypoint_q_loss,
            #     use_actor_dist_q_loss,
            #     learn_angle,
            #     print_log_freq,
            #     image_log_freq,
            #     num_images_log,
            #     use_wandb,
            # )
            #
            # # total_eval_loss = get_total_loss(test_dist_loss, test_action_loss, alpha)
            # # eval_total_losses.append(total_eval_loss)
            # eval_critic_losses.append(test_critic_loss)
            # eval_actor_losses.append(test_actor_loss)
            # # wandb.log({f"{dataset_type}_total_loss": total_eval_loss})
            # # print(f"{dataset_type}_total_loss: {total_eval_loss}")
            #
            # if use_wandb:
            #     wandb.log({f"{dataset_type}_critic_loss": test_critic_loss})
            #     wandb.log({f"{dataset_type}_actor_loss": test_actor_loss})
            #
            # print(f"{dataset_type}_critic_loss: {test_critic_loss}")
            # print(f"{dataset_type}_actor_loss: {test_actor_loss}")

            print(f"Start Pairwise Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            for dataset_type in test_dataloaders:
                if "pairwise" in test_dataloaders[dataset_type]:
                    pairwise_dist_loader = test_dataloaders[dataset_type]["pairwise"]
                    pairwise_accuracy, pairwise_auc = pairwise_acc(
                        model,
                        pairwise_dist_loader,
                        device,
                        project_folder,
                        epoch,
                        dataset_type,
                        print_log_freq,
                        image_log_freq,
                        num_images_log,
                        use_wandb=use_wandb,
                    )

                    if use_wandb:
                        wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
                        wandb.log({f"{dataset_type}_pairwise_auc": pairwise_auc})

                    print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
                    print(f"{dataset_type}_pairwise_auc: {pairwise_auc}")
    print()


def train(
    model: nn.Module,
    optimizer: Dict[str, Optimizer],
    train_dist_loader: DataLoader,
    train_waypoint_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    target_update_freq: int = 1,
    discount: float = 0.99,
    use_td: bool = True,
    bc_coef: float = 0.05,
    mle_gcbc_loss: bool = False,
    stop_grad_actor_img_encoder: bool = True,
    use_actor_waypoint_q_loss: bool = False,
    use_actor_dist_q_loss: bool = False,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        train_rl_loader: dataloader for RL training
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        use_td: whether to use C-Learning (TD) or Contrastive NCE (MC)
        bc_coef: behavioral cloning regularization coefficient.
        stop_grad_actor_img_encoder: whether to stop gradients from actor loss to policy image encoder
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    # critic loggers
    critic_loss_logger = Logger("critic_loss", "train", window_size=print_log_freq)
    waypoint_binary_acc_logger = Logger("critic/waypoint_binary_accuracy", "train", window_size=print_log_freq)
    waypoint_categorical_acc_logger = Logger("critic/waypoint_categorical_accuracy", "train", window_size=print_log_freq)
    waypoint_logits_pos_logger = Logger("critic/waypoint_logits_pos", "train", window_size=print_log_freq)
    waypoint_logits_neg_logger = Logger("critic/waypoint_logits_neg", "train", window_size=print_log_freq)
    waypoint_logits_logger = Logger("critic/waypoint_logits", "train", window_size=print_log_freq)
    dist_binary_acc_logger = Logger("critic/dist_binary_accuracy", "train", window_size=print_log_freq)
    dist_categorical_acc_logger = Logger("critic/dist_categorical_accuracy", "train", window_size=print_log_freq)
    dist_logits_pos_logger = Logger("critic/dist_logits_pos", "train", window_size=print_log_freq)
    dist_logits_neg_logger = Logger("critic/dist_logits_neg", "train", window_size=print_log_freq)
    dist_logits_logger = Logger("critic/dist_logits", "train", window_size=print_log_freq)

    # actor loggers
    actor_loss_logger = Logger("actor_loss", "train", window_size=print_log_freq)
    actor_waypoint_q_loss = Logger("actor/actor_waypoint_q_loss", "train", window_size=print_log_freq)
    actor_dist_q_loss = Logger("actor/actor_dist_q_loss", "train", window_size=print_log_freq)
    actor_q_loss_logger = Logger("actor/actor_q_loss", "train", window_size=print_log_freq)
    gcbc_mle_loss_logger = Logger("actor/gcbc_mle_loss", "train", window_size=print_log_freq)
    gcbc_mse_loss_logger = Logger("actor/gcbc_mse_loss", "train", window_size=print_log_freq)
    waypoint_gcbc_mle_loss_logger = Logger("actor/waypoint_gcbc_mle_loss", "train", window_size=print_log_freq)
    waypoint_gcbc_mse_loss_logger = Logger("actor/waypoint_gcbc_mse_loss", "train", window_size=print_log_freq)
    dist_gcbc_mle_loss_logger = Logger("actor/dist_gcbc_mle_loss", "train", window_size=print_log_freq)
    dist_gcbc_mse_loss_logger = Logger("actor/dist_gcbc_mse_loss", "train", window_size=print_log_freq)

    # waypoint loggers
    action_waypts_cos_sim_logger = Logger(
        "actor/action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "actor/multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )

    variables = [
        critic_loss_logger,
        waypoint_binary_acc_logger,
        waypoint_categorical_acc_logger,
        waypoint_logits_pos_logger,
        waypoint_logits_neg_logger,
        waypoint_logits_logger,
        dist_binary_acc_logger,
        dist_categorical_acc_logger,
        dist_logits_pos_logger,
        dist_logits_neg_logger,
        dist_logits_logger,
        actor_loss_logger,
        actor_q_loss_logger,
        actor_waypoint_q_loss,
        actor_dist_q_loss,
        gcbc_mle_loss_logger,
        gcbc_mse_loss_logger,
        waypoint_gcbc_mle_loss_logger,
        waypoint_gcbc_mse_loss_logger,
        dist_gcbc_mle_loss_logger,
        dist_gcbc_mse_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
    ]

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "actor/action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "actor/multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(train_dist_loader), len(train_waypoint_loader))
    for i, val in enumerate(zip(train_dist_loader, train_waypoint_loader)):
        dist_vals, waypoint_vals = val
        (
            dist_obs_image,
            dist_next_obs_image,
            dist_goal_image,
            dist_trans_obs_image,
            dist_trans_next_obs_image,
            dist_trans_goal_image,
            dist_label,
            dist_dataset_index,
        ) = dist_vals
        (
            waypoint_obs_image,
            waypoint_next_obs_image,
            waypoint_goal_image,
            waypoint_trans_obs_image,
            waypoint_trans_next_obs_image,
            waypoint_trans_goal_image,
            waypoint_goal_pos,
            waypoint_label,
            waypoint_oracle,
            waypoint_dataset_index,
        ) = waypoint_vals
        # (
        #     obs_image,
        #     next_obs_image,
        #     goal_image,
        #     trans_obs_image,
        #     trans_next_obs_image,
        #     trans_goal_image,
        #     goal_pos,
        #     action_label,
        #     oracle_action,
        #     dist_label,
        #     dataset_index,
        # ) = vals
        dist_obs_data = dist_trans_obs_image.to(device)
        dist_next_obs_data = dist_trans_next_obs_image.to(device)
        dist_goal_data = dist_trans_goal_image.to(device)
        dist_label = dist_label.to(device)

        waypoint_obs_data = waypoint_trans_obs_image.to(device)
        waypoint_next_obs_data = waypoint_trans_next_obs_image.to(device)
        waypoint_goal_data = waypoint_trans_goal_image.to(device)
        waypoint_label = waypoint_label.to(device)
        waypoint_oracle = waypoint_oracle[:num_images_log]

        obs_data = (waypoint_obs_data, dist_obs_data)
        next_obs_data = (waypoint_next_obs_data, dist_next_obs_data)
        action_data = (waypoint_label.reshape([waypoint_label.shape[0], -1]),
                       dist_label)
        goal_data = (waypoint_goal_data, dist_goal_data)

        # obs_data = trans_obs_image.to(device)
        # next_obs_data = trans_next_obs_image.to(device)
        # goal_data = trans_goal_image.to(device)
        # action_label = action_label.to(device)
        # oracle_action = oracle_action[:num_images_log].to(device)
        #
        # dist_label = dist_label.to(device)
        # action_data = torch.cat([
        #     action_label.reshape([action_label.shape[0], -1]),
        #     dist_label
        # ], dim=-1)

        # save oracle data to cpu cause we use cpu to do inference here
        waypoint_oracle_obs_data = waypoint_obs_data[:num_images_log, None].repeat_interleave(
            waypoint_oracle.shape[1], dim=1).flatten(0, 1).cpu()
        waypoint_oracle_goal_data = waypoint_goal_data[:num_images_log, None].repeat_interleave(
            waypoint_oracle.shape[1], dim=1).flatten(0, 1).cpu()
        dummy_dist = torch.zeros_like(dist_label)[:num_images_log, None].repeat_interleave(
            waypoint_oracle.shape[1], dim=1).flatten(0, 1).cpu()
        # waypoint_oracle = waypoint_oracle[:num_images_log].flatten(
        #     start_dim=0, end_dim=1).flatten(1).cpu()
        # oracle_action_data = torch.cat([
        #     oracle_action.flatten(0, 1).flatten(-2, -1).cpu(),
        #     dist_label[:num_images_log, None].repeat_interleave(oracle_action.shape[1], dim=1).flatten(0, 1).cpu()
        # ], dim=-1)

        # Important: the order of loss computation and optimizer update matter!
        # compute critic loss
        critic_loss, critic_info = get_critic_loss(
            model, obs_data, next_obs_data, action_data, goal_data,
            discount, use_td=use_td)

        # optimize critic
        optimizer["critic_optimizer"].zero_grad()
        critic_loss.backward()
        optimizer["critic_optimizer"].step()

        # compute actor loss
        actor_loss, actor_info = get_actor_loss(
            model, obs_data, action_data, goal_data,
            bc_coef=bc_coef, mle_gcbc_loss=mle_gcbc_loss,
            stop_grad_actor_img_encoder=stop_grad_actor_img_encoder,
            use_actor_waypoint_q_loss=use_actor_waypoint_q_loss,
            use_actor_dist_q_loss=use_actor_dist_q_loss)

        # optimize actor
        optimizer["actor_optimizer"].zero_grad()
        actor_loss.backward()
        optimizer["actor_optimizer"].step()

        # preds = model.policy_network(obs_data, goal_data).mean
        # action_data are not used here
        # model.cpu()
        waypoint_pred, dist_pred = model(waypoint_obs_data, dist_obs_data,
                                         waypoint_label.flatten(1), dist_label,
                                         waypoint_goal_data, dist_goal_data)[-4:-2]
        waypoint_pred = waypoint_pred.reshape(waypoint_label.shape)
        # preds, _ = model.policy_network(obs_data.cpu(), goal_data.cpu())
        # model = model.to(device)

        # The action of policy is different from the action here (waypoints).
        # action_pred = preds[:, :-1]
        # action_pred = action_pred.reshape(action_label.shape)
        # action_label = action_label.cpu()

        # dist_pred = preds[:, -1]

        action_waypts_cos_sim = F.cosine_similarity(
            waypoint_pred[:2], waypoint_label[:2], dim=-1
        ).mean()
        multi_action_waypts_cos_sim = F.cosine_similarity(
            torch.flatten(waypoint_pred[:2], start_dim=1),
            torch.flatten(waypoint_label[:2], start_dim=1),
            dim=-1,
        ).mean()
        if learn_angle:
            action_orien_cos_sim = F.cosine_similarity(
                waypoint_pred[2:], waypoint_label[2:], dim=-1
            ).mean()
            multi_action_orien_cos_sim = F.cosine_similarity(
                torch.flatten(waypoint_pred[2:], start_dim=1),
                torch.flatten(waypoint_label[2:], start_dim=1),
                dim=-1,
            ).mean()
            action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
            multi_action_orien_cos_sim_logger.log_data(
                multi_action_orien_cos_sim.item()
            )

        # prevent NAN via gradient clipping
        # total_loss.backward()
        # optimizer.step()
        # optimizer["actor_optimizer"].zero_grad()
        # actor_loss.backward()
        # torch.nn.utils.clip_grad_norm(
        #     model.policy_network.parameters(), 1.0)
        # optimizer["actor_optimizer"].step()

        # optimizer["critic_optimizer"].zero_grad()
        # critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(
        #     model.q_network.parameters(), 1.0)
        # optimizer["critic_optimizer"].step()
        # optimizer['optimizer'].zero_grad()
        # (actor_loss + critic_loss).backward()
        # optimizer['optimizer'].step()

        if i % target_update_freq == 0:
            model.soft_update_target_q_network()

        critic_loss_logger.log_data(critic_loss.item())
        waypoint_binary_acc_logger.log_data(critic_info["waypoint_binary_accuracy"].item())
        waypoint_categorical_acc_logger.log_data(critic_info["waypoint_categorical_accuracy"].item())
        waypoint_logits_pos_logger.log_data(critic_info["waypoint_logits_pos"].item())
        waypoint_logits_neg_logger.log_data(critic_info["waypoint_logits_neg"].item())
        waypoint_logits_logger.log_data(critic_info["waypoint_logits"].item())
        dist_binary_acc_logger.log_data(critic_info["dist_binary_accuracy"].item())
        dist_categorical_acc_logger.log_data(critic_info["dist_categorical_accuracy"].item())
        dist_logits_pos_logger.log_data(critic_info["dist_logits_pos"].item())
        dist_logits_neg_logger.log_data(critic_info["dist_logits_neg"].item())
        dist_logits_logger.log_data(critic_info["dist_logits"].item())

        actor_loss_logger.log_data(actor_loss.item())
        actor_waypoint_q_loss.log_data(actor_info["actor_waypoint_q_loss"].item())
        actor_dist_q_loss.log_data(actor_info["actor_dist_q_loss"].item())
        actor_q_loss_logger.log_data(actor_info["actor_q_loss"].item())
        gcbc_mle_loss_logger.log_data(actor_info["gcbc_mle_loss"].item())
        gcbc_mse_loss_logger.log_data(actor_info["gcbc_mse_loss"].item())
        waypoint_gcbc_mle_loss_logger.log_data(actor_info["waypoint_gcbc_mle_loss"].item())
        waypoint_gcbc_mse_loss_logger.log_data(actor_info["waypoint_gcbc_mse_loss"].item())
        dist_gcbc_mle_loss_logger.log_data(actor_info["dist_gcbc_mle_loss"].item())
        dist_gcbc_mse_loss_logger.log_data(actor_info["dist_gcbc_mse_loss"].item())

        action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
        multi_action_waypts_cos_sim_logger.log_data(multi_action_waypts_cos_sim.item())

        if use_wandb:
            data_log = {}
            for var in variables:
                data_log[var.full_name()] = var.latest()
            wandb.log(data_log)

        if i % print_log_freq == 0:
            log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
            for var in variables:
                print(log_display + var.display())
            print()

        if i % image_log_freq == 0:
            visualize_dist_pred(
                to_numpy(dist_obs_image),
                to_numpy(dist_goal_image),
                to_numpy(dist_pred),
                to_numpy(dist_label),
                "train",
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
            # visualize_traj_pred(
            #     to_numpy(obs_image),
            #     to_numpy(goal_image),
            #     to_numpy(dataset_index),
            #     to_numpy(goal_pos),
            #     to_numpy(action_pred),
            #     to_numpy(action_label),
            #     "train",
            #     normalized,
            #     project_folder,
            #     epoch,
            #     num_images_log,
            #     use_wandb=use_wandb,
            # )

            # TODO (chongyiz): move this code block above
            # critic prediction for oracle actions
            # we do inference on cpu cause the batch size is too large
            model.cpu()
            waypoint_obs_repr, _, waypoint_g_repr, _ = model.q_network(
                waypoint_oracle_obs_data, waypoint_oracle_obs_data,
                waypoint_oracle.flatten(start_dim=0, end_dim=1).flatten(1), dummy_dist,
                waypoint_oracle_goal_data, waypoint_oracle_goal_data)
            waypoint_oracle_logit = torch.einsum('ikl,jkl->ijl', waypoint_obs_repr, waypoint_g_repr)
            waypoint_oracle_logit = torch.diag(torch.mean(waypoint_oracle_logit, dim=-1)).reshape(
                waypoint_oracle.shape[0], waypoint_oracle.shape[1], 1)
            waypoint_oracle_critic = torch.sigmoid(waypoint_oracle_logit)
            model = model.to(device)
            visualize_critic_pred(
                to_numpy(waypoint_obs_image),
                to_numpy(waypoint_goal_image),
                to_numpy(waypoint_dataset_index),
                to_numpy(waypoint_goal_pos),
                to_numpy(waypoint_oracle),
                to_numpy(waypoint_oracle_critic),
                to_numpy(waypoint_pred),
                to_numpy(waypoint_label),
                "train",
                normalized,
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
    return


def evaluate(
    eval_type: str,
    model: nn.Module,
    eval_dist_loader: DataLoader,
    eval_action_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    discount: float = 0.99,
    use_td: bool = True,
    bc_coef: float = 0.05,
    mle_gcbc_loss: bool = False,
    use_actor_waypoint_q_loss: bool = False,
    use_actor_dist_q_loss: bool = False,
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
        # eval_rl_loader (DataLoader): dataloader for evaluating RL algorithm
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        target_update_freq (int):
        discount (float):
        learn_angle (bool): whether to learn the angle of the action
        print_log_freq (int): frequency of printing loss
        image_log_freq (int): frequency of logging images
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
    """
    model.eval()

    # critic loggers
    critic_loss_logger = Logger("critic_loss", eval_type, window_size=print_log_freq)
    waypoint_binary_acc_logger = Logger("critic/waypoint_binary_accuracy", eval_type, window_size=print_log_freq)
    waypoint_categorical_acc_logger = Logger("critic/waypoint_categorical_accuracy", eval_type,
                                             window_size=print_log_freq)
    waypoint_logits_pos_logger = Logger("critic/waypoint_logits_pos", eval_type, window_size=print_log_freq)
    waypoint_logits_neg_logger = Logger("critic/waypoint_logits_neg", eval_type, window_size=print_log_freq)
    waypoint_logits_logger = Logger("critic/waypoint_logits", eval_type, window_size=print_log_freq)
    dist_binary_acc_logger = Logger("critic/dist_binary_accuracy", eval_type, window_size=print_log_freq)
    dist_categorical_acc_logger = Logger("critic/dist_categorical_accuracy", eval_type, window_size=print_log_freq)
    dist_logits_pos_logger = Logger("critic/dist_logits_pos", eval_type, window_size=print_log_freq)
    dist_logits_neg_logger = Logger("critic/dist_logits_neg", eval_type, window_size=print_log_freq)
    dist_logits_logger = Logger("critic/dist_logits", eval_type, window_size=print_log_freq)

    # actor loggers
    actor_loss_logger = Logger("actor_loss", eval_type, window_size=print_log_freq)
    actor_waypoint_q_loss = Logger("actor/actor_waypoint_q_loss", eval_type, window_size=print_log_freq)
    actor_dist_q_loss = Logger("actor/actor_dist_q_loss", eval_type, window_size=print_log_freq)
    actor_q_loss_logger = Logger("actor/actor_q_loss", eval_type, window_size=print_log_freq)
    gcbc_mle_loss_logger = Logger("actor/gcbc_mle_loss", eval_type, window_size=print_log_freq)
    gcbc_mse_loss_logger = Logger("actor/gcbc_mse_loss", eval_type, window_size=print_log_freq)
    waypoint_gcbc_mle_loss_logger = Logger("actor/waypoint_gcbc_mle_loss", eval_type, window_size=print_log_freq)
    waypoint_gcbc_mse_loss_logger = Logger("actor/waypoint_gcbc_mse_loss", eval_type, window_size=print_log_freq)
    dist_gcbc_mle_loss_logger = Logger("actor/dist_gcbc_mle_loss", eval_type, window_size=print_log_freq)
    dist_gcbc_mse_loss_logger = Logger("actor/dist_gcbc_mle_loss", eval_type, window_size=print_log_freq)

    # waypoint loggers
    action_waypts_cos_sim_logger = Logger(
        "actor/action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "actor/multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )

    variables = [
        critic_loss_logger,
        waypoint_binary_acc_logger,
        waypoint_categorical_acc_logger,
        waypoint_logits_pos_logger,
        waypoint_logits_neg_logger,
        waypoint_logits_logger,
        dist_binary_acc_logger,
        dist_categorical_acc_logger,
        dist_logits_pos_logger,
        dist_logits_neg_logger,
        dist_logits_logger,
        actor_loss_logger,
        actor_q_loss_logger,
        actor_waypoint_q_loss,
        actor_dist_q_loss,
        gcbc_mle_loss_logger,
        gcbc_mse_loss_logger,
        waypoint_gcbc_mle_loss_logger,
        waypoint_gcbc_mse_loss_logger,
        dist_gcbc_mle_loss_logger,
        dist_gcbc_mse_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
    ]
    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "actor/action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "actor/multi_action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(eval_dist_loader), len(eval_action_loader))

    with torch.no_grad():
        for i, val in enumerate(zip(eval_dist_loader, eval_action_loader)):
            dist_vals, waypoint_vals = val
            (
                dist_obs_image,
                dist_next_obs_image,
                dist_goal_image,
                dist_trans_obs_image,
                dist_trans_next_obs_image,
                dist_trans_goal_image,
                dist_label,
                dist_dataset_index,
            ) = dist_vals
            (
                waypoint_obs_image,
                waypoint_next_obs_image,
                waypoint_goal_image,
                waypoint_trans_obs_image,
                waypoint_trans_next_obs_image,
                waypoint_trans_goal_image,
                waypoint_goal_pos,
                waypoint_label,
                waypoint_oracle,
                waypoint_dataset_index,
            ) = waypoint_vals
            dist_obs_data = dist_trans_obs_image.to(device)
            dist_next_obs_data = dist_trans_next_obs_image.to(device)
            dist_goal_data = dist_trans_goal_image.to(device)
            dist_label = dist_label.to(device)

            waypoint_obs_data = waypoint_trans_obs_image.to(device)
            waypoint_next_obs_data = waypoint_trans_next_obs_image.to(device)
            waypoint_goal_data = waypoint_trans_goal_image.to(device)
            waypoint_label = waypoint_label.to(device)
            waypoint_oracle = waypoint_oracle[:num_images_log]

            obs_data = (waypoint_obs_data, dist_obs_data)
            next_obs_data = (waypoint_next_obs_data, dist_next_obs_data)
            action_data = (waypoint_label.reshape([waypoint_label.shape[0], -1]),
                           dist_label)
            goal_data = (waypoint_goal_data, dist_goal_data)

            # save oracle data to cpu cause we use cpu to do inference here
            waypoint_oracle_obs_data = waypoint_obs_data[:num_images_log, None].repeat_interleave(
                waypoint_oracle.shape[1], dim=1).flatten(0, 1).cpu()
            waypoint_oracle_goal_data = waypoint_goal_data[:num_images_log, None].repeat_interleave(
                waypoint_oracle.shape[1], dim=1).flatten(0, 1).cpu()
            dummy_dist = torch.zeros_like(dist_label)[:num_images_log, None].repeat_interleave(
                waypoint_oracle.shape[1], dim=1).flatten(0, 1).cpu()

            critic_loss, critic_info = get_critic_loss(
                model, obs_data, next_obs_data, action_data, goal_data,
                discount, use_td=use_td)

            actor_loss, actor_info = get_actor_loss(
                model, obs_data, action_data, goal_data,
                bc_coef=bc_coef, mle_gcbc_loss=mle_gcbc_loss,
                use_actor_waypoint_q_loss=use_actor_waypoint_q_loss,
                use_actor_dist_q_loss=use_actor_dist_q_loss)
            # gcbc_loss = 0.5 * (1e-3 * dist_gcbc_loss) + 0.5 * waypoint_gcbc_loss

            # preds = model.policy_network(obs_data, goal_data).mean
            # action_data are not used here
            # model.cpu()
            waypoint_pred, dist_pred = model(waypoint_obs_data, dist_obs_data,
                                             waypoint_label.flatten(1), dist_label,
                                             waypoint_goal_data, dist_goal_data)[-4:-2]
            waypoint_pred = waypoint_pred.reshape(waypoint_label.shape)
            # preds, _ = model.policy_network(obs_data.cpu(), goal_data.cpu())
            # model = model.to(device)

            # The action of policy is different from the action here (waypoints).
            # action_pred = preds[:, :-1]
            # action_pred = action_pred.reshape(action_label.shape)
            # action_label = action_label.cpu()

            # dist_pred = preds[:, -1]

            action_waypts_cos_sim = F.cosine_similarity(
                waypoint_pred[:2], waypoint_label[:2], dim=-1
            ).mean()
            multi_action_waypts_cos_sim = F.cosine_similarity(
                torch.flatten(waypoint_pred[:2], start_dim=1),
                torch.flatten(waypoint_pred[:2], start_dim=1),
                dim=-1,
            ).mean()
            if learn_angle:
                action_orien_cos_sim = F.cosine_similarity(
                    waypoint_pred[2:], waypoint_label[2:], dim=-1
                ).mean()
                multi_action_orien_cos_sim = F.cosine_similarity(
                    torch.flatten(waypoint_pred[2:], start_dim=1),
                    torch.flatten(waypoint_label[2:], start_dim=1),
                    dim=-1,
                ).mean()
                action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
                multi_action_orien_cos_sim_logger.log_data(
                    multi_action_orien_cos_sim.item()
                )

            # total_loss = alpha * (1e-3 * dist_loss) + (1 - alpha) * action_loss

            # dist_loss_logger.log_data(dist_loss.item())
            # action_loss_logger.log_data(action_loss.item())

            critic_loss_logger.log_data(critic_loss.item())
            waypoint_binary_acc_logger.log_data(critic_info["waypoint_binary_accuracy"].item())
            waypoint_categorical_acc_logger.log_data(critic_info["waypoint_categorical_accuracy"].item())
            waypoint_logits_pos_logger.log_data(critic_info["waypoint_logits_pos"].item())
            waypoint_logits_neg_logger.log_data(critic_info["waypoint_logits_neg"].item())
            waypoint_logits_logger.log_data(critic_info["waypoint_logits"].item())
            dist_binary_acc_logger.log_data(critic_info["dist_binary_accuracy"].item())
            dist_categorical_acc_logger.log_data(critic_info["dist_categorical_accuracy"].item())
            dist_logits_pos_logger.log_data(critic_info["dist_logits_pos"].item())
            dist_logits_neg_logger.log_data(critic_info["dist_logits_neg"].item())
            dist_logits_logger.log_data(critic_info["dist_logits"].item())

            actor_loss_logger.log_data(actor_loss.item())
            actor_waypoint_q_loss.log_data(actor_info["actor_waypoint_q_loss"].item())
            actor_dist_q_loss.log_data(actor_info["actor_dist_q_loss"].item())
            actor_q_loss_logger.log_data(actor_info["actor_q_loss"].item())
            gcbc_mle_loss_logger.log_data(actor_info["gcbc_mle_loss"].item())
            gcbc_mse_loss_logger.log_data(actor_info["gcbc_mse_loss"].item())
            waypoint_gcbc_mle_loss_logger.log_data(actor_info["waypoint_gcbc_mle_loss"].item())
            waypoint_gcbc_mse_loss_logger.log_data(actor_info["waypoint_gcbc_mse_loss"].item())
            dist_gcbc_mle_loss_logger.log_data(actor_info["dist_gcbc_mle_loss"].item())
            dist_gcbc_mse_loss_logger.log_data(actor_info["dist_gcbc_mse_loss"].item())

            action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            multi_action_waypts_cos_sim_logger.log_data(multi_action_waypts_cos_sim.item())

            if i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            if i % image_log_freq == 0:
                visualize_dist_pred(
                    to_numpy(dist_obs_image),
                    to_numpy(dist_goal_image),
                    to_numpy(dist_pred),
                    to_numpy(dist_label),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    use_wandb=use_wandb,
                )
                # visualize_traj_pred(
                #     to_numpy(obs_image),
                #     to_numpy(goal_image),
                #     to_numpy(dataset_index),
                #     to_numpy(goal_pos),
                #     to_numpy(action_pred),
                #     to_numpy(action_label),
                #     eval_type,
                #     normalized,
                #     project_folder,
                #     epoch,
                #     num_images_log,
                #     use_wandb=use_wandb,
                # )

                model.cpu()
                waypoint_obs_repr, _, waypoint_g_repr, _ = model.q_network(
                    waypoint_oracle_obs_data, waypoint_oracle_obs_data,
                    waypoint_oracle.flatten(start_dim=0, end_dim=1).flatten(1), dummy_dist,
                    waypoint_oracle_goal_data, waypoint_oracle_goal_data)
                waypoint_oracle_logit = torch.einsum('ikl,jkl->ijl', waypoint_obs_repr, waypoint_g_repr)
                waypoint_oracle_logit = torch.diag(torch.mean(waypoint_oracle_logit, dim=-1)).reshape(
                    waypoint_oracle.shape[0], waypoint_oracle.shape[1], 1)
                waypoint_oracle_critic = torch.sigmoid(waypoint_oracle_logit)
                model = model.to(device)
                visualize_critic_pred(
                    to_numpy(waypoint_obs_image),
                    to_numpy(waypoint_goal_image),
                    to_numpy(waypoint_dataset_index),
                    to_numpy(waypoint_goal_pos),
                    to_numpy(waypoint_oracle),
                    to_numpy(waypoint_oracle_critic),
                    to_numpy(waypoint_pred),
                    to_numpy(waypoint_label),
                    eval_type,
                    normalized,
                    project_folder,
                    epoch,
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
    return critic_loss_logger.average(), actor_loss_logger.average()


def pairwise_acc(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    save_folder: str,
    epoch: int,
    eval_type: str,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    display: bool = False,
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
            ) = vals
            batch_size = transf_obs_image.shape[0]
            transf_obs_image = transf_obs_image.to(device)
            transf_close_image = transf_close_image.to(device)
            transf_far_image = transf_far_image.to(device)

            dummy_action = torch.zeros([batch_size, model.action_size], device=transf_obs_image.device)
            close_dist_pred = model(transf_obs_image, transf_obs_image,
                                    dummy_action[:, :model.action_size - 1], dummy_action[:, [-1]],
                                    transf_close_image, transf_close_image)[-3]

            far_dist_pred = model(transf_obs_image, transf_obs_image,
                                  dummy_action[:, :model.action_size - 1], dummy_action[:, [-1]],
                                  transf_far_image, transf_far_image)[-3]

            close_pred_flat = close_dist_pred.reshape(close_dist_pred.shape[0])
            far_pred_flat = far_dist_pred.reshape(far_dist_pred.shape[0])

            close_pred_flat = to_numpy(close_pred_flat)
            far_pred_flat = to_numpy(far_pred_flat)

            correct = np.where(far_pred_flat > close_pred_flat, 1, 0)
            correct_list.append(correct.copy())
            correct[batch_size // 2:] = np.logical_not(correct[batch_size // 2:]).astype(np.int)

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

            if i % image_log_freq == 0:
                visualize_dist_pairwise_pred(
                    to_numpy(obs_image),
                    to_numpy(close_image),
                    to_numpy(far_image),
                    to_numpy(close_dist_pred),
                    to_numpy(far_dist_pred),
                    to_numpy(close_dist_label),
                    to_numpy(far_dist_label),
                    eval_type,
                    save_folder,
                    epoch,
                    num_images_log,
                    use_wandb,
                    display,
                )
        if len(correct_list) == 0:
            return 0
        return np.concatenate(correct_list).mean(), np.asarray(auc_list).mean()


def load_model(model, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    loaded_model = checkpoint["model"]
    try:  # for DataParallel
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict)
    except (RuntimeError, AttributeError) as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict)
