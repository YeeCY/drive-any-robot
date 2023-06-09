import wandb
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict

# from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
# from stable_contrastive_rl_train.visualizing.critic_utils import visualize_critic_pred
from stable_contrastive_rl_train.evaluation.visualization_utils import visualize_critic_pred
from stable_contrastive_rl_train.training.train_utils import (
    get_critic_loss,
    get_actor_loss,
)
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.evaluation.visualization_utils import visualize_dist_pairwise_pred
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
    waypoint_gcbc_loss_scale: float = 1.0,
    learn_angle: bool = True,
    use_wandb: bool = True,
    pairwise_pred_use_critic: bool = False,
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
        # eval_critic_losses, eval_actor_losses = [], []
        eval_critic_losses, eval_actor_losses = [], []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} Stable Contrastive RL Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            dist_loader = test_dataloaders[dataset_type]["distance"]
            action_loader = test_dataloaders[dataset_type]["action"]
            test_critic_loss, test_actor_loss = evaluate(
                dataset_type,
                model,
                dist_loader,
                action_loader,
                device,
                project_folder,
                normalized,
                epoch,
                discount,
                use_td,
                bc_coef,
                mle_gcbc_loss,
                use_actor_waypoint_q_loss,
                use_actor_dist_q_loss,
                waypoint_gcbc_loss_scale,
                learn_angle,
                print_log_freq,
                image_log_freq,
                num_images_log,
                use_wandb,
            )

            # total_eval_loss = get_total_loss(test_dist_loss, test_action_loss, alpha)
            # eval_total_losses.append(total_eval_loss)
            eval_critic_losses.append(test_critic_loss)
            eval_actor_losses.append(test_actor_loss)
            # wandb.log({f"{dataset_type}_total_loss": total_eval_loss})
            # print(f"{dataset_type}_total_loss: {total_eval_loss}")

            if use_wandb:
                wandb.log({f"{dataset_type}_critic_loss": test_critic_loss})
                wandb.log({f"{dataset_type}_actor_loss": test_actor_loss})

            print(f"{dataset_type}_critic_loss: {test_critic_loss}")
            print(f"{dataset_type}_actor_loss: {test_actor_loss}")

        # FIXME (chongyi)
        # print(f"Start Pairwise Testing Epoch {epoch}/{current_epoch + epochs - 1}")
        # for dataset_type in test_dataloaders:
        #     if "pairwise" in test_dataloaders[dataset_type]:
        #         pairwise_dist_loader = test_dataloaders[dataset_type]["pairwise"]
        #         pairwise_accuracy, pairwise_auc = pairwise_acc(
        #             model,
        #             pairwise_dist_loader,
        #             device,
        #             project_folder,
        #             epoch,
        #             dataset_type,
        #             print_log_freq,
        #             image_log_freq,
        #             num_images_log,
        #             use_wandb=use_wandb,
        #             use_critic=pairwise_pred_use_critic,
        #         )
        #
        #         if use_wandb:
        #             wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
        #             wandb.log({f"{dataset_type}_pairwise_auc": pairwise_auc})
        #
        #         print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
        #         print(f"{dataset_type}_pairwise_auc: {pairwise_auc}")
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
    discount: float = 0.99,
    use_td: bool = True,
    bc_coef: float = 0.05,
    mle_gcbc_loss: bool = False,
    use_actor_waypoint_q_loss: bool = False,
    use_actor_dist_q_loss: bool = False,
    waypoint_gcbc_loss_scale: float = 1.0,
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

    # actor loggers
    actor_loss_logger = Logger("actor_loss", eval_type, window_size=print_log_freq)
    waypoint_actor_q_loss_logger = Logger("actor/waypoint_actor_q_loss", eval_type, window_size=print_log_freq)
    waypoint_gcbc_loss_logger = Logger("actor/waypoint_gcbc_loss_logger", eval_type, window_size=print_log_freq)
    waypoint_gcbc_mle_loss_logger = Logger("actor/waypoint_gcbc_mle_loss", eval_type, window_size=print_log_freq)
    waypoint_gcbc_mse_loss_logger = Logger("actor/waypoint_gcbc_mse_loss", eval_type, window_size=print_log_freq)

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
        waypoint_actor_q_loss_logger,
        waypoint_gcbc_loss_logger,
        waypoint_gcbc_mle_loss_logger,
        waypoint_gcbc_mse_loss_logger,
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
                dist_index_to_data,
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
                waypoint_curr_pos,
                waypoint_yaw,
                waypoint_dataset_index,
                waypoint_index_to_data,
            ) = waypoint_vals
            dist_obs_data = dist_trans_obs_image.to(device)
            dist_next_obs_data = dist_trans_next_obs_image.to(device)
            dist_goal_data = dist_trans_goal_image.to(device)
            dist_label = dist_label.to(device)

            waypoint_obs_data = waypoint_trans_obs_image.to(device)
            waypoint_next_obs_data = waypoint_trans_next_obs_image.to(device)
            waypoint_goal_data = waypoint_trans_goal_image.to(device)
            waypoint_label = waypoint_label.to(device)
            waypoint_oracle = waypoint_oracle.to(device)
            # waypoint_curr_pos = waypoint_curr_pos.to(device)
            # waypoint_yaw = waypoint_yaw.to(device)

            obs_data = (waypoint_obs_data, dist_obs_data)
            next_obs_data = (waypoint_next_obs_data, dist_next_obs_data)
            action_data = (waypoint_label.flatten(1), dist_label)
            goal_data = (waypoint_goal_data, dist_goal_data)

            # waypoint_oracle_obs_data = waypoint_obs_data[:, None].repeat_interleave(
            #     waypoint_oracle.shape[1], dim=1)
            # waypoint_oracle_goal_data = waypoint_goal_data[:, None].repeat_interleave(
            #     waypoint_oracle.shape[1], dim=1)

            critic_loss, critic_info = get_critic_loss(
                model, obs_data, next_obs_data, action_data, goal_data,
                discount, use_td=use_td)

            actor_loss, actor_info = get_actor_loss(
                model, obs_data, action_data, goal_data,
                bc_coef=bc_coef, mle_gcbc_loss=mle_gcbc_loss,
                use_actor_waypoint_q_loss=use_actor_waypoint_q_loss,
                use_actor_dist_q_loss=use_actor_dist_q_loss,
                waypoint_gcbc_loss_scale=waypoint_gcbc_loss_scale)

            waypoint_pred = model(
                waypoint_obs_data, waypoint_label.flatten(1), waypoint_goal_data)[-2]
            waypoint_pred = waypoint_pred.reshape(waypoint_label.shape)
            waypoint_pred_obs_repr, waypoint_pred_g_repr = model(
                waypoint_obs_data, waypoint_pred.flatten(1), waypoint_goal_data)[:2]
            waypoint_pred_logit = torch.einsum(
                'ikl,jkl->ijl', waypoint_pred_obs_repr, waypoint_pred_g_repr)
            waypoint_pred_logit = torch.diag(torch.mean(waypoint_pred_logit, dim=-1))
            waypoint_pred_critic = torch.sigmoid(waypoint_pred_logit)[:, None]

            waypoint_label_obs_repr, waypoint_label_g_repr = model(
                waypoint_obs_data, waypoint_label.flatten(1), waypoint_goal_data)[:2]
            waypoint_label_logits = torch.einsum(
                'ikl,jkl->ijl', waypoint_label_obs_repr, waypoint_label_g_repr)
            waypoint_label_logit = torch.diag(torch.mean(waypoint_label_logits, dim=-1))
            waypoint_label_critic = torch.sigmoid(waypoint_label_logit)[:, None]

            del waypoint_pred_logit
            del waypoint_pred_obs_repr
            del waypoint_pred_g_repr
            del waypoint_label_logit
            del waypoint_label_obs_repr
            del waypoint_label_g_repr
            torch.cuda.empty_cache()

            # (chongyiz): Since we are using DataParallel, we have to use the same batch size
            #   as training to make sure outputs from the networks are consistent (using for loop).
            #   Otherwise, the critic predictions are not correct.
            waypoint_oracle_critic = []
            for idx in range(waypoint_oracle.shape[1]):
                waypoint_oracle_obs_repr, waypoint_oracle_g_repr = model(
                    waypoint_obs_data, waypoint_oracle[:, idx].flatten(1), waypoint_goal_data)[:2]
                waypoint_oracle_logit = torch.einsum(
                    'ikl,jkl->ijl', waypoint_oracle_obs_repr, waypoint_oracle_g_repr)
                waypoint_oracle_logit = torch.diag(torch.mean(waypoint_oracle_logit, dim=-1))
                waypoint_oracle_critic.append(torch.sigmoid(waypoint_oracle_logit)[:, None])

                del waypoint_oracle_logit
                del waypoint_oracle_obs_repr
                del waypoint_oracle_g_repr
                torch.cuda.empty_cache()
            waypoint_oracle_critic = torch.stack(waypoint_oracle_critic, dim=1)

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

            critic_loss_logger.log_data(critic_loss.item())
            waypoint_binary_acc_logger.log_data(critic_info["waypoint_binary_accuracy"].item())
            waypoint_categorical_acc_logger.log_data(critic_info["waypoint_categorical_accuracy"].item())
            waypoint_logits_pos_logger.log_data(critic_info["waypoint_logits_pos"].item())
            waypoint_logits_neg_logger.log_data(critic_info["waypoint_logits_neg"].item())
            waypoint_logits_logger.log_data(critic_info["waypoint_logits"].item())

            actor_loss_logger.log_data(actor_loss.item())
            waypoint_actor_q_loss_logger.log_data(actor_info["waypoint_actor_q_loss"].item())
            waypoint_gcbc_loss_logger.log_data(actor_info["waypoint_gcbc_loss"].item())
            waypoint_gcbc_mle_loss_logger.log_data(actor_info["waypoint_gcbc_mle_loss"].item())
            waypoint_gcbc_mse_loss_logger.log_data(actor_info["waypoint_gcbc_mse_loss"].item())

            action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            multi_action_waypts_cos_sim_logger.log_data(multi_action_waypts_cos_sim.item())

            # release GPU memory
            del dist_obs_data
            del dist_next_obs_data
            del dist_goal_data

            del waypoint_obs_data
            del waypoint_next_obs_data
            del waypoint_goal_data

            del critic_loss
            del critic_info
            del actor_loss
            del actor_info

            del action_waypts_cos_sim
            del multi_action_waypts_cos_sim
            if learn_angle:
                del action_orien_cos_sim
                del multi_action_orien_cos_sim

            torch.cuda.empty_cache()

            if i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            if i % image_log_freq == 0:
                # visualize_dist_pred(
                #     to_numpy(dist_obs_image),
                #     to_numpy(dist_goal_image),
                #     to_numpy(dist_pred),
                #     to_numpy(dist_label),
                #     eval_type,
                #     project_folder,
                #     epoch,
                #     num_images_log,
                #     use_wandb=use_wandb,
                # )

                visualize_critic_pred(
                    to_numpy(waypoint_obs_image),
                    to_numpy(waypoint_goal_image),
                    to_numpy(waypoint_dataset_index),
                    to_numpy(waypoint_goal_pos),
                    to_numpy(waypoint_oracle),
                    to_numpy(waypoint_oracle_critic),
                    to_numpy(waypoint_pred),
                    to_numpy(waypoint_pred_critic),
                    to_numpy(waypoint_label),
                    to_numpy(waypoint_label_critic),
                    eval_type,
                    normalized,
                    project_folder,
                    epoch,
                    waypoint_index_to_data,
                    num_images_log,
                    use_wandb=use_wandb,
                )

            del dist_obs_image
            del dist_goal_image
            del dist_label

            del waypoint_obs_image
            del waypoint_goal_image
            del waypoint_dataset_index
            del waypoint_goal_pos
            del waypoint_oracle
            del waypoint_oracle_critic
            del waypoint_pred
            del waypoint_pred_critic
            del waypoint_label
            del waypoint_label_critic
            torch.cuda.empty_cache()
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
    use_critic: bool = False,
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
            try:
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
            except:
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
            close_dist_label = close_dist_label.to(device)
            far_dist_label = far_dist_label.to(device)

            dummy_action = torch.zeros([batch_size, model.action_size], device=transf_obs_image.device)
            if use_critic:
                # (b)
                close_dist_obs_repr, _, close_dist_g_repr = model(transf_obs_image, transf_obs_image,
                                                                  dummy_action[:, :model.action_size - 1], close_dist_label,
                                                                  transf_close_image[:, -3:], transf_close_image[:, -3:])[1:4]
                close_dist_logit = torch.einsum('ikl,jkl->ijl', close_dist_obs_repr, close_dist_g_repr)
                close_dist_logit = torch.diag(torch.mean(close_dist_logit, dim=-1))
                close_dist_pred = torch.sigmoid(close_dist_logit)

                far_dist_obs_repr, _, far_dist_g_repr = model(transf_obs_image, transf_obs_image,
                                                              dummy_action[:, :model.action_size - 1], far_dist_label,
                                                              transf_far_image, transf_far_image)[1:4]
                far_dist_logit = torch.einsum('ikl,jkl->ijl', far_dist_obs_repr, far_dist_g_repr)
                far_dist_logit = torch.diag(torch.mean(far_dist_logit, dim=-1))
                far_dist_pred = torch.sigmoid(far_dist_logit)

                close_dist_flat = to_numpy(close_dist_pred)
                far_dist_flat = to_numpy(far_dist_pred)

                correct = np.where(close_dist_flat >= far_dist_flat, 1, 0)
                correct_list.append(correct.copy())
                # correct[batch_size // 2:] = np.logical_not(correct[batch_size // 2:]).astype(np.int)

                auc = roc_auc_score(
                    np.concatenate([np.ones_like(correct[:batch_size // 2]),
                                    np.zeros_like(correct[batch_size // 2:])]),
                    np.concatenate([(close_dist_flat - far_dist_flat)[:batch_size // 2],
                                    (far_dist_flat - close_dist_flat)[batch_size // 2:]])
                )
                auc_list.append(auc)

                # (c)
                # close_dist_repr, _, far_dist_repr = model(transf_close_image, transf_close_image,
                #                                           dummy_action[:, :model.action_size - 1], far_dist_label - close_dist_label,
                #                                           transf_far_image, transf_far_image)[1:4]
                # dist_logit = torch.einsum('ikl,jkl->ijl', close_dist_repr, far_dist_repr)
                # dist_logit = torch.diag(torch.mean(dist_logit, dim=-1))
                # dist_pred = torch.sigmoid(dist_logit)
                #
                # dist_pred = to_numpy(dist_pred)
                #
                # correct = np.where(dist_pred >= 0.5, 1, 0)
                # correct_list.append(correct.copy())
                #
                # auc = roc_auc_score(
                #     np.concatenate([np.ones_like(correct[:batch_size // 2]),
                #                     np.zeros_like(correct[batch_size // 2:])]),
                #     np.concatenate([dist_pred[:batch_size // 2],
                #                     (1 - dist_pred)[batch_size // 2:]])
                # )
                # auc_list.append(auc)
            else:
                close_dist_pred = model(transf_obs_image, transf_obs_image,
                                        dummy_action[:, :model.action_size - 1], dummy_action[:, [-1]],
                                        transf_close_image[:, -3:], transf_close_image[:, -3:])[-3]
    
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
                    index_to_data,
                    num_images_log,
                    use_wandb,
                    display,
                )
        if len(correct_list) == 0:
            return 0
        return np.concatenate(correct_list).mean(), np.asarray(auc_list).mean()
