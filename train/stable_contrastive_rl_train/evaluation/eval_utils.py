import wandb
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict
import pickle as pkl

# from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
# from stable_contrastive_rl_train.visualizing.critic_utils import visualize_critic_pred
from stable_contrastive_rl_train.evaluation.visualization_utils import (
    visualize_critic_pred,
    visualize_traj_dist_pred,
)
from stable_contrastive_rl_train.training.train_utils import (
    get_critic_loss,
    get_actor_loss,
)
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.evaluation.visualization_utils import (
    # numpy_to_img,
    visualize_dist_pairwise_pred
)
from gnm_train.training.logger import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.optim import Optimizer
# from torch.distributions import (
#     Normal,
#     Independent
# )


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
    save_pairwise_dist_pred_freq: int = 5,
    save_traj_dist_pred_freq: int = 5,
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
    pairwise_dist_pred_eval_mode: str = "close_logit_diff",
    traj_dist_pred_eval_mode: str = "logit_sum",
    eval_waypoint: bool = True,
    eval_pairwise_dist_pred: bool = True,
    eval_traj_dist_pred: bool = True,
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
        if eval_waypoint:
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

        if eval_pairwise_dist_pred:
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
                        save_pairwise_dist_pred_freq,
                        eval_mode=pairwise_dist_pred_eval_mode,
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
                        eval_mode=traj_dist_pred_eval_mode,
                    )

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

    # use classifier_to_generative_model data to debug
    # from classifier_to_generative_model.utils import load_and_construct_dataset
    #
    # train_traj_dataset, train_dataset = load_and_construct_dataset(
    #     "/projects/rsalakhugroup/chongyiz/classifier_to_generative_model/datasets/fourrooms.pkl"
    # )

    # TODO (chongyiz): there are some bugs here, tried to fix them.
    # Fixed batch of data
    # batch_idxs = np.random.choice(len(train_dataset["state"]),
    #                               size=32, replace=False)
    # batch_idxs = np.arange(32)
    # batch = {k: torch.tensor(v[batch_idxs], device=device) for k, v in train_dataset.items()}

    with torch.no_grad():
        for i, val in enumerate(zip(eval_dist_loader, eval_action_loader)):
            try:
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
                    dist_data_info,
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
                    waypoint_data_info,
                ) = waypoint_vals
            except:
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
                    dist_data_info,
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
                    waypoint_data_info,
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

            # waypoint_f_curr, waypoint_obs_time, waypoint_f_goal, waypoint_g_time = waypoint_data_info
            # f_mask = torch.Tensor(np.array(waypoint_f_curr)[:, None] == np.array(waypoint_f_goal)[None])
            # time_mask = waypoint_obs_time[:, None] < waypoint_g_time[None]
            # mc_bce_labels = (f_mask * time_mask).to(device)
            # mc_bce_labels = torch.eye(mc_bce_labels.shape[0], device=device)
            critic_loss, critic_info = get_critic_loss(
                model, waypoint_obs_data, waypoint_next_obs_data,
                waypoint_label.flatten(1), waypoint_goal_data,
                waypoint_data_info, discount, use_td=use_td)

            # batch_idxs = np.random.choice(len(train_dataset["state"]),
            #                               size=512, replace=False)
            # batch = {k: torch.tensor(v[batch_idxs], device=device) for k, v in train_dataset.items()}

            # critic_loss, critic_info = get_critic_loss(
            #     model, batch["state"], batch["next_state"],
            #     batch["action"], batch["next_state"],
            #     discount, mc_bce_labels, use_td=use_td)

            actor_loss, actor_info = get_actor_loss(
                model, waypoint_obs_data,
                waypoint_label.reshape([waypoint_label.shape[0], -1]),
                waypoint_goal_data,
                bc_coef=bc_coef, mle_gcbc_loss=mle_gcbc_loss,
                use_actor_waypoint_q_loss=use_actor_waypoint_q_loss,
                use_actor_dist_q_loss=use_actor_dist_q_loss,
                waypoint_gcbc_loss_scale=waypoint_gcbc_loss_scale)

            # waypoint_pred = model(
            #     waypoint_obs_data, waypoint_label.flatten(1), waypoint_goal_data)[-2]
            # waypoint_pred = waypoint_pred.reshape(waypoint_label.shape)
            # waypoint_pred_obs_repr, waypoint_pred_g_repr = model(
            #     waypoint_obs_data, waypoint_pred.flatten(1), waypoint_goal_data)[:2]
            # waypoint_pred_logit = torch.einsum(
            #     'ikl,jkl->ijl', waypoint_pred_obs_repr, waypoint_pred_g_repr)
            # waypoint_pred_logit = torch.diag(torch.mean(waypoint_pred_logit, dim=-1))
            # waypoint_pred_critic = torch.sigmoid(waypoint_pred_logit)[:, None]
            #
            # waypoint_label_obs_repr, waypoint_label_g_repr = model(
            #     waypoint_obs_data, waypoint_label.flatten(1), waypoint_goal_data)[:2]
            # waypoint_label_logits = torch.einsum(
            #     'ikl,jkl->ijl', waypoint_label_obs_repr, waypoint_label_g_repr)
            # waypoint_label_logit = torch.diag(torch.mean(waypoint_label_logits, dim=-1))
            # waypoint_label_critic = torch.sigmoid(waypoint_label_logit)[:, None]

            # del waypoint_pred_logit
            # del waypoint_pred_obs_repr
            # del waypoint_pred_g_repr
            # del waypoint_label_logit
            # del waypoint_label_obs_repr
            # del waypoint_label_g_repr
            # torch.cuda.empty_cache()

            # (chongyiz): Since we are using DataParallel, we have to use the same batch size
            #   as training to make sure outputs from the networks are consistent (using for loop).
            #   Otherwise, the critic predictions are not correct.
            # waypoint_oracle_critic = []
            # for idx in range(waypoint_oracle.shape[1]):
            #     waypoint_oracle_obs_repr, waypoint_oracle_g_repr = model(
            #         waypoint_obs_data, waypoint_oracle[:, idx].flatten(1), waypoint_goal_data)[:2]
            #     waypoint_oracle_logit = torch.einsum(
            #         'ikl,jkl->ijl', waypoint_oracle_obs_repr, waypoint_oracle_g_repr)
            #     waypoint_oracle_logit = torch.diag(torch.mean(waypoint_oracle_logit, dim=-1))
            #     waypoint_oracle_critic.append(torch.sigmoid(waypoint_oracle_logit)[:, None])
            #
            #     del waypoint_oracle_logit
            #     del waypoint_oracle_obs_repr
            #     del waypoint_oracle_g_repr
            #     torch.cuda.empty_cache()
            # waypoint_oracle_critic = torch.stack(waypoint_oracle_critic, dim=1)
            #
            # action_waypts_cos_sim = F.cosine_similarity(
            #     waypoint_pred[:2], waypoint_label[:2], dim=-1
            # ).mean()
            # multi_action_waypts_cos_sim = F.cosine_similarity(
            #     torch.flatten(waypoint_pred[:2], start_dim=1),
            #     torch.flatten(waypoint_pred[:2], start_dim=1),
            #     dim=-1,
            # ).mean()
            # if learn_angle:
            #     action_orien_cos_sim = F.cosine_similarity(
            #         waypoint_pred[2:], waypoint_label[2:], dim=-1
            #     ).mean()
            #     multi_action_orien_cos_sim = F.cosine_similarity(
            #         torch.flatten(waypoint_pred[2:], start_dim=1),
            #         torch.flatten(waypoint_label[2:], start_dim=1),
            #         dim=-1,
            #     ).mean()
            #     action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
            #     multi_action_orien_cos_sim_logger.log_data(
            #         multi_action_orien_cos_sim.item()
            #     )

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

            # action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            # multi_action_waypts_cos_sim_logger.log_data(multi_action_waypts_cos_sim.item())

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

            # del action_waypts_cos_sim
            # del multi_action_waypts_cos_sim
            # if learn_angle:
            #     del action_orien_cos_sim
            #     del multi_action_orien_cos_sim

            torch.cuda.empty_cache()

            if i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            # if i % image_log_freq == 0:
            #     # visualize_dist_pred(
            #     #     to_numpy(dist_obs_image),
            #     #     to_numpy(dist_goal_image),
            #     #     to_numpy(dist_pred),
            #     #     to_numpy(dist_label),
            #     #     eval_type,
            #     #     project_folder,
            #     #     epoch,
            #     #     num_images_log,
            #     #     use_wandb=use_wandb,
            #     # )
            #
            #     visualize_critic_pred(
            #         to_numpy(waypoint_obs_image),
            #         to_numpy(waypoint_goal_image),
            #         to_numpy(waypoint_dataset_index),
            #         to_numpy(waypoint_goal_pos),
            #         to_numpy(waypoint_oracle),
            #         to_numpy(waypoint_oracle_critic),
            #         to_numpy(waypoint_pred),
            #         to_numpy(waypoint_pred_critic),
            #         to_numpy(waypoint_label),
            #         to_numpy(waypoint_label_critic),
            #         eval_type,
            #         normalized,
            #         project_folder,
            #         epoch,
            #         waypoint_index_to_data,
            #         num_images_log,
            #         use_wandb=use_wandb,
            #     )
            #
            # del dist_obs_image
            # del dist_goal_image
            # del dist_label
            #
            # del waypoint_obs_image
            # del waypoint_goal_image
            # del waypoint_dataset_index
            # del waypoint_goal_pos
            # del waypoint_oracle
            # del waypoint_oracle_critic
            # del waypoint_pred
            # del waypoint_pred_critic
            # del waypoint_label
            # del waypoint_label_critic
            # torch.cuda.empty_cache()
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
    save_result_freq: int = 1000,
    eval_mode: str = "close_logit_diff",
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
            transf_obs_image = transf_obs_image[:, -(model.context_size + 1) * 3:].to(device)
            transf_close_image = transf_close_image.to(device)
            transf_far_image = transf_far_image.to(device)
            close_dist_label = close_dist_label.to(device)
            far_dist_label = far_dist_label.to(device)

            dummy_action = torch.zeros(
                [batch_size, model.len_trajectory_pred, model.num_action_params],
                device=transf_obs_image.device
            )
            dummy_action[..., 2] = 1  # cos of yaws
            dummy_action = dummy_action.reshape([batch_size, model.action_size])
            if eval_mode == "close_far_mi_diff":
                obs_a_repr, close_g_repr = model(
                    transf_obs_image, dummy_action, transf_close_image[:, -3:])[0:2]
                obs_a_close_g_logit = torch.einsum('ikl,jkl->ijl', obs_a_repr, close_g_repr)
                obs_a_close_g_logit = torch.diag(torch.mean(obs_a_close_g_logit, dim=-1))

                far_g_a_repr, close_g_repr = model(
                    transf_far_image, dummy_action, transf_close_image[:, -3:])[0:2]
                far_g_a_close_g_logit = torch.einsum('ikl,jkl->ijl', far_g_a_repr, close_g_repr)
                far_g_a_close_g_logit = torch.diag(torch.mean(far_g_a_close_g_logit, dim=-1))

                obs_a_repr, far_g_repr = model(
                    transf_obs_image, dummy_action, transf_far_image[:, -3:])[0:2]
                obs_a_far_g_logit = torch.einsum('ikl,jkl->ijl', obs_a_repr, far_g_repr)
                obs_a_far_g_logit = torch.diag(torch.mean(obs_a_far_g_logit, dim=-1))

                close_g_a_repr, far_g_repr = model(
                    transf_close_image, dummy_action, transf_far_image[:, -3:])[0:2]
                close_g_a_far_g_logit = torch.einsum('ikl,jkl->ijl', close_g_a_repr, far_g_repr)
                close_g_a_far_g_logit = torch.diag(torch.mean(close_g_a_far_g_logit, dim=-1))

                close_g_logit_diff = obs_a_close_g_logit - far_g_a_close_g_logit
                far_g_logit_diff = obs_a_far_g_logit - close_g_a_far_g_logit

                close_dist_pred = close_g_logit_diff.clone()
                far_dist_pred = far_g_logit_diff.clone()

                close_g_logit_diff = to_numpy(close_g_logit_diff)
                far_g_logit_diff = to_numpy(far_g_logit_diff)

                correct = np.where(close_g_logit_diff > far_g_logit_diff, 1, 0)
                correct_list.append(correct.copy())

                auc = roc_auc_score(
                    np.concatenate([np.ones_like(correct[:batch_size // 2]),
                                    np.zeros_like(correct[batch_size // 2:])]),
                    np.concatenate([(close_g_logit_diff - far_g_logit_diff)[:batch_size // 2],
                                    (far_g_logit_diff - close_g_logit_diff)[batch_size // 2:]])
                )
                auc_list.append(auc)
            elif eval_mode == "close_logit_diff":
                obs_a_repr, close_g_repr = model(
                    transf_obs_image, dummy_action, transf_close_image[:, -3:])[0:2]
                obs_a_close_g_logit = torch.einsum('ikl,jkl->ijl', obs_a_repr, close_g_repr)
                obs_a_close_g_logit = torch.diag(torch.mean(obs_a_close_g_logit, dim=-1))

                far_g_a_repr, close_g_repr = model(
                    transf_far_image, dummy_action, transf_close_image[:, -3:])[0:2]
                far_g_a_close_g_logit = torch.einsum('ikl,jkl->ijl', far_g_a_repr, close_g_repr)
                far_g_a_close_g_logit = torch.diag(torch.mean(far_g_a_close_g_logit, dim=-1))

                close_dist_pred = obs_a_close_g_logit.clone()
                far_dist_pred = far_g_a_close_g_logit.clone()

                obs_a_close_g_logit = to_numpy(obs_a_close_g_logit)
                far_g_a_close_g_logit = to_numpy(far_g_a_close_g_logit)

                correct = np.where(obs_a_close_g_logit > far_g_a_close_g_logit, 1, 0)
                correct_list.append(correct.copy())

                auc = roc_auc_score(
                    np.concatenate([np.ones_like(correct[:batch_size // 2]),
                                    np.zeros_like(correct[batch_size // 2:])]),
                    np.concatenate([(obs_a_close_g_logit - far_g_a_close_g_logit)[:batch_size // 2],
                                    (far_g_a_close_g_logit - obs_a_close_g_logit)[batch_size // 2:]])
                )
                auc_list.append(auc)
            elif eval_mode == "far_logit_diff":
                close_g_a_repr, far_g_repr = model(
                    transf_close_image, dummy_action, transf_far_image[:, -3:])[0:2]
                close_g_a_far_g_logit = torch.einsum('ikl,jkl->ijl', close_g_a_repr, far_g_repr)
                close_g_a_far_g_logit = torch.diag(torch.mean(close_g_a_far_g_logit, dim=-1))

                obs_a_repr, far_g_repr = model(
                    transf_obs_image, dummy_action, transf_far_image[:, -3:])[0:2]
                obs_a_far_g_logit = torch.einsum('ikl,jkl->ijl', obs_a_repr, far_g_repr)
                obs_a_far_g_logit = torch.diag(torch.mean(obs_a_far_g_logit, dim=-1))

                close_dist_pred = close_g_a_far_g_logit.clone()
                far_dist_pred = obs_a_far_g_logit.clone()

                close_g_a_far_g_logit = to_numpy(close_g_a_far_g_logit)
                obs_a_far_g_logit = to_numpy(obs_a_far_g_logit)

                correct = np.where(close_g_a_far_g_logit > obs_a_far_g_logit, 1, 0)
                correct_list.append(correct.copy())

                auc = roc_auc_score(
                    np.concatenate([np.ones_like(correct[:batch_size // 2]),
                                    np.zeros_like(correct[batch_size // 2:])]),
                    np.concatenate([(close_g_a_far_g_logit - obs_a_far_g_logit)[:batch_size // 2],
                                    (obs_a_far_g_logit - close_g_a_far_g_logit)[batch_size // 2:]])
                )
                auc_list.append(auc)
            else:
                raise RuntimeError("Unknown evaluation mode: {}".format(eval_mode))

            if i % print_log_freq == 0:
                print(f"({i}/{num_batches}) batch of points processed")

            if i % save_result_freq == 0:
                save_dist_pairwise_pred(
                    to_numpy(close_dist_pred),
                    to_numpy(far_dist_pred),
                    to_numpy(close_dist_label),
                    to_numpy(far_dist_label),
                    eval_type,
                    save_folder,
                    epoch,
                    index_to_data,
                )
        if len(correct_list) == 0:
            return 0
        return np.concatenate(correct_list).mean(), np.asarray(auc_list).mean()


def save_dist_pairwise_pred(
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    index_to_data: dict,
    rounding: int = 4,
):
    save_dir = os.path.join(
        save_folder,
        "result",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    os.makedirs(save_dir, exist_ok=True)
    assert (
        len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )
    batch_size = batch_close_preds.shape[0]
    result_save_path = os.path.join(save_dir, f"results.pkl")
    batch_results = {}
    for i in range(batch_size):
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)

        result_label = "f_close={}_f_far={}_curr_time={}_close_time={}_far_time={}_context_times={}".format(
            index_to_data["f_close"][i],
            index_to_data["f_far"][i],
            int(index_to_data["curr_time"][i]),
            int(index_to_data["close_time"][i]),
            int(index_to_data["far_time"][i]),
            list(index_to_data["context_times"].numpy()[i]),
        )
        result = {
            "f_close": index_to_data["f_close"][i],
            "f_far": index_to_data["f_far"][i],
            "curr_time": int(index_to_data["curr_time"][i]),
            "close_time": int(index_to_data["close_time"][i]),
            "far_time": int(index_to_data["far_time"][i]),
            "context_times": list(index_to_data["context_times"].numpy()[i]),
            "far_dist_pred": far_dist_pred,
            "far_dist_label": far_dist_label,
            "close_dist_pred": close_dist_pred,
            "close_dist_label": close_dist_label,
        }
        batch_results[result_label] = result

    if os.path.exists(result_save_path):
        with open(result_save_path, "rb") as f:
            results = pkl.load(f)
        results.update(batch_results)
        with open(result_save_path, "wb") as f:
            pkl.dump(results, f)
    else:
        with open(result_save_path, "wb+") as f:
            pkl.dump(batch_results, f)


def planning_via_sorting(model, obs_image, goal_image, cand_image, ref_image):
    max_batch_size = max(obs_image.shape[0], goal_image.shape[0],
                         cand_image.shape[0], ref_image.shape[0])
    dummy_action = torch.zeros(
        [max_batch_size, model.len_trajectory_pred, model.num_action_params],
        device=obs_image.device
    )
    # dummy_action[..., 2] = 1  # cos of yaws
    dummy_action = dummy_action.reshape([max_batch_size, model.action_size])
    # dummy_action = torch.zeros(
    #     [traj_len, model.len_trajectory_pred, model.num_action_params],
    #     device=transf_obs_image.device
    # )
    # dummy_action[..., 2] = 1  # cos of yaws
    # dummy_action = dummy_action.reshape([traj_len, model.action_size])
    obs_zero_repr, goal_repr = model(
        obs_image,
        dummy_action[:obs_image.shape[0]],
        goal_image[:, -3:]
    )[:2]
    cand_zero_repr, cand_repr = model(
        cand_image,
        dummy_action[:cand_image.shape[0]],
        cand_image[:, -3:]
    )[:2]
    ref_zero_repr = model(
        ref_image,
        dummy_action[:ref_image.shape[0]],
        ref_image[:, -3:]  # not used
    )[0]

    ref_cand_logits = torch.mean(torch.einsum("ikl,jkl->ijl", ref_zero_repr, cand_repr), dim=-1)
    obs_cand_logits = torch.mean(torch.einsum("ikl,jkl->ijl", obs_zero_repr, cand_repr), dim=-1)
    # ref_cand_logits = []
    # for ref_image_ in ref_image:
    #     ref_logits_ = model.q_network(ref_image_[None].repeat_interleave(cand_image.shape[0], dim=0),
    #                                   dummy_action[:cand_image.shape[0]],
    #                                   cand_image).mean(-1)
    #     ref_cand_logits.append(ref_logits_)
    # ref_cand_logits = torch.stack(ref_cand_logits)
    # obs_cand_logits = []
    # for obs_image_ in obs_image:
    #     obs_logits_ = model.q_network(obs_image_[None].repeat_interleave(cand_image.shape[0], dim=0),
    #                                   dummy_action[:cand_image.shape[0]],
    #                                   cand_image).mean(-1)
    #     obs_cand_logits.append(obs_logits_)
    # obs_cand_logits = torch.stack(obs_cand_logits)

    ref_goal_logits = torch.mean(torch.einsum("ikl,jkl->ijl", ref_zero_repr, goal_repr), dim=-1)
    cand_goal_logits = torch.mean(torch.einsum("ikl,jkl->ijl", cand_zero_repr, goal_repr), dim=-1)
    # ref_goal_logits = model.q_network(ref_image, dummy_action[:ref_image.shape[0]], goal_image)[0].mean(-1)
    # cand_goal_logits = model.q_network(cand_image, dummy_action[:cand_image.shape[0]], goal_image)[0].mean(-1)
    # ref_goal_logits = []
    # for ref_image_ in ref_image:
    #     ref_logits_ = model.q_network(ref_image_[None].repeat_interleave(goal_image.shape[0], dim=0),
    #                                   dummy_action[:goal_image.shape[0]],
    #                                   goal_image).mean(-1)
    #     ref_goal_logits.append(ref_logits_)
    # ref_goal_logits = torch.stack(ref_goal_logits)
    # cand_goal_logits = []
    # for cand_image_ in cand_image:
    #     cand_logits_ = model.q_network(cand_image_[None].repeat_interleave(goal_image.shape[0], dim=0),
    #                                    dummy_action[:goal_image.shape[0]],
    #                                    goal_image).mean(-1)
    #     cand_goal_logits.append(cand_logits_)
    # cand_goal_logits = torch.stack(cand_goal_logits)

    ref_cand_logits = torch.cat([ref_cand_logits, obs_cand_logits], dim=0)
    ref_goal_logits = torch.cat([
        ref_goal_logits.repeat_interleave(cand_image.shape[0], dim=1),
        cand_goal_logits.T], dim=0
    )

    # use the index of f(s, a, g_s) in the sorted reference logit set as distance
    # use transpose here to make sure the index for candidates starts from 0 and ends at len(candidates)
    # (chongyiz): check the following lines
    dist_obs_to_cand = torch.where(
        (torch.argsort(ref_cand_logits, dim=0, descending=True) == len(ref_cand_logits) - 1).T)[1]
    dist_cand_to_goal = torch.where(
        (torch.argsort(ref_goal_logits, dim=0, descending=True) == len(ref_goal_logits) - 1).T)[1]
    subgoal_idx = int(torch.argmin(dist_obs_to_cand + dist_cand_to_goal))

    return subgoal_idx


def planning_on_graph(model, obs_image, goal_image, cand_image):
    max_batch_size = max(obs_image.shape[0], goal_image.shape[0],
                         cand_image.shape[0])
    dummy_action = torch.zeros(
        [max_batch_size, model.len_trajectory_pred, model.num_action_params],
        device=obs_image.device
    )
    dummy_action[..., 2] = 1  # cos of yaws
    dummy_action = dummy_action.reshape([max_batch_size, model.action_size])

    # construct graphs
    cand_zero_repr, cand_repr = model(
        cand_image,
        dummy_action[:cand_image.shape[0]],
        cand_image[:, -3:]
    )[:2]
    # cand_cand_logits = []
    # for cand_image_ in cand_image:
    #     cand_cand_logit = model(
    #         cand_image,
    #         dummy_action[:cand_image.shape[0]],
    #         cand_image_[-3:][None].repeat_interleave(cand_image.shape[0], dim=0)
    #     )[0]
    #     cand_cand_logits.append(cand_cand_logit)
    # cand_cand_logits = torch.mean(torch.stack(cand_cand_logits), dim=-1).T

    num_cands = cand_image.shape[0]
    cand_cand_logits = torch.mean(torch.einsum("ikl,jkl->ijl", cand_zero_repr, cand_repr), dim=-1)

    # use quantile to find connected candidates
    qvals = torch.quantile(
        cand_cand_logits * (1 - torch.eye(num_cands, device=cand_cand_logits.device)), 0.9, dim=0)
    adj_matrix = (cand_cand_logits > qvals[None]).float()

    shifted_logits = cand_cand_logits - cand_cand_logits.min(dim=0)[0]

    # find the index of each value after the array is sorted: https://stackoverflow.com/questions/54459554/numpy-find-index-in-sorted-array-in-an-efficient-way
    index_after_sorting = torch.argsort(torch.argsort(shifted_logits * adj_matrix, descending=True, dim=0), dim=0)

    # set min_dist to 1 (otherwise the dijkstra algorithm will ignore the edge)
    dists = ((index_after_sorting + 1) * adj_matrix).cpu().numpy()

    from scipy.sparse.csgraph import dijkstra
    dist_matrix, predecessors = dijkstra(dists, indices=0, return_predecessors=True)

    idx = -1
    path = []
    while predecessors[idx] != -9999:
        path.append(predecessors[idx])
        idx = predecessors[idx]
    path.reverse()
    if len(path) > 0:
        path.append(len(dists) - 1)

    return adj_matrix, path


def traj_dist_pred(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    save_folder: str,
    epoch: int,
    eval_type: str,
    print_log_freq: int = 100,
    save_result_freq: int = 1000,
    eval_mode: str = "logit_sum",
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
                cand_image,
                transf_cand_image,
                cand_latlong,
                global_cand_pos,
                index_to_traj,
            ) = tuple([v[0] for v in vals[:-1]] + [vals[-1]])
            traj_len = transf_obs_image.shape[0]
            traj_transf_obs_image = transf_obs_image.to(device)
            assert torch.all(
                transf_goal_image[0] == transf_goal_image[np.random.randint(len(transf_goal_image))]
            )
            traj_transf_goal_image = transf_goal_image.to(device)
            transf_goal_image = traj_transf_goal_image.to(device)[0][None]  # all goals are the same
            transf_cand_image = transf_cand_image.to(device)

            # planning with SCRL
            # dummy_action = torch.zeros(
            #     [traj_len, model.len_trajectory_pred, model.num_action_params],
            #     device=transf_obs_image.device
            # )
            # dummy_action[..., 2] = 1  # cos of yaws
            # dummy_action = dummy_action.reshape([traj_len, model.action_size])
            current_obs_idx = 0
            path_obs_idxs = [current_obs_idx]
            assert torch.all(transf_cand_image[current_obs_idx] == traj_transf_obs_image[current_obs_idx])

            # while current_obs_idx != traj_len - 1 and len(path_obs_idxs) < traj_len:
            #     if len(path_obs_idxs) > 3 and len(np.unique(np.array(path_obs_idxs)[-3:])) == 1:
            #         break
            #     # mask = torch.zeros(transf_obs_image.shape[0], dtype=torch.bool, device=device)
            #     # sg_indices = torch.arange(traj_len, dtype=torch.int, device=device)
            #     # mask[current_obs_idx] = True
            #     # transf_curr_obs_image = transf_obs_image[mask]
            #     # transf_sg_image = transf_obs_image[~mask]
            #     # sg_indices = sg_indices[~mask]
            #     # # tmp_curr_obs_a_repr, tmp_sg_repr = model(
            #     # #     transf_curr_obs_image, dummy_action[[0]], transf_curr_obs_image)[0:2]
            #     # # tmp_curr_obs_a_sg_logit = torch.einsum('ikl,jkl->ijl', tmp_curr_obs_a_repr, tmp_sg_repr)
            #     # # tmp_curr_obs_a_sg_logit = torch.diag(torch.mean(tmp_curr_obs_a_sg_logit, dim=-1))
            #     #
            #     # waypoint_pred = model(
            #     #     transf_curr_obs_image.repeat_interleave(transf_sg_image.shape[0], dim=0),
            #     #     dummy_action[:transf_sg_image.shape[0]],
            #     #     transf_sg_image
            #     # )[-2]
            #     # curr_obs_a_repr, sg_repr = model(
            #     #     transf_curr_obs_image.repeat_interleave(transf_sg_image.shape[0], dim=0),
            #     #     waypoint_pred,
            #     #     transf_sg_image
            #     # )[0:2]
            #     # curr_obs_a_sg_logit = torch.einsum('ikl,jkl->ijl', curr_obs_a_repr, sg_repr)
            #     # curr_obs_a_sg_logit = torch.diag(torch.mean(curr_obs_a_sg_logit, dim=-1))
            #     #
            #     # sg_a_repr, g_repr = model(
            #     #     transf_sg_image,
            #     #     dummy_action[:transf_sg_image.shape[0]],
            #     #     transf_goal_image[:transf_sg_image.shape[0]]
            #     # )[0:2]
            #     # sg_a_g_logit = torch.einsum('ikl,jkl->ijl', sg_a_repr, g_repr)
            #     # sg_a_g_logit = torch.diag(torch.mean(sg_a_g_logit, dim=-1))
            #     #
            #     # waypoint_pred = model(
            #     #     transf_curr_obs_image.repeat_interleave(transf_sg_image.shape[0], dim=0),
            #     #     dummy_action[:transf_sg_image.shape[0]],
            #     #     transf_goal_image[:transf_sg_image.shape[0]]
            #     # )[-2]
            #     # curr_obs_a_repr, g_repr = model(
            #     #     transf_curr_obs_image.repeat_interleave(transf_sg_image.shape[0], dim=0),
            #     #     waypoint_pred,
            #     #     transf_goal_image[:transf_sg_image.shape[0]]
            #     # )[0:2]
            #     # curr_obs_a_g_logit = torch.einsum('ikl,jkl->ijl', curr_obs_a_repr, g_repr)
            #     # curr_obs_a_g_logit = torch.diag(torch.mean(curr_obs_a_g_logit, dim=-1))
            #     #
            #     # waypoint_pred = model(
            #     #     transf_goal_image[:transf_sg_image.shape[0]],
            #     #     dummy_action[:transf_sg_image.shape[0]],
            #     #     transf_sg_image
            #     # )[-2]
            #     # g_a_repr, sg_repr = model(
            #     #     transf_goal_image[:transf_sg_image.shape[0]],
            #     #     waypoint_pred,
            #     #     transf_sg_image
            #     # )[0:2]
            #     # g_a_sg_logit = torch.einsum('ikl,jkl->ijl', g_a_repr, sg_repr)
            #     # g_a_sg_logit = torch.diag(torch.mean(g_a_sg_logit, dim=-1))
            #     #
            #     # if eval_mode == "logit_sum":
            #     #     sg_idx = int(sg_indices[torch.argmax(curr_obs_a_sg_logit + sg_a_g_logit)])
            #     # elif eval_mode == "close_logit_diff":
            #     #     sg_idx = int(sg_indices[torch.argmax(curr_obs_a_sg_logit - g_a_sg_logit)])
            #     # elif eval_mode == "far_logit_diff":
            #     #     sg_idx = int(sg_indices[torch.argmax(sg_a_g_logit - curr_obs_a_g_logit)])
            #     # elif eval_mode == "mi_diff":
            #     #     sg_idx = int(sg_indices[torch.argmax((curr_obs_a_sg_logit - g_a_sg_logit) - (curr_obs_a_g_logit - sg_a_g_logit))])
            #
            #     # planning via sorting
            #     transf_obs_image = transf_cand_image[current_obs_idx][None]
            #
            #     # selected_cand_idx = planning_via_sorting(
            #     #     model, transf_obs_image, transf_goal_image,
            #     #     # torch.cat([traj_transf_obs_image, transf_goal_image], dim=0),
            #     #     traj_transf_obs_image[1:],  # (chongyi): Do we need to include the starting state and the goal in the candidates?
            #     #     torch.cat([traj_transf_obs_image, transf_goal_image], dim=0)
            #     # )
            #     # model, obs_image, goal_image, cand_image, ref_image
            #     # selected_cand_idx = planning_via_sorting(
            #     #     model, transf_obs_image, transf_goal_image,
            #     #     # torch.cat([traj_transf_obs_image, transf_goal_image], dim=0),
            #     #     traj_transf_obs_image[1:],
            #     #     # (chongyi): Do we need to include the starting state and the goal in the candidates?
            #     #     traj_transf_obs_image,
            #     # )
            #     # subgoal_idx = planning_via_sorting(
            #     #     model, transf_obs_image, transf_goal_image,
            #     #     transf_cand_image,
            #     #     transf_cand_image,
            #     # )
            #
            #     subgoal_idx = planning_on_graph(
            #         model, transf_obs_image, transf_goal_image,
            #         transf_cand_image,
            #         transf_cand_image,
            #     )
            #
            #     # we used all the states except the starting and the goal states as the candidates.
            #     # subgoal_idx = selected_cand_idx + 1
            #
            #     # assume we can move to the subgoal exactly
            #     path_obs_idxs.append(subgoal_idx)
            #     current_obs_idx = subgoal_idx

            cand_adj_matrix, path_obs_idxs = planning_on_graph(
                model, transf_cand_image[current_obs_idx][None], transf_goal_image,
                torch.cat([transf_cand_image, transf_goal_image], dim=0),
            )

            if i % print_log_freq == 0:
                print(f"({i}/{num_trajs}) trajectories processed")

            if i % save_result_freq == 0:
                save_traj_dist_pred(
                    to_numpy(obs_latlong),
                    to_numpy(goal_latlong),
                    to_numpy(cand_latlong),
                    to_numpy(global_obs_pos),
                    to_numpy(global_goal_pos),
                    to_numpy(global_cand_pos),
                    to_numpy(cand_adj_matrix),
                    np.array(path_obs_idxs),
                    eval_type,
                    save_folder,
                    epoch,
                    index_to_traj,
                )

            # if i % image_log_freq == 0:
            #     visualize_traj_dist_pred(
            #         to_numpy(traj_transf_obs_image),
            #         to_numpy(traj_transf_goal_image),
            #         to_numpy(transf_cand_image),
            #         to_numpy(dataset_index),
            #         to_numpy(local_goal_pos),
            #         to_numpy(waypoint_label),
            #         to_numpy(global_obs_pos),
            #         to_numpy(global_goal_pos),
            #         to_numpy(global_cand_pos),
            #         np.array(path_obs_idxs),
            #         eval_type,
            #         True,
            #         save_folder,
            #         epoch,
            #     )


def save_traj_dist_pred(
    obs_latlong: np.ndarray,
    goal_latlong: np.ndarray,
    cand_latlong: np.ndarray,
    global_obs_pos: np.ndarray,
    global_goal_pos: np.ndarray,
    global_cand_pos: np.ndarray,
    cand_adj_matrix: np.ndarray,
    path_idxs: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    index_to_traj: dict,
):
    save_dir = os.path.join(
        save_folder,
        "result",
        eval_type,
        f"epoch{epoch}",
        "trajectory_distance_prediction",
    )
    os.makedirs(save_dir, exist_ok=True)
    result_save_path = os.path.join(save_dir, f"results.pkl")

    result_label = "f_traj={}_context_size={}_end_slack={}_subsampling_spacing={}_goal_time={}".format(
        index_to_traj["f_traj"],
        int(index_to_traj["context_size"]),
        int(index_to_traj["end_slack"]),
        int(index_to_traj["subsampling_spacing"]),
        int(index_to_traj["goal_time"]),
    )
    result = {
        "f_traj": index_to_traj["f_traj"],
        "context_size": int(index_to_traj["context_size"]),
        "end_slack": int(index_to_traj["end_slack"]),
        "subsampling_spacing": int(index_to_traj["subsampling_spacing"]),
        "goal_time": int(index_to_traj["goal_time"]),
        "neighbor_trajs": [neighbor_traj[0] for neighbor_traj in index_to_traj["neighbor_trajs"]],
        "cand_infos": [(cand_info[0][0], int(cand_info[1])) for cand_info in index_to_traj["cand_infos"]],
        "cand_adj_matrix": cand_adj_matrix,
        "obs_latlong": obs_latlong,
        "goal_latlong": goal_latlong,
        "cand_latlong": cand_latlong,
        "global_obs_pos": global_obs_pos,
        "global_goal_pos": global_goal_pos,
        "global_cand_pos": global_cand_pos,
        "path_idxs": path_idxs,
    }
    traj_results = {result_label: result}

    if os.path.exists(result_save_path):
        with open(result_save_path, "rb") as f:
            results = pkl.load(f)
        results.update(traj_results)
        with open(result_save_path, "wb") as f:
            pkl.dump(results, f)
    else:
        with open(result_save_path, "wb+") as f:
            pkl.dump(traj_results, f)
