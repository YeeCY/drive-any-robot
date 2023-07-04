import wandb
import os
import yaml
import numpy as np
from typing import Dict

from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
from stable_contrastive_rl_train.visualizing.critic_utils import visualize_critic_pred
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.data.data_utils import calculate_sin_cos
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

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def train_eval_rl_loop(
    model: nn.Module,
    optimizer: Dict,
    train_dist_loader: DataLoader,
    train_action_loader: DataLoader,
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
    target_update_freq: int = 1,
    discount: float = 0.99,
    use_td: bool = True,
    bc_coef: float = 0.05,
    mle_gcbc_loss: bool = False,
    stop_grad_actor_img_encoder: bool = True,
    use_actor_waypoint_q_loss: bool = False,
    use_actor_dist_q_loss: bool = False,
    waypoint_gcbc_loss_scale: float = 1.0,
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
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        print(
            f"Start Stable Contrastive RL Training Epoch {epoch}/{current_epoch + epochs - 1}"
        )
        train(
            model,
            optimizer,
            train_dist_loader,
            train_action_loader,
            device,
            project_folder,
            normalized,
            epoch,
            target_update_freq,
            discount,
            use_td,
            bc_coef,
            mle_gcbc_loss,
            stop_grad_actor_img_encoder,
            use_actor_waypoint_q_loss,
            use_actor_dist_q_loss,
            waypoint_gcbc_loss_scale,
            learn_angle,
            print_log_freq,
            image_log_freq,
            num_images_log,
            use_wandb,
        )

        # eval_total_losses = []
        eval_critic_losses, eval_actor_losses = [], []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} Stable Contrastive RL Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            dist_loader = test_dataloaders[dataset_type]["distance"]
            action_loader = test_dataloaders[dataset_type]["action"]
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
            #     waypoint_gcbc_loss_scale,
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

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_eval_critic_loss": np.mean(eval_critic_losses),
            "avg_eval_actor_loss": np.mean(eval_actor_losses),
        }

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

        # TODO: fix distance prediction and predict pairwise accuracy.
        # if (epoch - current_epoch) % pairwise_test_freq == 0:
        #     print(f"Start Pairwise Testing Epoch {epoch}/{current_epoch + epochs - 1}")
        #     for dataset_type in test_dataloaders:
        #         if "pairwise" in test_dataloaders[dataset_type]:
        #             pairwise_dist_loader = test_dataloaders[dataset_type]["pairwise"]
        #             pairwise_accuracy = pairwise_acc(
        #                 model,
        #                 pairwise_dist_loader,
        #                 device,
        #                 project_folder,
        #                 epoch,
        #                 dataset_type,
        #                 print_log_freq,
        #                 image_log_freq,
        #                 num_images_log,
        #                 use_wandb=use_wandb,
        #             )
        #
        #             if use_wandb:
        #                 wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
        #
        #             print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
    print()


def train(
    model: nn.Module,
    optimizer: Dict,
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
    waypoint_gcbc_loss_scale: float = 1.0,
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

    # actor loggers
    actor_loss_logger = Logger("actor_loss", "train", window_size=print_log_freq)
    waypoint_actor_q_loss_logger = Logger("actor/waypoint_actor_q_loss", "train", window_size=print_log_freq)
    waypoint_gcbc_loss_logger = Logger("actor/waypoint_gcbc_loss_logger", "train", window_size=print_log_freq)
    waypoint_gcbc_mle_loss_logger = Logger("actor/waypoint_gcbc_mle_loss", "train", window_size=print_log_freq)
    waypoint_gcbc_mse_loss_logger = Logger("actor/waypoint_gcbc_mse_loss", "train", window_size=print_log_freq)

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
        actor_loss_logger,
        waypoint_actor_q_loss_logger,
        waypoint_gcbc_loss_logger,
        waypoint_gcbc_mle_loss_logger,
        waypoint_gcbc_mse_loss_logger,
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

    # use classifier_to_generative_model data to debug
    # from classifier_to_generative_model.utils import load_and_construct_dataset

    # train_traj_dataset, train_dataset = load_and_construct_dataset(
    #     "/projects/rsalakhugroup/chongyiz/classifier_to_generative_model/datasets/fourrooms.pkl"
    # )

    # TODO (chongyiz): there are some bugs here, tried to fix them.
    # Fixed batch of data
    # # batch_idxs = np.random.choice(len(train_dataset["state"]),
    # #                               size=32, replace=False)
    # batch_idxs = np.arange(32)
    # batch = {k: torch.tensor(v[batch_idxs], device=device) for k, v in train_dataset.items()}

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
        waypoint_oracle = waypoint_oracle.to(device)
        # waypoint_curr_pos = waypoint_curr_pos.to(device)
        # waypoint_yaw = waypoint_yaw.to(device)

        obs_data = (waypoint_obs_data, dist_obs_data)
        next_obs_data = (waypoint_next_obs_data, dist_next_obs_data)
        action_data = (waypoint_label.flatten(1), dist_label)
        goal_data = (waypoint_goal_data, dist_goal_data)

        # Important: the order of loss computation and optimizer update matters!
        # compute critic loss
        waypoint_f_curr, waypoint_obs_time, waypoint_f_goal, waypoint_g_time = waypoint_data_info
        f_mask = torch.Tensor(np.array(waypoint_f_curr)[:, None] == np.array(waypoint_f_goal)[None])
        time_mask = waypoint_obs_time[:, None] < waypoint_g_time[None]
        mc_bce_labels = (f_mask * time_mask).to(device)
        mc_bce_labels = torch.eye(mc_bce_labels.shape[0], device=device)

        # batch_idxs = np.random.choice(len(train_dataset["state"]),
        #                               size=512, replace=False)
        # batch = {k: torch.tensor(v[batch_idxs], device=device) for k, v in train_dataset.items()}

        # critic_loss, critic_info = get_critic_loss(
        #     model, batch["state"], batch["next_state"],
        #     batch["action"], batch["next_state"],
        #     discount, mc_bce_labels, use_td=use_td)
        critic_loss, critic_info = get_critic_loss(
            model, waypoint_obs_data, waypoint_next_obs_data,
            waypoint_label.reshape([waypoint_label.shape[0], -1]), waypoint_goal_data,
            discount, mc_bce_labels, use_td=use_td)

        # compute actor loss
        # actor_loss, actor_info = get_actor_loss(
        #     model, batch["state"], batch["action"], batch["next_state"],
        #     bc_coef=bc_coef, mle_gcbc_loss=mle_gcbc_loss,
        #     stop_grad_actor_img_encoder=stop_grad_actor_img_encoder,
        #     use_actor_waypoint_q_loss=use_actor_waypoint_q_loss,
        #     use_actor_dist_q_loss=use_actor_dist_q_loss,
        #     waypoint_gcbc_loss_scale=waypoint_gcbc_loss_scale)
        actor_loss, actor_info = get_actor_loss(
            model, waypoint_obs_data,
            waypoint_label.reshape([waypoint_label.shape[0], -1]),
            waypoint_goal_data,
            bc_coef=bc_coef, mle_gcbc_loss=mle_gcbc_loss,
            stop_grad_actor_img_encoder=stop_grad_actor_img_encoder,
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
        #
        # del waypoint_pred_logit
        # del waypoint_pred_obs_repr
        # del waypoint_pred_g_repr
        # del waypoint_label_logit
        # del waypoint_label_obs_repr
        # del waypoint_label_g_repr
        # torch.cuda.empty_cache()
        #
        # # (chongyiz): Since we are using DataParallel, we have to use the same batch size
        # #   as training to make sure outputs from the networks are consistent (using for loop).
        # #   Otherwise, the critic predictions are not correct.
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
        #     torch.flatten(waypoint_label[:2], start_dim=1),
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

        # Do all the computations before updating neural networks.
        # Also, we need to update the actor network first because the gradients
        # of actor loss were backpropagated through the critic network.

        # optimize actor
        optimizer["actor_optimizer"].zero_grad()
        # actor_optimizer.zero_grad()
        actor_loss.backward()
        optimizer["actor_optimizer"].step()
        # actor_optimizer.step()

        # optimize critic
        optimizer["critic_optimizer"].zero_grad()
        # critic_optimizer.zero_grad()
        critic_loss.backward()
        optimizer["critic_optimizer"].step()
        # critic_optimizer.step()

        if i % target_update_freq == 0:
            model.soft_update_target_q_network()

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
        # del dist_obs_data
        # del dist_next_obs_data
        # del dist_goal_data
        #
        # del waypoint_obs_data
        # del waypoint_next_obs_data
        # del waypoint_goal_data
        #
        # del critic_loss
        # del critic_info
        # del actor_loss
        # del actor_info

        # del waypoint_pred_logit
        # del waypoint_pred_obs_repr
        # del waypoint_pred_g_repr
        # del waypoint_label_logit
        # del waypoint_label_obs_repr
        # del waypoint_label_g_repr

        # del action_waypts_cos_sim
        # del multi_action_waypts_cos_sim
        # if learn_angle:
        #     del action_orien_cos_sim
        #     del multi_action_orien_cos_sim

        # torch.cuda.empty_cache()

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

        # if i % image_log_freq == 0:
        #     # visualize_dist_pred(
        #     #     to_numpy(dist_obs_image),
        #     #     to_numpy(dist_goal_image),
        #     #     to_numpy(dist_pred),
        #     #     to_numpy(dist_label),
        #     #     "train",
        #     #     project_folder,
        #     #     epoch,
        #     #     num_images_log,
        #     #     use_wandb=use_wandb,
        #     # )
        #     # visualize_traj_pred(
        #     #     to_numpy(obs_image),
        #     #     to_numpy(goal_image),
        #     #     to_numpy(dataset_index),
        #     #     to_numpy(goal_pos),
        #     #     to_numpy(action_pred),
        #     #     to_numpy(action_label),
        #     #     "train",
        #     #     normalized,
        #     #     project_folder,
        #     #     epoch,
        #     #     num_images_log,
        #     #     use_wandb=use_wandb,
        #     # )
        #
        #     # # construct oracle waypoints via predicted waypoints
        #     # dataset_names = list(data_config.keys())
        #     # dataset_names.sort()
        #     # metric_waypoint_spacing = []
        #     # for idx in waypoint_dataset_index:
        #     #     waypoint_dataset_name = dataset_names[int(idx)]
        #     #     waypoint_spacing = data_config[waypoint_dataset_name]["metric_waypoint_spacing"]
        #     #     metric_waypoint_spacing.append(waypoint_spacing)
        #     # metric_waypoint_spacing = torch.tensor(
        #     #     metric_waypoint_spacing, dtype=waypoint_label.dtype, device=device)
        #     #
        #     # global_waypoint_pred = local_to_global_waypoint(
        #     #     waypoint_pred, waypoint_curr_pos, waypoint_yaw, metric_waypoint_spacing,
        #     #     learn_angle=learn_angle, normalized=normalized)
        #     #
        #     # oracle_angles = torch.linspace(
        #     #     -np.deg2rad(45), np.deg2rad(45),
        #     #     9, dtype=waypoint_yaw.dtype, device=device)
        #     #
        #     # waypoint_oracle = []
        #     # for oracle_angle in oracle_angles:
        #     #     global_waypoint_oracle = global_waypoint_pred.clone()
        #     #     global_waypoint_oracle[:, 1:, 2] += oracle_angle
        #     #     global_waypoint_oracle = batch_to_local_coords(
        #     #         global_waypoint_oracle, waypoint_curr_pos, waypoint_yaw - oracle_angle)
        #     #     global_waypoint_oracle[:, 0, 2] += oracle_angle
        #     #
        #     #     waypoint_oracle.append(global_waypoint_oracle)
        #     # waypoint_oracle = torch.stack(waypoint_oracle, dim=1)
        #     #
        #     # if learn_angle:  # localize the waypoint angles
        #     #     waypoint_oracle = calculate_sin_cos(waypoint_oracle)
        #     # if normalized:
        #     #     waypoint_oracle[..., :2] /= (
        #     #         metric_waypoint_spacing[:, None, None, None]
        #     #     )
        #
        #     # (chongyiz): Since we are using DataParallel, we have to use the same batch size
        #     #   as training to make sure outputs from the networks are consistent (using for loop).
        #     #   Otherwise, the critic predictions are not correct.
        #     # waypoint_oracle_critic = []
        #     # for idx in range(waypoint_oracle.shape[1]):
        #     #     waypoint_oracle_obs_repr, _, waypoint_oracle_g_repr = model(
        #     #         waypoint_obs_data, dist_obs_data,
        #     #         waypoint_oracle[:, idx].flatten(1), dist_label,
        #     #         waypoint_obs_data, dist_obs_data)[0:3]
        #     #     # waypoint_oracle_logit = torch.einsum(
        #     #     #     'ikl,jkl->ijl', waypoint_oracle_obs_repr, waypoint_oracle_g_repr)
        #     #     outer = torch.bmm(waypoint_oracle_obs_repr[..., 0].unsqueeze(0),
        #     #                       waypoint_oracle_g_repr[..., 0].permute(1, 0).unsqueeze(0))[0]
        #     #     outer2 = torch.bmm(waypoint_oracle_obs_repr[..., 1].unsqueeze(0),
        #     #                        waypoint_oracle_g_repr[..., 1].permute(1, 0).unsqueeze(0))[0]
        #     #     waypoint_oracle_logit = torch.stack([outer, outer2], dim=-1)
        #     #     waypoint_oracle_logit = torch.diag(torch.mean(waypoint_oracle_logit, dim=-1))
        #     #     waypoint_oracle_critic.append(torch.sigmoid(waypoint_oracle_logit)[:, None])
        #     #
        #     #     del waypoint_oracle_logit
        #     #     del waypoint_oracle_obs_repr
        #     #     del waypoint_oracle_g_repr
        #     #     torch.cuda.empty_cache()
        #     # waypoint_oracle_critic = torch.stack(waypoint_oracle_critic, dim=1)
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
        #         "train",
        #         normalized,
        #         project_folder,
        #         epoch,
        #         num_images_log,
        #         use_wandb=use_wandb,
        #     )
        #
        #     # del metric_waypoint_spacing
        #     # del global_waypoint_pred
        #     # del waypoint_oracle

        # del dist_obs_image
        # del dist_goal_image
        # del dist_label
        #
        # del waypoint_obs_image
        # del waypoint_goal_image
        # del waypoint_dataset_index
        # del waypoint_goal_pos
        # del waypoint_oracle
        # # del waypoint_oracle_critic
        # # del waypoint_pred
        # # del waypoint_pred_critic
        # del waypoint_label
        # # del waypoint_label_critic
        # torch.cuda.empty_cache()

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

            waypoint_f_curr, waypoint_obs_time, waypoint_f_goal, waypoint_g_time = waypoint_data_info
            f_mask = torch.Tensor(np.array(waypoint_f_curr)[:, None] == np.array(waypoint_f_goal)[None])
            time_mask = waypoint_obs_time[:, None] < waypoint_g_time[None]
            mc_bce_labels = (f_mask * time_mask).to(device)
            mc_bce_labels = torch.eye(mc_bce_labels.shape[0], device=device)
            critic_loss, critic_info = get_critic_loss(
                model, obs_data, next_obs_data, action_data, goal_data,
                discount, mc_bce_labels, use_td=use_td)

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

            # del waypoint_pred_logit
            # del waypoint_pred_obs_repr
            # del waypoint_pred_g_repr
            # del waypoint_label_logit
            # del waypoint_label_obs_repr
            # del waypoint_label_g_repr

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

                # # construct oracle waypoints via predicted waypoints
                # dataset_names = list(data_config.keys())
                # dataset_names.sort()
                # metric_waypoint_spacing = []
                # for idx in waypoint_dataset_index:
                #     waypoint_dataset_name = dataset_names[int(idx)]
                #     waypoint_spacing = data_config[waypoint_dataset_name]["metric_waypoint_spacing"]
                #     metric_waypoint_spacing.append(waypoint_spacing)
                # metric_waypoint_spacing = torch.tensor(
                #     metric_waypoint_spacing, dtype=waypoint_label.dtype, device=device)
                #
                # global_waypoint_pred = local_to_global_waypoint(
                #     waypoint_pred, waypoint_curr_pos, waypoint_yaw, metric_waypoint_spacing,
                #     learn_angle=learn_angle, normalized=normalized)
                #
                # oracle_angles = torch.linspace(
                #     -np.deg2rad(45), np.deg2rad(45),
                #     9, dtype=waypoint_yaw.dtype, device=device)

                # waypoint_oracle = []
                # for oracle_angle in oracle_angles:
                #     global_waypoint_oracle = global_waypoint_pred.clone()
                #     global_waypoint_oracle[:, 1:, 2] += oracle_angle
                #     global_waypoint_oracle = batch_to_local_coords(
                #         global_waypoint_oracle, waypoint_curr_pos, waypoint_yaw - oracle_angle)
                #     global_waypoint_oracle[:, 0, 2] += oracle_angle
                #
                #     waypoint_oracle.append(global_waypoint_oracle)
                # waypoint_oracle = torch.stack(waypoint_oracle, dim=1)
                #
                # if learn_angle:  # localize the waypoint angles
                #     waypoint_oracle = calculate_sin_cos(waypoint_oracle)
                # if normalized:
                #     waypoint_oracle[..., :2] /= (
                #         metric_waypoint_spacing[:, None, None, None]
                #     )

                # waypoint_oracle_critic = []
                # for idx in range(waypoint_oracle.shape[1]):
                #     waypoint_oracle_obs_repr, _, waypoint_oracle_g_repr = model(
                #         waypoint_oracle_obs_data[:, idx], waypoint_oracle_obs_data[:, idx],
                #         waypoint_oracle[:, idx].flatten(1), dist_label,
                #         waypoint_oracle_goal_data[:, idx], waypoint_oracle_goal_data[:, idx])[0:3]
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
            transf_obs_image = transf_obs_image.to(device)
            transf_close_image = transf_close_image.to(device)
            transf_far_image = transf_far_image.to(device)

            dummy_action = torch.zeros([transf_obs_image.shape[0], model.action_size], device=transf_obs_image.device)
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
            correct_list.append(correct)
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
        return np.concatenate(correct_list).mean()


def get_critic_loss(model, obs, next_obs, action, goal, discount, mc_bce_labels, use_td=False):
    # waypoint_obs, dist_obs = obs
    # waypoint_next_obs, dist_next_obs = next_obs
    # waypoint, dist = action
    # waypoint_goal, dist_goal = goal
    waypoint_obs = obs
    waypoint_next_obs = next_obs
    waypoint = action
    waypoint_goal = goal

    # dummy_waypoint = torch.zeros(
    #     [waypoint_obs.shape[0], model.len_trajectory_pred, model.num_action_params],
    #     device=waypoint_obs.device
    # )
    # dummy_waypoint[..., 2] = 1  # cos of yaws
    # dummy_waypoint = dummy_waypoint.reshape([waypoint_obs.shape[0], model.action_size])

    batch_size = waypoint_obs.shape[0]

    bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

    waypoint_new_goal = waypoint_goal
    if use_td:
        # extract next goal from fusion of observations and contexts.
        waypoint_new_goal = waypoint_next_obs[:, -3:]
    # I = torch.eye(batch_size, device=waypoint_obs.device)

    # obs_waypoint_repr, obs_dist_repr, g_repr = model(obs, action, new_goal)[0:3]
    # obs_repr, g_repr = model(waypoint_obs, waypoint, waypoint_new_goal)[:2]
    # logits = torch.einsum('ikl,jkl->ijl', obs_repr, g_repr)
    # logits = torch.diag(model(waypoint_obs, waypoint, waypoint_new_goal)[0].mean(-1))
    # logits = model(waypoint_obs, waypoint, waypoint_new_goal)[0]
    #
    # goal_indices = torch.roll(
    #     torch.arange(batch_size, dtype=torch.int64), -1)
    # waypoint_rand_goal = waypoint_new_goal[goal_indices]
    # # rand_logits = torch.diag(model(waypoint_obs, waypoint, waypoint_rand_goal)[0].mean(-1))
    # rand_logits = model(waypoint_obs, waypoint, waypoint_rand_goal)[0]

    # logits = model.q_network(waypoint_obs, waypoint, waypoint_new_goal)[0]

    # Make sure to use the twin Q trick.
    # assert len(logits.shape) == 3

    pos_logits = model(waypoint_obs, waypoint, waypoint_new_goal)[0]

    goal_indices = torch.roll(
        torch.arange(batch_size, dtype=torch.int64), -1)
    waypoint_rand_goal = waypoint_new_goal[goal_indices]
    neg_logits = model(waypoint_obs, waypoint, waypoint_rand_goal)[0]

    if use_td:
        # action was not used here
        next_waypoint_mean, next_waypoint_std = model(
            waypoint_next_obs, waypoint, waypoint_rand_goal)[-2:]

        next_waypoint_dist = Independent(
            Normal(next_waypoint_mean, next_waypoint_std, validate_args=False),
            reinterpreted_batch_ndims=1
        )
        next_waypoint = next_waypoint_dist.rsample()

        # use target critic
        # next_obs_repr, next_rand_g_repr = model(
        #     waypoint_next_obs, next_waypoint, waypoint_rand_goal)[2:4]
        # next_q = torch.einsum('ikl,jkl->ijl', next_obs_repr, next_rand_g_repr)
        next_q = model(waypoint_next_obs, next_waypoint, waypoint_rand_goal)[1]

        next_q = torch.sigmoid(next_q)
        next_v = torch.min(next_q, dim=-1)[0].detach()
        # next_v = torch.diag(next_v)
        w = next_v / (1 - next_v)
        w_clipping = 20.0
        w = torch.clamp(w, min=0.0, max=w_clipping)

        # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
        # pos_logits = torch.diagonal(logits).permute(1, 0)
        loss_pos = bce_with_logits_loss(
            pos_logits, torch.ones_like(pos_logits))

        # neg_logits = logits[torch.arange(batch_size), goal_indices]
        loss_neg1 = w[:, None] * bce_with_logits_loss(
            neg_logits, torch.ones_like(neg_logits))
        loss_neg2 = bce_with_logits_loss(
            neg_logits, torch.zeros_like(neg_logits))

        qf_loss = (1 - discount) * loss_pos + discount * loss_neg1 + loss_neg2
    else:
        # decrease the weight of negative term to 1 / (B - 1)
        # qf_loss_weights = torch.ones(
        #     (batch_size, batch_size), device=waypoint_obs.device) / (batch_size - 1)
        # qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = 1
        # qf_loss_weights = torch.ones(
        #     (batch_size, batch_size), device=waypoint_obs.device)
        # qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = (batch_size - 1)
        # logits.shape = (B, B, 2) with 1 term for positive pair
        # and (B - 1) terms for negative pairs in each row
        # if len(logits.shape) == 3:
        #     qf_loss = qf_loss_weights[..., None] * bce_with_logits_loss(
        #         logits, mc_bce_labels.unsqueeze(-1).repeat_interleave(2, dim=-1))
        # else:
        #     qf_loss = qf_loss_weights * bce_with_logits_loss(logits, mc_bce_labels)
        # qf_loss *= qf_loss_weights

        # use separate positive and negative logits
        # logits = torch.stack([logits, rand_logits], dim=1)

        qf_loss = bce_with_logits_loss(pos_logits, torch.ones_like(pos_logits)) \
                  + bce_with_logits_loss(neg_logits, torch.zeros_like(neg_logits))

    critic_loss = torch.mean(qf_loss)

    waypoint_logits = torch.mean(torch.stack([pos_logits, neg_logits], dim=1), dim=-1)
    # # waypoint_correct = mc_bce_labels[torch.arange(batch_size), torch.argmax(waypoint_logits, dim=-1)]
    # waypoint_correct = (torch.argmax(waypoint_logits, dim=-1) == torch.argmax(mc_bce_labels, dim=-1))
    # waypoint_logits_pos = torch.sum(waypoint_logits * mc_bce_labels) / torch.sum(mc_bce_labels)
    # waypoint_logits_neg = torch.sum(waypoint_logits * (1 - mc_bce_labels)) / torch.sum(1 - mc_bce_labels)

    binary_acc_targets = torch.stack([
        torch.ones_like(pos_logits.mean(-1)),
        torch.zeros_like(neg_logits.mean(-1))
    ], dim=1)
    waypoint_correct = (torch.argmax(waypoint_logits, dim=-1) == 0)
    waypoint_logits_pos = waypoint_logits[:, 0]
    waypoint_logits_neg = waypoint_logits[:, 1]

    return critic_loss, {
        # "waypoint_binary_accuracy": torch.mean(((waypoint_logits > 0) == mc_bce_labels).float()),
        "waypoint_binary_accuracy": torch.mean(((waypoint_logits > 0) == binary_acc_targets).float()),
        "waypoint_categorical_accuracy": torch.mean(waypoint_correct.float()),
        "waypoint_logits_pos": waypoint_logits_pos.mean(),
        "waypoint_logits_neg": waypoint_logits_neg.mean(),
        "waypoint_logits": waypoint_logits.mean(),
    }


def get_actor_loss(model, obs, orig_action, goal, bc_coef=0.05,
                   mle_gcbc_loss=False, stop_grad_actor_img_encoder=True,
                   use_actor_waypoint_q_loss=True, use_actor_dist_q_loss=True,
                   waypoint_gcbc_loss_scale=1.0):
    """
    We might need to add alpha and GCBC term.
    """
    assert use_actor_waypoint_q_loss or use_actor_dist_q_loss

    # waypoint_obs, dist_obs = obs
    # waypoint, dist = orig_action
    # waypoint_goal, dist_goal = goal
    waypoint_obs = obs
    waypoint = orig_action
    waypoint_goal = goal

    waypoint_mean, waypoint_std = model(
        waypoint_obs, waypoint, waypoint_goal)[-2:]

    waypoint_dist = Independent(Normal(waypoint_mean, waypoint_std,
                                       validate_args=False), reinterpreted_batch_ndims=1)

    sampled_waypoint = waypoint_dist.rsample()

    # obs_repr, g_repr = model(
    #     waypoint_obs, sampled_waypoint, waypoint_goal)[:2]
    # waypoint_q = torch.einsum('ikl,jkl->ijl', obs_repr, g_repr)
    waypoint_q = model(waypoint_obs, sampled_waypoint, waypoint_goal)[0]

    # if len(waypoint_q.shape) == 3:  # twin q trick
    #     assert waypoint_q.shape[2] == 2
    #     waypoint_q = torch.min(waypoint_q, dim=-1)[0]
    if waypoint_q.shape[-1] == 2:
        waypoint_q = torch.min(waypoint_q, dim=-1)[0]

    actor_q_loss, actor_waypoint_q_loss = \
        torch.zeros_like(waypoint_mean)[:, 0], \
        torch.zeros_like(waypoint_mean)[:, 0]
    if use_actor_waypoint_q_loss:
        # actor_waypoint_q_loss = -torch.diag(waypoint_q)
        actor_waypoint_q_loss = -waypoint_q
        actor_q_loss += actor_waypoint_q_loss

    waypoint_gcbc_mle_loss = -waypoint_dist.log_prob(waypoint)
    waypoint_gcbc_mse_loss = F.mse_loss(waypoint_mean, waypoint)

    if mle_gcbc_loss:
        gcbc_loss = waypoint_gcbc_mle_loss
    else:
        gcbc_loss = waypoint_gcbc_mse_loss

    actor_loss = bc_coef * gcbc_loss + (1 - bc_coef) * actor_q_loss

    actor_loss = torch.mean(actor_loss)
    actor_waypoint_q_loss = torch.mean(actor_waypoint_q_loss)

    gcbc_loss = torch.mean(gcbc_loss)
    waypoint_gcbc_mle_loss = torch.mean(waypoint_gcbc_mle_loss)
    waypoint_gcbc_mse_loss = torch.mean(waypoint_gcbc_mse_loss)

    return actor_loss, {
        "waypoint_actor_q_loss": actor_waypoint_q_loss,
        "waypoint_gcbc_loss": gcbc_loss,
        "waypoint_gcbc_mle_loss": waypoint_gcbc_mle_loss,
        "waypoint_gcbc_mse_loss": waypoint_gcbc_mse_loss,
    }


def load_model(model, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    loaded_model = checkpoint["model"]
    try:  # for DataParallel
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict)
    except (RuntimeError, AttributeError) as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict)


def get_saved_optimizer(
    checkpoint: dict, device: torch.device
) -> torch.optim.Optimizer:
    optimizer = checkpoint["optimizer"]
    optimizer_to(optimizer, device)
    return optimizer


def optimizer_to(optim, device):
    """Move optimizer state to device."""
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def batch_yaw_rotmat(yaw):
    return torch.stack([
        torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw)], dim=-1),
        torch.stack([torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], dim=-1),
        torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1),
        ],
        dim=1
    )


def batch_to_local_coords(
    positions, curr_pos, curr_yaw
):
    rotmat = batch_yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:, :2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return torch.matmul(positions - curr_pos[:, None], rotmat)


def batch_to_global_coords(
    positions, curr_pos, curr_yaw
):
    rotmat = batch_yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:, :2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return torch.matmul(positions, torch.linalg.inv(rotmat)) + curr_pos[:, None]


def local_to_global_waypoint(
        waypoint, curr_pos, yaw, metric_waypoint_spacing, learn_angle=True, normalized=True):
    waypoint_pos = waypoint[..., :2]
    if normalized:
        waypoint_pos *= metric_waypoint_spacing[:, None, None]
    if learn_angle:
        waypoint_angle = torch.arctan2(
            waypoint[..., [3]], waypoint[..., [2]])
        waypoint_angle[:, 1:] += waypoint_angle[:, [0]]

    local_waypoint = torch.cat(
        [waypoint_pos, waypoint_angle], dim=-1)

    global_waypoint = batch_to_global_coords(
        local_waypoint, curr_pos, yaw)

    # force angles to stay in [-pi, pi)
    global_waypoint[..., 2] = torch.arctan2(
        torch.sin(global_waypoint[..., 2]), torch.cos(global_waypoint[..., 2]))

    # if learn_angle:
    #     global_waypoint[:, 1:, 2] += global_waypoint[:, [0], 2]

    return global_waypoint
