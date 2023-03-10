import wandb
import os
import numpy as np
from typing import List, Optional, Dict

from gnm_train.visualizing.action_utils import visualize_traj_pred
from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.training.logger import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.optim import Adam
from torch.optim import Optimizer


def train_eval_rl_loop(
    model: nn.Module,
    optimizer: Dict[str, Optimizer],
    # train_dist_loader: DataLoader,
    # train_action_loader: DataLoader,
    train_rl_loader: DataLoader,
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
            f"Start GNM Training Epoch {epoch}/{current_epoch + epochs - 1}"
        )
        train(
            model,
            optimizer,
            train_rl_loader,
            device,
            project_folder,
            normalized,
            epoch,
            target_update_freq,
            discount,
            use_td,
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
                f"Start {dataset_type} GNM Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            rl_loader = test_dataloaders[dataset_type]["rl"]
            test_critic_loss, test_actor_loss = evaluate(
                dataset_type,
                model,
                rl_loader,
                device,
                project_folder,
                normalized,
                epoch,
                discount,
                use_td,
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
            wandb.log({f"{dataset_type}_critic_loss": test_critic_loss})
            print(f"{dataset_type}_critic_loss: {test_critic_loss}")
            wandb.log({f"{dataset_type}_actor_loss": test_actor_loss})
            print(f"{dataset_type}_actor_loss: {test_actor_loss}")

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

        if (epoch - current_epoch) % pairwise_test_freq == 0:
            print(f"Start Pairwise Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            for dataset_type in test_dataloaders:
                if "pairwise" in test_dataloaders[dataset_type]:
                    pairwise_dist_loader = test_dataloaders[dataset_type]["pairwise"]
                    pairwise_accuracy = pairwise_acc(
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
                    wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
                    print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
    print()


def train(
    model: nn.Module,
    optimizer: Dict[str, Optimizer],
    train_rl_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    target_update_freq: int = 1,
    discount: float = 0.99,
    use_td: bool = True,
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
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    critic_loss_logger = Logger("critic_loss", "train", window_size=print_log_freq)
    actor_loss_logger = Logger("actor_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )

    variables = [
        critic_loss_logger,
        actor_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
    ]

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = len(train_rl_loader)
    for i, vals in enumerate(train_rl_loader):
        # FIXME (chongyiz)
        # dist_vals, action_vals = val
        # (
        #     dist_obs_image,
        #     dist_goal_image,
        #     dist_trans_obs_image,
        #     dist_trans_goal_image,
        #     dist_label,
        #     dist_dataset_index,
        # ) = dist_vals
        # (
        #     action_obs_image,
        #     action_goal_image,
        #     action_trans_obs_image,
        #     action_trans_goal_image,
        #     action_goal_pos,
        #     action_label,
        #     action_dataset_index,
        # ) = action_vals
        # dist_obs_data = dist_trans_obs_image.to(device)
        # dist_goal_data = dist_trans_goal_image.to(device)
        # dist_label = dist_label.to(device)

        (
            obs_image,
            next_obs_image,
            goal_image,
            trans_obs_image,
            trans_next_obs_image,
            trans_goal_image,
            goal_pos,
            action_label,
            dist_label,
            dataset_index,
        ) = vals
        obs_data = trans_obs_image.to(device)
        next_obs_data = trans_next_obs_image.to(device)
        goal_data = trans_goal_image.to(device)
        action_label = action_label.to(device)
        dist_label = dist_label.to(device)
        action_data = torch.cat([
            action_label.reshape([action_label.shape[0], -1]),
            dist_label],
            dim=-1
        )

        optimizer["critic_optimizer"].zero_grad()
        optimizer["actor_optimizer"].zero_grad()

        # dist_pred, _ = model(dist_obs_data, dist_goal_data)
        # dist_loss = F.mse_loss(dist_pred, dist_label)
        #
        # action_obs_data = action_trans_obs_image.to(device)
        # action_goal_data = action_trans_goal_image.to(device)
        # action_label = action_label.to(device)

        # _, action_pred = model(action_obs_data, action_goal_data)
        # action_loss = F.mse_loss(action_pred, action_label)

        critic_loss = get_critic_loss(
            model, obs_data, next_obs_data, action_data, goal_data,
            discount, use_td=use_td)

        actor_loss = get_actor_loss(model, obs_data, goal_data)

        # preds = model.policy_network(obs_data, goal_data).mean
        # action_data are not used here
        _, preds = model(obs_data, action_data, goal_data).mean

        # The action of policy is different from the action here (waypoints).
        action_pred = preds[:, :-1]
        action_pred = action_pred.reshape(action_label.shape)

        dist_pred = preds[:, -1]

        action_waypts_cos_similairity = F.cosine_similarity(
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

        # prevent NAN via gradient clipping
        # total_loss.backward()
        # optimizer.step()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm(
        #     model.policy_network.parameters(), 1.0)
        optimizer["actor_optimizer"].step()

        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(
        #     model.q_network.parameters(), 1.0)
        optimizer["critic_optimizer"].step()

        if i % target_update_freq == 0:
            model.soft_update_target_q_network()

        critic_loss_logger.log_data(critic_loss.item())
        actor_loss_logger.log_data(actor_loss.item())
        action_waypts_cos_sim_logger.log_data(action_waypts_cos_similairity.item())
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
                to_numpy(obs_image),
                to_numpy(goal_image),
                to_numpy(dist_pred),
                to_numpy(dist_label),
                "train",
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
            visualize_traj_pred(
                to_numpy(obs_image),
                to_numpy(goal_image),
                to_numpy(dataset_index),
                to_numpy(goal_pos),
                to_numpy(action_pred),
                to_numpy(action_label),
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
    eval_rl_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    discount: float = 0.99,
    use_td: bool = True,
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
    critic_loss_logger = Logger("critic_loss", eval_type, window_size=print_log_freq)
    actor_loss_logger = Logger("actor_loss", eval_type, window_size=print_log_freq)
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
        critic_loss_logger,
        actor_loss_logger,
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

    # num_batches = min(len(eval_dist_loader), len(eval_action_loader))
    num_batches = len(eval_rl_loader)

    with torch.no_grad():
        for i, vals in enumerate(eval_rl_loader):
            # dist_vals, action_vals = val
            # (
            #     dist_obs_image,
            #     dist_goal_image,
            #     dist_trans_obs_image,
            #     dist_trans_goal_image,
            #     dist_label,
            #     dist_dataset_index,
            # ) = dist_vals
            # (
            #     action_obs_image,
            #     action_goal_image,
            #     action_trans_obs_image,
            #     action_trans_goal_image,
            #     action_goal_pos,
            #     action_label,
            #     action_dataset_index,
            # ) = action_vals
            # dist_obs_data = dist_trans_obs_image.to(device)
            # dist_goal_data = dist_trans_goal_image.to(device)
            # dist_label = dist_label.to(device)

            (
                obs_image,
                next_obs_image,
                goal_image,
                trans_obs_image,
                trans_next_obs_image,
                trans_goal_image,
                goal_pos,
                action_label,
                dist_label,
                dataset_index,
            ) = vals
            obs_data = trans_obs_image.to(device)
            next_obs_data = trans_next_obs_image.to(device)
            goal_data = trans_goal_image.to(device)
            action_label = action_label.to(device)
            dist_label = dist_label.to(device)
            action_data = torch.cat([
                action_label.reshape([action_label.shape[0], -1]),
                dist_label],
                dim=-1
            )

            # dist_pred, _ = model(dist_obs_data, dist_goal_data)
            # dist_loss = F.mse_loss(dist_pred, dist_label)
            #
            # action_obs_data = action_trans_obs_image.to(device)
            # action_goal_data = action_trans_goal_image.to(device)
            # action_label = action_label.to(device)

            # _, action_pred = model(action_obs_data, action_goal_data)
            # action_loss = F.mse_loss(action_pred, action_label)

            critic_loss = get_critic_loss(
                model, obs_data, next_obs_data, action_data, goal_data,
                discount, use_td=use_td)

            actor_loss = get_actor_loss(model, obs_data, goal_data)

            # preds = model.policy_network(obs_data, goal_data).mean
            # action_data are not used here
            _, preds = model(obs_data, action_data, goal_data).mean

            # The action of policy is different from the action here (waypoints).
            action_pred = preds[:, :-1]
            action_pred = action_pred.reshape(action_label.shape)

            dist_pred = preds[:, -1]

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

            # total_loss = alpha * (1e-3 * dist_loss) + (1 - alpha) * action_loss

            # dist_loss_logger.log_data(dist_loss.item())
            # action_loss_logger.log_data(action_loss.item())

            critic_loss_logger.log_data(critic_loss.item())
            actor_loss_logger.log_data(actor_loss.item())
            action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            multi_action_waypts_cos_sim_logger.log_data(
                multi_action_waypts_cos_sim.item()
            )
            # total_loss_logger.log_data(total_loss.item())

            if i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            if i % image_log_freq == 0:
                visualize_dist_pred(
                    to_numpy(obs_image),
                    to_numpy(goal_image),
                    to_numpy(dist_pred),
                    to_numpy(dist_label),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    use_wandb=use_wandb,
                )
                visualize_traj_pred(
                    to_numpy(obs_image),
                    to_numpy(goal_image),
                    to_numpy(dataset_index),
                    to_numpy(goal_pos),
                    to_numpy(action_pred),
                    to_numpy(action_label),
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

            # close_pred, _ = model(transf_obs_image, transf_close_image)
            # far_pred, _ = model(transf_obs_image, transf_far_image)
            # close_pred = model.policy_network(transf_obs_image, transf_close_image).mean
            # far_pred = model.policy_network(transf_obs_image, transf_far_image).mean
            dummy_action = torch.zeros([transf_obs_image.shape[0], model.action_size], device=transf_obs_image.device)
            _, close_pred = model(transf_obs_image, dummy_action, transf_close_image).mean
            _, far_pred = model(transf_obs_image, dummy_action, transf_far_image).mean
            close_dist_pred, far_dist_pred = close_pred[:, -1], far_pred[:, -1]

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


def get_critic_loss(model, obs, next_obs, action, goal, discount, use_td=False):
    batch_size = obs.shape[0]

    bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

    I = torch.eye(batch_size, device=obs.device)
    # logits = model.q_network(obs, action, goal)
    logits, _ = model(obs, action, goal)

    if use_td:
        # Make sure to use the twin Q trick.
        assert len(logits.shape) == 3

        goal_indices = torch.roll(
            torch.arange(batch_size, dtype=torch.int64), -1)

        random_goal = goal[goal_indices]

        # next_dist = model.policy_network(next_obs, random_goal)
        # action was not used here
        _, next_dist = model(next_obs, action, random_goal)
        next_action = next_dist.rsample()

        next_q = model.target_q_network(
            next_obs, next_action, random_goal)

        next_q = torch.sigmoid(next_q)
        next_v = torch.min(next_q, dim=-1)[0].detach()
        next_v = torch.diag(next_v)
        w = next_v / (1 - next_v)

        w_clipping = 20.0
        w = torch.clamp(w, min=0.0, max=w_clipping)

        pos_logits = torch.diagonal(logits).permute(1, 0)
        loss_pos = bce_with_logits_loss(
            pos_logits, torch.ones_like(pos_logits))

        neg_logits = logits[torch.arange(batch_size), goal_indices]
        loss_neg1 = w[:, None] * bce_with_logits_loss(
            neg_logits, torch.ones_like(neg_logits))
        loss_neg2 = bce_with_logits_loss(
            neg_logits, torch.zeros_like(neg_logits))

        critic_loss = (1 - discount) * loss_pos + \
                      discount * loss_neg1 + loss_neg2
        critic_loss = torch.mean(critic_loss)
    else:
        # TODO
        raise NotImplementedError

    return critic_loss


def get_actor_loss(model, obs, goal):
    """
    We might need to add alpha and GCBC term.
    """

    dummy_actions = torch.zeros([obs.shape[0], model.action_size], device=obs.device)
    # _, dist = model.policy_network(obs, dummy_actions, goal)
    _, dist = model(obs, dummy_actions, goal)
    sampled_action = dist.rsample()
    log_prob = dist.log_prob(sampled_action)

    # q_action = model.q_network(obs, sampled_action, goal)
    q_action, _ = model(obs, sampled_action, goal)

    if len(q_action.shape) == 3:  # twin q trick
        assert q_action.shape[2] == 2
        q_action = torch.min(q_action, dim=-1)[0]

    actor_loss = -torch.diag(q_action)

    actor_loss = torch.mean(actor_loss)

    return actor_loss


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
