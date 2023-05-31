import wandb
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict

# from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
# from stable_contrastive_rl_train.visualizing.critic_utils import visualize_critic_pred
# from stable_contrastive_rl_train.training.train_utils import (
#     get_critic_loss,
#     get_actor_loss,
# )
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.evaluation.visualization_utils import visualize_dist_pairwise_pred
# from gnm_train.training.logger import Logger

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
        # for dataset_type in test_dataloaders:
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
                    use_critic=pairwise_pred_use_critic,
                )

                if use_wandb:
                    wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
                    wandb.log({f"{dataset_type}_pairwise_auc": pairwise_auc})

                print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
                print(f"{dataset_type}_pairwise_auc: {pairwise_auc}")
    print()


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
