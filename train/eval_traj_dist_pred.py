import os
import wandb
import argparse
import numpy as np
import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from gnm_train.models.gnm import GNM
from gnm_train.models.siamese import SiameseModel
from gnm_train.models.stacked import StackedModel
from gnm_train.data.gnm_dataset import (
    GNM_Dataset,
    GNM_EvalDataset,
)
# from gnm_train.data.pairwise_distance_dataset import PairwiseDistanceDataset
from gnm_train.data.pairwise_distance_dataset import (
    PairwiseDistanceEvalDataset,
    PairwiseDistanceFailureDataset
)
from gnm_train.training.train_utils import load_model
from gnm_train.evaluation.eval_utils import eval_loop

from stable_contrastive_rl_train.data.rl_dataset import (
    RLTrajDataset,
)
from stable_contrastive_rl_train.models.base_model import DataParallel
from stable_contrastive_rl_train.models.stable_contrastive_rl import StableContrastiveRL
from stable_contrastive_rl_train.evaluation.eval_utils import eval_rl_loop


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = [
        transforms.ToTensor(),
        transforms.Resize(
            (config["image_size"][1], config["image_size"][0])
        ),  # torch does (h, w)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform)
    aspect_ratio = config["image_size"][0] / config["image_size"][1]
    if config.get("img_encoder_kwargs", None):
        config["img_encoder_kwargs"]["input_width"] = config["image_size"][1]
        config["img_encoder_kwargs"]["input_height"] = config["image_size"][0]

    # Load the data
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    data_split_type = "test"
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        output_type = "traj_dist_pred"
        if config["eval"] == "rl":
            dataset = RLTrajDataset(
                data_folder=data_config["data_folder"],
                traj_names=data_config["test_traj_names"],
                goal_idxs=data_config["goal_idxs"],
                dataset_name=dataset_name,
                transform=transform,
                aspect_ratio=aspect_ratio,
                waypoint_spacing=data_config["waypoint_spacing"],
                subsampling_spacing=data_config["subsampling_spacing"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=data_config["end_slack"],
                normalize=config["normalize"],
            )
        else:
            dataset = GNM_EvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config[data_split_type],
                dataset_name=dataset_name,
                is_action=(output_type == "action"),
                transform=transform,
                aspect_ratio=aspect_ratio,
                waypoint_spacing=data_config["waypoint_spacing"],
                min_dist_cat=config[output_type]["min_dist_cat"],
                max_dist_cat=config[output_type]["max_dist_cat"],
                negative_mining=data_config["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=data_config["end_slack"],
                goals_per_obs=data_config["goals_per_obs"],
                normalize=config["normalize"],
            )
        dataset_type = f"{dataset_name}_{data_split_type}"
        if dataset_type not in test_dataloaders:
            test_dataloaders[dataset_type] = {}
        test_dataloaders[dataset_type][output_type] = dataset

    if "traj_eval_batch_size" not in config:
        config["traj_eval_batch_size"] = config["traj_batch_size"]
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type in test_dataloaders:
        for loader_type in test_dataloaders[dataset_type]:
            if loader_type == "traj_dist_pred":
                test_dataloaders[dataset_type][loader_type] = DataLoader(
                    test_dataloaders[dataset_type][loader_type],
                    batch_size=config["traj_eval_batch_size"],
                    shuffle=False,  # (chongyiz): we don't need to shuffle the trajectory dataset
                    num_workers=config["num_workers"],
                )
            else:
                raise RuntimeError(f"Unknown loader type {loader_type}")

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "siamese":
        model = SiameseModel(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "stacked":
        model = StackedModel(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obsgoal_encoding_size"],
        )
    elif config["model_type"] == "stable_contrastive_rl":
        model = StableContrastiveRL(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["soft_target_tau"],
            config["img_encoder_kwargs"],
            config["contrastive_critic_kwargs"],
            config["policy_kwargs"]
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if len(config["gpu_ids"]) > 1:
        model = DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path, map_location=device)
        load_model(model, latest_checkpoint)
        # optimizer = get_saved_optimizer(latest_checkpoint, device)
        current_epoch = latest_checkpoint["epoch"] + 1

    torch.autograd.set_detect_anomaly(True)
    if config["eval"] == "supervised":
        try:
            assert type(model) != StableContrastiveRL
        except AssertionError:
            assert type(model.module) != StableContrastiveRL

        eval_loop(
            model=model,
            # optimizer=optimizer,
            # train_dist_loader=train_dist_loader,
            # train_action_loader=train_action_loader,
            test_dataloaders=test_dataloaders,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            # pairwise_test_freq=config["pairwise_test_freq"],
            save_pairwise_dist_pred_freq=config["save_pairwise_dist_pred_freq"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            save_failure_index_to_data=config["save_failure_index_to_data"],
            eval_waypoint=config["eval_waypoint"],
            eval_pairwise_dist_pred=config["eval_pairwise_dist_pred"],
        )
    elif config["eval"] == "rl":
        try:
            assert type(model) == StableContrastiveRL
        except AssertionError:
            assert type(model.module) == StableContrastiveRL

        assert config["eval_waypoint"] is False
        assert config["eval_pairwise_dist_pred"] is False
        assert config["eval_traj_dist_pred"] is True

        eval_rl_loop(
            model=model,
            # optimizer=optimizer,
            # train_dist_loader=train_dist_loader,
            # train_action_loader=train_action_loader,
            test_dataloaders=test_dataloaders,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            save_pairwise_dist_pred_freq=config["save_pairwise_dist_pred_freq"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            discount=config["discount"],
            use_td=config["use_td"],
            bc_coef=config["bc_coef"],
            mle_gcbc_loss=config["mle_gcbc_loss"],
            use_actor_waypoint_q_loss=config["use_actor_waypoint_q_loss"],
            use_actor_dist_q_loss=config["use_actor_dist_q_loss"],
            waypoint_gcbc_loss_scale=config["waypoint_gcbc_loss_scale"],
            use_wandb=config["use_wandb"],
            pairwise_dist_pred_eval_mode=config["pairwise_dist_pred_eval_mode"],
            eval_waypoint=config["eval_waypoint"],
            eval_pairwise_dist_pred=config["eval_pairwise_dist_pred"],
            eval_traj_dist_pred=config["eval_traj_dist_pred"],
        )
    else:
        raise ValueError(f"Evaluation type {config['train']} not supported")
    print("FINISHED EVALUATION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Robot Agnostic Learning")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/gnm/gnm_public.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"], settings=wandb.Settings(start_method="fork")
        )
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
