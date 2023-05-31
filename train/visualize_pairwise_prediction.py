import os
import wandb
import argparse
import numpy as np
import yaml
import glob
import pickle as pkl
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from gnm_train.models.gnm import GNM
from gnm_train.models.siamese import SiameseModel
from gnm_train.models.stacked import StackedModel
from gnm_train.data.gnm_dataset import GNM_Dataset
# from gnm_train.data.pairwise_distance_dataset import PairwiseDistanceDataset
from gnm_train.data.pairwise_distance_dataset import (
    PairwiseDistanceEvalDataset,
    PairwiseDistanceFailureDataset
)
from gnm_train.training.train_utils import load_model
from gnm_train.evaluation.eval_utils import eval_loop

from stable_contrastive_rl_train.data.rl_dataset import RLDataset
from stable_contrastive_rl_train.models.base_model import DataParallel
from stable_contrastive_rl_train.models.stable_contrastive_rl import StableContrastiveRL
from stable_contrastive_rl_train.evaluation.eval_utils import eval_rl_loop


def main(config):
    # read results
    gnm_dir = config["result_dirs"]["gnm"]
    rl_mc_reg_dir = config["result_dirs"]["stable_contrastive_rl_mc_regression"]
    gnm_filename = os.path.join(gnm_dir, 'results.pkl')
    rl_mc_reg_filename = os.path.join(rl_mc_reg_dir, 'results.pkl')

    with open(gnm_filename, "rb") as f:
        gnm_results = pkl.load(f)
    with open(rl_mc_reg_filename, "rb") as f:
        rl_mc_reg_results = pkl.load(f)

    for label, gnm_result in gnm_results.items():
        f_close = gnm_result["f_close"]
        f_far = gnm_result["f_far"]
        curr_time = gnm_result["curr_time"]
        close_time = gnm_result["close_time"]
        far_time = gnm_result["far_time"]
        context_times = gnm_result["context_times"]
        gnm_far_dist_pred = gnm_result["far_dist_pred"]
        gnm_far_dist_label = gnm_result["far_dist_label"]
        gnm_close_dist_pred = gnm_result["close_dist_pred"]
        gnm_close_dist_label = gnm_result["close_dist_label"]

        rl_mc_reg_result = rl_mc_reg_results[label]
        print()

        # with open(rl_mc_reg_filename, "rb") as f:
        #     rl_mc_reg_result = pkl.load(f)
        #
        # assert f_close == rl_mc_reg_result["f_close"]
        # assert f_far == rl_mc_reg_result["f_far"]
        # assert curr_time == rl_mc_reg_result["curr_time"]
        # assert close_time == rl_mc_reg_result["close_time"]
        # assert far_time == rl_mc_reg_result["far_time"]
        # assert context_times == rl_mc_reg_result["context_times"]
        # # gnm_far_dist_pred = gnm_result["far_dist_pred"]
        # # gnm_far_dist_label = gnm_result["far_dist_label"]
        # # gnm_close_dist_pred = gnm_result["close_dist_pred"]
        # # gnm_close_dist_label = gnm_result["close_dist_label"]

    print()



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

    print(config)
    main(config)
