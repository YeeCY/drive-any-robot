import pickle as pkl
import numpy as np


def load_and_construct_dataset(traj_dataset_filepath):
    with open(traj_dataset_filepath, "rb") as f:
        traj_dataset = pkl.load(f)
    print("Load dataset from: {}".format(traj_dataset_filepath))
    print("Number of transitions in the trajectory dataset: {}".format(
        sum([len(traj) for traj in traj_dataset])))

    dataset = dict()
    for traj_idx, traj in enumerate(traj_dataset):
        for t in range(0, len(traj) - 1):
            state, action = traj[t]
            next_state, next_action = traj[t + 1]

            if isinstance(state, (int, np.int, np.int64)):
                state = [state]
                next_state = [next_state]
            if isinstance(action, (int, np.int, np.int64)):
                action = [action]
                next_action = [next_action]

            dataset.setdefault("state", []).append(state)
            dataset.setdefault("action", []).append(action)
            dataset.setdefault("next_state", []).append(next_state)
            dataset.setdefault("next_action", []).append(next_action)

    for k, v in dataset.items():
        dataset[k] = np.stack(dataset[k])
    print("Number of transitions in the processed dataset: {}".format(len(dataset["state"])))

    return traj_dataset, dataset
