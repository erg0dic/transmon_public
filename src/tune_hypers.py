from ast import Assert
from src.ppo import PPO
from src.env_sampler import MBPPO
from src.mbsac import MBSAC
import torch.optim as optim
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
import os
import ray
import numpy as np
import sys
import pandas as pd
from utilities import get_timelog

# num_cpus = int(sys.argv[1])

# ray.init(address="auto", num_cpus=6)

# print("Nodes in the Ray cluster:")
# print(ray.nodes())


def parse_algo(algo: str):
    if algo.lower() == "ppo":
        return PPO
    elif algo.lower() == "mbsac":
        return MBSAC
    else:
        raise Exception(f"Specified algo {algo} in not in the implemented suite.")


def train(config):
    # device = torch.device("cuda" if torch.c uda.is_available() else "cpu")
    ALGO = config.pop("ALGO")
    if ALGO == PPO:
        config_dic = dict(
            hidden=config["hidden"],
            layers=config["layers"],
            # gamma=config["gamma"],
            buffer_size=config["buf_size"],
            lam=config["lamb"],
            clip_ratio=config["clip_ratio"],
            train_pi_iters=config["pi_iter"],
            train_v_iters=config["v_iter"],
        )
        algo = ALGO(
            **config_dic,
            verbose=False,
            dissipate=True,
            decay_1=-0.005,
            decay_2=-0.005,
            epochs=10,
        )
    elif ALGO == MBPPO:
        config_dic = dict(
            hidden=config["hidden"],
            layers=config["layers"],
            # # gamma=config["gamma"],
            # buffer_size=config["buf_size"],
            # lam=config["lamb"],
            clip_ratio=config["clip_ratio"],
            train_pi_iters=config["pi_iter"],
            train_v_iters=config["v_iter"],
            model_train_freq=config["model_train_freq"],
            rollout_batch_size=config["rollout_batch_size"],
        )
        algo = ALGO(
            **config_dic,
            verbose=False,
            dissipate=True,
            decay_1=-0.005,
            decay_2=-0.005,
            epochs=10,
        )

    elif ALGO == MBSAC:
        config_dic = dict(
            # hidden=config["hidden"],
            # layers=config["layers"],
            # # gamma=config["gamma"],
            # buffer_size=config["buf_size"],
            # lam=config["lamb"],
            clip_ratio=config["clip_ratio"],
            model_train_freq=config["model_train_freq"],
            rollout_batch_size=config["rollout_batch_size"],
            automatic_entropy_tuning=config["automatic_entropy_tuning"],
            min_pool_size=config["min_pool_size"],
            rollout_max_epoch=config["rollout_max_epoch"],
            rollout_max_length=config["rollout_max_length"],
            real_ratio=config["real_ratio"],
            network_size=config["network_size"],
            elite_size=config["elite_size"],
            max_path_length=config["max_path_length"],
            lr=config["lr"],
        )
        algo = ALGO(
            **config_dic,
            verbose=False,
            dissipate=True,
            decay_1=-0.005,
            decay_2=-0.005,
            epochs=1,
        )
    else:
        raise NotImplementedError
    # algo.ac.to(device)
    if ALGO != MBSAC:
        pi_optimizer = optim.Adam(algo.ac.pi.parameters(), lr=config["pi_lr"])
        vf_optimizer = optim.Adam(algo.ac.v.parameters(), lr=config["v_lr"])
        optimizer = (pi_optimizer, vf_optimizer)
        for i in range(10):
            acc = algo.run(optimizer=optimizer)
            # Send the current training result back to Tune
            tune.report(
                max_fid_seen=acc,
                pi_lr=config["pi_lr"],
                v_lr=config["v_lr"],
                **config_dic,
            )
            # don't need to save the model: TODO problems with ray paths
            # if i % 5 == 0:
            #     timelog = str(datetime.date(datetime.now())) + "_" + str(
            #                 datetime.time(datetime.now()))
            #     # This saves the model to the trial directory
            #     torch.save(algo.ac.state_dict(), f"{SAVE_DIR}/model.pth")
    else:
        raise NotImplementedError
        if config["automatic_entropy_tuning"]:
            algo.agent.alpha_optim, algo.agent.critic_optim, algo.agent.policy_optim = (
                optim.Adam([algo.agent.log_alpha], lr=config["lr"]),
                optim.Adam(algo.agent.critic.parameters(), lr=config["lr"]),
                optim.Adam(algo.agent.policy.parameters(), lr=config["lr"]),
            )
            optimizer = (
                algo.agent.alpha_optim,
                algo.agent.critic_optim,
                algo.agent.policy_optim,
            )
        else:
            algo.agent.critic_optim, algo.agent.policy_optim = (
                optim.Adam(algo.agent.critic.parameters(), lr=config["lr"]),
                optim.Adam(algo.agent.policy.parameters(), lr=config["lr"]),
            )
            optimizer = (algo.agent.critic_optim, algo.agent.policy_optim)

        for i in range(10):
            acc = algo.run(optimizer=optimizer)
            # Send the current training result back to Tune
            tune.report(max_fid_seen=acc, **config_dic)


def execute_bayesian_hyperparameter_opt(
    savedir, grid_version=False, samples=10, hyper_exp_name="HBOPT", ALGO=PPO
):
    # needs to be modified, as currently only configured with hyperopt
    if ALGO == PPO:
        search_space = {
            "hidden": hp.choice("hidden", [50, 100, 200]),
            "layers": hp.choice("layers", [1, 2, 3, 4]),
            # "gamma": hp.uniform("gamma", 0.95, 1),
            "lamb": hp.loguniform("lamb", np.log(0.9), np.log(0.9999)),
            "clip_ratio": hp.uniform("clip", 0.05, 0.95),
            "pi_iter": hp.loguniform("pi_iter", np.log(10), np.log(1000)),
            "v_iter": hp.loguniform("v_iter", np.log(10), np.log(1000)),
            "v_lr": hp.loguniform("v_lr", np.log(1e-5), np.log(1e-2)),
            "pi_lr": hp.loguniform("pi_lr", np.log(1e-5), np.log(1e-2)),
            "buf_size": hp.choice("buf_size", [100, 1000, 10000, 2000, 5000]),
        }
    elif ALGO == MBPPO:
        search_space = {
            "hidden": hp.choice("hidden", [50, 100, 200]),
            "layers": hp.choice("layers", [1, 2, 3, 4]),
            # "gamma": hp.uniform("gamma", 0.95, 1),
            "clip_ratio": hp.uniform("clip", 0.05, 0.95),
            "pi_iter": hp.loguniform("pi_iter", np.log(10), np.log(1000)),
            "v_iter": hp.loguniform("v_iter", np.log(10), np.log(1000)),
            "v_lr": hp.loguniform("v_lr", np.log(1e-5), np.log(1e-2)),
            "pi_lr": hp.loguniform("pi_lr", np.log(1e-5), np.log(1e-2)),
            "model_train_freq": hp.choice(
                "model_train_freq", [100, 1000, 10000, 20000, 50000]
            ),
            "rollout_batch_size": hp.choice(
                "rollout_batch_size", [100, 1000, 10000, 20000, 50000]
            ),
        }
    elif ALGO == MBSAC:
        search_space = {
            "clip_ratio": hp.uniform("clip", 0.05, 0.95),
            "model_train_freq": hp.choice(
                "model_train_freq", [100, 1000, 10000, 20000, 50000]
            ),
            "rollout_batch_size": hp.choice(
                "rollout_batch_size", [100, 1000, 10000, 20000, 50000]
            ),
            "automatic_entropy_tuning": hp.choice(
                "automatic_entropy_tuning", [False, True]
            ),
            "min_pool_size": hp.choice("min_pool_size", [100, 1000, 10000]),
            "rollout_max_epoch": hp.choice("rollout_max_epoch", [20, 50, 100]),
            "rollout_max_length": hp.choice("rollout_max_length", [15, 30, 100]),
            "real_ratio": hp.choice("real_ratio", [0.01, 0.05, 0.1, 0.5]),
            "network_size": hp.choice("network_size", [1, 2, 4, 7, 10]),
            "elite_size": hp.choice("elite_size", [1, 2, 5, 8]),
            "max_path_length": hp.choice("max_path_length", [100, 1000, 10000]),
            "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
        }
    else:
        raise NotImplementedError

    search_space["ALGO"] = ALGO

    if grid_version:
        # raise Exception("Make sure to modify the search space again using `tune.loguniform` etc.")
        # tune.run(train, fail_fast="raise")
        analysis = tune.run(train, config=search_space)

    else:
        # tune.run(trainit, fail_fast="raise")
        hyperopt_search = HyperOptSearch(
            search_space, metric="max_fid_seen", mode="max"
        )
        analysis = tune.run(
            train,
            num_samples=samples,
            search_alg=hyperopt_search,
            resources_per_trial={"cpu": 5},
        )

    timelog = get_timelog()
    df = pd.concat(list(analysis.trial_dataframes.values()))
    df = df.sort_values("max_fid_seen", ascending=False)
    df.to_csv(savedir + "/" + hyper_exp_name + "_" + timelog + ".csv")


def population_based_training_model_selection(savedir):
    from ray.tune import run, sample_from
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune.schedulers.pb2 import PB2
    import random

    timelog = get_timelog()

    pbt = PopulationBasedTraining(
        time_attr="funccalls",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=50000,
        resample_probability=0.25,
        quantile_fraction=0.25,  # copy bottom % with top %
        # Specifies the search space for these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.1, 0.5),
            "lr": lambda: random.uniform(1e-3, 1e-5),
            "train_batch_size": lambda: random.randint(1000, 60000),
        },
        # custom_explore_fn=explore
    )

    config = {
        "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
        "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
        "train_batch_size": sample_from(lambda spec: random.randint(1000, 60000)),
    }
    algo = PPO()

    analysis = tune.run(
        algo,
        scheduler=pbt,
        verbose=1,
        num_samples=10,
        stop={"funccalls": 100000},
        config=config,
    )

    df = pd.concat(list(analysis.trial_dataframes.values())).reset_index(drop=True)
    df = df.sort_values("max_fid_seen", ascending=False)
    df.to_csv(savedir + "/" + "HBOPT_" + timelog + ".csv")


if __name__ == "__main__":
    SAVE_DIR = "hyper_tests"
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    import argparse

    parser = argparse.ArgumentParser(description="CLA")
    parser.add_argument(
        "--hyper_exp_name",
        type=str,
        default="HBOPT",
        help="hyper_experiment_name",
        required=False,
    )
    parser.add_argument(
        "--debug", type=bool, required=False, default=False, help="debugging flag"
    )

    parser.add_argument(
        "--samples",
        type=int,
        required=False,
        default=10,
        help="hyper opt number of experiment samples",
    )

    parser.add_argument(
        "--algo", type=str, required=False, default="PPO", help="algo to tune"
    )

    args = parser.parse_args()

    if args.debug == True:
        # For debuggin only
        # config = dict(
        #         hidden=3,
        #         layers=100,
        #         clip_ratio=0.2,
        #         pi_iter=44.2,
        #         v_iter=50.3,
        #         model_train_freq = 1000,
        #         rollout_batch_size = 1,
        #         ALGO = MBSAC,
        #         v_lr = 1e-3,
        #         pi_lr=1e-3,
        # )
        config = dict(
            clip_ratio=0.2,
            model_train_freq=1,
            rollout_batch_size=100,
            automatic_entropy_tuning=False,
            min_pool_size=100,
            rollout_max_epoch=100,
            rollout_max_length=1000,
            real_ratio=0.04,
            network_size=10,
            elite_size=10,
            max_path_length=100,
            ALGO=MBSAC,
            lr=3e-4,
        )
        train(config)

    else:
        from parser import parse_algo

        execute_bayesian_hyperparameter_opt(
            SAVE_DIR,
            hyper_exp_name=args.hyper_exp_name,
            ALGO=parse_algo(args.algo),
            samples=args.samples,
        )
