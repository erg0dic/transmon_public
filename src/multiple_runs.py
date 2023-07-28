from src.ppo import PPO, CNOT, MLPActorCritic
import secrets, glob, json, os
import torch
import numpy as np
from torch.optim import Adam
from typing import List
from tqdm import tqdm
import re
from src.utilities import add_args

args = add_args()


def cleanup():
    # clean up leftovers
    cpaths = glob.glob("conts_dicts/*")
    mpaths = glob.glob("models/*")
    for cpath, mpath in zip(cpaths, mpaths):
        x = cpath.split("/")[1].split("_")[1]
        if re.search("[0-9a-zA-Z]{16}|[0-9a-zA-Z]{32}", x):
            os.remove(cpath)
        x = mpath.split("_")[1]
        if re.search("[0-9a-zA-Z]{16}", x):
            os.remove(mpath)


def update_records(records: dict, exp_hash, clean=False):
    # find all the trials/multiple runs of this experiment
    files_conts = glob.glob(f"conts_dicts/conts_{exp_hash}*")
    # sort because glob messes up the order
    files_conts.sort(key=lambda x: int(x.split("__")[1]))

    files_models = glob.glob(f"models/model_{exp_hash}*")
    files_models.sort(key=lambda x: int(x.split("__")[1]))

    # update each run's epoch checkpoint appropriately
    for i, cont_file in enumerate(files_conts):
        record = json.load(open(cont_file, "rb"))
        if i not in records:
            records[i] = {}
            for key in record:
                records[i][key] = record[key]
        else:
            for key in record:
                if key == "controllers":
                    records[i][key].update(record[key])
        if clean:
            os.remove(cont_file)
    return files_models


def update_singulars(
    singulars, models: List[str], reg_loss: List = None, capacity_loss_exp=False
):
    "models is a list of paths to the model checkpoints"
    checkpoint_ = torch.load(models[0])
    _buffer = np.array(checkpoint_["buffer"])  # fix buffer at origin
    inds = np.random.randint(low=0, high=len(_buffer), size=10000)
    buffer = torch.as_tensor(_buffer[inds], dtype=torch.float32)  # augment

    for i, model in enumerate(models):
        checkpoint = torch.load(model)
        x = checkpoint["model_state_dict"]
        # breakpoint()
        n1 = MLPActorCritic(buffer.shape[-1], 2, hidden_sizes=(100, 100, 100))
        n1.load_state_dict(x)
        n1.eval()

        # remove the final layer
        n1.pi.mu_net = n1.pi.mu_net[:5]
        n1.v.v_net = n1.v.v_net[:5]

        # forward pass through the networks
        features_pi = n1.pi.mu_net(buffer).detach().numpy()
        features_vf = n1.v.v_net(buffer).detach().numpy()

        # compute the effective dimension using SVD
        _, s, _ = np.linalg.svd(features_pi)
        _, s2, _ = np.linalg.svd(features_vf)
        if i not in singulars:
            singulars[i] = {}
            singulars[i]["pi"] = [s.tolist()]
            singulars[i]["vf"] = [s2.tolist()]
        else:
            singulars[i]["pi"] += [s.tolist()]
            singulars[i]["vf"] += [s2.tolist()]

        # add regression target evalution using randomly initialized neural network outputs
        if capacity_loss_exp:
            loss_aux_pi = np.zeros(10)
            loss_aux_vf = np.zeros(10)
            # breakpoint()
            for i in range(5):
                # load checkpointed model
                n1 = MLPActorCritic(buffer.shape[-1], 2, hidden_sizes=(100, 100, 100))
                n1.load_state_dict(x)
                n1_pi_optimizer = Adam(n1.pi.parameters(), lr=3e-3)
                n1_vf_optimizer = Adam(n1.v.parameters(), lr=8e-4)
                n1_vf_optimizer.load_state_dict(checkpoint["vf_optimizer_state_dict"])
                n1_pi_optimizer.load_state_dict(checkpoint["pi_optimizer_state_dict"])
                n1.pi.mu_net = n1.pi.mu_net[:5]
                n1.v.v_net = n1.v.v_net[:5]
                n1.train()
                # init a random network
                n2 = MLPActorCritic(buffer.shape[-1], 2, hidden_sizes=(100, 100, 100))
                n2.pi.mu_net = n2.pi.mu_net[:5]
                n2.v.v_net = n2.v.v_net[:5]
                reg_out_pi = n2.pi.mu_net(buffer).detach()  # no need to backprop here
                reg_out_vf = n2.v.v_net(buffer).detach()
                # try to backprop fit random network's output
                for s in range(100):
                    n1_pi_optimizer.zero_grad()
                    n1_vf_optimizer.zero_grad()
                    reg_in_pi = n1.pi.mu_net(buffer)
                    reg_in_vf = n1.v.v_net(buffer)
                    loss_pi = ((reg_in_pi - reg_out_pi) ** 2).mean()
                    loss_vf = ((reg_in_vf - reg_out_vf) ** 2).mean()
                    loss_pi.backward()
                    loss_vf.backward()
                    n1_pi_optimizer.step()
                    n1_vf_optimizer.step()
                # store the loss for the final iteration
                loss_aux_pi[i] = loss_pi.detach().numpy()
                loss_aux_vf[i] = loss_vf.detach().numpy()
            # store mean loss over different random networks i.e. capacity loss
            reg_loss.append([loss_aux_pi.mean(), loss_aux_vf.mean()])

        os.remove(model)


def run_pretraining_exp_multiple_times(
    reps: int = 10, epoch_save_freq=10, target=CNOT, args=args
):
    pbar = tqdm(total=reps * 3)
    records = {}
    records2 = {}
    records3 = {}
    singulars = {}
    singulars2 = {}
    singulars3 = {}
    reg_loss1 = []
    reg_loss2 = []
    reg_loss3 = []

    do_capacity_loss_exp = True if args.extension == "" else False

    # for safety
    if args.epochs < epoch_save_freq:
        epoch_save_freq = args.epochs
    for rep in range(reps):
        # use internal save but delete files immediately after merging with a global
        exp_hash = secrets.token_hex(nbytes=16)
        # no noise pre-train
        trial = PPO(
            target=target,
            trl=2,
            num_timesteps=20,
            verbose=args.verbose,
            epochs=args.epochs,
            dissipate=True,
            decay_1=0,
            decay_2=0,
            epoch_checkpoint_rate=epoch_save_freq,
            save_topc=args.save_topc,
            experiment_name=exp_hash,
            buffer_size=args.buffer_size,
        )
        trial.run()
        pbar.update(1)
        files_models = update_records(records, exp_hash)
        assert len(files_models) == args.epochs // epoch_save_freq

        # noisy pretrain test
        pretrain_path = files_models[-1][13:-4]
        # breakpoint()
        exp_hash = secrets.token_hex(nbytes=16)
        trial = PPO(
            target=target,
            trl=2,
            num_timesteps=20,
            verbose=args.verbose,
            epochs=args.epochs,
            dissipate=True,
            decay_1=-0.14,
            decay_2=-0.14,
            saved_model_path=pretrain_path,
            epoch_checkpoint_rate=epoch_save_freq,
            experiment_name=exp_hash,
        )
        trial.run()
        pbar.update(1)

        update_singulars(singulars, files_models, reg_loss1, do_capacity_loss_exp)

        files_models2 = update_records(records2, exp_hash)
        assert len(files_models2) == args.epochs // epoch_save_freq
        update_singulars(singulars2, files_models2, reg_loss2, do_capacity_loss_exp)

        # noisy but not pretrained
        exp_hash = secrets.token_hex(nbytes=16)
        trial = PPO(
            target=target,
            trl=2,
            num_timesteps=20,
            verbose=args.verbose,
            epochs=args.epochs,
            dissipate=True,
            decay_1=-0.14,
            decay_2=-0.14,
            saved_model_path=None,
            epoch_checkpoint_rate=epoch_save_freq,
            experiment_name=exp_hash,
        )
        trial.run()
        pbar.update(1)

        files_models3 = update_records(records3, exp_hash)
        assert len(files_models3) == args.epochs // epoch_save_freq
        update_singulars(singulars3, files_models3, reg_loss3, do_capacity_loss_exp)

        records_ = [records, records2, records3]
        singulars_ = [singulars, singulars2, singulars3]
        reg_losses = [reg_loss1, reg_loss2, reg_loss3]
        exp_types = ["pretrain", "pretrain_test", "from_scratch_lind"]
        save(records_, singulars_, reg_losses, exp_types)
        # print(f"saved {rep}")

    # os.system("git add .")
    # os.system('git commit -m "data update"')
    # os.system("git push")


def save(records_, singulars_, reg_losses, exp_types):
    for recordsi, singularsi, reg_lossi, exp_type in zip(
        records_, singulars_, reg_losses, exp_types
    ):
        json.dump(
            recordsi, open(f"plot_data/conts_{exp_type}{args.extension}.json", "w")
        )
        json.dump(
            singularsi,
            open(f"plot_data/singulars_{exp_type}{args.extension}.json", "w"),
        )
        json.dump(
            reg_lossi, open(f"plot_data/regloss_{exp_type}{args.extension}.json", "w")
        )


# run_pretraining_exp_multiple_times(reps=args.reps)
# cleanup()


def run_finite_meas_exp(reps: int = 10, epoch_save_freq=20, target=CNOT, args=args):
    Ms = [1, 2, 5, 10, 50]
    pbar = tqdm(total=reps * len(Ms))
    allrecords = {}
    allsingulars = {}
    for M in Ms:
        records = {}
        singulars = {}
        # for safety
        if args.epochs < epoch_save_freq:
            epoch_save_freq = args.epochs
        for rep in range(reps):
            # use internal save but delete files immediately after merging with a global
            exp_hash = secrets.token_hex(nbytes=16)
            # no noise pre-train
            trial = PPO(
                target=target,
                trl=2,
                num_timesteps=20,
                verbose=args.verbose,
                epochs=args.epochs,
                dissipate=True,
                decay_1=0,
                decay_2=0,
                epoch_checkpoint_rate=epoch_save_freq,
                save_topc=args.save_topc,
                experiment_name=exp_hash,
                buffer_size=args.buffer_size,
                M=M,
            )
            trial.run()
            pbar.update(1)
            model_files = update_records(records, exp_hash)
            for model in model_files:
                os.remove(model)
            # update_singulars(singulars, model_files)
        allrecords[M] = records
        # allsingulars[M] = singulars
    json.dump(
        allrecords,
        open(f"plot_data/conts_finite_meas_unitary_{args.extension}.json", "w"),
    )
    # json.dump(allsingulars, open(f"models/models_finite_meas_unitary_{args.extension}.json", "w"))


# run_finite_meas_exp(reps=args.reps)


def run_fourier_exp(
    reps: int = 10, epoch_save_freq=20, target=CNOT, args=args, clean=True
):
    stds = [0.01, 0.1, 1, "baseline"]
    pbar = tqdm(total=reps * len(stds))
    allrecords = {}
    allsingulars = {}
    for std in stds:
        records = {}
        singulars = {}
        # for safety
        if args.epochs < epoch_save_freq:
            epoch_save_freq = args.epochs
        for rep in range(reps):
            # use internal save but delete files immediately after merging with a global
            exp_hash = secrets.token_hex(nbytes=16)
            # baseline without any fourier rep
            if std == "baseline":
                trial = PPO(
                    target=target,
                    trl=2,
                    num_timesteps=20,
                    verbose=args.verbose,
                    epochs=args.epochs,
                    dissipate=True,
                    decay_1=0,
                    decay_2=0,
                    epoch_checkpoint_rate=epoch_save_freq,
                    save_topc=args.save_topc,
                    experiment_name=exp_hash,
                    buffer_size=args.buffer_size,
                    use_learned_fourier=False,
                    fourier_std=std,
                    concat_fourier=False,
                )
            else:
                trial = PPO(
                    target=target,
                    trl=2,
                    num_timesteps=20,
                    verbose=args.verbose,
                    epochs=args.epochs,
                    dissipate=True,
                    decay_1=0,
                    decay_2=0,
                    epoch_checkpoint_rate=epoch_save_freq,
                    save_topc=args.save_topc,
                    experiment_name=exp_hash,
                    buffer_size=args.buffer_size,
                    use_learned_fourier=True,
                    fourier_std=std,
                    concat_fourier=True,
                )

            trial.run()
            pbar.update(1)
            model_files = update_records(records, exp_hash, clean)
            for model in model_files:
                os.remove(model)
            # update_singulars(singulars, model_files)
        allrecords[std] = records
        # allsingulars[M] = singulars
    json.dump(allrecords, open(f"plot_data/conts_fourier_{args.extension}.json", "w"))


# run_fourier_exp(reps=args.reps)


def single_run_exp(
    reps: int = 10,
    epoch_save_freq=1,
    target=CNOT,
    args=args,
    clean=True,
    ALGO: PPO = PPO,
):
    allrecords = {}
    pbar = tqdm(total=reps)
    if args.epochs < epoch_save_freq:
        epoch_save_freq = args.epochs
    for rep in range(reps):
        # use internal save but delete files immediately after merging with a global
        exp_hash = secrets.token_hex(nbytes=16)
        # baseline without any fourier rep
        trial = ALGO(
            target=target,
            trl=2,
            num_timesteps=20,
            verbose=args.verbose,
            epochs=args.epochs,
            dissipate=False,
            decay_1=0,
            decay_2=0,
            epoch_checkpoint_rate=epoch_save_freq,
            save_topc=args.save_topc,
            experiment_name=exp_hash,
            use_learned_fourier=False,
            concat_fourier=False,
        )

        trial.run()
        pbar.update(1)
        model_files = update_records(allrecords, exp_hash, clean)
        for model in model_files:
            os.remove(model)
        # update_singulars(singulars, model_files)
    # allsingulars[M] = singulars
    json.dump(allrecords, open(f"plot_data/single_exp_{args.extension}.json", "w"))


from tune_hypers import parse_algo

single_run_exp(reps=args.reps, ALGO=parse_algo(args.algo))
