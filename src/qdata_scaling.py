from src.mbsac import *
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from src.non_para_dynamics import make_holdout_dataset
from typing import Any
from src.utilities import save_fig

KWARGS = dict(
    learn_diss=False,
    learn_diss_coeffs_only=False,
    learn_H_cont=False,
    static_lind=False,
    use_ansatz_model=True,
    use_model=True,
)


def load(fname: str) -> Any:
    return pickle.load(open(fname, "rb"))


def save(fname: str, obj: Any) -> None:
    pickle.dump(obj, open(fname, "wb"))


class LearnableHamTrainer(MBSAC):
    def __init__(
        self,
        num_full_trajectories,
        num_timesteps=20,
        data_train_epochs=2,
        use_shots=False,
        M=int(pow(10, 7)),
        override_ham_init=True,
        checkpoint=True,
        imperfection_delta=0.2,
        **extra,
    ):
        # np.random.seed(1) # make sure the random ham is always the same
        self.num_timesteps = 20
        buffer_size = self.set_dataset_size(num_full_trajectories)
        super().__init__(
            **KWARGS,
            buffer_size=buffer_size,
            use_shots_to_recon_state=use_shots,
            M=M,
            override_ham_init=override_ham_init,
            imperfection_delta=imperfection_delta,
            **extra,
        )
        # make dataset
        self.exploration_before_start()
        self.checkpoint = checkpoint
        if not os.path.exists("datascaling"):
            os.mkdir("datascaling")
        self.save_name = None
        if checkpoint:
            self.setup_savename(
                buffer_size,
                use_shots,
            )
        self.data_train_epochs = data_train_epochs

    def set_dataset_size(self, num_full_trajectories):
        buffer_size = self.num_timesteps * num_full_trajectories
        return buffer_size

    def setup_savename(self, buffer_size, use_shots):
        self.save_name = f"datascaling/datascaling_exp_size_{buffer_size}"
        if use_shots:
            self.save_name += f"_shots_{use_shots}"

    def train_learnable_ham_model(
        self, custom_holdout_dataset=None, use_full_dataset=False
    ):
        if use_full_dataset:
            self.save_name += "full_data"
        if self.save_name and os.path.exists(self.save_name):
            return load(self.save_name)
        state, action, reward, next_state, _, _ = self.env_pool.sample(
            len(self.env_pool)
        )
        batch_size = (
            256  # if len(self.env_pool) > 256*1/0.2 else int(0.2*(len(self.env_pool)))
        )
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate(
            (np.reshape(reward, (reward.shape[0], -1)), next_state), axis=-1
        )
        train_losses, holdout_losses, other_losses = self.env_model.train(
            inputs,
            labels,
            batch_size=batch_size,
            holdout_ratio=0.2,
            data_scaling_exp=True,
            epochs=self.data_train_epochs,
            custom_holdouts=custom_holdout_dataset,
            use_full_dataset=use_full_dataset,
        )
        if self.checkpoint:
            save(self.save_name, (train_losses, holdout_losses, other_losses))
        return train_losses, holdout_losses, other_losses


def gen_holdout_dataset(size=2000, use_shots=False):
    fname = f"datascaling/holdout_dataset_size_{size}"
    if use_shots:
        fname += f"_shots_{use_shots}"
    if os.path.exists(fname):
        return load(fname)
    og_model = MBSAC(**KWARGS, buffer_size=size, use_shots_to_recon_state=use_shots)
    # updates the env_pool object where our dataset resides
    og_model.exploration_before_start()
    inputs, labels = og_model.get_sup_learning_ingredients()
    pickleable = make_holdout_dataset(inputs, labels, network_size=1)
    save(fname, pickleable)
    return pickleable


def get_holdout_err_vs_shots(
    shot_range,
    imperfection_delta=0.0,
    fig=None,
    ax=None,
    fname_prefix="",
    figlabel=None,
):
    fname = (
        fname_prefix
        + f"datascaling/shots_losses_imperfection_delta_{imperfection_delta}_ss{shot_range}"
    )
    fsize = 12

    if os.path.exists(fname):
        trains, holdouts = load(fname)
    else:
        trains, holdouts = [], []
        for shots in shot_range:
            runner = LearnableHamTrainer(
                num_full_trajectories=100,
                M=shots,
                override_ham_init=False,
                data_train_epochs=1,
                checkpoint=False,
                use_ruthless_delta=True,
                use_shots=True,
                imperfection_delta=imperfection_delta,
            )
            train_losses, holdout_losses, _ = runner.train_learnable_ham_model()
            train_losses = train_losses[0].tolist()
            trains.append(train_losses)
            holdouts.extend(holdout_losses)
            save(fname, (trains, holdouts))
    if fig is None:
        fig, ax = plt.subplots()

    ax.set_ylim(1e-3, int(1e6))
    ax.loglog(shot_range, trains, label="train loss", marker="o")
    ax.loglog(shot_range, holdouts, label="holdout loss", marker="o")
    ax.set_xlabel(r"$M$", fontsize=fsize)
    ax.set_ylabel("Loss", fontsize=fsize)
    ax.set_title(figlabel + r" $\delta$=" + f"{imperfection_delta}", fontsize=fsize)
    ax.legend(fontsize=fsize)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    # plt.show()
    fig.savefig("plots/" + fname.split("/")[-1] + ".pdf", dpi=1000)
    # return (trains, holdouts)
    return fig, ax


def get_holdout_err_vs_deltas(
    delta_range, fig=None, ax=None, fname_prefix="", train_epochs=50
):
    fname = (
        fname_prefix
        + f"datascaling/delta_losses_imperfection_delta_{delta_range}_ss_1e6_test"
    )
    fsize = 12

    if os.path.exists(fname):
        trains, holdouts = load(fname)
    else:
        trains, holdouts = [], []
        for delta in delta_range:
            runner = LearnableHamTrainer(
                num_full_trajectories=17,
                M=int(pow(10, 7)),
                override_ham_init=False,
                checkpoint=False,
                use_ruthless_delta=True,
                use_shots=False,
                imperfection_delta=delta,
                data_train_epochs=train_epochs,
            )
            train_losses, holdout_losses, _ = runner.train_learnable_ham_model()
            train_losses = train_losses[0].tolist()
            trains.append(train_losses)
            holdouts.extend(holdout_losses)
            save(fname, (trains, holdouts))
    if fig is None:
        fig, ax = plt.subplots()

    ax.set_ylim(1e-3, int(1e6))
    ax.loglog(delta_range, trains, label="train loss", marker="o")
    ax.loglog(delta_range, holdouts, label="holdout loss", marker="o")
    ax.set_xlabel(r"$\delta$", fontsize=fsize)
    ax.set_ylabel("Loss", fontsize=fsize)
    ax.legend(fontsize=fsize)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    # plt.show()
    fig.savefig("plots/" + fname.split("/")[-1] + ".pdf", dpi=1000)
    # return (trains, holdouts)
    return fig, ax


def run_data_exp_scaling_exps(
    datasizes: list, epochs=2, use_shots=False, use_full_dataset=False
):
    if len(datasizes) % 4 != 0:
        fig1, ax = plt.subplots(nrows=1, ncols=len(datasizes), figsize=(20, 10))
        import warnings

        warnings.warn(
            f"Need a multiple of 4 length datasizes not {len(datasizes)} for optimal plotting experience."
        )
    else:
        fig1, ax = plt.subplots(nrows=len(datasizes) // 4, ncols=4, figsize=(15, 10))
    ax = ax.ravel()
    final_train_losses = []
    final_holdout_losses = []
    final_ham_errors = []
    fsize = 20
    custom_holdout_dataset = gen_holdout_dataset(size=2000, use_shots=use_shots)
    for i, datasize in enumerate(datasizes):
        runner = LearnableHamTrainer(
            datasize, data_train_epochs=epochs, use_shots=use_shots
        )

        train_losses, holdout_losses, other_losses = runner.train_learnable_ham_model(
            custom_holdout_dataset, use_full_dataset
        )
        train_losses, holdout_losses = np.array(train_losses), np.array(holdout_losses)
        final_train_losses.append(train_losses[-1])
        final_holdout_losses.append(holdout_losses[-1])
        final_ham_errors.append(np.real(other_losses[-1]["ham"][0]))
        ham_errors = [other_losses[i]["ham"] for i in range(epochs)]
        # # TODO normalize here for the plots to look prettier
        # train_losses /= train_losses.sum()
        # holdout_losses /= holdout_losses.sum()
        ax[i].semilogy(range(1, 1 + epochs), train_losses, label="train", linewidth=2)
        ax[i].semilogy(
            range(1, 1 + epochs), holdout_losses, label="holdout", linewidth=2
        )
        ax[i].semilogy(range(1, 1 + epochs), ham_errors, label="H error", linewidth=2)
        ax[i].set_xlabel("epochs", fontsize=fsize)
        ax[i].set_ylabel("Loss", fontsize=fsize)
        ax[i].set_title(f"datasize={datasize*20}", fontsize=fsize)
        # ax[i].set_ylim(int(-1e1), int(1e5)) # TODO limits don't work!
        ax[i].tick_params(axis="both", which="major", labelsize=fsize)
    ax[i].legend(fontsize=fsize)
    fig2, ax = plt.subplots()
    ax3 = ax.twinx()
    ax.plot(np.array(datasizes) * 20, final_train_losses, label="train", marker="o")
    ax.plot(np.array(datasizes) * 20, final_holdout_losses, label="holdout", marker="o")
    ax3.plot(
        np.array(datasizes) * 20,
        final_ham_errors,
        label="final ham error",
        marker="o",
        c="red",
    )
    ax.set_xlabel("datasize")
    ax.set_ylabel(f"final Error after {epochs} training epochs")
    ax3.set_ylabel("H error", c="red")
    ax.legend()
    fig2.tight_layout()
    fig1.tight_layout()

    save_fig(fig1, f"plots/ham_error_and_prop_error.pdf", copyto="../paper/figures")
    save_fig(
        fig2, f"plots/final_ham_error_and_prop_error.pdf", copyto="../paper/figures"
    )
    # ax[i].set_ylim(0,1)
    fig1.tight_layout()


if __name__ == "__main__":
    # get_holdout_err_vs_shots([int(pow(10,i)) for i in [3,4,5,6,7,8,9,10,11,12]], imperfection_delta=0.5)
    # run_data_exp_scaling_exps([1, 2, 10, 100], epochs=200, use_shots=True, use_full_dataset=True)
    # figlabels = ["({})".format(i) for i in "abcdefghijklmnopqrstuvwxyz"]
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    # axs =axs.ravel()
    # i=0
    # for d, ax in zip([0.0, 0.1, 0.2, 0.5], axs):
    #     fig, ax = get_holdout_err_vs_shots([int(pow(10,i)) for i in [3,4,5,6,7,8,9,10,11,12]],
    #                     imperfection_delta=d , ax=ax, fig=fig, figlabel=figlabels[i])
    #     i+=1
    # fig.tight_layout()
    # save_fig(fig, "holdout_err_vs_shots.pdf", copyto="../paper/figures")
    # plt.show()
    get_holdout_err_vs_deltas([0, 0.1, 0.2, 0.5, 0.8, 1, 2, 10, 20], train_epochs=1000)
