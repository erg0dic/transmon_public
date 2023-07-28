"""
A general script to parse the experimental data in `conts_dicts` with identifiers and plot the results 
"""
import glob
import re
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import json
from itertools import zip_longest, product
from src.utilities import (
    DidntFindAnyFiles,
    save_fig,
    get_dissipation_discrimating_fids,
    save,
    load,
)
from src.str_to_qsys import str_to_qsys
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm


def typify(l: List, conv_to_type=int) -> List:
    for i in range(len(l)):
        try:
            l[i] = conv_to_type(l[i])
        except:
            try:
                l[i] = float(l[i])
            except:
                pass
    return l


def get_hypernames(
    exp_name_str: str, regex_tag: str = None, return_dict=True
) -> Union[Dict[str, int], List[Tuple[str, int]]]:
    strlist = re.findall(regex_tag, exp_name_str)
    if strlist:
        if isinstance(strlist[0], tuple):
            strlist = list(
                map(lambda x: x[0], strlist)
            )  # ignore the second group found by re
    else:
        warnings.warn(
            "Don't try an array plot since the hyperparameters strings weren't regex parsed."
        )
    n_and_vals = list(map(lambda x: typify(x.split("_"), conv_to_type=int), strlist))
    if return_dict:
        n_v_dict = {}
        for i in range(len(n_and_vals)):
            n_v_dict[n_and_vals[i][0]] = n_and_vals[i][1]
        return n_v_dict
    return n_and_vals


def list2list(list1_element: Union[Any, List, np.ndarray], list2: List) -> List:
    if not isinstance(list1_element, list) and not isinstance(
        list1_element, np.ndarray
    ):
        return list2[list1_element]
    else:
        return list(map(lambda x: list2[x], list1_element))


def get_hyp_exp_marker(exp_dict: Dict) -> str:
    hyp_exp_marker = ""
    for key in exp_dict:
        if key != "seed":
            hyp_exp_marker += f"{key}-{exp_dict[key]}_"
    return hyp_exp_marker


def get_hyp_exp_marker_all(unique_hypers_dict: Dict) -> str:
    all_hyp_exp_markers = []
    hyper_params = list(unique_hypers_dict.keys())
    hyper_param_vals = list(unique_hypers_dict.values())
    # arbitrary depth for loop!
    hp_enumeration = list(product(*hyper_param_vals))
    for val_tuple in hp_enumeration:
        hyp_exp_marker = ""
        for i in range(len(val_tuple)):
            hyp_exp_marker += f"{hyper_params[i]}-{val_tuple[i]}_"
        all_hyp_exp_markers.append(hyp_exp_marker)
    return all_hyp_exp_markers


class ExpResults:
    def __init__(
        self,
        file_marker: str,
        max_epochs: int = 20,
        filter_seeds: List = [],
        save_dir="conts_dicts",
    ):
        self.figlabels = ["({})".format(i) for i in "abcdefghijklmnopqrstuvwxyz"]
        self.exp_identifier = file_marker
        self.save_dir = save_dir
        self.files_ = self.filter_f()
        if filter_seeds:
            ffiles = []
            for seed in filter_seeds:
                ffiles += self.filter_f(f"seed_{seed}")
            self.files_ = ffiles
        self.files_ = self.remove_higher_epochs(max_epochs)

        if len(self.files_) == 0:
            raise DidntFindAnyFiles(
                f"No files with the tag {self.exp_identifier} were found."
            )
        # remove any parent directory appendages (in other words, we only want the .json files)
        self.fnames = list(map(lambda x: x.split("/")[-1], self.files_))

        self.epoch_counts = list(map(self.get_epoch, self.fnames))

        # this is a universal experiment tag (a hyperparameter label followed by its value)
        self.regex_tag = "([a-zA-Z]+_\d+(\.\d+)?)"  # "([a-zA-Z]+_\d+)"
        # make sure all hyper dicts are homogeneous
        self.hypers_dicts = list(
            map(
                lambda x: get_hypernames(x, self.regex_tag, return_dict=True),
                self.fnames,
            )
        )
        self.unique_hypers_dict = self.get_unique_hypers_dict()
        self.sorted_markers = get_hyp_exp_marker_all(self.unique_hypers_dict)
        # add epoc
        # for i in range(len(self.epoch_counts)):
        #     self.hypers_dicts[i]["epochs"] = self.epoch_counts[i]
        self.check_files_of_interest()

    def filter_f(self, expression: str = ""):
        nexp = lambda x: expression in x and self.exp_identifier in x
        return list(filter(nexp, glob.glob(f"{self.save_dir}/*")))

    def remove_higher_epochs(self, max_epoch: Union[int, str]) -> List:
        filter_expr = lambda fname: self.get_epoch(fname) <= max_epoch
        return list(filter(filter_expr, self.files_))

    def get_epoch(self, fname: str):
        return int(fname.split("__")[1])

    def get_unique_hypers_dict(self):
        unique_hypers = {}
        for d in self.hypers_dicts:
            for key in d:
                if key != "seed":
                    if key not in unique_hypers:
                        unique_hypers[key] = [d[key]]
                    elif d[key] not in unique_hypers[key]:
                        unique_hypers[key].append(d[key])
        for hyper_param in unique_hypers:
            unique_hypers[hyper_param].sort(reverse=False)
        return unique_hypers

    def group_all_seeds(self):
        "group experiment id by seed number"
        seed_groups = {}
        for expi in range(len(self.hypers_dicts)):
            exp_dict = self.hypers_dicts[expi]
            hyp_exp_marker = get_hyp_exp_marker(exp_dict)
            if hyp_exp_marker not in seed_groups:
                seed_groups[hyp_exp_marker] = [expi]
            else:
                seed_groups[hyp_exp_marker].append(expi)
        return seed_groups

    def check_files_of_interest(self):
        try:
            keys_list = list(map(lambda x: list(x.keys()), self.hypers_dicts))
            for i in range(len(keys_list)):
                for j in range(len(keys_list)):
                    for k in range(len(keys_list[i])):
                        if keys_list[i][k] != keys_list[j][k]:
                            raise AttributeError(
                                f"Terminating the plotting protocol! Hyper exp name {keys_list[i][k]} != {keys_list[j][k]}"
                            )
        except Exception as e:
            print(
                "File hyperparameter extraction failed!. Here is the internal error. \n"
            )
            # raise e

    @staticmethod
    def check_needs_padding(fid_array, num_conts):
        # helper
        if num_conts != len(fid_array):
            return True
        else:
            return False

    def plot_results(
        self,
        epoch_length: int = 2000,
        figsize: Tuple = (20, 10),
        two_d=None,
        pltshow=False,
        should_save_fig=True,
        fig=None,
        ax=None,
        plot_save_name=None,
        plterr=False,
        reduce_op=np.mean,
        legend: str = None,
        plt_sup_labels=True,
        add_title=False,
        remove_maxes=False,
        reshape_ax=False,
        add_title_for_explore_exploit=False,
        plot_external_legend=True,
        return_x_y=False,
        get_dnorm_fids=False,
        System="Transmon",
        ylim=1e-3,
        **plotting_kwargs,
    ):
        x, y, y_upper, y_lower = [], [], [], []
        grouped_seeds = self.group_all_seeds()
        if two_d is None:
            two_d = True if len(self.unique_hypers_dict) <= 2 else False
        if two_d:
            pltdims = []
            for key in self.unique_hypers_dict:
                pltdims.append(len(self.unique_hypers_dict[key]))
            if len(pltdims) == 1:
                pltdims.insert(0, 1)
            if fig is None and ax is None:
                fig, ax = plt.subplots(
                    nrows=pltdims[0], ncols=pltdims[1], figsize=figsize
                )
                ax = ax.ravel()
        else:
            if fig is None and ax is None and not return_x_y:
                fig, ax = plt.subplots(figsize=figsize)
            if not isinstance(ax, tuple) and not isinstance(ax, list):
                ax = [ax]
        plti = 0
        for seed_group in self.sorted_markers:
            exp_ids = np.array(grouped_seeds[seed_group])
            epochs_ = np.array(list2list(exp_ids, self.epoch_counts))
            epoch_order = np.argsort(epochs_)
            sorted_exp_ids = exp_ids[epoch_order]
            epochs = epochs_[epoch_order]
            prev_epoch = epochs[0]
            fid_array = None
            num_conts = None
            fid_array_allseeds = []
            needs_padding = False
            pbar = tqdm(total=len(epochs), leave=False)
            for exp_id, epoch in zip(sorted_exp_ids, epochs):
                try:
                    load_dict = json.load(open(self.files_[exp_id], "rb"))
                except:
                    continue
                # possibly manipulate `load_dict` and produce some scalars here later
                if get_dnorm_fids:
                    system, target, *extra = str_to_qsys(System)
                    # save time by caching 😼
                    cache = f"caches/cache_{self.exp_identifier}_{exp_id}.pkl"
                    # unnecessary vestige
                    if "high_diss" in cache:
                        cache += "_high_diss"
                    if not os.path.exists(cache):
                        if "low_diss" in cache:
                            decay_1, decay_2 = 0.1, 0.1
                        else:
                            decay_1, decay_2 = 0.5, 0.5
                        fids = get_dissipation_discrimating_fids(
                            list(load_dict["controllers"].values()),
                            system,
                            target,
                            decay_1=decay_1,
                            decay_2=decay_2,
                        )
                        save(fids, cache)
                    else:
                        fids = load(cache)
                else:
                    fids = np.array(
                        list(load_dict["controllers"].keys()), dtype=np.float32
                    )
                # plotting fidelities right now
                # iter 0 case
                if fid_array is None:
                    fid_array = fids
                elif epoch == prev_epoch:
                    fid_array = np.append(fid_array, fids)
                elif epoch != prev_epoch:
                    fid_array_allseeds.append(fid_array)
                    if num_conts is None:
                        num_conts = len(fid_array)
                    needs_padding = self.check_needs_padding(fid_array, num_conts)
                    fid_array = fids
                    prev_epoch = epoch
                # terminal iter case
                if exp_id == sorted_exp_ids[-1]:
                    fid_array_allseeds.append(fid_array)
                    needs_padding = self.check_needs_padding(fid_array, num_conts)
                pbar.update(1)
            # get max seed number
            try:
                max_seed = len(
                    list(set(map(lambda x: self.hypers_dicts[x]["seed"], exp_ids)))
                )
            except KeyError:
                max_seed = None
            # shape: (epochs, seeds)
            # try to convert to array and if needed convert with padded zeros
            if needs_padding:
                fid_array_allseeds = np.array(
                    list(zip_longest(*fid_array_allseeds, fillvalue=0))
                ).T
            else:
                fid_array_allseeds = np.array(fid_array_allseeds)

            means = reduce_op(fid_array_allseeds, axis=-1)
            try:
                maxes = np.max(fid_array_allseeds, axis=-1)
            except:
                print(
                    "oops, unexpected expected behavior in processing fids in ExpResults"
                )
                breakpoint()
            x_axis = (
                epoch_length * 2 * np.arange(fid_array_allseeds.shape[0]) + epoch_length
            ) / 1000
            std = fid_array_allseeds.std(axis=-1)
            if return_x_y:
                x.append(x_axis)
                y.append(means)
                y_upper.append(means + std)
                y_lower.append(means - std)
                continue
            ax[plti].tick_params(axis="both", which="major", labelsize=17)
            # x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=True)
            # x_formatter.set_powerlimits((-1,3))
            # ax[plti].get_xaxis().set_major_formatter(x_formatter)
            ax[plti].semilogy(
                x_axis,
                1 - means,
                label=self.make_legend(legend, exp_id, two_d, max_seed),
                linewidth=3,
                **plotting_kwargs,
            )
            color = ax[plti].get_lines()[-1].get_color()
            label = "max fid" if legend == "no train unitary " + r"$M=\infty$" else None
            if not remove_maxes:
                ax[plti].semilogy(
                    x_axis,
                    1 - maxes,
                    label=None,
                    #  "max fid" if legend == "no train unitary "+r"$M=\infty$" else None,
                    linestyle="--",
                    linewidth=3,
                    color=color,
                    **plotting_kwargs,
                )

            ax[plti].set_ylim(ylim, 1)
            if add_title:
                # TODO hack (maybe a waste of time if there isn't much variety to the figures anyhow!)
                if add_title_for_explore_exploit:
                    ax[plti].set_title(
                        self.figlabels[plti]
                        + " "
                        + self.make_title(self.hypers_dicts[exp_id], two_d),
                        fontsize=15,
                    )
                else:

                    ax[plti].set_title(
                        self.figlabels[plti]
                        + " "
                        + r"$\delta=$"
                        + str(self.hypers_dicts[exp_id]["delta"]),
                        fontsize=20,
                    )

            # xticks = ['{:,.2f}'.format(x) + r'x$10^3$' for x in np.round(ax[plti].get_xticks(), 1)/1000]
            # ax[plti].set_xticklabels(xticks)
            # ax[plti].set_xscale('log')
            if plti == 0:
                # ax[plti].legend(loc=4)
                # if not add_title_for_explore_exploit: # hack (no legend for this plot)
                if plot_external_legend:
                    ax[plti].legend(
                        loc="upper center",
                        bbox_to_anchor=(2.7, 1.2),
                        fancybox=True,
                        shadow=True,
                        ncol=4,
                        fontsize=23,
                    )
            if plterr:
                q025 = np.quantile(fid_array_allseeds, 0.025, axis=-1)
                q975 = np.quantile(fid_array_allseeds, 0.975, axis=-1)
                ax[plti].fill_between(
                    x_axis,
                    # np.clip(means-q025, 0.001, 1),
                    # means+q975,
                    np.clip(1 - (means + std), 0.001, 1),
                    1 - (means - std),
                    alpha=0.4,
                )
            if two_d:
                plti += 1
        if return_x_y:
            return x, y, y_upper, y_lower
        # plt.legend()
        if plt_sup_labels:
            fig.text(
                0.01,
                0.55,
                reduce_op.__name__ + " fid",
                va="center",
                rotation="vertical",
                fontsize=15,
            )
            fig.text(0.5, 0.04, r"$\mathcal{E}$ calls", va="center", fontsize=15)
        if should_save_fig:
            if not os.path.exists("plots"):
                os.mkdir("plots")
            if not plot_save_name:
                plot_save_name = self.exp_identifier
            save_fname = "plots/" + plot_save_name + ".pdf"
            save_fig(fig, save_fname)
            if pltshow:
                plt.show()
        if reshape_ax:
            ax = ax.reshape(pltdims)
        return fig, ax

    def make_legend(self, legend, exp_id, two_d, max_seed):
        if legend:
            label = legend
        else:
            label = (
                self.make_title(self.hypers_dicts[exp_id], two_d) + f" seeds {max_seed}"
            )
        return label

    def make_title(self, exp_dict: Dict, two_d=False) -> str:
        if not two_d:
            return self.exp_identifier
        out = ""
        for hyp in exp_dict:
            if hyp != "seed":
                out += f"{hyp}={exp_dict[hyp]} "
        return out
