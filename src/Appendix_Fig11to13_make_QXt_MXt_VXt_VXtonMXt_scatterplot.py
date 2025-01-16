import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from utils import get_model_names

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def save_plot(layer_idx2token_stats, model_name):
    layer_idxs = sorted(list(layer_idx2token_stats.keys()))

    n = len(layer_idxs) // 2
    layer_idxs = [
        layer_idxs[0],
        layer_idxs[n // 2],
        layer_idxs[n],
        layer_idxs[n + n // 2],
        layer_idxs[-1],
    ]
    fig, axes = plt.subplots(
        len(layer_idxs), 4, figsize=(20, int(len(layer_idxs) * 2.5))
    )

    def format_func(value, tick_number):
        return f"{value:.2f}"

    right_ls = 25
    ls = 18
    tick_s = 15
    lw = 2
    legend_s = 9
    for lx, layer_idx in tqdm(enumerate(layer_idxs)):
        n_t_list = layer_idx2token_stats[layer_idx]["n_t"]
        Q_X_t_list = layer_idx2token_stats[layer_idx]["Q_X_t"]
        M_X_t_list = layer_idx2token_stats[layer_idx]["M_X_t"]
        V_X_t_list = layer_idx2token_stats[layer_idx]["V_X_t"]

        log10_n_t_list = np.log10(n_t_list)
        Q_X_t_list = np.array(Q_X_t_list)
        M_X_t_list = np.array(M_X_t_list)
        V_X_t_list = np.array(V_X_t_list)

        data = [
            ("orange", r"$\log_{10}(1+Q(X_t))$", np.log10(Q_X_t_list + 1)),
            ("green", r"$\log_{10}(1+M(X_t))$", np.log10(M_X_t_list + 1)),
            ("blue", r"$\log_{10}(1+V(X_t))$", np.log10(V_X_t_list + 1)),
        ]

        for idx, (color, y_label, ys) in enumerate(data):
            ax = axes[lx][idx]
            ax.scatter(log10_n_t_list, ys, s=1, color=color, alpha=0.1)

            # select tokens whose log10(n_t) is in [1, 5]
            sel_log10_n_t_list = []
            sel_ys = []
            for log10_n_t, y in zip(log10_n_t_list, ys):
                if 1 <= log10_n_t <= 5:
                    sel_log10_n_t_list.append(log10_n_t)
                    sel_ys.append(y)
            sel_log10_n_t_list = np.array(sel_log10_n_t_list)
            sel_ys = np.array(sel_ys)

            # linear regression
            regr = LinearRegression()
            regr.fit(sel_log10_n_t_list.reshape(-1, 1), sel_ys)
            slope = regr.coef_[0]
            intercept = regr.intercept_
            score = regr.score(sel_log10_n_t_list.reshape(-1, 1), sel_ys)

            xs_ = np.linspace(min(log10_n_t_list), max(log10_n_t_list), 100)
            ax.plot(
                xs_,
                [slope * j + intercept for j in xs_],
                color="red",
                linewidth=lw,
                label=f"Slope: {slope:.3f}, " + r"$R^2" + f": {score:.3f}" + r"$",
            )

            ax.set_xlabel(r"$\log_{10} n_t$", fontsize=ls)
            ax.set_ylabel(y_label, fontsize=ls)
            if idx == 2:
                ax.legend(fontsize=legend_s, loc="lower right")
            else:
                ax.legend(fontsize=legend_s, loc="upper right")
            ax.tick_params(axis="both", which="major", labelsize=tick_s)

            y_mean = np.mean(ys)
            std = np.std(ys)
            ax.set_ylim(y_mean - 2 * std, y_mean + 2 * std)

            ax.yaxis.set_major_formatter(FuncFormatter(format_func))

        # V(X_t) on M(X_t)
        ax = axes[lx][3]

        sorted_log10_n_t_list, sorted_M_X_t_list, sorted_V_X_t_list = zip(
            *sorted(zip(log10_n_t_list, M_X_t_list, V_X_t_list), key=lambda x: x[0])
        )

        scatter = ax.scatter(
            sorted_M_X_t_list,
            sorted_V_X_t_list,
            s=1,
            c=sorted_log10_n_t_list,
            cmap="rainbow",
            alpha=0.5,
        )

        # color bar ticks
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.ax.tick_params(labelsize=tick_s)
        cbar.set_label(r"$\log_{10} n_t$", fontsize=ls, labelpad=25, rotation=270)
        ax.set_xlabel(r"$M(X_t)$", fontsize=ls)
        ax.set_ylabel(r"$V(X_t)$", fontsize=ls)
        ax.tick_params(axis="both", which="major", labelsize=tick_s)

        # linear regression
        regr = LinearRegression()
        sel_M_X_t_list = []
        sel_V_X_t_list = []
        for log10_n_t, M_X_t, V_X_t in zip(log10_n_t_list, M_X_t_list, V_X_t_list):
            if 1 <= log10_n_t <= 5:
                sel_M_X_t_list.append(M_X_t)
                sel_V_X_t_list.append(V_X_t)
        regr.fit(np.array(sel_M_X_t_list).reshape(-1, 1), sel_V_X_t_list)
        # slope
        slope = regr.coef_[0]
        # intercept
        intercept = regr.intercept_
        score = regr.score(np.array(sel_M_X_t_list).reshape(-1, 1), sel_V_X_t_list)
        xs_ = np.linspace(min(M_X_t_list), max(M_X_t_list), 100)
        ax.plot(
            xs_,
            [slope * j + intercept for j in xs_],
            color="red",
            label=f"Slope: {slope:.3f}, " + r"$R^2$" + f": {score:.3f}",
        )
        ax.legend(loc="upper right", fontsize=legend_s)
        ax.tick_params(labelsize=tick_s)

        ax.set_xlim(np.percentile(M_X_t_list, 0.1), np.percentile(M_X_t_list, 99.9))
        ax.set_ylim(np.percentile(V_X_t_list, 0.1), np.percentile(V_X_t_list, 99.9))

        ax2 = ax.twinx()
        ax2.set_ylabel(
            f"Layer {layer_idx}", fontsize=right_ls, rotation=270, labelpad=100
        )
        # ax2 has "no" ticks
        ax2.tick_params(
            axis="both",
            which="both",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )
    # save
    fig.subplots_adjust(
        left=0.05, right=0.96, bottom=0.055, top=0.985, wspace=0.3, hspace=0.45
    )
    output_dir = Path("output/images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f"QXt_MXt_VXt_VXtonMXt_scatterplot_{model_name}.png"
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    encoder_model_names, decoder_model_names = get_model_names()
    model_names = encoder_model_names + decoder_model_names
    logger.info(f"model_names: {model_names}")

    dataset_name = "bookcorpus"
    data_split = "train"
    max_len = 64
    pct = 0.01
    seed = 0

    pctstr = f"{int(pct * 100):03d}"
    input_dir = Path(
        f"output/token_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}"
    )  # noqa

    for model_name in model_names:
        logger.info(f"model_name: {model_name}")
        input_path = input_dir / f"{model_name}.pkl"
        logger.info(f"loading token stats from {input_path}")
        with open(input_path, "rb") as f:
            layer_idx2token_stats = pickle.load(f)
        save_plot(layer_idx2token_stats, model_name)


if __name__ == "__main__":
    main()
