import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import get_model_names, get_model2dict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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

    model2layer_idx2token_stats = {}
    for model_name in model_names:

        input_path = input_dir / f'{model_name}.pkl'
        logger.info(f'loading token stats from {input_path}')
        with open(input_path, 'rb') as f:
            layer_idx2token_stats = pickle.load(f)
        model2layer_idx2token_stats[model_name] = layer_idx2token_stats

    model2color, model2marker, model2linestyle = get_model2dict()

    fig, axes = plt.subplots(1, 3, figsize=(28, 7))
    idx2y_label = {0: r'$\log_{10}(1+Q(X_t))$',
                   1: r'$\log_{10}(1+M(X_t))$', 2: r'$\log_{10}(1+V(X_t))$'}

    for mx, (model_name, layer_idx2token_stats) in enumerate(
            model2layer_idx2token_stats.items()):

        layer_idxs = sorted(list(layer_idx2token_stats.keys()))

        slopes_dict = defaultdict(list)
        for layer_idx in layer_idxs:

            n_t_list = layer_idx2token_stats[layer_idx]['n_t']
            Q_X_t_list = layer_idx2token_stats[layer_idx]['Q_X_t']
            M_X_t_list = layer_idx2token_stats[layer_idx]["M_X_t"]
            V_X_t_list = layer_idx2token_stats[layer_idx]['V_X_t']

            log10_n_t_list = np.log10(n_t_list)
            Q_X_t_list = np.array(Q_X_t_list)
            M_X_t_list = np.array(M_X_t_list)
            V_X_t_list = np.array(V_X_t_list)

            data = [np.log10(Q_X_t_list+1),
                    np.log10(M_X_t_list+1),
                    np.log10(V_X_t_list+1)]

            for idx, ys in enumerate(data):

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
                slopes_dict[idx].append(slope)

        xs = list(range(len(layer_idxs)))
        for idx in range(3):
            ax = axes[idx]
            slopes = slopes_dict[idx]

            ax.plot(xs, slopes, label=model_name,
                    linestyle=model2linestyle[model_name],
                    color=model2color[model_name],
                    marker=model2marker[model_name],
                    markersize=8, linewidth=2)

    legend_s = 28
    ls = 25
    yls = 24
    ts = 25

    for idx, ax in enumerate(axes):

        ax.set_xlabel('Layer Index', fontsize=ls)

        ax.tick_params(axis='both', which='major', labelsize=ts)

        # x label
        ax.set_xticks(np.arange(0, len(layer_idxs),
                                max(len(layer_idxs) // 4, 1)))

        # legend set outside the plot
        # shorten the horizontal distance between each legend
        if idx == 1:

            ax.legend(loc='center', fontsize=legend_s,
                      columnspacing=1, ncol=6, bbox_to_anchor=(0.5, 1.15),
                      frameon=False)

        # y label
        ax.set_ylabel('Slope of ' + idx2y_label[idx] +
                      ' on ' + r'$\log_{10} n_t$',
                      fontsize=yls)

        # grid
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.85,
                        wspace=0.25, hspace=0.4)
    output_dir = Path("output/images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / 'Slope_of_QXt_MXt_VXt_plot.pdf'
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()
