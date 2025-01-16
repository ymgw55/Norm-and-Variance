import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import get_model2dict, get_model_names

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
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    idx2y_label = {0: r'$M(X)/Q(X)$',
                   1: r'$V(X)/Q(X)$',
                   2: r'$V_W(X)/Q(X)$',
                   3: r'$V_B(X)/Q(X)$'}

    for mx, (model_name, layer_idx2token_stats) in enumerate(
            model2layer_idx2token_stats.items()):

        layer_idxs = sorted(list(layer_idx2token_stats.keys()))

        normed_M_X_list = []
        normed_V_X_list = []
        normed_Vw_X_list = []
        normed_Vb_X_list = []

        for layer_idx in tqdm(layer_idxs):

            n_t_list = layer_idx2token_stats[layer_idx]['n_t']
            mu_X_t_list = layer_idx2token_stats[layer_idx]['mu_X_t']
            Q_X_t_list = layer_idx2token_stats[layer_idx]['Q_X_t']
            V_X_t_list = layer_idx2token_stats[layer_idx]['V_X_t']

            n = 0
            mu_X = 0
            Q_X = 0
            Vw_X = 0
            # To compute n, mu_X, Q_X, Vw_X, 
            # we select tokens whose log10(n_t) is in [1, 5]
            for n_t, mu_X_t, Q_X_t, V_X_t in zip(
                    n_t_list, mu_X_t_list, Q_X_t_list, V_X_t_list):
                if 1 <= np.log10(n_t) <= 5:
                    n += n_t
                    mu_X += n_t * mu_X_t
                    Q_X += n_t * Q_X_t
                    Vw_X += n_t * V_X_t

            # normalize by n
            mu_X /= n
            Q_X /= n
            Vw_X /= n

            # M_X is the squared norm of the mean embedding mu_X
            M_X = np.linalg.norm(mu_X)**2
            # compute V_X using Q_X = M_X + V_X
            V_X = Q_X - M_X
            # compute Vb_X using V_X = Vb_X + Vw_X
            Vb_X = V_X - Vw_X

            # normalize by Q_X
            normed_M_X = M_X / Q_X
            normed_V_X = V_X / Q_X
            normed_Vb_X = Vb_X / Q_X
            normed_Vw_X = Vw_X / Q_X

            normed_M_X_list.append(normed_M_X)
            normed_V_X_list.append(normed_V_X)
            normed_Vw_X_list.append(normed_Vw_X)
            normed_Vb_X_list.append(normed_Vb_X)

        xs = list(range(len(layer_idxs)))
        data = [normed_M_X_list, normed_V_X_list, normed_Vw_X_list, normed_Vb_X_list]
        for idx, ys in enumerate(data):
            ax = axes[idx]
            ax.plot(xs, ys, label=model_name,
                    linestyle=model2linestyle[model_name],
                    color=model2color[model_name],
                    marker=model2marker[model_name],
                    markersize=8, linewidth=2)
    
    legend_s = 30
    ls = 30
    yls = ls
    ts = 30
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
                      columnspacing=1, ncol=6, bbox_to_anchor=(1, 1.1),
                      frameon=False)

        # y label
        ax.set_ylabel(idx2y_label[idx], fontsize=yls)

        # grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # y limit
        ax.set_ylim(-0.05, 1.05)

        # y ticks
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.16,
                        top=0.88, wspace=0.28, hspace=0.4)
    output_dir = Path("output/images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / 'MX_VX_VwX_VbX_per_QX_plot.pdf'
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()
