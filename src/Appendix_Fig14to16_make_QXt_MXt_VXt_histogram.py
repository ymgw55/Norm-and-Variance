import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    fig, axes = plt.subplots(len(layer_idxs), 3,
                             figsize=(15, int(len(layer_idxs)*2.5)))

    right_ls = 25
    ls = 18
    tick_s = 15
    legend_s = 10

    idx2y_max = {0: 0, 1: 0, 2: 0}
    for lx, layer_idx in tqdm(enumerate(layer_idxs)):

        Q_X_t_list = np.array(layer_idx2token_stats[layer_idx]["Q_X_t"])
        M_X_t_list = np.array(layer_idx2token_stats[layer_idx]["M_X_t"])
        V_X_t_list = np.array(layer_idx2token_stats[layer_idx]["V_X_t"])

        data = [('orange', r'$Q(X_t)$', Q_X_t_list),
                ('green', r'$M(X_t)$', M_X_t_list),
                ('blue', r'$V(X_t)$', V_X_t_list)]

        for idx, (color, x_label, ys) in enumerate(data):
            ax = axes[lx][idx]

            # trim top 1% of values for better visualization
            trim_ys = ys[(ys < np.percentile(ys, 99))]

            ax.hist(trim_ys, bins=50, color=color, alpha=0.5)
            y_max = np.max(np.histogram(trim_ys, bins=50)[0])
            idx2y_max[idx] = max(idx2y_max[idx], y_max)

            y_mean = np.mean(ys)
            ax.axvline(y_mean, color='black', linestyle='dashed',
                       linewidth=1, label=f'mean: {y_mean:.1f}')

            ax.legend(fontsize=legend_s, loc='upper right')
            ax.tick_params(axis='both', which='major', labelsize=tick_s)

            ax.set_xlabel(x_label, fontsize=ls)
            ax.set_ylabel('Frequency', fontsize=ls)

        ax2 = ax.twinx()
        ax2.set_ylabel(f'Layer {layer_idx}', fontsize=right_ls,
                       rotation=270, labelpad=40)
        ax2.tick_params(axis='both', which='both', left=False, right=False,
                        labelleft=False, labelright=False)

    for lx, layer_idx in enumerate(layer_idxs):
        for idx in range(3):
            ax = axes[lx][idx]
            ax.set_ylim(0, idx2y_max[idx] * 1.1)

    # save
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.99,
                        wspace=0.3, hspace=0.45)
    output_dir = Path("output/images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir /\
        f"QXt_MXt_VXt_histogram_{model_name}.png"
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
