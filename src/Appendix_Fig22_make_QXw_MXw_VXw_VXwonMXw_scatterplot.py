import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def save_plot(layer_idx2word_stats, model_name, word2token_count):
    layer_idxs = sorted(list(layer_idx2word_stats.keys()))

    n = len(layer_idxs) // 2
    layer_idxs = [
        layer_idxs[0],
        layer_idxs[n // 2],
        layer_idxs[n],
        layer_idxs[n + n // 2],
        layer_idxs[-1],
    ]
    fig, axes = plt.subplots(len(layer_idxs), 4,
                             figsize=(20, int(len(layer_idxs)*2.75)))

    def format_func(value, tick_number):
        return str(int(value))

    right_ls = 25
    title_s = 25
    ls = 18
    tick_s = 15
    legend_s = 9
    for lx, layer_idx in tqdm(enumerate(layer_idxs)):

        words = layer_idx2word_stats[layer_idx]["words"]
        n_w_list = layer_idx2word_stats[layer_idx]["n_w"]
        Q_X_w_list = layer_idx2word_stats[layer_idx]["Q_X_w"]
        M_X_w_list = layer_idx2word_stats[layer_idx]["M_X_w"]
        V_X_w_list = layer_idx2word_stats[layer_idx]["V_X_w"]

        log10_n_w_list = np.log10(n_w_list)
        Q_X_w_list = np.array(Q_X_w_list)
        M_X_w_list = np.array(M_X_w_list)
        V_X_w_list = np.array(V_X_w_list)

        data = [r'$Q(X_w)$', r'$M(X_w)$', r'$V(X_w)$']

        token_count2logn_QMV_list = defaultdict(list)
        for word, log10_n_w, Q_X_w, M_X_w, V_X_w in zip(words, log10_n_w_list,
                Q_X_w_list, M_X_w_list, V_X_w_list):
            token_count = word2token_count[word]
            QMV = (Q_X_w, M_X_w, V_X_w)
            token_count2logn_QMV_list[min(token_count, 4)].append((log10_n_w, QMV))

        token_count2color = {}
        for token_count in range(1, 5):
            token_count2color[token_count] = plt.cm.tab20(token_count)

        for idx, y_label in enumerate(data):
            ax = axes[lx][idx]

            for token_count in range(1, 5):
                xs, ys = zip(*[(log10_n_w, QMV[idx]) for log10_n_w, QMV in
                               token_count2logn_QMV_list[token_count]])
                color = token_count2color[token_count]
                ax.scatter(xs, ys, s=1, color=color, alpha=1)

            # only show legend
            for token_count in range(1, 5):
                color = token_count2color[token_count]
                if token_count == 4:
                    label = '4+'
                else:
                    label = f'{token_count}'
                ax.scatter([], [], s=10, color=color, alpha=1, label=label)

            ax.set_xlabel(r'$\log_{10} n_w$', fontsize=ls)
            ax.set_ylabel(y_label, fontsize=ls)
            if idx == 2:
                ax.legend(fontsize=legend_s, loc='lower right')
            else:
                ax.legend(fontsize=legend_s, loc='upper right')
            ax.tick_params(axis='both', which='major', labelsize=tick_s)

            ax.yaxis.set_major_formatter(FuncFormatter(format_func))

        # V(X_w) on M(X_w)
        ax = axes[lx][3]
        for token_count in range(1, 5):
            ms, vs = zip(*[(QMV[1], QMV[2]) for _, QMV in
                         token_count2logn_QMV_list[token_count]])
            color = token_count2color[token_count]
            ax.scatter(ms, vs, s=1, color=color, alpha=1)
        # only show legend
        for token_count in range(1, 5):
            color = token_count2color[token_count]
            if token_count == 4:
                label = '4+'
            else:
                label = f'{token_count}'
            ax.scatter([], [], s=10, color=color, alpha=1, label=label)

        ax.set_xlabel(r'$M(X_w)$', fontsize=ls)
        ax.set_ylabel(r'$V(X_w)$', fontsize=ls)
        ax.tick_params(axis='both', which='major', labelsize=tick_s)
        ax.legend(loc='upper right', fontsize=legend_s)

        ax2 = ax.twinx()
        ax2.set_ylabel(f'Layer {layer_idx}', fontsize=right_ls,
                       rotation=270, labelpad=35)
        # ax2 has "no" ticks
        ax2.tick_params(axis='both', which='both', left=False, right=False,
                        labelleft=False, labelright=False)

    fig.suptitle(f'{model_name}', fontsize=title_s, y=0.99)

    # save
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.055, top=0.955,
                        wspace=0.3, hspace=0.45)
    output_dir = Path("output/images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f"QXw_MXw_VXw_VXwonMXw_scatterplot_{model_name}.png"
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():

    model_name = "bert-base-uncased"
    dataset_name = "bookcorpus"
    data_split = "train"
    max_len = 64
    pct = 0.01
    seed = 0
    logger.info(f"model_name: {model_name}")


    pctstr = f"{int(pct * 100):03d}"
    input_dir = Path(
        f"output/word_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}"
    )  # noqa

    logger.info(f"model_name: {model_name}")
    input_path = input_dir / f"{model_name}.pkl"
    logger.info(f"loading word stats from {input_path}")
    with open(input_path, "rb") as f:
        layer_idx2word_stats = pickle.load(f)

    input_path = input_dir /\
        f'{dataset_name}_{model_name}_word2token_count.pkl'
    logger.info(f"loading word2token_count from {input_path}")
    with open(input_path, 'rb') as f:
        word2token_count = pickle.load(f)

    save_plot(layer_idx2word_stats, model_name, word2token_count)


if __name__ == "__main__":
    main()
