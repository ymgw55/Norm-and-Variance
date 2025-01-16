

# def save_plot(model_name2layer_idx2wfmuQMV, output_dir, dataset_name):

#     model2color, model2marker, model2linestyle = get_model2dict()

#     fig, axes = plt.subplots(1, 3, figsize=(28, 7))
#     axes2 = [axes[0].twinx(), axes[1].twinx()]
#     idx2y_max = defaultdict(float)
#     idx2y_max2 = defaultdict(float)
#     idx2y_label = {0: r'$V_W(X)$', 1: r'$V_B(X)$', 2: r'$V_B(X)\,/\,V(X)$'}
#     for model_name, layer_idx2wfmuQMV in \
#             model_name2layer_idx2wfmuQMV.items():
#         layer_idxs = sorted(list(layer_idx2wfmuQMV.keys()))
#         max_y = 0
#         Vwc_list = []
#         Vbc_list = []
#         Vbc_Vc_list = []
#         for layer_idx in tqdm(layer_idxs):

#             fw_list = layer_idx2wfmuQMV[layer_idx]['fs']
#             Qw_list = layer_idx2wfmuQMV[layer_idx]['Qs']
#             Vw_list = layer_idx2wfmuQMV[layer_idx]['Vs']
#             mu_w_list = layer_idx2wfmuQMV[layer_idx]['Mus']
#             Qc = 0
#             mu_c = 0
#             Vw_c = 0
#             total_f = 0
#             for fw, Qw, mu_w, Vw in zip(
#                     fw_list, Qw_list, mu_w_list, Vw_list):
#                 if 1 <= np.log10(fw) <= 5:
#                     Qc += fw * Qw
#                     mu_c += fw * mu_w
#                     Vw_c += fw * Vw
#                     total_f += fw
#             Qc /= total_f
#             mu_c /= total_f
#             Vw_c /= total_f
#             Mc = np.linalg.norm(mu_c)**2
#             Vc = Qc - Mc
#             Vb_c = Vc - Vw_c

#             Vwc_list.append(Vw_c)
#             Vbc_list.append(Vb_c)
#             Vbc_Vc_list.append(Vb_c / Vc)



import logging
import pickle
from collections import defaultdict
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
    fig, axes = plt.subplots(1, 3, figsize=(28, 7))
    axes2 = [axes[0].twinx(), axes[1].twinx()]
    idx2y_max = defaultdict(float)
    idx2y_max2 = defaultdict(float)
    idx2y_label = {0: r'$V_W(X)$', 1: r'$V_B(X)$', 2: r'$V_B(X)\,/\,V(X)$'}

    for mx, (model_name, layer_idx2token_stats) in enumerate(
            model2layer_idx2token_stats.items()):

        layer_idxs = sorted(list(layer_idx2token_stats.keys()))

        Vw_X_list = []
        Vb_X_list = []
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
            # normalize Vb_X by V_X
            normed_Vb_X = Vb_X / V_X

            Vw_X_list.append(Vw_X)
            Vb_X_list.append(Vb_X)
            normed_Vb_X_list.append(normed_Vb_X)

        xs = list(range(len(layer_idxs)))
        for idx, ys in enumerate([Vw_X_list, Vb_X_list, normed_Vb_X_list]):

            if 'gpt2' in model_name and idx != 2:
                ax = axes2[idx]
                idx2y_max2[idx] = max(idx2y_max2[idx], max(ys))
            else:
                ax = axes[idx]
                idx2y_max[idx] = max(idx2y_max[idx], max(ys))

            ax.plot(xs, ys, label=model_name, linestyle=model2linestyle[model_name],
                    color=model2color[model_name], marker=model2marker[model_name],
                    markersize=8, linewidth=2)

    legend_s = 28
    ls = 25
    ts = 30

    for idx, ax in enumerate(axes):

        ax.set_xlabel('Layer Index', fontsize=ls)

        ts = 25
        ax.tick_params(axis='both', which='major', labelsize=ts)

        # x label
        ax.set_xticks(np.arange(0, len(layer_idxs),
                                max(len(layer_idxs) // 4, 1)))
        # y label
        if idx == 2:
            max_y = idx2y_max[idx]
        else:
            max_y = max(idx2y_max[0], idx2y_max[1])
        ax.set_ylim(0.0, max_y * 1.05)

        # legend set outside the plot
        # shorten the horizontal distance between each legend
        if idx == 1:

            # dummy plot for legend
            for model_name in ['gpt2', 'gpt2-medium']:
                ax.plot([], [], label=model_name, linestyle=model2linestyle[model_name],
                        color=model2color[model_name], marker=model2marker[model_name],
                        markersize=8, linewidth=2)

            ax.legend(loc='center', fontsize=legend_s,
                      columnspacing=1, ncol=6, bbox_to_anchor=(0.5, 1.1),
                      frameon=False)

        # y label
        if idx == 2:
            ax.set_ylabel(idx2y_label[idx], fontsize=ls)
        else:
            ax.set_ylabel(idx2y_label[idx] + ' for BERT & RoBERTa',
                          fontsize=ls)

        # grid
        ax.grid(True, linestyle='--', alpha=0.5)

    max_y = max(idx2y_max2.values())
    for idx, ax in enumerate(axes2):
        # y label is log scale
        ax.set_ylim(1, max_y * 2)
        ax.set_yscale('log')
        ax.set_ylabel(idx2y_label[idx] + ' for GPT-2',
                      fontsize=ls, rotation=270, labelpad=35)
        ax.tick_params(axis='y', which='major', labelsize=ts)

    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.13, top=0.85,
                        wspace=0.45, hspace=0.4)
    output_dir = Path("output/images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / 'VwX_VbX_VbXperVX_plot.pdf'
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()
