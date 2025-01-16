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


def main():

    encoder_model_names, decoder_model_names = get_model_names()
    model_names = encoder_model_names + decoder_model_names
    logger.info(f'model_names: {model_names}')

    dataset_name = 'bookcorpus'
    data_split = 'train'
    max_len = 64
    pct = 0.01
    seed = 0

    pctstr = f"{int(pct*100):03d}"
    input_dir = Path(
        f'output/token_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}')  # noqa

    model2layer_idx2token_stats = {}
    for model_name in model_names:

        input_path = input_dir / f'{model_name}.pkl'
        logger.info(f'loading token stats from {input_path}')
        with open(input_path, 'rb') as f:
            layer_idx2token_stats = pickle.load(f)
        model2layer_idx2token_stats[model_name] = layer_idx2token_stats

    half = len(model_names) // 2
    fig, axes = plt.subplots(2, half, figsize=(20, 11))

    for mx, (model_name, layer_idx2token_stats) in enumerate(
            model2layer_idx2token_stats.items()):

        ax = axes[mx % 2][(mx // 2) % half]

        layer_idxs = sorted(list(layer_idx2token_stats.keys()))

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
            normed_Vb_X = Vb_X / Q_X
            normed_Vw_X = Vw_X / Q_X

            # draw ratio bar
            ax.bar(layer_idx, normed_M_X, color='limegreen')
            ax.bar(layer_idx, normed_Vb_X, bottom=normed_M_X,
                   color='lightsteelblue')
            ax.bar(layer_idx, normed_Vw_X, bottom=normed_M_X + normed_Vb_X,
                   color='deepskyblue')

        legend_s = 32
        ls = 28
        title_s = 30
        ts = 24
        ax.set_title(model_name, fontsize=title_s, pad=10)
        ax.set_xlabel('Layer Index', fontsize=ls)
        # x label
        ax.set_xticks(np.arange(0, len(layer_idxs),
                                max(len(layer_idxs) // 4, 1)))

        ax.tick_params(axis='both', which='major', labelsize=ts)

        # y label 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ax.set_yticks(np.arange(0, 1.1, 0.2))

        if mx == 2:
            # dummy plot for legend
            ax.bar(0, 0, color='limegreen', label=r'$M(X)/Q(X)$')
            ax.bar(0, 0, color='deepskyblue', label=r'$V_W(X)/Q(X)$')
            ax.bar(0, 0, color='lightsteelblue', label=r'$V_B(X)/Q(X)$')
            ax.legend(fontsize=legend_s, loc='center', facecolor='white',
                      ncol=3, columnspacing=0.5, framealpha=1, frameon=False,
                      bbox_to_anchor=(0.5, 1.25))

    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.88,
                        wspace=0.18, hspace=0.45)
    
    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'MXVwXVbX_per_QX_bargraph.pdf'
    logger.info(f'saving the figure to {output_path}')
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    main()
