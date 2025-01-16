import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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

    model2color, model2marker, model2linestyle = get_model2dict()

    # CV, slope, R2
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))

    for (model_name, 
         layer_idx2token_stats) in model2layer_idx2token_stats.items():

        layer_idxs = sorted(list(layer_idx2token_stats.keys()))
        CVs = []
        slopes = []
        R2s = []
        for layer_idx in layer_idxs:

            n_t_list = layer_idx2token_stats[layer_idx]['n_t']
            Q_X_t_list = layer_idx2token_stats[layer_idx]['Q_X_t']
            M_X_t_list = layer_idx2token_stats[layer_idx]['M_X_t']
            V_X_t_list = layer_idx2token_stats[layer_idx]['V_X_t']

            log10_n_t_list = np.log10(n_t_list)

            # select tokens whose log10(n_t) is in [1, 5]
            sel_Q_X_t_list = []
            sel_M_X_t_list = []
            sel_V_X_t_list = []
            for Q_X_t, M_X_t, V_X_t, log10_n_t in zip(
                    Q_X_t_list, M_X_t_list, V_X_t_list, log10_n_t_list):
                if 1 <= log10_n_t <= 5:
                    sel_Q_X_t_list.append(Q_X_t)
                    sel_M_X_t_list.append(M_X_t)
                    sel_V_X_t_list.append(V_X_t)

            # CV of Q(X_t)
            cv = np.std(sel_Q_X_t_list) / np.mean(sel_Q_X_t_list)
            CVs.append(cv)

            # linear regression of V(X_t) on M(X_t)
            regr = LinearRegression()
            regr.fit(np.array(sel_M_X_t_list).reshape(-1, 1), sel_V_X_t_list)

            # slope of V(X_t) on M(X_t)
            slope = regr.coef_[0]
            slopes.append(slope)

            # R2 of V(X_t) on M(X_t)
            score = regr.score(np.array(sel_M_X_t_list).reshape(-1, 1),
                               sel_V_X_t_list)
            R2s.append(score)

        xs = list(range(len(layer_idxs)))
        for idx, ys in enumerate([CVs, slopes, R2s]):
            ax = axes[idx]
            ax.plot(xs, ys, label=model_name,
                    linestyle=model2linestyle[model_name],
                    marker=model2marker[model_name], markersize=8,
                    linewidth=2, color=model2color[model_name])

    legend_s = 33
    ls = 32
    ts = 26

    ax = axes[0]
    ax.set_ylabel('C.V. of ' + r'$Q(X_t)$', fontsize=ls, labelpad=10)
    ax.set_yticks(np.arange(0.0, 0.4, 0.1))

    ax = axes[1]
    ax.legend(loc='center', fontsize=legend_s, columnspacing=0.5, ncol=6,
              bbox_to_anchor=(0.43, 1.06), frameon=False)
    ax.set_ylabel('Slope of ' + r'$V(X_t)$' + ' on ' +
                  r'$M(X_t)$', fontsize=ls, labelpad=10)
    ax.set_yticks(np.arange(-1.0, 0.1, 0.2))
    ax.set_ylim(-1.1, 0.1)

    ax = axes[2]
    ax.set_ylabel(r'$R^2$' ' of ' + r'$V(X_t)$' + ' on ' +
                  r'$M(X_t)$', fontsize=ls, labelpad=10)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_ylim(-0.1, 1.1)

    for ax in axes:
        ax.set_xlabel('Layer Index', fontsize=ls)

        ax.tick_params(axis='both', which='major', labelsize=ts)

        # x label
        ax.set_xticks(np.arange(0, len(layer_idxs),
                                max(len(layer_idxs) // 4, 1)))

        # grid
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.13, top=0.9,
                        wspace=0.28, hspace=0.35)
    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'QXtCV_MXtVXtSlope_MXtVXtR2_plot.pdf'
    logger.info(f'saving the figure to {output_path}')
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    main()