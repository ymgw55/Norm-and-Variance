import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
    fig, axes = plt.subplots(2, half, figsize=(18, 9))

    # left bottom width height
    cb_ax = fig.add_axes([0.935, 0.08, 0.015, 0.87])
    titile_s = 22
    label_s = 22
    legend_s = 13
    tick_s = 16
    cbar_label_s = label_s
    cbar_tick_s = 20

    # NOTE: unify the range of the color bar in all scatter plots.
    all_log10_n_t_values = []
    for mx, (model_name, layer_idx2token_stats) in enumerate(
            model2layer_idx2token_stats.items()):
        for layer_idx, token_stats in layer_idx2token_stats.items():
            all_log10_n_t_values.extend(np.log10(token_stats['n_t']))
    vmin = min(all_log10_n_t_values)
    vmax = max(all_log10_n_t_values)


    for mx, (model_name, layer_idx2token_stats) in enumerate(
            model2layer_idx2token_stats.items()):

        ax = axes[mx % 2][(mx // 2) % half]

        layer_idxs = sorted(list(layer_idx2token_stats.keys()))
        middle_layer_idx = layer_idxs[len(layer_idxs)//2]

        n_t_list = layer_idx2token_stats[middle_layer_idx]['n_t']
        M_X_t_list = layer_idx2token_stats[middle_layer_idx]['M_X_t']
        V_X_t_list = layer_idx2token_stats[middle_layer_idx]['V_X_t']

        log10_n_t_list = np.log10(n_t_list)

        sorted_log10_n_t_list, sorted_M_X_t_list, sorted_V_X_t_list = zip(
            *sorted(zip(log10_n_t_list, M_X_t_list, V_X_t_list),
                    key=lambda x: x[0]))

        scatter = ax.scatter(sorted_M_X_t_list, sorted_V_X_t_list, s=1,
                             c=sorted_log10_n_t_list,
                             cmap='rainbow', alpha=0.5, vmin=vmin, vmax=vmax)
        if mx == 0:
            cbar = fig.colorbar(scatter, cax=cb_ax)
            cbar.set_label(r'$\log_{10} n_t$', rotation=270, labelpad=30,
                           fontsize=cbar_label_s)
            cbar.ax.tick_params(labelsize=cbar_tick_s)

        # label
        title = f'{model_name}  Layer {middle_layer_idx}'
        ax.set_title(title, fontsize=titile_s, pad=10)
        ax.set_xlabel(r'$M(X_t)$', fontsize=label_s)
        ax.set_ylabel(r'$V(X_t)$', fontsize=label_s)

        # select tokens whose log10(n_t) is in [1, 5]
        sel_M_X_t_list = []
        sel_V_X_t_list = []
        for M_X_t, V_X_t, log10_n_t in zip(
                M_X_t_list, V_X_t_list, log10_n_t_list):
            if 1 <= log10_n_t <= 5:
                sel_M_X_t_list.append(M_X_t)
                sel_V_X_t_list.append(V_X_t)

        # linear regression
        regr = LinearRegression()
        regr.fit(np.array(sel_M_X_t_list).reshape(-1, 1), sel_V_X_t_list)
        # slope
        slope = regr.coef_[0]
        # intercept
        intercept = regr.intercept_
        score = regr.score(np.array(sel_M_X_t_list).reshape(-1, 1),
                           sel_V_X_t_list)
        xs = np.linspace(min(sel_M_X_t_list), max(sel_M_X_t_list), 100)
        ax.plot(xs, [slope*j + intercept for j in xs],
                color='red',
                label=f'Slope: {slope:.3f}, ' + r'$R^2$' + f': {score:.3f}')
        ax.legend(loc='upper right', fontsize=legend_s)

        ax.set_xlim(np.percentile(M_X_t_list, 0.1),
                    np.percentile(M_X_t_list, 99.9))
        ax.set_ylim(np.percentile(V_X_t_list, 0.1),
                    np.percentile(V_X_t_list, 99.9))
        ax.tick_params(labelsize=tick_s)

    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.95,
                        wspace=0.32, hspace=0.42)

    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'VXt_on_MXt_scatterplot.png'
    logger.info(f'saving the figure to {output_path}')
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    main()