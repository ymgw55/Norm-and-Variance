import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
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

    model_name = 'bert-base-uncased'

    input_path = input_dir / f'{model_name}.pkl'
    logger.info(f'loading token stats from {input_path}')
    with open(input_path, 'rb') as f:
        layer_idx2token_stats = pickle.load(f)

    layer_idxs = sorted(list(layer_idx2token_stats.keys()))
    n = len(layer_idxs) // 2
    middle_layer_idx = layer_idxs[n]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    title_s = 25
    ls = 22
    tick_s = 18
    lw = 2
    legend_s = 13

    n_t_list = layer_idx2token_stats[middle_layer_idx]['n_t']
    Q_X_t_list = layer_idx2token_stats[middle_layer_idx]['Q_X_t']
    M_X_t_list = layer_idx2token_stats[middle_layer_idx]['M_X_t']
    V_X_t_list = layer_idx2token_stats[middle_layer_idx]['V_X_t']

    log10_n_t_list = np.log10(n_t_list)

    data = [('orange', r'$Q(X_t)$', Q_X_t_list),
            ('green', r'$M(X_t)$', M_X_t_list),
            ('blue', r'$V(X_t)$', V_X_t_list)]

    def format_func(value, tick_number):
        return f'{int(value)}'

    for idx, (color, y_label, ys) in enumerate(data):
        ax = axes[idx]
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
        ax.plot(xs_, [slope*j + intercept for j in xs_],
                color='red', linewidth=lw, label=f'Slope: {slope:.3f}, ' + 
                r'$R^2' + f': {score:.3f}' + r'$')

        ax.set_xlabel(r'$\log_{10} n_t$', fontsize=ls)
        ax.set_ylabel(y_label, fontsize=ls)
        if idx == 2:
            ax.legend(fontsize=legend_s, loc='lower right')
        else:
            ax.legend(fontsize=legend_s, loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=tick_s)

        y_mean = np.mean(ys)
        std = np.std(ys)
        ax.set_ylim(y_mean - 3*std, y_mean + 3*std)

        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    fig.suptitle(f'{model_name} Layer {middle_layer_idx}',
                 fontsize=title_s, y=0.98)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.86,
                        wspace=0.3, hspace=0.45)

    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'BERTbase_QXt_MXt_VXt_scatterplot.png'
    logger.info(f'saving the figure to {output_path}')
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == '__main__':
    main()