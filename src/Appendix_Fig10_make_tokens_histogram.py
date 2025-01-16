import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_model_names

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def save_plot(model2layer_idx2fQMV, output_dir):

    model_names = list(model2layer_idx2fQMV.keys())
    model_names = sorted(model_names)

    # freq correlation


def main():

    model_names = ['bert-base-uncased', 'roberta-base', 'gpt2']
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

    layer_idx = 0
    model2color = {'bert-base-uncased': 'red',
                   'roberta-base': 'green',
                   'gpt2': 'blue'}
    model2label = {'bert-base-uncased': 'BERT',
                   'roberta-base': 'RoBERTa', 'gpt2': 'GPT-2'}

    tick_s = 20
    label_s = 25
    xlabel_s = label_s
    ylabel_s = label_s
    title_s = 30
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for mx, model_name in enumerate(['bert-base-uncased', 'roberta-base', 'gpt2']):
        layer_idx2token_stats = model2layer_idx2token_stats[model_name]
        tokens_list = layer_idx2token_stats[layer_idx]['tokens']
        assert len(tokens_list) == len(set(tokens_list))
        n_t_list = layer_idx2token_stats[layer_idx]['n_t']

        row = {'label': model2label[model_name],
               '|T|': len(tokens_list), 'n': sum(n_t_list)}
        logger.info(row)

        log10_n_t_list = np.log10(n_t_list)
        bin = np.linspace(0, 6.5, 30)
        ax = axes[mx]
        ax.hist(log10_n_t_list, bins=bin, alpha=0.5, label=model2label[model_name],
                color=model2color[model_name], edgecolor='black', linewidth=1.2)
        ax.tick_params(axis='both', which='major', labelsize=tick_s)
        ax.set_xticks(
            range(1, 7), [r'$' + str(i) + '$' for i in range(1, 7)])
        ax.set_title(model2label[model_name], fontsize=title_s)
        ax.set_xlabel(r'$\log_{10} n_t$', fontsize=xlabel_s)
        ax.set_xlim(0, 6.5)
        ax.set_ylabel('Frequency of ' + r'$\log_{10} n_t$', fontsize=ylabel_s)
        ax.set_ylim(0, 2900)
        ax.set_yticks(range(0, 3000, 500))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.9,
                        wspace=0.3, hspace=0.4)

    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'tokens_histogram.pdf'
    logger.info(f'Saving to {save_path}')
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()
