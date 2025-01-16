import logging
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_sentences

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():

    dataset_name = 'bookcorpus'
    data_split = 'train'
    max_len = 64
    pct = 0.01
    seed = 0

    sentences = get_sentences(dataset_name, data_split, max_len, pct, seed)

    lengths = []
    for sentence in tqdm(sentences):
        lengths.append(len(sentence.split(' ')))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(lengths, bins=30, alpha=0.7, color='gray', edgecolor='black')

    label_s = 20
    title_s = 22
    tick_s = 18
    ax.set_xlabel('Sentence Length', fontsize=label_s)
    ax.set_ylabel('Frequency', fontsize=label_s)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=tick_s)

    ax.set_title('Sampled Sentence Length Histogram',
                 fontsize=title_s, pad=10)

    output_dir = Path('output/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'sentences_histogram.pdf'
    logger.info(f'Saving to {output_path}')
    fig.subplots_adjust(
        left=0.13, right=0.99, bottom=0.12, top=0.93)
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    main()