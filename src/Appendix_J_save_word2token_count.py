import logging
import pickle
from pathlib import Path

import gensim
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import get_sentences

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():

    model_name = 'bert-base-uncased'
    dataset_name = 'bookcorpus'
    data_split = 'train'
    max_len = 64
    pct = 0.01
    seed = 0
    logger.info(f'model_name: {model_name}')

    sentences = get_sentences(dataset_name, data_split, max_len, pct, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    word2token_count = {}
    for sentence in tqdm(sentences):

        # gensim tokenization
        words = gensim.utils.simple_preprocess(
            sentence, min_len=2)

        for word in words:
            if word in word2token_count:
                continue
            word2token_count[word] = len(tokenizer.tokenize(word))

    # save the results
    pctstr = f"{int(pct*100):03d}"
    output_dir = Path(
        f'output/word_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}')  # noqa
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir /\
        f'{dataset_name}_{model_name}_word2token_count.pkl'
    logger.info(f'saving word2token_count to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(word2token_count, f)


if __name__ == '__main__':
    main()