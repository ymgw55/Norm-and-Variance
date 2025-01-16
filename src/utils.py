import logging
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_sentences(dataset_name, data_split, max_len, pct, seed):

    np.random.seed(seed)

    input_dir = Path('output/datasets')
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f'{dataset_name}.pkl'

    if input_path.exists():
        logger.info(f'loading original dataset from {input_path}')
        with open(input_path, 'rb') as f:
            dataset = pickle.load(f)[data_split]
    else:
        logger.info(f'loading {dataset_name} dataset')
        dataset = load_dataset(dataset_name)
        logger.info(f'dataset keys: {list(dict(dataset).keys())}')
        logger.info(f'saving dataset to {input_path}')
        with open(input_path, 'wb') as f:
            pickle.dump(dataset, f)
        dataset = dataset[data_split]

    pctstr = f"{int(pct*100):03d}"
    sentences_path = input_dir / \
        f'{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}.pkl'

    if sentences_path.exists():
        logger.info(f'loading sentences from {sentences_path}')
        with open(sentences_path, 'rb') as f:
            sentences = pickle.load(f)

    else:
        logger.info(f'selecting sentences from {input_path}')

        # random choice
        use_idx_set = np.random.choice(
            len(dataset), int(pct*len(dataset)), replace=False)
        use_idx_set = set(use_idx_set)

        valid_sentences = []
        idx = -1
        for sentence_dict in tqdm(dataset):
            idx += 1
            if idx not in use_idx_set:
                continue
            if len(sentence_dict['text'].split(' ')) < max_len:
                valid_sentences.append(sentence_dict['text'])

        # sorting the sentences by length
        sentences = sorted(
            valid_sentences, key=lambda x: len(x.split(' ')))

        with open(sentences_path, 'wb') as f:
            pickle.dump(sentences, f)

    logger.info(f'use {len(sentences)} sentences, '
          f'{len(sentences)/len(dataset)*100:.3f}% of the original dataset')

    return sentences


def get_model_names():

    encoder_model_names = [
        'bert-base-uncased',
        'bert-large-uncased',
        'roberta-base',
        'roberta-large',
    ]

    decoder_model_names = [
        'gpt2',
        'gpt2-medium',
    ]

    return encoder_model_names, decoder_model_names


def get_batch_size(model_name):

    # Our setting: NVIDIA RTX A6000, 48GB
    # You may need to adjust the batch size depending on the GPU memory
    model_name2batch_size = {
        'bert-base-uncased': 1024,
        'bert-large-uncased': 128,
        'roberta-base': 512,
        'roberta-large': 128,
        'gpt2': 128,
        'gpt2-medium': 64,
    }

    return model_name2batch_size[model_name]

