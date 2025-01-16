import logging
import pickle
from pathlib import Path

import gensim
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import get_batch_size, get_sentences

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_batch_results(batch_sentences, tokenizer):

    batch_gensim_words = []
    batch_indexs = []
    batch_segments_ids = []
    batch_word_idx2token_embedding_indexs = []

    bos_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token

    for sentence in batch_sentences:

        # gensim tokenization
        gensim_words = gensim.utils.simple_preprocess(
            sentence, min_len=2)
        batch_gensim_words.append(gensim_words)

        # Model tokenization
        tokens = []
        tokens.append(bos_token)

        # index of the current word embedding
        word_idx2token_embedding_indexs = []
        for gensim_word in gensim_words:
            len_now = 0
            for token in tokenizer.tokenize(gensim_word):
                # current word embedding index
                len_now += 1
                tokens.append(token)
            word_idx2token_embedding_indexs.append(
                [len(tokens) - 1 - i for i in range(len_now)])

        tokens.append(eos_token)
        indexs = tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)
        assert len(indexs) == len(segments_ids)

        batch_indexs.append(indexs)
        batch_segments_ids.append(segments_ids)
        batch_word_idx2token_embedding_indexs.append(word_idx2token_embedding_indexs)

    batch_max_len = max([len(indexs) for indexs in batch_indexs])

    for indexs, segments_ids in zip(
            batch_indexs, batch_segments_ids):
        assert len(indexs) == len(segments_ids)
        indexs.extend([0]*(batch_max_len-len(indexs)))
        segments_ids.extend([0]*(batch_max_len-len(segments_ids)))

    return batch_gensim_words, batch_indexs,\
        batch_segments_ids, batch_word_idx2token_embedding_indexs


def main():

    model_name = 'bert-base-uncased'
    dataset_name = 'bookcorpus'
    data_split = 'train'
    max_len = 64
    pct = 0.01
    seed = 0
    logger.info(f'model_name: {model_name}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    model.eval()
    batch_size = get_batch_size(model_name)
    logger.info(f"batch_size: {batch_size}")


    sentences = get_sentences(dataset_name, data_split, max_len, pct, seed)
    sentences = sentences[::-1]

    batch_size = get_batch_size(model_name)
    logger.info(f'batch_size: {batch_size}')

    logger.info("Sequential computation for each word")
    layer_idx2avg_dict = dict()
    for batch_index in tqdm(
            list(range((len(sentences) + batch_size - 1) // batch_size))):

        batch_sentences = sentences[batch_index *
                                    batch_size:(batch_index+1)*batch_size]
        
        (batch_gensim_words,
         batch_indexs,
         batch_segments_ids, 
         batch_word_idx2token_embedding_indexs
         ) = get_batch_results(batch_sentences, tokenizer)

        indexs_tensor = torch.tensor(batch_indexs).to('cuda')
        segments_tensors = torch.tensor(batch_segments_ids).to('cuda')

        with torch.no_grad():
            outputs = model(input_ids=indexs_tensor,
                            attention_mask=segments_tensors,
                            output_hidden_states=True)
            hidden_states = outputs.hidden_states
        del outputs

        for bx, (gensim_words, word_idx2token_embedding_indexs) in enumerate(
                zip(batch_gensim_words, batch_word_idx2token_embedding_indexs)):

            for wx in range(len(gensim_words)):
                word = gensim_words[wx]
                token_embedding_indexs = word_idx2token_embedding_indexs[wx]
                for layer_index in range(len(hidden_states)):

                    if layer_index not in layer_idx2avg_dict:
                        layer_idx2avg_dict[layer_index] = dict()

                    batch_layer_embeddings = hidden_states[layer_index]
                    assert len(batch_layer_embeddings) == len(batch_gensim_words)
                    layer_embeddings = batch_layer_embeddings[bx]

                    # creating word embedding
                    x = sum([layer_embeddings[token_embedding_index].to('cpu')
                             for token_embedding_index in token_embedding_indexs]
                            )/len(token_embedding_indexs)

                    # squared norm of the embedding
                    squared_norm = np.linalg.norm(x)**2

                    if word in layer_idx2avg_dict[layer_index]:
                        k, u, q, _, _ = layer_idx2avg_dict[layer_index][word]
                        k_ = k + 1
                        u_ = k*u / (k+1) + x / (k+1)
                        q_ = k*q / (k+1) + squared_norm / (k+1) 
                        m_ = np.linalg.norm(u_)**2
                        v_ = q_ - m_
                        layer_idx2avg_dict[layer_index][word] = (
                            k_, u_, q_, m_, v_)
                    else:
                        layer_idx2avg_dict[layer_index][word] = (
                            1, x, squared_norm, squared_norm, 0)

    layer_idxs = sorted(list(layer_idx2avg_dict.keys()))
    layer_idx2word_stats = dict()
    for layer_idx in layer_idxs:
        avg_dict = layer_idx2avg_dict[layer_idx]
        word_list = []
        n_w_list = []
        mu_X_w_list = []
        Q_X_w_list = []
        M_X_w_list = []
        V_X_w_list = []
        for word, word_stats in avg_dict.items():
            n_w, mu_X_w, Q_X_w, M_X_w, V_X_w = word_stats
            word_list.append(word)
            n_w_list.append(n_w)
            mu_X_w_list.append(mu_X_w)
            Q_X_w_list.append(Q_X_w)
            M_X_w_list.append(M_X_w)
            V_X_w_list.append(V_X_w)

        layer_idx2word_stats[layer_idx] = {
            'words': word_list, 'n_w': n_w_list, 'mu_X_w': mu_X_w_list,
            'Q_X_w': Q_X_w_list, 'M_X_w': M_X_w_list, 'V_X_w': V_X_w_list}

    pctstr = f"{int(pct*100):03d}"
    output_dir = Path(
        f'output/word_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}')  # noqa
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{model_name}.pkl'
    logger.info(f'saving word stats to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(layer_idx2word_stats, f)


if __name__ == '__main__':
    main()
