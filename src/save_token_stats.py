import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from utils import get_batch_size, get_model_names, get_sentences

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_batch_results(batch_sentences, tokenizer):

    # bos_token
    if 'gpt2' in tokenizer.name_or_path or 'roberta' in tokenizer.name_or_path:
        bos_token = tokenizer.bos_token
    elif 'bert' in tokenizer.name_or_path:
        bos_token = tokenizer.cls_token
    else:
        raise NotImplementedError

    # eos_token
    if 'gpt2' in tokenizer.name_or_path or 'roberta' in tokenizer.name_or_path:
        eos_token = tokenizer.eos_token
    elif 'bert' in tokenizer.name_or_path:
        eos_token = tokenizer.sep_token
    else:
        raise NotImplementedError

    batch_tokens = []
    batch_indexs = []
    batch_segments_ids = []

    for sentence in batch_sentences:

        # Model tokenization
        tokens = [bos_token] + \
            tokenizer.tokenize(sentence) + [eos_token]
        batch_tokens.append(tokens)

        indexs = tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)

        batch_indexs.append(indexs)
        batch_segments_ids.append(segments_ids)

    batch_max_len = max([len(indexs)
                         for indexs in batch_indexs])

    for indexs, segments_ids in zip(
            batch_indexs, batch_segments_ids):
        assert len(indexs) == len(segments_ids)
        indexs.extend([0]*(batch_max_len-len(indexs)))
        segments_ids.extend([0]*(batch_max_len-len(segments_ids)))

    return batch_tokens, batch_indexs, batch_segments_ids


def main(model_name='bert-base-uncased'):

    dataset_name = 'bookcorpus'
    data_split = 'train'
    max_len = 64
    pct = 0.01
    seed = 0

    encoder_model_names, decoder_model_names = get_model_names()
    model_names = encoder_model_names + decoder_model_names
    logger.info(f"model_names: {model_names}")

    assert model_name in model_names

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name in encoder_model_names:
        model = AutoModel.from_pretrained(model_name).to('cuda')
        if 'roberta' in model_name:
            model_type = 'roberta'
        elif 'bert' in model_name:
            model_type = 'bert'
    elif model_name in decoder_model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        tokenizer.pad_token = tokenizer.eos_token
        model_type = 'gpt2'
    model.eval()

    batch_size = get_batch_size(model_name)
    logger.info(f"batch_size: {batch_size}")

    sentences = get_sentences(dataset_name, data_split, max_len, pct, seed)
    # Reverse the order for checking OOM error ASAP.
    # I have should do in get_sentences function, but I forgot :(.
    # Since the function was used in other scripts, 
    # I just keep it here for reproducibility.
    sentences = sentences[::-1]
    logger.info(f"len(sentences): {len(sentences)}")

    logger.info("Sequential computation for each token t")
    layer_idx2avg_dict = dict()
    for batch_index in tqdm(
            list(range((len(sentences) + batch_size - 1) // batch_size))):

        batch_sentences = sentences[batch_index *
                                    batch_size:(batch_index+1)*batch_size]
        
        batch_tokens, batch_indexs, batch_segments_ids = \
            get_batch_results(batch_sentences, tokenizer)

        indexs_tensor = torch.tensor(batch_indexs).to('cuda')
        segments_tensors = torch.tensor(batch_segments_ids).to('cuda')

        with torch.no_grad():
            outputs = model(input_ids=indexs_tensor,
                            attention_mask=segments_tensors,
                            output_hidden_states=True)
            hidden_states = outputs.hidden_states
        del outputs

        for layer_index in range(len(hidden_states)):
            batch_layer_embeddings = hidden_states[layer_index
                                                   ].to('cpu').numpy()
            if layer_index not in layer_idx2avg_dict:
                layer_idx2avg_dict[layer_index] = dict()

            for tokens, layer_embeddings in zip(
                    batch_tokens, batch_layer_embeddings):

                assert len(tokens) <= len(layer_embeddings)
                layer_embeddings = layer_embeddings[:len(tokens)]

                for token, x in zip(tokens, layer_embeddings):

                    # remove the special token prefix (e.g., '##' or 'Ġ')
                    if model_type == 'bert':
                        if token.startswith('##'):
                            token = token[2:]
                    else:
                        if token.startswith('Ġ'):
                            token = token[1:]

                    squared_norm = np.linalg.norm(x)**2

                    if token in layer_idx2avg_dict[layer_index]:
                        k, u, q, _, _ = layer_idx2avg_dict[layer_index][token]
                        k_ = k + 1
                        u_ = k*u / (k+1) + x / (k+1)
                        q_ = k*q / (k+1) + squared_norm / (k+1) 
                        m_ = np.linalg.norm(u_)**2
                        v_ = q_ - m_
                        layer_idx2avg_dict[layer_index][token] = (
                            k_, u_, q_, m_, v_)
                    else:
                        layer_idx2avg_dict[layer_index][token] = (
                            1, x, squared_norm, squared_norm, 0)
                        
    layer_idxs = sorted(list(layer_idx2avg_dict.keys()))
    layer_idx2token_stats = dict()
    for layer_idx in layer_idxs:
        avg_dict = layer_idx2avg_dict[layer_idx]
        token_list = []
        n_t_list = []
        mu_X_t_list = []
        Q_X_t_list = []
        M_X_t_list = []
        V_X_t_list = []
        for token, token_stats in avg_dict.items():
            n_t, mu_X_t, Q_X_t, M_X_t, V_X_t = token_stats
            token_list.append(token)
            n_t_list.append(n_t)
            mu_X_t_list.append(mu_X_t)
            Q_X_t_list.append(Q_X_t)
            M_X_t_list.append(M_X_t)
            V_X_t_list.append(V_X_t)

        layer_idx2token_stats[layer_idx] = {
            'tokens': token_list, 'n_t': n_t_list, 'mu_X_t': mu_X_t_list,
            'Q_X_t': Q_X_t_list, 'M_X_t': M_X_t_list, 'V_X_t': V_X_t_list}
        
    pctstr = f"{int(pct*100):03d}"
    output_dir = Path(
        f'output/token_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}')  # noqa
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{model_name}.pkl'
    logger.info(f'saving token stats to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(layer_idx2token_stats, f)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
