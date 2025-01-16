import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from adjustText import adjust_text
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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

    pctstr = f"{int(pct*100):03d}"
    input_dir = Path(
        f'output/token_stats/{dataset_name}_{data_split}_lt{max_len}_pct{pctstr}_seed{seed}')
    input_path = input_dir / f'{model_name}.pkl'
    with open(input_path, 'rb') as f:
        layer_idx2token_stats = pickle.load(f)

    layer_idx = 6
    token_list = layer_idx2token_stats[layer_idx]['tokens']
    n_t_list = layer_idx2token_stats[layer_idx]['n_t']
    M_X_t_list = layer_idx2token_stats[layer_idx]['M_X_t']
    V_X_t_list = layer_idx2token_stats[layer_idx]['V_X_t']

    # select tokens whose log10(n_t) is in [1, 5]
    sel_token_list = []
    sel_n_t_list = []
    sel_M_X_t_list = []
    sel_V_X_t_list = []
    for token, n_t, M_X_t, V_X_t in zip(
            token_list, n_t_list, M_X_t_list, V_X_t_list):
        if 1 <= np.log10(n_t) <= 5:
            sel_token_list.append(token)
            sel_n_t_list.append(n_t)
            sel_M_X_t_list.append(M_X_t)
            sel_V_X_t_list.append(V_X_t)

    # split selected tokens into 10 bins by n_t
    sel_n_t_list = np.array(sel_n_t_list)
    bin_size = 10
    bin_edges = np.linspace(np.log10(sel_n_t_list.min()),
                            np.log10(sel_n_t_list.max()), (bin_size + 1))

    bin_edges = np.power(10, bin_edges)
    bin_idx2bin_range = dict()
    for i in range(len(bin_edges) - 1):
        bin_idx2bin_range[i] = (bin_edges[i], bin_edges[i + 1])

    # linear regression for M(X_t) and V(X_t)
    regr = LinearRegression()
    regr.fit(np.array(M_X_t_list).reshape(-1, 1), np.array(V_X_t_list))
    slope = regr.coef_[0]
    intercept = regr.intercept_
    logger.info(f'slope: {slope}, intercept: {intercept}')

    bx2cands = defaultdict(list)
    for token, n_t, M_X_t, V_X_t in zip(
            sel_token_list, sel_n_t_list, sel_M_X_t_list, sel_V_X_t_list):
        
        if len(token) < 3:
            continue

        bin_idx = np.digitize(n_t, bin_edges) - 1
        bin_range = bin_idx2bin_range[bin_idx]
        bx2cands[bin_range].append((token, n_t, M_X_t, V_X_t))

    bx2size = {}
    for i in range(len(bin_edges) - 1):
        bin_range = bin_idx2bin_range[i]
        cands = bx2cands[bin_range] 
        logger.info(f'{bin_range}: {len(cands)}')
        bx2size[bin_range] = len(cands)
    max_size = max(bx2size.values())
    logger.info(f'max_size: {max_size}')

    # hyperparameters, which are determined by ad-hoc experiments
    seed_for_fig1 = 3
    np.random.seed(seed_for_fig1)
    k = 2
    a = 4
    b = 0.5
    # BUG: k + int((a * (size / max_size))**b) in the paper was wrong.
    # NOTE: k + int(a * (size / max_size)**b) below is correct.
    bx2sample_size = {bin_range: k + int(a * (size / max_size) ** b)
                for bin_range, size in bx2size.items()}

    sampled_tokens = []
    special_sampled_tokens = []
    for bin_range, cands in bx2cands.items():
        # random select tokens
        sampled_idxs = np.random.choice(
            len(cands), bx2sample_size[bin_range], replace=False)
        sampled_cands = [cands[i] for i in sampled_idxs]
        sampled_tokens.extend([cand[0] for cand in sampled_cands
                               if not cand[0].startswith('##')])
        special_sampled_tokens.extend(
            [f'##{cand[0]}' for cand in sampled_cands
             if not cand[0].startswith('##')])

    output_dir = Path('output/images/pca_figs')
    output_dir.mkdir(parents=True, exist_ok=True)

    # M(X_t) and V(X_t) for all tokens
    cbar_label_s = 25
    cbar_tick_s = 20
    text_s = 12
    label_s = 25
    tick_s = 20
    title_s = 30
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sorted_M_X_t_list, sorted_V_X_t_list, sorted_n_t_list = zip(
        *sorted(zip(M_X_t_list, V_X_t_list, n_t_list), key=lambda x: x[2]))
    scatter = ax.scatter(sorted_M_X_t_list, sorted_V_X_t_list,
                         c=np.log10(sorted_n_t_list),
                         cmap='rainbow', s=1, alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10} n_t$', fontsize=cbar_label_s,
                    labelpad=30, rotation=270)
    cbar.ax.tick_params(labelsize=cbar_tick_s)

    texts = []
    data = []
    sampled_token2n_t = {}
    for token, n_t, M_X_t, V_X_t in zip(
            token_list, n_t_list, M_X_t_list, V_X_t_list):
        if token in sampled_tokens:
            ax.scatter(M_X_t, V_X_t, color='black', s=10, zorder=10)
            # with line
            texts.append(ax.text(M_X_t, V_X_t, token, fontsize=text_s))
            data.append(
                {('token ' + r'$t$'): r'\textit{' + token + '}',
                 r'$n_t$': f'{n_t}',
                 r'$Q(X_t)$': 
                    f"{float(f'{M_X_t:.1f}') + float(f'{V_X_t:.1f}'):.1f}",
                 r'$M(X_t)$': f'{M_X_t:.1f}',
                 r'$V(X_t)$': f'{V_X_t:.1f}'})
            sampled_token2n_t[token] = n_t
    data.sort(key=lambda x: int(x[r'$n_t$']), reverse=True)
    df = pd.DataFrame(data)
    output_path = output_dir / \
        f'MV_{dataset_name}_{model_name}_{layer_idx}.tex'
    df.to_latex(output_path, index=False, escape=False)

    adjust_text(texts, ax=ax, arrowprops=dict(
                    arrowstyle='-', color='k', lw=0.5, zorder=10),)

    # regression line
    xs = np.linspace(min(M_X_t_list), max(M_X_t_list), 100)
    ys = slope * xs + intercept
    ax.plot(xs, ys, color='red')

    ax.set_xlim(np.percentile(M_X_t_list, 0.1),
                np.percentile(M_X_t_list, 99.9))
    ax.set_ylim(np.percentile(V_X_t_list, 0.1),
                np.percentile(V_X_t_list, 99.9))

    ax.set_xlabel(r'$M(X_t)$', fontsize=label_s)
    ax.set_ylabel(r'$V(X_t)$', fontsize=label_s)
    ax.tick_params(labelsize=tick_s)

    ax.set_title(
        f'{model_name} Layer {layer_idx}', fontsize=title_s, pad=8)

    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.11, top=0.93,
                        wspace=0.45, hspace=0.4)

    # save
    output_path = output_dir / \
        f'linear_regression_{dataset_name}_{model_name}_{layer_idx}.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sentences = get_sentences(dataset_name, data_split, max_len, pct, seed)
    tokenized_sentences = []
    for sentence in tqdm(sentences):
        tokens = tokenizer.tokenize(sentence)
        for token in tokens:
            if token in sampled_tokens or token in special_sampled_tokens:
                tokenized_sentences.append(
                    ['[CLS]'] + tokens + ['[SEP]'])
                break
    tokenized_sentences = sorted(
        tokenized_sentences, key=lambda x: len(x))

    model = AutoModel.from_pretrained(model_name).to('cuda')
    model.eval()

    bsz = 1024

    token2X_t = defaultdict(list)
    for batch_idx in tqdm(
            list(range(0, (len(tokenized_sentences) + bsz-1)//bsz))):
        batch_tokenized_sentences = \
            tokenized_sentences[batch_idx * bsz: (batch_idx+1)*bsz]
        max_len = max([len(tokens)
                        for tokens in batch_tokenized_sentences])
        batch_indexs = []
        batch_segments_ids = []
        for tokens in batch_tokenized_sentences:
            indexs = tokenizer.convert_tokens_to_ids(tokens)
            segments_ids = [1] * len(tokens)
            assert len(indexs) == len(segments_ids)

            indexs.extend([0]*(max_len-len(indexs)))
            segments_ids.extend([0]*(max_len-len(segments_ids)))
            batch_indexs.append(indexs)
            batch_segments_ids.append(segments_ids)

        batch_indexs = torch.tensor(batch_indexs).to('cuda')
        batch_segments_ids = torch.tensor(
            batch_segments_ids).to('cuda')
        with torch.no_grad():
            outputs = model(
                input_ids=batch_indexs,
                attention_mask=batch_segments_ids,
                output_hidden_states=True)
            hidden_states = outputs.hidden_states
        del outputs

        batch_layer_embeddings = hidden_states[layer_idx].to(
            'cpu').numpy()
        for tokens, layer_embeddings in zip(batch_tokenized_sentences,
                                            batch_layer_embeddings):
            tokens = tokens[1:-1]
            layer_embeddings = layer_embeddings[1:len(tokens)+1]

            for token, x in zip(tokens, layer_embeddings):
                if token in sampled_tokens:
                    token2X_t[token].append(x)
                if token in special_sampled_tokens:
                    # remove the "##" prefix
                    token2X_t[token[2:]].append(x)

    x_index = [0]
    xs = []
    mu_X_ts = []
    n_ts = []
    sampled_tokens = sorted(sampled_tokens, key=lambda x: -sampled_token2n_t[x])
    for token in sampled_tokens:
        n_t = sampled_token2n_t[token]
        X_t = token2X_t[token]
        assert len(X_t) == n_t
        logger.info(f'{token}: n_t={n_t}')
        n_ts.append(n_t)
        x_index.append(x_index[-1] + n_t)
        xs.extend(X_t)
        mu_X_t = np.mean(X_t, axis=0)
        mu_X_ts.append(mu_X_t)

    zero_vec = np.zeros_like(xs[0])
    mu_X_ts.append(zero_vec)
    mu_X_ts = np.array(mu_X_ts)
    xs = np.array(xs)

    # pca
    pca = PCA(n_components=2)
    pca.fit(mu_X_ts)
    pca_mu_X_ts = pca.transform(mu_X_ts)
    pca_origin = pca_mu_X_ts[-1]
    pca_mu_X_ts = pca_mu_X_ts[:-1] - pca_origin
    pca_xs = pca.transform(xs) - pca_origin

    token2pca_X_t = dict()
    for i, token in enumerate(sampled_tokens):
        token2pca_X_t[token] = pca_xs[x_index[i]:x_index[i+1]]

    # plot
    fig, ax = plt.subplots()

    cmap = plt.cm.rainbow
    normalizer = Normalize(vmin=np.log10(min(n_ts)), vmax=np.log10(max(n_ts)))
    scatter = ax.scatter(pca_mu_X_ts[:, 0], pca_mu_X_ts[:, 1],
                         c=np.log10(n_ts), cmap='rainbow', s=15, alpha=1,
                         edgecolors='black', linewidth=0.5, zorder=10)
    cbar = fig.colorbar(scatter, ax=ax)

    cbar_label_s = 18
    cbar_tick_s = 15
    cbar.set_label(r'$\log_{10} n_t$', fontsize=cbar_label_s,
                   labelpad=30, rotation=270)
    # tick
    cbar.ax.tick_params(labelsize=cbar_tick_s)

    token2color = dict()
    for token, n_t in zip(sampled_tokens, n_ts):
        token2color[token] = cmap(normalizer(np.log10(n_t)))

    ano_s = 7.5
    texts = []
    for token, pca_mu_X_t in zip(sampled_tokens, pca_mu_X_ts):
        ax.scatter(pca_mu_X_t[0], pca_mu_X_t[1]+0.15, color='black', s=0)
        texts.append(ax.text(pca_mu_X_t[0], pca_mu_X_t[1]+0.15, token,
                                fontsize=ano_s, color='black', zorder=100,
                                ha='center', va='bottom'))

    adjust_text(texts, ax=ax,
                arrowprops=dict(
                    arrowstyle='-', color='k', lw=0.5, zorder=10),
                force_text=0.1,
                force_points=0.1,
                expand_text=(1.05, 1.05),
                expand_points=(1.05, 1.05),  
                )

    # token vecs
    for token in sampled_tokens:
        pca_X_t = token2pca_X_t[token]
        ax.scatter(pca_X_t[:, 0], pca_X_t[:, 1],
                    color=token2color[token], s=1, alpha=0.25)

    # origin
    ax.scatter(0, 0, color='black', s=20, zorder=10, marker='x')

    # equal aspect ratio
    ax.set_aspect('equal', adjustable='datalim')

    # label
    label_s = 15
    title_s = 18
    ax.set_xlabel('PC1', fontsize=label_s)
    ax.set_ylabel('PC2', fontsize=label_s)
    ax.tick_params(labelsize=label_s)
    # x and y ticks are integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.set_title(
        f'{model_name} Layer {layer_idx}', fontsize=title_s, pad=8)

    fig.subplots_adjust(left=0.11, right=0.95, bottom=0.12, top=0.935)

    # save
    output_path = output_dir / \
        f'a{a}_b{b:.3f}_k{k}_seed{seed_for_fig1}.png'
    print(output_path)
    plt.savefig(output_path, dpi=150)


if __name__ == '__main__':
    main()
