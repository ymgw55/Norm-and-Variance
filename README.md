# Norm-and-Variance

> [Norm of Mean Contextualized Embeddings Determines their Variance](https://arxiv.org/abs/2409.11253)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/), [Hidetoshi Shimodaira](http://stat.sys.i.kyoto-u.ac.jp/members/shimo/)          
> *COLING 2025*

<div align="center">
<img src=".github/images/embeds_variance.png" alt="fig3.png" width="75%">
</div>

## Setup

This repository is intended to be run in a Docker environment. If you are not familiar with Docker, please install the packages listed in [requirements.txt](requirements.txt).

### Docker build

Create a Docker image as follows:

```bash
$ bash script/docker/build.sh
```

### Environment variable

Set the `DOCKER_HOME` environment variable to specify the path of the directory to be mounted as the home directory inside the Docker container.
```bash
export DOCKER_HOME="path/to/your/docker_home"
```

### Docker run
Run the Docker container by passing the GPU ID as an argument:
```bash
$ bash script/docker/run.sh 0
```

## Code

### Saving statistical measures of $X_t$

#### Using Preprocessed Data from the Experiments

- [Sentences extracted from bookcorpus (Google Drive)](https://drive.google.com/file/d/1Ety83AiSanikRsDDe_JtV189UUsXKRIk/view?usp=drive_link)
- [Statistical measures for each model (Google Drive)](https://drive.google.com/drive/folders/1nVX4eNE2T8TlMg7olmJjElcICZFoo56c?usp=drive_link)

Place the downloaded data in the following structure:

```bash
output/
├── datasets
│   └── bookcorpus_train_lt64_pct001_seed0.pkl
└── token_stats
    └── bookcorpus_train_lt64_pct001_seed0
        ├── bert-base-uncased.pkl
        ├── bert-large-uncased.pkl
        ├── gpt2-medium.pkl
        ├── gpt2.pkl
        ├── roberta-base.pkl
        └── roberta-large.pkl
```

#### For Reproducibility
To regenerate statistical measures:

```bash
python src/save_token_stats.py --model_name model_name
```

The `model_name` values supported are `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `roberta-large`, `gpt2`, `gpt2-medium`.


### PCA Plot in Fig. 1

```bash
python src/Fig1_make_pca_scatterplot.py
```

<div align="center">
<img src=".github/images/a4_b0.500_k2_seed3.png" alt="fig1.png" width="75%">
</div>

This script also generates Fig. 8. and Table 2. See [README.Appendix.md](README.Appendix.md)  for more details.

### Trade-off between $M(X_t)$ and $V(X_t)$ in Fig.2

```bash
python src/Fig2_make_VXt_on_MXt_scatterplot.py
```

<div align="center">
<img src=".github/images/VXt_on_MXt_scatterplot.png" alt="fig2.png" width="90%">
</div>

🚨 Note: The color bar range in the published figure was incorrect. While the color bar for BERT was shown, the ranges were not unified across models. This issue has been fixed, and its impact is minimal.

### C.V. of $Q(X_t)$, regression slopes of $V (X_t)$ on $M(X_t)$, and the corresponding $R^2$ in Fig4

```bash
python src/Fig4_make_QXtCV_MXtVXtSlope_MXtVXtR2_plot.py
```

<div align="center">
<img src=".github/images/QXtCV_MXtVXtSlope_MXtVXtR2_plot.png" alt="fig4.png" width="90%">
</div>

### Bar Graphs for $M(X)/Q(X)$, $V_W(X)/Q(X)$, $V_B(X)/Q(X)$ in Fig. 5 

```bash
python src/Fig5_make_MXVwXVbX_per_QX_bargraph.py
```

<div align="center">
<img src=".github/images/MXVwXVbX_per_QX_bargraph.png" alt="fig5.png" width="90%">
</div>

###  Plot of $V_W(X)/V(X)$ in Fig.6

```bash
python src/Fig6_make_VwX_per_VX_plot.py
```

<div align="center">
<img src=".github/images/VwX_per_VX_plot.png" alt="fig6.png" width="75%">
</div>

### Scatter plots of $Q(X_t)$, $M(X_t)$, and $V(X_t)$ against $\textrm{log}_{10}n_t$ in Fig.7

```bash
python src/Fig7_make_BERTbase_QXt_MXt_VXt_scatterplot.py
```
<div align="center">
<img src=".github/images/BERTbase_QXt_MXt_VXt_scatterplot.png" alt="fig7.png" width="90%">
</div>

## References

The code for generating embeddings was inspired by:

> Wannasuphoprasit et al. [Solving Cosine Similarity Underestimation between High Frequency Words by $\ell_2$ Norm Discounting](https://aclanthology.org/2023.findings-acl.550/). ACL 2023 Findings.
 
We sincerely thank the authors for sharing their [LivNLP/cosine-discounting](https://github.com/LivNLP/cosine-discounting) codebase.

## Appendix
See [README.Appendix.md](README.Appendix.md) for the experiments in the Appendix.

## Note
- Since the URLs of published datasets may change, please refer to the GitHub repository URL instead of the direct URL when referencing in papers, etc.
- This directory was created by [Hiroaki Yamagiwa](https://ymgw55.github.io/).