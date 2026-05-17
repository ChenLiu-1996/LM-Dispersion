<div align="center">

  <h1><code>LM-Dispersion</code></h1>

  [![arXiv](https://img.shields.io/badge/arXiv-Dispersion-firebrick)](https://arxiv.org/abs/2602.00217)
  [![PDF](https://img.shields.io/badge/PDF-DADBDD)](https://arxiv.org/pdf/2602.00217)
  [![Project_Page](https://img.shields.io/badge/Project_Page-B9DEF1)](https://chenliu-1996.github.io/projects/LM-Dispersion/)
  [![ICML 2026](https://img.shields.io/badge/ICML_2026-purple)](https://openreview.net/pdf?id=pd6A7jB5D6)
  [![OpenReview](https://img.shields.io/badge/OpenReview-eeeeee)](https://openreview.net/forum?id=pd6A7jB5D6)
  [![GitHub Stars](https://img.shields.io/github/stars/ChenLiu-1996/LM-Dispersion.svg?style=social\&label=Stars)](https://github.com/ChenLiu-1996/LM-Dispersion)
  <br>[![Latest PyPI version](https://img.shields.io/pypi/v/embedding-condensation.svg)](https://pypi.org/project/embedding-condensation/)
  [![PyPI download 3 month](https://static.pepy.tech/badge/embedding-condensation)](https://pepy.tech/projects/embedding-condensation)
  [![PyPI download month](https://img.shields.io/pypi/dm/embedding-condensation.svg)](https://pypistats.org/packages/embedding-condensation)
  <br>[![LinkedIn](https://img.shields.io/badge/LinkedIn-Chen-blue)](https://www.linkedin.com/in/chenliu1996/)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Xingzhi-blue)](https://www.linkedin.com/in/xingzhi-sun)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Xiao-blue)](https://www.linkedin.com/in/xi-xiao-4800272a5)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Alex-blue)](https://www.linkedin.com/in/alexandre-van-tassel)
  <br>[![Google Scholar](https://img.shields.io/badge/Scholar-Chen-4a86cf?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=3rDjnykAAAAJ&sortby=pubdate)
  [![Google Scholar](https://img.shields.io/badge/Scholar-Xingzhi-4a86cf?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=tUvfTd8AAAAJ)
  <br>[![Twitter Follow](https://img.shields.io/twitter/follow/Chen.svg?style=social)](https://x.com/ChenLiu_1996)
  [![Twitter Follow](https://img.shields.io/twitter/follow/Xingzhi.svg?style=social)](https://x.com/https://x.com/XingzhiSun)
  [![Twitter Follow](https://img.shields.io/twitter/follow/Xiao.svg?style=social)](https://x.com/markshawww99)
  [![Twitter Follow](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://x.com/KrishnaswamyLab)

</div>

This is the author's repository for the ICML 2026 paper
<br>[Dispersion loss counteracts embedding condensation and improves generalization in small language models](https://arxiv.org/pdf/2602.00217).

The official version is hosted at the [Lab GitHub repo](https://github.com/KrishnaswamyLab/LM-Dispersion).

**You are encouraged to read the illustrated walkthrough of the paper on the [project website](https://chenliu-1996.github.io/projects/LM-Dispersion/).**

<br>

## A 5-minute intro to this paper
**This paper presents an observation-driven improvement on language model training.**

We observe a geometric phenomenon which we term **embedding condensation**, where token embeddings collapse into a narrow cone-like subspace in smaller language models. We then design a training objective called dispersion loss to counteract the effect.

<img src="assets/motivation.png" width="800">

**Feature 1**: Larger model, less condensation.
<br>Within the same model family, smaller models exhibit more severe embedding condensation, with token embeddings collapsing toward near-parallel directions, while larger models resist this collapse.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/observation.png" width="800">

This effect is also quite robust to the choice of input datasets.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/supp_change_dataset.png" width="800">

**Feature 2**: Reproducible when controlling for confounders.
<br>To isolate the effect of model size from other confounding factors, we conduct a controlled experiment where we pre-train GPT2-like models, varying only the MLP dimension while keeping all other components fixed, including the number of layers, embedding dimension, dataset, and training settings. The same phenomenon is observed.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/controlled_experiment.png" width="800">

**Feature 3**: Condensation occurs early on.
<br>The embedding condensation phenomenon emerges at model initialization and is gradually mitigated, not exacerbated, by pre-training.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/observation_training.png" width="400">

**Feature 4**: Distillation is not a solution.
<br>Knowledge distillation from a larger model does not transfer the desired resistance to embedding condensation.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/observation_distillation.png" width="800">

**Dispersion loss**
<br>Embedding condensation reduces the expressivity of Transformers by collapsing token embedding vectors into narrow cones, under-utilizing the representation space. We hypothesize that by dispersing embeddings during training, smaller models can achieve representational qualities more similar to larger models, thus narrowing the performance gap without increasing the number of parameters.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/loss_illustration.png" width="800">

Our dispersion loss is inspired by the "[Diffuse and Disperse](https://arxiv.org/abs/2506.09027)" paper with practical modifications.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/table_loss.png" width="800">

Dispersion loss counteracts the embedding condensation effect during mid-training and pre-training. A qualitative result is shown below, while more quantitative results can be found in the paper.

<img src="https://raw.githubusercontent.com/ChenLiu-1996/LM-Dispersion/main/assets/results_condensation_counteract.png" width="800">

## Disclaimers and future directions
Please see our [project website](https://chenliu-1996.github.io/projects/LM-Dispersion/) for disclaimers and some future directions we suggest.

## [New] PyPI support: embedding condensation

We have provided the computation and visualzation of embedding condensation into a PyPI package!

1. Install or upgrade the package.

```sh
pip install embedding-condensation --upgrade
```

2. Use it by simply passing in a `transformers` model and tokenizer, as shown in the example below.

- `max_length` determines the number of tokens in the context.
- `dataset` currently supports [`wikipedia`, `pubmed`, `imdb`, `squad`].
- `min_word_count` and `max_word_count` faciliates the text parser when grabbing a random part from the `dataset` corpse.
- If you have a specific text corpse, you can pass it in using the `texts` argument (expected format is `Sequence[str]`). This would bypass `dataset`, `min_word_count` and `max_word_count`.


```python3
import numpy as np
from transformers import AutoModel, AutoTokenizer
from embedding_condensation import measure_embedding_condensation

model = AutoModel.from_pretrained("gpt2")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

result = measure_embedding_condensation(
    model,
    tokenizer,
    repetitions=10,
    max_length=512,
    dataset="wikipedia",
    min_word_count=1024,
    max_word_count=1280,
    plot=True,
    show_progress=True,
    save_path="./test_embedding_condensation.png",
)
print(result.cossim_by_layer.shape)
```

## Citation
```bibtex
@inproceedings{liu2026dispersion,
  title={Dispersion loss counteracts embedding condensation and improves generalization in small language models},
  author={Liu, Chen and Sun, Xingzhi and Xiao, Xi and Van Tassel, Alexandre and Xu, Ke and Reimann, Kristof and Liao, Danqi and Gerstein, Mark and Wang, Tianyang and Wang, Xiao and Krishnaswamy, Smita},
  booktitle={International conference on machine learning},
  year={2026},
  organization={PMLR}
}
```

## Acknowledgements
1. This work was initially motivated by the paper "[A mathematical perspective on Transformers](https://arxiv.org/abs/2312.10794)". We started this project early Apr 2025 after we watched [a talk on that paper](https://www.youtube.com/watch?v=3McmEtA3t_0).
2. The design of the dispersion loss was largely inspired by [Runqian](https://raywang4.github.io/) and [Kaiming](https://scholar.google.com/citations?user=DhtAFkwAAAAJ)'s paper "[Diffuse and Disperse: Image Generation with Representation Regularization](https://arxiv.org/abs/2506.09027)".
