<h1 align="center">
<a href="https://interactiveaudiolab.github.io/assets/papers/oreilly_awasthi_vijayaraghavan_pardo_2021.pdf">
Inconspicuous and Effective Over-the-Air Adversarial Examples via Adaptive Filtering
</a>
</h1>

<h4 align="center"><i>ICASSP</i> '22</h4>

<div align="center">
<h4>
    <p>
    by <em>Patrick O'Reilly, Pranjal Awasthi, Aravindan Vijayaragavan, Bryan Pardo</em>
    </p>
</h4>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreillyp/filter_icassp_22/blob/master/notebooks/filter_icassp_22.ipynb)
[![Demo](https://img.shields.io/badge/Web-Demo-blue)](https://interactiveaudiolab.github.io/project/audio-adversarial-examples.html)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](/LICENSE)
    

</div>

<p align="center"><img src="https://interactiveaudiolab.github.io/assets/images/projects/filters.png" width="400"/></p>

We demonstrate a novel audio-domain adversarial attack that modifies benign audio using a time-varying FIR filter. Unlike existing state-of-the-art attacks, our proposed method does not require a complex optimization procedure or generative model, relying only on a simple variant of gradient descent to tune filter parameters. In order to craft attacks that work well in real-world playback scenarios (“over-the-air”), we optimize through simulated distortions that mimic the properties of acoustic environments (e.g. reverb, noise, time-domain offset).

## Citation
```
@inproceedings{oreilly2022filter,
        title={Inconspicuous and effective over-the-air adversarial examples via adaptive filtering},
        author={O'Reilly, Patrick and Awasthi, Pranjal and Vijayaragavan, Aravindan and Pardo, Bryan},
        booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        year={2022}}
```
