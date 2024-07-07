# TAGE: Bag of Metrics for Generated Audio Evaluation

This toolbox aims to evaluate audio generative model for a fair comparison.

## Quick Start

For now, ones should quickly setup environment using `Audioldm_eval` kit.
First, prepare the environment
```shell
pip install git+https://github.com/haoheliu/audioldm_eval
```

To evaluate the generated audio,
```shell
python tage \
  --generated-audio-dir '/directory/to/generated/audio' \
  --target-audio-dir '/directory/to/target/audio' \
  --reference_text_path '/path/to/reference/text/json'
```

## Evaluation metrics
We have the following metrics in this toolbox: 

- Metrics:
  - CLAP score: Correspondence between audio and text description
  - LPAPS: Distance between generated and target audio.
  - FAD: Frechet audio distance
  - ISc: Inception score
  - FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model
  - KID: Kernel inception score
  - KL: KL divergence (softmax over logits)
  - KL_Sigmoid: KL divergence (sigmoid over logits)
  - PSNR: Peak signal noise ratio
  - SSIM: Structural similarity index measure
  - LSD: Log-spectral distance

The evaluation function will accept the paths of two folders as main parameters. 
1. If two folder have **files with same name and same numbers of files**, the evaluation will run in **paired mode**.
2. If two folder have **different numbers of files or files with different name**, the evaluation will run in **unpaired mode**.

**These metrics will only be calculated in paried mode**: KL, KL_Sigmoid, PSNR, SSIM, LSD. 
In the unpaired mode, these metrics will return minus one.


## TODO

- [ ] Environment setup
- [ ] Test run
- [ ] finish doc

## Cite this repo

If you found this tool useful, please also consider citing
```bibtex
@article{liu2023audioldm,
  title={AudioLDM: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2301.12503},
  year={2023}
}
```
```bibtex
@misc{manor2024zeroshotunsupervisedtextbasedaudio,
      title={Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion}, 
      author={Hila Manor and Tomer Michaeli},
      year={2024},
      eprint={2402.10009},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2402.10009}, 
}
```
## Reference

> https://github.com/haoheliu/audioldm_eval
