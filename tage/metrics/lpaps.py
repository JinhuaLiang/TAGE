from __future__ import absolute_import

import os
import torch
import librosa
import glob
import numpy as np
from logging import getLogger
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from torch import nn
from tqdm import tqdm

from tage.metrics.factory import MetricMixin, MetricOutput
from tage.modules.clap import CLAP_base


log = getLogger(__name__)


r"""Modified from https://github.com/HilaManor/AudioEditingCode/blob/codeclean/evals/lpaps.py."""
class LPAPS(nn.Module, MetricMixin):
    def __init__(self, net: str = 'clap', device: str = 'cpu' if torch.cuda.is_available() else 'cuda',
                 net_kwargs: dict = {'model_arch': 'HTSAT-base',
                             'chkpt': 'music_speech_audioset_epoch_15_esc_89.98.pt', # we use another ckpt to fit environmental sound
                             'enable_fusion': False}, 
                 req_grad=False, spatial=False):
        
        r"""Initializes a perceptual loss torch.nn.Module
            :param str net: the network to use
            :param str device: the device to use
            :param dict net_kwargs: the network keyword arguments
            :param str checkpoint_path: the checkpoint path
            :param bool req_grad: whether to require gradients
            :param bool spatial: whether to use spatial information
        """

        super(LPAPS, self).__init__()

        self.pnet_type = net
        self.extra_kwargs = net_kwargs
        self.req_grad = req_grad

        self.spatial = spatial

        if (self.pnet_type in 'clap'):
            net_type = CLAP_base

        self.net = net_type(requires_grad=self.req_grad, device=device, **self.extra_kwargs)
        self.L = self.net.get_num_layers()
        self.eval()

    def read_audio(self, file_path: str):
        # Load the waveform of the shape (T,), should resample to 48000
        audio_waveform, sr = librosa.load(file_path, sr=48000)           
        # Quantization
        audio_waveform = self.int16_to_float32(self.float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float().to(self.net.device)

        return audio_waveform, sr

    def forward(self, 
            generated_audio_path: Union[list, str], 
            reference_audio_path: Union[list, str],
            generated_edit_duration: Optional[Tuple[float]] = None,
            reference_edit_duration: Optional[Tuple[float]] = None,
            recalculate=False,
        ):
        gen_cache_dir = self.get_cache_dir(generated_audio_path)
        gen_cache_pth = gen_cache_dir + '-lpaps.npy'
        ref_cache_dir = self.get_cache_dir(reference_audio_path)
        ref_cache_pth = ref_cache_dir + '-lpaps.npy'

        gen_audio_files = self.maybe_collate_data(generated_audio_path)
        ref_audio_files = self.maybe_collate_data(reference_audio_path)

        gen_audio_files, ref_audio_files = self.get_paired_audio(
            gen_audio_files, ref_audio_files,threshold=0.99, limited_num=None
            )

        if gen_audio_files is None:
            log.warning("Cannot compute LPAPS score because File names do NOT match.")
            return -1

        if generated_edit_duration is None:
            generated_edit_duration = [None] * len(gen_audio_files)
        if reference_edit_duration is None:
            reference_edit_duration = [None] * len(ref_audio_files)
             
        if not os.path.exists(gen_cache_pth) or recalculate:
            log.info(f"Getting embeddings from {gen_cache_dir}.")
            gen_embeds = []
            with tqdm(total=len(gen_audio_files)) as pbar:
                for gen_f, gen_duration in zip(gen_audio_files, generated_edit_duration):
                    gen_wav, gen_sr = self.read_audio(gen_f)  # (T,)
                    gen_m = self.maybe_generate_mask(gen_duration, gen_wav, gen_sr)
                    gen_emb = self.get_emb_with_window(gen_wav, gen_sr, gen_m)
                    gen_embeds.append(gen_emb)
                    pbar.update(1)
            self.dump_intermediate_result(gen_embeds, gen_cache_pth)
        else:
            gen_embeds = self.load_intermediate_result(gen_cache_pth)

        if not os.path.exists(ref_cache_pth) or recalculate:
            log.info(f"Getting embeddings from {ref_cache_dir}.")
            ref_embeds = []
            with tqdm(total=len(ref_audio_files)) as pbar:
                for ref_f, ref_duration in zip(ref_audio_files, reference_edit_duration):
                    ref_wav, ref_sr = self.read_audio(ref_f)
                    ref_m = self.maybe_generate_mask(ref_duration, ref_wav, ref_sr)
                    ref_emb = self.get_emb_with_window(ref_wav, ref_sr, ref_m)
                    ref_embeds.append(ref_emb)
                    pbar.update(1)
            self.dump_intermediate_result(ref_embeds, ref_cache_pth)
        else:
            ref_embeds = self.load_intermediate_result(ref_cache_pth)

        log.info("Computing LPAPS score...")
        scores = []
        with tqdm(total=len(ref_audio_files)) as pbar:
            for gen_emb, ref_emb in zip(gen_embeds, ref_embeds):
                s, _ = self.compute_lpaps(gen_emb, ref_emb)
                scores.append(s.item())
                pbar.update(1)

        return MetricOutput(mean=np.mean(scores), std=np.std(scores, ddof=1))

    def get_emb_with_window(self, audio: torch.Tensor, sr: int, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.ones_like(audio)

        # '*' op is used to avoid changing the shape of model input
        windowed_aud = audio * mask

        outs = self.net.forward(
            windowed_aud.unsqueeze(0).unsqueeze(0), torch.tensor([sr], device=self.net.device))
        
        return outs
    
    def compute_lpaps(self, emb1, emb2, output_size=None):
        r"""Compute lpaps between `emb1` and `emb2`.
        :param: tuple: size of output tensor for upsampling. Only work when 'self.spatial' is True.
        """
        if self.spatial and output_size is None: 
            raise Exception('`output_size` required when `sptial` is True')
        
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = self.normalize_tensor(emb1[kk]), self.normalize_tensor(emb2[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if (self.spatial):
            res = [self.upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=output_size.shape[2:]) for kk in range(self.L)]
        else:
            res = [self.spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        return val, res

    @staticmethod
    def normalize_tensor(in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
        return in_feat/(norm_factor+eps)

    @staticmethod
    def spatial_average(in_tens, keepdim=True):
        return in_tens.mean([1, 2], keepdim=keepdim)

    @staticmethod
    def upsample(in_tens, out_HW=(64, 64)):
        return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)
    
    @staticmethod
    def int16_to_float32(x):
        return (x / 32767.0).astype(np.float32)

    @staticmethod
    def float32_to_int16(x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)


if __name__ == "__main__":
    # from engine import debugger
    from engine import load_json

    lpaps = LPAPS()
    # Use paired data by specifying dir without windows
    res = lpaps(
        generated_audio_path="/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/remove/mini_data", 
        reference_audio_path="/mnt/bn/jliang-lq-nas/workplace/SoundEdit-TrainingFree/outputs/remove/remove-TEST-noTan-withText-ES1.6-New",
        )
    # Use paired data by loading file lists with windows
    json_path = "/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/remove/mini_data/val.json"
    data = load_json(json_path)
    duration_dict = {}
    for datum in data:
        duration_dict[datum["target_audio"]["audio_path"]] = datum["edit"]["timestamps"]

    gen_audio_files = lpaps.maybe_collate_data("/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/remove/mini_data")
    ref_audio_files = lpaps.maybe_collate_data("/mnt/bn/jliang-lq-nas/workplace/SoundEdit-TrainingFree/outputs/remove/remove-TEST-noTan-withText-ES1.6-New")
    durations = [duration_dict[f] for f in gen_audio_files]
    res = lpaps(
        generated_audio_path="/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/remove/mini_data", 
        reference_audio_path="/mnt/bn/jliang-lq-nas/workplace/SoundEdit-TrainingFree/outputs/remove/remove-TEST-noTan-withText-ES1.6-New",
        generated_edit_duration=durations,
        reference_edit_duration=durations,
        )
    import ipdb; ipdb.set_trace()