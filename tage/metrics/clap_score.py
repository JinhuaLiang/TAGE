from __future__ import absolute_import

import os
import torch
import json
import librosa
import laion_clap
import numpy as np
import torch.nn.functional as F
from logging import getLogger
from typing import List, Optional, Union, Tuple
from pathlib import Path
from torch import nn
from tqdm import tqdm
from clap_module.factory import load_state_dict
from transformers import RobertaTokenizer

from tage.metrics.factory import MetricMixin, MetricOutput
from tage.modules.clap import CLAP_base


home_dir = os.path.expanduser("~")
log = getLogger(__name__)

class CLAPScore(nn.Module, MetricMixin):
    r"""Implementing CLAP score by calculating similarity between text and audio embeddings."""
    def __init__(self, 
                 model_path: Union[str, Path] = f"{home_dir}/.cache/tage/ckpt/music_speech_audioset_epoch_15_esc_89.98.pt", 
                 model_arch: str = 'HTSAT-base', 
                 enable_fusion: bool = False,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        super(CLAPScore, self).__init__()
        self._initialize_model(model_path, model_arch, enable_fusion)
        self.model.to(device)
        self.model.device = device
 
    def _initialize_model(self, model_path: Union[str, Path], model_arch: str, enable_fusion: bool):
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=model_arch)
        self.model_sample_rate = 48_000
        self.load_clap_state_dict(self.model, model_path)
        self.model.eval()

    def read_audio(self, file_path: str):
        # Load the waveform of the shape (T,), should resample to 48000
        audio_waveform, sr = librosa.load(file_path, sr=48000)           
        # Quantization
        audio_waveform = self.int16_to_float32(self.float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float().to(self.model.device)

        return audio_waveform, sr
    
    def repeat_pad(self, audio, target_length):
        repeat_time = target_length // audio.size(-1) + 1
        audio = audio.repeat(repeat_time)[:target_length]
        return audio

    def batch_encode_audio_with_mask(self, audio, mask=[None]):
        r"""Stack audio into the shape of [B, T]."""
        if mask[0] is not None:
            max_length = audio[0].size(-1)
            audio = [self.repeat_pad(a[m.bool()], max_length) for a, m in zip(audio, mask)]
        audio = torch.stack(audio) 
        audio_embeds = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
        return audio_embeds

    def tokenizer(self, text: Union[str, List[str]]) -> dict:
        # Use the default params from CLAP module here as well
        return self.tokenize(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

    def get_paired_captions_with_window(self, meta, audio, threshold=0.99):
        # Assume meta is a json file or data structure
        # e.g., {`audio_basename`: {'caption': , 'window':}}
        captions, windows, missing_files = [], [], []
        for aud in audio:
            aud_basename = os.path.basename(aud)
            try:
                cap = meta[aud_basename]['caption']  # type str
                win = meta[aud_basename].get('window', None)
                captions.append(cap)
                windows.append(win)
            except KeyError as e:
                missing_files.append(aud)

        if len(captions) / len(audio) < threshold:
            raise Exception(f"captions and audio do not match. Missing audio file {missing_files}")
        
        return captions, windows
    
    @torch.no_grad()
    def forward(self,
                generated_audio_path: Union[list, str], 
                reference_text_path: Union[dict, str, None] = None,
                text_template='A sound of {}',
                recalculate=False,):
        r"""Text should be store in the json file in the format."""
        if reference_text_path is None or not os.path.exists(reference_text_path):
            return MetricOutput(mean=-1)
        
        gen_cache_dir = self.get_cache_dir(generated_audio_path)
        gen_cache_pth = gen_cache_dir + '-clap_score.npy'
        gen_audio_files = self.maybe_collate_data(generated_audio_path)

        reference_meta = self.load_json(reference_text_path)
        ref_captions, windows = self.get_paired_captions_with_window(reference_meta, gen_audio_files)

        if windows[0] is None:
            log.warning(f"CLAP score is now calculated on the whole segment. Please check if this is expected.")
            gen_cache_pth = gen_cache_pth[:-4] + '-without_win.npy'
             
        if not os.path.exists(gen_cache_pth) or recalculate:
            log.info(f"Getting embeddings from {gen_cache_dir}.")
            with tqdm(total=len(gen_audio_files)) as pbar:
                gen_waveforms, gen_masks = [], []
                for gen_f, win in zip(gen_audio_files, windows):
                    gen_wav, gen_sr = self.read_audio(gen_f)
                    gen_m = self.maybe_generate_mask(win, gen_wav, gen_sr)
                    gen_waveforms.append(gen_wav)
                    gen_masks.append(gen_m)
                    pbar.update(1)
                gen_embeds = self.batch_encode_audio_with_mask(gen_waveforms, mask=gen_masks)
                del gen_waveforms
            self.dump_intermediate_result(gen_embeds.detach().cpu(), gen_cache_pth)
        else:
            gen_embeds = self.load_intermediate_result(gen_cache_pth)
            gen_embeds = torch.tensor(gen_embeds.tolist(), device=self.model.device)

        ref_embeds = self.model.get_text_embedding(
            ref_captions, tokenizer=self.tokenizer, use_tensor=True)

        # cosine similarity between the text and the audio embedding
        cosine_sim = F.cosine_similarity(gen_embeds, ref_embeds, dim=1)

        return MetricOutput(
            mean=cosine_sim.mean(0).item(),
            std=cosine_sim.std(0).item(),
            )

    @staticmethod
    def load_clap_state_dict(clap_model, path: Union[str, Path]):
        """Wrapper around state dict loading of CLAP model
        addressing compatibility issues between CLAP and AudioCraft
        HuggingFace transformer version.
        See: https://github.com/LAION-AI/CLAP/issues/118
        """
        pkg = load_state_dict(path)
        pkg.pop('text_branch.embeddings.position_ids', None)
        clap_model.model.load_state_dict(pkg)
        del pkg

    def load_json(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
        
    @staticmethod
    def int16_to_float32(x):
        return (x / 32767.0).astype(np.float32)

    @staticmethod
    def float32_to_int16(x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)


if __name__ == "__main__":
    # from engine import load_json, write_json

    # def generate_json(json_pth, out_pth):
    #     ori_data = load_json(json_pth)
    #     out = {}
    #     for datum in ori_data:
    #         basename = os.path.basename(datum['target_audio']['audio_path'])
    #         cap = datum['edit']['event']
    #         win = datum['edit']['timestamps']
    #         out[basename] = {'caption': cap, 'window': win}
    #     write_json(out, out_pth)

    # generate_json(
    #     json_pth='/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val.json',
    #     out_pth='/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt-with_win.json',
    #     )

    csc = CLAPScore()
    
    # # Use paired data by specifying dir without windows
    res = csc(
        generated_audio_path='/data/scratch/eey340/data_package/SoundEdit-TrainingFree/style_transfer/generation', 
        reference_text_path='/data/EECS-MachineListeningLab/datasets/AudioSet-E/dataset/mldbdown/eval-without_source_caption-subset.json',
        )
    print(res)
    # # Use paired data by loading file lists with windows
    # res = csc(
    #     generated_audio_path='/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/mini_mini_data', 
    #     reference_text_path='/mnt/bn/jliang-lq-nas/workplace/AudioSet-E/dataset/add/val-gt.json',
    #     )
   
    import ipdb; ipdb.set_trace()