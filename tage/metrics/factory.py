import os
import glob
import torch
import numpy as np
from dataclasses import dataclass
from typing import Union
from random import randint


@dataclass
class MetricOutput:
    mean: float
    std: float = None

    @property
    def result(self,):
        if self.std is not None:
            return f"{self.mean:.5f}±{self.std:.5f}"
        else:
            return f"{self.mean:.5f}"

    def to_dict(self,):
        if self.std is not None:
            return {'mean': float(self.mean), 'std': float(self.std)}
        else:
            return {'mean': float(self.mean)}


class MetricMixin:
    def get_cache_dir(self, audio_path: Union[str, list]):
        if isinstance(audio_path, str):
            return audio_path
        elif isinstance(audio_path, list):
            audio_dir = audio_path[0].split('/')[-1]
            audio_dir = os.path.join(*audio_dir)
            return audio_dir
        else:
            raise Exception(f'Cannot handle the data type {type(audio_path)}.')
    def maybe_collate_data(self, data_dir: Union[str, list]):
        if isinstance(data_dir, list):
            return data_dir
        
        wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
        return wav_files
    
    def dump_intermediate_result(self, data, dump_path):
        np.save(dump_path, data)

    def load_intermediate_result(self, path):
        return np.load(path, allow_pickle=True)

    def get_paired_audio(self, audio1, audio2, threshold=0.99, limited_num=None):
        r"""Get the paired audio with the same filename in `audio1` and `audio2`."""
        audio1_files = sorted(audio1)
        audio2_files = sorted(audio2)

        audio1_loc = {os.path.basename(x): x for x in audio1_files}
        audio2_loc = {os.path.basename(x): x for x in audio2_files}

        audio1_basename = set(audio1_loc.keys())
        audio2_basename = set(audio2_loc.keys())

        joint_audio_basename = audio1_basename.intersection(audio2_basename)
        if (
            len(joint_audio_basename) / len(audio1_basename) < threshold
            or len(joint_audio_basename) / len(audio2_basename) < threshold
        ):
            return None

        audio1_files = [audio1_loc[n] for n in joint_audio_basename]
        audio2_files = [audio2_loc[n] for n in joint_audio_basename]

        if limited_num is not None:
            audio1_files = audio1_files[:limited_num]
            audio2_files = audio2_files[:limited_num]

        return audio1_files, audio2_files

    def maybe_generate_mask(self, window, audio, sr):
        if window is None:
            return None
    
        t_on, t_off = window
        m = torch.zeros_like(audio)
        s_on, s_off = int(t_on*sr), int(t_off*sr)
        m[s_on:s_off] += 1

        return m

    def get_duration(self, window):
        stt, end = window
        return (end - stt)
    
    def truncate_window(self, window, target_duration, enable_random=True):
        ori_stt, ori_end = window
        if enable_random:
            new_stt = randint(ori_stt, ori_end-target_duration)
        else:
            new_stt = 0
        new_end = new_stt + target_duration
        return new_stt, new_end

    def get_paired_windows(self, win1=None, win2=None, mode="union"):
        if win1 is None and win2 is None:
            return None, None
        elif win1 is None:
            return win2, win2
        elif win2 is None:
            return win1, win1
        
        du1, du2 = self.get_duration(win1), self.get_duration(win2)

        if du1 < du2:
            win2 = self.truncate_window(win2, du1, enable_random=True)
        else:
            win1 = self.truncate_window(win1, du2, enable_random=True)
        return win1, win2
