import os
import argparse
import datetime
import torch
import numpy as np
from tqdm import tqdm
from logging import getLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ssr_eval.metrics import AudioMetrics

import tage.audio as Audio
from tage.metrics.factory import MetricOutput
from tage.metrics.fad import FrechetAudioDistance
from tage import calculate_fid, calculate_isc, calculate_kid, calculate_kl, LPAPS, CLAPScore
from tage.modules.panns import Cnn14
from tage.audio.tools import save_pickle, load_pickle, write_json, load_json
from tage.datasets.load_mel import load_npy_data, MelPairedDataset, WaveDataset


log = getLogger(__name__)

class EvaluationHelper:
    def __init__(self, sampling_rate, device, backbone="cnn14") -> None:
        self.device = device
        self.backbone = backbone
        self.sampling_rate = sampling_rate
        self.lpaps = LPAPS()
        self.csc = CLAPScore()
        self.frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )

        self.lsd_metric = AudioMetrics(self.sampling_rate)
        self.frechet.model = self.frechet.model.to(device)

        features_list = ["2048", "logits"]
        if self.sampling_rate == 16000:
            self.mel_model = Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
        elif self.sampling_rate == 32000:
            self.mel_model = Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )

        if self.sampling_rate == 16000:
            self._stft = Audio.TacotronSTFT(512, 160, 512, 64, 16000, 50, 8000)
        elif self.sampling_rate == 32000:
            self._stft = Audio.TacotronSTFT(1024, 320, 1024, 64, 32000, 50, 14000)
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )

        self.mel_model.eval()
        self.mel_model.to(self.device)
        self.fbin_mean, self.fbin_std = None, None

    def main(
        self,
        generate_files_path,
        groundtruth_path,
        reference_text_path=None,
        calculate_psnr_ssim=True,
        calculate_lsd=False,
        recalculate=False,
        limit_num=None,
    ):
        log.info(f"Generted files {generate_files_path}")
        log.info(f"Target files {groundtruth_path}")

        self.file_init_check(generate_files_path)
        self.file_init_check(groundtruth_path)

        same_name = self.get_filename_intersection_ratio(
            generate_files_path, groundtruth_path, limit_num=limit_num
        )

        metrics = self.calculate_metrics(
            generate_files_path,
            groundtruth_path,
            same_name,
            reference_text_path=reference_text_path,
            limit_num=limit_num,
            calculate_psnr_ssim=calculate_psnr_ssim,
            calculate_lsd=calculate_lsd,
            recalculate=recalculate,
        )

        return metrics

    def file_init_check(self, dir):
        assert os.path.exists(dir), "The path does not exist %s" % dir
        assert len(os.listdir(dir)) > 1, "There is no files in %s" % dir

    def get_filename_intersection_ratio(
        self, dir1, dir2, threshold=0.99, limit_num=None
    ):
        self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
        self.datalist1 = sorted(self.datalist1)

        self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
        self.datalist2 = sorted(self.datalist2)

        data_dict1 = {os.path.basename(x): x for x in self.datalist1}
        data_dict2 = {os.path.basename(x): x for x in self.datalist2}

        keyset1 = set(data_dict1.keys())
        keyset2 = set(data_dict2.keys())

        intersect_keys = keyset1.intersection(keyset2)
        if (
            len(intersect_keys) / len(keyset1) > threshold
            and len(intersect_keys) / len(keyset2) > threshold
        ):
            log.info(
                f"Two path have {len(intersect_keys)} intersection files out of total {len(keyset1)} & {len(keyset2)} files. Processing two folder with same_name=True"
                )
            return True
        else:
            log.info(
                f"Two path have {len(intersect_keys)} intersection files out of total {len(keyset1)} & {len(keyset2)} files. Processing two folder with same_name=False"
                )
            return False

    def calculate_lsd(self, pairedloader, same_name=True, time_offset=160*7, eps=1e-8):
        if same_name == False:
            return {
                "lsd": -1,
                "ssim_stft": -1,
            }
        log.info(f"Calculating LSD using a time offset of {time_offset} ...")
        lsd_avg = []
        ssim_stft_avg = []
        for _, _, filename, (audio1, audio2) in tqdm(pairedloader):
            audio1 = audio1.cpu().numpy()[0, 0]
            audio2 = audio2.cpu().numpy()[0, 0]

            # If you use HIFIGAN (verified on 2023-01-12), you need seven frames' offset
            audio1 = audio1[time_offset:]

            audio1 = audio1 - np.mean(audio1)
            audio2 = audio2 - np.mean(audio2)

            audio1 = audio1 / (np.max(np.abs(audio1)) + eps)
            audio2 = audio2 / (np.max(np.abs(audio2)) + eps)

            min_len = min(audio1.shape[0], audio2.shape[0])

            audio1, audio2 = audio1[:min_len], audio2[:min_len]

            result = self.lsd(audio1, audio2)

            lsd_avg.append(result["lsd"])
            ssim_stft_avg.append(result["ssim"])

        return {"lsd": MetricOutput(mean=np.mean(lsd_avg)), "ssim_stft": MetricOutput(mean=np.mean(ssim_stft_avg))}

    def lsd(self, audio1, audio2):
        result = self.lsd_metric.evaluation(audio1, audio2, None)
        return result

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if same_name == False:
            return {'psnr': MetricOutput(mean=-1), 'ssim': MetricOutput(mean=-1)}
        
        psnr_avg = []
        ssim_avg = []
        for mel_gen, mel_target, filename, _ in tqdm(pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if np.isinf(psnrval):
                log.info(f"Infinite value encountered in psnr {filename}.")
                continue
            psnr_avg.append(psnrval)
            data_range = max(np.max(mel_gen), np.max(mel_target)) - min(
                np.min(mel_gen), np.min(mel_target)
            )
            ssim_avg.append(ssim(mel_gen, mel_target, data_range=data_range))

        return {'psnr': MetricOutput(mean=np.mean(psnr_avg)), 'ssim': MetricOutput(mean=np.mean(ssim_avg))}

    def calculate_metrics(
        self,
        generate_files_path,
        groundtruth_path,
        same_name,
        reference_text_path=None,
        limit_num=None,
        calculate_psnr_ssim=False,
        calculate_lsd=False,
        recalculate=False,
        num_workers=6,
    ):
        torch.manual_seed(42)

        outputloader = DataLoader(
            WaveDataset(
                generate_files_path,
                self.sampling_rate,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        resultloader = DataLoader(
            WaveDataset(
                groundtruth_path,
                self.sampling_rate,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )


        out, summary = OmegaConf.create(), []

        r"""Calcualte CLAP score."""
        clap_score = self.csc(
            generated_audio_path=generate_files_path, 
            reference_text_path=reference_text_path,
            recalculate=recalculate,
            )
        # out.update({f"clap_score_{k}": v for k, v in clap_score.to_dict().items()})
        out.update({"CLAP_Score": clap_score.to_dict()})
        summary.append(f"CLAP_Score: {clap_score.result}.")

        r"""Calculate Perceptual Simmilarity (LPAPS)."""
        lpaps_score = self.lpaps(
            generated_audio_path=generate_files_path,
            reference_audio_path=groundtruth_path,
            recalculate=recalculate,
        )
        out.update({'Perceptual similary (LPAPS)': lpaps_score.to_dict()})
        summary.append(f"LPAPS: {lpaps_score.result}")

        r"""Calculate Frechet Audio Distance (FAD)."""
        if recalculate:
            log.info("Calculate FAD score from scratch.")
        fad_score = self.frechet.score(
            generate_files_path,
            groundtruth_path,
            limit_num=limit_num,
            recalculate=recalculate,
        )
        out.update({'Frechet Audio Distance (FAD)': fad_score.to_dict()})
        summary.append(f"FAD: {fad_score.result}")


        r"""Calculate KL Divergence(KL)."""
        cache_path = groundtruth_path + "classifier_logits_feature_cache.pkl"
        if os.path.exists(cache_path) and not recalculate:
            log.info(f"reload {cache_path}")
            featuresdict_2 = load_pickle(cache_path)
        else:
            log.info(f"Extracting features from {groundtruth_path}.")
            featuresdict_2 = self.get_featuresdict(resultloader)
            save_pickle(featuresdict_2, cache_path)

        cache_path = generate_files_path + "classifier_logits_feature_cache.pkl"
        if os.path.exists(cache_path) and not recalculate:
            log.info(f"reload {cache_path}")
            featuresdict_1 = load_pickle(cache_path)
        else:
            log.info(f"Extracting features from {generate_files_path}.")
            featuresdict_1 = self.get_featuresdict(outputloader)
            save_pickle(featuresdict_1, cache_path)

        kl, kl_ref, paths_1 = calculate_kl(
            featuresdict_1, featuresdict_2, "logits", same_name
        )

        out.update({'Kullback Leibler divergence (KL)': kl['kl_softmax'].to_dict(), 'KL sigmoid': kl['kl_sigmoid'].to_dict()})
        summary.append(f"KL: {kl['kl_softmax'].result}, KL_sigmoid: {kl['kl_sigmoid'].result}")

        r"""Calculate Inception Score (IS)."""
        metric_isc = calculate_isc(
            featuresdict_1,
            feat_layer_name="logits",
            splits=10,
            samples_shuffle=True,
            rng_seed=2020,
        )
        out.update({'Inception Score (IS)': metric_isc.to_dict()})
        summary.append(f'IS: {metric_isc.result}')

        r"""Calculate FD."""
        # if "2048" in featuresdict_1.keys() and "2048" in featuresdict_2.keys():
        #     metric_fid = calculate_fid(
        #         featuresdict_1, featuresdict_2, feat_layer_name="2048"
        #     )
        #     out.update({'Frechet Distance (FD)': metric_fid.to_dict()})
        #     summary.append(f'FD: {metric_fid.result}')

        r"""Calculate psnr, ssim, lad with paired data."""
        if calculate_psnr_ssim or calculate_lsd:
            pairedloader = DataLoader(
                MelPairedDataset(
                    generate_files_path,
                    groundtruth_path,
                    self._stft,
                    self.sampling_rate,
                    self.fbin_mean,
                    self.fbin_std,
                    limit_num=limit_num,
                ),
                batch_size=1,
                sampler=None,
                num_workers=16,
            )

        if calculate_lsd:
            metric_lsd = self.calculate_lsd(pairedloader, same_name=same_name)
            out.update({
                'Log-Spectrogram Distance (LSD)': metric_lsd['lsd'].to_dict(),
                'SSIM on Spectrogram': metric_lsd['ssim_stft'].to_dict(),})
            summary.append(f"LSD: {metric_lsd['lsd'].result}, SSIM_on_Spec: {metric_lsd['ssim_stft'].result}")

        if calculate_psnr_ssim:
            psnr_ssim = self.calculate_psnr_ssim(
                pairedloader, same_name=same_name
            )
            out.update({
                'Peak Signal-to-Noise Ratio (PSNR)': psnr_ssim['psnr'].to_dict(), 
                'Structural Similarity (SSIM)': psnr_ssim['ssim'].to_dict()})
            summary.append(f"PSNR: {psnr_ssim['psnr'].result}, SSIM: {psnr_ssim['ssim'].result}")

        metric_kid = calculate_kid(
            featuresdict_1,
            featuresdict_2,
            feat_layer_name="2048",
            subsets=100,
            subset_size=1000,
            degree=3,
            gamma=None,
            coef0=1,
            rng_seed=2020,
        )
        out.update({'Kernel Inception Distance (KID)': metric_kid.to_dict()})
        summary.append(f"KID: {metric_kid.result}")

        log.info(OmegaConf.to_yaml(out))
        log.info('=====Summary=====')
        num = len(generate_files_path) if limit_num is None else limit_num
        log.info(f'Evaluate the model on {num} audio files:')
        log.info(', '.join(summary))

        json_path = os.path.join(
            os.path.dirname(generate_files_path),
            self.get_current_time()
            + "_"
            + os.path.basename(generate_files_path)
            + ".json",
        )
        out = OmegaConf.to_container(out, resolve=True)
        write_json(out, json_path)
        return out

    def get_current_time(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d-%H:%M:%S")

    def get_featuresdict(self, dataloader):
        out = None
        out_meta = None

        for waveform, filename in tqdm(dataloader):
            try:
                metadict = {
                    "file_path_": filename,
                }
                waveform = waveform.squeeze(1)

                waveform = waveform.float().to(self.device)

                with torch.no_grad():
                    featuresdict = self.mel_model(waveform)  # "logits": [1, 527]

                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

                if out is None:
                    out = featuresdict
                else:
                    out = {k: out[k] + featuresdict[k] for k in out.keys()}

                if out_meta is None:
                    out_meta = metadict
                else:
                    out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

            except Exception as e:
                log.warning(f"Classifier Inference error: {e}")
                continue

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[: samples.shape[0]], :]


if __name__ == "__main__":
    import argparse
    import torch
    import logging

    from tage import EvaluationHelper


    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s]%(name)s:\n%(message)s',
        handlers=[
            # logging.FileHandler('eval.log'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--generated-audio-dir", type=str, 
        help="folder of generation result."
    )
    parser.add_argument(
        "-t", "--target-audio-dir", type=str, 
        help="folder of reference result."
    )
    parser.add_argument(
        "-r", "--reference_text_path", type=str, 
        help="Reference audio description during evaluation."
    )
    parser.add_argument(
        "-sr", "--sampling_rate", type=int, default=16000, 
        help="audio sampling rate."
    )
    parser.add_argument(
        "-l",
        "--limit_num",
        type=int,
        required=False,
        help="Audio clip numbers limit for evaluation",
        default=None,
    )
    parser.add_argument(
        "--recalculate", action="store_true", default=False, 
        help="Recalculate metrics when applicable."
    )
    args = parser.parse_args()


    evaluator = EvaluationHelper(
        sampling_rate=args.sampling_rate, 
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

    metrics = evaluator.main(
        args.generated_audio_dir,
        args.target_audio_dir,
        reference_text_path=args.reference_text_path,
        limit_num=args.limit_num,
    )
    # import ipdb; ipdb.set_trace()