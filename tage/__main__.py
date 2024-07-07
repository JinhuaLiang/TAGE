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