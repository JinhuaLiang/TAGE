r"""Creates a set of audio files to test measurement calculation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import errno
import scipy.io.wavfile

from absl import app
from absl import flags
import numpy as np


_SAMPLE_RATE = 16000

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "test_files", "", "Directory where the test files should be located"
)

current_dir = os.path.dirname(os.path.abspath(__file__))


def create_dir(output_dir):
    """Ignore directory exists error."""
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise


def add_noise(data, stddev):
    """Adds Gaussian noise to the samples.

    Args:
      data: 1d Numpy array containing floating point samples. Not necessarily
        normalized.
      stddev: The standard deviation of the added noise.

    Returns:
       1d Numpy array containing the provided floating point samples with added
       Gaussian noise.

    Raises:
      ValueError: When data is not a 1d numpy array.
    """
    if len(data.shape) != 1:
        raise ValueError("expected 1d numpy array.")
    max_value = np.amax(np.abs(data))
    num_samples = data.shape[0]
    gauss = np.random.normal(0, stddev, (num_samples)) * max_value
    return data + gauss


def gen_sine_wave(freq=600, length_seconds=6, sample_rate=_SAMPLE_RATE, param=None):
    """Creates sine wave of the specified frequency, sample_rate and length."""
    t = np.linspace(0, length_seconds, int(length_seconds * sample_rate))
    samples = np.sin(2 * np.pi * t * freq)
    if param:
        samples = add_noise(samples, param)
    return np.asarray(2**15 * samples, dtype=np.int16)


def main(argv):
    del argv  # Unused.
    meta = {}
    for traget, count, param in [
        ("reference", 50, 0.0),
        ("paired", 50, 0.001),
        ("unpaired", 25, 0.001),
    ]:
        output_dir = os.path.join(FLAGS.test_files, "example", traget)
        output_json = os.path.join(FLAGS.test_files, "example/reference_captions.json")
        create_dir(output_dir)
        print("output_dir:", output_dir)
        frequencies = np.linspace(100, 1000, count).tolist()
        for freq in frequencies:
            samples = gen_sine_wave(freq, param=param)
            filename = os.path.join(current_dir, output_dir, "sin_%.0f.wav" % freq)
            print("Creating: %s with %i samples." % (filename, samples.shape[0]))
            scipy.io.wavfile.write(filename, _SAMPLE_RATE, samples)
            meta[filename] = {
                'caption': f"A {freq:.2f}Hz sine wave.",
                'window': [0, 10]
            }

    with open(output_json, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Write reference data and windows into {output_json}")
        


if __name__ == "__main__":
    os.makedirs("example", exist_ok=True)
    app.run(main)
