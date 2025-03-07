import librosa
from scipy import signal
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import numpy as np
from praatio import textgrid
import os
import argparse
import torch
from beat_detector import BD

parser = argparse.ArgumentParser()

parser.add_argument('-in', '--audio_dir', type=str, help='directory with audio files')
parser.add_argument('-out', '--output_dir', type=str, help='directory to save output')
parser.add_argument('-l', '--bandpass_left', type=int, help='bandpass filters left cutoff')
parser.add_argument('-r', '--bandpass_right', type=int, help='bandpass filters right cutoff')
parser.add_argument('-sr', '--sampling_rate', type=int, help='audio sampling rate, automatic resampling is performed if specified sr is different from native')
parser.add_argument('-tg', '--save_textgrids', action='store_true', help='whether the output should be textgrid files')

args = parser.parse_args()

bd = BD( sr = args.sampling_rate, bandpass_left=args.bandpass_left, bandpass_right=args.bandpass_right)

timestamps = {}

for filename in os.listdir(args.audio_dir):
  filepath = os.path.join(args.audio_dir, filename)
  waveform, sr = librosa.load(filepath, sr = args.sampling_rate)
  beats = bd.detect_beats(waveform)

  if args.save_textgrids:
    tg_path = os.path.join(args.output_dir, 'textgrids')
    os.makedirs(tg_path, exist_ok=True)
    out_file = os.path.join(tg_path, os.path.splitext(filename)[0] + ".TextGrid")
    bd.save_textgrid(beats, out_file)

  else:
    timestamps[filename] = beats

torch.save(timestamps, os.path.join(args.output_dir, "beat_timestamps.pt"))