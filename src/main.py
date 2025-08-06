import librosa
import os
import argparse
import numpy as np
from tqdm import tqdm
from beat_detector import BD

parser = argparse.ArgumentParser()

parser.add_argument('-in', '--audio_dir', type=str, required=True, help='directory with audio files')
parser.add_argument('-out', '--output_dir', type=str, required=True, help='directory to save output')
parser.add_argument('-sr', '--sampling_rate', type=int, required=True, help='audio sampling rate, automatic resampling is performed if specified sr is different from native')
parser.add_argument('-l', '--bandpass_left', type=int, default=800, help='bandpass filters left cutoff')
parser.add_argument('-r', '--bandpass_right', type=int, default=1500, help='bandpass filters right cutoff')
parser.add_argument('-tg', '--save_textgrids', action='store_true', help='if used, textgrid will be saved in [output_dir]/textgrids')
parser.add_argument('-w', '--save_filtered', action='store_true', help='if used, filtered audios will be saved in [output_dir]/filtered_wavs')
parser.add_argument('-plt', '--save_plots', action='store_true', help='if used, plots will be saved in [output_dir]/plots')

args = parser.parse_args()

bd = BD( sr = args.sampling_rate, bandpass_left=args.bandpass_left, bandpass_right=args.bandpass_right)

timestamps_data = {}
normTimestamps_data = {}

# Script will only read .wav files in the specified directory, ignoring subdirectories

file_list = [f"{args.audio_dir}/{f}" for f in os.listdir(args.audio_dir) if os.path.isfile(f"{args.audio_dir}/{f}") and f.split('.')[-1] == 'wav']

for i, file_path in enumerate(tqdm(file_list, desc="Processing audios")):
  waveform, sr = librosa.load(file_path, sr = args.sampling_rate)
  beats = bd.detect_beats(waveform)

  # Generate a unique ID
  identifier = os.path.basename(file_path)

  timestamps_data[identifier] = beats[0]
  normTimestamps_data[identifier] = beats[1]

  if args.save_textgrids:
    tg_path = os.path.join(args.output_dir, 'textgrids')
    os.makedirs(tg_path, exist_ok=True)
    out_file = os.path.join(tg_path, identifier[:-4] + ".TextGrid")
    bd.save_textgrid(beats[0], out_file)
  

  if args.save_filtered:
    filtered_path = os.path.join(args.output_dir, 'filtered_wavs')
    os.makedirs(filtered_path, exist_ok=True)
    out_file = os.path.join(filtered_path, identifier[:-4] + ".wav")
    bd.write_wav(out_file)

  if args.save_plots:
    plots_path = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    out_file = os.path.join(plots_path, identifier[:-4] + ".png")
    bd.plot_filtered(out_file)

os.makedirs(args.output_dir, exist_ok=True)
np.savez(os.path.join(args.output_dir, "beat_timestamps.npz"), **timestamps_data)
np.savez(os.path.join(args.output_dir, "beat_timeNorm.npz"), **normTimestamps_data)