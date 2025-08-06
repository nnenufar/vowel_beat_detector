import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--arrays_dir', type=str, required=True, help='directory with .npz files')
args = parser.parse_args()

arrays_dir = args.arrays_dir
timestamps_path = os.path.join(arrays_dir, "beat_timestamps.npz")
timeNorm_path = os.path.join(arrays_dir, "beat_timeNorm.npz")

all_timestamps = np.load(timestamps_path)
all_timeNorm = np.load(timeNorm_path)

identifier = "JN.wav"

timestamps = all_timestamps[identifier]
timeNorm = all_timeNorm[identifier]

print(f'timestamps: {timestamps}\n\ntime normalized: {timeNorm}')
