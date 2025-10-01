###
# Receives: waveform, beat timestamps
# Outputs: array of shape [len(timestamps), feat_dimension]

from librosa.feature import melspectrogram, mfcc

class MFCC():
    def __init__(self, sr, n_mfcc, frame_length):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length

    def extract_mfcc(self, waveform, beat_index):
        assert self.frame_length % 2 == 0, "Frame length must be a multiple of 2"

        center = beat_index
        start = center - self.frame_length // 2
        end = center + self.frame_length // 2
        
        frame = waveform[start:end]
        feats = mfcc(y=frame, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.frame_length, center=False)

        return feats






     