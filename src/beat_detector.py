from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from praatio import textgrid
from sklearn.preprocessing import MaxAbsScaler

class BD():
  def __init__(self, sr, bandpass_left, bandpass_right, bandpass_order = 1, lowpass_order = 3, lowpass_cutoff = 15, min_dur = 0.03, min_peak_prominence = 0.2):
    self.bandpass_order = bandpass_order
    self.bandpass_left = bandpass_left
    self.bandpass_right = bandpass_right
    self.lowpass_order = lowpass_order
    self.lowpass_cutoff = lowpass_cutoff
    self.min_dur = min_dur
    self.min_peak_prominence = min_peak_prominence
    self.sr = sr
    self.nyquist = sr / 2

  def bandpass(self):
    '''
    First filter. Used to eliminate fricative noise and F0 energy,
    leaving energy in the formants region intact.
    '''
    bandpass_right = self.bandpass_right / self.nyquist
    bandpass_left = self.bandpass_left / self.nyquist
    b, a = signal.butter(self.bandpass_order, [bandpass_left, bandpass_right], btype = 'bandpass')
    return (b, a)

  def rectify(self, waveform):
    '''
    Signal rectification using absolute values.
    '''
    return np.abs(waveform)

  def lowpass(self):
    '''
    Second filter. Used to produce a smooth amplitude envelope.
    '''
    cutoff = self.lowpass_cutoff / self.nyquist
    b, a = signal.butter(self.lowpass_order, cutoff, btype = 'lowpass')
    return (b, a)

  def detect_beats(self, waveform):
    '''
    Beats are associated with local maxima in the first derivative of the spectral envelope.
    Outputs arrays with timestamps (in seconds) of the detected beats.
    '''
    # Store waveform and intermediate signals as instance attributes
    self.waveform = waveform
    self.tmax_s = len(self.waveform) / self.sr 
    self.filtered = signal.lfilter(*self.bandpass(), self.waveform)
    self.rectified = self.rectify(self.filtered)
    self.envelope = signal.lfilter(*self.lowpass(), self.rectified)
    self.derivative = np.gradient(self.envelope)
    self.second_derivative = np.gradient(self.derivative)
    scaler = MaxAbsScaler()
    self.derivative = scaler.fit_transform(self.derivative.reshape(-1, 1)).reshape(-1)
    self.second_derivative = scaler.fit_transform(self.second_derivative.reshape(-1, 1)).reshape(-1)

    self.peaks, _ = signal.find_peaks(self.derivative, distance = self.sr * self.min_dur, prominence = self.min_peak_prominence)
    self.peaks_second, _ = signal.find_peaks(self.second_derivative, distance = self.sr * self.min_dur, prominence = self.min_peak_prominence)

    # Note: self.beats will store the sample indices of the beats
    self.beats = []
    min_ref = 0.01
    for index in self.peaks:
      peak_index = index
      if self.derivative[peak_index] > min_ref:
        self.beats.append(peak_index)

      else:
        continue

    self.beats_second = []
    min_ref = 0.01
    for index in self.peaks_second:
      peak_index = index
      if self.second_derivative[peak_index] > min_ref:
        self.beats_second.append(peak_index)

    beat_timestamps =  np.array(self.beats) / self.sr

    time_normalized_timestamps = beat_timestamps / self.tmax_s

    beat_timestamps_second = np.array(self.beats) / self.sr

    return beat_timestamps, time_normalized_timestamps, beat_timestamps_second

  def plot_filtered(self, out_file, ground_truth = None):
    '''
    Plotting resources. Must be called after detect_beats().
    Shows the original and bandpass filtered waveforms, the amplitude envelope, its derivative
    with peaks and beats marks, and the frequency response of the bandpass filter.
    '''
    plt.figure(figsize=(12, 8))

    plt.subplot(5,1,1)
    plt.plot(self.waveform)
    plt.title("Original Waveform")

    plt.subplot(5,1,2)
    plt.plot(self.filtered)
    plt.title("Bandpass Filtered Waveform")

    plt.subplot(5,1,3)
    plt.plot(self.rectified)
    plt.title("Rectified Waveform")

    plt.subplot(5,1,4)
    plt.plot(self.envelope)
    plt.title("Amplitude envelope")

    plt.subplot(5,1,5)
    plt.plot(self.derivative)
    for beat in self.beats:
      plt.axvline(beat, color='red', linestyle='--', alpha=0.7, label="Detected Beat" if beat == self.beats[0] else "")

    if ground_truth is not None:
      tg = textgrid.openTextgrid(ground_truth, includeEmptyIntervals=True)
      gt_tier = tg.getTier('beats')
      gt_timestamps = [entry.time * self.sr for entry in gt_tier.entries]
      for i, beat in enumerate(gt_timestamps):
        label = "Ground truth" if i == 0 else ""
        plt.axvline(beat, color='red', linestyle='-', alpha=0.7, label = label)
      plt.legend(fontsize='small')
      plt.title("Derivative of the envelope")
      plt.tight_layout()
      plt.savefig(out_file)
    else:
      plt.legend()
      plt.title("Derivative of the envelope")
      plt.tight_layout()
      plt.savefig(out_file)

    # plt.subplot(6,1,6)
    # plt.plot(self.second_derivative)
    # plt.plot(self.peaks_second, self.second_derivative[self.peaks_second], "*", color = 'purple', label = 'Peak_2nd derivative')
    # for beat in self.beats_second:
    #   plt.axvline(beat, color='red', linestyle='-', alpha=0.7, label="Beat_2nd derivative" if beat == self.beats_second[0] else "")
    # plt.legend()
    # plt.title("Second derivative of the envelope")

    ## Bandpass frequency response
    # plt.figure(figsize=(6, 4))
    # w, h = signal.freqz(*self.bandpass(), worN=1024)
    # # Convert frequency from radians/sample to Hz
    # frequencies = w * self.sr / (2 * np.pi)
    # # Magnitude in dB
    # magnitude = 20 * np.log10(abs(h))
    # plt.plot(frequencies, magnitude)
    # plt.title("Frequency Response of the Bandpass Filter")
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Magnitude [dB]")
    # plt.ylim([-30, 0])
    # plt.axhline(-3, color = 'grey', linestyle='--')
    # center = (self.bandpass_left + self.bandpass_right) / 2
    # plt.axvline(center, color = 'black', linestyle='--', label='Center frequency')
    # plt.axvline(self.bandpass_right, color = 'green', linestyle='--', label='Right Cutoff')
    # plt.axvline(self.bandpass_left, color = 'red', linestyle='--', label='Left Cutoff')
    # plt.xticks([0, self.bandpass_left, center, self.bandpass_right, self.sr/2], rotation = 45)
    # plt.yticks(range(0, -30, -3))
    # plt.legend()
    # plt.show()

  def write_wav(self, out_file):
    '''
    Writes the bandpass filtered audio to disk. Must be called after detect_beats().
    '''
    sf.write(out_file, self.filtered, self.sr)

  def save_textgrid(self, timestamps, out_file, xmin = 0):
    '''
    Writes a Praat .TextGrid file with one point tier containing beat annotations
    at the detected timestamps.
    '''
    tmax_s = self.tmax_s
    points = [(t, "Beat") for t in timestamps]
    point_tier = textgrid.PointTier("Beats", points, xmin, tmax_s)
    tg = textgrid.Textgrid(xmin, tmax_s)
    tg.addTier(point_tier)
    tg.save(out_file, format = "long_textgrid", includeBlankSpaces = True)
    print(f"TextGrid saved to {out_file}")