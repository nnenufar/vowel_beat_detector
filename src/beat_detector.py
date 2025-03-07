from scipy import signal
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import numpy as np
from praatio import textgrid
from sklearn.preprocessing import MaxAbsScaler

class BD():
  def __init__(self, sr, bandpass_left, bandpass_right, bandpass_order = 1, lowpass_order = 3, lowpass_cutoff = 10, min_dur = 0.03, min_peak_prominence = 0.2):
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

  def detect_beats(self, waveform, plot = False, play = False):
    '''
    Beats are associated with the points at which each local amplitude rise is about 40% complete.
    Amplitude rises are detected using the first derivative of the spectral envelope.

    Outputs an array with timestamps (in seconds) of the detected beats.
    '''
    self.xmax = len(waveform) / self.sr
    filtered = signal.lfilter(*self.bandpass(), waveform)
    rectified = self.rectify(filtered)
    envelope = signal.lfilter(*self.lowpass(), rectified)
    derivative = np.gradient(envelope)
    scaler = MaxAbsScaler()
    derivative = scaler.fit_transform(derivative.reshape(-1, 1)).reshape(-1)

    peaks, _ = signal.find_peaks(derivative, distance = self.sr * self.min_dur, prominence = self.min_peak_prominence)

    beats = []
    min_ref = 0.01
    for index in peaks:
      peak_index = index
      if derivative[peak_index] > min_ref:
        while index > 0 and derivative[index - 1] > min_ref:
          index -= 1

        rise_amp = derivative[peak_index] - derivative[index]
        beat_amp = derivative[index] + rise_amp * 0.4

        beat_index = np.argmin([abs(amp - beat_amp) for amp in derivative[index:peak_index]])
        beats.append(index + beat_index)

      else:
        continue

    if plot:
      '''
      Plotting resources. Not used by default.
      Shows the original and bandpass filtered waveforms, the amplitude envelope, its derivative
      with peaks and beats marks, and the frequency response of the bandpass filter.
      '''
      plt.figure(figsize=(12, 8))

      plt.subplot(5,1,1)
      plt.plot(waveform)
      plt.title("Original Waveform")

      plt.subplot(5,1,2)
      plt.plot(filtered)
      plt.title("Bandpass Filtered Waveform")

      plt.subplot(5,1,3)
      plt.plot(rectified)
      plt.title("Rectified Waveform")

      plt.subplot(5,1,4)
      plt.plot(envelope)
      plt.title("Amplitude envelope")

      plt.subplot(5,1,5)
      plt.plot(derivative)
      plt.plot(peaks, derivative[peaks], "*", color = 'green', label = 'Peak')
      for beat in beats:
        plt.axvline(beat, color='red', linestyle='--', alpha=0.7, label="Beat" if beat == beats[0] else "")
      plt.legend()
      plt.title("Derivative of the envelope")
      plt.tight_layout()
      plt.show()

      ## Bandpass frequency response
      plt.figure(figsize=(6, 4))
      w, h = signal.freqz(*self.bandpass(), worN=1024)
      # Convert frequency from radians/sample to Hz
      frequencies = w * self.sr / (2 * np.pi)
      # Magnitude in dB
      magnitude = 20 * np.log10(abs(h))
      plt.plot(frequencies, magnitude)
      plt.title("Frequency Response of the Bandpass Filter")
      plt.xlabel("Frequency [Hz]")
      plt.ylabel("Magnitude [dB]")
      plt.ylim([-30, 0])
      plt.axhline(-3, color = 'grey', linestyle='--')
      center = (self.bandpass_left + self.bandpass_right) / 2
      plt.axvline(center, color = 'black', linestyle='--', label='Center frequency')
      plt.axvline(self.bandpass_right, color = 'green', linestyle='--', label='Right Cutoff')
      plt.axvline(self.bandpass_left, color = 'red', linestyle='--', label='Left Cutoff')
      plt.xticks([0, self.bandpass_left, center, self.bandpass_right, self.sr/2], rotation = 45)
      plt.yticks(range(0, -30, -3))
      plt.legend()
      plt.show()

    if play:
      display(Audio(filtered, rate=self.sr))

    return np.array(beats) / self.sr

  def save_textgrid(self, timestamps, out_file, xmin = 0):
    '''
    Writes a Praat .TextGrid file with one point tier containing beat annotations
    at the detected timestamps.
    '''
    xmax = self.xmax
    points = [(t, "Beat") for t in timestamps]
    point_tier = textgrid.PointTier("Beats", points, xmin, xmax)
    tg = textgrid.Textgrid(xmin, xmax)
    tg.addTier(point_tier)
    tg.save(out_file, format = "long_textgrid", includeBlankSpaces = True)
    print(f"TextGrid saved to {out_file}")