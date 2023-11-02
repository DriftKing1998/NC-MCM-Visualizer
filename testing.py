import numpy as np
from scipy import signal

def bandpass(traces, f_l, f_h, sampling_freq):
    """
    Apply a bandpass filter to the input traces.

    Parameters:
        traces (np.ndarray): Input traces to be filtered.
        f_l (float): Lower cutoff frequency in Hz.
        f_h (float): Upper cutoff frequency in Hz.
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        filtered (np.ndarray): Filtered traces.

    """
    cut_off_h = f_h * sampling_freq / 2  ## in units of sampling_freq/2
    cut_off_l = f_l * sampling_freq / 2  ## in units of sampling_freq/2
    #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.

    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered

fps = 1.3
X = np.array([[0.2, 0.03, 0.041],[0.2, 0.03, 1.001],[1.2, 0.003, 0.1]])
filtered = bandpass(X.T, f_l=0.01, f_h=0.1, sampling_freq=fps).T
print(filtered)
