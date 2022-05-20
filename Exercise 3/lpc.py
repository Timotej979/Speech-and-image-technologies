import sys, os
import numpy as np
import scipy
from scipy.io.wavfile import read, write
from scipy.signal import lfilter
import scipy.signal
from sinsum import sinsum
import matplotlib.pyplot as plt

def lpc_ref(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)

    Notes
    ----
    This is just for reference, as it is using the direct inversion of the
    toeplitz matrix, which is really slow"""
    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(scipy.linalg.inv(scipy.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype = signal.dtype)

def lpc_to_formants(lpc, sr):
    """Convert LPC to formants directly"""

    # extract roots, get angle and radius
    roots = np.roots(lpc)

    pos_roots = roots[np.imag(roots) >= 0]
    if len(pos_roots) < len(roots) // 2:
        pos_roots = list(pos_roots) + [0] * (len(roots) // 2 - len(pos_roots))
    if len(pos_roots) > len(roots) // 2:
        pos_roots = pos_roots[: len(roots) // 2]

    w = np.angle(pos_roots)
    a = np.abs(pos_roots)

    order = np.argsort(w)
    w = w[order]
    a = a[order]

    freqs = w * (sr / (2 * np.pi))
    bws = -0.5 * (sr / (2 * np.pi)) * np.log(a)

    # exclude DC and sr/2 frequencies
    return freqs, bws

def lpc_okno(okno_signala, red, fs):
    """Funkcija, ki poračuna frekvence in amplitude formantov v danem oknu
    signala z uporabo LPC analize."""
    # filtriranje vhodnega signala
    okno_signala = okno_signala * np.hanning(okno_signala.shape[0])
    okno_signala = lfilter([1], [1.0, 0.63], okno_signala)

    # določitev koeficientov LPC filtra z linearno regresijo
    A = lpc_ref(okno_signala, red)

    # Določitev frekvenc in pasovnih širin formantov
    formanti, pasovne_sirine = lpc_to_formants(A, fs)
    amplitude = np.exp(pasovne_sirine / 60)
    return formanti[1:5],  amplitude[1:5]
