import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def sinsum(f, a, T, fs):
    """Funkcija za sintezo sinusne vrste. Vhodni argumenti:
        f: seznam frekvenc frekvenčnih komponent sinusne vrste
        a: seznam ojačanj frekvenčnih komponent, v linearni skali
        T: dolžina sintetiziranega signala, v sekundah
        fs: željena vzorčna frekvenca sintetiziranega signala."""
    
    # definicija časovne osi, preko katere sintetiziramo signal
    t = np.arange(0, T, 1/fs).astype("float32")
    # inicializacija praznega arraya za sintetiziran signal
    signal = np.zeros_like(t)
    print(len(signal))
    
    for i in range(len(f)):
        for j in range(len(signal)):    	
	        signal[j] = signal[j] + a[i]*np.sin( 2*np.pi*f[i]* t[j] )
    
    
    # normalizacija signala na zalogo vrednosti [-1, 1]
    signal -= signal.mean()
    signal /= np.abs(signal).max()
    return signal

if __name__ == "__main__":
    x = sinsum([220, 440, 880, 1760], 
               [1, 0.5, 0.25, 0.125], 
               1.00002, 
               44100)
    x_prav = np.load("sinsum_test.npy")
    # oblika dejanskega in pravilnega signala
    print(x.shape, x_prav.shape)
    # maksimalno odstopanje
    print(np.abs(x - x_prav).max())

    plt.plot(x_prav)
    plt.plot(x)
    plt.tight_layout()
    plt.grid(True)
    plt.show()
