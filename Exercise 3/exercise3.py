import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from lpc import lpc_ref, lpc_to_formants, lpc_okno
from sinsum import sinsum


# 2. EXERCISE
Flocal1 = np.array([330, 640, 270, 540, 290, 470, 380, 430])
Flocal2 = np.array([700,1300,1160,1850,2240,860,2200,1530])

Fwiki1 = np.array([360, 850,  250,  610,  240,  500,  390,  460])
Fwiki2 = np.array([640, 1610, 595, 1900, 2400, 700, 2300, 1310])

AMPlocal1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
AMPlocal2 = Flocal2/Flocal1

sam = ["o","a","u","E","i","O","e","@"]

Tsample = 0.5
fs = 16000
aeiou = [sam.index("a"),sam.index("E"),sam.index("i"),sam.index("O"),sam.index("u")]

siglocal = np.array([0.0])
sigwiki = np.array([0.0])

for k in aeiou:
    formantlocal = sinsum([Flocal1[k], Flocal2[k]], [AMPlocal1[k], AMPlocal2[k]], Tsample, fs)
    formantwiki = sinsum([Fwiki1[k], Fwiki2[k]], [AMPlocal1[k], AMPlocal2[k]], Tsample, fs)
    
    siglocal = np.append(siglocal, formantlocal, 0)
    sigwiki = np.append(sigwiki, formantwiki, 0)
    
wavfile.write("aeiou-local.wav", fs, siglocal)
wavfile.write("aeiou-wiki.wav", fs, sigwiki)


# 3. EXERCISE
fs, signal = wavfile.read("govor.wav")

# Window size 200 with half overstep (100)
n_window = 200
n_step = 100

n_total = int( (len(signal)/n_step) - 1 )

# Order of lpc coeficients is 16 + 2 for begining and end
order = 18 
T_step = n_step/fs

formant = [0] * n_total
amplitude = [0] * n_total

finalsig = np.array([0.0])

 
# 4. EXERCISE 
finalsig = np.zeros(n_step*(n_total + 1), dtype = np.float32)
    
for k in range(n_total):
    temp_window = signal[(k*n_step):(k*n_step + n_window)]

    temp_formant, temp_amplitude = lpc_okno(temp_window, order, fs)
    
    formant[k] = temp_formant
    amplitude[k] = temp_amplitude

    temp_sig = np.hanning(n_window)*sinsum(np.ndarray.tolist(temp_formant), np.ndarray.tolist(temp_amplitude), T_step*2, fs)
    
    
    finalsig[k*n_step:k*n_step+n_window] = finalsig[(k*n_step):(k*n_step+n_window)] + temp_sig
    
wavfile.write("final-hann.wav", fs, finalsig)
