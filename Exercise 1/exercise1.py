import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy.signal as sig


#----------------------------------------SETUP PARAMETERS----------------------------------------#
wideLim = [10e-3, 20e-3]									     
shortLim = [50e-3, 200e-3]									     
												     
# READ FILE											     
fs,signal = read("govor.wav")									     
												     
time = signal.shape[0] / fs
B, A = sig.butter(3, 0.7, output='ba')									     
signal = sig.filtfilt(B, A, signal)
												     
												     
#---------------------------------------------EXERCISE1---------------------------------------------#
print("#-----------------------VAJA1-----------------------#")				      
print("#---------------------------------------------------#")				      	
print("#  Sound wave sampling frequency: ", fs, " Hz")				      
print("#  Sound wave length: ", time, " sekund")				      
print("#  Sound wave data type: ", signal.dtype)			      
print("#  Sound wave range min: ", np.min(signal))			      
print("#  Sound wave range max: ", np.max(signal))			      
print("#  Sound wave delta(max-min): ", np.max(signal) - np.min(signal))	      
print("#---------------------------------------------------#")				      
												      
#---------------------------------------------EXERCISE2---------------------------------------------#
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(80, 20), ncols = 4)			      
												      
# BASIC SIGNAL											      
ax1.plot(signal)										      
ax1.set_xlabel("Sample")									      
ax1.set_ylabel("Amplitude")									      
ax1.title.set_text("Signal graph of govor.wav")						      
												      
# GENERAL SPECTROGRAM										      
ps2, f2, t2, im2 = ax2.specgram(signal, Fs = fs)						      
ax2.set_xlabel("Time [sec]")									      
ax2.set_ylabel("Frequency [Hz]")								      
ax2.title.set_text("Spectrogram of govor.wav")						      
												      
# WIDE SPECTROGRAM										      
ps3, f3, t3, im3 = ax3.specgram(signal, NFFT = int(fs*(wideLim[1] - wideLim[0])), Fs = fs)	      
ax3.set_xlabel("Time [sec]")									      
ax3.set_ylabel("Frequency [Hz]")								      
ax3.title.set_text("Wide spectrogram of govor.wav")						      
												      
# SHORT SPECTROGRAM										      
ps4, f4, t4, im4 = ax4.specgram(signal, NFFT = int(fs*(shortLim[1] - shortLim[0])), Fs = fs)      
ax4.set_xlabel("Time [sec]")									      
ax4.set_ylabel("Frequency [Hz]")								      
ax4.title.set_text("Short spectrogram of govor.wav")						      
												      
plt.show()											      
												      
												      
#---------------------------------------------EXERCISE3---------------------------------------------#
# VOWELS
# 	    á,    ê,     é,     í,    ó,    ô,     ú,    ŕ
tstart = [0.878, 1.541, 2.053, 2.546, 3.034, 3.377, 3.908, 4.185]
tstop =  [0.933, 1.593, 2.104, 2.563, 3.109, 3.432, 3.946, 4.224]

# INDEX START/STOP ARRAY
istart = np.dot(tstart, fs)
istart = istart.astype(int)

istop = np.dot(tstop, fs)
istop = istop.astype(int)

# PRESET CALCULATED ARRAYS
sectionsList = []										     
NFFTList = []											     
PampList = []
fAxisList = []

# CALCULATING ARRAY VALUES
for i in range(0, len(istart)):

	sectionsList.append(signal[istart[i]:istop[i]])
	NFFTList.append(sectionsList[i].shape[0])
	
	PampList.append( 20 * np.log10(np.abs(np.fft.fft(sectionsList[i]*4)[:len(sectionsList[i])//2])) )	
	fAxisList.append( np.arange(0, fs/2, fs/NFFTList[i]) )

# GRAPHING ALL VOCALS
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(figsize=(80, 20), nrows = 2, ncols = 4)

ax1.plot(fAxisList[0], PampList[0])
ax1.grid(True)
ax1.set_xlabel("Frekvenca [Hz]")
ax1.set_ylabel("P [dB]")
ax1.title.set_text("Ozki á (ostrivec)")

ax2.plot(fAxisList[1], PampList[1])
ax2.grid(True)
ax2.set_xlabel("Frekvenca [Hz]")
ax2.set_ylabel("P [dB]")
ax2.title.set_text("Široki ê (strešica)")

ax3.plot(fAxisList[2], PampList[2])
ax3.grid(True)
ax3.set_xlabel("Frekvenca [Hz]")
ax3.set_ylabel("P [dB]")
ax3.title.set_text("Ozki é (ostrivec)")

ax4.plot(fAxisList[3], PampList[3])
ax4.grid(True)
ax4.set_xlabel("Frekvenca [Hz]")
ax4.set_ylabel("P [dB]")
ax4.title.set_text("Črka í")

ax5.plot(fAxisList[4], PampList[4])
ax5.grid(True)
ax5.set_xlabel("Frekvenca [Hz]")
ax5.set_ylabel("P [dB]")
ax5.title.set_text("Ozki ó (ostrivec)")

ax6.plot(fAxisList[5], PampList[5])
ax6.grid(True)
ax6.set_xlabel("Frekvenca [Hz]")
ax6.set_ylabel("P [dB]")
ax6.title.set_text("Široki ô (strešica)")

ax7.plot(fAxisList[6], PampList[6])
ax7.grid(True)
ax7.set_xlabel("Frekvenca [Hz]")
ax7.set_ylabel("P [dB]")
ax7.title.set_text("Ozki ú (ostrivec)")

ax8.plot(fAxisList[7], PampList[7])
ax8.grid(True)
ax8.set_xlabel("Frekvenca [Hz]")
ax8.set_ylabel("P [dB]")
ax8.title.set_text("Črka ŕ")

plt.show()


# FORMANT FREQUENCIES
F1 = np.array([330, 640, 270, 540, 290, 470, 380, 430])
F2 = np.array([700,1300,1160,1850,2240,860,2200,1530])

for i in range(8):
 plt.plot(F2[i], F1[i], "o")
plt.xlim((2500, 500))
plt.ylim((850, 100))
plt.xlabel("F2 [Hz]")
plt.ylabel("F1 [Hz]")
plt.legend("o a u E i O e @".split(" "))
plt.show()

