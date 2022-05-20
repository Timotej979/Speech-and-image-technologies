import numpy as np
import plotext as plt
from python_speech_features import mfcc, delta
from scipy.io.wavfile import read
import sys, os
	

####################################### FUNCTIONS #######################################
def mfcc_feats(sig, fs):
    N = 1 
    
    #window 25 ms, 10 ms overlap -> step 15 ms for 10 ms overlap
    mfcc_feat = mfcc(sig ,fs, winlen=0.025,winstep=0.015,numcep=13, appendEnergy=True) 
    
    delta_1 = delta(mfcc_feat, 1) 
    delta_2 = delta(delta_1, 1)
     
    mfcc_final = np.concatenate((mfcc_feat, delta_1, delta_2), axis=1) 	

    return mfcc_final


def DTW_distance(sig1, sig2, fs):

    # MFCC from both signals
    f1 = mfcc_feats(sig1,fs)
    f2 = mfcc_feats(sig2,fs)    
    
    # DTW matrix dimension setup
    P = f1.shape[0]
    R = f2.shape[0]
    
    # Create DTW matrix
    D = np.ones((P+1,R+1))*10**9
    
    # Set first field to zero
    D[0,0] = 0  
    
    for i in range(P):
        for j in range(R): 
            D_min = min(D[i,j],D[i+1,j],D[i,j+1]) 
            D_razd = np.sqrt(np.sum(np.subtract(f1[i],f2[j])**2))
            D[i+1,j+1] = D_min + D_razd
    
    D_dist = D[P,R] 
    return D_dist


def sample_comparison():

    recordings = os.listdir("posnetki1")
    samples = []
    for recording in recordings:
        fs, read_sig = read(os.path.join("posnetki1",recording))
        samples.append(read_sig)
        
    recording_len = len(recordings) 
    distance_list = []
    all_rec = 16;
    valid_rec = 0;
    percentage_rec = 0.0;
    neighbour_index = np.zeros(recording_len)
    neighbour_distance = np.zeros(recording_len)
    
    f1 = [];
    f2 = [];
    

    print("\n\n########################################################################")
    print("|----------------------------All distances-----------------------------|")
    print("|       Recording       |       Neighbour       |       Distance       |")
    print("|-----------------------|-----------------------|----------------------|")
	
    # FOR-LOOPS over I and J to compare everyone
    for i in range(recording_len):
        distance_i = np.zeros(recording_len)
        for j in range(recording_len):
            distance_i[j] = DTW_distance(samples[i], samples[j], fs)            	
            print("|  ", recordings[i], "               ", recordings[j], "        ", distance_i[j])
    	
        # exclude same comparison, saving min distance values
        distance_i[i] = 10000 
        neighbour_index[i] = np.argmin(distance_i) 
        neighbour_distance[i] = np.min(distance_i) 
        distance_list.append(distance_i)

    print("------------------------------------------------------------------------")
    
    # 23/23/22 spaces per sector
    print("\n\n########################################################################")
    print("|--------------------------Closest distances---------------------------|")
    print("|       Recording       |   Closest neighbour   |       Distance       |")
    print("|-----------------------|-----------------------|----------------------|")

    for i in range(recording_len): 
    	print("|  ", recordings[i] , "               ", recordings[int(neighbour_index[i])], "        ", neighbour_distance[i])
    print("------------------------------------------------------------------------")	
    
    return samples, recordings, recording_len    
    
 
def local_sample_comparison(samples, recordings, recording_len):

    ## LOCAL -> LOCAL COMPARISON ##
    local_recordings = os.listdir("posnetki2") 
    local_samples = []
    percentage_sum = 0.0
    percentage = 0.0
    
    for recording in local_recordings:
    	fs, read_sig = read(os.path.join("posnetki2", recording)) 
    	local_samples.append(read_sig) 
   
    recording_len_AX = len(local_recordings)

    neighbour_index_AX = np.zeros(recording_len_AX) 
    neighbour_distance_AX = np.zeros(recording_len_AX) 
    
    print("\n\n\n################### LOCAL TO LOCAL COMPARISON ############################")  
    print("\n########################################################################")
    print("|----------------------------All distances-----------------------------|")
    print("|       Recording       |       Neighbour       |       Distance       |")
    print("|-----------------------|-----------------------|----------------------|")
    
    for i in range(recording_len_AX): 
        distance_i = np.zeros(recording_len_AX)       
        for j in range(recording_len): 
            distance_i[j] = DTW_distance(local_samples[i],samples[j], fs)
            print("|  ", local_recordings[i], "               ", recordings[j], "        ", distance_i[j])
            
    
        distance_i[i] = 10000 
        neighbour_index_AX[i] = np.argmin(distance_i) 
        neighbour_distance_AX[i] = np.min(distance_i) 
    
    print("------------------------------------------------------------------------")
    
    print("\n\n########################################################################")
    print("|--------------------------Closest distances---------------------------|")
    print("|       Recording       |   Closest neighbour   |       Distance       |")
    print("|-----------------------|-----------------------|----------------------|")
    
    for i in range(recording_len_AX): 
        print("|  ", local_recordings[i] , "               ", local_recordings[int(neighbour_index_AX[i])], "        ", neighbour_distance_AX[i])
        for j in range(recording_len):
           if(distance_i[j] < 7000):
               percentage_sum = percentage_sum + 1
        print("|  PERCENTAGE: ", percentage_sum/16*100, "%")
        percentage_sum = 0
    print("------------------------------------------------------------------------")

    #####################################################################################################################################################

    ## LOCAL -> ORIGINAL COMPARISON ##
    neighbour_index_XA = np.zeros(recording_len_AX)  
    neighbour_distance_XA = np.zeros(recording_len_AX) 

    print("\n\n\n################ LOCAL TO ORIGINAL COMPARISON ##########################")  
    print("\n########################################################################")
    print("|----------------------------All distances-----------------------------|")
    print("|       Recording       |       Neighbour       |       Distance       |")
    print("|-----------------------|-----------------------|----------------------|")

    for i in range(recording_len):  
        distance_i = np.zeros(recording_len) 
        for j in range(recording_len_AX): 
            distance_i[j] = DTW_distance(samples[i], local_samples[j], fs)             
            print("|  ", local_recordings[i], "               ", recordings[j], "        ", distance_i[j])            
            
        distance_i[i] = 10000 
        neighbour_index_XA[i] = np.argmin(distance_i) 
        neighbour_distance_XA[i] = np.min(distance_i) 

    
##################################### BASIC CODE #####################################
if __name__ == "__main__":
    fs, signal = read("posnetki1/gor1.wav")
    actual_mfcc = mfcc_feats(signal, fs=16000)
    correct_mfcc = np.load("mfcc_test.npy")

    print("\n\n############################ EXERCISE 2 ############################\nBasic test:")
    print("  - Correct shape: ", correct_mfcc.shape)
    print("  - Actual shape: ", actual_mfcc.shape)

    print("  - Maximum deviation: ",  np.abs(correct_mfcc - actual_mfcc).max())


###### CALL FUNCTIONS ######
samplesOG, recordingsOG, recording_lenOG = sample_comparison()
local_sample_comparison(samplesOG, recordingsOG, recording_lenOG)
	
