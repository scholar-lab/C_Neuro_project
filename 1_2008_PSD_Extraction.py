
import numpy as np
import pandas as pd
from scipy.signal import welch


sampling_rate = 250  # Sampling rate (Hz)
window_size = sampling_rate * 6     # Set the window size
overlap = window_size // 2  # 50% overlap
step = window_size - overlap # used for calculating beginning of a segment


name = {'Feet':'Sub','Left_Hand':'Sub','Right_Hand':'Sub','Tongue':'Sub'}
for n in name: # Folder Name Loop all subject of one class
    # Generate the string series
    psd_all_subject = np.empty((0,1936)) # used for storing PSD of one class (ALL Subjects)
    for sl in range(1, 9): # Loop through all 9 subjects
        filename = f"{name[n]}{sl}.csv"
        
        # Load EEG data from the filename CSV file
        input_file = fr".\REQUIRED_PART_BCI_2008\{n}\{filename}"  # Replace with the actual file path
        raw_data = pd.read_csv(input_file)
        print('Drop last three column containing monopolar EOG')
        data = raw_data.iloc[:,:-3]
        data = data.to_numpy()
        print(f'Filename: {filename},Original shape: {raw_data.shape},Modified shape: {data.shape}')
        num_segments = (len(data) - overlap) // step
        # Create a 3D array to store segments
        num_channel = data.shape[1]
        segments = np.zeros((num_channel,num_segments, window_size))
        psd_out = np.empty((num_segments, 0))#psd_out.reshape(1,-1)

        
        for channel in range(num_channel): # Loop through all channels 
            #print(channel)
            #os.mkdir(f'{channel}')
            psd_one_channel = np.empty((0, 88)) #np.array([])
            #psd_one_channel = psd_one_channel.reshape(1,-1)
            for i in range(num_segments): # Loop through all segments
                start = i * step
                end = start + window_size
                #segments.append(data[start:end][channel])
                #y=data[start:end][channel]
                segments[channel][i] = data[start:end,channel]
                
                # Get the signal for the current channel
                signal = data[start:end,channel]
            
                # Perform Welch's method to calculate Power Spectral Density (PSD)
                freqs, psd = welch(signal, fs=sampling_rate, nperseg=4*sampling_rate)
                psd = psd.reshape(1,-1) # convert to row array
                freqs = freqs.reshape(1,-1)
                psd_segment = psd[:,32:120] # index corresponds to 8-30 HZ
                # save the psd for later use
                #psd_one_channel = np.append(psd_one_channel, psd[:,32:120], axis=0)
                psd_one_channel = np.vstack((psd_one_channel,psd_segment))
                #print(f'Start = {start}\nend = {end}\ni = {i}')
                # Save to CSV
                
            psd_out = np.hstack((psd_out,psd_one_channel))
            
            #print(f'Segments: {segments}')
            #np.savetxt(rf'{n}{filename}.csv', psd_out, delimiter=',')
        # Vertically stack after every subject is processed
        psd_all_subject = np.vstack((psd_all_subject,psd_out))
        print(f'Current PSD data Size: {psd_all_subject.shape}')
    #Save the file when all subject is processed
    np.savetxt(rf'2008_PSD_{n}.csv', psd_all_subject, delimiter=',')