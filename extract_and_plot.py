import numpy as np
import pandas as pd
import os
from scipy.signal import welch
import matplotlib.pyplot as plt

# ================= GLOBAL PARAMETERS =====================

sampling_rate = 250
window_size = sampling_rate * 6       # 6 seconds
overlap = window_size // 2            # 50 percent
step = window_size - overlap          # step for segmentation


# ================= HELPER FUNCTIONS ======================

def band_indices(freqs, f_low, f_high):
    idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    return idx[0], idx[-1] + 1


def plot_time_domain(data, sampling_rate, class_name, subject_id):
    """Plot first 3 seconds of channel 1"""
    plt.figure(figsize=(10, 4))

    t = np.arange(len(data)) / sampling_rate
    plt.plot(t[:3 * sampling_rate], data[:3 * sampling_rate, 0])

    plt.title("Time Domain EEG - Class: {} | Subject {}".format(class_name, subject_id))
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (uV)")
    plt.grid()

    outfile = "timeplot_{}_sub{}.png".format(class_name, subject_id)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Saved:", outfile)


def plot_psd_class(psd_matrix, sampling_rate, class_name):
    """Plot mean PSD of entire class"""

    mean_psd = np.mean(psd_matrix, axis=0)

    # Compute frequency vector
    freqs, _ = welch(np.zeros(window_size), fs=sampling_rate, nperseg=1000)

    # Compute band indices for 8-30 Hz
    f1, f2 = band_indices(freqs, 8, 30)

    # Slice frequency vector to match PSD vector length
    freqs_band = freqs[f1:f2]

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_band, mean_psd, linewidth=2)

    plt.title("PSD - Class: {}".format(class_name))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid()

    outfile = "PSD_{}.png".format(class_name)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Saved:", outfile)


# ================= MAIN PIPELINE ========================

name = {
    "Feet": "Sub",
    "Left_Hand": "Sub",
    "Right_Hand": "Sub",
    "Tongue": "Sub"
}

for class_name in name:
    print("\n================ Processing Class: {} ================".format(class_name))

    psd_all_subject = []

    for subject_id in range(1, 9):

        filename = "{}{}.csv".format(name[class_name], subject_id)
        input_file = r".\REQUIRED_PART_BCI_2008\{}\{}".format(class_name, filename)

        if not os.path.exists(input_file):
            print("Missing file:", input_file)
            continue

        raw_data = pd.read_csv(input_file)
        data = raw_data.iloc[:, :-3].to_numpy()  # remove last 3 EOG channels
        print("Loaded", filename, "Shape:", data.shape)

        # ---------- TIME DOMAIN PLOT ----------
        plot_time_domain(data, sampling_rate, class_name, subject_id)

        # ---------- PSD CALCULATION ----------
        num_samples, num_channels = data.shape
        num_segments = (num_samples - overlap) // step

        # Frequency indices for 8-30 Hz band
        freqs, _ = welch(data[:window_size, 0], fs=sampling_rate, nperseg=1000)
        f1, f2 = band_indices(freqs, 8, 30)

        subject_psd = []

        for ch in range(num_channels):
            for seg in range(num_segments):
                start = seg * step
                end = start + window_size

                signal = data[start:end, ch]

                freqs, psd = welch(signal, fs=sampling_rate, nperseg=1000)

                # Slice PSD for 8-30 Hz
                subject_psd.append(psd[f1:f2])

        subject_psd = np.array(subject_psd)
        psd_all_subject.append(subject_psd)
        print("Subject PSD shape:", subject_psd.shape)

    # -------- STACK ALL SUBJECTS OF THE CLASS --------
    psd_all_subject = np.vstack(psd_all_subject)
    print("Final PSD shape for class {}: {}".format(class_name, psd_all_subject.shape))

    # -------- PSD CLASS PLOT --------
    plot_psd_class(psd_all_subject, sampling_rate, class_name)

    # -------- SAVE CSV --------
    outfile = "2008_PSD_{}.csv".format(class_name)
    np.savetxt(outfile, psd_all_subject, delimiter=",")
    print("Saved:", outfile)

print("\nAll processing completed successfully.")
