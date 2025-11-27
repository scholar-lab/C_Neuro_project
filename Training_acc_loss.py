# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:37:51 2025

@author: scholar
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  7 00:38:01 2025

@author: umaka
"""
import pickle
import matplotlib.pyplot as plt
import statistics
import numpy as np

def bin_average(data, bin_size):
    return np.mean(data[:len(data) // bin_size * bin_size].reshape(-1, bin_size), axis=1)

"""
CHANGE :
    1. HISTORY FILENAME
    2. PLOT TITLE
    3. PLOT FILENAME
"""


# Load history from Pickle file
with open('2008shahtalelebi2020_history_list.pkl', 'rb') as f:
    history_list = pickle.load(f)


training_accuracy = []
training_loss = []
validation_accuracy = []
validation_loss = []

for i in range(len(history_list)):
    history = history_list[i] 
    av_acc = statistics.mean(history.history['accuracy']) 
    av_loss = statistics.mean(history.history['loss'])
    av_val_acc = statistics.mean(history.history['val_accuracy'])
    av_val_loss = statistics.mean(history.history['val_loss'])
    training_accuracy.append(av_acc)
    training_loss.append(av_loss)
    validation_accuracy.append(av_val_acc)
    validation_loss.append(av_val_loss)



chunk_size = 200
training_accuracy = bin_average(np.array(training_accuracy), bin_size=chunk_size)
training_loss = bin_average(np.array(training_loss), bin_size=chunk_size)

validation_accuracy = bin_average(np.array(validation_accuracy), bin_size=chunk_size)
validation_loss = bin_average(np.array(validation_loss), bin_size=chunk_size)

# Step 6: Plotting the curves
epochs = range(1, len(training_accuracy) + 1)

plt.figure(figsize=(12, 5))

Model = "Training"
# Accuracy plot
#plt.subplot(1, 2, 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o')
plt.title(f'{Model} Accuracy Curve')
plt.xlabel('Binned Loops ')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("M3_accuracy.png")

# Loss plot
#plt.subplot(1, 2, 2)
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_loss, label='Training Loss', marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
plt.title(f'{Model} Loss Curve')
plt.xlabel('Binned Loops')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("M3_loss.png")

plt.tight_layout()
plt.show()


