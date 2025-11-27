
print('Importing Libraries')
import numpy as np
import pandas as pd
import itertools
from scipy.stats import zscore
from tensorflow.keras.models import load_model
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns





print('Loading Models')    
model = load_model(os.getcwd()+"/2008shahtalelebi2020_eeg_encoder.keras")
Siamese_model = load_model(os.getcwd()+"/2008shahtalelebi2020_eeg_siamese_model.keras")
# Load data and add label to last column
print('Reading the data from CSV files')
ft = pd.read_csv('./2008_PSD_Feet.csv')
lh = pd.read_csv('./2008_PSD_Left_Hand.csv')
rh = pd.read_csv('./2008_PSD_Right_Hand.csv')
te = pd.read_csv('./2008_PSD_Tongue.csv')


#Use all data set [0,1356] x 2816
Feet = np.array(ft)
Left = np.array(lh)
Right = np.array(rh)
Tongue = np.array(te)

# Standardize data along axis 1 (row-wise)
Feet = zscore(Feet, axis=1)
Left = zscore(Left, axis=1)
Right = zscore(Right, axis=1)
Tongue = zscore(Tongue, axis=1)

# Make Combinations (Positive and Negative) of Left, Right and Both
print('Making data pairs for training Siamese network')
Positive_Feet = list(itertools.combinations(Feet,2))
Positive_Left = list(itertools.combinations(Left, 2)) # combinations AB != BA
Positive_Right = list(itertools.combinations(Right, 2))
Positive_Tongue = list(itertools.combinations(Tongue, 2))

 # all possible combinations
Negative1 = list(itertools.product(Feet,Left))
Negative2 = list(itertools.product(Feet,Right))
Negative3 = list(itertools.product(Feet,Tongue))
Negative4 = list(itertools.product(Left,Right))
Negative5 = list(itertools.product(Left,Tongue))
Negative6 = list(itertools.product(Right,Tongue))

# Positive and Negative Samples
Positive_Samples = Positive_Feet + Positive_Left + Positive_Right + Positive_Tongue
Negative_Samples = Negative1 + Negative2 + Negative3 + Negative4 + Negative5 + Negative6

# Split into pairs
Pair1 = []
Pair2 = []
Label = []

for sample in Positive_Samples:
    Pair1.append(sample[0])
    Pair2.append(sample[1])
    Label.append(1)
for sample in Negative_Samples:
    Pair1.append(sample[0])
    Pair2.append(sample[1])
    Label.append(0)

print('Splitting Train test data')
# Split the training and testing data
Pair1_train, Pair1_test, Pair2_train, Pair2_test, Label_train, Label_test = train_test_split(Pair1, Pair2, Label, test_size=0.2,random_state=23, shuffle=True)
"""Pair1_test, Pair2_test = np.array(Pair1_test[:100]), np.array(Pair2_test[:100])  ## selecting only 100 due to memory issue
Label_test = np.array(Label_test[:100])
"""
Pair1_test, Pair2_test = np.array(Pair1_test), np.array(Pair2_test)  ## selecting only 100 due to memory issue
Label_test = np.array(Label_test)
# Standardize data along axis 1 (row-wise)

predictions=Siamese_model.predict([Pair1_test,Pair2_test])
predicted_labels = (predictions >= 0.5).astype(int)
# Confusion Matrix
predicted_labels = predicted_labels.tolist()
actual = Label_test.tolist()


cm = confusion_matrix(actual,predicted_labels)
plt.figure(figsize=(8, 6))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(actual), yticklabels=np.unique(actual))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Different Class', 'Same Class'],
            yticklabels=['Different Class', 'Same Class'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("3m3_ConfusionMatrix.png")
#plt.show()

predicted_labels = np.array(predicted_labels).flatten()
actual = np.array(actual).flatten()

# Calculate accuracy
accuracy = (predicted_labels == actual).mean() * 100
print(f"Test Accuracy: {accuracy:.2f}%")

