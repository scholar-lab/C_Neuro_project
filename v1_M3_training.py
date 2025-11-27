
print('Importing Libraries')
import numpy as np
import pandas as pd
import itertools
from scipy.stats import zscore
from keras.layers import Input, Dense, dot, Dropout      
from keras.models import Model 
from sklearn.model_selection import train_test_split
import os 
#from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import pickle
from tensorflow.keras.optimizers import SGD  # Stochastic Gradient Descent (GD when batch_size = full)
from tensorflow.keras.losses import Huber

# Load data and add label to last column
print('Reading the data from CSV files')
ft = pd.read_csv('../2008_PSD_Feet.csv')
lh = pd.read_csv('../2008_PSD_Left_Hand.csv')
rh = pd.read_csv('../2008_PSD_Right_Hand.csv')
te = pd.read_csv('../2008_PSD_Tongue.csv')

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
Pair1_test, Pair2_test = np.array(Pair1_test[:100]), np.array(Pair2_test[:100])  ## selecting only 100 due to memory issue
Label_test = np.array(Label_test[:100])
# Divide the large list into chunks to process and train model
chunk_size = len(Pair1_train) // 10000
learning_rate = 0.001 # smaller is better to reduce loss but slower
shape=(1936,)
print('Creating Base and Siamese Model')   
input_layer = Input(shape=shape)
x = Dense(512, activation='relu')(input_layer)  # First hidden layer
x = Dropout(0.5)(x)  # Dropout layer for overfitting prevention
x = Dense(512, activation='relu')(x)
embeddings = x         

# Create model
model = Model(inputs=input_layer, outputs=embeddings)
# Create siamese model
input1 = Input(shape=shape)
input2 = Input(shape=shape)
# Create left and right twin models
left_model = model(input1)
right_model = model(input2)
# Dot product layer
dot_product = dot([left_model, right_model], axes=1, normalize=True) # Cosine Similiarity [-1,1]
siamese_model = Model(inputs=[input1, input2], outputs=dot_product)
# Model summary 
print(siamese_model.summary())
# Compile model    
siamese_model.compile(optimizer=SGD(learning_rate=learning_rate), loss= Huber(),metrics=['accuracy'])

# Fit model
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
history_list = []
counter = 1
for i in range(0, len(Pair1_train), chunk_size):
    X1 = Pair1_train[i:i + chunk_size]
    X2 = Pair2_train[i:i + chunk_size]
    y = np.array(Label_train[i:i + chunk_size])
    # Test Data 
    print(f' Loop {counter}: Standardizing data rows {i} to {i + chunk_size}' )
    counter+= 1
    # Standardize data along axis 1 (row-wise)
    # X1= zscore(X1, axis=1)
    # X2 = zscore(X2, axis=1)
    X1=np.array(X1)
    X2 = np.array(X2)
    
    fold_history= siamese_model.fit([X1, X2], y, epochs=15, batch_size=1, shuffle=True, 
                                        verbose=2,validation_data=([Pair1_test,Pair2_test], Label_test),
                                        callbacks=[early_stopping])
    history_list.append(fold_history)
    # # Evaluate the model on test data
    # loss, accuracy = siamese_model.evaluate([X1,X2], y, verbose=2)
    # print(f"Test Loss: {loss}")
    # print(f"Test Accuracy: {accuracy}")
    # embedding = model.predict(X1)
    # print(f"Embedding:{embedding}")
    # Save history to a file using Pickle at the end of each loop
    with open('2008shahtalelebi2020_history_list.pkl', 'wb') as f:
        pickle.dump(history_list, f)
    model.save(os.getcwd()+"/2008shahtalelebi2020_eeg_encoder.keras")
    siamese_model.save(os.getcwd()+"/2008shahtalelebi2020_eeg_siamese_model.keras")
 
# Save history to a file using Pickle at the end of program
with open('2008shahtalelebi2020_All_history_list.pkl', 'wb') as f:
    pickle.dump(history_list, f)
model.save(os.getcwd()+"/2008shahtalelebi2020_Final_eeg_encoder.keras")
siamese_model.save(os.getcwd()+"/2008shahtalelebi2020_Final_eeg_siamese_model.keras")
    

  

