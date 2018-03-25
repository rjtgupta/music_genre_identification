
import glob, os
import scipy.io.wavfile
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_confusion_matrix

artists = ['Pop', 'Rap', 'Rock']

song_data = []
sample_rate = []

for artist in artists:
    
    files = 'Full_Songs/'+artist+'/*.mp3'
    songs_list = glob.glob(files)
    
    for song in songs_list:
        sound = AudioSegment.from_mp3(song)
        
        folder_names=song.split('/')
        folder_mp3 = folder_names[0]+'/'+folder_names[1]+'/'+'Wav_files/'+folder_names[2]
        wav_file = folder_mp3[:-4]+".wav"
        sound.export(wav_file, format='wav')
        
        samplerate, data = scipy.io.wavfile.read(wav_file)
        
        #We are just keeping the 5 seconds into our song for the analysis
        #We are keeping the 0th index (sound coming from left speaker- making audio mono instead of stereo)
        for i in range(0,20):
            song_data.append(data[5*i*samplerate:5*(i+1)*samplerate,0])
        
            sample_rate.append(samplerate)

spec,time,freq,image = plt.specgram(song_data[0], Fs = sample_rate[0])

rows, cols = spec.shape

a = np.zeros(rows*cols, float)
X = np.zeros((rows*cols, len(song_data)), float)

for s in range (len(song_data)):
    k = 0
    spec,_,_,_ = plt.specgram(song_data[s], Fs = sample_rate[s])
    for i in range (rows):
        for j in range (cols):
            a[k] = spec[i][j]
            k += 1
    X[:,s] = a

U, sigma, V = np.linalg.svd(X, full_matrices=False)
#As V returned by np.linalg.svd is actually V'
#We will take do V =V.T to get its correct shape
V = V.T

length = []
for i in range (sigma.shape[0]):
    length.append(i)
    
plt.scatter(length, sigma/sum(sigma), s = sigma*0.0000005)
plt.ylim([0,0.07])
plt.title('Singular Value Spectrum')
plt.xlabel('mode number')
plt.ylabel('Variance in mode')

x1 = np.random.permutation(480)
x2 = np.random.permutation(480)
x3 = np.random.permutation(480)

xpop = V[0:480,0:191]
xrap = V[480:960,0:191]
xrock= V[960:1440,0:191]

xtrain = xpop[x1[:450],:]
xtrain = np.append(xtrain, xrap[x2[:450],:],0)
xtrain = np.append(xtrain, xrock[x3[:450],:],0)

xtest = xpop[x1[450:],:]
xtest = np.append(xtest, xrap[x2[450:],:],0)
xtest = np.append(xtest, xrock[x3[450:],:],0)

ytrain = []
ytest = []
for i in range(1350):
    if(i<450):
        ytrain.append('Pop')
    elif(i>=450 and i<900):
        ytrain.append('Rap')
    else:
        ytrain.append('Rock')
        
for i in range(90):
    if(i<30):
        ytest.append('Pop')
    elif(i>=30 and i<60):
        ytest.append('Rap')
    else:
        ytest.append('Rock')

pipeline = Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel='rbf', C=21, gamma = .00095) )
            ))
pipeline.fit(xtrain, ytrain)
ypred_svm = pipeline.predict(xtest)

accuracy_score(ytest, ypred_svm)

conf_svm = confusion_matrix(ytest, ypred_svm)
fig, ax = plot_confusion_matrix(conf_mat=conf_svm)
plt.show()

