import os
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed

spec_dir = '/data1/zhaojw/Harmonix_mel.npy'
audio_dir= '/data1/zhaojw/dataset/Harmonix/audio/'

spectrogram = np.load(spec_dir, allow_pickle=True)
failed_songs = []
def convert(i):
    try:
        print('processing song', str(i), flush=True)
        audio_signal  = librosa.feature.inverse.mel_to_audio(spectrogram[i].T, sr=22050, n_fft=2048, hop_length=1024)
        sf.write(os.path.join(audio_dir, str(i).zfill(3)+'.wav'), audio_signal, 22050)
        print('finish song', str(i), flush=True)
    except:
        failed_songs.append(i)

Parallel(n_jobs=16)(delayed(convert)(idx) for idx in range(spectrogram.shape[0]))
print('failed_songs:', failed_songs)