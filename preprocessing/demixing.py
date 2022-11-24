from spleeter.audio.adapter import AudioAdapter
from spleeter.audio import Codec, STFTBackend
from spleeter.separator import Separator
from spleeter.audio import STFTBackend
from tqdm import tqdm

import librosa
import numpy as np 
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)


class new_Separator(Separator):
    def __init__(self, params_descriptor, MWF=False, stft_backend=STFTBackend.AUTO, multiprocess=True):
        super(new_Separator, self).__init__(params_descriptor, MWF, stft_backend, multiprocess)
        self.mel_f = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
        self.audio_adapter = AudioAdapter.default()
  
    def _separate_librosa(self, waveform, audio_descriptor):
        """
        re-define _separate_librosa so that it outputs spectrogram instead of audio
        """
        with self._tf_graph.as_default():
            #out = []
            out = {}
            features = self._get_features()
            # TODO: fix the logic, build sometimes return,
            #       sometimes set attribute.
            outputs = self._get_builder().outputs
            stft = self._stft(waveform)
            if stft.shape[-1] == 1:
                stft = np.concatenate([stft, stft], axis=-1)
            elif stft.shape[-1] > 2:
                stft = stft[:, :2]
            sess = self._get_session()
            outputs = sess.run(
                outputs,
                feed_dict=self._get_input_provider().get_feed_dict(
                    features, stft, audio_descriptor
                ),
            )
            for inst in self._get_builder().instruments:
                out[inst] = self._stft(
                    outputs[inst], inverse=True, length=waveform.shape[0]
                )
            
            for inst in self._get_builder().instruments:
                #print(outputs[inst].shape, outputs[inst][100, 1024:1034, 0])
                stft = np.mean(outputs[inst], axis=-1)
                magnitude = np.abs(stft)
                mel_spec = np.dot(magnitude**2, self.mel_f)
                out.append(mel_spec)
            return np.stack(out)


class prepare_spectrgram_data(object):
    def __init__(self):
        super(prepare_spectrgram_data, self).__init__()
        self.data_dir = {
            'ballroom': '/data1/zhaojw/dataset/Ballroom/',
            'carnetic': '/data1/zhaojw/dataset/Carnetic/audio/',
            'gtzan': '/data1/zhaojw/dataset/GTZAN/', 
            'hainsworth': '/data1/zhaojw/dataset/Hainsworth/wavs/',
            'smc': '/data1/zhaojw/dataset/SMC/SMC_MIREX_Audio/',
            'harmonix': '/data1/zhaojw/dataset/Harmonix/audio/'
        }
        self.beat_annotation_dir = {
            'ballroom': '/data1/zhaojw/dataset/ISMIR2019-master/ballroom/annotations/beats/',
            'carnetic': '/data1/zhaojw/dataset/Carnetic/annotations/beats/',
            'gtzan': '/data1/zhaojw/dataset/ISMIR2019-master/gtzan/annotations/beats/',
            'hainsworth': '/data1/zhaojw/dataset/ISMIR2019-master/hainsworth/annotations/beats/',
            'smc': '/data1/zhaojw/dataset/ISMIR2019-master/smc/annotations/beats/',
            'harmonix': '/data1/zhaojw/dataset/Harmonix/annotation/'
        }

        self.SR = 44100
        self.separator = new_Separator('spleeter:5stems')
        self.audio_loader = AudioAdapter.default()

        print('initailize harmonix ...')
        harmonix_data, harmonix_annotation, not_found = self.initialize_harmonix()
        print(harmonix_data.shape, harmonix_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/data1/zhaojw/dataset/linear_spectrogram_harmonix.npy', harmonix_data, allow_pickle=True)

        print('initailize carnetic ...')
        carnetic_data, carnetic_annotation, not_found = self.initialize_carnetic()
        print(carnetic_data.shape, carnetic_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/data1/zhaojw/dataset/linear_spectrogram_carnetic.npy', carnetic_data, allow_pickle=True)

        print('initailize gtzan ...')
        gtzan_data, gtzan_annotation, not_found = self.initialize_gtzan()
        print(gtzan_data.shape, gtzan_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/data1/zhaojw/dataset/linear_spectrogram_gtzan.npy', gtzan_data, allow_pickle=True)

        print('initailize hainsworth ...')
        hainsworth_data, hainsworth_annotation, not_found = self.initialize_hainsworth()
        print(hainsworth_data.shape, hainsworth_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/data1/zhaojw/dataset/linear_spectrogram_hainsworth.npy', hainsworth_data, allow_pickle=True)

        print('initailize smc ...')
        smc_data, smc_annotation, not_found = self.initialize_smc()
        print(smc_data.shape, smc_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/data1/zhaojw/dataset/linear_spectrogram_smc.npy', smc_data, allow_pickle=True)

        print('initailize ballroom ...')
        ballroom_data, ballroom_annotation, not_found = self.initialize_ballroom()
        print(ballroom_data.shape, ballroom_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/data1/zhaojw/dataset/linear_spectrogram_ballroom.npy', ballroom_data, allow_pickle=True)

        np.savez_compressed('/data1/zhaojw/dataset/demix_spectrogram_data.npz', ballroom=ballroom_data,\
                                            carnetic = carnetic_data,\
                                            gtzan = gtzan_data,\
                                            hainsworth = hainsworth_data,\
                                            smc = smc_data,\
                                            harmonix = harmonix_data)

        np.savez_compressed('/data1/zhaojw/dataset/full_beat_annotation.npz', ballroom=ballroom_annotation,\
                                            carnetic = carnetic_annotation,\
                                            gtzan = gtzan_annotation,\
                                            hainsworth = hainsworth_annotation,\
                                            smc = smc_annotation,\
                                            harmonix = harmonix_annotation)

    def initialize_ballroom(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['ballroom']
        annotation_dir = self.beat_annotation_dir['ballroom']
        for gnere in tqdm(os.listdir(data_dir)):
            if gnere[0].isupper():
                gnere_dir = os.path.join(data_dir, gnere)
                for audio_name in os.listdir(gnere_dir):
                    #load audio
                    audio_dir = os.path.join(gnere_dir, audio_name)
                    waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
                    #load beat annotations
                    beat_dir = os.path.join(annotation_dir, 'ballroom_'+audio_name.split('.')[0]+'.beats')
                    try:
                        values = np.loadtxt(beat_dir, ndmin=1)
                    except OSError:
                        not_found_error.append(audio_name)
                    specs = self.separator.separate(waveform).transpose(1, 0, 2)
                    break
                    #print('specs', specs.shape)
                    spectrogram.append(specs)
                    annotation.append(values)
        return np.array(spectrogram, dtype=object), np.array(annotation), not_found_error

    def initialize_carnetic(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['carnetic']
        annotation_dir = self.beat_annotation_dir['carnetic']
        for audio_name in tqdm(os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            #print(audio.shape, np.mean(audio))
            beat_dir = os.path.join(annotation_dir, audio_name.split('.')[0]+'.beats')
            try:
                values = np.loadtxt(beat_dir, delimiter=',', ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = self.separator.separate(waveform).transpose(1, 0, 2)
            spectrogram.append(specs)
            annotation.append(values)
        return np.array(spectrogram), np.array(annotation), not_found_error
    
    def initialize_gtzan(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['gtzan']
        annotation_dir = self.beat_annotation_dir['gtzan']
        for gnere in tqdm(os.listdir(data_dir)):
            if '.' in gnere:
                continue
            gnere_dir = os.path.join(data_dir, gnere)
            for audio_name in os.listdir(gnere_dir):
                audio_dir = os.path.join(gnere_dir, audio_name)
                waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
                beat_dir = os.path.join(annotation_dir, 'gtzan_'+audio_name.split('.')[0]+'_'+audio_name.split('.')[1]+'.beats')
                try:
                    values = np.loadtxt(beat_dir, ndmin=1)
                except OSError:
                    not_found_error.append(audio_name)
                specs = self.separator.separate(waveform).transpose(1, 0, 2)
                spectrogram.append(specs)
                annotation.append(values)
        return np.array(spectrogram), np.array(annotation), not_found_error

    def initialize_hainsworth(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['hainsworth']
        annotation_dir = self.beat_annotation_dir['hainsworth']
        for audio_name in tqdm(os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            beat_dir = os.path.join(annotation_dir, 'hainsworth_'+audio_name.split('.')[0]+'.beats')
            try:
                values = np.loadtxt(beat_dir, ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = self.separator.separate(waveform).transpose(1, 0, 2)
            spectrogram.append(specs)
            annotation.append(values)
        return np.array(spectrogram), np.array(annotation), not_found_error

    def initialize_smc(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['smc']
        annotation_dir = self.beat_annotation_dir['smc']
        for audio_name in tqdm(os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            beat_dir = os.path.join(annotation_dir, audio_name.lower().split('.')[0]+'.beats')
            try:
                values = np.loadtxt(beat_dir, ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = self.separator.separate(waveform).transpose(1, 0, 2)
            spectrogram.append(specs)
            annotation.append(values)
        return np.array(spectrogram), np.array(annotation), not_found_error

    def initialize_harmonix(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['harmonix']
        annotation_dir = self.beat_annotation_dir['harmonix']
        for audio_name in tqdm(os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            beat_dir = os.path.join(annotation_dir, audio_name.split('.')[0]+'.txt')
            try:
                values = np.loadtxt(beat_dir, ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = self.separator.separate(waveform).transpose(1, 0, 2)
            spectrogram.append(specs)
            annotation.append(values)
        return np.array(spectrogram), np.array(annotation), not_found_error


if __name__ == '__main__':
    data = prepare_spectrgram_data()