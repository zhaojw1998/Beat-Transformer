import os
import time
import madmom
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm
from matplotlib import pyplot as plt
import librosa.display
from scipy.interpolate import interp1d
from scipy.signal import argrelmax



class dataset_processing(Dataset):
    def __init__(self, full_data, 
                    full_annotation, 
                    audio_files,
                    mode='train', 
                    fold=0, 
                    fps=44100/1024,
                    sample_size = 512,
                    num_folds=8,
                    mask_value=-1,
                    test_only = []  
                    ):
        self.fold = fold
        self.num_folds = num_folds
        self.fps = fps
        self.mode = mode
        self.sample_size = sample_size
        self.MASK_VALUE = mask_value

        self.data = []
        self.beats = []
        self.downbeats = []
        self.tempi = []
        self.root = []

        if self.mode == 'train':
            self.dataset_name = []
            self.train_clip(full_data, full_annotation, test_only=test_only)
        
        elif self.mode == 'validation' or self.mode == 'test':
            self.dataset_name = []
            self.audio_files = []
            self.val_and_test_clip(full_data, full_annotation, audio_files, test_only=test_only)

        full_data = None
        full_annotation = None
            
    def train_clip(self, full_data, full_annotation, num_tempo_bins=300, test_only=[]):
        for fold_idx in tqdm(range(self.num_folds)):
            if (fold_idx != self.fold) and (fold_idx != (self.fold+1)%self.num_folds):
                for key in full_data:
                    if key == test_only:
                        continue
                    #print(f'processing {key} under fold {fold_idx}')
                    for song_idx in range(len(full_data[key][fold_idx])):
                        song = full_data[key][fold_idx][song_idx]   #(t, 5, mel)
                        annotation = full_annotation[key][fold_idx][song_idx]
                        try:
                            #print(annotation, annotation.shape)
                            if len(annotation.shape) == 2:
                                beat = madmom.utils.quantize_events(annotation[:, 0], fps=self.fps, length=len(song))
                            else:
                                beat = madmom.utils.quantize_events(annotation[:], fps=self.fps, length=len(song))
                            beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                            beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                        except:
                            beat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                            print(f'beat load error at {key} dataset, skip it')
                        
                        try:
                            downbeat = annotation[annotation[:, 1] == 1][:, 0]
                            downbeat = madmom.utils.quantize_events(downbeat, fps=self.fps, length=len(song))
                            downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                            downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                        except:
                            downbeat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                            if not ((key == 'smc') or (key == 'musicnet')):
                                print(f'downbeat load error at {key} dataset, skip it')

                        try:
                            #tempo = self.infer_tempo(annotation[:, 0])
                            #tempo = np.array([int(np.round(tempo))])
                            tempo = np.zeros(num_tempo_bins, dtype='float32')
                            if len(annotation.shape) == 2:
                                tempo[int(np.round(self.infer_tempo(annotation[:, 0])))] = 1
                            else:
                                tempo[int(np.round(self.infer_tempo(annotation[:])))] = 1
                            tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                            tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                            tempo = tempo/sum(tempo)
                            #tempo += np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.25)
                        except:
                            #tempo = np.array([self.MASK_VALUE]) 
                            tempo = np.ones(num_tempo_bins, dtype='float32') * self.MASK_VALUE
          
                        if self.sample_size is None:
                                self.dataset_name.append(key)
                                self.data.append(song)
                                self.beats.append(beat)
                                self.downbeats.append(downbeat)
                                self.tempi.append(tempo)
                        else:
                            if len(song) <= self.sample_size:
                                self.dataset_name.append(key)
                                self.data.append(song)
                                self.beats.append(beat)
                                self.downbeats.append(downbeat)
                                self.tempi.append(tempo)
                            else:
                                for i in range(0, len(song)-self.sample_size+1, self.sample_size):
                                    self.dataset_name.append(key)
                                    self.data.append(song[i: i+self.sample_size])
                                    self.beats.append(beat[i: i+self.sample_size])
                                    self.downbeats.append(downbeat[i: i+self.sample_size])
                                    self.tempi.append(tempo)
                                if i + self.sample_size < len(song):
                                    self.dataset_name.append(key)
                                    self.data.append(song[len(song)-self.sample_size:])
                                    self.beats.append(beat[len(song)-self.sample_size:])
                                    self.downbeats.append(downbeat[len(song)-self.sample_size:])
                                    self.tempi.append(tempo)


        #print(len(self.data), len(self.beats), len(self.downbeats))

    def val_and_test_clip(self, full_data, full_annotation, audio_files, num_tempo_bins=300, test_only=[]):
        if self.mode == 'validation':
            fold_idx = (self.fold+1)%self.num_folds
        elif self.mode == 'test':
            fold_idx = self.fold
        for key in tqdm(full_data, total=len(full_data)):
            #print(f'processing {key}')
            if ((self.mode == 'validation') and (key in test_only)):
                continue
            for song_idx in range(len(full_data[key][fold_idx])):
                song = full_data[key][fold_idx][song_idx]
                annotation = full_annotation[key][fold_idx][song_idx]
                audio_file = audio_files[key][fold_idx][song_idx]
                try:
                    if len(annotation.shape) == 2:
                        beat = madmom.utils.quantize_events(annotation[:, 0], fps=self.fps, length=len(song))
                    else: 
                        beat = madmom.utils.quantize_events(annotation[:], fps=self.fps, length=len(song))
                    beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                    beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                except:
                    beat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                    print(f'beat load error at {key} dataset, skip it')

                try:
                    downbeat = annotation[annotation[:, 1] == 1][:, 0]
                    downbeat = madmom.utils.quantize_events(downbeat, fps=self.fps, length=len(song))
                    downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                    downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                except:
                    downbeat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                    if not ((key == 'smc') or (key == 'musicnet')):
                        print(f'downbeat load error at {key} dataset, skip it')

                try:
                    #tempo = self.infer_tempo(annotation[:, 0])
                    #tempo = np.array([int(np.round(tempo))])
                    tempo = np.zeros(num_tempo_bins, dtype='float32')
                    if len(annotation.shape) == 2:
                        tempo[int(np.round(self.infer_tempo(annotation[:, 0])))] = 1
                    else:
                        tempo[int(np.round(self.infer_tempo(annotation[:])))] = 1
                    tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                    tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                    tempo = tempo/sum(tempo)
                except:
                    #tempo = np.array([self.MASK_VALUE]) 
                    tempo = np.ones(num_tempo_bins, dtype='float32') * self.MASK_VALUE
                
                if self.sample_size is None:
                        self.dataset_name.append(key)
                        self.root.append(audio_file)
                        self.data.append(song)
                        self.beats.append(beat)
                        self.downbeats.append(downbeat)
                        self.tempi.append(tempo)
                else:
                    eval_sample_size = int(44100/1024 * 420)
                    if len(song) <= eval_sample_size:
                        self.dataset_name.append(key)
                        self.root.append(audio_file)
                        self.data.append(song)
                        self.beats.append(beat)
                        self.downbeats.append(downbeat)
                        self.tempi.append(tempo)
                    else:
                        for i in range(0, len(song)-eval_sample_size+1, eval_sample_size):
                            self.dataset_name.append(key)
                            self.root.append(audio_file)
                            self.data.append(song[i: i+eval_sample_size])
                            self.beats.append(beat[i: i+eval_sample_size])
                            self.downbeats.append(downbeat[i: i+eval_sample_size])
                            self.tempi.append(tempo)
                        if i + eval_sample_size < len(song):
                            self.dataset_name.append(key)
                            self.root.append(audio_file)
                            self.data.append(song[len(song)-eval_sample_size:])
                            self.beats.append(beat[len(song)-eval_sample_size:])
                            self.downbeats.append(downbeat[len(song)-eval_sample_size:])
                            self.tempi.append(tempo)

    def infer_tempo(self, beats, hist_smooth=4, no_tempo=-1):
        ibis = np.diff(beats) * self.fps
        bins = np.bincount(np.round(ibis).astype(int))
        # if no beats are present, there is no tempo
        if not bins.any():
            return no_tempo
        # smooth histogram bins
        if hist_smooth > 0:
            bins = madmom.audio.signal.smooth(bins, hist_smooth)
        #print(bins)
        intervals = np.arange(len(bins))       
        # create interpolation function
        interpolation_fn = interp1d(intervals, bins, 'quadratic')
        # generate new intervals with 1000x the resolution
        intervals = np.arange(intervals[0], intervals[-1], 0.001)
        tempi = 60.0 * self.fps / intervals
        # apply quadratic interpolation
        bins = interpolation_fn(intervals)
        peaks = argrelmax(bins, mode='wrap')[0]
        if len(peaks) == 0:
            # no peaks, no tempo
            return no_tempo
        else:
            # report only the strongest tempo
            sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
            return tempi[sorted_peaks][0]
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """x = np.sum(self.data[index], axis=1).transpose(1, 0) #(dmodel, T)
        x = librosa.power_to_db(x, ref=np.max)
        x = x.T[np.newaxis, :, :]
        x = np.repeat(x, 5, axis=0)
        return self.dataset_name[index], x, self.beats[index], self.downbeats[index], self.tempi[index]"""
        
        x = np.transpose(self.data[index],( 1, 2, 0))   #5, dmodel, T
        #x = x + .25 * np.sum(x, axis=0, keepdims=True)
        #x = [librosa.power_to_db(x[i], ref=np.max) for i in range(x.shape[0])]

        np.random.seed()
        if self.mode == 'train':
            p = np.random.rand()
            if p < .5:  #50% time use 5 subspectrograms
                pass
            else:
                idx_sum = np.random.choice(len(x), size=2, replace=False)
                x = [x[i] for i in range(len(x)) if i not in idx_sum] + [x[idx_sum[0]] + x[idx_sum[1]]]
                q = np.random.rand()
                if q < .6:  #30% time use 4 subspectrograms
                    pass
                else:
                    idx_sum = np.random.choice(len(x), size=2, replace=False)
                    x = [x[i] for i in range(len(x)) if i not in idx_sum] + [x[idx_sum[0]] + x[idx_sum[1]]]
                    r = np.random.rand()
                    if r < .5:  #10% time use 3 subspectrograms
                        pass
                    else:  #10% time use 2 subspectrograms
                        idx_sum = np.random.choice(len(x), size=2, replace=False)
                        x = [x[i] for i in range(len(x)) if i not in idx_sum] + [x[idx_sum[0]] + x[idx_sum[1]]]
            
        x = [librosa.power_to_db(x[i], ref=np.max) for i in range(len(x))]
        x = np.transpose(np.array(x), (0, 2, 1))    #T, instr, dmodel

        if self.mode == 'test':
            return self.dataset_name[index], x, self.beats[index], self.downbeats[index], self.tempi[index], self.root[index]
        else:
            return self.dataset_name[index], x, self.beats[index], self.downbeats[index], self.tempi[index]
        




class audioDataset(object):
    def __init__(self, data_to_load=['ballroom', 'carnetic', 'gtzan', 'hainsworth', 'smc', 'harmonix'],
                        test_only_data = ['hainsworth'],
                        data_path="/data1/zhaojw/dataset/linear_spectrogram_data.npz", 
                        annotation_path="/data1/zhaojw/dataset/beat_annotation.npz",
                        fps=44100/1024,
                        SEED = 0,
                        num_folds=8,
                        mask_value = -1,
                        sample_size = 512
                ):

        self.fps = fps
        self.sample_size = sample_size
        self.mask_value = mask_value
        self.num_folds = num_folds
        self.test_only_data = test_only_data

        load_linear_spectr = np.load(data_path, allow_pickle=True)
        load_annotation = np.load(annotation_path, allow_pickle=True)

        self.full_data = {}
        self.full_annotation = {}
        self.audio_files = {}
        for key in load_linear_spectr:
            if key in data_to_load:
                time1 = time.time()
                print(f'loading {key} dataset ...')
                data = load_linear_spectr[key]
                annotation = load_annotation[key]
                assert(len(data) == len(annotation))

                with open(f'./data/audio_lists/{key}.txt', 'r') as f:
                    audio_root = f.readlines()
                audio_root = [item.replace('\n', '') for item in audio_root]
                assert(len(data) == len(audio_root))
                print(f'finish loading {key} with shape {data.shape}, using {time.time()-time1}s.')
                #fold split
                self.full_data[key] = {}
                self.full_annotation[key] = {}
                self.audio_files[key] = {}
                if key in self.test_only_data:
                    FOLD_SIZE = len(data) // num_folds
                    np.random.seed(SEED)
                    np.random.shuffle(data)
                    np.random.seed(SEED)
                    np.random.shuffle(annotation)
                    np.random.seed(SEED)
                    np.random.shuffle(audio_root)
                    for i in range(num_folds):
                        self.full_data[key][i] = data[:]
                        self.full_annotation[key][i] = annotation[:]
                        self.audio_files[key][i] = audio_root[:]
                else:
                    FOLD_SIZE = len(data) // num_folds
                    np.random.seed(SEED)
                    np.random.shuffle(data)
                    np.random.seed(SEED)
                    np.random.shuffle(annotation)
                    np.random.seed(SEED)
                    np.random.shuffle(audio_root)
                    for i in range(num_folds-1):
                        self.full_data[key][i] = data[i*FOLD_SIZE: (i+1)*FOLD_SIZE]
                        self.full_annotation[key][i] = annotation[i*FOLD_SIZE: (i+1)*FOLD_SIZE]
                        self.audio_files[key][i] = audio_root[i*FOLD_SIZE: (i+1)*FOLD_SIZE]
                    self.full_data[key][num_folds-1] = data[(num_folds-1)*FOLD_SIZE: len(data)]
                    self.full_annotation[key][num_folds-1] = annotation[(num_folds-1)*FOLD_SIZE: len(annotation)]
                    self.audio_files[key][num_folds-1] = audio_root[(num_folds-1)*FOLD_SIZE: len(audio_root)]
                data = None
                annotation = None

    def get_fold(self, fold=0):
        print('processing train_set')
        train_set = dataset_processing(full_data=self.full_data, 
                                        full_annotation=self.full_annotation, 
                                        audio_files=None,
                                        mode='train', 
                                        fps=self.fps,
                                        fold=fold, 
                                        sample_size = self.sample_size,
                                        num_folds=self.num_folds,
                                        mask_value=self.mask_value,
                                        test_only=self.test_only_data
                                        )

        print('processing val_set')
        val_set = dataset_processing(full_data=self.full_data, 
                                        full_annotation=self.full_annotation, 
                                        audio_files=self.audio_files,
                                        mode='validation', 
                                        fps=self.fps,
                                        fold=fold, 
                                        sample_size=self.sample_size,
                                        num_folds=self.num_folds,
                                        mask_value=self.mask_value,
                                        test_only=self.test_only_data
                                        )

        print('processing test_set')
        test_set = dataset_processing(full_data=self.full_data, 
                                        full_annotation=self.full_annotation, 
                                        audio_files=self.audio_files,
                                        mode='test', 
                                        fps=self.fps,
                                        fold=fold, 
                                        sample_size=self.sample_size,
                                        num_folds=self.num_folds,
                                        mask_value=self.mask_value,
                                        test_only=self.test_only_data
                                        )
        return train_set, val_set, test_set


    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    #data_to_load=['ballroom', 'carnetic', 'gtzan', 'hainsworth', 'smc', 'harmonix']
    dataset = audioDataset(data_to_load=['smc'],
                        test_only_data = ['gtzan'],
                        data_path = "./data/demix_spectrogram_data.npz", 
                        annotation_path = "./data/full_beat_annotation.npz",
                        fps = 44100/1024,
                        sample_size = None,
                        num_folds = 8)
    # Fold Splitting
    train_set, val_set, test_set = dataset.get_fold(fold=0)
    #train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    #val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    #for i, (key, data, beat, downbeat, tempo) in enumerate(val_data):
    for i, (key, data, beat, downbeat, tempo, root) in enumerate(test_loader):
        print('key:', key)
        print('data:', data.shape)
        print('beat:', beat.shape)
        #print('beat:', torch.nonzero(beat))
        print('downbeat:', downbeat.shape)
        print('tempo:', tempo.shape)
        print('audio_root:', root)
        #print('downbeat:', torch.nonzero(downbeat))
        break
    
