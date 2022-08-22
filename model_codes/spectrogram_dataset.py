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
                    downsample_size=1,
                    hop_size = 128,
                    num_folds=8,
                    mask_value=-1  
                    ):
        self.fold = fold
        self.num_folds = num_folds
        self.fps = fps
        self.mode = mode
        self.sample_size = sample_size
        self.DS = downsample_size
        self.MASK_VALUE = mask_value

        self.data = []
        self.beats = []
        self.downbeats = []
        self.tempi = []
        self.root = []

        if self.mode == 'train':
            self.dataset_name = []
            self.hop_size = hop_size
            self.train_clip(full_data, full_annotation)
        
        elif self.mode == 'validation' or self.mode == 'test':
            self.dataset_name = []
            self.audio_files = []
            self.hop_size = hop_size
            self.val_and_test_clip(full_data, full_annotation, audio_files)

        full_data = None
        full_annotation = None
            
    def train_clip(self, full_data, full_annotation, num_tempo_bins=300):
        for fold_idx in tqdm(range(self.num_folds)):
            if (fold_idx != self.fold) and (fold_idx != (self.fold+1)%self.num_folds):
                for key in full_data:
                    if key == 'gtzan':
                        continue
                    #print(f'processing {key} under fold {fold_idx}')
                    for song_idx in range(len(full_data[key][fold_idx])):
                        song = full_data[key][fold_idx][song_idx]   #(t, 5, mel)
                        annotation = full_annotation[key][fold_idx][song_idx]
                        try:
                            #print(annotation, annotation.shape)
                            if len(annotation.shape) == 2:
                                beat = madmom.utils.quantize_events(annotation[:, 0], fps=self.fps/self.DS, length=len(song)//self.DS)
                            else:
                                beat = madmom.utils.quantize_events(annotation[:], fps=self.fps/self.DS, length=len(song)//self.DS)
                            beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                            beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                        except:
                            beat = np.ones(len(song)//self.DS, dtype='float32') * self.MASK_VALUE
                            print(f'beat load error at {key} dataset, skip it')
                        
                        try:
                            downbeat = annotation[annotation[:, 1] == 1][:, 0]
                            downbeat = madmom.utils.quantize_events(downbeat, fps=self.fps/self.DS, length=len(song)//self.DS)
                            downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                            downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                        except:
                            downbeat = np.ones(len(song)//self.DS, dtype='float32') * self.MASK_VALUE
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

    def val_and_test_clip(self, full_data, full_annotation, audio_files, num_tempo_bins=300):
        if self.mode == 'validation':
            fold_idx = (self.fold+1)%self.num_folds
        elif self.mode == 'test':
            fold_idx = self.fold
        for key in tqdm(full_data, total=len(full_data)):
            #print(f'processing {key}')
            if ((self.mode == 'validation') and (key == 'gtzan')):
                continue
            for song_idx in range(len(full_data[key][fold_idx])):
                song = full_data[key][fold_idx][song_idx]
                annotation = full_annotation[key][fold_idx][song_idx]
                audio_file = audio_files[key][fold_idx][song_idx]
                try:
                    if len(annotation.shape) == 2:
                        beat = madmom.utils.quantize_events(annotation[:, 0], fps=self.fps/self.DS, length=len(song)//self.DS)
                    else: 
                        beat = madmom.utils.quantize_events(annotation[:], fps=self.fps/self.DS, length=len(song)//self.DS)
                    beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                    beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                except:
                    beat = np.ones(len(song)//self.DS, dtype='float32') * self.MASK_VALUE
                    print(f'beat load error at {key} dataset, skip it')

                try:
                    downbeat = annotation[annotation[:, 1] == 1][:, 0]
                    downbeat = madmom.utils.quantize_events(downbeat, fps=self.fps/self.DS, length=len(song)//self.DS)
                    downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                    downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                except:
                    downbeat = np.ones(len(song)//self.DS, dtype='float32') * self.MASK_VALUE
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
                        sample_size = 512,
                        downsample_size = 1,
                        hop_size = 128  
                ):

        self.fps = fps
        self.sample_size = sample_size
        self.mask_value = mask_value
        self.downsample_size = downsample_size
        self.hop_size = hop_size
        self.num_folds = num_folds

        load_linear_spectr = np.load(data_path, allow_pickle=True)
        load_annotation = np.load(annotation_path, allow_pickle=True)

        audio_dir = {
            'ballroom': '/data1/zhaojw/dataset/Ballroom/',
            'carnetic': '/data1/zhaojw/dataset/Carnetic/audio/',
            'gtzan': '/data1/zhaojw/dataset/GTZAN/', 
            'hainsworth': '/data1/zhaojw/dataset/Hainsworth/wavs/',
            'smc': '/data1/zhaojw/dataset/SMC/SMC_MIREX_Audio/',
            'harmonix': '/data1/zhaojw/dataset/Harmonix/audio/',
            'musicnet': '/data1/zhaojw/dataset/musicnet/',
            'rwc': '/data1/zhaojw/dataset/RWC_P/AUDIO/'
        }

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
                if (key == 'ballroom') or (key == 'gtzan'):
                    audio_root = []
                    for genre in os.listdir(audio_dir[key]):
                        genere_dir = os.path.join(audio_dir[key], genre)
                        audio_root += [os.path.join(genere_dir, x) for x in os.listdir(genere_dir)]
                elif key == 'musicnet':
                    audio_root = []
                    data_splits = ['train_data', 'test_data']
                    for split in data_splits:
                        split_dir = os.path.join(audio_dir[key], split)
                        for x in os.listdir(split_dir):
                            if not x in ['2116.wav', '2117.wav', '1813.wav', '2119.wav', '1812.wav', '2159.wav', '2118.wav']:
                                audio_root.append(os.path.join(split_dir, x))
                else:
                    audio_root = [os.path.join(audio_dir[key], x) for x in os.listdir(audio_dir[key])]
                if len(audio_root) != len(data):
                    print(f'found {len(audio_root)} original audio files and {len(data)} spectrograms.')
                print(f'finish loading {key} with shape {data.shape}, using {time.time()-time1}s.')
                #fold split
                self.full_data[key] = {}
                self.full_annotation[key] = {}
                self.audio_files[key] = {}
                if key in test_only_data:
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
                                        downsample_size=self.downsample_size,
                                        hop_size = self.hop_size,
                                        num_folds=self.num_folds,
                                        mask_value=self.mask_value
                                        )

        print('processing val_set')
        val_set = dataset_processing(full_data=self.full_data, 
                                        full_annotation=self.full_annotation, 
                                        audio_files=self.audio_files,
                                        mode='validation', 
                                        fps=self.fps,
                                        fold=fold, 
                                        sample_size=self.sample_size,
                                        downsample_size=self.downsample_size,
                                        hop_size = self.hop_size,
                                        num_folds=self.num_folds,
                                        mask_value=self.mask_value
                                        )

        print('processing test_set')
        test_set = dataset_processing(full_data=self.full_data, 
                                        full_annotation=self.full_annotation, 
                                        audio_files=self.audio_files,
                                        mode='test', 
                                        fps=self.fps,
                                        fold=fold, 
                                        sample_size=self.sample_size,
                                        downsample_size=self.downsample_size,
                                        num_folds=self.num_folds,
                                        mask_value=self.mask_value
                                        )
        return train_set, val_set, test_set


    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    #data_to_load=['ballroom', 'carnetic', 'gtzan', 'hainsworth', 'smc', 'harmonix']
    dataset = audioDataset(data_to_load=['smc'],
                        test_only_data = ['gtzan'],
                        data_path = "/data1/zhaojw/dataset/linear_spectrogram_data.npz", 
                        annotation_path = "/data1/zhaojw/dataset/beat_annotation.npz",
                        fps = 44100/1024,
                        sample_size = None,
                        downsample_size=1,
                        hop_size = 128,
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
        print('root:', root)
        #print('downbeat:', torch.nonzero(downbeat))
        break
    