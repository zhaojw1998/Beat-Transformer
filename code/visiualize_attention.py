import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import madmom
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from DilatedTransformer import Demixed_DilatedTransformerModel
from spectrogram_dataset import audioDataset
import scipy
#import seaborn as sns
import matplotlib.pyplot as plt

from utils import AverageMeter

import warnings
warnings.filterwarnings('ignore')


#data
SAMPLE_SIZE = None
FPS = 44100/1024
NUM_FOLDS = 8
FOLD = 0
#model
DEVICE='cuda:0'
NORM_FIRST=True
ATTN_LEN=5
INSTR=5
NTOKEN=2
DMODEL=256 
NHEAD=8
DHID=1024
NLAYER=9
#directories
DATASET_PATH = './data/demix_spectrogram_data.npz'
ANNOTATION_PATH = 'data/full_beat_annotation.npz'
MODEL_PATH = "./checkpoints/fold_6_trf_param.pt"
DATA_TO_LOAD = ['gtzan']   #'carnetic', 'harmonix'   # ballroom,  'gtzan', 'hainsworth', 'smc'
TEST_ONLY = ['gtzan']
DEMO_SAVE_ROOT = './save/visualization'
if not os.path.exists(DEMO_SAVE_ROOT):
    os.mkdir(DEMO_SAVE_ROOT)


model = Demixed_DilatedTransformerModel(attn_len=ATTN_LEN,
                                                    instr=INSTR,
                                                    ntoken=NTOKEN, 
                                                    dmodel=DMODEL, 
                                                    nhead=NHEAD, 
                                                    d_hid=DHID, 
                                                    nlayers=NLAYER, 
                                                    norm_first=NORM_FIRST
                                                    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['state_dict'])
model.to(DEVICE)
model.eval()


dataset = audioDataset(data_to_load=DATA_TO_LOAD,
                            test_only_data = TEST_ONLY,
                            data_path = DATASET_PATH, 
                            annotation_path = ANNOTATION_PATH,
                            fps = FPS,
                            sample_size = None,
                            num_folds = NUM_FOLDS)

train_set, val_set, test_set = dataset.get_fold(fold=0)
loader = DataLoader(test_set, batch_size=1, shuffle=False)

beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=10, threshold=0.05)
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=10)

#thresh_beat_meter = AverageMeter()
#pick_beat_meter = AverageMeter()
beat_DBN_meter = AverageMeter()
#thresh_downbeat_meter = AverageMeter()
#pick_downbeat_meter = AverageMeter()
downbeat_DBN_meter = AverageMeter()

with torch.no_grad():
    for idx, (dataset_key, data, beat, downbeat, tempo, root) in tqdm(enumerate(loader), total=len(loader)):
        #if idx == 0:
        #    continue
        data = data.float().to(DEVICE)  #(1, 5, T', 128)

        dataset = dataset_key[0]
        print(root)

        #inference
        pred, pred_t, attn = model.inference(data)

        beat_pred = torch.sigmoid(pred[0, :, 0]).detach().cpu().numpy()
        #np.savetxt(os.path.join(save_dir, 'beat_activation.txt'), beat_pred[:, np.newaxis])
        downbeat_pred = torch.sigmoid(pred[0, :, 1]).detach().cpu().numpy()
        #np.savetxt(os.path.join(save_dir, 'downbeat_activation.txt'), downbeat_pred[:, np.newaxis])

        #gt beat
        beat_gt = torch.nonzero(beat[0]>.5)[:, 0].detach().numpy() / (FPS)
        dnb_beat_pred = beat_tracker(beat_pred)

        downbeat_gt = torch.nonzero(downbeat[0]>.5)[:, 0].detach().numpy() / (FPS)
        combined_act = np.concatenate((np.maximum(beat_pred - downbeat_pred, np.zeros(beat_pred.shape))[:, np.newaxis], downbeat_pred[:, np.newaxis]), axis=-1)   #(T, 2)
        dbn_downbeat_pred = downbeat_tracker(combined_act)
        dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

        beat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dnb_beat_pred, beat_gt)

        downbeat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_downbeat_pred, downbeat_gt)


        fig = plt.figure(figsize=(20, 60))
        for i in range(1, 10):
            layer_attn = attn[i].transpose(-2, -1).squeeze(0).cpu().detach().numpy()
            #layer_attn = np.mean(layer_attn, axis=0)
            layer_attn = layer_attn[2]
            #print(layer_attn.shape)

            fig.add_subplot(9, 4, 4*i-3)
            plt.imshow(layer_attn[0, :, :], cmap='viridis')
            plt.vlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='r', linestyle=':', linewidth=.01)
            plt.hlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='g', linestyle=':', linewidth=.01)

            fig.add_subplot(9, 4, 4*i-2)
            plt.imshow(layer_attn[1, :, :], cmap='viridis')
            plt.vlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='r', linestyle=':', linewidth=.01)
            plt.hlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='g', linestyle=':', linewidth=.01)

            fig.add_subplot(9, 4, 4*i-1)
            plt.imshow(layer_attn[2, :, :], cmap='viridis')
            plt.vlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='r', linestyle=':', linewidth=.01)
            plt.hlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='g', linestyle=':', linewidth=.01)

            fig.add_subplot(9, 4, 4*i)
            plt.imshow(layer_attn[3, :, :], cmap='viridis')
            plt.vlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='r', linestyle=':', linewidth=.01)
            plt.hlines(torch.nonzero(beat[0, :]>.5)[:, 0].detach().numpy(), 0, layer_attn.shape[-1], label='Beats', color='g', linestyle=':', linewidth=.01)
        
        plt.show()
        #fig = ax.get_figure()
        print('saving...')
        plt.savefig(f"{DEMO_SAVE_ROOT}/{root[0].split('/')[-1].replace('.wav', '')}_attention_paterns.pdf", format='pdf', dpi=1200)
        #ax = sns.heatmap(attn[1])
        #print(attn)


        print('beat accuracy:', beat_score_DBN.fmeasure, beat_score_DBN.cmlt, beat_score_DBN.amlt)
        print('dowbbeat accuracy:', downbeat_score_DBN.fmeasure, downbeat_score_DBN.cmlt, downbeat_score_DBN.amlt)
        break
