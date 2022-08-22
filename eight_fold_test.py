import os
import pickle
import torch
import madmom
import numpy as np
from utils import AverageMeter
from torch.utils.data import DataLoader
from DilatedTransformer import Demixed_DilatedTransformerModel
from spectrogram_dataset import audioDataset
from tqdm import tqdm
import shutil

import warnings
warnings.filterwarnings('ignore')


FPS = 44100 / 1024
NUM_FOLDS = 8
DEVICE='cuda:3'
#model
NORM_FIRST=True
ATTN_LEN=5
INSTR=5
NTOKEN=2
DMODEL=256
NHEAD=8
DHID=1024
NLAYER=9
#directories
DATASET_PATH = "/data1/zhaojw/dataset/full_linear_spectrogram_data_11k.npz"
ANNOTATION_PATH = '/data1/zhaojw/dataset/full_beat_annotation.npz'
#MODEL_PATH = '/data1/zhaojw/test_logs_and_weights/20220414-15-54-32_SS-tempo-rpe-EDT-9layers-attn_len_5_asymmetric-RAdam_look_ahead-patience_2-rwc_enriched/'
DATA_TO_LOAD = ['ballroom', 'hainsworth', 'smc', 'rwc', 'carnetic', 'harmonix']   #'ballroom', 'gtzan', 'hainsworth', 'smc', 'rwc', 'carnetic', 'harmonix'
TEST_ONLY = []
DEMO_SAVE_ROOT = '/data1/zhaojw/shell_beat_transformer_final/'# + MODEL_PATH.replace('/data1/zhaojw/test_logs_and_weights/', '')
if not os.path.exists(DEMO_SAVE_ROOT):
    os.makedirs(DEMO_SAVE_ROOT)


PARAM_PATH = {
    0: "./model_weights/fold_0_trf_param.pt",
    1: "./model_weights/fold_1_trf_param.pt",
    2: "./model_weights/fold_2_trf_param.pt",
    3: "./model_weights/fold_3_trf_param.pt",
    4: "./model_weights/fold_4_trf_param.pt",
    5: "./model_weights/fold_5_trf_param.pt",
    6: "./model_weights/fold_6_trf_param.pt",
    7: "./model_weights/fold_7_trf_param.pt"
}



def infer_activation():
    #占坑
    model = Demixed_DilatedTransformerModel().to(DEVICE)

    dataset = audioDataset(data_to_load=DATA_TO_LOAD,
                            test_only_data = TEST_ONLY,
                            data_path = DATASET_PATH, 
                            annotation_path = ANNOTATION_PATH,
                            fps = FPS,
                            sample_size = None,
                            downsample_size=1,
                            hop_size = None,
                            num_folds = NUM_FOLDS)

    inference_pred = {}
    beat_gt = {}
    downbeat_gt = {}

    for FOLD in range(NUM_FOLDS):
        print(f'\nFold {FOLD}')
        train_set, val_set, test_set = dataset.get_fold(fold=FOLD)
        #loader = DataLoader(val_set, batch_size=1, shuffle=False)
        loader = DataLoader(test_set, batch_size=1, shuffle=False)

        model = Demixed_DilatedTransformerModel(attn_len=ATTN_LEN,
                                                    instr=INSTR,
                                                    ntoken=NTOKEN, 
                                                    dmodel=DMODEL, 
                                                    nhead=NHEAD, 
                                                    d_hid=DHID, 
                                                    nlayers=NLAYER, 
                                                    norm_first=NORM_FIRST
                                                    )
        #model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'Fold_{FOLD}', 'model', 'trf_param_012.pt'), map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(PARAM_PATH[FOLD], map_location=torch.device('cpu'))['state_dict'])
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            for idx, (dataset_key, data, beat, downbeat, tempo, root) in tqdm(enumerate(loader), total=len(loader)):
                #data
                data = data.float().to(DEVICE)
                pred, _ = model(data)
                beat_pred = torch.sigmoid(pred[0, :, 0]).detach().cpu().numpy()
                downbeat_pred = torch.sigmoid(pred[0, :, 1]).detach().cpu().numpy()

                beat = torch.nonzero(beat[0]>.5)[:, 0].detach().numpy() / (FPS)
                downbeat = torch.nonzero(downbeat[0]>.5)[:, 0].detach().numpy() / (FPS)

                dataset_key = dataset_key[0]
                root = root[0]
                if not dataset_key in inference_pred:
                    inference_pred[dataset_key] = []
                    beat_gt[dataset_key] = []
                    downbeat_gt[dataset_key] = []
                inference_pred[dataset_key].append(np.stack((beat_pred, downbeat_pred), axis=0))
                beat_gt[dataset_key].append(beat)
                downbeat_gt[dataset_key].append(downbeat)

                #audio_name = root.split('/')[-1]
                #save_dir = os.path.join(DEMO_SAVE_ROOT, dataset_key, audio_name)
                #if not os.path.exists(save_dir):
                #    os.makedirs(save_dir)
                #if not os.path.isfile(os.path.join(save_dir, audio_name)):
                #    shutil.copy(root, os.path.join(save_dir, audio_name))
                #np.savetxt(os.path.join(save_dir, 'beat_activation.txt'), beat_pred[:, np.newaxis])
                #np.savetxt(os.path.join(save_dir, 'downbeat_activation.txt'), downbeat_pred[:, np.newaxis])
                #np.savetxt(os.path.join(save_dir, 'gt_beat.txt'), beat[:, np.newaxis])
                #np.savetxt(os.path.join(save_dir, 'gt_downbeat.txt'), downbeat[:, np.newaxis])


    print('saving prediction ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'inference_pred.pkl'), 'wb') as f:
        pickle.dump( inference_pred, f)
    print('saving gt ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gt.pkl'), 'wb') as f:
        pickle.dump(beat_gt, f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'down_gt.pkl'), 'wb') as f:
        pickle.dump(downbeat_gt, f)


def demo_inference_dbn():
    beat_DBN_meter = AverageMeter()
    downbeat_DBN_meter = AverageMeter()

    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=100, 
                                                                    observation_lambda=6, 
                                                                    num_tempi=None, 
                                                                    threshold=0.2)

    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=100, 
                                                                    observation_lambda=6, 
                                                                    num_tempi=None, 
                                                                    threshold=0.2)
    for dataset in DATA_TO_LOAD:
        save_dir = os.path.join(DEMO_SAVE_ROOT, dataset)
        print(f'Inferencing on {dataset} dataset ...')
        for song in tqdm(os.listdir(save_dir)):
            song_dir = os.path.join(save_dir, song)
            beat_pred = np.loadtxt(os.path.join(song_dir, 'beat_activation.txt'))
            downbeat_pred = np.loadtxt(os.path.join(song_dir, 'downbeat_activation.txt'))
            beat_gt = np.loadtxt(os.path.join(song_dir, 'gt_beat.txt'))
            
            dbn_beat_pred = beat_tracker(beat_pred)
            np.savetxt(os.path.join(song_dir, 'dbn_beat_pred.txt'), dbn_beat_pred[:, np.newaxis])
            beat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_beat_pred, beat_gt)
            
            accuracy = [f'fmeasure\t{beat_score_DBN.fmeasure}\n', \
                        f'cmlt\t{beat_score_DBN.cmlt}\n', \
                        f'amlt\t{beat_score_DBN.amlt}\n']
            with open(os.path.join(song_dir, 'accuracy.txt'), 'w') as f:
                f.writelines(accuracy)

            beat_DBN_meter.update(f'{dataset}-fmeasure', beat_score_DBN.fmeasure)
            beat_DBN_meter.update(f'{dataset}-cmlt', beat_score_DBN.cmlt)
            beat_DBN_meter.update(f'{dataset}-amlt', beat_score_DBN.amlt)


            combined_act = np.concatenate((np.maximum(beat_pred - downbeat_pred, np.zeros(beat_pred.shape))[:, np.newaxis], downbeat_pred[:, np.newaxis]), axis=-1)   #(T, 2)
            #print(combined_act.shape)
            dbn_downbeat_pred = downbeat_tracker(combined_act)
            dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]
            np.savetxt(os.path.join(song_dir, 'dbn_downbeat_pred.txt'), dbn_downbeat_pred[:, np.newaxis])

            #downbeat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_downbeat_pred, downbeat)
            #downbeat_DBN_meter.update(f'{dataset}-fmeasure', downbeat_score_DBN.fmeasure)
            #downbeat_DBN_meter.update(f'{dataset}-cmlt', downbeat_score_DBN.cmlt)
            #downbeat_DBN_meter.update(f'{dataset}-amlt', downbeat_score_DBN.amlt)

    print('DBN beat detection')
    for key in beat_DBN_meter.avg.keys():
        print('\t', key, beat_DBN_meter.avg[key])

    print('DBN downbeat detection')
    for key in downbeat_DBN_meter.avg.keys():
        print('\t', key, downbeat_DBN_meter.avg[key])


def inference_dbn():
    beat_DBN_meter = AverageMeter()
    downbeat_DBN_meter = AverageMeter()

    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=100, 
                                                                    observation_lambda=6, 
                                                                    num_tempi=None, 
                                                                    threshold=0.2)

    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=100, 
                                                                    observation_lambda=6, 
                                                                    num_tempi=None, 
                                                                    threshold=0.2)

    print('loading activations ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'inference_pred.pkl'), 'rb') as f:
        activations = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gt.pkl'), 'rb') as f:
        beat_gt = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'down_gt.pkl'), 'rb') as f:
        downbeat_gt = pickle.load(f)
    
    for dataset_key in activations:
        print(f'inferencing on {dataset_key} ...')
        beat_error = 0
        downbeat_error = 0
        for i in tqdm(range(len(activations[dataset_key]))):
            pred = activations[dataset_key][i]
            #print(pred.shape)
            beat = beat_gt[dataset_key][i]
            downbeat = downbeat_gt[dataset_key][i]

            try:
                dbn_beat_pred = beat_tracker(pred[0])
                beat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_beat_pred, beat)
                beat_DBN_meter.update(f'{dataset_key}-fmeasure', beat_score_DBN.fmeasure)
                beat_DBN_meter.update(f'{dataset_key}-cmlt', beat_score_DBN.cmlt)
                beat_DBN_meter.update(f'{dataset_key}-amlt', beat_score_DBN.amlt)
                
            except Exception as e:
                #print(f'beat inference encounter exception {e}')
                beat_error += 1


            try:
                combined_act = np.concatenate((np.maximum(pred[0] - pred[1], np.zeros(pred[0].shape))[:, np.newaxis], pred[1][:, np.newaxis]), axis=-1)   #(T, 2)
                #print(combined_act.shape)
                dbn_downbeat_pred = downbeat_tracker(combined_act)
                dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

                downbeat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_downbeat_pred, downbeat)
                downbeat_DBN_meter.update(f'{dataset_key}-fmeasure', downbeat_score_DBN.fmeasure)
                downbeat_DBN_meter.update(f'{dataset_key}-cmlt', downbeat_score_DBN.cmlt)
                downbeat_DBN_meter.update(f'{dataset_key}-amlt', downbeat_score_DBN.amlt)
            except Exception as e:
                #print(f'downbeat inference encounter exception {e}')
                downbeat_error += 1
        print(f'beat error: {beat_error}; downbeat error: {downbeat_error}')

    print('DBN beat detection')
    for key in beat_DBN_meter.avg.keys():
        print('\t', key, beat_DBN_meter.avg[key])

    print('DBN downbeat detection')
    for key in downbeat_DBN_meter.avg.keys():
        print('\t', key, downbeat_DBN_meter.avg[key])



def infer_gtzan_activation():
    #占坑
    model = Demixed_DilatedTransformerModel().to(DEVICE)

    dataset = audioDataset(data_to_load=['gtzan'],
                            test_only_data = ['gtzan'],
                            data_path = DATASET_PATH, 
                            annotation_path = ANNOTATION_PATH,
                            fps = FPS,
                            sample_size = None,
                            downsample_size=1,
                            hop_size = None,
                            num_folds = NUM_FOLDS)

    inference_pred = {}
    beat_gt = {}
    downbeat_gt = {}

    for FOLD in range(NUM_FOLDS):
        if not FOLD in inference_pred:
            inference_pred[FOLD] = {}
        print(f'\nFold {FOLD}')
        train_set, val_set, test_set = dataset.get_fold(fold=FOLD)
        #loader = DataLoader(val_set, batch_size=1, shuffle=False)
        loader = DataLoader(test_set, batch_size=1, shuffle=False)

        model = Demixed_DilatedTransformerModel(attn_len=ATTN_LEN,
                                                    instr=INSTR,
                                                    ntoken=NTOKEN, 
                                                    dmodel=DMODEL, 
                                                    nhead=NHEAD, 
                                                    d_hid=DHID, 
                                                    nlayers=NLAYER, 
                                                    norm_first=NORM_FIRST
                                                    )
        #model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'Fold_{FOLD}', 'model', 'trf_param_012.pt'), map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(PARAM_PATH[FOLD], map_location=torch.device('cpu'))['state_dict'])
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            for idx, (dataset_key, data, beat, downbeat, tempo, root) in tqdm(enumerate(loader), total=len(loader)):
                #data
                data = data.float().to(DEVICE)
                pred, _ = model(data)
                beat_pred = torch.sigmoid(pred[0, :, 0]).detach().cpu().numpy()
                downbeat_pred = torch.sigmoid(pred[0, :, 1]).detach().cpu().numpy()

                beat = torch.nonzero(beat[0]>.5)[:, 0].detach().numpy() / (FPS)
                downbeat = torch.nonzero(downbeat[0]>.5)[:, 0].detach().numpy() / (FPS)

                dataset_key = dataset_key[0]
                if not dataset_key in inference_pred[FOLD]:
                    inference_pred[FOLD][dataset_key] = []
                    beat_gt[dataset_key] = []
                    downbeat_gt[dataset_key] = []
                inference_pred[FOLD][dataset_key].append(np.stack((beat_pred, downbeat_pred), axis=0))
                beat_gt[dataset_key].append(beat)
                downbeat_gt[dataset_key].append(downbeat)


    print('saving prediction ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'inference_gtzan_pred.pkl'), 'wb') as f:
        pickle.dump( inference_pred, f)
    print('saving gt ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gtzan_gt.pkl'), 'wb') as f:
        pickle.dump(beat_gt, f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'down_gtzan_gt.pkl'), 'wb') as f:
        pickle.dump(downbeat_gt, f)

def inference_gtzan_dbn():
    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=100, 
                                                                    observation_lambda=6, 
                                                                    num_tempi=None, 
                                                                    threshold=0.2)

    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=100, 
                                                                    observation_lambda=6, 
                                                                    num_tempi=None, 
                                                                    threshold=0.2)

    print('loading activations ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'inference_gtzan_pred.pkl'), 'rb') as f:
        activations = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gtzan_gt.pkl'), 'rb') as f:
        beat_gt = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'down_gtzan_gt.pkl'), 'rb') as f:
        downbeat_gt = pickle.load(f)
    
    dataset_key ='gtzan'
    for FOLD in range(NUM_FOLDS):
        print(f'inferencing FOLD {FOLD} on {dataset_key} ...')
        beat_DBN_meter = AverageMeter()
        downbeat_DBN_meter = AverageMeter()
        beat_error = 0
        downbeat_error = 0
        for i in tqdm(range(len(activations[FOLD][dataset_key]))):
            pred = activations[FOLD][dataset_key][i]
            #print(pred.shape)
            beat = beat_gt[dataset_key][i]
            downbeat = downbeat_gt[dataset_key][i]

            try:
                dbn_beat_pred = beat_tracker(pred[0])
                beat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_beat_pred, beat)
                beat_DBN_meter.update(f'{dataset_key}-fmeasure', beat_score_DBN.fmeasure)
                beat_DBN_meter.update(f'{dataset_key}-cmlt', beat_score_DBN.cmlt)
                beat_DBN_meter.update(f'{dataset_key}-amlt', beat_score_DBN.amlt)
            except Exception as e:
                #print(f'beat inference encounter exception {e}')
                beat_error += 1


            try:
                combined_act = np.concatenate((np.maximum(pred[0] - pred[1], np.zeros(pred[0].shape))[:, np.newaxis], pred[1][:, np.newaxis]), axis=-1)   #(T, 2)
                #print(combined_act.shape)
                dbn_downbeat_pred = downbeat_tracker(combined_act)
                dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

                downbeat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_downbeat_pred, downbeat)
                downbeat_DBN_meter.update(f'{dataset_key}-fmeasure', downbeat_score_DBN.fmeasure)
                downbeat_DBN_meter.update(f'{dataset_key}-cmlt', downbeat_score_DBN.cmlt)
                downbeat_DBN_meter.update(f'{dataset_key}-amlt', downbeat_score_DBN.amlt)
            except Exception as e:
                #print(f'downbeat inference encounter exception {e}')
                downbeat_error += 1
        print(f'beat error: {beat_error}; downbeat error: {downbeat_error}')

        print('DBN beat detection')
        for key in beat_DBN_meter.avg.keys():
            print('\t', key, beat_DBN_meter.avg[key])

        print('DBN downbeat detection')
        for key in downbeat_DBN_meter.avg.keys():
            print('\t', key, downbeat_DBN_meter.avg[key])


def tuning_dbn():
    print('loading activations ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'inference_pred.pkl'), 'rb') as f:
        activations = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gt.pkl'), 'rb') as f:
        beat_gt = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'down_gt.pkl'), 'rb') as f:
        downbeat_gt = pickle.load(f)

    beat_error_record = 0
    downbeat_error_record = 0

    beat_record = [0, []]
    downbeat_record = [0, []]

    transition_lambda_set= [50, 100, 150]
    observation_lambda_set = [4, 8, 12]
    threshold_set = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
    num_tempi_set = [None, 20, 40]

    for threshold in threshold_set:
        for transition_lambda in transition_lambda_set:
            for observation_lambda in observation_lambda_set:
                for num_tempi in num_tempi_set:

                    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                    transition_lambda=transition_lambda, 
                                                                    observation_lambda=observation_lambda, 
                                                                    num_tempi=num_tempi, 
                                                                    threshold=threshold)

                    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, 
                                                                                    transition_lambda=transition_lambda, 
                                                                                    observation_lambda=observation_lambda, 
                                                                                    num_tempi=num_tempi, 
                                                                                    threshold=threshold)

                    beat_DBN_meter = AverageMeter()
                    downbeat_DBN_meter = AverageMeter()

                    beat_error = 0
                    downbeat_error = 0
                  
                    for dataset_key in activations:
                        print(f'inferencing on {dataset_key} ...')
                        
                        for i in tqdm(range(len(activations[dataset_key]))):
                            pred = activations[dataset_key][i]
                            beat = beat_gt[dataset_key][i]
                            downbeat = downbeat_gt[dataset_key][i]

                            try:
                                dbn_beat_pred = beat_tracker(pred[0])
                                beat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_beat_pred, beat)
                                beat_DBN_meter.update(f'{dataset_key}-fmeasure', beat_score_DBN.fmeasure)
                                beat_DBN_meter.update(f'{dataset_key}-cmlt', beat_score_DBN.cmlt)
                                beat_DBN_meter.update(f'{dataset_key}-amlt', beat_score_DBN.amlt)
                            except Exception as e:
                                beat_error += 1


                            try:
                                combined_act = np.concatenate((np.maximum(pred[0] - pred[1], np.zeros(pred[0].shape))[:, np.newaxis], pred[1][:, np.newaxis]), axis=-1)   #(T, 2)
                                dbn_downbeat_pred = downbeat_tracker(combined_act)
                                dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

                                downbeat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_downbeat_pred, downbeat)
                                downbeat_DBN_meter.update(f'{dataset_key}-fmeasure', downbeat_score_DBN.fmeasure)
                                downbeat_DBN_meter.update(f'{dataset_key}-cmlt', downbeat_score_DBN.cmlt)
                                downbeat_DBN_meter.update(f'{dataset_key}-amlt', downbeat_score_DBN.amlt)
                            except Exception as e:
                                downbeat_error += 1

                    beat_score = 0
                    for key in beat_DBN_meter.avg.keys():
                        beat_score += beat_DBN_meter.avg[key]
                    if beat_score > beat_record[0]:
                        beat_record[0] = beat_score
                        beat_record[1] = [threshold, transition_lambda, observation_lambda, num_tempi]


                    downbeat_score = 0
                    for key in downbeat_DBN_meter.avg.keys():
                        if key == 'smc':
                            continue
                        downbeat_score += downbeat_DBN_meter.avg[key]
                    if downbeat_score > downbeat_record[0]:
                        downbeat_record[0] = downbeat_score
                        downbeat_record[1] = [threshold, transition_lambda, observation_lambda, num_tempi]

                    if beat_error > beat_error_record:
                        beat_error_record = beat_error
                    if downbeat_error > downbeat_error_record:
                        downbeat_error_record = downbeat_error

                    print(beat_score, beat_error, beat_record, beat_error_record)
                    print(downbeat_score, downbeat_error, downbeat_record, downbeat_error_record)


    
if __name__ == '__main__':
    infer_activation()
    inference_dbn()
    #demo_inference_dbn()
    infer_gtzan_activation()
    inference_gtzan_dbn()
    #tuning_dbn()

    

            