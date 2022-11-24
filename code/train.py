import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import time
import madmom
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from optimizer import Lookahead
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import AverageMeter, epoch_time, infer_beat_with_DBN, infer_downbeat_with_DBN
from spectrogram_dataset import audioDataset

from DilatedTransformer import Demixed_DilatedTransformerModel

import warnings
warnings.filterwarnings('ignore')


DEBUG_MODE = int(sys.argv[1])
FOLD = int(sys.argv[2])
GPU = int(sys.argv[3])
PROJECT_NAME = 'Beat_Transformer'

###############################################################################
# Load config
###############################################################################
#data
SAMPLE_SIZE = int(44100 / 1024 * 180)
INSTR =5
FPS = 44100 / 1024
NUM_FOLDS = 8
#model
NORM_FIRST=True
ATTN_LEN=5
NTOKEN=2
DMODEL=256
NHEAD=8
DHID=1024
NLAYER=9
DROPOUT=.1
#training
DEVICE=f'cuda:{GPU}'
TRAIN_BATCH_SIZE = 1
LEARNING_RATE = 1e-3
DECAY = 0.99995
N_EPOCH = 30
CLIP=.5
#directories
DATASET_PATH = './data/demix_spectrogram_data.npz'
ANNOTATION_PATH = './data/full_beat_annotation.npz'
DATA_TO_LOAD = ['ballroom', 'ballroom', 'gtzan', 'hainsworth', 'smc', 'harmonix', 'carnetic']
TEST_ONLY = ['gtzan']

SAVE_PATH = f'./save/train_log/{str(GPU).zfill(2)}_{PROJECT_NAME}'

if DEBUG_MODE:
    N_EPOCH = 1
    TRAIN_BATCH_SIZE = 1
    DECAY = 0.9995
    DATA_TO_LOAD = ['hainsworth']   #hainsworth, smc
    SAVE_PATH = os.path.join(SAVE_PATH, 'debug')

print(f'\nProject initialized: {PROJECT_NAME}\n', flush=True)


print(f'\nFold {FOLD}')
###############################################################################
# Initialize fold
###############################################################################
project_path = os.path.join(SAVE_PATH, f'Fold_{FOLD}')

MODEL_PATH = os.path.join(project_path, 'model') 
LOG_PATH = os.path.join(project_path, 'log')

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

loss_writer = SummaryWriter(os.path.join(LOG_PATH, 'loss'))
#beat_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_acc'))
beat_ll_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_likelihood'))
downbeat_ll_writer = SummaryWriter(os.path.join(LOG_PATH, 'downbeat_likelihood'))
beat_pr_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_precision'))
downbeat_pr_writer = SummaryWriter(os.path.join(LOG_PATH, 'downbeat_precision'))
beat_DBN_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_DBN_acc'))
#downbeat_writer = SummaryWriter(os.path.join(LOG_PATH, 'downbeat_acc'))
downbeat_DBN_writer = SummaryWriter(os.path.join(LOG_PATH, 'downbeat_DBN_acc'))


###############################################################################
# model parameter
###############################################################################
model = Demixed_DilatedTransformerModel(attn_len=ATTN_LEN,
                                                instr=INSTR,
                                                ntoken=NTOKEN, 
                                                dmodel=DMODEL, 
                                                nhead=NHEAD, 
                                                d_hid=DHID, 
                                                nlayers=NLAYER, 
                                                norm_first=NORM_FIRST,
                                                dropout=DROPOUT
                                                )

model.to(DEVICE)


###############################################################################
# load data
###############################################################################
dataset = audioDataset(data_to_load=DATA_TO_LOAD,
                        test_only_data = TEST_ONLY,
                        data_path = DATASET_PATH, 
                        annotation_path = ANNOTATION_PATH,
                        fps = FPS,
                        sample_size = SAMPLE_SIZE,
                        num_folds = NUM_FOLDS)
# Fold Splitting
train_set, val_set, test_set = dataset.get_fold(fold=FOLD)
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
#test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


###############################################################################
# Optimizer and Criterion
###############################################################################
optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
#scheduler = MinExponentialLR(optimizer, gamma=DECAY, minimum=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=2, threshold=1e-3, min_lr=1e-7)
loss_func = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.LongTensor([1, 1]).to(DEVICE))
loss_tempo = nn.BCELoss(reduction='none')

beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=10, threshold=0.05)
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=10)


###############################################################################
# Main
###############################################################################
def train(model, train_loader, optimizer, scheduler, loss_func, loss_tempo, clip, epoch, device):
    print('training ...', flush=True)
    num_batch = len(train_loader)
    loss_meter_b = AverageMeter()
    loss_meter_t = AverageMeter()
    beat_meter = AverageMeter()
    beat_DBN_meter = AverageMeter()
    downbeat_meter = AverageMeter()
    downbeat_DBN_meter = AverageMeter()
    nan_count = []
    for idx, (dataset_key, data, beat_gt, downbeat_gt, tempo_gt) in tqdm(enumerate(train_loader), total=num_batch):
        #try:
        #data
        data = data.float().to(device)
        #annotation
        beat_gt = beat_gt.to(device)
        downbeat_gt = downbeat_gt.to(device)
        gt = torch.cat([beat_gt.unsqueeze(-1), downbeat_gt.unsqueeze(-1)], dim=-1).float().to(device) #(batch, T', 2)
        tempo_gt = tempo_gt.to(device)
        
        optimizer.zero_grad()
        pred, tempo = model(data)
        #print(pred.shape, gt.shape)
        valid_gt = gt.clone()
        valid_gt[gt == -1] = 0
        loss = loss_func(pred, valid_gt)
        weight = (1 - torch.as_tensor(gt == -1, dtype=torch.int32)).to(device)
        loss = (weight * loss).mean(dim=(0, 1)).sum()

        valid_tempo_gt = tempo_gt.clone()
        valid_tempo_gt[tempo_gt == -1] = 0
        loss_t = loss_tempo(torch.softmax(tempo, dim=-1), valid_tempo_gt)
        weight = (1 - torch.as_tensor(tempo_gt == -1, dtype=torch.int32)).to(device)
        loss_t = (weight * loss_t).mean()
        #except RuntimeError:
        #    continue
        
        loss_meter_t.update('train/loss', loss_t.item())
        loss_meter_b.update('train/loss', loss.item())
        if ((dataset_key[0] == 'musicnet') and (-1 in tempo_gt)):
            loss =  loss * 0 #do not trust musicnet beat annotation if tempo is none

        #try:
        loss = loss + loss_t
        #except RuntimeError:
        #    continue
        if torch.isnan(loss):
            nan_count.append(str(dataset_key)+'\n')
            with open('./home/zhaojw/workspace/efficient_dilated_MultiSpec_Transformer/nancount.txt', 'w') as f:
                f.writelines(nan_count)
            continue

        #downbeat_loss = loss_func(downbeat_pred, downbeat_gt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        #scheduler.step()

        #binary_acc = binary_accuracy(pred[:, :, :2], gt[:, :, :2])
        #loss_meter.update('train/loss', loss.item())
        #loss_meter.update('train/binary_acc', binary_acc.item())

        #beat_acc = beat_accuracy(pred[:, :, 0], gt[:, :, 0], FPS/DS_RATIO)
        #for key in beat_acc:
        #    beat_meter.update('train/' + key, beat_acc[key])

        
        #downbeat_acc = beat_accuracy(pred[:, :, 1], gt[:, :, 1], FPS/DS_RATIO)
        #if not dataset_key[0] == 'smc':
        #    for key in downbeat_acc:
        #        downbeat_meter.update('train/' + key, downbeat_acc[key])


        if DEBUG_MODE:
            print('------------training------------', flush=True)
            print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, idx, num_batch), flush=True)
            print('train beat loss:', loss.item()-loss_t.item(), flush=True)
            print('train tempo loss:', loss_t.item(), flush=True)
            #print('train binary batch accuracy', binary_acc.item(), flush=True)
            #print('beat accuracy:', list(beat_acc.values()), flush=True)
            #print('downbeat accuracy:', list(downbeat_acc.values()), flush=True)

        loss_writer.add_scalar('train/loss_beat', loss_meter_b.avg['train/loss'], epoch * num_batch + idx)
        loss_writer.add_scalar('train/loss_tempo', loss_meter_t.avg['train/loss'], epoch * num_batch + idx)
        loss_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * num_batch + idx)

        #for key in beat_meter.avg.keys():
        #    if 'train' in key:    
        #        beat_writer.add_scalar(key, beat_meter.avg[key], epoch * num_batch + idx)
        #for key in beat_DBN_meter.avg.keys():
        #    if 'train' in key:    
        #        beat_DBN_writer.add_scalar(key, beat_DBN_meter.avg[key], epoch * num_batch + idx)
        #for key in downbeat_meter.avg.keys():
        #    if 'train' in key:    
        #        downbeat_writer.add_scalar(key, downbeat_meter.avg[key], epoch * num_batch + idx)
        #for key in downbeat_DBN_meter.avg.keys():
        #    if 'train' in key:    
        #        downbeat_DBN_writer.add_scalar(key, downbeat_DBN_meter.avg[key], epoch * num_batch + idx)
    return loss_meter_b, loss_meter_t, beat_meter, beat_DBN_meter, downbeat_meter, downbeat_DBN_meter


def evaluate(model, val_loader, loss_func, loss_tempo, epoch, device):
    print('validating ...', flush=True)
    num_batch = len(val_loader)
    loss_meter_b = AverageMeter()
    loss_meter_t = AverageMeter()
    beat_meter = AverageMeter()
    beat_DBN_meter = AverageMeter()
    downbeat_meter = AverageMeter()
    downbeat_DBN_meter = AverageMeter()
    with torch.no_grad():
        for idx, (dataset_key, data, beat_gt, downbeat_gt, tempo_gt) in tqdm(enumerate(val_loader), total=num_batch):
            #try:
            #data
            data = data.float().to(device)
            #annotation
            beat_gt = beat_gt.to(device)
            downbeat_gt = downbeat_gt.to(device)
            gt = torch.cat([beat_gt.unsqueeze(-1), downbeat_gt.unsqueeze(-1)], dim=-1).float().to(device) #(batch, T', 2)
            #tempo_gt = tempo_gt.reshape(-1).long().to(device)
            tempo_gt = tempo_gt.float().to(device)

            pred, tempo = model(data)
            
            valid_gt = gt.clone()
            valid_gt[gt == -1] = 0
            loss = loss_func(pred, valid_gt)
            weight = (1 - torch.as_tensor(gt == -1, dtype=torch.int32)).to(device)
            loss = (weight * loss).mean(dim=(0, 1)).sum()

            valid_tempo_gt = tempo_gt.clone()
            valid_tempo_gt[tempo_gt == -1] = 0
            loss_t = loss_tempo(torch.softmax(tempo, dim=-1), valid_tempo_gt)
            weight = (1 - torch.as_tensor(tempo_gt == -1, dtype=torch.int32)).to(device)
            loss_t = (weight * loss_t).mean()
            #except RuntimeError:
            #    continue
            
            if not dataset_key[0] == 'gtzan':
                loss_meter_b.update('val/loss', loss.item())
            else:
                loss_meter_b.update('val/loss_nontrain', loss.item())

            if not dataset_key[0] == 'gtzan':
                loss_meter_t.update('val/loss', loss_t.item())
            else:
                loss_meter_t.update('val/loss_nontrain', loss_t.item())
            
            #binary_acc = binary_accuracy(pred[:, :, :2], gt[:, :, :2])
            #if not dataset_key[0][0] == 'gtzan':
            #    loss_meter.update('val/loss', loss.item())
            #    loss_meter.update('val/binary_acc', binary_acc.item())
            #else:
            #    loss_meter.update('val/loss_nontrain', loss.item())
            #    loss_meter.update('val/binary_acc_nontrain', binary_acc.item())
            

            #try:
            #beat_acc = beat_accuracy(pred[:, :, 0], gt[:, :, 0], FPS/DS_RATIO)
            #for key in beat_acc:
            #    beat_meter.update(f'val-{dataset_key[0][0]}/{key}', beat_acc[key])

            beat_acc_DBN = infer_beat_with_DBN(pred[:, :, 0], beat_gt, beat_tracker, FPS)
            for key in beat_acc_DBN:
                beat_DBN_meter.update(f'val-{dataset_key[0]}/{key}', beat_acc_DBN[key])

            
            #downbeat_acc = beat_accuracy(pred[:, :, 1], gt[:, :, 1], FPS/DS_RATIO)
            #if not dataset_key[0][0] == 'smc':
            #    for key in downbeat_acc:
            #        downbeat_meter.update(f'val-{dataset_key[0][0]}/{key}', downbeat_acc[key])

            downbeat_DBN_acc = infer_downbeat_with_DBN(pred[:, :, 0], pred[:, :, 1], downbeat_gt, downbeat_tracker, FPS)
            if not dataset_key[0] == 'smc':
                for key in downbeat_DBN_acc:
                    downbeat_DBN_meter.update(f'val-{dataset_key[0]}/{key}', downbeat_DBN_acc[key])


            if DEBUG_MODE:
                print('------------validation------------', flush=True)
                print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, idx, num_batch), flush=True)
                print('val beat loss:', loss.item(), flush=True)
                print('train tempo loss:', loss_t.item(), flush=True)
                #print('val batch binary accuracy:', binary_acc.item(), flush=True)
                #print('beat accuracy:', list(beat_acc.values()), flush=True)
                print('beat accuracy with DBN:', list(beat_acc_DBN.values()), flush=True)
                #print('downbeat accuracy:', list(downbeat_acc.values()), flush=True)
                print('downbeat accuracy with DBN:', list(downbeat_DBN_acc.values()), flush=True)
            #except Exception as e:
            #    print(e)


        if not dataset_key[0] == 'gtzan':
            loss_writer.add_scalar('val/loss_beat', loss_meter_b.avg['val/loss'], epoch)
            loss_writer.add_scalar('val/loss_tempo', loss_meter_t.avg['val/loss'], epoch)
        else:
            loss_writer.add_scalar('val/loss_beat_nontrain', loss_meter_b.avg['val/loss_nontrain'], epoch)
            loss_writer.add_scalar('val/loss_tempo_nontrain', loss_meter_t.avg['val/loss_nontrain'], epoch)


        #for key in beat_meter.avg.keys():
        #    if 'val' in key:    
        #        beat_writer.add_scalar(key, beat_meter.avg[key], epoch)
        for key in beat_DBN_meter.avg.keys():
            if 'val' in key:    
                beat_DBN_writer.add_scalar(key, beat_DBN_meter.avg[key], epoch)
        #for key in downbeat_meter.avg.keys():
        #    if 'val' in key:    
        #        downbeat_writer.add_scalar(key, downbeat_meter.avg[key], epoch)
        for key in downbeat_DBN_meter.avg.keys():
            if 'val' in key:    
                downbeat_DBN_writer.add_scalar(key, downbeat_DBN_meter.avg[key], epoch)
    return loss_meter_b, loss_meter_t, beat_meter, beat_DBN_meter, downbeat_meter, downbeat_DBN_meter


for epoch in range(N_EPOCH):
    print(f'Start Epoch: {epoch + 1:02}', flush=True)
    start_time = time.time()

    model.train()
    _, _, _, _, _, _ = train(model, train_loader, optimizer, scheduler, loss_func, loss_tempo, CLIP, epoch, DEVICE)

    model.eval()
    #optimizer._backup_and_load_cache()
    loss_meter_b, loss_meter_t, beat_meter, beat_DBN_meter, downbeat_meter, downbeat_DBN_meter = evaluate(model, val_loader, loss_func, loss_tempo, epoch, DEVICE)
    #optimizer._clear_and_load_backup()

    scheduler.step(loss_meter_b.avg['val/loss'] + loss_meter_t.avg['val/loss'])

    #torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'trf_param_'+str(epoch).zfill(3)+'.pt'))


    torch.save({ 'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),  
    }, os.path.join(MODEL_PATH, 'trf_param_'+str(epoch).zfill(3)+'.pt'))

    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s', flush=True)

    print('val beat loss:', loss_meter_b.avg['val/loss'], flush=True)
    print('val tempo loss:', loss_meter_t.avg['val/loss'], flush=True)
    #print('beat accuracy:', [(key.split('/')[-1], beat_meter.avg[key]) for key in beat_meter.avg.keys() if 'val' in key], flush=True)
    print('beat accuracy with DBN:', [(key.split('/')[-1], beat_DBN_meter.avg[key]) for key in beat_DBN_meter.avg.keys() if 'val' in key], flush=True)
    #print('downbeat accuracy:', [(key.split('/')[-1], downbeat_meter.avg[key]) for key in downbeat_meter.avg.keys() if 'val' in key], flush=True)
    print('downbeat accuracy with DBN:', [(key.split('/')[-1], downbeat_DBN_meter.avg[key]) for key in downbeat_DBN_meter.avg.keys() if 'val' in key], flush=True)
    print('\n')
