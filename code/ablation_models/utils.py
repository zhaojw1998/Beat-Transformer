import numpy as np
import json, sys, os
from torch import nn
import torch
from torch.distributions import kl_divergence, Normal
from torch.optim.lr_scheduler import ExponentialLR
import random
import madmom

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, key, val, n=1):
        if not key in self.val:
            self.val[key] = val
            self.sum[key] = val * n
            self.count[key] = n
            self.avg[key] = self.sum[key] / self.count[key]
        else:
            self.val[key] = val
            self.sum[key] += val * n
            self.count[key] += n
            self.avg[key] = self.sum[key] / self.count[key]

def binary_accuracy(beat_pred, beat_gt):
    #beat: (B, T)
    weight = (1 - torch.as_tensor(beat_gt == -1, dtype=torch.int32))
    beat_pred = torch.as_tensor((torch.sigmoid(beat_pred) >= 0.5), dtype=torch.int32)
    beat_gt = torch.as_tensor((beat_gt > 0.6), dtype=torch.int32)
    positives = torch.as_tensor((beat_pred == beat_gt), dtype=torch.int32)
    positives = positives * weight
    binary_accuracy = positives.sum() / (weight.sum() + 1e-4)
    return binary_accuracy

def beat_accuracy(beat_pred, beat_gt, fps):
    #beat_pred: (B, L), estimation result
    weight = (1 - torch.as_tensor(beat_gt == -1, dtype=torch.int32))
    beat_pred = torch.sigmoid(beat_pred) * weight
    beat_pred = torch.as_tensor((beat_pred - 0.5) > 0, dtype=torch.int32).detach().cpu().numpy()
    #beat_pred = (beat_pred / fps)
    beat_gt = torch.as_tensor((beat_gt - 0.5) > 0, dtype=torch.int32).detach().cpu().numpy()
    #beat_gt = (beat_gt / fps)
    #print(beat_gt)
    batch_score = []
    for idx in range(beat_pred.shape[0]):
        #if (beat_gt[idx] == 0).all():
        #    continue
        if np.sum(beat_gt[idx]) < 2:
            continue
        beat_pred_batch = np.nonzero(beat_pred[idx])[0] / fps
        beat_gt_batch = np.nonzero(beat_gt[idx])[0] / fps
        #print(beat_gt_batch)
        score = madmom.evaluation.beats.BeatEvaluation(beat_pred_batch, beat_gt_batch)
        batch_score.append(score)
    batch_score = madmom.evaluation.beats.BeatMeanEvaluation(batch_score)
    return {"fmeasure": batch_score.fmeasure, \
            #"cemgil": batch_score.cemgil, \
            #"cmlc": batch_score.cmlc, \
            "cmlt": batch_score.cmlt, \
            #"amlc": batch_score.amlc, \
            "amlt": batch_score.amlt}
    

def infer_beat_with_DBN(beat_pred, beat_gt, dbn_model, fps):
    #beat_pred: (B, L), estimation result
    weight = (1 - torch.as_tensor(beat_gt == -1, dtype=torch.int32))
    beat_pred = (torch.sigmoid(beat_pred) * weight).detach().cpu().numpy()
    #beat_pred = (beat_pred / fps)
    beat_gt = torch.as_tensor((beat_gt - 0.5) > 0, dtype=torch.int32).detach().cpu().numpy()
    batch_score = []
    for idx in range(beat_pred.shape[0]):
        #if (beat_gt[idx] == 0).all():
        #    continue
        if np.sum(beat_gt[idx]) < 2:
            continue
        try:
            beat_pred_batch = dbn_model(beat_pred[idx])
        except:
            return {"fmeasure": 0, "cmlt": 0, "amlt": 0}
        beat_gt_batch = np.nonzero(beat_gt[idx])[0] / fps
        score = madmom.evaluation.beats.BeatEvaluation(beat_pred_batch, beat_gt_batch)
        batch_score.append(score)
    batch_score = madmom.evaluation.beats.BeatMeanEvaluation(batch_score)
    return {"fmeasure": batch_score.fmeasure if not np.isnan(batch_score.fmeasure) else 0, \
            #"cemgil": batch_score.cemgil, \
            #"cmlc": batch_score.cmlc, \
            "cmlt": batch_score.cmlt if not np.isnan(batch_score.cmlt) else 0, \
            #"amlc": batch_score.amlc, \
            "amlt": batch_score.amlt if not np.isnan(batch_score.amlt) else 0}


def infer_downbeat_with_DBN(beat_pred, downbeat_pred, downbeat_gt, dbn_model, fps):
    #beat_pred: (B, L), estimation result
    beat_pred = torch.sigmoid(beat_pred).detach().cpu()
    downbeat_pred = torch.sigmoid(downbeat_pred).detach().cpu()
    combined_act = torch.cat((torch.maximum(beat_pred - downbeat_pred, torch.zeros(beat_pred.shape)).unsqueeze(-1), downbeat_pred.unsqueeze(-1)), dim=-1)
    #beat_pred = (beat_pred / fps)
    weight = (1 - torch.as_tensor(downbeat_gt == -1, dtype=torch.int32)).unsqueeze(-1).detach().cpu()
    combined_act = (combined_act * weight).numpy()

    beat_gt = torch.as_tensor((downbeat_gt - 0.5) > 0, dtype=torch.int32).detach().cpu().numpy()
    batch_score = []
    for idx in range(beat_pred.shape[0]):
        #if (beat_gt[idx] == 0).all():
        #    continue
        if np.sum(beat_gt[idx]) < 2:
            continue
        try:
            beat_pred_batch = dbn_model(combined_act[idx])
            beat_pred_batch = beat_pred_batch[beat_pred_batch[:, 1]==1][:, 0]
        except:
            return {"fmeasure": 0, "cmlt": 0, "amlt": 0}
        beat_gt_batch = np.nonzero(beat_gt[idx])[0] / fps
        score = madmom.evaluation.beats.BeatEvaluation(beat_pred_batch, beat_gt_batch)
        batch_score.append(score)
    batch_score = madmom.evaluation.beats.BeatMeanEvaluation(batch_score)
    return {"fmeasure": batch_score.fmeasure if not np.isnan(batch_score.fmeasure) else 0, \
            #"cemgil": batch_score.cemgil, \
            #"cmlc": batch_score.cmlc, \
            "cmlt": batch_score.cmlt if not np.isnan(batch_score.cmlt) else 0, \
            #"amlc": batch_score.amlc, \
            "amlt": batch_score.amlt if not np.isnan(batch_score.amlt) else 0}



def load_dataset_path(fn='model_config.json'):
    with open(fn) as f:
        paths = json.load(f)['dataset_path']

    train_val_path = paths['hpc_data_path']
    return train_val_path

def load_params_dict(key, fn='model_config.json'):
    with open(fn) as f:
        dict = json.load(f)[key]
    return dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def standard_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function_vae(recon_pitch, pitch, dist, pitch_criterion, normal,
                  weights=(1, .1)):
    # bs = dist.mean.size(0)
    #print(recon_pitch.shape, pitch.shape, recon_rhythm.shape, rhythm.shape)
    pitch_loss = pitch_criterion(recon_pitch, pitch)
    kl_div = kl_divergence(dist, normal).mean()
    loss = weights[0] * pitch_loss + weights[1] * kl_div
    return loss, pitch_loss, kl_div

def loss_function_discr(recon_mask, mask_gt, dist, mask_criterion, normal,
                  weights=(1, .1)):
    # bs = dist.mean.size(0)
    #print(recon_pitch.shape, pitch.shape, recon_rhythm.shape, rhythm.shape)
    mask_loss = mask_criterion(recon_mask, mask_gt)
    kl_div = kl_divergence(dist, normal).mean()
    loss = weights[0] * mask_loss + weights[1] * kl_div
    return loss, mask_loss, kl_div

def get_complement(mask_gt):
    #mask_gt: (BT, 128)
    complement = torch.zeros(mask_gt.shape).long().cuda()
    for i in range(mask_gt.shape[0]):
        if random.random() < 0.5:
            low = max(mask_gt[i].max(0)[-1].item() - 5, 0)
            high = min(mask_gt[i].max(0)[-1].item() + 6, 127)
        else:
            low = max(mask_gt[i].max(0)[-1].item() - 6, 0)
            high = min(mask_gt[i].max(0)[-1].item() + 5, 127)
        #print(low, high)
        complement[i, low: high+1] = 1.
    return complement - mask_gt


# Useful function for how long epochs take
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def scheduled_sampling(i, high=0.7, low=0.05):
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y



def piano_roll_to_target(pr):
    #  pr: (32, 128, 3), dtype=bool

    # Assume that "not (first_layer or second layer) = third_layer"
    pr[:, :, 1] = np.logical_not(np.logical_or(pr[:, :, 0], pr[:, :, 2]))
    # To int dtype can make addition work
    pr = pr.astype(int)
    # Initialize a matrix to store the duration of a note on the (32, 128) grid
    pr_matrix = np.zeros((32, 128))

    for i in range(31, -1, -1):
        # At each iteration
        # 1. Assure that the second layer accumulates the note duration
        # 2. collect the onset notes in time step i, and mark it on the matrix.

        # collect
        onset_idx = np.where(pr[i, :, 0] == 1)[0]
        pr_matrix[i, onset_idx] = pr[i, onset_idx, 1] + 1
        if i == 0:
            break
        # Accumulate
        # pr[i - 1, :, 1] += pr[i, :, 1]
        # pr[i - 1, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i - 1, :, 1] += pr[i, :, 1]

    return pr_matrix


def target_to_3dtarget(pr_mat, max_note_count=11, max_pitch=107, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87):
    """
    :param pr_mat: (32, 128) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((32, max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(32, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(int(pr_mat[t, p]) - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        cur_idx[t] += 1
    pr_mat3d[np.arange(0, 32), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def get_low_high_dur_count(pr_mat):
    # pr_mat (32, 128)
    # return the maximum duration
    # return the pitch range
    # return the number of notes at each column

    pitch_range = np.where(pr_mat != 0)[1]
    low_pitch = pitch_range.min()
    high_pitch = pitch_range.max()
    pitch_dur = pr_mat.max()
    num_notes = np.count_nonzero(pr_mat, axis=-1)
    return low_pitch, high_pitch, pitch_dur, num_notes
