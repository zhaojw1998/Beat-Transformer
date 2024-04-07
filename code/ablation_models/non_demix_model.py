import torch 
from torch import nn 
from DilatedTransformerLayer import DilatedTransformerLayer


class DilatedTransformerModel(nn.Module):
    def __init__(self, attn_len=5, ntoken=2, dmodel=128, nhead=2, d_hid=512, nlayers=9, norm_first=True, dropout=.1):
        super(DilatedTransformerModel, self).__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.attn_len = attn_len
        self.head_dim = dmodel // nhead
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"

        #self.Er = nn.Parameter(torch.randn(nlayers, nhead, self.head_dim, attn_len))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 3), stride=1, padding=(2, 0))#126
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=(1, 0))#79
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#26
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#31
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#15
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#5
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 6), stride=1, padding=(1, 0))#5
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 3), stride=1, padding=(1, 0))#3
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#1
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.Transformer_layers = nn.ModuleDict({})
        for idx in range(nlayers):
            self.Transformer_layers[f'Transformer_layer_{idx}'] = DilatedTransformerLayer(dmodel, nhead, d_hid, dropout, Er_provided=False, attn_len=attn_len, norm_first=norm_first)

        self.out_linear = nn.Linear(dmodel, ntoken)

        self.dropout_t = nn.Dropout(p=.5)
        self.out_linear_t = nn.Linear(dmodel, 300)
        

    def forward(self, x):
        #x: (batch, time, dmodel), FloatTensor
        x = x.unsqueeze(1)  #(batch, channel, time, dmodel)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch, channel, time, 1)
        x = x.transpose(1, 3).squeeze(1).contiguous()    #(batch, time, channel=dmodel)
        
        batch, time, dmodel = x.shape
        t = []
        for layer in range(self.nlayers):
            x, skip = self.Transformer_layers[f'Transformer_layer_{layer}'](x, layer=layer)
            t.append(skip)

        x = torch.relu(x)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t

    def inference(self, x):
        #x: (batch, time, dmodel), FloatTensor
        x = x.unsqueeze(1)  #(batch, channel, time, dmodel)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch, channel, time, 1)
        x = x.transpose(1, 3).squeeze(1).contiguous()    #(batch, time, channel=dmodel)
        
        batch, time, dmodel = x.shape
        t = []
        attn = [torch.eye(time, device=x.device).repeat(batch, self.nhead, 1, 1)]
        for layer in range(self.nlayers):
            x, skip, layer_attn = self.Transformer_layers[f'Transformer_layer_{layer}'].inference(x, layer=layer)
            t.append(skip)
            attn.append(torch.matmul(attn[-1], layer_attn.transpose(-2, -1)))


        x = torch.relu(x)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t, attn


if __name__ == '__main__':
    from non_demix_spectrogram_dataset import audioDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    import madmom
    from utils import AverageMeter
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
    DHID=512
    NLAYER=9
    DROPOUT=.1

    DEVICE=f'cuda:{0}'
    TRAIN_BATCH_SIZE = 1

    DATASET_PATH = './data/demix_spectrogram_data.npz'
    ANNOTATION_PATH = './data/full_beat_annotation.npz'
    DATA_TO_LOAD = ['gtzan']
    TEST_ONLY = ['gtzan']

    model = DilatedTransformerModel(attn_len=ATTN_LEN,
                                    ntoken=NTOKEN, 
                                    dmodel=DMODEL, 
                                    nhead=NHEAD, 
                                    d_hid=DHID, 
                                    nlayers=NLAYER, 
                                    norm_first=NORM_FIRST,
                                    dropout=DROPOUT
                                    )
    model.load_state_dict(torch.load("/mnt/c/Users/zhaoj/Desktop/trf_param_018.pt", map_location=torch.device(DEVICE)))
    model.to(DEVICE)


    dataset = audioDataset(data_to_load=DATA_TO_LOAD,
                        test_only_data = TEST_ONLY,
                        data_path = DATASET_PATH, 
                        annotation_path = ANNOTATION_PATH,
                        fps = FPS,
                        sample_size = SAMPLE_SIZE,
                        num_folds = 1)
    _, _, test_set = dataset.get_fold(fold=0)
    #loader = DataLoader(val_set, batch_size=1, shuffle=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)


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

    activations = {}
    beat_gt = {}
    downbeat_gt = {}

    count = 0
    with torch.no_grad():
        for idx, (dataset_key, data, beat, downbeat, tempo, root) in tqdm(enumerate(loader), total=len(loader)):
            #data
            data = data.float().to(DEVICE)
            print(data.shape)
            pred, _ = model(data)
            beat_pred = torch.sigmoid(pred[0, :, 0]).detach().cpu().numpy()
            downbeat_pred = torch.sigmoid(pred[0, :, 1]).detach().cpu().numpy()

            beat = torch.nonzero(beat[0]>.5)[:, 0].detach().numpy() / (FPS)
            downbeat = torch.nonzero(downbeat[0]>.5)[:, 0].detach().numpy() / (FPS)

            dataset_key = dataset_key[0]
            root = root[0]
            if not dataset_key in activations:
                activations[dataset_key] = []
                beat_gt[dataset_key] = []
                downbeat_gt[dataset_key] = []
            activations[dataset_key].append(np.stack((beat_pred, downbeat_pred), axis=0))
            beat_gt[dataset_key].append(beat)
            downbeat_gt[dataset_key].append(downbeat)

            #count += 1
            #if count == 50:
            #    break

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