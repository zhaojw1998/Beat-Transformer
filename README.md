# Beat Transformer
<a href="https://colab.research.google.com/drive/1IdrpMO1AivWmy-Bm8ktmMy14ED9jllux?usp=sharing)" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

Repository for paper: [Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention](https://arxiv.org/abs/2209.07140) in Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR 2022), Bengaluru, India.

Welcome to test our model on your own music at our [Google Colab](https://colab.research.google.com/drive/1IdrpMO1AivWmy-Bm8ktmMy14ED9jllux?usp=sharing).


## Code and File Directory

This repository is organized as follows:

```
root
│
└───checkpoint                          PyTorch model checkpoints
    │   ···
│   
└───code
    └───ablation_models                 ablation models
        │   ···                            
    │   DilatedTransformer.py           Beat Transformer model
    │   DilatedTransformerLayer.py      Dilated Self-Attention
    │   spectrogram_dataset.py          data loader
    │   train.py                        training script
    │   ...                             code for other utilities
│   
└───data
    └───audio_lists                     Order info of pieces in each dataset
        │   ···                     
    │   demix_spectrogram_data.npz      demixed spectrogram data (33GB, to be downloaded)
    │   full_beat_annotation.npz        beat/downbeat annotation
│   
└───preprocessing                       code for data pre-processing
    │   ···
│   
└───save                                training log and more
    │   ···
```


## How to run
* To quickly reproduce the accuracy reported in our paper, simply run `./code/eight_fold_test.py`.
* To quickly test our model with your own music, welcome to our [Google Colab](https://colab.research.google.com/drive/1IdrpMO1AivWmy-Bm8ktmMy14ED9jllux?usp=sharing).
* If you wish to train our model from scratch, first download our [processed dataset](https://drive.google.com/file/d/1LamSAEY5QsnY57cF6qH_0niesGGKkHtI/view?usp=sharing) (33GB in total, including demixed spectrogram data of Ballroom, Hainsworth, Carnetic, Harmonix, SMC, and GTZAN). 
* Executing `./code/train.sh` will train our model in 8-fold cross validation. If you wish to train one single fold, you can run `./code/train.py` after specifying `DEBUG_MODE`, `FOLD`, and `GPU`. When `DEBUG_MODE=1`, it will load a small portion of data to quickly run through with a smaller bach size.
* We also release out ablation model architectures in `./code/ablation_models`. We release our data processing scripts in `./preprocessing/demixing.py`, where we call [Spleeter](https://github.com/deezer/spleeter) to demix each piece and save the demixed spectrogram.

## Audio Data
We use a total of 7 datasets for model training and testing. If you wish to acquire the audio data, you can follow the following guidelines:
* Ballroom Dataset (audio) is available [here](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html). There are 13 duplicated pieces and I discarded them in my experiments. For more information, see [here](https://github.com/CPJKU/BallroomAnnotations/blob/master/README.md).

* Hainsworth Dataset (audio) is no longer accessible via the original link. Since Hainsworth is a well-known public dataset, I guess it's okay to share my copy. You can download Hainsworth [here](https://drive.google.com/file/d/1ctMDHAoeTBG5LSbtQIQBIv4vTI0oB0u1/view).

* GTZAN Dataset (audio) is available on [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). You need a registered Kaggle account to download it.

* SMC Dataset (audio) is available [here](https://joserzapata.github.io/publication/selective-sampling-beat-tracking/).

* Carnetic Dataset (audio) is on [Zenodo](https://zenodo.org/record/1264394). You can download it by request.

* Harmonix Dataset (mel-spectrogram) is available [here](https://github.com/urinieto/harmonixset). I used the Griffin-Lim algorithm in Librosa to convert mel-spectrogram to audio, which (however) is lossful. My conversion code is [here](https://github.com/zhaojw1998/Beat-Transformer/blob/main/preprocessing/harmonix_mel2wav.py).

* RWC POP (audio) seems NOT royalty-free so I'm afraid I cannot share the audio. For more info about this dataset, you can go to its [official webpage](https://staff.aist.go.jp/m.goto/RWC-MDB/).

For the beat/downbeat annotation of Ballroom, GTZAN, SMC, and Hainsworth, I used the annotation released by Sebastian Böck [here](https://github.com/superbock/ISMIR2019).



## Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

Nov. 24, 2022
