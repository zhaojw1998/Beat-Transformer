# Beat Transformer
<a href="https://colab.research.google.com/drive/1IdrpMO1AivWmy-Bm8ktmMy14ED9jllux?usp=sharing)" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

Paper: [Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention](https://arxiv.org/abs/2209.07140)

Welcome to test our model on your own music through our [Google Colab](https://colab.research.google.com/drive/1IdrpMO1AivWmy-Bm8ktmMy14ED9jllux?usp=sharing).


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


## Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

Nov. 24, 2022
