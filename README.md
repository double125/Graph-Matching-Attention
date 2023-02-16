# Graph-Matching-Attention

This code provides a pytorch implementation of our Graph Matching Attention method for Visual Question Answering as described in [Bilateral Cross-Modality Graph Matching Attention for Feature Fusion in Visual Question Answering](https://arxiv.org/pdf/2112.07270.pdf)

TODO:
1. The GQA dataset process
2. Result Table (which can find in our paper)
3. Trained models release.
4. Other details.

## Model diagram
  This is the first version of the code of Graph Matching Attention.
### Pipeline of Graph Matching Attention
![Pipeline of Graph Matching Attention](https://github.com/double125/Graph-Matching-Attention/raw/master/figures/GMA%20Pipeline.png)

### Framework of Graph Matching Attention
![Framework of Graph Matching Attention](https://github.com/double125/Graph-Matching-Attention/raw/master/figures/GMA%20Framework.png)

### Modules of Graph Matching Attention
![Modules of Graph Matching Attention](https://github.com/double125/Graph-Matching-Attention/raw/master/figures/GMA%20Module.png)

## Getting Started
  * [pytorch (0.3.1) (with CUDA)](https://pytorch.org/)
  * [zarr (2.2.0)](https://github.com/zarr-developers/zarr)
  * [tdqm](https://github.com/tqdm/tqdm)
  * [spacy](https://spacy.io/usage/)
  * [stanfordcorenlp](https://www.jianshu.com/p/c93fb950c2b3)
  * pandas
  * h5py

## Data
To download and unzip the required datasets, change to the data folder and run
```
cd VQAdata_process; python tools/download_data.py
```

To preprocess the image data and text data the following commands can be executed respectively. 

``` 
sh build_graph.sh
```

### Pretrained features for VQA dataset
For ease-of-use, we use the [pretrained features](https://github.com/peteanderson80/bottom-up-attention#pretrained-features) available for the entire MSCOCO dataset. Features are stored in tsv (tab-separated-values) format that can be downloaded from the links below,

10 to 100 features per image (adaptive):
* [2014 Train/Val Image Features (120K / 23GB)](https://storage.googleapis.com/up-down-attention/trainval.zip)
* [2014 Testing Image Features (40K / 7.3GB)](https://storage.googleapis.com/up-down-attention/test2014.zip)
* [2015 Testing Image Features (80K / 15GB)](https://storage.googleapis.com/up-down-attention/test2015.zip)

36 features per image (fixed):
* [2014 Train/Val Image Features (120K / 25GB)](https://storage.googleapis.com/up-down-attention/trainval_36.zip)
* [2014 Testing Image Features (40K / 9GB)](https://storage.googleapis.com/up-down-attention/test2014_36.zip)
* [2015 Testing Image Features (80K / 17GB)](https://storage.googleapis.com/up-down-attention/test2015_36.zip)

### VG dataset
we use extra data from [Visual Genome](http://visualgenome.org/). The question and answer pairs can be downloaded from the links below,
* [visualgenome_qa.zip](https://drive.google.com/file/d/1QKe4TKiYnn4pk_48z_mLxF7xaf2VH8Ut/view?usp=sharing)
* [image_data.json.zip](https://drive.google.com/file/d/1SLvZ9GbnRMCM-MmJBBP4OGjuDeTMsEBF/view?usp=sharing)
* [question_answers.json.zip](https://drive.google.com/file/d/16RYCz58WxKf7INk3ai268fXIfjr5Czqy/view?usp=sharing)
* [imgids.zip](https://drive.google.com/file/d/1X7xcbKDZB6oSe_qLwrVAGn9Zz3rcjxo-/view?usp=sharing)

## GQA dataset
Download the GQA dataset from [https://cs.stanford.edu/people/dorarad/gqa/](https://cs.stanford.edu/people/dorarad/gqa/)

## Training
To train a model on the train set with our default parameters run
```
python3 -u train.py --train --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model
```

and to train a model on the train and validation set for evaluation on the test set run
```
python3 -u train.py --trainval --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model
```
## Evaluation
Models can be validated via
```
python3 -u train.py --eval --model_path ./trained_model/model.pth.tar --data_type VQA --data_dir ../data/VQA --bsize 256
```

and a json of results from the test set can be produced with
```
python3 -u train.py --test --model_path ./trained_model/model.pth.tar --data_type VQA --data_dir ../data/VQA --bsize 256
```
## Citation
We hope our paper, data and code can help in your research. If this is the case, please cite:
```
@ARTICLE{Cao2022GMA,
  author={Cao, Jianjian and Qin, Xiameng and Zhao, Sanyuan and Shen, Jianbing},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Bilateral Cross-Modality Graph Matching Attention for Feature Fusion in Visual Question Answering}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2021.3135655}}
```

## Acknowledgements
Our code is based on this implementation of [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](https://github.com/aimbrain/vqa-project)

## Contact Us
If you have any problem about this work, please feel free to reach us out at caojianjianbit@gmail.com.

