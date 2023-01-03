# Graph-Matching-Attention

This code provides a pytorch implementation of our Graph Matching Attention method for Visual Question Answering as described in [Bilateral Cross-Modality Graph Matching Attention for Feature Fusion in Visual Question Answering](https://arxiv.org/pdf/2112.07270.pdf)

TODO:
1. The note of VQA & GQA dataset process
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

## Data
To download and unzip the required datasets, change to the data folder and run
```
cd VQAdata_process; python tools/download_data.py
```

To preprocess the image data and text data the following commands can be executed respectively. 

``` 
sh build_graph.sh
```

## Training
To train a model on the train set with our default parameters run
```
python3 -u train.py --train --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model
```

and to train a model on the train and validation set for evaluation on the test set run
```
python3 -u train.py --trainval --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model
```

Models can be validated via
```
python3 -u train.py --eval --model_path ./trained_model/model.pth.tar --data_type VQA --data_dir ../data/VQA --bsize 256
```

and a json of results from the test set can be produced with
```
python3 -u train.py --test --model_path ./trained_model/model.pth.tar --data_type VQA --data_dir ../data/VQA --bsize 256
```

## Acknowledgements
Our code is based on this implementation of [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](https://github.com/aimbrain/vqa-project)


