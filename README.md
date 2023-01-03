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


' cd data; python download_data.py '


## Training

## Acknowledgements
Our code is based on this implementation of [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](https://github.com/aimbrain/vqa-project)


