# Highway Convolutional Network

Implementation of simple highway convolutional network in PyTorch 0.4.0 (tested on MNIST).

* Paper:
  * [Highway Networks](https://arxiv.org/abs/1505.00387)
  * [Training Very Deep Networks](https://arxiv.org/abs/1507.06228)
* Dataset:
  * [MNIST](http://yann.lecun.com/exdb/mnist/)

## Execution
1. Install [PyTorch](https://pytorch.org/).
2. Install other dependencies.
```
pip install --upgrade numpy
pip install --upgrade matplotlib
pip install --upgrade scikit-learn
```
3. Clone the repository.
```
git clone https://github.com/MajerMartin/highway_convolutional_network.git
cd highway_convolutional_network
```
4. Run the main script.
```
python main.py
```

## Notes
* Training parameters are currently defined in *main.py*.
* Model parameters are currently defined as keyword arguments only.
* Stratified train/validation split is used to split original training data.

## TODO
* Use argparse for training related and model parameters.
* Add max pooling to the highway blocks.
