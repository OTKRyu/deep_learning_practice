# OTKRyu's deep learning practice
## information
- dataset : mnist
- language : python
- requirements : numpy, matploblib
- source : deep learning form scratch
- activater : ReLU
## design
### layers
- set W1, W2,W3 with He
- input
- layer1
  - affine
  - batch_normalization
  - ReLU
- dropout
- layer2
  - affine
  - batch_normalization
  - ReLU
- output
  - affine
  - softmax
### optimizer
optimizer : SGD with weight decay
## process
### 0413
project start and basic settings

### 0415

add source code from deep_learning_from_scratch github

add find.py for searching hyper parameters and find_data.txt for saving those hyper parameters

add train.py for finding weight matrix and bias and final_data.txt for saving those matrixes and biases

add final.py for actual use

#### 0416

add train_data.txt for training

change find.py because of overflow issue

run find.py and get hyper parameters

run train.py and get weights matrix and bias

## reuslt