# OTKRyu's deep learning practice
## information
- dataset : mnist
- language : python
- requirements : numpy, matploblib
- source : deep learning form scratch
- activater : ReLU
## design
### take one
#### layers
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
### take two
#### layers
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
- dropout
- layer3
  - affine
  - batch_normalization
  - ReLU
- dropout
- layer4
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

#### 0419
change train.py because of saving weights matrix and bias
run train.py and get result
complete final.py
#### 0420
try take two
change layers and searching area for hyper parameters
search hyper parameters
train and get weight matrix and bias
## reuslt
### take one
maybe just two layers are too few to get better result.
total fail. i don't know whether make this better or go to another object
i compared my take one and take two, there are many difference
some of those are main issue
first one is hyper parameters. i searched too short area for hyper parameters so i couldn't get meaningful result.
and another one is layers. at take one, i only use 2 layers but it wasn't many enough for decribing 28 * 28 picture.
### take two
in training time, i was satisfied.
test_acc is alomost reached 97%, so i thought in this depth and design that's limit and it's hard to overcome.
but in actual use in final.py, it didn't work well.
i guess weights matrixs and bias are enough for actual use, but it's not true or it's limit of linear affine.
i don't know which reason dominate these problem.