# lstm-matlab
Matlab LSTM deep learning framework by Moritz Nakatenus.

Overview
=========
This work is part of a Honors Thesis supervised by Elmar Rueckert. The framework can handle a LSTM cell with peephole connections. All gradients are derived fully analytically. For the exact derivations of all gradients used in this implementation, see the file [LSTMGradientsDerivations.pdf](LSTMGradientsDerivations.pdf). In this work, the 'Backpropagation Through Time' and 'Truncated Backpropagation Through Time' algorithm are also explained.

Features
=========
* Backpropagation Through Time algorithm
* Truncated Backpropagation Through Time algorithm
* RMSProp and Momentum Optimizer
* MSE and Cross-Entropy-Cost
* Softmax-output layer and unsaturated output layer can be used
* A example training-script using cross-validation and evaluation plots

Usage
======
To create a new LSTM network you just have to write one line of code like so

`network = lstm_network(timesteps, inputDimension, hiddenNeurons, optimizer);`

where optimizer can be `'Momentum'` or `'RMSProp'`. The specific optimization parameters can be defined in `lstm_network.m`. To run the BPTT algorithm on your code, add the following line

`[error, pred] = network.BPTT(Y, X);`

which takes input `X` and target values `Y` and gives you back the specific error and all predictions over time. For running 'Truncated Backpropagation Through Time', add this

`[error, pred] = network.TruncatedBPTT(Y, X, k1, k2);`

`k1` means, that BPTT runs every k1 timesteps and `k2` is for looking k2 timesteps back in time to compute the gradients.

It is also possible to seperately do forward propagation on your network

`pred = network.forwardPropagation(X);`

`X` again stands for the input data. If you want to manually compute the MSE or Cross-Entropy cross, add the regarding lines like so

`MSE = network.mseCost(pred, Y);`

`CrossEntropyErr = network.crossEntropyCost(pred, Y);`

Evaluation of a Toy Task
========================
In this case, a simple sin-function was used. The training and test datasets for cross-validation are sin-functions over 20 timesteps. 5 datasets are used and each one is trained over 50 epochs. The standard BPTT training algorithm is choosed using the momentum optimizer. The momentum optimizer has a learning rate of 0.02 and a momentum term of 0.8. The network has a hidden layer with 20 hidden neurons (10 for the hidden output and 10 for the hidden state) for each LSTM gate. As the function to learn has no probabilistic interpretation, the network is trained with a MSE loss function. On the below figure you can see the MSE over epochs

![MSESin](https://github.com/MoritzN89/lstm-matlab/blob/master/images/MSESinPred.svg)

Citation
=========
If you want to use my code, please cite my work in your studies:
```
@techreport{Nakatenus2017,
 title = {Derivation of Backpropagation Through Time Gradients for LSTM Neural Networks},
 author = {Moritz Nakatenus},
 year = {2017},
 institution = {Technische Universität Darmstadt},
 type = {M.Sc. Project},
 keywords = {RNN, Matrix Calculus},
 pubstate = {published}
}
```
Contact **Moritz Nakatenus** [:envelope:](mailto:moritznakatenus@yahoo.de) for questions, comments and reporting bugs.

