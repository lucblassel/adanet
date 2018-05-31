# ISI-10 Project
Implementation of ADANET neural network in Python framework  
According to [this paper](https://arxiv.org/pdf/1607.01097.pdf) _(AdaNet: Adaptive Structural Learning of Artificial Neural Networks)_

### Authors:
* Luc Blassel
* Romain Gautron

### Idea:
This method will be tested on a binary classification task, extracted from the CIFAR-10 dataset.  
The network will start with only an input layer and an output layer. Then iteratively the network will have a subnetwork added in between the inout and output layers. This subnetwork will be chosen between one that has the same number of layers as the one on the previous step, or one with a single layer more. The subnetwork that performs the best will be the one added to the general network.  
Each layer of depth _k_ of the subnetwork will be connected to some of the layers of depth _k-1_ of itself and other subnetworks previously generated.  
The subnetworks themselves will be randomly generated.  

![The two subnetworks (in red) being evaluated](schema.png)
