# CHASe

CHASe: Client Heterogeneity-Aware Data Selection for Effective Federated Active Learning
![Overview](./assets/intro-new.png)

## Requirements
- PyTorch
- NumPy

## Files
- options.py: Hyperparameter setting for CHASe.
- sampling.py: Sample the MNIST, EMNIST, CIFAR10, CIFAR-100 and Shakespeare in a IID/NonIID manner.
- utils.py: 
  - Construction of labeled, unlabeled and global test sets for sampled dataset；
  - Server's aggregation & Definition of log detail.
- model.py: Models for  MNIST, EMNIST, CIFAR10, CIFAR-100 and Shakespeare datasets.
- localtraining.py: 
  - Clients' local training ; 
    - Quantify Epistemic Variation;
    - Calibrate Decision Boundary;
  - Inference of local & global model.
- slected_strategy.py: Definition of the sampling with EV.
- main: Core code for CHASe, Logic and interaction throughout the pipeline.

## Usage
1. Download MNIST, EMNIST, CIFAR10 , CIFAR-100 and Shakespeare datasets or Execute the program default download; 
2. Set parameters in options.py;
3. Execute main.py to run the CHASe.
