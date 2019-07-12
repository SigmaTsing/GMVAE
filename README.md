# Code for "Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"
(https://arxiv.org/abs/1611.02648)
By
Nat Dilokthanakul, Pedro A.M. Mediano, Marta Garnelo, Matthew C.H. Lee, Hugh Salimbeni, Kai Arulkumaran, Murray Shanahan

# Abstract
We study a variant of the variational autoencoder model with a Gaussian mixture as a prior distribution, with the goal of performing unsupervised clustering through deep generative models. We observe that the standard variational approach in these models is unsuited for unsupervised clustering, and mitigate this problem by leveraging a principled information-theoretic regularisation term known as consistency violation. Adding this term to the standard variational optimisation objective yields networks with both meaningful internal representations and well-defined clusters. We demonstrate the performance of this scheme on synthetic data, MNIST and SVHN, showing that the obtained clusters are distinct, interpretable and result in achieving higher performance on unsupervised clustering classification than previous approaches.

# Requirements
Luarocks packages:
- mnist ( torch-rocks install https://raw.github.com/andresy/mnist/master/rocks/mnist-scm-1.rockspec )

Python packages:
- torchfile (pip install torchfile)

# Instructions

To run a spiral experiment,

	./run.sh spiral

To visualise spiral experiment (can be used while training)

	cd plot
	python plot_latent.py
	python plot_recon.py


To run MNIST

	./run.sh mnist

To use GPU

set flag

	./run.sh mnist -gpu 1

To run quick MNIST on fully-connected network

	./run.sh mnist_fc

# Acknowledgements

I would like to thanks the following, whose github's repos were used as inital templates for implementing this idea.

1. Rui Shu https://github.com/RuiShu/cvae

2. Joost van Amersfoort https://github.com/y0ast/VAE-Torch

3. Kai Arulkumaran https://github.com/Kaixhin/Autoencoders

# Extensions (LYL)

I added code for Fashion-MNIST, STL-10, CIFAR-10, CIFAR-100, SVHN using fully connected layers.

## HOWTO

1. Install LuaJIT and luarocks following http://torch.ch/docs/getting-started.html. Don't ever try to install it on Ubuntu 18.04 (yes, you can install torch7, but you will eventually fail to install matio package). Ubuntu 16.04.5 LTS is fine. Still you may encounter "error: more than one operator "==" matches these operands", for which you `export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__` before building.

2. Install mnist, fashion-mnist, matio. Should be unnecessary to configure Python.

   ```shell
   luarocks install mnist
   luarocks install https://raw.github.com/mingloo/fashion-mnist/master/rocks/fashion-mnist-scm-1.rockspec`
   apt-get install libmatio2	
   # You can only find libmatio4 on Ubuntu 18.04, which I think is not compatible with matio
   luarocks install matio
   # Or: luarocks install https://raw.github.com/soumith/matio-ffi.torch/master/matio-scm-1.rockspec
   ```

3. Copy the following files into `datasets`

   ```
   cifar10_feature.mat
   cifar100_feature.mat
   stl10_feature.mat
   train_gist.mat
   test_gist.mat	
   ```

4. Run `./run.sh {spiral,mnist,mnist_fc,fashion-mnist-fc,stl-10,cifar-10,cifar-00,svhn} [-gpu GPU]` (note that gpu index starts with 1)

## NOTE

I can't find what's wrong about `./run.sh fashion-mnist [-gpu GPU]`,  but the tensor shapes do not match.

# Notes by XSQ
(to someone who doesn't know lua at all, like me)

./run.sh (or simply th main.lua) runs with several args, a simple way is to tag uncomplete order like "./run.sh -gpu", and program will show you all possible options

usually runs with orders like "./run.sh -gpu 1 -seed 1" (enable gpu and runs with random seed)

(btw the randomlize progress of lua is so amazing that if you forget -seed 1, you will discover all your programs get exactly same result :(

And if you get CUDA in your system, just run "CUDA_VISIBLE_DEVICES=* ./run.sh -gpu 1 -seed 1"
