<h1 align="center">
    <img width="200" height="auto" src="resources/molecule.png" />
    <br>
    PyTorch Implementation of MolGAN
    <br>
</h1>

<p align="center">
  <a href="#mi-estimator">MI Estimator</a> •
  <a href="#pc-softmax">PC Softmax</a> •
  <a href="#InfoCAM">InfoCAM</a> •
  <a href="#credits">Credits</a> •
  <a href="#licence">Licence</a>
</p>

<p align="center">
    <img width=100% height="auto" src="resources/Info-CAM-Illustration.png" alt="InfoCAM-Procedure" />
</p>

<h3 align="center">
Overview
</h3>

<p align="center">
<strong align="center">
This repository contains a PyTorch implementation of MolGAN: An implicit generative model for small molecular graphs 
(https://arxiv.org/abs/1805.11973). 
</strong>
</p>

## Usage
You can either run MolGAN or MolVAE. Please download and preprocess the dataset before training molGAN. 
### Data Download 
Just simply run a bash script. 
You can find such a bash script in the data 
directory. 
If you see warnings or even errors, 
please just don't worry. 

If you wish to use the QM9 dataset, 
you can skip the data downloading and 
directed proceed to data preprocessing. 
```bash
bash download_dataset.sh
```
### Data Preprocessing 
Just run the python script below. 
You will need to comment different different code 
sections in the main function. 
It is too easy to figure out how to do. 
```bash
python sparse_molecular_dataset.py
```
### MolGAN
Simply run the following command to train. 
```bash
python main_gan.py
```
### MolVAE
For your convenience, 
a VAE version is also implemented. 
```bash
python main_vae.py
```
### Testing
You will need to change some arguments 
in ``args.py`` to test the saved model. 
It is too easy and I believe you can figure out 
how to do it. 
### Kind Reminder
If you witness low or even zero validity for 
generated molecules during training, that is 
normal. Please just don't worry. 

<p align="center">
    <img width=15% height="auto" src="resources/rl_1.png" alt="rl_1" />
    <img width=15% height="auto" src="resources/rl_2.png" alt="rl_2" />
    <img width=15% height="auto" src="resources/rl_3.png" alt="rl_3" />
    <img width=15% height="auto" src="resources/rl_4.png" alt="rl_4" />
    <img width=15% height="auto" src="resources/rl_5.png" alt="rl_5" />
    <img width=15% height="auto" src="resources/rl_6.png" alt="rl_6" />
</p>


## Dependencies 
I use PyTorch 1.5. There is no magic for installing 
packages. You can just install all the required 
packages if you run into ``no-such-package`` issues :-) 

## Credits
This repository uses the following implementations: 
- [Fine-Grained Inception-V3](https://github.com/richardaecn/cvpr18-inaturalist-transfer)
- [ADL](https://github.com/junsukchoe/ADL/tree/master/Pytorch)
- [PyTorch-Example](https://github.com/pytorch/examples/tree/master/mnist)

## Licence
[CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)

