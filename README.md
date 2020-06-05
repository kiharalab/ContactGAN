# ContactGAN
ContactGAN takes a predicted protein contact map as input and outputs a new contact map that better captures the residue-residue contacts

Copyright (C) 2020 Sai Raghavendra Maddhuri, Aashish Jain, Yuki Kagaya, Genki Terashi, Daisuke Kihara, and Purdue University.

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)

Contact: Daisuke Kihara (dkihara@purdue.edu)

Cite : Sai Raghavendra Maddhuri Venkata Subramaniya, Aashish Jain, Yuki Kagaya, Genki Terashi, and Daisuke Kihara, Protein Contact Map De-noising using Generative Adversarial Networks. In Submission. (2020)

## About ContactGAN  
> ContactGAN is a novel contact map denoising and refinement method based on Generative adversarial networks.  
> ContactGAN can be trained and combined with any contact prediction method to improve and correct similar patterns of errors that creep into the method.
> Currently ContactGAN is trained and works with 4 contact prediction methods - CCMPred, DeepContact, DeepCov and trRosetta
![](https://github.com/kiharalab/ContactGAN/blob/master/data/git/fig1.png)   


## Pre-required software
```

Python 3 : https://www.python.org/downloads/  
pytorch : pip/conda install pytorch  
CCMPred : A freely available software. It can be downloaded and installed from here : https://github.com/soedinglab/CCMpred  
DeepContact : A freely available software. It can be downloaded and installed from here : https://github.com/largelymfs/deepcontact   
DeepCov : A freely available software. It can be downloaded and installed from here : https://github.com/psipred/DeepCov  

```
## Instructions  
Generate an input contact map file using a method of your choice from the 4 methods described.  
### ContactGAN Usage  
```
python test/denoising_gan_test.py --input=<INPUT Contact Prediction File> --G_res_blocks=3 --D_res_blocks=3 --G_path=model/CCMPred/G_epoch_6000_50 --D_path=model/CCMPredD_epoch_6000_50
  --input               Input Contact Map    
  --G_res_blocks        Number of ResNet blocks in Generator (Default : 3)
  --D_res_blocks        Number of ResNet blocks in Discriminator (Default : 3)
  --G_path              Specify path of Generator model
  --D_path              Specify path of Discriminator model
  
```

### Output interpretation  
Generated output contact map file is the denoised version of the input map.Output file looks exactly same as input file structure-wise.  
An example contact map can be found at the bottom of this page.  
### Visualization    
```
python util/plot_cmap.py --input=<OUTPUT Contact Prediction File>
  --input               Input Contact Map    
  
```

## Tutorial: 
***For the purpose of this tutorial, please refer to example contact map input and output are provided in data/example_files/***   

### ContactGAN Usage  
To run ContactGAN, you will need an input contact map from one of the following 4 methods - CCMpred, DeepCov, DeepContact, or trRosetta.  
An example contact map can be found [here](https://github.com/kiharalab/ContactGAN/tree/master/data/example_files).  
Model files required to run ContactGAN can be found [here](https://github.com/kiharalab/ContactGAN/tree/master/model/)  
Once you have a contact map e.g. CCMpred, you can run ContactGAN as follows:  
1) Specify input map to --input argument
2) G_res_blocks - Number of Generator ResNet blocks. Specify 6 for trRosetta and 3 for others.  
3) D_res_blocks - Number of Disciminator ResNet blocks. Specify 3.  
4) G_path - Generator Model Path. If you're using CCMpred you can use this [path](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred/G_epoch_6000_50)  
5) D_path - Discriminator Model Path. If you're using CCMpred you can use this [path](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred/D_epoch_6000_50)  

```
python test/denoising_gan_test.py --input=data/example_files/5OHQA_input.ccmpred --G_res_blocks=3 --D_res_blocks=3 --G_path=model/CCMPred/G_epoch_6000_50 --D_path=model/CCMPredD_epoch_6000_50

```
### Output contact map Visualization  
```
python util/plot_cmap.py --input=data/example_files/5OHQA_output.npy
 
```
Below is an example visualization for contact maps of CCMpred before and after ContactGAN for protein with PDB ID: [5OHQA](http://www.rcsb.org/structure/5OHQ).      
![](https://github.com/kiharalab/ContactGAN/blob/master/data/git/fig2.jpg)   
