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
![](https://github.com/kiharalab/ContactGAN/blob/master/data/git/fig1.jpg)   


## Pre-required software

Python 3 : https://www.python.org/downloads/  
pytorch : pip/conda install pytorch  
CCMPred : A freely avalibale software. It can be downloaded and installed from here : https://github.com/soedinglab/CCMpred  
DeepContact : A freely avalibale software. It can be downloaded and installed from here : https://github.com/largelymfs/deepcontact   
DeepCov : A freely avalibale software. It can be downloaded and installed from here : https://github.com/psipred/DeepCov  

## Instructions  
Generate an input contact map file using a method of your choice from the 4 methods decribed.  
Example file given - data/example_files/5OHQA.ccmpred  
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
Generated output contact map file is the denoised version of the input map.  
Output file looks exactly same as input file structure-wise.  

### Visualization    
```
python util/plot_cmap.py --input=<OUTPUT Contact Prediction File>
  --input               Input Contact Map    
  
```
