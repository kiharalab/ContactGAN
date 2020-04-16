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

