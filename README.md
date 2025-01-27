# Demo code implementation for the TFGNN (<u>T</u>rigonometric <u>F</u>ilter <u>G</u>raph <u>N</u>eural <u>N</u>etwork)

## Environment
Ubunto-22.04  
CUDA-11.8  
python == 3.10  
torch == 2.4.0  
torchvision == 0.19.1  
torchaudio == 2.4.1  
torch-geometric == 2.6.0  
pyg_lib == 0.4.0  
torch_cluster == 1.6.3  
torch_scatter == 2.1.2  
torch_sparse == 0.6.18  
torch_spline_conv == 1.2.2  


## Executions
You can input the code below to reproduce partial of experiments.

**Cora:**
`python train.py --dataset cora --r_train 0.6 --r_val 0.2 --lr 0.001 --prop_lr 0.01 --weight_decay 0 --prop_wd 0.5 --dropout 0.5 --dprate 0.5 --K 6 --omega 0.3`

**Citeseer:**
`python train.py --dataset citeseer --r_train 0.6 --r_val 0.2 --lr 0.01 --prop_lr 0.1 --weight_decay 0.0005 --prop_wd 0.5 --dropout 0 --dprate 0.7 --K 2 --omega 0.5`

**Pubmed:**
`python train.py --dataset pubmed --r_train 0.6 --r_val 0.2 --lr 0.1 --prop_lr 0.5 --weight_decay 0 --prop_wd 0.5 --dropout 0.2 --dprate 0 --K 4 --omega 0.2`

**Roman-empire:**
`python train.py --dataset roman-empire --r_train 0.5 --r_val 0.25 --lr 0.001 --prop_lr 0.5 --weight_decay 0.0005 --prop_wd 0 --dropout 0.2 --dprate 0 --K 2 --omega 0.2`

**Amazon-ratings**
`python train.py --dataset amazon-ratings --r_train 0.5 --r_val 0.25 --lr 0.001 --prop_lr 0.5 --weight_decay 0.0005 --prop_wd 0 --dropout 0.5 --dprate 0 --K 4 --omega 0.3`

## Notes
- This code offers a reproducible demo for the conference submission as well as fast experiments for further research. Full implementations will be shared in the near future.
