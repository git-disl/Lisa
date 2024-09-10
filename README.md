<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning</h1>



Lisa is a fine-tuning stage defense against the threat of harmful fine-tuning.

Check out our [paper](https://arxiv.org/abs/2405.18641).


## Main code logistic
We implement a customized trainer on top of the original HuggingFace Trainer. To achieve Bi-state optimization,  we append one line of code in function ` training_step()` of `trainer.py`. 

```
inputs = self.check_mode(inputs) //Appended code: switch dataset/model according to steps number
loss = step()  //Gradient backward with the data and model
```

To introduce a proximal term towards consensus, we need to add the following regularization to the loss in function `step()`.

```
if self.status =="alignment":
  for name, param in model.named_parameters():
    if param.requires_grad and self.args.rho>0:
        loss +=  self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
else:
  for name, param in model.named_parameters():
    if param.requires_grad and self.args.rho>0:
         loss += self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
```
 



## Package requirement
The package requirement is listed in `lisa.yml` and `lisa_pip.txt`. Run the following code to install the packages with anaconda and pip.  
```
conda env create -f lisa.yml
pip install -r lisa_pip.txt
```

## Data  preparation
For finetuning task, we first need to run the following scripts to prepare the sueprvised finetuning data.
```
cd sst2
python build_dataset.py
cd ../gsm8k
python build_dataset.py
cd ../ag_news
python build_dataset.py
cd ..
```

## Huggingface Llama2 access
Llama2-7B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Llama-2-7b-hf.
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



## Example command to run

We prepare scripts for re-producing all the experiments in the paper. We recommend to use Slurm to reproduce the results as the logging file will be automatically organized into the script directory (if you don't use Slurm, just replace `sbatch` with `bash` in our example).

We first run SFT to produce the aligned model. 
```
cd script/alignment
sbatch  SFT.sh
```
Then we finetune the model using 10% of harmful data with a total number of 5000 samples from SST2 dataset. 
```
cd ../finetune
sbatch  lisa_poison_ratio.sh 0.1
```


For comparison, we finetune the model with SFT in the same data setting.

```
sbatch  sft_poison_ratio.sh 0.1
cd ../..
```

## A line of defense designs

We are commited to design defenses from different angles to harmful fine-tuning. The current avaialble defenses built in the disl group include:

* Alignment stage defense: [Vaccine](https://github.com/git-disl/Vaccine), [Booster](https://github.com/git-disl/Booster/tree/main)
* Fine-tuning stage defense: [Lisa](https://github.com/git-disl/Lisa)
* Post-fine-tuning stage defense: [Antidote](https://arxiv.org/abs/2408.09600)

We always welcome different forms of collaboration. If you are interested, please reach out Tiansheng Huang (thuang374@gatech.edu) for discussion. 



## Citation
If you feel our project is useful, you may cite our paper with the following bibtex.
```
@article{huang2024lazy,
  title={Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning},
  author={Huang, Tiansheng and Hu, Sihao and Ilhan, Fatih and Tekin, Selim Furkan and Liu, Ling},
  journal={arXiv preprint arXiv:2405.18641},
  year={2024}
}

@article{huang2024vaccine,
  title={Vaccine: Perturbation-aware alignment for large language model},
  author={Huang, Tiansheng and Hu, Sihao and Liu, Ling},
  journal={arXiv preprint arXiv:2402.01109},
  year={2024}
}

@article{huang2024booster,
  title={Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation},
  author={Huang, Tiansheng and Hu, Sihao and Ilhan, Fatih and Tekin, Selim Furkan and Liu, Ling},
  journal={arXiv preprint arXiv:2409.01586},
  year={2024}
}

@article{huang2024antidote,
  title={Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning},
  author={Huang, Tiansheng and Bhattacharya, Gautam and Joshi, Pratik and Kimball, Josh and Liu, Ling},
  journal={arXiv preprint arXiv:2408.09600},
  year={2024}
}

```


