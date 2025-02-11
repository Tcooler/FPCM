# FPCM

FPCM is an algorithm that, under small sample conditions, distinguishes therapeutic peptides from non-therapeutic peptides with high accuracy based on peptide sequences, aiming to address the issue of limited training data for certain therapeutic peptides.
FPCM fine-tunes the protein language pre-training model using LoRA to transfer data from the original model to the target task. It employs a clustering-based metric learning approach to minimize the disruption to the original space during the fine-tuning process, thereby improving prediction accuracy. The structure of FPCM is shown in the figure below.
<img width="525" alt="image" src="https://github.com/user-attachments/assets/6e2952ec-3867-413c-abe9-bf0b388a171e" />

## Getting Started



### Prerequisites

FPCM is fine-tuned based on the [ProtST model published by Xu et al.](https://arxiv.org/abs/2301.12040), which is built on the [esm-2 650M model architecture by Lin et al](https://www.science.org/doi/10.1126/science.ade2574). Therefore, the esm Python package needs to be installed, and the installation instructions will be provided in the installation section.

Additionally, download the ProtST-ESM-2 model parameters by [clicking here](https://protsl.s3.us-east-2.amazonaws.com/checkpoints/protst_esm2.pth). You can name the file protst_esm2_protein.pt and store it in the FPCM folder so that the program can automatically locate the parameter file during subsequent training follow the default without needing to provide the file path. Of course, you can also store it anywhere, but you will need to specify the path when running the file.

### Installing
```bash
# clone project
git clone https://github.com/Tcooler/FPCM.git
cd SenseXAMP

# create conda virtual environment
conda create -n FPCM python=3.9 
conda activate FPCM

# install all requirements
pip install fair-esm
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0
pip install pandas
pip install numpy
pip install scikit-learn
```

## Running the tests

### Training

Train the model and save the best parameters. The best result obtained on the validation set during the training process will be output in the result.csv file.

```bash
python run_train.py --positive_file='../dataset/ACP/positive.fasta' --negative_file='../dataset/ACP/negative.fasta' --pre_trained_model_dir='../protst_esm2_protein.pt' --device='cuda:1'
```

### Testing

Load the fine-tuned model parameters and test on the input test set. The results will be saved in the test_result.csv file.

```bash
python run_test.py --positive_file='../dataset/ACP/positive.fasta' --negative_file='../dataset/ACP/negative.fasta' --pre_trained_model_dir='../protst_esm2_protein.pt' --device='cuda:1' --model_file='../model/123_2_0.006_23_0.5.pt'
```

## Predicting

Make a prediction on the therapeutic activity of the samples contained in the input FASTA file. Samples with label 1 are considered to have therapeutic activity, while samples with label 0 are considered to have no therapeutic activity.

```bash
python run_predict.py --samples='../ACP/positive.fasta' --pre_trained_model_dir='../protst_esm2_protein.pt' --device='cuda:1' --model_file='../model/123_2_0.006_23_0.5.pt'
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


