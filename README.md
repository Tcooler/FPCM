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


## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

/```
Give an example
/```

### And coding style tests

Explain what these tests test and why

/```
Give an example
/```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

