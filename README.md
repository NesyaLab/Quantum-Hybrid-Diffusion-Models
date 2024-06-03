# Quantum-Hybrid-Diffusion-Models

## Overview

This repository contains the implementation of our [Quantum Hybrid Diffusion Models](https://arxiv.org/abs/2402.16147) paper, leveraging both quantum computing principles and classical diffusion models based on U-Net for advanced synthetic data generation tasks. The codebase is structured to facilitate experimentation with hybrid approaches, combining the strengths of quantum variational circuits and classical machine learning techniques.

## Features

- **Quantum Variational Circuits**: Integration of quantum circuits to enhance model expressiveness.
- **Hybrid Architecture**: Combines quantum and classical methodologies.
- **Efficient Encoding**: Techniques for embedding classical data into quantum states.
- **Model Training and Evaluation**: Scripts for training models and evaluating performance on synthetic and real-world datasets.

## Repository Structure

- `configs/`: Configuration files for various experimental setups.
- `main.py`: Main script to run the models.
- `sampling.py`: Functions for data sampling and preprocessing.
- `train.py`: Training routines for the hybrid models.
- `unet.py`: Implementation of the U-Net architecture.
- `utils.py`: Utility functions.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries listed in `requirements.txt`

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/NesyaLab/Quantum-Hybrid-Diffusion-Models.git
cd Quantum-Hybrid-Diffusion-Models
pip install -r requirements.txt
```

### Usage

Run the main script with a specified configuration:

```bash
python main.py --config configs/config.yaml
```

## Reference

If you use this code in your research, please cite the following reference:

```
@article{de2024towards,
  title={Towards Efficient Quantum Hybrid Diffusion Models},
  author={De Falco, Francesca and Ceschini, Andrea and Sebastianelli, Alessandro and Saux, Bertrand Le and Panella, Massimo},
  journal={arXiv preprint arXiv:2402.16147},
  year={2024}
}
```

## Acknowledgments

We acknowledge the code from [denoising-diffusion-flax](https://github.com/yiyixuxu/denoising-diffusion-flax) by Yiyi Xu, which inspired our implementation of the U-Net and Denoising Diffusion parts. Special thanks to all contributors and the open-source community for their invaluable support.

## Contributing

We welcome contributions to enhance the functionality and performance of the models. Please submit pull requests or open issues for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Paper

For detailed information on the theory and implementation of our models, please refer to our [Quantum Hybrid Diffusion Models paper](https://arxiv.org/abs/2402.16147).





