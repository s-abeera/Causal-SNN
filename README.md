# Causal SNN

This repository hosts the codebase for our ongoing research on evaluating a Spiking Neural Network (SNN) pipeline using the Abstract Causal Reasoning (ACRE) dataset. Our study aims to explore the capabilities of SNNs in identifying causal relationships within a complex dataset designed for testing cognitive reasoning.

## Data

The data is available at [https://wellyzhang.github.io/project/acre.html](https://wellyzhang.github.io/project/acre.html) from the paper Zhang, C., Jia, B., Edmonds, M., Zhu, S.-C., & Zhu, Y. (2021). ACRE: Abstract Causal REasoning Beyond Covariation (arXiv:2103.14232). arXiv. [http://arxiv.org/abs/2103.14232](http://arxiv.org/abs/2103.14232)

## Installation

To create a virtual environment with conda, run the following commands:

```
conda env create -f environment.yml -n causal-snn
conda activate causal-snn
```

## Usage

After activating the environment, you can run the main script to start training the model:

```
python train_main.py
```

For testing the model with the ACRE dataset, use:

```
python test.main.py
```

## Requirements

- TensorFlow 2.x
- Sonnet 2.x
- Numpy
- Matplotlib (for visualization)

Please refer to environment.yml for a detailed list of dependencies.

## Contact

For any queries related to the codebase, please contact me at syeda.abeera.amir@gmail.com.
