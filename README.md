Implementation of the LogTokU uncertainty estimation method for Large Language Models as in the paper [Estimating LLM Uncertainty with Evidence, Ma et al., 2025](https://arxiv.org/abs/2502.00290), showcased during the [ESSAI 2025](essai2025.eu) summer school.

#### Requirements

This repo requires the following libraries

- transformers
- bitsandbytes
- huggingface_hub (optional, for using models which require a huggingface access token)

#### Execution

The library `uncertainty.py` contains the `logTokU` function (plus some helper functions and a couple of output-based dummy uncertainty quantification techniques).

The walkthrough given during the ESSAI course is in the `llms.ipynb` notebook.
