# Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Huggingface Account Setup and Access Token

Create an account on huggingface.co.

Create an access token from the following link: https://huggingface.co/settings/tokens
Access tokens programmatically authenticate your identity to the Hugging Face Hub, allowing applications to perform specific actions specified by the scope of permissions granted. Visit the documentation to discover how to use them.

# Weights and Bias Account Setup and API Key

Create an account on [Weights and Biases](https://wandb.ai/)

Create an api key from the following link: https://wandb.ai/settings#api
API Keys enable huggingface training scripts to log training runs in wandb.ai to visualize the performance of the model on datasets.

# How to train the model

To train the model, run the following command:
```
export TOKENIZERS_PARALLELISM=true
cd scripts
python hf_model_train.py
```