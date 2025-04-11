# COMP34812 Natural Language Understanding — NLI Project

Welcome to our submission for the COMP34812 shared task on Natural Language Inference (NLI).
Our project implements and compares two distinct deep learning approaches for pairwise sequence classification.
Project Overview

# We explored two powerful neural approaches:
Approach B: BiLSTM
- A bidirectional LSTM network that captures contextual dependencies in both directions.
- Model Card: [View Model Card](/NLU_Method_B/my_model_card.md)
- Demo Notebook: [NLU_Method_B/BILSTM_Demo.ipynb](/NLU_Method_B/BILSTM_Demo.ipynb)
- Training Notebook: [NLU_Method_B/BILSTM_RNN_Trainer.ipynb](/NLU_Method_B/BILSTM_RNN_Trainer.ipynb)

Approach C: Transformer Ensemble
- An ensemble of transformer-based models leveraging the strength of multiple pre-trained language models.
- Model Card: [View Model Card](/NLU_Method_C/my_model_card.md)
- Demo Notebook: [NLU_Method_C/NLI_Transformer_demo.ipynb](/NLU_Method_C/NLI_Transformer_demo.ipynb)
- Training Notebook: [NLU_Method_C/Transformer_Train_and_Evaluate.ipynb](/NLU_Method_C/Transformer_Train_and_Evaluate.ipynb)

# Running the Demo Scripts

Both demo scripts are designed for Google Colab and will automatically install required Python libraries.

Steps to run:
- Open the relevant notebook in Colab.
- Follow the instructions at the top to upload your test dataset.
- Run the notebook cells to generate predictions.

BiLSTM Demo: [NLU_Method_B/BILSTM_Demo.ipynb](/NLU_Method_B/BILSTM_Demo.ipynb)

Transformer Demo: [NLU_Method_C/NLI_Transformer_demo.ipynb](/NLU_Method_C/NLI_Transformer_demo.ipynb)


Outputs will be in the required CSV format for submission.

# Model Training

Our training notebooks are also Colab-friendly and fully documented.

BiLSTM Training: [NLU_Method_B/BILSTM_RNN_Trainer.ipynb](/NLU_Method_B/BILSTM_RNN_Trainer.ipynb)

Transformer Training: [NLU_Method_C/Transformer_Train_and_Evaluate.ipynb](/NLU_Method_C/Transformer_Train_and_Evaluate.ipynb)

Both scripts handle:
- Dataset loading
- Model training
- Evaluation on the dev set
- Saving models for reuse in demo notebooks

# Additional Resources
## Training Datasets

We used the official dataset provided in the coursework:

[Dataset on Hugging Face](https://huggingface.co/datasets/aap9002/NLU-Coursework)

## Pre-trained Models

Our trained models are stored on Hugging Face for easy reuse:

[Transformer Ensemble Model](https://huggingface.co/aap9002/NLI-Transformer-Ensemble-Model)

[BiLSTM Model](https://huggingface.co/aap9002/NLI-BILSTM)
