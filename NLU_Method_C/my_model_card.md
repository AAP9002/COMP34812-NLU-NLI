---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/AAP9002/COMP34812-NLU-NLI

---

# Model Card for z72819ap-e91802zc-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to detect whether a premise and hypothesis entail each other or not, using binary classification.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a ensemble of RoBERTa models that was fine-tuned using over 24K premise-hypothesis pairs from the shared task dataset for Natural Language Inference (NLI).

- **Developed by:** Alan Prophett and Zac Curtis
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** roberta-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/FacebookAI/roberta-base
- **Paper or documentation:** https://arxiv.org/abs/1907.11692

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K+ premise-hypothesis pairs from the shared task dataset provided for Natural Language Inference (NLI).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


    All Models and datasets
      - seed: 42

    Roberta Large NLI Binary Classification Model
      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - num_epochs: 5

    Semantic Textual Similarity Binary Classification Model
      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - num_epochs: 5

    Ensemble Meta Model
      - learning_rate: 2e-05
      - train_batch_size: 128
      - eval_batch_size: 16
      - num_epochs: 3
      

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 309 minutes 30 seconds

    Roberta Large NLI Binary Classification Model
      - duration per training epoch: 11 minutes
      - model size: 1.42 GB

    Semantic Textual Similarity Binary Classification Model
      - duration per training epoch: 4 minutes 30 seconds
      - model size: 501 MB

    Ensamble Meta Model
      - duration per training epoch: 4 minutes
      - model size: 1.92 GB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 6K+ pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The Ensemble Model obtained an F1-score of 91% and an accuracy of 91%.

## Technical Specifications

### Hardware


      - RAM: at least 10 GB
      - Storage: at least 4GB,
      - GPU: a100 40GB

### Software


      - Tensorflow 2.18.0+cu12.4
      - Transformers 4.50.3
      - Pandas 2.2.2
      - NumPy 2.0.2
      - Seaborn 0.13.2
      - Huggingface_hub 0.30.1
      - Matplotlib 3.10.0
      - Scikit-learn 1.6.1

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
