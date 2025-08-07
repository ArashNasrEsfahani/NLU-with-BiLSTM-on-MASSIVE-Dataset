# 🤖 Intent Classification and Slot Filling Project

This project implements and trains deep learning models for two core Natural Language Understanding (NLU) tasks: Intent Classification and Slot Filling. The models are trained on the Persian subset of the multilingual [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) dataset.

## 📝 Project Overview

The project is divided into two main parts:

1.  **Slot Filling**: The goal is to identify and label key pieces of information (slots) within a user's utterance. For example, in "set an alarm for 7 am," the model should identify "7 am" as a `time` slot. An LSTM-based model is implemented for this token-level classification task.

2.  **Intent Classification**: The goal is to determine the overall purpose (intent) of a user's utterance. For example, the intent for "play the latest hits" would be `play_music`. A separate LSTM-based model is implemented for this sentence-level classification task.

## ✨ Features

-   **Dataset Analysis**: In-depth analysis of the MASSIVE dataset's structure, class distribution, and data splits.
-   **Slot Filling Model**: A Bidirectional LSTM model that predicts a slot label for each token in a sentence.
    - Achieved a final **F1 Micro score of 0.8439** and an **F1 Macro score of 0.6922** on the development set.
-   **Intent Classification Model**: A Bidirectional LSTM model that classifies the intent of the entire sentence.
    - Reached a **validation accuracy of 82.75%** and an **F1 Macro score of 0.784**.
-   **Custom Data Handling**: Includes a `DatasetHandler` class and custom batching functions for efficient processing in PyTorch.
-   **Inference**: Demonstrates model capabilities with successful inference on new Persian sentences.

## 📊 Dataset

This project uses the **MASSIVE** dataset, a large, open-source, multilingual dataset created by Amazon for NLU tasks. It contains over 16,000 utterances in English and 10,000+ in other languages, annotated for intent and slots. This project focuses on the Persian portion of the dataset.

A key challenge in this dataset is the significant class imbalance, where some intents are much more frequent than others. This was a critical consideration during model training and evaluation.

## 🤖 Models & Tasks

### 1. Slot Filling
This task is treated as a sequence-labeling problem.
-   **Architecture**: The model uses an Embedding layer followed by a Bidirectional LSTM. A final linear layer projects the LSTM's output for each token to the space of possible slot labels.
-   **Performance**: The model shows a strong learning trend, with F1 scores steadily improving over 10 epochs. The gap between Micro and Macro F1 scores highlights the challenge of identifying less frequent slot types.

### 2. Intent Classification
This task involves classifying the entire input sentence into one of 60 predefined intents.
-   **Architecture**: The model uses a 4-layer Bidirectional LSTM. The final hidden states of the LSTM are concatenated and passed through a linear layer to produce logits for each intent class.
-   **Performance**: The model demonstrates excellent improvement, with validation accuracy rising from 53% to over 82% and the F1 Macro score increasing from 0.319 to 0.784, indicating its growing ability to classify even rare intents correctly.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## 🛠️ Technologies Used

-   Python
-   PyTorch
-   Pandas
-   Hugging Face `datasets`
-   Scikit-learn
-   Matplotlib & Seaborn
-   Jupyter Notebook
-   NumPy