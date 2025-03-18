# Novel Pipeline for Cyberbullying Detection using Deep Learning Methods

## Project Overview

This project presents a robust and novel pipeline for the detection of cyberbullying in online text, specifically targeting social media content.  Cyberbullying is a growing concern, and this research aims to develop an effective and transparent deep learning-based solution to identify and mitigate online harassment.

This repository contains the code and documentation for my MSc Data Science project, focusing on leveraging advanced Natural Language Processing (NLP) techniques and deep learning models to achieve high accuracy in cyberbullying detection.

## Methodology and Approach

The core of this project lies in a hybrid approach that combines cutting-edge NLP techniques with a sophisticated deep learning architecture. The key components of the pipeline include:

1.  **Hybrid Word Embeddings:**  Utilizing a combination of **FastText** and **BERT** embeddings to capture both subword information and contextual semantic meaning from the text data. This hybrid approach aims to create richer and more informative word representations.

2.  **Autoencoder for Feature Extraction and Dimensionality Reduction:** An **autoencoder neural network** is employed to extract meaningful features from the high-dimensional hybrid embeddings and reduce dimensionality, removing noise and redundancy while preserving essential information.

3.  **Principal Component Analysis (PCA) for Further Dimensionality Reduction:** **Principal Component Analysis (PCA)** is applied to further reduce the dimensionality of the autoencoder-extracted features, enhancing computational efficiency and focusing on variance preservation.

4.  **Temporal Convolutional Network (TCN) for Sequential Modeling:** A **one-dimensional Temporal Convolutional Network (1D-TCN)** is used to model the sequential nature of text data and capture temporal dependencies within cyberbullying messages. TCNs are well-suited for processing sequential data and capturing long-range context.

5.  **ExtraTrees Classifier Ensemble:** An **ExtraTrees Classifier** is integrated as an ensemble method, trained on the features extracted by the TCN. This ensemble approach enhances the robustness and generalization capability of the final classification model.

## Key Features and Highlights

*   **Novel Hybrid Model:**  Combines TCN with ExtraTrees Classifier and hybrid word embeddings for improved accuracy and robustness in cyberbullying detection.
*   **High Accuracy:** Achieves a high accuracy of **90.30%** on the evaluation dataset, outperforming several baseline models and demonstrating the effectiveness of the proposed pipeline.
*   **Emphasis on Transparency and Interpretability:**  While leveraging deep learning, the use of PCA and ExtraTrees Classifier, along with a clear methodology, contributes to a more interpretable and understandable system compared to purely black-box deep learning approaches.
*   **Addresses Class Imbalance:**  The evaluation methodology prioritizes metrics like F1-score and ROC-AUC, which are suitable for imbalanced datasets common in cyberbullying detection.
*   **Comprehensive Documentation:** This repository includes well-documented code, a detailed project report, and a comprehensive README to facilitate understanding and reproducibility.

## Dataset

The project utilizes the publicly available **"Tweets Dataset for Detection of Cyber-Trolls"** from Kaggle. This dataset contains human-labeled tweets for binary classification (aggressive vs. non-aggressive) and is widely used in cyberbullying research.

## Code Structure

The repository is organized as follows:

*   **`main(cyber).ipynb`:**  Jupyter Notebook containing the main Python code for the cyberbullying detection pipeline. This notebook includes:
    *   Data loading and exploration
    *   Preprocessing and feature engineering (hybrid embeddings, autoencoder, PCA)
    *   Model building and training (TCN, ExtraTrees Classifier)
    *   Model evaluation and results analysis
*   **`EMBEDDING CODE (IPYNB)`:** Jupyter Notebook containing the code for generating and saving the hybrid word embeddings (FastText + BERT).
*   **`data/`:**  Directory to store the dataset.  *(Note: You may need to download the "Tweets Dataset for Detection of Cyber-Trolls" from Kaggle and place it in this directory).*
*   **`saved_files/`:** Directory to store saved model weights, embeddings, and other output files.
*   **`report/`:** (Optional) Directory to store the project report (e.g., PDF or Word document).
*   **`README.md`:**  This file, providing an overview of the project.

## Dependencies

The project code is written in Python and requires the following libraries:

*   `pandas`
*   `numpy`
*   `plotly.express`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn (sklearn)`
*   `tensorflow`
*   `keras-tcn`
*   `transformers`
*   `fasttext`
*   `textblob`
*   `emoji`
*   `contractions`
*   `wordcloud`

You can install these dependencies using `pip`:
