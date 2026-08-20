# 📰 Fake News Detection Using Deep Learning

<p align="center">

**An NLP-powered deep learning system for detecting potentially misleading news content**

<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Classification-8A2BE2?style=for-the-badge)](#-methodology)

</p>

---

## 📌 Overview

The rapid spread of misinformation across digital platforms has made automated news credibility analysis an increasingly important Natural Language Processing (NLP) problem.

**Fake News Detection Using Deep Learning** is a machine-learning application designed to analyze the textual content of a news article and classify it as:

> 🟢 **REAL NEWS**
> 🔴 **FAKE NEWS**

The system combines **Natural Language Processing, TF-IDF feature extraction, and a trained Deep Neural Network** to transform raw news text into numerical representations and generate a classification prediction.

A lightweight **Streamlit web interface** makes the trained model accessible through an interactive browser-based application.

> ⚠️ **Important:** This project is an educational machine-learning system. Its prediction should not be treated as a definitive fact-checking verdict. Real-world misinformation detection requires source verification, contextual analysis, cross-referencing, and human judgment.

---

## ✨ Key Features

* 📰 **Real vs. Fake Classification**
  Classifies submitted news content into REAL or FAKE categories.

* 🧠 **Deep Learning Model**
  Uses a trained neural-network classifier for text classification.

* 🔤 **Natural Language Processing**
  Performs text preprocessing before model inference.

* 📊 **TF-IDF Feature Engineering**
  Converts processed news text into numerical features suitable for machine learning.

* 🎯 **Prediction Confidence**
  Displays a prediction score/confidence estimate alongside the classification.

* 🌐 **Interactive Streamlit Interface**
  Provides a simple interface for testing news articles without writing additional code.

* 💾 **Saved Model Artifacts**
  Includes the trained model and fitted TF-IDF vectorizer for reproducible inference.

* 🧪 **Ablation Analysis**
  The project evaluates how preprocessing, feature configuration, dropout, and architecture choices affect performance.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     User Input       │
                    │   News Article Text  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Text Preprocessing │
                    │                      │
                    │ • Cleaning           │
                    │ • Stopword Removal   │
                    │ • Lemmatization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    TF-IDF Vectorizer │
                    │                      │
                    │ Text → Numerical     │
                    │ Feature Representation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Deep Neural Network│
                    │      Classifier      │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │       Classification       │
                 │                            │
                 │     REAL  /  FAKE          │
                 │     + Confidence           │
                 └────────────────────────────┘
```

---

# 🔬 Methodology

The complete prediction pipeline consists of four major stages.

## 1. Text Preprocessing

Raw news articles contain noise that can negatively affect text classification.

The preprocessing pipeline cleans the input and prepares it for feature extraction.

Typical operations include:

* Text normalization
* Removal of unnecessary characters
* Stopword removal
* Lemmatization
* Tokenization / text preparation

The objective is to retain meaningful linguistic information while reducing irrelevant variation.

---

## 2. TF-IDF Feature Extraction

After preprocessing, the text is transformed into numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

TF-IDF assigns importance to words based on:

* how frequently they occur in a document
* how informative they are across the complete dataset

Conceptually:

```text
News Article
     ↓
Preprocessing
     ↓
Clean Text
     ↓
TF-IDF Vectorization
     ↓
Numerical Feature Vector
```

The fitted vectorizer is persisted as:

```text
vectorizer.pkl
```

This is important because inference must use the **same feature representation** that was used during model training.

---

# 🧠 Deep Learning Model

The project uses a **Deep Neural Network** trained on TF-IDF representations of news articles.

### Model configuration

| Component          | Details                    |
| ------------------ | -------------------------- |
| Learning approach  | Supervised Deep Learning   |
| Problem type       | Binary Text Classification |
| Feature extraction | TF-IDF                     |
| Classifier         | Deep Neural Network        |
| Training epochs    | 15                         |
| Output             | REAL / FAKE                |
| Model format       | `.h5`                      |
| Vectorizer format  | `.pkl`                     |

The trained model is stored in:

```text
model.h5
```

During inference, the application loads both:

```text
model.h5
vectorizer.pkl
```

and applies the same preprocessing + vectorization pipeline before generating a prediction.

---

# 📊 Dataset

The project was trained using a combination of:

```text
Fake.csv
True.csv
Additional Dataset
```

The combination of multiple sources is intended to provide examples of both real and fabricated news content.

However, dataset composition can strongly influence the model's behavior. A model trained on historical datasets may not generalize perfectly to modern misinformation, newly emerging topics, or intentionally adversarial content.

---

# 🧪 Model Evaluation

An ablation study was conducted to investigate the impact of different modeling decisions, including:

* Text preprocessing strategies
* Feature configurations
* Dropout settings
* Neural-network architecture choices

The final reported model achieved approximately:

## **~85% Accuracy**

> **Note:** Accuracy is dataset-dependent and should not be interpreted as a guarantee of real-world fact-checking performance.

For a production-grade system, additional evaluation should include:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Cross-validation
* Per-class performance
* Evaluation on an unseen external dataset

---

# 🖥️ Application

The project provides a browser-based interface built with **Streamlit**.

The workflow is intentionally simple:

```text
1. Open the application
        ↓
2. Enter / paste news text
        ↓
3. Submit the article
        ↓
4. NLP preprocessing
        ↓
5. TF-IDF transformation
        ↓
6. Deep-learning inference
        ↓
7. REAL / FAKE result
```

The interface also provides a prediction score/confidence estimate to help users understand the model's output.

---

# 📂 Project Structure

```text
Fake-News-Detection-Using-Deep-Learning/
│
├── app_streamlit.py       # Streamlit application
├── model.h5               # Trained deep-learning model
├── vectorizer.pkl         # Fitted TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

### File Responsibilities

| File               | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| `app_streamlit.py` | Runs the interactive prediction application     |
| `model.h5`         | Stores trained neural-network weights/model     |
| `vectorizer.pkl`   | Stores the fitted TF-IDF vectorizer             |
| `requirements.txt` | Defines required Python packages                |
| `.gitignore`       | Prevents unnecessary files from being committed |
| `README.md`        | Project documentation                           |

---

# ⚙️ Installation

## Prerequisites

Make sure the following are installed:

* Python 3.x
* pip
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/komal-alis/Fake-News-Detection-Using-Deep-Learning.git
```

Navigate into the project:

```bash
cd Fake-News-Detection-Using-Deep-Learning
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Launch the Streamlit application:

```bash
streamlit run app_streamlit.py
```

Streamlit will start a local development server and provide a browser URL.

Open the displayed URL and enter a news article to test the classifier.

---

# 🔍 Example Workflow

### Input

```text
Scientists have announced a new discovery after conducting
a large-scale study involving thousands of participants...
```

### Processing

```text
Raw Text
   ↓
Cleaning
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
TF-IDF
   ↓
Neural Network
```

### Output

```text
Prediction: REAL NEWS

Confidence: XX%
```

> The displayed result is a model prediction, not a verified statement about the factual accuracy of the article.

---

# 🧩 Why TF-IDF + Deep Learning?

TF-IDF provides a strong and interpretable baseline for representing textual information.

Combining it with a neural network allows the system to learn nonlinear relationships between the extracted textual features and the target classes.

The architecture also has a practical advantage:

```text
Lightweight Feature Extraction
            +
Deep Learning Classification
            =
Fast Local Inference
```

This makes the approach suitable for an educational project and lightweight experimentation.

---

# 🧪 Reproducibility

The repository includes the trained artifacts required for inference:

```text
model.h5
vectorizer.pkl
```

Keeping the vectorizer together with the model is critical.

A model trained using one vocabulary/TF-IDF configuration should not be paired with a newly fitted vectorizer during inference, because the resulting feature space may differ.

---

# ⚠️ Limitations

Fake-news detection is an inherently difficult NLP problem.

This implementation has several important limitations:

### 1. Dataset Dependence

The model can only learn patterns represented in its training data.

### 2. Temporal Generalization

Language, topics, websites, and misinformation strategies evolve over time.

### 3. Context Understanding

Text-based classification alone may not understand:

* sarcasm
* satire
* missing context
* manipulated quotes
* misleading headlines
* source credibility
* political or social context
* deliberately adversarial wording

### 4. Prediction ≠ Fact Checking

A neural network prediction does not establish whether a claim is objectively true.

### 5. Domain Shift

Performance may decrease when the model encounters content from domains or sources that differ significantly from its training data.

---

# 🚀 Future Roadmap

The project can be extended significantly beyond the current TF-IDF + neural-network approach.

### 🤖 Model Improvements

* [ ] Experiment with LSTM / BiLSTM
* [ ] Explore GRU architectures
* [ ] Compare CNN-based text classifiers
* [ ] Fine-tune transformer models such as BERT
* [ ] Evaluate modern sentence embeddings
* [ ] Perform systematic hyperparameter optimization

### 📚 Dataset Improvements

* [ ] Expand the training dataset
* [ ] Add newer news sources
* [ ] Improve class balance
* [ ] Introduce multilingual datasets
* [ ] Evaluate against an external unseen dataset

### 📊 Evaluation Improvements

* [ ] Add precision / recall / F1-score
* [ ] Add confusion matrix
* [ ] Add ROC-AUC
* [ ] Perform k-fold cross-validation
* [ ] Add error analysis
* [ ] Evaluate robustness against adversarial text

### 🌐 Product Improvements

* [ ] Deploy the application publicly
* [ ] Build a REST API
* [ ] Add batch prediction
* [ ] Add prediction history
* [ ] Improve UI/UX
* [ ] Add multilingual support
* [ ] Add source-level credibility analysis

### 🔎 Explainability

A future version could explain *why* a prediction was generated using techniques such as:

* SHAP
* LIME
* Feature importance
* Token-level attribution

This would make the system more transparent and useful for research.

---

# 🔐 Responsible AI Considerations

Automated misinformation detection can have real-world consequences.

A responsible system should therefore avoid presenting predictions as absolute truth.

Recommended production behavior:

```text
Model Prediction
       ↓
Confidence / Uncertainty
       ↓
Evidence Retrieval
       ↓
Source Verification
       ↓
Human Review
       ↓
Final Assessment
```

The model should be considered a **decision-support tool**, rather than an autonomous authority on factual truth.

---

# 🛠️ Technology Stack

| Technology             | Role                        |
| ---------------------- | --------------------------- |
| **Python**             | Core programming language   |
| **TensorFlow / Keras** | Deep learning               |
| **Scikit-learn**       | TF-IDF and ML utilities     |
| **NLTK**               | Natural Language Processing |
| **Pandas**             | Data processing             |
| **NumPy**              | Numerical computation       |
| **Streamlit**          | Web application             |
| **Pickle**             | Vectorizer serialization    |

---

# 📈 Project Highlights

```text
                 FAKE NEWS DETECTION
                         │
          ┌──────────────┼──────────────┐
          │              │              │
         NLP           TF-IDF      Deep Learning
          │              │              │
          └──────────────┼──────────────┘
                         │
                    Classification
                         │
                 ┌───────┴───────┐
                 │               │
              REAL NEWS       FAKE NEWS
```

### Current reported performance

**~85% accuracy**

### Core pipeline

**NLP → TF-IDF → Neural Network → Classification**

### Interface

**Streamlit Web Application**

---

# 🎓 Educational Purpose

This repository demonstrates an end-to-end machine-learning workflow:

```text
Problem Definition
       ↓
Data Preparation
       ↓
Text Preprocessing
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Application Development
       ↓
Inference
```

It is particularly useful for learning about:

* Natural Language Processing
* Text classification
* Feature engineering
* Deep learning
* Model deployment
* Streamlit application development
* Responsible AI

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve the project:

```bash
# Fork the repository

# Create a feature branch
git checkout -b feature/your-feature

# Make your changes

# Commit
git commit -m "Add your feature"

# Push
git push origin feature/your-feature
```

Then open a Pull Request with a clear description of the proposed improvement.

---

# 📄 License

This project is intended primarily for **educational and research purposes**.

Please review the repository's licensing terms and dataset licenses before redistributing or deploying the project commercially.

---

# 👩‍💻 Authors

**Astha Shrivastava**

GitHub: [@astha-alis](https://github.com/astha-alis/TruthLens-AI-Powered-Fake-News-Detection)

---

# ⭐ Support the Project

If you found this project useful for learning or research:

* ⭐ Star the repository
* 🍴 Fork it
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute through pull requests

---

## 📰 Final Note

Fake news detection is not simply a binary classification problem.

A reliable misinformation-detection platform ultimately needs to combine:

**Natural Language Processing + Machine Learning + Source Verification + Context + Evidence + Human Judgment**

This project provides the foundation for exploring that larger problem.

---

<p align="center">

**Built with Python, NLP, Deep Learning & Streamlit**

⭐ If this project helped you, consider giving it a star.

</p>
