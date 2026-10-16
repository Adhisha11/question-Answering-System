# Question Answering Chatbot System

## Overview

The Question Answering Chatbot System is a Python-based web application that demonstrates two approaches to answering user queries: Rule-Based Chatbot and Information Retrieval (IR)-Based Chatbot.

The application uses the **Stanford Question Answering Dataset (SQuAD)** to retrieve relevant passages and answers from a real-world question-answering dataset. It is built using Streamlit and Scikit-learn.

## Features

- **Rule-Based Chatbot:** Responds to predefined greetings, common queries, and keywords using programmed rules.
- **IR-Based Chatbot:** Searches the SQuAD dataset to identify relevant questions and passages using TF-IDF similarity.
- **Dataset Exploration:** Displays information about the dataset used by the application.
- **Evaluation and Comparison:** Provides a way to compare the behavior of rule-based and information-retrieval approaches.
- **Interactive Web Interface:** Allows users to interact with both chatbots through a Streamlit application.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- SQuAD Dataset

## Dataset

This project uses the Stanford Question Answering Dataset (SQuAD), which contains questions, corresponding passages, and annotated answers.

Dataset source: https://huggingface.co/datasets/rajpurkar/squad

The dataset CSV is excluded from this repository to avoid uploading a large data file. It can be downloaded separately using the provided dataset download script.

## Project Structure

```text
qa_chatbot/
├── app.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── squad_train.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/question-answering-chatbot-system.git
cd question-answering-chatbot-system
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Run the dataset download script:

```bash
python download_dataset.py
```

Ensure the downloaded CSV is saved as `data/squad_train.csv`, as expected by the application.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## How It Works

### Rule-Based Chatbot

The rule-based chatbot checks user input against predefined rules and patterns. It is suitable for simple, predictable queries but cannot answer arbitrary questions unless corresponding rules are implemented.

### Information Retrieval-Based Chatbot

The IR-based chatbot uses TF-IDF vectorization and cosine similarity to find relevant questions or passages from the SQuAD dataset. It uses textual similarity to select relevant content rather than generating answers like a large language model.

The retrieved result may not always be correct, particularly when a query uses different wording from the dataset or when several passages contain similar terms.

## Limitations

- The rule-based chatbot is limited to the rules implemented in the application.
- The IR-based chatbot depends on the coverage and quality of the dataset.
- TF-IDF measures lexical similarity and does not fully understand the meaning or context of a question.
- Retrieval results should be verified when answer accuracy is important.

## Future Enhancements

- Integrate semantic embeddings for improved retrieval.
- Implement a reranking model to select more relevant passages.
- Evaluate retrieval performance using metrics such as Precision@K, Recall@K, and Mean Reciprocal Rank.
- Improve answer extraction and confidence-based fallback responses.
- Add a larger set of test questions and compare both chatbot approaches systematically.

## License

This project is intended for educational and academic purposes. Add a suitable open-source license if you decide to distribute the project under specific reuse terms.
