
import ast
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="QA Chatbot System",
    page_icon="💬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "squad_train.csv"


# ============================================================
# 2. LOAD AND PREPARE DATASET
# ============================================================


@st.cache_data
def load_dataset(file_path, file_modified):
    import re
    import pandas as pd
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    required = {"question", "context"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["question"] = data["question"].fillna("").astype(str).str.strip()
    data["context"] = data["context"].fillna("").astype(str).str.strip()

    if "title" not in data.columns:
        data["title"] = "Wikipedia"

    data["title"] = data["title"].fillna("Wikipedia").astype(str)

    # Keep records with valid questions and passages.
    data = data[
        (data["question"] != "") &
        (data["context"] != "")
    ].copy()

    def extract_answer(value):
        if not isinstance(value, str) or not value.strip():
            return ""

        # Handles answers stored as NumPy array representations.
        patterns = [
            r"""array\(\s*\[\s*['"]([^'"]+)['"]""",
            r"""['"]text['"]\s*:\s*\[\s*['"]([^'"]+)['"]""",
        ]

        for pattern in patterns:
            match = re.search(pattern, value)
            if match:
                return match.group(1).strip()

        # Handles standard JSON/Python dictionary strings.
        try:
            import ast
            parsed = ast.literal_eval(value)

            if isinstance(parsed, dict):
                texts = parsed.get("text", [])
                if isinstance(texts, (list, tuple)) and texts:
                    return str(texts[0]).strip()

            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0]).strip()

        except (ValueError, SyntaxError, TypeError):
            pass

        return ""

    if "answers" in data.columns:
        data["answer"] = data["answers"].apply(extract_answer)
    else:
        data["answer"] = ""

    # IMPORTANT: Do not discard valid passages just because answer
    # extraction failed. The IR chatbot can still retrieve passages.
    data = data.reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid records found. Check question and context columns."
        )

    return data


    
def extract_answer(value):
    if not isinstance(value, str) or not value.strip():
        return ""

    # Handle the NumPy-style answer representation in your CSV.
    match = re.search(
        r"""['"]text['"]\s*:\s*array\(\s*\[\s*(['"])(.*?)\1""",
        value
    )

    if match:
        return match.group(2).strip()

    # Fallback for ordinary Python dictionaries and lists.
    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, dict):
            texts = parsed.get("text", [])
            if isinstance(texts, (list, tuple)) and texts:
                return str(texts[0]).strip()

        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0]).strip()

    except (ValueError, SyntaxError, TypeError):
        pass

    return ""


    data["answer"] = data["answers"].apply(extract_answer)

    data = data[
        data["answer"].fillna("").astype(str).str.strip() != ""
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No usable question-answer records were found in the CSV."
        )

    return data


# ============================================================
# 3. BUILD SEARCH INDEX
# ============================================================

@st.cache_resource
def build_search_index(contexts, questions):
    # Ensure the input contains valid strings.
    contexts = [
        str(value).strip()
        for value in contexts
        if value is not None and str(value).strip()
    ]

    questions = [
        str(value).strip()
        for value in questions
        if value is not None and str(value).strip()
    ]

    if not contexts:
        raise ValueError(
            "No valid passages were supplied to the search index."
        )

    if not questions:
        raise ValueError(
            "No valid questions were supplied to the search index."
        )

    # Character n-grams avoid empty vocabulary errors caused by
    # word tokenization and stop-word filtering.
    passage_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100000,
        dtype=float
    )

    question_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100000,
        dtype=float
    )

    passage_matrix = passage_vectorizer.fit_transform(contexts)
    question_matrix = question_vectorizer.fit_transform(questions)

    return (
        passage_vectorizer,
        passage_matrix,
        question_vectorizer,
        question_matrix
    )


# ============================================================
# 4. RULE-BASED CHATBOT
# ============================================================

def rule_based_response(user_text):
    text = user_text.lower().strip()

    # Normalize punctuation and whitespace.
    normalized = re.sub(r"[^a-z0-9\s]", " ", text)
    normalized = " ".join(normalized.split())
    words = set(normalized.split())

    if not normalized:
        return "Please enter a question."

    if words.intersection({"hello", "hi", "hey"}):
        return (
            "Hello! Welcome to the Rule-Based Chatbot. "
            "Ask me about this project or type 'help'."
        )

    if "good morning" in normalized:
        return "Good morning! How can I help you?"

    if "good evening" in normalized:
        return "Good evening! What would you like to know?"

    if any(term in normalized for term in [
        "who are you", "what are you", "your name"
    ]):
        return (
            "I am the Rule-Based Chatbot. I respond using "
            "predefined rules and patterns."
        )

    if "help" in words or "what can you do" in normalized:
        return (
            "I can respond to greetings and predefined questions "
            "about this project. For open-ended questions, try "
            "the IR-Based Chatbot."
        )

    if "dataset" in words or "squad" in words:
        return (
            "This project uses SQuAD, a dataset containing "
            "Wikipedia passages, questions, and annotated answers."
        )

    if "rule based" in normalized or "rule based chatbot" in normalized:
        return (
            "A rule-based chatbot follows predefined conditions "
            "and returns the response associated with a matching rule."
        )

    if "information retrieval" in normalized or "ir chatbot" in normalized:
        return (
            "An Information Retrieval chatbot searches documents "
            "or passages and retrieves information relevant to "
            "the user's question."
        )

    if "thank" in words or "thanks" in words:
        return "You're welcome!"

    if words.intersection({"bye", "goodbye"}):
        return "Goodbye! Thank you for trying the chatbot."

    return (
        "Sorry, I do not have a predefined rule for that question. "
        "Please try another question or use the IR-Based Chatbot."
    )


# ============================================================
# 5. IR-BASED CHATBOT
# ============================================================

def retrieve_answer(
    user_question,
    df,
    passage_vectorizer,
    passage_matrix,
    question_vectorizer,
    question_matrix,
    unique_contexts
):
    question = user_question.strip()

    if not question:
        return "Please enter a question.", None

    # Find the most relevant passage.
    query_vector = passage_vectorizer.transform([question])

    passage_scores = cosine_similarity(
        query_vector,
        passage_matrix
    ).ravel()

    best_passage_index = int(passage_scores.argmax())
    best_passage_score = float(passage_scores[best_passage_index])

    best_context = unique_contexts[best_passage_index]

    # Get dataset questions associated with the retrieved passage.
    matching_rows = df[
        df["context"] == best_context
    ].copy()

    if matching_rows.empty:
        return (
            "A passage was retrieved, but no annotated answers "
            "were associated with it.",
            {
                "title": "Wikipedia",
                "context": best_context,
                "matched_question": "",
                "passage_score": best_passage_score,
                "question_score": 0.0
            }
        )

    # Find the closest annotated question within this passage.
    matching_indices = matching_rows.index.to_list()

    query_question_vector = question_vectorizer.transform([question])

    question_scores = cosine_similarity(
        query_question_vector,
        question_matrix[matching_indices]
    ).ravel()

    best_local_index = int(question_scores.argmax())
    best_row = matching_rows.iloc[best_local_index]
    best_question_score = float(question_scores[best_local_index])

    source = {
        "title": str(best_row["title"]),
        "context": best_context,
        "matched_question": str(best_row["question"]),
        "passage_score": best_passage_score,
        "question_score": best_question_score
    }

    # Avoid presenting unrelated answers when retrieval confidence is low.
    if best_passage_score < 0.05:
        return (
            "I could not find a sufficiently relevant passage. "
            "Please rephrase your question or try another topic.",
            None
        )

    if best_question_score >= 0.10:
        return str(best_row["answer"]), source

    return (
        "I found a potentially relevant passage, but could not "
        "confidently match your question to an annotated answer. "
        "Please inspect the retrieved passage below.",
        source
    )


# ============================================================
# 6. LOAD DATA AND CREATE INDEX
# ============================================================

st.title("Question Answering Chatbot System")
st.write(
    "Compare a Rule-Based Chatbot with an Information Retrieval "
    "(IR) Chatbot using the SQuAD question-answer dataset."
)

try:
    dataset = load_dataset(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime
    )

    # Keep one copy of each distinct passage.
    unique_contexts = dataset["context"].drop_duplicates().tolist()

    # Ensure the order of unique passages matches the index.
    if not unique_contexts:
        st.error("No valid passages were found in the dataset.")
        st.stop()

    unique_questions = dataset["question"].tolist()

    if not unique_questions:
        st.error("No valid questions were found in the dataset.")
        st.stop()

    with st.spinner("Building the search index. Please wait..."):
        (
            passage_vectorizer,
            passage_matrix,
            question_vectorizer,
            question_matrix
        ) = build_search_index(
            unique_contexts,
            unique_questions
        )

except Exception as error:
    st.error(f"Unable to load the dataset or build the search index: {error}")
    st.info(
        "Verify that data/squad_train.csv exists and contains "
        "non-empty question and context columns."
    )
    st.stop()

st.success("Dataset loaded and search index built successfully.")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("QA Records", f"{len(dataset):,}")
metric2.metric("Unique Passages", f"{len(unique_contexts):,}")
metric3.metric("Search Questions", f"{len(unique_questions):,}")

rule_tab, ir_tab, evaluation_tab = st.tabs([
    "Rule-Based Chatbot",
    "IR-Based Chatbot",
    "Dataset & Evaluation"
])


# ============================================================
# 7. RULE-BASED CHAT INTERFACE
# ============================================================

with rule_tab:
    st.subheader("Rule-Based Chatbot")
    st.caption("Uses predefined rules and responses.")

    if "rule_messages" not in st.session_state:
        st.session_state.rule_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the Rule-Based Chatbot. "
                    "Try saying hello or asking about the dataset."
                )
            }
        ]

    for message in st.session_state.rule_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(
        "Ask the Rule-Based Chatbot...",
        key="rule_chat_input"
    )

    if prompt:
        st.session_state.rule_messages.append(
            {"role": "user", "content": prompt}
        )

        response = rule_based_response(prompt)

        st.session_state.rule_messages.append(
            {"role": "assistant", "content": response}
        )

        st.rerun()

    if st.button("Clear Rule-Based Chat", key="clear_rule"):
        st.session_state.rule_messages = []
        st.rerun()


# ============================================================
# 8. IR-BASED CHAT INTERFACE
# ============================================================

with ir_tab:
    st.subheader("IR-Based Chatbot")
    st.caption(
        "Uses TF-IDF to retrieve a relevant passage and match "
        "your question against annotated dataset questions."
    )

    if "ir_messages" not in st.session_state:
        st.session_state.ir_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Ask a question about a topic covered "
                    "in the SQuAD dataset."
                )
            }
        ]

    for message in st.session_state.ir_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            source = message.get("source")

            if source:
                st.caption(
                    f"Passage similarity: {source['passage_score']:.3f} | "
                    f"Question similarity: {source['question_score']:.3f}"
                )

                with st.expander("View retrieved passage"):
                    st.write("**Title:**", source["title"])

                    if source["matched_question"]:
                        st.write(
                            "**Matched dataset question:**",
                            source["matched_question"]
                        )

                    st.write("**Passage:**")
                    st.write(source["context"])

    prompt = st.chat_input(
        "Ask the IR-Based Chatbot...",
        key="ir_chat_input"
    )

    if prompt:
        st.session_state.ir_messages.append(
            {"role": "user", "content": prompt}
        )

        answer, source = retrieve_answer(
            prompt,
            dataset,
            passage_vectorizer,
            passage_matrix,
            question_vectorizer,
            question_matrix,
            unique_contexts
        )

        st.session_state.ir_messages.append({
            "role": "assistant",
            "content": answer,
            "source": source
        })

        st.rerun()

    if st.button("Clear IR-Based Chat", key="clear_ir"):
        st.session_state.ir_messages = []
        st.rerun()


# ============================================================
# 9. DATASET AND EVALUATION INTERFACE
# ============================================================

with evaluation_tab:
    st.subheader("Dataset Overview")

    st.write(
        "The dataset contains Wikipedia passages with questions "
        "and annotated reference answers."
    )

    st.dataframe(
        dataset[["title", "question", "answer", "context"]].head(10),
        use_container_width=True
    )

    st.subheader("Try a dataset question")

    sample_questions = (
        dataset["question"].drop_duplicates().head(100).tolist()
    )

    selected_question = st.selectbox(
        "Select a question",
        sample_questions
    )

    if st.button("Test IR retrieval"):
        answer, source = retrieve_answer(
            selected_question,
            dataset,
            passage_vectorizer,
            passage_matrix,
            question_vectorizer,
            question_matrix,
            unique_contexts
        )

        expected = dataset[
            dataset["question"] == selected_question
        ].iloc[0]["answer"]

        st.write("**Question:**", selected_question)
        st.write("**IR chatbot answer:**", answer)
        st.write("**Reference answer:**", expected)

        if source:
            with st.expander("Inspect retrieved evidence"):
                st.write(source["context"])

    st.warning(
        "This section demonstrates retrieval; it does not calculate "
        "formal Exact Match or F1 evaluation metrics. For a fair "
        "evaluation, test on a separate SQuAD validation dataset."
    )
