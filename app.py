import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search


@st.cache
def read_data(get_posts="data/posts.json"):
    """Read the data from S3."""
    return pd.read_json(get_posts)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="models/stack_faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def main():
    # Load data and models
    df = pd.read_json("data/posts.json")
    # df['date'] = pd.to_datetime(df['creation_date'])
    # answers = pd.read_json("data/answers.json")
    model = load_bert_model()
    faiss_index = load_faiss_index()
    st.markdown("## StackOverflow search")
    st.markdown("### Vector-based searches with Sentence Transformers and Faiss")
    st.write("StackOverflow reference includes " + str(df.shape[0]) + " posts")

    # User search
    user_input = st.text_area("Search box", "natural language processing")

    # Filters
    st.sidebar.markdown("**Filters**")
    # filter_year = st.sidebar.slider("Publication year", 2010, 2021, (2010, 2021), 1)
    filter_answers = st.sidebar.slider("Answers", 0, 25, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 100, 50)

    # Fetch results
    if user_input:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on year
        frame = df[
            # (data.year >= filter_year[0])
            # & (data.year <= filter_year[1])
            (df.answer_count >= filter_answers)
        ]
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(frame.id):
                f = frame[(frame.id == id_)]
            else:
                continue

            st.write(
                f"""**{f.iloc[0].title}**  
            **Answers**: {f.iloc[0].answer_count}  
            **Creation UNIX date**: {f.iloc[0].creation_date}  
            """
            )
            st.markdown(f.iloc[0].body, unsafe_allow_html=True)

            # a = answers[
            #     (answers.parent_id == f.iloc[0].id)
            # ]
            # if f.iloc[0].answer_count > 0:
            #     st.markdown("## Answers")
            #     st.markdown(a.body, unsafe_allow_html=True)


if __name__ == "__main__":
    main()