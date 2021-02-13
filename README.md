# Building a semantic search engine with Transformers and Faiss
This post was created with reference from:
- [How to build a semantic search engine with Transformers and Faiss](https://kstathou.medium.com/how-to-build-a-semantic-search-engine-with-transformers-and-faiss-dcbea307a0e8?source=friends_link&sk=6974c79b86e2f257c32f77d49583a524)
- [How to deploy a machine learning model on AWS Elastic Beanstalk with Streamlit and Docker](https://kstathou.medium.com/how-to-deploy-a-semantic-search-engine-with-streamlit-and-docker-on-aws-elastic-beanstalk-42ddce0422f3?source=friends_link&sk=dcc7bbf8d172f2cd18aefcdf0c2c6b49)

Check out the blogs if you want to learn how to create a semantic search engine with Sentence Transformers and Faiss.  

You can [run the notebook on Google Colab](https://colab.research.google.com/github/kstathou/vector_engine/blob/master/notebooks/001_vector_search.ipynb) and leverage the free GPU to speed up the computation!

## Data

Data has been replaced with sample of StackOverflow: see https://gist.github.com/wesslen/1bb5a860723e74fdb85295c6c913783c

## How to deploy the Streamlit app locally with Docker ##
Assuming docker is running on your machine and you have a docker account, do the following:
- Build the image

``` bash
docker build -t <USERNAME>/<YOUR_IMAGE_NAME> .
```

- Run the image

``` bash
docker run -p 8501:8501 <USERNAME>/<YOUR_IMAGE_NAME>
```

- Open your browser and go to `http://localhost:8501/`

## How to modify data set used / create index

``` python
import pandas as pd

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer

# Used to create and store the Faiss index.
import faiss
import numpy as np
import pickle
from pathlib import Path

# Used to do vector searches and display the results.
from vector_engine.utils import vector_search, id2details

# assume file is in data/posts.json
pd.read_json("data/posts.json")

# Instantiate the sentence-level DistilBERT
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Check if GPU is available and use it
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
print(model.device)

# Convert abstracts to vectors
embeddings = model.encode(posts.title.to_list(), show_progress_bar=True)

# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, posts.id.values)

print(f"Number of vectors in the Faiss index: {index.ntotal}")

# Define project base directory
# Change the index from 1 to 0 if you run this on Google Colab
project_dir = Path('notebooks').resolve().parents[1]
print(project_dir)

# Serialise index and store it as a pickle
with open(f"{project_dir}/models/stack_faiss_index.pickle", "wb") as h:
    pickle.dump(faiss.serialize_index(index), h)

```

Then update the `app.py` file to modify the data being loaded (update according columns) and the new index (pickle file).