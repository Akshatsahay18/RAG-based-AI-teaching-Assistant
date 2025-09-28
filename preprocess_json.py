import requests#to make HTTP POST requests to Ollama’s API (/api/embed) for embeddings.
import os#To interact with the filesystem. Used earlier for os.listdir("jsons") to loop through JSON files.
import json#To read/write JSON files that contain text chunks.
import numpy as np#For numerical operations on embeddings (e.g., converting lists to arrays).
import pandas as pd#For creating and manipulating a DataFrame of chunks + embeddings.
from sklearn.metrics.pairwise import cosine_similarity#To compute semantic similarity between embeddings (for search / retrieval).
import joblib#To save and load large Python objects (like your embeddings.joblib DataFrame with vectors).Faster and more memory-efficient than CSV/JSON for this use case.

# Function to create embeddings using Ollama
def create_embedding(text_list):
    # Send POST request to Ollama's embedding endpoint
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",      # Model to use for embeddings
        "input": text_list      # List of input texts to embed
    })

    # Extract embeddings from response
    embedding = r.json()["embeddings"] 
    return embedding


# List all JSON files in the "jsons" folder
jsons = os.listdir("jsons")  
my_dicts = []   # Will hold processed chunks with metadata + embeddings
chunk_id = 0    # Unique counter for chunk IDs across all files
"""Suppose you have one file in jsons/ called doc1.json:
{
  "chunks": [
    {"text": "Deep learning is a subset of machine learning."},
    {"text": "Neural networks are inspired by the human brain."}
  ]}
And another file doc2.json:
{
  "chunks": [
    {"text": "Computer vision helps machines understand images."
    }]}
jsons = ["doc1.json", "doc2.json"]"""

# Loop over each JSON file in the folder
for json_file in jsons:
    # Open and load JSON file
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
        """for first content={
  "chunks": [
    {"text": "Deep learning is a subset of machine learning."},
    {"text": "Neural networks are inspired by the human brain."}
  ]
}"""
    print(f"Creating Embeddings for {json_file}")

    # Generate embeddings for all text chunks in the file
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    """embeddings = [
  [0.12, -0.34, 0.56],   # embedding for "Deep learning..."
  [0.44, 0.02, -0.11]    # embedding for "Neural networks..."
]"""
       
    # Attach embeddings and chunk_id to each chunk
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id          # Assign unique chunk ID
        chunk['embedding'] = embeddings[i]    # Add embedding vector
        chunk_id += 1                         # Increment chunk counter
        my_dicts.append(chunk)                # Add chunk to list
        """my_dict=[
  {first chunk all data
    "text": "Deep learning is a subset of machine learning.",
    "chunk_id": 0,
    "embedding": [0.12, -0.34, 0.56]
  },
  {
    "text": "Neural networks are inspired by the human brain.",
    "chunk_id": 1,
    "embedding": [0.44, 0.02, -0.11]
  }
]
.......middle chunks data
[
  {last chunk data
    "text": "Computer vision helps machines understand images.",
    "chunk_id": 2,
    "embedding": [-0.21, 0.67, 0.15]
  }
]"""

# Convert all chunks into a pandas DataFrame
df = pd.DataFrame.from_records(my_dicts)

# Save the DataFrame to disk for later use
joblib.dump(df, 'embeddings.joblib')

"""DF is
text	chunk_id	embedding
Deep learning is a subset of machine learning.	0	[0.12, -0.34, 0.56]
Neural networks are inspired by the human brain.	1	[0.44, 0.02, -0.11]
Computer vision helps machines understand images.	2	[-0.21, 0.67, 0.15]
"""

"""
"""