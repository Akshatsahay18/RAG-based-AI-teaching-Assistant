# ---------------------------------------------
# Importing required libraries
# ---------------------------------------------
import pandas as pd                         # For working with data tables (loading embeddings, manipulation)
from sklearn.metrics.pairwise import cosine_similarity  # For similarity calculations between vectors
import numpy as np                           # For numerical operations on embeddings
import joblib                                # For saving and loading Python objects (like DataFrames)
import requests                              # To send HTTP requests to Ollama API (local LLM server)

# ---------------------------------------------
# Function: Create Embeddings
# ---------------------------------------------
def create_embedding(text_list):
    """
    Sends a list of text strings to the Ollama local embedding endpoint
    to generate vector embeddings using the BGE-M3 model.
    
    Args:
        text_list (list): A list of text strings to embed.

    Returns:
        list: A list of embedding vectors where each text is represented
              as a list of floating-point numbers.
    """
    
    # POST request to Ollama's /api/embed endpoint
    # The model used here is 'bge-m3', a multilingual embedding model.
    # This converts text into high-dimensional vectors that capture meaning.
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",    # Embedding model name
        "input": text_list    # List of texts to generate embeddings for
    })

    # Parse the JSON response and extract embeddings
    embedding = r.json()["embeddings"] 
    return embedding
"""output---{
  "model": "bge-m3",
  "created_at": "2025-09-28T17:45:30.123Z",
  "embeddings": [
    [0.0123, -0.8854, 0.4321, ..., 0.0765],
    [0.2234, -0.6743, 0.1890, ..., -0.4521]
  ],
  "done": true}
"""
# ---------------------------------------------
# Function: Inference using a Generative Model
# ---------------------------------------------
def inference(prompt):
    """
    Sends a prompt to a generative model running locally on Ollama
    to get a natural language response.
    Args:
        prompt (str): The full input prompt including context and user query.

    Returns:
        dict: JSON response from the LLM containing the generated text and metadata.
    """
    # POST request to Ollama's /api/generate endpoint
    # This is for text generation, not embeddings.
    r = requests.post("http://localhost:11434/api/generate", json={
        # You can swap between models like deepseek-r1 or llama3.2
        # "model": "deepseek-r1",
        "model": "llama3.2",   # Llama 3.2 model is currently being used
        "prompt": prompt,      # The full structured prompt to pass to the model
        "stream": False        # 'False' means we want the entire response at once (not streaming chunks)
    })

    # Convert HTTP response to a Python dictionary
    response = r.json()

    # Print raw response for debugging (optional)
    print(response)

    return response
"""input---I am teaching web development in my Sigma course.

Here are related video chunks:
[{"title": "CSS Basics", "number": "03", "start": 30.0, "end": 60.0, "text": "Create a CSS file and link it..."}]

User question: "How do I style my webpage using CSS?"

Answer by pointing to the correct video and timestamp.
"""
"""output---{
  "model": "llama3.2",
  "created_at": "2025-09-28T16:40:00Z",
  "response": "Check Video 03 between 30 and 60 seconds where I explain how to create and link a CSS file."
}
"""
# ---------------------------------------------
# Step 1: Load Precomputed Embeddings from Disk
# ---------------------------------------------
# The embeddings.joblib file was previously created by processing JSONs
# containing video transcripts and generating embeddings for each chunk.
df = joblib.load('embeddings.joblib')

# ---------------------------------------------
# Step 2: Accept a User Query
# ---------------------------------------------
incoming_query = input("Ask a Question: ")

# Convert the user query into an embedding
# Since create_embedding expects a list, we wrap the query in [ ]
# [0] extracts the first embedding since we're only sending one query
question_embedding = create_embedding([incoming_query])[0]
"""input--->Ask a Question: How do I link a CSS file to HTML?
incoming_query == "How do I link a CSS file to HTML?"
create_embedding-->{
  "model": "bge-m3",
  "created_at": "2025-09-28T12:34:56Z",
  "embeddings": [
    [0.0123, -0.4456, 0.2210, ..., 0.5544]
  ],
  "done": true
}
output--->question_embedding = [0.0123, -0.4456, 0.2210, -0.3341, 0.1092, 0.5567, -0.1182, 0.1058, -0.4412, 0.0321, ...]"""

# ---------------------------------------------
# Step 3: Compare Query with Video Chunks using Cosine Similarity
# ---------------------------------------------
# Cosine similarity tells us how "close in meaning" two embeddings are.
# Higher values mean more similar.
#
# 1. df['embedding'] is a column where each cell contains a list of floats.
# 2. np.vstack stacks all these lists into a 2D array shaped like (num_chunks, embedding_dim).
# 3. cosine_similarity compares each chunk vector with the query vector.

similarities = cosine_similarity(
    np.vstack(df['embedding']),      # Matrix of all chunk embeddings
    [question_embedding]             # The query embedding as a 2D array
).flatten()                          # Flatten result to get a 1D array of similarity scores
"""
# Example embeddings
embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]
question_embedding = [0.2, 0.3, 0.4]"""

"""Similarities: [0.98198051 0.99385869 0.99819089]"""

# ---------------------------------------------
# Step 4: Select Top N Most Relevant Chunks
# ---------------------------------------------
top_results = 5  # Number of chunks to retrieve for context
"""similarities = [0.55, 0.72, 0.88, 0.93, 0.40, 0.60]
"""

# Sort similarities in descending order, take top 5 indices
max_indx = similarities.argsort()[::-1][0:top_results]
"""similarities.argsort() → [4, 0, 5, 1, 2, 3]
[::-1] → [3, 2, 1, 5, 0, 4]
max_indx = [3, 2, 1, 5, 0]

"""

# Retrieve the actual top matching rows from DataFrame
new_df = df.loc[max_indx]
"""index	title	number	start	end	text	similarity
2	JavaScript	03	30.0	60.0	"Handling JS events..."	0.96
4	CSS Styling	02	15.0	30.0	"Adding style to HTML..."	0.88
1	HTML Basics	01	0.0	10.0	"HTML is the structure..."	0.85
0	HTML Basics	01	10.0	20.0	"Tags and elements..."	0.72
3	Miscellaneous	04	40.0	50.0	"Unrelated content..."	0.33
"""
# ---------------------------------------------
# Step 5: Build Final Prompt for LLM
# ---------------------------------------------
# Combine retrieved chunks and user query into a structured prompt.
# The LLM will read the subtitles and user question, then respond with
# a human-friendly answer pointing to relevant videos and timestamps.

prompt = f'''
I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, it's just for you) where and how much content is taught in which video (specify the video number and timestamp), and guide the user to go to that particular video.
If the user asks an unrelated question, politely tell them that you can only answer questions related to the course.'''


"""expected pompt--"I am teaching web development in my Sigma web development course. 
Here are video subtitle chunks containing video title, video number, 
start time in seconds, end time in seconds, and the text at that time:

[{"title":"CSS Basics","number":"03","start":30.0,"end":60.0,"text":"Create a style.css file and link it using the `<link>` tag in HTML."},
{"title":"HTML Head","number":"02","start":12.0,"end":22.0,"text":"The `<link>` tag should be placed inside the `<head>` section of your HTML document."},
{"title":"CSS Selectors","number":"04","start":0.0,"end":20.0,"text":"Selectors are used to target elements for styling."}]

---------------------------------
"How do I link a CSS file to my HTML page?"
User asked this question related to the video chunks, you have to answer in a human way 
(don't mention the above format, it's just for you) where and how much content is taught 
in which video (specify the video number and timestamp), and guide the user to go to that 
particular video.

If the user asks an unrelated question, politely tell them that you can only answer 
questions related to the course.
"""
"""output---To link a CSS file to your HTML page, start by watching Video 02 between 12s and 22s,
where I explain how to place the `<link>` tag inside the `<head>` section.
Then, check Video 03 from 30s to 60s, where I demonstrate how to create a separate
style.css file and properly link it to your HTML document.
If you want to learn more about targeting elements, go to Video 04 between 0s and 20s,
where I explain CSS selectors.

"""
# ---------------------------------------------
# Step 6: Save Prompt for Debugging
# ---------------------------------------------
# This helps you see exactly what is being sent to the model
# Useful for prompt engineering and troubleshooting.
with open("prompt.txt", "w") as f:
    f.write(prompt)
    """prompt.txt--I am teaching web development in my Sigma web development course. 
Here are video subtitle chunks containing video title, video number, 
start time in seconds, end time in seconds, and the text at that time:
[{"title":"HTML Head","number":"02","start":12.0,"end":22.0,"text":"The `<link>` tag should be placed in the `<head>` section of your HTML."},
{"title":"CSS Basics","number":"03","start":30.0,"end":60.0,"text":"Create a style.css file and link it using the `<link>` tag."},
{"title":"CSS Selectors","number":"04","start":0.0,"end":20.0,"text":"Selectors are used to target specific HTML elements."}]
---------------------------------
"How do I link a CSS file to my HTML page?"
User asked this question related to the video chunks, you have to answer in a human way 
(don't mention the above format, it's just for you) where and how much content is taught 
in which video (specify the video number and timestamp), and guide the user to go to that 
particular video.
If the user asks an unrelated question, politely tell them that you can only answer 
questions related to the course.
"""

# ---------------------------------------------
# Step 7: Run Inference (Generate Answer)
# ---------------------------------------------
# Send the final prompt to the LLM and get the generated response
response = inference(prompt)["response"]

# Print the generated answer to console
print(response)
"""response--To link a CSS file to your HTML page, first check Video 02 between 12s and 22s,
where I explain how to place the <link> tag inside the <head> section.
Then, go to Video 03 between 30s and 60s, where I show you how to create a separate
style.css file and correctly link it to your HTML.
For more about targeting elements, you can also check Video 04 between 0s and 20s,
where I explain CSS selectors.
"""

# Save the response to a text file for record-keeping
with open("response.txt", "w") as f:
    f.write(response)
"""response.txt--To link a CSS file to your HTML page, first check Video 02 between 12s and 22s,
where I explain how to place the <link> tag inside the <head> section.
Then, go to Video 03 between 30s and 60s, where I show you how to create a separate
style.css file and correctly link it to your HTML.
For more about targeting elements, you can also check Video 04 between 0s and 20s,
where I explain CSS selectors.
"""