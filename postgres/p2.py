from connect import connect
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosine_similarity(embedding1, embedding2):
    dot_product = torch.sum(embedding1 * embedding2)
    norm1 = torch.sqrt(torch.sum(embedding1 ** 2))
    norm2 = torch.sqrt(torch.sum(embedding2 ** 2))
    return dot_product / (norm1 * norm2)

# Connecting to the database
connection = connect(load_config())
cursor = connection.cursor()

# Sentences we chose
sentences_to_process = [
    "usually , he would be tearing around the living room , playing with his toys .",
    "but just one look at a minion sent him practically catatonic .",
    "that had been megan 's plan when she got him dressed earlier .",
    "he 'd seen the movie almost by mistake , considering he was a little young for the pg cartoon , but with older cousins , along with her brothers , mason was often exposed to things that were older .",
    "she liked to think being surrounded by adults and older kids was one reason why he was a such a good talker for his age .",
    "`` are n't you being a good boy ? ''",
    "she said .",
    "mason barely acknowledged her .",
    "instead , his baby blues remained focused on the television .",
    "`` beau ? ''"
]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Convert the 10 sentences to embeddings
embeddings = []
for sentence in sentences_to_process:
    encoded_input = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings.append((sentence, embedding))

# List to store computation times
computation_times = []

# Compute the top-2 most similar sentences (among all other sentences) for each of them using two different distance metrics
for original_sentence, embedding in embeddings:
    float_embedding = embedding.numpy().flatten().tolist()
    
    cursor.execute("SELECT sentence, embedding FROM sentences")
    all_sentences = cursor.fetchall()
    
    closest_euclidean = None
    closest_cosine = None
    min_euclidean_distance = float('inf')
    max_cosine_similarity = -float('inf')
    
    start_time = time.time()
    
    for db_sentence, db_embedding in all_sentences:
        if db_sentence == original_sentence:
            continue
        
        db_embedding = torch.tensor(db_embedding)
        
        euclidean_distance = torch.sqrt(torch.sum((embedding - db_embedding) ** 2)).item()
        similarity_cosine = cosine_similarity(embedding, db_embedding).item()
        
        if euclidean_distance < min_euclidean_distance:
            min_euclidean_distance = euclidean_distance
            closest_euclidean = (db_sentence, euclidean_distance)
        
        if similarity_cosine > max_cosine_similarity:
            max_cosine_similarity = similarity_cosine
            closest_cosine = (db_sentence, similarity_cosine)
    
    end_time = time.time()
    computation_times.append(end_time - start_time)
    
    print(f"For the sentence: \"{original_sentence}\"")
    print(f"Closest (Euclidean): \"{closest_euclidean[0]}\" with distance {closest_euclidean[1]}")
    print(f"Closest (Cosine Similarity): \"{closest_cosine[0]}\" with similarity {closest_cosine[1]}")
    print()

computation_times = np.array(computation_times)
min_time = np.min(computation_times)
max_time = np.max(computation_times)
mean_time = np.mean(computation_times)
std_dev_time = np.std(computation_times)

print(f"Minimum computation time: {min_time} seconds")
print(f"Maximum computation time: {max_time} seconds")
print(f"Average computation time: {mean_time} seconds")
print(f"Standard deviation of computation time: {std_dev_time} seconds")

# Close connection
cursor.close()
connection.close()