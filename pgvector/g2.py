from connect import connect
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
computation_times = []

for sentence in sentences_to_process:
    encoded_input = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1).numpy().flatten().tolist()

    # Convert the embedding to a list of floats
    float_embedding = np.array(embedding).tolist()

    start_time = time.time()
    
    cursor.execute("""
        SELECT sentence, embedding <-> %s::vector AS euclidean_distance, embedding <#> %s::vector AS cosine_similarity
        FROM sentences_pg
        WHERE sentence <> %s
        ORDER BY euclidean_distance ASC
        LIMIT 2
        """, (float_embedding, float_embedding, sentence))


    closest_sentences = cursor.fetchall()

    end_time = time.time()
    computation_times.append(end_time - start_time)
    
    print(f"For the sentence: \"{sentence}\":")
    for closest_sentence, euclidean_distance, cosine_similarity in closest_sentences:
        print(f"Closest sentence: \"{closest_sentence}\"")
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
