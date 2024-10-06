from connect import connect
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import psycopg2
import time
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Connecting to the database
connection = connect(load_config())
cursor = connection.cursor()

# Read the sentences from the database
cursor.execute("SELECT id, sentence FROM sentences")
sentences = cursor.fetchall()
ids = [row[0] for row in sentences]
sentences = [row[1] for row in sentences]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

group_sz = 500

# List to store update times
update_times = []

for i in range(0, len(sentences), group_sz):
    batch = sentences[i:i + group_sz]
    batch_ids = ids[i:i + group_sz]
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    start_time = time.time()
    for id, embedding in zip(batch_ids, embeddings):
        # Convert the embedding to numpy array and then to binary
        float_embedding = embedding.numpy().flatten().tolist()
        
        # Update the sentence with its embedding
        cursor.execute("UPDATE sentences SET embedding = ARRAY[%s]::FLOAT[] WHERE id = %s", (float_embedding, id))
    end_time = time.time()
    
    # Append the time taken for this batch
    update_times.append(end_time - start_time)

connection.commit()

# Close connection
cursor.close()
connection.close()

# Convert to numpy array and compute statistics
update_times = np.array(update_times)
min_time = np.min(update_times)
max_time = np.max(update_times)
mean_time = np.mean(update_times)
std_dev_time = np.std(update_times)

# Print the results
print(f"Minimum update time: {min_time} seconds")
print(f"Maximum update time: {max_time} seconds")
print(f"Average update time: {mean_time} seconds")
print(f"Standard deviation of update time: {std_dev_time} seconds")