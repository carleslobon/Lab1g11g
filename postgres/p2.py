from connect import connect
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#Connecting to the database
connection = connect(load_config())
cursor = connection.cursor()

#Creating the stored procedure
create_procedure_sql = """
CREATE OR REPLACE FUNCTION get_closest_sentences(input_embedding float[])
RETURNS TABLE(sentence text, distance_type text, distance_value double precision) AS $$
BEGIN
    RETURN QUERY
    (
        SELECT 
            s.sentence,
            'euclidean' AS distance_type,
            sqrt(sum(pow(s.embedding[i] - input_embedding[i], 2))) AS distance_value
        FROM 
            sentences s,
            generate_series(1, array_length(s.embedding, 1)) AS i
        GROUP BY s.sentence
        ORDER BY distance_value ASC
        LIMIT 1
    )
    UNION ALL
    (
        SELECT 
            s.sentence,
            'manhattan' AS distance_type,
            sum(abs(s.embedding[i] - input_embedding[i])) AS distance_value
        FROM 
            sentences s,
            generate_series(1, array_length(s.embedding, 1)) AS i
        GROUP BY s.sentence
        ORDER BY distance_value ASC
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql;
"""

try:
    cursor.execute(create_procedure_sql)
    print("Stored procedure created successfully.")
except Exception as e:
    print(f"Error while creating the stored procedure: {e}")

connection.commit()

#Sentences we chose
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

#Conver the 10 sentences to embeddings
embeddings = []
for sentence in sentences_to_process:
    encoded_input = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings.append(embedding)

#Compute the top-2 most similar sentences (among all other sentences) for each of them using two different distance metrics
for embedding in embeddings:
    #Convert each sentence to Float[]
    float_embedding = embedding.numpy().flatten().tolist()
    cursor.execute("SELECT * FROM get_closest_sentences(ARRAY[%s]::float[])", (float_embedding,))
    similar_sentences = cursor.fetchall()
    
    print("Similar sentences:")
    for sentence in similar_sentences:
        print(sentence)

#Close connection
cursor.close()
connection.close()
