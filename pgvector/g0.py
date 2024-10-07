from datasets import load_dataset
import time
from connect import connect
from config import load_config
import numpy as np

# Conneting to the database
connection = connect(load_config())
cursor = connection.cursor()

# Creating the table
cursor.execute("DROP TABLE IF EXISTS sentences_pg")
cursor.execute("""CREATE TABLE sentences_pg
               (id SERIAL PRIMARY KEY,
               sentence TEXT,
               embedding VECTOR(384))""")
connection.commit()

# Charging the sentences
ds = load_dataset("williamkgao/bookcorpus100mb")
sentences = ds['train']['text'][:10000]

insertion_times = []

for sentence in sentences:
    start_time = time.time()
    cursor.execute("INSERT INTO sentences_pg (sentence) VALUES (%s)", (sentence,))
    connection.commit()
    end_time = time.time()
    insertion_times.append(end_time - start_time)

# Close connection
cursor.close()
connection.close()

min_time = np.min(insertion_times)
max_time = np.max(insertion_times)
avg_time = np.mean(insertion_times)
std_dev_time = np.std(insertion_times)

print(f"Minimum insertion time: {min_time} seconds")
print(f"Maximum insertion time: {max_time} seconds")
print(f"Average insertion time: {avg_time} seconds")
print(f"Standard deviation of insertion time: {std_dev_time} seconds")
