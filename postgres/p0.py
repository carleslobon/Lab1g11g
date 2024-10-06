from datasets import load_dataset
import time
from connect import connect
from config import load_config
import numpy as np

# Connecting to the database
connection = connect(load_config())
cursor = connection.cursor()

# Creating the table
cursor.execute("DROP TABLE IF EXISTS sentences")
cursor.execute("""CREATE TABLE sentences
               (id SERIAL PRIMARY KEY,
               sentence TEXT,
               embedding FLOAT[])""")
connection.commit()

# Charging the sentences
ds = load_dataset("williamkgao/bookcorpus100mb")
sentences = ds['train']['text'][:10000]

# List to store insertion times
insertion_times = []

# Inserting to the database and calculing time
for sentence in sentences:
    start_time = time.time()
    cursor.execute("INSERT INTO sentences (sentence) VALUES (%s)", (sentence,))
    connection.commit()
    end_time = time.time()
    insertion_times.append(end_time - start_time)

# Close connection
cursor.close()
connection.close()

# Convert to numpy array and compute statistics
insertion_times = np.array(insertion_times)
min_time = np.min(insertion_times)
max_time = np.max(insertion_times)
mean_time = np.mean(insertion_times)
std_dev_time = np.std(insertion_times)

# Print the results
print(f"Minimum insertion time: {min_time} seconds")
print(f"Maximum insertion time: {max_time} seconds")
print(f"Average insertion time: {mean_time} seconds")
print(f"Standard deviation of insertion time: {std_dev_time} seconds")