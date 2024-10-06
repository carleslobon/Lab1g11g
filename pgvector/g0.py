from datasets import load_dataset
import time
from connect import connect
from config import load_config

#Conneting to the database
connection = connect(load_config())
cursor = connection.cursor()

#Creating the table
cursor.execute("DROP TABLE IF EXISTS sentences_pg")
cursor.execute("""CREATE TABLE sentences_pg
               (id SERIAL PRIMARY KEY,
               sentence TEXT,
               embedding vector(384))""")
connection.commit()

#Charging the sentences
ds = load_dataset("williamkgao/bookcorpus100mb")
sentences = ds['train']['text'][:10000]

#Inserting to the database and calculing time
start_time = time.time()
for sentence in sentences:
    cursor.execute("INSERT INTO sentences_pg (sentence) VALUES (%s)", (sentence,))
connection.commit()
end_time = time.time()

#Close connection
cursor.close()
connection.close()

print(f"The insertion time with text is: {end_time - start_time} seconds")