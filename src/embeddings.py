from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

#create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text1 = "Employee can work remotely upto three days"
text2 = "How many days employee can work from home"
text3 = "The company cafeteria serves lunch at noon"

#Generate the embeddings
vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)
vector3 = embeddings.embed_query(text3)


print("Numeber of dimensions vector1 ")
print(len(vector1))
print("Numeber of dimensions vector2")
print(len(vector2))
print("Numeber of dimensions vector3")
print(len(vector3))


def consine_similarity(vector_a,vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a,b) / (np.linalg.norm(a)* np.linalg.norm(b))

similarity_1 = consine_similarity(vector1,vector2)
similiarty_2 = consine_similarity(vector1,vector3)

print(similarity_1)
print(similiarty_2)
