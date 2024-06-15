import streamlit as st
import openai
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Configura la chiave API di OpenAI
openai.api_key = 'YOUR_OPENAI_API_KEY_3'

# Funzione per ottenere embeddings dai documenti
def get_embeddings(document):
    response = openai.embeddings.create(
        input=document,
        model="text-embedding-ada-002"  # Specifica il modello di embedding di OpenAI
    )
    return response.data[0].embedding

# Funzione per calcolare la similarità coseno
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Interfaccia Streamlit
st.title('Document Similarity Checker')

st.write('Verifica la similarità tra due documenti presenti nel repository GitHub.')

# Inserisci il nome del repository e i percorsi dei file

# Scarica i contenuti dei documenti
content1 = "doc_1.txt"
content2 = "doc_2.txt"

if content1 and content2:
    # Ottieni gli embeddings per entrambi i documenti
    embedding1 = get_embeddings(content1)
    embedding2 = get_embeddings(content2)

    # Calcola la similarità coseno
    similarity = calculate_similarity(embedding1, embedding2)

    st.write(f"La similarità tra i documenti è: {similarity:.4f}")
else:
    st.error("Per favore, compila tutti i campi.")