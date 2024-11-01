import streamlit as st
import duckdb
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def get_connection():
    conn = duckdb.connect('embeddings.db')
    extensoes_duckdb = conn.execute("""SELECT extension_name, installed, description
                FROM duckdb_extensions();""").fetchdf()
    if extensoes_duckdb.query('extension_name == "vss"')['installed'].values[0] == False:
        conn.execute("""INSTALL vss;""")
    conn.execute("""LOAD vss;""")
    return conn


@st.cache_resource
def load_embeddings():
    embeddings_path = 'embeddings.parquet'
    return pd.read_parquet(embeddings_path)

# Função para realizar a busca
def perform_search(conn, embedding):
    query = f"""
    SELECT * 
    FROM embeddings 
    ORDER BY array_distance(emb, {embedding.tolist()}::FLOAT[512]) 
    LIMIT 5;
    """
    return conn.execute(query).fetch_df()

st.sidebar.title("Como usar a Busca de Imagens")

st.sidebar.markdown("""
Esta ferramenta utiliza busca vetorial para encontrar imagens semelhantes à imagem que for inserida.

**Como funciona:**

1. **Realizar Busca:** Ao clicar no botão realizar busca, uma imagem da base será selecionada aleatóriamente.
2. **Obtenha resultados:** A ferramenta irá procurar em sua base de dados por imagens visualmente semelhantes.
3. **Entenda os resultados:** A similaridade é calculada com base em elementos como cores, texturas e composição da imagem, não em características faciais.

""")

# Exibe a primeira imagem separadamente
def show_images(image_paths, title):
    #st.header(title)
    
    # Exibe a primeira imagem separadamente
    if image_paths:
        first_image_path = image_paths[0]
        first_img = Image.open(first_image_path)
        first_filename = first_image_path.removeprefix('fotos_celebs/').removesuffix('.jpg')
        st.image(first_img, width= 80, caption=first_filename)
        
        # Exibe as imagens restantes em colunas
        remaining_image_paths = image_paths[1:]
        if remaining_image_paths:
            cols = st.columns(len(remaining_image_paths))
            for i, path in enumerate(remaining_image_paths):
                img = Image.open(path)
                filename = path.removeprefix('fotos_celebs/').removesuffix('.jpg')
                cols[i].image(img, use_column_width=True, caption=filename)

st.title('Imagens semelhantes com DuckDB 🦆')

conn = get_connection()
embeddings = load_embeddings()

tabela, imagens = st.columns([1,1])

if st.button('Realizar Busca'):
    aleatorio = np.random.randint(0, len(embeddings))
    embedding = embeddings.iloc[aleatorio]['emb']
    filename = embeddings.iloc[aleatorio]['filename']

    with tabela:
        # Realizando a busca
        t_init = time.time()
        df_resultado = perform_search(conn, embedding)
        t_end = time.time()
        st.write(f'A busca demorou {t_end-t_init} s')
        st.write(df_resultado)
    with imagens:
        # Carregar caminhos das imagens
        st.write(f'Realizando busca com {filename}')
        image_paths = [os.path.join('fotos_celebs', filename)]  # Imagem consultada
        image_paths.extend([os.path.join('fotos_celebs', fn) for fn in df_resultado['id']])  # Imagens retornadas

        show_images(image_paths, f'Imagem Consultada: {filename}')