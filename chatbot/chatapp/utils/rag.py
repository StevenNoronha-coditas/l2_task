from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from dotenv import load_dotenv
import psycopg2, PyPDF2
from chatapp.utils.pydantic_files import RAGResult

load_dotenv()

embedding_model = SentenceTransformer('all-mpnet-base-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def establish_connection():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD")
    )
    return conn

def semantic_search(query, top_k=10, threshold=0.6, rerank_top_k=3):
    query_embedding = embedding_model.encode(query)
    
    connection = establish_connection()
    cursor = connection.cursor()

    query_embedding_list = query_embedding.tolist()

    cursor.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) as similarity
        FROM documents
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY similarity DESC
        LIMIT %s
    """, (query_embedding_list, query_embedding_list, threshold, top_k))
    
    results = cursor.fetchall()
    initial_results = [(content, float(similarity)) for content, similarity in results]
    cursor.close()
    connection.close()
    
    if initial_results:
        content_texts = [content for content, _ in initial_results]
        scores = cross_encoder.predict([(query, content) for content in content_texts])
        ranked_results = sorted(zip(content_texts, scores), key=lambda x: x[1], reverse=True)
        final_results = [RAGResult(content=content, similarity=similarity) for content, similarity in ranked_results[:rerank_top_k]]
    else:
        final_results = []
    
    return final_results

def store_embeddings():
    chunks = extract_text_from_pdf()
    connection = establish_connection()
    cursor = connection.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(768)
        );
        CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("No existing embeddings found. Generating new embeddings...")
        for chunk in chunks:
            embedding = embedding_model.encode(chunk)
            
            cursor.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (chunk, embedding.tolist())
            )
        print(f"Successfully stored {len(chunks)} embeddings.")
    else:
        print(f"Found {count} existing embeddings. Skipping embedding generation.")
    
    connection.commit()
    cursor.close()
    connection.close()

from django.conf import settings
def extract_text_from_pdf():
    pdf_path = os.path.join(settings.BASE_DIR, 'chatapp', 'utils', 'knowledge.pdf')
    text_chunks = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        words = text.split()
        chunk_size = 1000  
        overlap = 200     
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                text_chunks.append(chunk)
    return text_chunks