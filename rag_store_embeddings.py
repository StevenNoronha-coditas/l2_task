from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2, PyPDF2

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
            print(embedding)
            
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

def extract_text_from_pdf():
    pdf_path = './knowledge.pdf'
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

store_embeddings()