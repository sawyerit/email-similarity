from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup

# Initialize the Sentence-BERT model
# model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', 'cuda')

# Sample multi-paragraph emails
email1 = """"""
email2 = """"""
email3 = """"""
email4 = """"""

# Define a function to compute the embedding for multi-paragraph text
def get_embedding_for_long_text(text):
    
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(['p', 'div', 'br']):
        tag.replace_with(tag.get_text() + '\n\n')
    clean_text = (soup.get_text())


    # Split the text by paragraphs
    paragraphs = clean_text.split('\n\n')  # Assumes paragraphs are separated by two newlines
    
    # Compute embeddings for each paragraph
    paragraph_embeddings = [model.encode(paragraph, convert_to_tensor=False) 
                            for paragraph in paragraphs if paragraph.strip()]
    
    # Average the paragraph embeddings to get a single embedding for the full text
    if paragraph_embeddings:
        return sum(paragraph_embeddings) / len(paragraph_embeddings)
    else:
        return None  # Handle case where there are no valid paragraphs
    

# Get embeddings for each email
embedding1 = get_embedding_for_long_text(email1)
embedding2 = get_embedding_for_long_text(email2)
embedding3 = get_embedding_for_long_text(email3)
embedding4 = get_embedding_for_long_text(email4)


# Calculate cosine similarity between the two embeddings
embeddings = [
    ('1 vs 2', embedding1, embedding2),
    ('1 vs 3', embedding1, embedding3),
    ('1 vs 4', embedding1, embedding4),
    ('2 vs 3', embedding2, embedding3),
    ('2 vs 4', embedding2, embedding4),
    ('3 vs 4', embedding3, embedding4),
]

for description, emb1, emb2 in embeddings:
    similarity_score = util.pytorch_cos_sim(emb1, emb2).item()
    print(f"Similarity Score {description}: {similarity_score}")
