from sentence_transformers import SentenceTransformer, util

# Initialize the Sentence-BERT model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('all-mpnet-base-v2', 'cuda')


# Sample multi-paragraph emails
email1 = """Hi John,

I wanted to follow up on the project details. Please let me know if you need any additional information.

Also, make sure you have reviewed the latest documentation we shared last week.

Best regards,
Alice"""

email2 = """Hello John,

Just checking in regarding the project updates. Do you need any further info from my end?

Please review the documentation I sent earlier to stay updated.

Thanks,
Alice"""

# Define a function to compute the embedding for multi-paragraph text
def get_embedding_for_long_text(text):
    # Split the text by paragraphs
    paragraphs = text.split('\n\n')  # Assumes paragraphs are separated by two newlines
    
    # Compute embeddings for each paragraph
    paragraph_embeddings = [model.encode(paragraph, convert_to_tensor=True) 
                            for paragraph in paragraphs if paragraph.strip()]
    
    # Average the paragraph embeddings to get a single embedding for the full text
    if paragraph_embeddings:
        return sum(paragraph_embeddings) / len(paragraph_embeddings)
    else:
        return None  # Handle case where there are no valid paragraphs

# Get embeddings for each email
embedding1 = get_embedding_for_long_text(email1)
embedding2 = get_embedding_for_long_text(email2)

# Ensure that both embeddings were computed successfully
if embedding1 is not None and embedding2 is not None:
    # Calculate cosine similarity between the two embeddings
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    print(f"Similarity Score: {similarity_score}")
else:
    print("One of the embeddings could not be computed.")
