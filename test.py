import nltk
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download NLTK data
# I had to export as an env var where the data were downloaded : export NLTK_DATA=/home/hay4hi/nltk_data
nltk.set_proxy('http://rb-proxy-de.bosch.com:8080')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load retriever models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Load generator model and tokenizer
generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
generator_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Example large document
large_document = """
Paris is the capital of France. It is known for its art, fashion, and culture. The Eiffel Tower is one of the most famous landmarks in Paris.
Berlin is the capital of Germany. It has a rich history and is known for its museums and historical sites.
Madrid is the capital of Spain. It is famous for its vibrant nightlife and cultural heritage.
"""

# Step 1: Divide the text into chunks (e.g., sentences)
chunks = nltk.sent_tokenize(large_document)

# Step 2: Encode the chunks using the context encoder
chunk_embeddings = [context_encoder(**context_tokenizer(chunk, return_tensors='pt').to(device)).pooler_output for chunk in chunks]

# Input query
query = "What is the capital of Germany?"

# Step 3: Encode the query using the question encoder
query_embedding = question_encoder(**question_tokenizer(query, return_tensors='pt').to(device)).pooler_output

# Step 4: Retrieve the most relevant chunk (simplified)
import torch
similarities = [torch.cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
retrieved_chunk = chunks[torch.argmax(torch.tensor(similarities))]

# Step 5: Generate response using the retrieved chunk
input_ids = generator_tokenizer(query + " " + retrieved_chunk, return_tensors='pt').input_ids.to(device)
output_ids = generator.generate(input_ids)
response = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(response)