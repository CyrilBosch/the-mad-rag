import nltk

class Chunking():

    def __init__(self):
        self.chunks = []
        self.sentences = []

    def chunking_into_sentences(self, input_document):
        # Tokenize the document into sentences
        self.chunks = nltk.sent_tokenize(input_document)

    def chunking_sliding_window(self, input_document, window_size, stride):
        # Using a sliding window approach to create overlapping chunks of text. This ensures that important context is preserved across chunks.
        # window_size, Number of sentences per chunk
        # stride, Number of sentences to move the window each step

        # Tokenize the document into sentences
        sentences = nltk.sent_tokenize(input_document)

        chunks = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = " ".join(sentences[i:i + window_size])
            chunks.append(chunk)
        self.chunks = chunks
        