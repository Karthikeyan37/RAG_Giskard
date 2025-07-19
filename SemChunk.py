import stanza
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download and load Stanza English pipeline
stanza.download('en')  # Only needed once
nlp = stanza.Pipeline(lang='en', processors='tokenize')

# Initialize HuggingFace embeddings (MiniLM)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def semantic_chunk_resume_with_stanza(text: str) -> List[str]:
    """
    Chunk the input resume text into semantically meaningful pieces
    using Stanza for sentence tokenization and HuggingFace embeddings.
    """
    if not text.strip():
        logging.warning("Input text is empty.")
        return []

    try:
        # Sentence tokenization using Stanza
        doc = nlp(text)
        sentences = [
            sentence.text.strip()
            for sent in doc.sentences
            for sentence in [sent]
            if sentence.text.strip()
        ]

        # Join sentences with a custom boundary marker
        joined_text = " @@@SENTENCE_BOUNDARY@@@ ".join(sentences)

        # Create a semantic chunker with custom sentence split regex
        chunker = SemanticChunker(
            embeddings=embeddings,
            sentence_split_regex=r" @@@SENTENCE_BOUNDARY@@@ ",
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.0
        )

        # Generate semantic chunks
        chunks = chunker.create_documents([joined_text])
        return [doc.page_content.strip() for doc in chunks if doc.page_content.strip()]

    except Exception as e:
        logging.error(f"Error during semantic chunking: {e}")
        return []

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    file_path = "res1.pdf"  # Replace with your actual resume file

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            resume_text = f.read()

        chunks = semantic_chunk_resume_with_stanza(resume_text)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i + 1} ---\n{chunk}")

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
