from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)


class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_and_split(self, file_path: str) -> List[str]:
        """Load a document and split it into chunks"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Select appropriate loader based on file extension
        if path.suffix.lower() == ".txt":
            loader = TextLoader(file_path)
        elif path.suffix.lower() == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # Load the document
        documents = loader.load()

        # Split the documents into chunks
        splits = self.text_splitter.split_documents(documents)

        # Extract the text content from splits
        return [doc.page_content for doc in splits]
