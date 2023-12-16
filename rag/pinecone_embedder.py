import pinecone
from typing import List, Iterable
import itertools
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .utils import load_json, generate_random_string

from .config import API_KEY, PINECONE_ENVIRONMENT

pinecone.init(api_key=API_KEY, environment=PINECONE_ENVIRONMENT)


class PineconeEmbedder:
    def __init__(self, index_name, pool_threads=30):
        self.index_name = index_name
        self.index = pinecone.Index(self.index_name, pool_threads=pool_threads)
        self.dimension = self.index.describe_index_stats()["dimension"]
        self.model = None
        self.vectorstore = None
        self.model_name = None

    def load_model(self, model_path: str, device="cuda"):
        """Load a sentence transformers model and create a vectorstore

        Args:
            model_path (str): model path
        """
        self.model = SentenceTransformer(model_path, device=device)
        self.model_name = model_path
        assert (
            self.model.get_sentence_embedding_dimension() == self.dimension
        ), "Dimension mismatch"

    def create_vectorstore(self, namespace="default", text_key="text"):
        """Create a vectorstore

        Args:
            namespace (str, optional): Pinecone namespace to use. Defaults to "default".
            text_key (str, optional): Text key. Defaults to "text".
        """
        assert (
            self.model is not None
        ), "You need to load a sentence transformers model first!"
        self.vectorstore = Pinecone(
            self.index, self.encode, text_key=text_key, namespace=namespace
        )

    @staticmethod
    def _add_prefix(texts: List[str], prefix: str = "query: "):
        return [prefix + text for text in texts]

    def encode(self, texts: List[str]):
        """Encode a list of texts

        Args:
            texts (List[str]): list of texts

        Returns:
            2d list: list of embeddings
        """
        embeddings = self.model.encode(texts).tolist()
        return embeddings

    @staticmethod
    def _chunks(iterable: Iterable, batch_size: int = 100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def upsert_parallel(self, texts: List[str], namespace="default"):
        """Upsert a list of texts in parallel to pinecone

        Args:
            texts (List[str]): list of texts
            namespace (str, optional): pinecone namespace. Defaults to "default".

        Returns:
            list: api responses
        """
        # need to add prefix
        if "e5" in self.model_name:
            texts = self._add_prefix(texts)

        embeddings = self.encode(texts)

        vectors = []
        for _, data in enumerate(zip(embeddings, texts)):
            vectors.append((generate_random_string(), data[0], {"text": data[1]}))

        # Send requests in parallel
        async_results = [
            self.index.upsert(
                vectors=ids_vectors_chunk, async_req=True, namespace=namespace
            )
            for ids_vectors_chunk in self._chunks(vectors)
        ]

        # Wait for and retrieve responses (this raises in case of error)
        responses = [async_result.get() for async_result in async_results]

        return responses

    def chunk_document(
        self, document_path: str, chunk_size: int = 300, chunk_overlap: int = 20
    ):
        """Chunk a document into smaller chunks"""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        extracted_text = load_json(document_path)

        all_chunks = []
        for chunk in extracted_text:
            texts = text_splitter.create_documents([chunk])
            texts = [text.page_content for text in texts]
            all_chunks.extend(texts)

        return all_chunks

    def get_retriever(self):
        """Return the vectorstore object as a retriever for RAG"""
        return self.vectorstore.as_retriever()

    def upsert(self, texts, namespace="default"):
        """Upsert a list of texts to pinecone

        Args:
            texts (List[str]): list of texts
            namespace (str, optional): pinecone namespace. Defaults to "default".

        Returns:
            UpsertResponse: api responses
        """
        # need to add prefix
        if "e5" in self.model_name:
            texts = self._add_prefix(texts)

        embeddings = self.encode(texts)

        vectors = []
        for _, data in enumerate(zip(embeddings, texts)):
            vectors.append((generate_random_string(), data[0], {"text": data[1]}))

        upsert_response = self.index.upsert(vectors=vectors, namespace=namespace)
        assert upsert_response["upserted_count"] == len(
            embeddings
        ), "Upsert count mismatch"

        return upsert_response

    def index_info(self):
        """Get index info

        Returns:
            DescribeIndexStatsResponse: index info
        """
        return self.index.describe_index_stats()

    def delete_id(self, ids: List[str], namespace="default"):
        """Delete a list of ids

        Args:
            ids (List[str]): ids to delete
            namespace (str, optional): pinecone namespace. Defaults to "default".
        """
        self.index.delete(ids=ids, namespace=namespace)

    def delete_namespace(self, namespace="default"):
        """Deletes all vectors in a namespace

        Args:
            namespace (str, optional): pinecone namespace. Defaults to "default".
        """
        self.index.delete(delete_all=True, namespace=namespace)

    def query(
        self, query_list: List[str], top_k=5, namespace="default", text_key="text"
    ):
        """Query the index

        Args:
            query_vector (List[str]): the list of queries
            top_k (int, optional): number of results. Defaults to 5.
            namespace (str, optional): pinecone namespace. Defaults to "default".

        Returns:
            list: list of ids
        """
        if self.model is None:
            raise ValueError("You need to load a sentence transformers model first!")
        vectorstore = Pinecone(
            self.index, self.encode, text_key=text_key, namespace=namespace
        )

        return vectorstore.similarity_search(
            query_list,  # our search query
            k=top_k,  # return k most relevant docs
            namespace=namespace,
        )
