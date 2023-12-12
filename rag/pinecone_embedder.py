import pinecone
from typing import List, Iterable
import itertools
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer

from .config import API_KEY, PINECONE_ENVIRONMENT

pinecone.init(api_key=API_KEY, environment=PINECONE_ENVIRONMENT)

# TODO: fixa bättre ids till index


class PineconeEmbedder:
    def __init__(self, index_name, pool_threads=30, text_field="text"):
        self.index_name = index_name
        self.index = pinecone.Index(self.index_name, pool_threads=pool_threads)
        self.dimension = self.index.describe_index_stats()["dimension"]
        self.model = None
        self.vectorstore = None

    def load_model(self, model_path: str, device="cuda"):
        """Load a sentence transformers model and create a vectorstore

        Args:
            model_path (str): model path
        """
        self.model = SentenceTransformer(model_path, device=device)
        assert (
            self.model.get_sentence_embedding_dimension() == self.dimension
        ), "Dimension mismatch"

    def create_vectorstore(self, namespace="default", text_key="text"):
        assert self.model is not None, "You need to load a sentence transformers model first!"
        self.vectorstore = Pinecone(self.index, self.encode, text_key=text_key, namespace=namespace)

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
        embeddings = self.encode(texts)

        vectors = []
        for ind, data in enumerate(zip(embeddings, texts)):
            vectors.append((str(ind), data[0], {"text": data[1]}))

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
        embeddings = self.encode(texts)

        vectors = []
        for ind, data in enumerate(zip(embeddings, texts)):
            vectors.append((str(ind), data[0], {"text": data[1]}))

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
          

    def query(self, query_list: List[str], top_k=5, namespace="default"):
        """Query the index

        Args:
            query_vector (List[str]): the list of queries
            top_k (int, optional): number of results. Defaults to 5.
            namespace (str, optional): pinecone namespace. Defaults to "default".

        Returns:
            list: list of ids
        """
        if self.vectorstore is None:
            raise ValueError("You need to load a sentence transformers model first!")

        return self.vectorstore.similarity_search(
            query_list,  # our search query
            k=top_k,  # return k most relevant docs
            namespace=namespace,
        )
