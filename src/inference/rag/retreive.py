from openai import OpenAI
import numpy as np
import faiss
import json
from sentence_transformers import CrossEncoder

from src.utils.api_keys import get_api_key


class RunRAG:
    def __init__(
        self,
        index_path,
        chunks_path,
        embedding_model="text-embedding-3-large",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize RAG system:
        - Loads OpenAI client
        - Loads FAISS index
        - Loads chunk database
        - Loads reranker model
        """

        # -----------------------------
        # LOAD OPENAI API KEY
        # -----------------------------
        self.client = OpenAI(api_key=get_api_key("openai"))

        # -----------------------------
        # LOAD FAISS INDEX
        # -----------------------------
        self.index = faiss.read_index(index_path)

        # -----------------------------
        # LOAD CHUNKS
        # -----------------------------
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # -----------------------------
        # LOAD MODELS
        # -----------------------------
        self.embedding_model = embedding_model

        self.reranker = CrossEncoder(reranker_model, device="cpu")
        print("[RAG] reranker on device: cpu", flush=True)

    # --------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------
    def _embed_query(self, query):
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )

        embedding = np.array(
            res.data[0].embedding,
            dtype="float32"
        )

        return embedding

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------
    def _retrieve(self, query, k=30):
        q_emb = self._embed_query(query)

        # Normalize for cosine similarity
        faiss.normalize_L2(q_emb.reshape(1, -1))

        distances, indices = self.index.search(
            q_emb.reshape(1, -1),
            k
        )

        return indices[0]

    # --------------------------------------------------
    # RERANKING
    # --------------------------------------------------
    def _rerank(self, query, candidate_indices):
        pairs = []
        candidate_chunks = []

        for idx in candidate_indices:
            chunk = self.chunks[idx]

            pairs.append((query, chunk["text"]))
            candidate_chunks.append(chunk)

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(scores, candidate_chunks),
            key=lambda x: x[0],
            reverse=True
        )

        return [chunk for _, chunk in ranked]

    # --------------------------------------------------
    # MAIN RAG INFERENCE
    # --------------------------------------------------
    def rag_infer(self, query, top_k=3, retrieval_k=30):
        """
        Returns ONLY the retrieved chunk text,
        formatted for direct LLM prompting.
        """

        # Retrieve candidate chunks
        candidate_indices = self._retrieve(
            query,
            k=retrieval_k
        )

        # Rerank
        reranked_chunks = self._rerank(
            query,
            candidate_indices
        )

        # Keep top-k
        top_chunks = reranked_chunks[:top_k]

        # Return text-only context
        context = "\n\n".join(
            [chunk["text"] for chunk in top_chunks]
        )

        return context


# --------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------
if __name__ == "__main__":

    rag = RunRAG(
        index_path="/Users/vysri/Desktop/conversational-filler/src/inference/rag/uw_phd.index",
        chunks_path="/Users/vysri/Desktop/conversational-filler/src/inference/rag/uw_chunks.json",
    )

    query = "Yeah, I don't know how I can do that. When do I need to take quals?"

    context = rag.rag_infer(
        query,
        top_k=2
    )

    print("\n=== RAG CONTEXT ===\n")
    print(context)