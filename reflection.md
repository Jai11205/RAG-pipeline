# Team C: Govind Singh , Jai Prasanth , Subham Dey

## Local RAG System Components

### 1. Semantic Chunker
- **Function:** Slices documents into ~300-character segments using a hierarchy of separators (`\n\n` -> `.` -> `,`).
- **Importance:** Preserves context and ensures text fragments remain meaningful for the embedding model.

### 2. Embedding Engine (Sentence-Transformers)
- **Function:** Encodes text into 384-dimensional vectors.
- **Importance:** Enables "semantic" retrieval, allowing the system to understand the intent of a query rather than just matching keywords.

### 3. Document Filter & Retriever
- **Function:** Uses Cosine Similarity to rank chunks and Regex to filter by Document ID.
- **Importance:** Increases precision by allowing users to target specific files and providing the most relevant context to the LLM.

### 4. Local LLM (Ollama/Phi-3)
- **Function:** Generates a human-like response based *only* on the provided context.
- **Importance:** Provides a private, secure way to interact with data while strictly preventing information "hallucinations" through specialized prompting.
## Where Did the Pipeline Fail?

### 1. Document Identification Issue
- The pipeline performs correctly when the document is explicitly referenced in numeric form (e.g., *“document 3”*).
- However, it fails when the document is mentioned in textual format (e.g., *“document three”*, *“document five”*).
- In such cases, the system is unable to correctly identify the intended document.
- As a fallback, it relies on cosine similarity and returns the top 3 most relevant chunks, which may not correspond to the correct document.

---

### 2. Out-of-Scope Responses (Hallucination)
- The LLM tends to go beyond the provided context, even when explicitly instructed not to do so in the prompt.
- It generates answers using its own knowledge instead of restricting itself to the retrieved context.
- To handle evaluation constraints (e.g., hallucination checks), incorrect answers were intentionally forced in such cases.

---

### 3. Uncontrolled Creativity
- Despite clear instructions to stay within context and avoid using external reasoning, the LLM still produces creative or inferred responses.
- It does not reliably refuse to answer out-of-context queries, as expected.
- This leads to responses that may sound plausible but are not grounded in the provided data.
## 🔧 What Would I Change First If I Had More Time?

I would focus on improving the prompt design and enforcing stricter control over the LLM’s behavior.

### Key Changes:

1. **Refusal for Out-of-Context Queries**
   - The LLM should strictly refuse to answer any query that goes beyond the provided context.

2. **Restriction on Unnecessary Creativity**
   - The LLM should avoid generating creative or inferred responses.
   - It must strictly adhere to the given information without adding its own assumptions.

### Goal:
- To make the system more reliable, controlled, and strictly context-driven.
## Limitation of Cosine Similarity as a Retrieval Method

Cosine similarity is effective for retrieving semantically similar content. However, it has limitations when handling queries that require references to a **specific document**.

### Key Limitation:
- When a query explicitly asks for content from a particular document (e.g., by document number), cosine similarity may fail to prioritize chunks from that exact document.
- Instead, it retrieves the most semantically similar chunks across all documents, which may not belong to the intended source.

### Suggested Improvement:
- Combine semantic similarity with structured metadata (such as document ID or document number).
- Treat **document reference and text content as a unified entity** during retrieval.
- This ensures that both *relevance* and *correct document context* are preserved.

### Goal:
- Improve retrieval accuracy by aligning semantic matching with explicit document-level constraints.
