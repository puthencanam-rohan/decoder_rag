# decoder_rag
Retrieval Augmented Generation Tutorial

## Some preliminary findings:
1. You need a model with 8B parameters or more for effective RAG. Lighter models ignore context or hallucinate i.e. make up answers.
2. Using the 8B models to embed is time-consuming. Using an embedding-oriented model may be quicker with no loss in quality of RAG.
3. Decoder's YouTube tutorial was used for this project. Though the parse_file function was replaced with custom logic. In addition, the file-based embedding storage would be too slow and inefficient and so was replaced with a Chroma vector database.

# Reference
Decoder's RAG tutorial
https://www.youtube.com/watch?v=V1Mz8gMBDMo&pp=ygUSZGVjb2RlciBvbGxhbWEgcmFn

# Model
Mistral 7B works well for RAG on my CPU-based laptop.
https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/blob/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
