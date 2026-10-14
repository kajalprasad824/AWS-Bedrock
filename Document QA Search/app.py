
import boto3
import streamlit as st

from langchain_aws import BedrockEmbeddings, BedrockLLM, ChatBedrockConverse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


# --------------------------------------------------
# 1. AWS BEDROCK CLIENT AND EMBEDDINGS
# --------------------------------------------------

bedrock = boto3.client(service_name="bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock
)


# --------------------------------------------------
# 2. PDF DATA INGESTION
# --------------------------------------------------

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No PDF files found. Place your PDFs inside the data folder."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)
    return docs


# --------------------------------------------------
# 3. CREATE FAISS VECTOR STORE
# --------------------------------------------------

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        documents=docs,
        embedding=bedrock_embeddings
    )

    vectorstore_faiss.save_local("faiss_index")


# --------------------------------------------------
# 4. GEMMA LLM
# --------------------------------------------------

def get_gemma_llm():
    return ChatBedrockConverse(
        model="google.gemma-3-4b-it",
        region_name=bedrock.meta.region_name,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9
    )

# --------------------------------------------------
# 5. NEMOTRON LLM
# --------------------------------------------------

def get_nemotron_llm():
    llm = ChatBedrockConverse(
        model="nvidia.nemotron-super-3-120b",
        region_name=bedrock.meta.region_name,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9
    )

    return llm


# --------------------------------------------------
# 6. PROMPT TEMPLATE
# --------------------------------------------------

prompt_template = """
Use the following context to answer the question.

Provide a clear and detailed answer of approximately 250 words
when the context contains enough information.

If the answer is not available in the context, say that you
don't know. Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# --------------------------------------------------
# 7. RETRIEVAL QA
# --------------------------------------------------

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa.invoke({"query": query})

    answer = response["result"]
    source_documents = response["source_documents"]

    return answer, source_documents


# --------------------------------------------------
# 8. STREAMLIT APPLICATION
# --------------------------------------------------

def main():
    st.set_page_config(
        page_title="Chat PDF",
        page_icon="📄",
        layout="wide"
    )

    st.header("Chat With PDF Using AWS Bedrock")

    user_question = st.text_input(
        "Ask a question about your PDF files"
    )

    # Model selector
    model_choice = st.selectbox(
        "Choose an LLM",
        ["Gemma", "Nemotron"]
    )

    # Sidebar: create or update vector store
    with st.sidebar:
        st.title("Update or Create Vector Store")

        if st.button("Update Vectors"):
            try:
                with st.spinner("Processing PDF files..."):
                    docs = data_ingestion()
                    get_vector_store(docs)

                st.success(
                    f"Vector store created successfully "
                    f"from {len(docs)} text chunks."
                )

            except Exception as e:
                st.error(f"Error creating vector store: {e}")

    # Generate answer
    if st.button("Generate Answer"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
            return

        try:
            with st.spinner(
                f"Generating answer using {model_choice}..."
            ):
                # Load the saved FAISS index
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )

                # Select the requested model
                if model_choice == "Gemma":
                    llm = get_gemma_llm()
                else:
                    llm = get_nemotron_llm()

                # Retrieve context and generate answer
                answer, source_documents = get_response_llm(
                    llm,
                    faiss_index,
                    user_question
                )

            st.subheader(f"Answer ({model_choice})")
            st.write(answer)

            # Display source documents
            with st.expander("View source documents"):
                for i, doc in enumerate(source_documents, start=1):
                    st.markdown(f"**Source {i}**")
                    st.write(doc.page_content[:1500])

                    st.caption(
                        f"File: {doc.metadata.get('source', 'Unknown')} "
                        f"| Page: {doc.metadata.get('page', 'Unknown')}"
                    )

        except Exception as e:
            st.error(
                f"Error while generating the answer with "
                f"{model_choice}: {e}"
            )


# --------------------------------------------------
# 9. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    main()
