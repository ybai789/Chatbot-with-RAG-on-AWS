import os
import logging
import json
import boto3
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pinecone

# AWS S3 client and SageMaker runtime
s3 = boto3.client('s3')
sagemaker_runtime = boto3.client('runtime.sagemaker')
sagemaker_endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Lambda Handler definition
def lambda_handler(event, context):
    """Handler for S3-triggered Lambda function"""
    # Extract bucket name and object key (file path) from the event
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        logging.info(f"Processing file from bucket: {bucket_name}, key: {object_key}")
        
        # Determine file type based on object key (e.g., 'md/' for markdown or 'pdf/' for PDF)
        if object_key.startswith('md/'):
            file_type = 'md'
        elif object_key.startswith('pdf/'):
            file_type = 'pdf'
        else:
            logging.error(f"Unsupported file type for object key: {object_key}")
            continue

        # Load the document from S3 and process it
        knowledge_base = KnowledgeBase(
            s3_bucket=bucket_name,
            prefix_md='md/',  # This can be dynamic or based on event data
            prefix_pdf='pdf/',  # Same for PDF prefix
            index_name='your-pinecone-index'
        )

        # Generate the knowledge base with the newly uploaded document
        knowledge_base.process_single_document(object_key, file_type)

    return {
        'statusCode': 200,
        'body': json.dumps('Knowledge Base updated with new document.')
    }


class KnowledgeBase:
    def __init__(self, s3_bucket: str, prefix_md: str, prefix_pdf: str, index_name: str, chunk_size: int = 1000, chunk_overlap: int = 500):
        self.s3_bucket = s3_bucket
        self.prefix_md = prefix_md
        self.prefix_pdf = prefix_pdf
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Pinecone index
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=768)  # Assuming Hugging Face embedding dimension is 768
        self.index = pinecone.Index(self.index_name)

    def process_single_document(self, s3_key: str, file_type: str):
        """Process a single document that was uploaded to S3."""
        logging.info(f"Processing {file_type} document: {s3_key}")
        document = self.load_single_document_from_s3(s3_key, file_type)
        chunks = self.split_text([document])
        self.save_to_pinecone(chunks)

    def load_single_document_from_s3(self, s3_key: str, file_type: str) -> Document:
        """Load a single document from S3 based on its file type."""
        local_path = f"/tmp/{os.path.basename(s3_key)}"
        s3.download_file(self.s3_bucket, s3_key, local_path)

        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(local_path)
        elif file_type == 'pdf':
            loader = PyPDFLoader(local_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        document = loader.load()[0]  # Assume each file contains one document
        os.remove(local_path)  # Clean up temporary file
        return document

    def split_text(self, documents: List[Document]) -> List[Document]:
        """Split the documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        chunks = []
        for doc in documents:
            chunks.extend(text_splitter.split_documents([doc]))
        return chunks

    def embed_documents_with_sagemaker(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SageMaker's Hugging Face model."""
        payload = {'inputs': texts}
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sagemaker_endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response['Body'].read().decode())
        embeddings = result['embeddings']
        return embeddings

    def save_to_pinecone(self, chunks: List[Document]):
        """Save the chunks to Pinecone index."""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embed_documents_with_sagemaker(texts)
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.index.upsert(vectors=list(zip(ids, embeddings)))

