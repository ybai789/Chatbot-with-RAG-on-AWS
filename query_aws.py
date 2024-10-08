import os
import sys
from dotenv import load_dotenv, find_dotenv
import argparse
from tqdm import tqdm
import requests
import json
import boto3
import pinecone
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

# Load environment variables
_ = load_dotenv(find_dotenv())
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
SAGEMAKER_ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME')

if not PINECONE_API_KEY or not SAGEMAKER_ENDPOINT_NAME:
    print("Error: Missing required environment variables.")
    sys.exit(1)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = 'your_pinecone_index_name'
index = pinecone.Index(index_name)

# SageMaker client
sagemaker_runtime = boto3.client('runtime.sagemaker')

PROMPT_TEMPLATE = """
Answer the question using only the following context:
{context}
-------------------------------------------------------------
Based on the above context, answer this question:
{question}
"""

def embed_query_with_sagemaker(query_text):
    """Generate embeddings for the query using the SageMaker endpoint."""
    payload = {
        'inputs': [query_text]
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response['Body'].read().decode())
    embeddings = result['embeddings'][0]
    return embeddings

def query_knowledge_base(query_text):
    """Query Pinecone index using the SageMaker-generated embeddings."""
    query_embedding = embed_query_with_sagemaker(query_text)
    
    # Query Pinecone for relevant documents
    response = index.query(queries=[query_embedding], top_k=5, include_metadata=True)

    return response['matches']

def format_source_path(full_path):
    return os.path.basename(full_path)

def extract_relevant_excerpt(doc_content, query):
    """Extract relevant excerpt from the document."""
    lower_content = doc_content.lower()
    query_keywords = query.lower().split()
    for keyword in query_keywords:
        if keyword in lower_content:
            start_pos = max(0, lower_content.find(keyword) - 100)
            end_pos = min(len(doc_content), lower_content.find(keyword) + 150)
            return doc_content[start_pos:end_pos]
    return doc_content[:200]  # Return the first 200 characters as fallback

def rerank_results(query_text, results):
    """Rerank the results based on relevance to the query using a SageMaker model."""
    sagemaker_runtime = boto3.client('runtime.sagemaker')
    sagemaker_endpoint_name = os.getenv('SAGEMAKER_RERANK_ENDPOINT_NAME')  # Deployed reranking model endpoint
    
    # Create a list of tuples (document text, similarity score)
    ranked_results = []
    
    for match in results:
        document_text = match['metadata']['text'][:300]  # Limiting to first 300 characters
        body = json.dumps({
            "inputs": {
                "query": query_text,
                "document": document_text
            }
        })
        
        # Call SageMaker endpoint to get similarity score
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sagemaker_endpoint_name,
            ContentType='application/json',
            Body=body
        )
        response_body = json.loads(response['Body'].read().decode())
        similarity_score = response_body['similarity_score']
        
        # Append the result with its score
        ranked_results.append((match, similarity_score))
    
    # Sort results by similarity score (descending)
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return only the sorted results
    return [match for match, score in ranked_results]

# Initialize Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Replace with the appropriate region

# Function to call AWS Bedrock model
def call_bedrock_model(prompt):
    """Call AWS Bedrock model with a prompt and return the response."""
    response = bedrock_client.invoke_model(
        modelId='your-bedrock-model-id',  # Replace with your specific Bedrock model ID
        contentType='application/json',
        body=json.dumps({
            "inputText": prompt
        })
    )

    # Parse the response
    response_body = json.loads(response['body'].read().decode())
    return response_body['results'][0]['outputText']

# Define the lambda_handler for the Query and Rerank process
def lambda_handler_query(event, context):
    """Lambda handler for query processing and reranking."""
    try:
        # Extract the query text from the Lambda event input
        query_text = event.get('query', '')

        if not query_text:
            return {
                'statusCode': 400,
                'body': json.dumps('No query provided.')
            }

        # Query the knowledge base
        results = query_knowledge_base(query_text)

        if not results:
            return {
                'statusCode': 404,
                'body': json.dumps(f"No matching results for query: '{query_text}'")
            }

        # Rerank the results based on relevance
        ranked_results = rerank_results(query_text, results)

        # Prepare the context for the LLM
        context = "\n\n---\n\n".join([match['metadata']['text'][:300] for match in ranked_results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query_text)

        # Use AWS Bedrock model to answer the question
        response_text = call_bedrock_model(prompt)

        # Compile response with answer and sources
        response_body = {
            "query": query_text,
            "answer": response_text,
            "sources": []
        }

        unique_sources = set()
        for match in ranked_results:
            source = format_source_path(match['metadata'].get("source", "Unknown"))
            if source not in unique_sources:
                unique_sources.add(source)
                relevant_excerpt = extract_relevant_excerpt(match['metadata'].get("text", ""), query_text)
                response_body["sources"].append({
                    "source": source,
                    "excerpt": relevant_excerpt[:200]
                })

        # Return the final response with sources
        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {str(e)}")
        }

# Optionally, add a main function for local testing or CLI access
def main():
    parser = argparse.ArgumentParser(description="Query the Knowledge Base")
    parser.add_argument("query", nargs="?", type=str, help="The question to ask")
    args = parser.parse_args()

    if args.query:
        event = {'query': args.query}
        print(lambda_handler_query(event, None))  # Simulate the Lambda function call for testing
    else:
        print("Welcome to the Knowledge Base. Enter your question below, or type 'quit' to exit.")
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            elif not query.strip():
                print("Please enter a valid question.")
            else:
                event = {'query': query}
                print(lambda_handler_query(event, None))  # Simulate the Lambda function call for testing

if __name__ == "__main__":
    main()
