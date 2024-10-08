## Introduction

This repository provides a fully AWS-deployed system that integrates a knowledge base with a chatbot interface, leveraging a **Retrieval-Augmented Generation (RAG)** architecture. The system utilizes two **SageMaker-deployed Hugging Face models** for embedding generation and document reranking, as well as an **AWS Bedrock model** for generating final responses. The solution is built using **LangChain** for document processing and retrieval, **Amazon S3** for document storage, **Pinecone** as the vector database for storing and retrieving document embeddings, and **AWS Lambda** for handling serverless execution.

The architecture ensures scalable knowledge base management, real-time query handling, and accurate, AI-driven responses by combining embedding-based document retrieval and sophisticated language modeling. This system is ideal for managing large document repositories, offering contextually aware, precise responses.

## Features

- **Knowledge Base Creation**:
  - The `build_knowledge_base_aws.py` script processes markdown and PDF documents, splits them into smaller chunks, and generates embeddings using a SageMaker-deployed Hugging Face model. The generated embeddings are stored in Pinecone for efficient retrieval.

- **Querying the Knowledge Base**:
  - The `query_aws.py` script allows users to interact with the knowledge base via command-line queries. It retrieves relevant documents from Pinecone, reranks them using a second SageMaker model to ensure relevance, and then generates a final response using a Bedrock model for language generation.

## Architecture Overview

- **SageMaker Models**:
  1. **Embedding Model**: Used to generate embeddings for document chunks and store them in Pinecone.
  2. **Reranking Model**: Used to rerank retrieved documents based on their relevance to the user's query.

- **Bedrock Model**: Provides the language generation capabilities, processing the reranked documents and formulating the final response.

- **AWS Lambda Functions**:
  1. **Document Processing Lambda**: Processes documents, generates embeddings using the SageMaker embedding model, and stores the embeddings in Pinecone.
  2. **Query Handling Lambda**: Handles user queries, retrieves and reranks documents, and invokes the Bedrock model to generate the final response.

This system seamlessly integrates multiple AWS services, providing a powerful, scalable solution for knowledge management and AI-powered query response generation.

## How It Works

<img src=".\images\arch.png" alt="arch" style="zoom:40%;" />

1. **Build the Knowledge Base**:
   - Upload your markdown and PDF documents to the designated **Amazon S3** bucket.
   - This will automatically trigger the **Document Processing Lambda Function**, which will process the documents by chunking them into smaller pieces, generate embeddings using a SageMaker-deployed model, and store the embeddings in a **Pinecone** vector database for efficient retrieval.
2. **Query the Knowledge Base**:
   - To query the knowledge base, invoke the **Query Handling Lambda Function** (e.g., via AWS API Gateway or AWS CLI). The system will retrieve relevant documents from **Pinecone**, rerank them using a SageMaker reranking model, and generate the final response using an **AWS Bedrock** language model.

## AWS Setup

### 1. **Create and Configure Amazon S3 Bucket**

   - **Steps**:
     1. Log in to [AWS Management Console](https://aws.amazon.com/console/).
     2. Go to the **Amazon S3** console and click **Create Bucket**.
     3. Enter a bucket name (e.g., `my-knowledge-base-bucket`).
     4. Select the appropriate region and create the bucket.
     5. Create two folders within the bucket: one for markdown files (`md/`), and one for PDF files (`pdf/`).
     6. Upload the markdown and PDF documents you want to process to their respective folders.

### 2. **Deploy Hugging Face Model to Amazon SageMaker**

**Add embedding genertion model deployment**:

   - **Steps**:
     1. Log in to **AWS Management Console**.
     2. Go to **Amazon SageMaker** console, and choose **Models** > **Deploy model**.
     3. Click **Create model** and select **Hugging Face Model**.
     4. Enter the model ID, such as a Hugging Face model for embedding generation (e.g., `sentence-transformers/paraphrase-MiniLM-L6-v2`).
     5. Configure an instance type (e.g., `ml.m5.large`) for inference.
     6. Deploy the model as a SageMaker inference endpoint, and note the endpoint name (e.g., `huggingface-embedding-endpoint`), which will be used in the code.
   - **Reference**: The official Hugging Face model library provides models for text embeddings. You can choose the appropriate model here: [Hugging Face Transformers](https://huggingface.co/models).

#### **Add rerank model deployment**:

In addition to deploying the Hugging Face model for embedding generation, you can deploy another model for reranking the results. This rerank model will accept the query and document text and output a similarity score.

- **Steps**:
  1. In the SageMaker console, go to **Models** > **Deploy model** again.
  2. Enter the model ID for the rerank model (for example, a `sentence-transformers` model that can assess text similarity).
  3. Deploy the model and note the SageMaker endpoint name (e.g., `huggingface-rerank-endpoint`).
- **Note**:
  This rerank model will compare the query with each document and return a relevance score, which will be used to reorder the results before sending them to the LLM for final processing.

### 3. **Create Pinecone Vector Database**

   - **Steps**:
     1. [Sign up for Pinecone](https://www.pinecone.io/) and log in to the Pinecone console.
     2. Create a new Pinecone index, setting the dimension to 768 (matching the output dimension of your chosen Hugging Face embedding model).
     3. Record your Pinecone API key and environment, which you will use in the code.

### 4. **Create IAM Role and Configure Permissions**

   - **Steps**:
     1. Go to the **AWS IAM** console and create a new role, selecting **Lambda** service as the trusted entity.
     2. Attach necessary permissions to the role, including:
        - **S3 permissions**: `s3:GetObject`, `s3:ListBucket` (for reading files from S3).
        - **SageMaker permissions**: `sagemaker:InvokeEndpoint` (for calling SageMaker endpoints).
        - **CloudWatch log permissions**: `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` (for writing CloudWatch logs).
     3. Name the role and remember the ARN, which will be used in the Lambda function.

### 5. **Create Lambda Functions**

You will need to create two Lambda functions: one for **document processing and embedding generation**, and another for **query handling and reranking**. Below are the steps for creating each of these Lambda functions.

#### **Steps for Document Processing and Embedding Generation Lambda Function**:

1. **Go to the AWS Lambda Console**:
   - Click **Create Function**.
   - Choose **Author from scratch**.

2. **Set up the function**:
   - **Function name**: Enter a name (e.g., `DocumentEmbeddingGenerator`).
   - **Runtime**: Choose Python 3.8 or higher as the runtime.

3. **Permissions**:
   - In the permissions section, select the IAM role you created earlier. This role should have permissions to access S3, invoke the SageMaker embedding endpoint, and store embeddings in Pinecone.

4. **Set environment variables**:
   - Go to the **Configuration** tab and choose **Environment Variables**.
   - Add the following environment variables:
     - `SAGEMAKER_ENDPOINT_NAME`: Your SageMaker-deployed Hugging Face embedding model endpoint name.
     - `PINECONE_API_KEY`: Your Pinecone API key.
     - `PINECONE_ENVIRONMENT`: The environment set in Pinecone.
     - `S3_BUCKET_NAME`: The name of the S3 bucket where your documents are stored.
     - `S3_PREFIX_MD`: The prefix (folder path) for markdown documents.
     - `S3_PREFIX_PDF`: The prefix (folder path) for PDF documents.

5. **Upload code**:
   - Either upload your code file or paste the code directly in the Lambda console.
   - This code should read documents from S3, process them to generate embeddings using SageMaker, and store them in Pinecone.

6. **Upload Lambda Layer**:
   - Package the necessary dependencies such as `boto3`, `requests`, and `pinecone-client` into a **Lambda Layer**. Upload this layer so your Lambda function has access to the required Python packages.

#### **Steps for Query Handling and Reranking Lambda Function**:

1. **Go to the AWS Lambda Console**:
   - Click **Create Function**.
   - Choose **Author from scratch**.

2. **Set up the function**:
   - **Function name**: Enter a name (e.g., `QueryRerankHandler`).
   - **Runtime**: Choose Python 3.8 or higher as the runtime.

3. **Permissions**:
   - In the permissions section, select the IAM role you created earlier. This role should have permissions to invoke the SageMaker rerank endpoint, query Pinecone, and use other necessary AWS resources (e.g., CloudWatch).

4. **Set environment variables**:
   - Go to the **Configuration** tab and choose **Environment Variables**.
   - Add the following environment variables:
     - `SAGEMAKER_RERANK_ENDPOINT_NAME`: Your SageMaker-deployed rerank model endpoint name.
     - `SAGEMAKER_LLM_ENDPOINT_NAME`: (Optional) Your SageMaker-deployed LLM endpoint name if you are not using external LLM services.
     - `PINECONE_API_KEY`: Your Pinecone API key.
     - `PINECONE_ENVIRONMENT`: The environment set in Pinecone.

5. **Upload code**:
   - Either upload your code file or paste the code directly in the Lambda console.
   - This code should query Pinecone for relevant documents, rerank them using the SageMaker rerank endpoint, and then pass the reranked results to an LLM for generating the final response.

6. **Upload Lambda Layer**:
   - Package the necessary dependencies (`boto3`, `requests`, and `pinecone-client`) into a **Lambda Layer** and upload it to ensure the Lambda function has access to the required Python packages.

#### **Summary of Lambda Functions**:
- **DocumentEmbeddingGenerator**: Reads documents from S3, generates embeddings using SageMaker, and stores them in Pinecone.
- **QueryRerankHandler**: Retrieves embeddings from Pinecone, reranks them using SageMaker, and uses an LLM to generate responses.

Each Lambda function has distinct roles and environment variables, and both rely on their respective SageMaker endpoints for embedding generation and reranking.

### 6  Create and Deploy a Model on AWS Bedrock for chat response

AWS Bedrock allows you to access various foundation models (such as **Anthropic Claude**, **AI21 Jurassic-2**, or **Stability AI**) and use them to power your applications. Here are the steps to create and deploy a model on AWS Bedrock:

### **Step 1: AWS Bedrock Access**
1. **Request Access to AWS Bedrock (if required)**:
   - Bedrock might be in preview or limited release, so ensure that your AWS account has access to AWS Bedrock services. If it's in preview, you may need to request access via the AWS Management Console.

2. **Open the AWS Bedrock Console**:
   - Navigate to the AWS Management Console and search for **Bedrock**.
   - Once inside the Bedrock service, you will have access to foundation models (FMs) from various providers such as Anthropic, AI21, Stability AI, and Amazon's own Titan models.

### **Step 2: Choose a Foundation Model**
1. **Select the Model Provider**:
   - AWS Bedrock provides access to different model providers. Choose the one that fits your needs (e.g., **Anthropic Claude** for conversational AI or **AI21 Jurassic-2** for natural language understanding).
   
2. **Select the Specific Model**:
   - Under the selected provider, pick the specific model you want to use. Some examples include:
     - **Claude-v1** or **Claude-v2** (Anthropic)
     - **Jurassic-2** (AI21 Labs)
     - **Stable Diffusion** (Stability AI) for image generation tasks
     - **Amazon Titan** for general-purpose large language model tasks.
   
3. **Review Model Specifications**:
   - Read the model documentation to ensure that the model fits your task (e.g., conversational AI, text generation, or summarization).

### **Step 3: Set Up Model Invocation**
1. **Choose Deployment Options**:
   - In Bedrock, you do not need to manage infrastructure directly. Models are hosted by AWS, and you can invoke them directly using API calls.
   
2. **Configure IAM Roles**:
   - Make sure you have an IAM role with the necessary permissions:
     - **`bedrock:InvokeModel`**: Permission to invoke the model.
     - **`logs:CreateLogGroup`**, **`logs:CreateLogStream`**, and **`logs:PutLogEvents`**: Permissions for CloudWatch Logs to monitor model invocations.
   
3. **Get API Endpoint**:
   - AWS Bedrock will provide an API endpoint that you can use to call the model. This endpoint will be used later in your application (like Lambda functions).



### 7. **Monitoring and Debugging**

   - **CloudWatch Logs**:
     1. When the Lambda function executes, all logs are recorded in **CloudWatch Logs**.
     2. Use CloudWatch to check the execution status, error messages, and whether the SageMaker endpoint calls were successful.

### Summary:

- **Amazon S3**: Stores markdown and PDF files that will be processed.
- **Amazon SageMaker**: Deploys two Hugging Face models:
   1. **Embedding Model**: Used for embedding generation.
   2. **Reranking Model**: Used for reranking the retrieved documents based on relevance to the query.
- **AWS IAM**: Manages permissions for Lambda to access S3, invoke SageMaker endpoints, and interact with Pinecone. It also provides access to CloudWatch logs for monitoring and debugging.
- **Amazon CloudWatch**: Monitors and debugs the execution of both Lambda functions, providing insights into performance and potential issues.
- **Pinecone**: Stores the generated embeddings from the documents, enabling efficient retrieval during queries.
- **AWS Lambda**: Two serverless functions are used:
   1. **Document Processing Lambda Function**: Processes documents, generates embeddings using SageMaker, and stores them in Pinecone.
   2. **Query Handling and Reranking Lambda Function**: Handles user queries, retrieves documents from Pinecone, reranks them using the SageMaker reranking model, and generates the final response with LLM.

### Summary:

- **Amazon S3**: Stores markdown and PDF files that will be processed.
- **Amazon SageMaker**: Deploys two Hugging Face models:
  1. **Embedding Model**: Used for embedding generation.
  2. **Reranking Model**: Used for reranking the retrieved documents based on relevance to the query.
- **AWS Bedrock**: Deploys the language model (LLM) used to generate the final query response. This model can be from providers like **Anthropic Claude**, **AI21 Jurassic-2**, or **Stability AI**.
- **AWS IAM**: Manages permissions for Lambda to access S3, invoke SageMaker and Bedrock endpoints, and interact with Pinecone. It also provides access to CloudWatch logs for monitoring and debugging.
- **Amazon CloudWatch**: Monitors and debugs the execution of both Lambda functions, providing insights into performance and potential issues.
- **Pinecone**: Stores the generated embeddings from the documents, enabling efficient retrieval during queries.
- **AWS Lambda**: Three serverless functions are used:
  1. **Document Processing Lambda Function**: Processes documents, generates embeddings using SageMaker, and stores them in Pinecone.
  2. **Query Handling and Reranking Lambda Function**: Handles user queries, retrieves documents from Pinecone, reranks them using the SageMaker reranking model.
  3. **Bedrock Response Generation Lambda Function**: Uses the AWS Bedrock-deployed LLM to generate the final response based on the reranked query results.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ybai789/Chatbot-with-RAG-on-AWS.git
   cd Chatbot-with-RAG-on-AWS
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a `.env` file in the root directory:

   ```bash
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   SAGEMAKER_EMBEDDING_ENDPOINT_NAME=your_sagemaker_embedding_endpoint_name
   SAGEMAKER_RERANK_ENDPOINT_NAME=your_sagemaker_rerank_endpoint_name
   
   # Bedrock-specific environment variables
   BEDROCK_MODEL_ID=your_bedrock_model_id  # Bedrock model ID, e.g., 'anthropic.claude-v1'
   BEDROCK_REGION=your_aws_region  # AWS region where Bedrock is deployed, e.g., 'us-east-1'
   ```

## Usage

### 1. Build the Knowledge Base

Invoke the **Document Processing Lambda Function** (triggered automatically when new documents are uploaded to S3 or manually via AWS Console/API/CLI).

**Option 1: Automatic S3 Trigger**
- Upload markdown or PDF documents to the designated S3 bucket.
- The **Document Processing Lambda Function** will automatically process the documents, generate embeddings using SageMaker, and store them in Pinecone.

**Option 2: Manual Trigger via AWS CLI**
- Use AWS CLI to invoke the function manually if needed:

```bash
aws lambda invoke --function-name <DocumentProcessingLambdaFunction> \
  --payload '{"bucket_name": "your-bucket", "prefix_md": "md/", "prefix_pdf": "pdf/", "index_name": "your-index"}' response.json
```

### 2. Query the Knowledge Base

Invoke the **Query Handling Lambda Function** using the AWS Console, API Gateway, or AWS CLI:

**Example via AWS CLI**:

```bash
aws lambda invoke --function-name <QueryHandlingLambdaFunction> \
  --payload '{"query": "What is the capital of France?"}' response.json
```

The system will:
- Retrieve relevant documents from Pinecone.
- Rerank them using the SageMaker reranking model.
- Generate the final response using the AWS Bedrock language model.

---

### Summary:

- In a **Lambda-based setup**, the actual Python scripts (`build_knowledge_base_aws.py` and `query_aws.py`) are part of the Lambda functions deployed to AWS. You don't run them directly.
- The **Lambda functions** should be invoked either via triggers (such as S3 events or API Gateway) or manually through AWS CLI/Console.
- The command-line examples should reference how to invoke the Lambda functions instead of running the scripts directly.

If you still want to test these scripts locally, you could keep the original usage, but for **deployment** and **production use**, the usage should reflect the serverless architecture.