# Document Intelligence Application - Data Model & Architecture

## Overview

This document provides a deep dive into the data model, database schema, and architectural flow of the Document Intelligence application. The application uses PostgreSQL with the pgvector extension for vector similarity search and conversation management.

## Application Flow

### 1. User Authentication & Session Management
- **Entry Point**: User opens the application
- **Authentication**: Verified through Databricks workspace client
- **Database Verification**: User record is created/verified in the database
- **Session State**: User session is established with access to their data

### 2. Conversation Management
- **Existing Conversations**: User sees list of previous conversations
- **New Conversation**: User can start a fresh conversation
- **Resume Conversation**: User can continue previous conversations with full context

### 3. Document Processing & Vector Search
- **Document Upload**: Files are uploaded to Databricks Unity Catalog volumes
- **Async Processing**: Documents are sent to Databricks serverless jobs for AI parsing
- **Chunking & Embedding**: Documents are split into chunks and vectorized
- **Vector Storage**: Chunks and embeddings are stored in PostgreSQL with pgvector

### 4. RAG (Retrieval Augmented Generation)
- **Context Retrieval**: Vector similarity search finds relevant document chunks
- **LLM Integration**: Databricks LLM generates responses using retrieved context
- **Conversation Persistence**: All interactions are stored in the database

## Database Schema

### Core Tables

#### 1. Users Table
**Purpose**: Store user authentication and profile information
**Key Features**: Maps Databricks user IDs to application users

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,                    -- Databricks user ID
    username VARCHAR(255) UNIQUE NOT NULL,     -- Username for display
    email VARCHAR(255),                        -- Optional email
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Relationships**:
- One-to-many with `documents` table
- One-to-many with `conversations` table

**Indexes**:
- Primary key on `id` (Databricks user ID)
- Unique index on `username`

#### 2. Documents Table
**Purpose**: Store document metadata and processing status
**Key Features**: Tracks document lifecycle from upload to processing completion

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_hash VARCHAR(64) UNIQUE NOT NULL,      -- SHA256 hash of file content
    filename VARCHAR(512) NOT NULL,            -- Original filename
    original_path VARCHAR(1024),               -- Databricks volume path
    processed_path VARCHAR(1024),              -- Processed document path
    status VARCHAR(50) DEFAULT 'uploaded',     -- uploaded, processing, processed, failed
    file_size INTEGER,                         -- File size in bytes
    content_type VARCHAR(100),                 -- MIME type
    doc_metadata JSONB,                        -- Additional metadata
    user_id INTEGER NOT NULL,                  -- Foreign key to users.id
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Status Values**:
- `uploaded`: Document uploaded, waiting for processing
- `processing`: Document being processed by Databricks job
- `processed`: Document successfully processed and chunked
- `failed`: Document processing failed

**Relationships**:
- Many-to-one with `users` table
- One-to-many with `document_chunks` table

**Indexes**:
- Primary key on `id`
- Unique index on `doc_hash`
- Composite index on `(user_id, status)` for user document queries
- Index on `created_at` for chronological ordering

#### 3. Document Chunks Table
**Purpose**: Store document text chunks with vector embeddings for semantic search
**Key Features**: Enables vector similarity search across document content

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,                 -- Foreign key to documents.id
    chunk_index INTEGER NOT NULL,              -- Sequential chunk order
    content TEXT NOT NULL,                     -- Text content of the chunk
    embedding VECTOR(768),                     -- Vector embedding (pgvector)
    chunk_metadata JSONB,                      -- Page number, section, etc.
    token_count INTEGER,                       -- Approximate token count
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Vector Dimensions**: 768 (optimized for Databricks GTE-Large embeddings)

**Relationships**:
- Many-to-one with `documents` table

**Indexes**:
- Primary key on `id`
- Index on `document_id` for document-specific queries
- HNSW index on `embedding` for fast similarity search
- IVFFlat index on `embedding` for approximate search

**Vector Indexes**:
```sql
-- HNSW index for high-quality similarity search
CREATE INDEX idx_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat index for approximate search
CREATE INDEX idx_chunks_embedding_ivfflat 
ON document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### 4. Conversations Table
**Purpose**: Manage conversation sessions and thread context
**Key Features**: Links conversations to users and associated documents

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,                  -- Foreign key to users.id
    title VARCHAR(512) NOT NULL,               -- Conversation title
    thread_id VARCHAR(255) UNIQUE NOT NULL,    -- LangGraph thread identifier
    document_ids JSONB,                        -- Array of document UUIDs
    status VARCHAR(50) DEFAULT 'active',       -- active, archived, deleted
    conv_metadata JSONB,                       -- Additional conversation metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Status Values**:
- `active`: Currently active conversation
- `archived`: Archived conversation (read-only)
- `deleted`: Soft-deleted conversation

**Relationships**:
- Many-to-one with `users` table
- One-to-many with `messages` table

**Indexes**:
- Primary key on `id`
- Unique index on `thread_id`
- Composite index on `(user_id, status)` for user conversation queries
- Index on `updated_at` for recent conversation ordering

#### 5. Messages Table
**Purpose**: Store individual chat messages within conversations
**Key Features**: Maintains conversation history with metadata

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,              -- Foreign key to conversations.id
    role VARCHAR(20) NOT NULL,                 -- user, assistant, system
    content TEXT NOT NULL,                     -- Message content
    msg_metadata JSONB,                        -- Tokens, model used, etc.
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Role Values**:
- `user`: User input messages
- `assistant`: AI-generated responses
- `system`: System messages and prompts

**Relationships**:
- Many-to-one with `conversations` table

**Indexes**:
- Primary key on `id`
- Index on `conversation_id` for conversation message queries
- Index on `created_at` for chronological ordering

#### 6. Vector Search Cache Table
**Purpose**: Cache vector search results for improved performance
**Key Features**: Reduces redundant embedding generation and search operations

```sql
CREATE TABLE vector_search_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(64) UNIQUE NOT NULL,    -- Hash of query text
    query_text TEXT NOT NULL,                  -- Original query text
    query_embedding VECTOR(768) NOT NULL,      -- Cached query embedding
    results JSONB NOT NULL,                    -- Cached search results
    user_id INTEGER,                           -- Foreign key to users.id (optional)
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP                        -- Cache expiration time
);
```

**Relationships**:
- Many-to-one with `users` table (optional)

**Indexes**:
- Primary key on `id`
- Unique index on `query_hash`
- Index on `expires_at` for cache cleanup
- Index on `user_id` for user-specific cache queries

## Data Flow Architecture

### 1. Document Processing Pipeline

```
User Upload → Storage Service → Document Service → Databricks Job → Database Storage
     ↓              ↓              ↓              ↓              ↓
File Content → Volume Storage → Processing Queue → AI Parsing → Chunks + Embeddings
```

**Steps**:
1. **Upload**: File stored in Databricks Unity Catalog volume
2. **Queue**: Document processing job queued in Databricks
3. **Processing**: AI parsing extracts text and structure
4. **Chunking**: Document split into semantic chunks
5. **Embedding**: Vector embeddings generated for each chunk
6. **Storage**: Chunks and embeddings stored in PostgreSQL

### 2. Conversation Flow

```
User Input → Vector Search → Context Retrieval → LLM Generation → Response Storage
     ↓            ↓              ↓              ↓              ↓
Message → Query Embedding → Similarity Search → Context + History → AI Response
```

**Steps**:
1. **Input**: User sends message
2. **Search**: Query embedded and vector similarity search performed
3. **Retrieval**: Relevant document chunks retrieved
4. **Generation**: LLM generates response using context
5. **Storage**: Message and response stored in database

### 3. RAG Integration

```
Query → Embedding → Vector Search → Document Retrieval → Context Building → LLM Response
  ↓         ↓          ↓              ↓              ↓              ↓
Text → Vector → Similarity → Top Chunks → Prompt Context → Generated Text
```

**Components**:
- **Embedding Model**: Databricks GTE-Large for text vectorization
- **Vector Store**: PostgreSQL with pgvector extension
- **Search Algorithm**: Cosine similarity with HNSW indexing
- **LLM**: Databricks Claude Sonnet 4 for response generation

## Database Extensions

### pgvector Extension
The application requires the pgvector extension for vector operations:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Supported Operations**:
- Vector similarity search (cosine, L2, inner product)
- HNSW indexing for fast approximate search
- IVFFlat indexing for approximate search with configurable accuracy

## Performance Considerations

### 1. Vector Search Optimization
- **HNSW Index**: Provides fast approximate similarity search
- **IVFFlat Index**: Alternative indexing for different accuracy/speed trade-offs
- **Chunk Size**: Optimal chunk size of 500-1000 tokens for search relevance

### 2. Database Performance
- **Connection Pooling**: SQLAlchemy connection pool with 10-20 connections
- **Index Strategy**: Composite indexes for common query patterns
- **Query Optimization**: Efficient joins and filtering for user-specific data

### 3. Caching Strategy
- **Vector Cache**: Cache query embeddings and results
- **Session State**: Maintain conversation context in memory
- **Document Cache**: Cache frequently accessed document metadata

## Security & Access Control

### 1. User Isolation
- **Row-Level Security**: Users can only access their own data
- **Foreign Key Constraints**: Enforce data integrity across tables
- **Authentication**: Databricks workspace authentication required

### 2. Data Privacy
- **Document Isolation**: Users cannot access other users' documents
- **Conversation Privacy**: Conversations are user-specific
- **Audit Trail**: All operations logged with timestamps

## Migration & Schema Evolution

### 1. Schema Versioning
- **Version Tracking**: Track schema changes in metadata
- **Backward Compatibility**: Maintain compatibility with existing data
- **Migration Scripts**: Automated schema updates

### 2. Data Migration
- **Vector Migration**: Handle embedding dimension changes
- **Index Rebuilding**: Rebuild indexes after schema changes
- **Data Validation**: Verify data integrity after migrations

## Monitoring & Maintenance

### 1. Performance Monitoring
- **Query Performance**: Monitor slow queries and optimize
- **Vector Search**: Track search latency and accuracy
- **Storage Usage**: Monitor database size and growth

### 2. Maintenance Tasks
- **Index Maintenance**: Regular index rebuilding and optimization
- **Cache Cleanup**: Remove expired cache entries
- **Data Archiving**: Archive old conversations and documents

## Future Enhancements

### 1. Advanced Vector Features
- **Multi-Modal Embeddings**: Support for images and other media types
- **Hybrid Search**: Combine vector and keyword search
- **Semantic Clustering**: Group similar documents automatically

### 2. Scalability Improvements
- **Sharding**: Distribute data across multiple database instances
- **Read Replicas**: Scale read operations with database replicas
- **CDN Integration**: Cache document content for faster access

### 3. Advanced Analytics
- **Usage Analytics**: Track user behavior and document usage
- **Search Analytics**: Analyze search patterns and improve relevance
- **Performance Metrics**: Monitor system performance and bottlenecks
