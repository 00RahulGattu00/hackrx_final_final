import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import tempfile

from extract import extract_text_from_pdf, validate_pdf, get_pdf_info
from search import DocumentProcessor, SemanticSearch
from llm_processor import LLMProcessor
from decision_engine import DecisionEngine

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    """Centralized configuration"""
    def __init__(self):
        self.bearer_token = os.getenv('BEARER_TOKEN')
        if not self.bearer_token:
            raise ValueError("BEARER_TOKEN environment variable is required")
        
        self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
        self.max_questions = int(os.getenv('MAX_QUESTIONS', '10'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '180'))
        
        # Processing settings
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '500'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
        self.top_k_chunks = int(os.getenv('TOP_K_CHUNKS', '5'))

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Global components with proper lifecycle management
class ComponentManager:
    def __init__(self):
        self.document_processor: Optional[DocumentProcessor] = None
        self.semantic_search: Optional[SemanticSearch] = None
        self.llm_processor: Optional[LLMProcessor] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing system components...")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            # Initialize processing components
            settings = get_settings()
            self.document_processor = DocumentProcessor(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            self.semantic_search = SemanticSearch()
            self.llm_processor = LLMProcessor()
            self.decision_engine = DecisionEngine(self.llm_processor)
            
            # Log configuration
            api_count = len(self.llm_processor.apis) if self.llm_processor.apis else 0
            logger.info(f"Successfully initialized all components with {api_count} LLM APIs")
            
            if api_count == 0:
                logger.warning("No LLM APIs configured - system will use fallback mode")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Components cleaned up")

# Global component manager
component_manager = ComponentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await component_manager.initialize()
    yield
    # Shutdown
    await component_manager.cleanup()

# FastAPI app with lifespan management
app = FastAPI(
    title="LLM-Powered Query-Retrieval System",
    version="1.0.0",
    description="AI-powered document analysis with multi-API fallback support",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to analyze")
    questions: List[str] = Field(..., min_items=1, max_items=10, description="List of questions to answer")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers to the submitted questions")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional processing metadata")

class HealthResponse(BaseModel):
    status: str
    service: str
    components: Dict[str, str]
    api_endpoints: int
    processing_time: float
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: float

# Dependency injection
def get_components() -> ComponentManager:
    if not component_manager._initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    return component_manager

async def verify_authorization(authorization: str = Header(None), settings: Settings = Depends(get_settings)):
    """Verify bearer token authorization"""
    expected_token = f"Bearer {settings.bearer_token}"
    if not authorization or authorization != expected_token:
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing authorization token"
        )
    return True

# Enhanced error handling
class DocumentProcessingError(Exception):
    pass

class ValidationError(Exception):
    pass

@app.exception_handler(DocumentProcessingError)
async def handle_document_error(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(ValidationError)
async def handle_validation_error(request, exc):
    return HTTPException(status_code=422, detail=str(exc))

# Main endpoint
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    req: QueryRequest,
    _: bool = Depends(verify_authorization),
    components: ComponentManager = Depends(get_components),
    settings: Settings = Depends(get_settings)
):
    """
    Main endpoint for processing document queries using LLM-powered retrieval system
    """
    start_time = time.time()
    
    try:
        # Validate request
        await validate_request(req, settings)
        
        logger.info(f"Processing {len(req.questions)} questions for document: {req.documents}")
        
        # Step 1: Download and extract document content
        document_content = await download_and_extract_document(req.documents, components.http_client, settings)
        
        # Step 2: Process document into chunks and build search index
        chunks = components.document_processor.create_chunks(document_content)
        if not chunks:
            raise DocumentProcessingError("No content chunks could be created from the document")
        
        search_index = components.semantic_search.build_index(chunks)
        
        # Step 3: Process each question
        answers = await process_questions(req.questions, chunks, search_index, components, settings)
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed all {len(req.questions)} questions in {processing_time:.2f}s")
        
        return QueryResponse(
            answers=answers,
            processing_time=processing_time,
            metadata={
                "document_chunks": len(chunks),
                "total_characters": len(document_content),
                "api_endpoints_available": len(components.llm_processor.apis)
            }
        )
        
    except (DocumentProcessingError, ValidationError) as e:
        processing_time = time.time() - start_time
        logger.error(f"Validation/Processing error after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in run_query after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def validate_request(req: QueryRequest, settings: Settings):
    """Validate incoming request"""
    if not req.documents or not req.documents.strip():
        raise ValidationError("Document URL is required")
    
    if not req.documents.startswith(('http://', 'https://')):
        raise ValidationError("Invalid URL format - must start with http:// or https://")
    
    if len(req.questions) > settings.max_questions:
        raise ValidationError(f"Maximum {settings.max_questions} questions allowed per request")
    
    for i, question in enumerate(req.questions):
        if not question or not question.strip():
            raise ValidationError(f"Question {i + 1} cannot be empty")

async def download_and_extract_document(document_url: str, http_client: httpx.AsyncClient, settings: Settings) -> str:
    """
    Download document from URL and extract text content with robust error handling
    """
    temp_path = None
    try:
        logger.info(f"Downloading document from: {document_url}")
        
        # Download document with streaming
        async with http_client.stream('GET', document_url) as response:
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not document_url.lower().endswith('.pdf'):
                logger.warning(f"Content type '{content_type}' may not be PDF")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                total_size = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    temp_file.write(chunk)
                    total_size += len(chunk)
                    
                    # Prevent downloading extremely large files
                    if total_size > settings.max_file_size_mb * 1024 * 1024:
                        raise DocumentProcessingError(f"Document too large (>{settings.max_file_size_mb}MB)")
                
                temp_path = temp_file.name
        
        logger.info(f"Downloaded {total_size / 1024:.1f}KB to temporary file")
        
        # Validate PDF
        if not validate_pdf(temp_path):
            raise DocumentProcessingError("Downloaded file is not a valid PDF")
        
        # Get PDF info for logging
        pdf_info = get_pdf_info(temp_path)
        logger.info(f"PDF info: {pdf_info.get('page_count', 0)} pages")
        
        # Extract text
        document_content = extract_text_from_pdf(temp_path)
        
        if not document_content.strip():
            raise DocumentProcessingError("No text content extracted from document")
        
        logger.info(f"Extracted {len(document_content)} characters from document")
        return document_content
        
    except httpx.TimeoutException:
        logger.error("Document download timed out")
        raise DocumentProcessingError("Document download timed out")
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading document: {e.response.status_code}")
        raise DocumentProcessingError(f"Failed to download document: HTTP {e.response.status_code}")
        
    except Exception as e:
        logger.error(f"Failed to process document: {str(e)}")
        raise DocumentProcessingError(f"Failed to process document: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug("Cleaned up temporary file")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

async def process_questions(questions: List[str], chunks: List[Dict], search_index, components: ComponentManager, settings: Settings) -> List[str]:
    """Process all questions and return answers"""
    answers = []
    
    for i, question in enumerate(questions):
        try:
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            # Parse and understand the query using LLM
            parsed_query = components.llm_processor.parse_query(question)
            
            # Retrieve relevant chunks using semantic search
            relevant_chunks = components.semantic_search.search(
                search_index, question, chunks, top_k=settings.top_k_chunks
            )
            
            # Generate answer with explainable reasoning
            answer_result = components.decision_engine.generate_answer(
                question=question,
                parsed_query=parsed_query,
                context_chunks=relevant_chunks
            )
            
            # Extract just the answer text for the response
            answer_text = answer_result.get("answer", "Unable to generate answer")
            answers.append(answer_text)
            
            logger.info(f"Successfully processed question {i+1}")
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {str(e)}")
            error_msg = f"Unable to process question: {str(e)[:100]}"
            answers.append(error_msg)
    
    return answers

@app.get("/health", response_model=HealthResponse)
async def health_check(components: ComponentManager = Depends(get_components)):
    """Comprehensive health check endpoint with component status"""
    start_time = time.time()
    
    try:
        # Test document processor
        doc_status = await test_component_health(
            lambda: components.document_processor.create_chunks("Health check test document."),
            "document_processor"
        )
        
        # Test search engine
        search_status = await test_component_health(
            lambda: test_search_functionality(components),
            "semantic_search"
        )
        
        # Test LLM processor
        llm_status, api_count = await test_llm_health(components.llm_processor)
        
        # Test decision engine
        decision_status = "healthy" if llm_status != "unhealthy" else "degraded"
        
        # Overall status
        all_statuses = [doc_status, search_status, llm_status, decision_status]
        overall_status = determine_overall_status(all_statuses)
        
        processing_time = time.time() - start_time
        
        return HealthResponse(
            status=overall_status,
            service="LLM Query-Retrieval System",
            components={
                "document_processor": doc_status,
                "semantic_search": search_status,
                "llm_processor": llm_status,
                "decision_engine": decision_status,
                "http_client": "healthy" if components.http_client else "unhealthy"
            },
            api_endpoints=api_count,
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            service="LLM Query-Retrieval System",
            components={"error": str(e)},
            api_endpoints=0,
            processing_time=round(processing_time, 3)
        )

async def test_component_health(test_func, component_name: str) -> str:
    """Test individual component health"""
    try:
        result = test_func()
        if result:
            return "healthy"
        else:
            return "degraded"
    except Exception as e:
        logger.warning(f"{component_name} health check failed: {str(e)}")
        return "unhealthy"

def test_search_functionality(components: ComponentManager) -> bool:
    """Test search functionality"""
    test_chunks = components.document_processor.create_chunks("Test document for search functionality.")
    if not test_chunks:
        return False
    
    test_index = components.semantic_search.build_index(test_chunks)
    test_results = components.semantic_search.search(test_index, "test query", test_chunks, top_k=1)
    return bool(test_results)

async def test_llm_health(llm_processor: LLMProcessor) -> tuple[str, int]:
    """Test LLM processor health"""
    api_count = len(llm_processor.apis) if llm_processor.apis else 0
    
    if api_count == 0:
        return "degraded", 0
    
    try:
        # Test fallback parsing (doesn't require API)
        test_result = llm_processor._fallback_parse_query("test query")
        if test_result:
            return "healthy", api_count
        else:
            return "degraded", api_count
    except Exception:
        return "unhealthy", api_count

def determine_overall_status(statuses: List[str]) -> str:
    """Determine overall system status"""
    if all(s == "healthy" for s in statuses):
        return "healthy"
    elif any(s == "unhealthy" for s in statuses):
        return "unhealthy"
    else:
        return "degraded"

@app.get("/api-status")
async def api_status(components: ComponentManager = Depends(get_components)):
    """Check status and statistics of available LLM APIs"""
    try:
        llm_processor = components.llm_processor
        
        if not hasattr(llm_processor, 'apis'):
            return {"error": "LLM processor not properly initialized"}
        
        api_info = []
        for api in llm_processor.apis:
            api_info.append({
                "id": api.get('id', 'unknown'),
                "type": api['type'],
                "model": api['model'],
                "priority": api['priority'],
                "configured": True
            })
        
        # Get usage statistics if available
        stats = {}
        if hasattr(llm_processor, 'get_api_stats'):
            stats = llm_processor.get_api_stats()
        
        return {
            "total_apis": len(llm_processor.apis),
            "apis": api_info,
            "fallback_available": len(llm_processor.apis) > 1,
            "usage_stats": stats.get('api_stats', {}),
            "system_ready": len(llm_processor.apis) > 0
        }
        
    except Exception as e:
        logger.error(f"Error getting API status: {str(e)}")
        return {"error": f"Failed to get API status: {str(e)}"}

@app.get("/")
async def root():
    """Root endpoint with basic system information"""
    return {
        "service": "LLM-Powered Query-Retrieval System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "api_status": "/api-status",
            "metrics": "/metrics"
        },
        "description": "AI-powered document analysis with multi-API fallback support"
    }

@app.get("/metrics")
async def get_metrics(components: ComponentManager = Depends(get_components)):
    """Get system metrics and performance data"""
    try:
        llm_processor = components.llm_processor
        
        metrics = {
            "system": {
                "total_apis": len(llm_processor.apis) if hasattr(llm_processor, 'apis') else 0,
                "components_initialized": component_manager._initialized
            }
        }
        
        # Add API statistics if available
        if hasattr(llm_processor, 'get_api_stats'):
            api_stats = llm_processor.get_api_stats()
            metrics["api_usage"] = api_stats.get('api_stats', {})
        
        # Add decision engine stats if available
        if hasattr(components.decision_engine, 'get_stats'):
            metrics["decision_engine"] = components.decision_engine.get_stats()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {"error": f"Failed to get metrics: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    
    # Startup checks
    try:
        settings = get_settings()
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.info("Please check your environment variables and .env file")
        exit(1)
    
    # Start server
    logger.info("🚀 Starting LLM Query-Retrieval System server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)