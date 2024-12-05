from cache import CacheManager
from compliance import ComplianceManager
from integrations import IntegrationManager
from models import DocumentMetadata, ProcessingResult, ProcessingOptions
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
import anthropic
import multiprocessing
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import hashlib
from langdetect import detect
from transformers import TrOCRProcessor
import numpy as np
from torch.cuda.amp import autocast
from imports import *
from models import *
import aiofiles
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import CrossEncoder
from torch.cuda.amp import autocast
import asyncio
import time

class EnhancedDocumentProcessor:
    def __init__(
        self,
        claude_key: str,
        config: Dict[str, Any],
        batch_size: int = 64,
        max_concurrent: int = multiprocessing.cpu_count()
    ):
        self.config = config
        self.setup_models()
        self.claude = anthropic.Anthropic(api_key=claude_key)
        self.qdrant = QdrantClient(path="vectors.db")
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.metrics = AnalyticsMetrics()
        self.cache_manager = CacheManager(config['redis'])
        self.integration_manager = IntegrationManager(config)
        self.compliance_manager = ComplianceManager(config)
        self.processing_options = ProcessingOptions()
        self.logger = logging.getLogger(__name__)
        
        # Initialize file type handlers
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.pptx': self._process_pptx,
            '.xlsx': self._process_excel,
            '.html': self._process_html,
            '.eml': self._process_email,
            '.msg': self._process_email,
            '.txt': self._process_text
        }
        
        # Initialize NLP
        self.nlp = spacy.load("en_core_web_lg")
        
        # Performance optimizations
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Initialize semaphore for concurrent operations
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.task_queue = asyncio.Queue()
        
    def setup_models(self):
        """Initialize all required models with GPU optimization"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Object Detection
            self.object_detector = YOLO('yolov8x.pt')
            
            # Chart Understanding
            self.vlm = AutoModel.from_pretrained("microsoft/git-base")
            self.vlm_processor = AutoProcessor.from_pretrained("microsoft/git-base")
            
            # Table Extraction
            self.table_ocr = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
            self.table_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
            
            # Embedding Model
            self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            
            # Move models to GPU and optimize
            for model in [self.vlm, self.table_model, self.embedding_model]:
                model.to(self.device)
                if self.device.type == "cuda":
                    model.half()  # FP16 for efficiency
                    
        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}")
            raise

    async def _safe_async_operation(self, coroutine, error_message: str):
        """Wrapper for safe async operations with error handling"""
        try:
            async with self.semaphore:
                return await coroutine
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout: {error_message}")
            raise TimeoutError(f"Operation timed out: {error_message}")
        except asyncio.CancelledError:
            self.logger.warning(f"Operation cancelled: {error_message}")
            raise
        except Exception as e:
            self.logger.error(f"Error in {error_message}: {str(e)}")
            raise

    async def process_documents(
        self,
        files: List[str],
        user_id: str,
        batch_size: int = 64
    ) -> List[ProcessingResult]:
        try:
            # Group files by type for optimized batch processing
            grouped_files = self._group_files_by_type(files)
            results = []
            
            # Process each group with specialized handlers
            for file_type, file_group in grouped_files.items():
                handler = self.supported_formats.get(file_type)
                if not handler:
                    continue
                    
                # Process in batches
                for batch in self._create_batches(file_group, batch_size):
                    batch_results = await self._process_batch(
                        batch,
                        handler,
                        user_id
                    )
                    results.extend(batch_results)
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Document processing error: {str(e)}")
            raise
            
    async def _process_batch(
        self,
        batch: List[str],
        handler: Callable,
        user_id: str
    ) -> List[ProcessingResult]:
        tasks = []
        async with ProcessPoolExecutor(max_workers=self.max_concurrent) as executor:
            for file_path in batch:
                task = asyncio.create_task(
                    self._process_single_file(
                        file_path,
                        handler,
                        executor,
                        user_id
                    )
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and log them
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Processing error: {str(result)}")
                    continue
                processed_results.append(result)
                
            return processed_results
            
    async def _process_single_file(
        self,
        file_path: str,
        handler: Callable,
        executor: ProcessPoolExecutor,
        user_id: str
    ) -> ProcessingResult:
        try:
            # Check cache first
            cache_key = self._generate_cache_key(file_path, user_id)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
                
            # Process file content
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                executor,
                handler,
                file_path
            )
            
            # Extract metadata
            metadata = await self._extract_metadata(file_path, content)
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(content)
            
            # Create result
            result = ProcessingResult(
                metadata=metadata,
                content=content,
                embeddings=embeddings,
                processing_time=time.time(),
                status="success"
            )
            
            # Cache result
            await self.cache_manager.set(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def _create_error_result(self, file_path: str, error: str, start_time: datetime) -> ProcessingResult:
        """Create standardized error result"""
        return ProcessingResult(
            metadata=DocumentMetadata(file_path=file_path, error=error),
            content={},
            embeddings=np.array([]),
            error=error,
            status="failed",
            processing_time=(datetime.now() - start_time).total_seconds()
        )

    async def _enhance_content(self, content: Dict, metadata: DocumentMetadata) -> Dict:
        """Apply enhanced processing features"""
        enhanced = {'original': content}
        
        if self.processing_options.language_detection:
            enhanced['language'] = detect(content['text'])
            
        if self.processing_options.extract_citations:
            enhanced['citations'] = await self._extract_citations(content['text'])
            
        if self.processing_options.generate_summaries:
            enhanced['summary'] = await self._generate_summary(content['text'])
            
        if self.processing_options.detect_duplicates:
            enhanced['duplicate_check'] = await self._check_duplicates(content['text'])
            
        # Extract entities and relationships
        doc = self.nlp(content['text'])
        enhanced['entities'] = [
            {'text': ent.text, 'label': ent.label_} 
            for ent in doc.ents
        ]
        
        return enhanced

    async def query(
        self,
        query: str,
        filters: Optional[Dict] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        departments: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        include_metadata: bool = False,
        aggregation: Optional[str] = None,
        user_id: str = None
    ) -> Dict:
        """Enhanced query with filtering and aggregation"""
        try:
            # Check cache first
            cached_result = await self.cache_manager.get_cached_query(query)
            if cached_result:
                return cached_result
            
            # Generate query embedding
            with torch.no_grad(), autocast(enabled=True):
                query_embedding = self.embedding_model.encode(query)
            
            # Build search filters
            search_filters = self._build_search_filters(
                filters, date_range, departments, document_types
            )
            
            # Initial retrieval
            initial_results = self.qdrant.search(
                collection_name="docs",
                query_vector=query_embedding,
                query_filter=search_filters,
                limit=self.config['search']['initial_results_multiplier'] * self.config['search']['top_k']
            )
            
            # Rerank results
            reranked_results = await self._rerank_results(query, initial_results)
            top_results = reranked_results[:self.config['search']['top_k']]
            
            # Filter by confidence
            top_results = [r for r in top_results if r.score >= min_confidence]
            
            # Get context from top results
            context = await self._get_context(top_results)
            
            # Get LLM response
            response = await self._get_llm_response(query, context)
            
            result = {
                "answer": response,
                "sources": [hit.payload["path"] for hit in top_results],
                "confidence_scores": [hit.score for hit in top_results]
            }
            
            if include_metadata:
                result["metadata"] = await self._get_results_metadata(top_results)
                
            if aggregation:
                result["aggregations"] = await self._aggregate_results(top_results, aggregation)
                
            # Cache result
            await self.cache_manager.cache_frequent_queries(query, result)
            
            # Log query
            if user_id:
                await self.compliance_manager.log_access(
                    document_id="query",
                    user_id=user_id,
                    action="query",
                    metadata={'query': query}
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    async def process_with_llm(self, content: str, llm_name: str, model: str = None) -> Dict:
        """Process content using specified LLM through integration manager"""
        try:
            # Use the integration manager's process_with_llm method directly
            response = await self.integration_manager.process_with_llm(
                content=content,
                llm_name=llm_name,
                model=model
            )
            
            # Update processing metrics
            processing_time = response.get('processing_time', 0)
            self.metrics.processing_time += processing_time
            self.metrics.performance_metrics[f'llm_{llm_name}_time'] = processing_time
            
            # Log the LLM usage for compliance
            await self.compliance_manager.log_access(
                document_id="llm_process",
                user_id=self.config.get('user_id', 'system'),
                action=f"llm_process_{llm_name}",
                metadata={
                    'model': response.get('model'),
                    'provider': response.get('provider'),
                    'processing_time': processing_time
                }
            )
            
            # Cache the result if needed
            if self.processing_options.cache_llm_results:
                cache_key = f"llm_{llm_name}_{hashlib.md5(content.encode()).hexdigest()}"
                await self.cache_manager.cache_document_metadata(cache_key, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing with LLM: {str(e)}")
            # Update error metrics
            self.metrics.processing_errors[f'llm_{llm_name}_error'] = \
                self.metrics.processing_errors.get(f'llm_{llm_name}_error', 0) + 1
            raise

    async def _generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using a process pool for CPU-bound tasks"""
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor() as executor:
            embeddings = await loop.run_in_executor(executor, self.embedding_model.encode, text)
        return embeddings

    def setup_vector_search(self):
        # Initialize multiple embedding models for different content types
        self.embedders = {
            'default': SentenceTransformer('BAAI/bge-large-en-v1.5'),
            'code': SentenceTransformer('microsoft/codebert-base'),
            'multilingual': SentenceTransformer('intfloat/multilingual-e5-large')
        }
        
        # Initialize Qdrant with optimized settings
        self.qdrant = QdrantClient(
            path="vectors.db",
            optimize_storage=True,
            timeout=60.0
        )
        
        # Create collections with different vector configs
        self.collections = {
            'default': {
                'name': 'default_vectors',
                'vector_size': 1024,
                'distance': 'Cosine'
            },
            'code': {
                'name': 'code_vectors',
                'vector_size': 768,
                'distance': 'Dot'
            }
        }
        
    async def hybrid_search(
        self,
        query: str,
        collection: str = 'default',
        top_k: int = 5,
        threshold: float = 0.7,
        rerank: bool = True
    ) -> List[Dict]:
        try:
            # Generate query embedding
            query_embedding = self.embedders[collection].encode(query)
            
            # Vector search
            vector_results = await self.qdrant.search(
                collection_name=self.collections[collection]['name'],
                query_vector=query_embedding,
                limit=top_k * 2  # Get more results for re-ranking
            )
            
            if not rerank:
                return vector_results[:top_k]
            
            # Re-ranking with cross-encoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
            
            rerank_pairs = [
                (query, self.get_document_text(hit.id)) 
                for hit in vector_results
            ]
            
            rerank_scores = reranker.predict(rerank_pairs)
            
            # Combine vector similarity and reranking scores
            results = []
            for idx, (hit, rerank_score) in enumerate(zip(vector_results, rerank_scores)):
                if rerank_score > threshold:
                    results.append({
                        'id': hit.id,
                        'score': (0.7 * hit.score + 0.3 * rerank_score),
                        'metadata': hit.metadata,
                        'content': self.get_document_text(hit.id)
                    })
            
            # Sort by combined score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {str(e)}")
            raise