from cache import CacheManager
from compliance import ComplianceManager
from integrations import IntegrationManager
from models import DocumentMetadata, ProcessingResult, ProcessingOptions
from typing import Dict, Any, List, Optional, Tuple
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

    async def process_documents(self, file_paths: List[str], user_id: str) -> List[ProcessingResult]:
        """Enhanced processing pipeline with Windows-optimized async handling"""
        results = []
        unique_files = {xxhash.xxh64(f.encode()).hexdigest(): f for f in file_paths}
        
        # Create processing tasks
        tasks = []
        for batch in self._create_batches(list(unique_files.values())):
            task = self._safe_async_operation(
                self._process_batch(batch, user_id),
                f"processing batch of {len(batch)} files"
            )
            tasks.append(task)
        
        # Process with graceful error handling
        try:
            # Use gather with return_exceptions=True for fault tolerance
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error: {str(result)}")
                    continue
                results.extend(result)
                
        except Exception as e:
            self.logger.error(f"Critical error in document processing: {str(e)}")
            raise
        finally:
            # Ensure analytics are updated even if processing fails
            await self._safe_async_operation(
                self._update_analytics(results),
                "updating analytics"
            )
            
        return results

    async def _process_batch(self, file_paths: List[str], user_id: str) -> List[ProcessingResult]:
        """Process batch with resource management"""
        results = []
        tasks = []
        
        for file_path in file_paths:
            task = self._safe_async_operation(
                self._process_single_file(file_path, user_id),
                f"processing file {file_path}"
            )
            tasks.append(task)
        
        try:
            file_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in file_results:
                if isinstance(result, Exception):
                    self.logger.error(f"File processing error: {str(result)}")
                    continue
                results.append(result)
                
        finally:
            # Ensure all resources are released
            for task in tasks:
                task.cancel()
            
        return results

    async def _read_file_async(self, file_path: str) -> str:
        """Read file content asynchronously"""
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            return await f.read()

    async def _process_single_file(self, file_path: str, user_id: str) -> ProcessingResult:
        """Process single file with async I/O"""
        start_time = datetime.now()
        
        try:
            # Read file content asynchronously
            content = await self._read_file_async(file_path)
            
            # Process content
            metadata = await self._extract_metadata(file_path)
            compliance_status = await self._safe_async_operation(
                self.compliance_manager.verify_compliance(metadata),
                f"verifying compliance for {file_path}"
            )
            
            if not compliance_status['compliant']:
                raise ValueError(f"Document fails compliance checks: {compliance_status['reasons']}")
            
            # Enhanced processing
            processed_content = await self._safe_async_operation(
                self._enhance_content({'text': content}, metadata),
                "enhancing content"
            )
            
            # Generate embeddings
            embeddings = await self._safe_async_operation(
                self._generate_embeddings(processed_content['text']),
                "generating embeddings"
            )
            
            # Store results
            await self._safe_async_operation(
                self._store_results(metadata.document_id, processed_content, embeddings),
                "storing results"
            )
            
            return ProcessingResult(
                metadata=metadata,
                content=processed_content,
                embeddings=embeddings,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return self._create_error_result(file_path, str(e), start_time)

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