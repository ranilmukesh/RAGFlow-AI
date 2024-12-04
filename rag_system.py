from imports import *
from processor import EnhancedDocumentProcessor
from compliance import ComplianceManager
from integrations import IntegrationManager
from cache import CacheManager
from models import DocumentMetadata, ProcessingResult, AnalyticsMetrics

class RAGSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = EnhancedDocumentProcessor(
            claude_key=config.get('claude_api_key'),
            config=config,
            batch_size=config['processing']['batch_size'],
            max_concurrent=config['processing']['max_concurrent']
        )
        self.integration_manager = IntegrationManager(config)
        self.compliance_manager = ComplianceManager(config)
        self.cache_manager = CacheManager(config['redis'])
        
    async def process_documents(self, files: List[str], user_id: str) -> List[ProcessingResult]:
        return await self.processor.process_documents(files, user_id)
        
    async def query(self, query: str, **kwargs) -> Dict:
        return await self.processor.query(query, **kwargs)
        
    async def get_analytics(self) -> AnalyticsMetrics:
        return self.processor.metrics