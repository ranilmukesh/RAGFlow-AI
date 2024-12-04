from dataclasses import dataclass, field, fields
from typing import Dict, Any, List, Optional
import hashlib
from datetime import datetime
from imports import *
@dataclass
class DocumentMetadata:
    file_path: str
    file_type: str
    creation_date: datetime
    last_modified: datetime
    author: str
    version: str
    department: str
    tags: List[str]
    language: str
    confidence_score: float
    processing_status: str
    document_id: str = field(default_factory=lambda: hashlib.sha256().hexdigest())
    checksum: str = ""
    size_bytes: int = 0
    page_count: int = 0
    classification: str = ""
    retention_period: int = 0
    access_level: str = "public"
    last_accessed: datetime = field(default_factory=datetime.now)
    processing_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format"""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentMetadata':
        """Create metadata instance from dictionary"""
        return cls(**{
            k: v for k, v in data.items()
            if k in {f.name for f in fields(cls)}
        })
    
    def update(self, **kwargs) -> None:
        """Update metadata fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """Validate metadata fields"""
        required_fields = ['file_path', 'file_type', 'creation_date', 'author']
        return all(getattr(self, field) for field in required_fields)

@dataclass
class ProcessingMetrics:
    docs_processed: int = 0
    charts_processed: int = 0
    tables_processed: int = 0
    processing_time: float = 0.0
    failed_docs: List[str] = field(default_factory=list)
    processing_dates: List[datetime] = field(default_factory=list)

@dataclass
class AnalyticsMetrics(ProcessingMetrics):
    processing_errors: Dict[str, int] = field(default_factory=dict)
    average_processing_time: float = 0.0
    document_types_distribution: Dict[str, int] = field(default_factory=dict)
    storage_usage: float = 0.0
    query_latency: List[float] = field(default_factory=list)
    user_feedback_scores: List[float] = field(default_factory=list)
    daily_usage_stats: Dict[str, int] = field(default_factory=dict)
    error_distribution: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class AuditLog:
    timestamp: datetime
    user_id: str
    action: str
    document_id: str
    changes: Dict
    ip_address: str = ""
    session_id: str = ""
    status: str = "success"
    details: Dict = field(default_factory=dict)

@dataclass
class ProcessingResult:
    metadata: DocumentMetadata
    content: Dict[str, Any]
    embeddings: np.ndarray
    error: Optional[str] = None
    processing_time: float = 0.0
    status: str = "success"

@dataclass
class ProcessingOptions:
    language_detection: bool = True
    extract_citations: bool = True
    generate_summaries: bool = True
    detect_duplicates: bool = True
    cache_llm_results: bool = True
    ocr_enabled: bool = True
    chart_analysis: bool = True
    table_extraction: bool = True
    entity_recognition: bool = True
    sentiment_analysis: bool = False
    batch_size: int = 64
    max_concurrent: int = 8
    confidence_threshold: float = 0.7