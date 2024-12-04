from imports import *
from models import DocumentMetadata
import sqlite3
import json
import hashlib
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
class CacheStatistics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_size = 0
        self.queries_cached = 0
        self.cache_evictions = 0
        self.avg_query_time = 0.0

class CacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config.get('cache', {}).get('path', "cache/cache.db"))
        self.db_path.parent.mkdir(exist_ok=True)
        self.max_size_bytes = config.get('cache', {}).get('max_size_mb', 500) * 1024 * 1024  # Default 500MB
        self.stats = CacheStatistics()
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database with enhanced tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enhanced cache table with size tracking
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    size_bytes INTEGER,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Enhanced metadata table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    doc_id TEXT PRIMARY KEY,
                    metadata TEXT,
                    size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Cache statistics table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hits INTEGER,
                    misses INTEGER,
                    total_size_bytes INTEGER,
                    queries_cached INTEGER,
                    evictions INTEGER
                )
                """)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total size
                cursor.execute("SELECT SUM(size_bytes) FROM cache")
                total_size = cursor.fetchone()[0] or 0
                
                # Get item count
                cursor.execute("SELECT COUNT(*) FROM cache")
                item_count = cursor.fetchone()[0]
                
                # Get hit rate
                hit_rate = self.stats.hits / (self.stats.hits + self.stats.misses) if (self.stats.hits + self.stats.misses) > 0 else 0
                
                return {
                    "total_size_mb": total_size / (1024 * 1024),
                    "item_count": item_count,
                    "hit_rate": hit_rate,
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "evictions": self.stats.cache_evictions,
                    "avg_query_time_ms": self.stats.avg_query_time,
                    "space_utilization": (total_size / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0
                }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {}

    async def _enforce_size_limit(self):
        """Enforce cache size limit using LRU policy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current total size
                cursor.execute("SELECT SUM(size_bytes) FROM cache")
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_size_bytes:
                    # Calculate how much space we need to free
                    space_to_free = total_size - (self.max_size_bytes * 0.9)  # Free up 10% extra
                    
                    # Delete least recently accessed items
                    cursor.execute("""
                    DELETE FROM cache 
                    WHERE key IN (
                        SELECT key FROM cache 
                        ORDER BY last_accessed ASC 
                        LIMIT (SELECT COUNT(*) * 10 / 100 FROM cache)
                    )
                    """)
                    
                    evicted_count = cursor.rowcount
                    self.stats.cache_evictions += evicted_count
                    conn.commit()
                    
                    self.logger.info(f"Evicted {evicted_count} items from cache")
                    
        except Exception as e:
            self.logger.error(f"Error enforcing size limit: {str(e)}")

    async def get_cached_query(self, query: str) -> Optional[Dict]:
        """Get cached query result with enhanced monitoring"""
        start_time = datetime.now()
        cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?",
                    (cache_key,)
                )
                result = cursor.fetchone()
                
                if result:
                    value, expires_at = result
                    if expires_at and datetime.fromisoformat(expires_at) > datetime.now():
                        # Update access statistics
                        cursor.execute("""
                        UPDATE cache 
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE key = ?
                        """, (cache_key,))
                        conn.commit()
                        
                        self.stats.hits += 1
                        return json.loads(value)
                    else:
                        cursor.execute("DELETE FROM cache WHERE key = ?", (cache_key,))
                        conn.commit()
                
                self.stats.misses += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {str(e)}")
            return None
        finally:
            # Update average query time
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats.avg_query_time = (self.stats.avg_query_time + query_time) / 2

    async def cache_frequent_queries(self, query: str, result: Dict, ttl: int = 3600):
        """Cache query results with size tracking"""
        cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
        expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        value = json.dumps(result)
        size_bytes = len(value.encode('utf-8'))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, size_bytes, last_accessed, expires_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """,
                    (cache_key, value, size_bytes, expires_at)
                )
                conn.commit()
                
                self.stats.queries_cached += 1
                await self._enforce_size_limit()
                
        except Exception as e:
            self.logger.error(f"Cache storage error: {str(e)}")

    async def cache_document_metadata(self, doc_id: str, metadata: Dict):
        """Store document metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO document_metadata (doc_id, metadata)
                    VALUES (?, ?)
                    """,
                    (doc_id, json.dumps(metadata))
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Metadata storage error: {str(e)}")

    async def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT metadata FROM document_metadata WHERE doc_id = ?",
                    (doc_id,)
                )
                result = cursor.fetchone()
                return json.loads(result[0]) if result else None
                
        except Exception as e:
            self.logger.error(f"Metadata retrieval error: {str(e)}")
            return None

    async def clear_expired_cache(self):
        """Clear expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),)
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {str(e)}")

    async def get_cache_analysis(self) -> Dict[str, Any]:
        """Get detailed cache analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                analysis = {
                    "size_distribution": {},
                    "age_distribution": {},
                    "access_patterns": {},
                    "performance_metrics": await self.get_cache_stats()
                }
                
                # Size distribution
                cursor.execute("""
                SELECT 
                    CASE 
                        WHEN size_bytes < 1024 THEN '< 1KB'
                        WHEN size_bytes < 1048576 THEN '< 1MB'
                        ELSE '≥ 1MB'
                    END as size_category,
                    COUNT(*) as count
                FROM cache
                GROUP BY size_category
                """)
                analysis["size_distribution"] = dict(cursor.fetchall())
                
                # Age distribution
                cursor.execute("""
                SELECT 
                    CASE 
                        WHEN julianday('now') - julianday(created_at) < 1 THEN '< 1 day'
                        WHEN julianday('now') - julianday(created_at) < 7 THEN '< 1 week'
                        ELSE '≥ 1 week'
                    END as age_category,
                    COUNT(*) as count
                FROM cache
                GROUP BY age_category
                """)
                analysis["age_distribution"] = dict(cursor.fetchall())
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error getting cache analysis: {str(e)}")
            return {}