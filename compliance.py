from models import DocumentMetadata, AuditLog
from imports import *

class ComplianceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audit_collection = "audit_logs"

    async def _store_audit_log(self, audit_entry: AuditLog):
        try:
            log_data = {
                'timestamp': audit_entry.timestamp,
                'user_id': audit_entry.user_id,
                'action': audit_entry.action,
                'document_id': audit_entry.document_id,
                'changes': audit_entry.changes,
                'ip_address': audit_entry.ip_address,
                'session_id': audit_entry.session_id,
                'status': audit_entry.status,
                'details': audit_entry.details
            }
            
            self.logger.info(f"Stored audit log: {log_data}")
            
        except Exception as e:
            self.logger.error(f"Error storing audit log: {str(e)}")
            raise
