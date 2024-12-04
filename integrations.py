from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import aiohttp
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
from google.cloud import storage, bigquery
from simple_salesforce import Salesforce
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from openai import OpenAI as openai
import groq
from anthropic import Anthropic as anthropic

class IntegrationManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_clients()
        
    def setup_clients(self):
        try:
            # SharePoint setup
            self.sharepoint_context = self.connect_sharepoint(self.config['sharepoint_url'], self.config['sharepoint_client_id'], self.config['sharepoint_client_secret'])
            
            # Salesforce setup
            self.sf = Salesforce(
                username=self.config['sf_username'],
                password=self.config['sf_password'],
                security_token=self.config['sf_token']
            )
            
            # GCP setup
            self.storage_client = storage.Client()
            self.bigquery_client = bigquery.Client()
            
            # OpenAI setup
            self.openai_client = openai.Client(api_key=self.config['openai_api_key'])
            
            # Groq setup - using official client
            self.groq_client = groq.Client(api_key=self.config['groq_api_key'])
            
            # Claude setup (if not already initialized)
            self.claude = anthropic.Anthropic(api_key=self.config['claude_api_key'])
            
        except Exception as e:
            self.logger.error(f"Failed to setup integration clients: {str(e)}")
            raise
            
    async def connect_sharepoint(self, url: str, username: str, password: str):
        try:
            ctx = ClientContext(url).with_credentials(
                ClientCredential(username, password)
            )
            return ctx
        except Exception as e:
            self.logger.error(f"SharePoint connection error: {str(e)}")
            raise
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def sync_with_sharepoint(self, library_name: str) -> List[Dict]:
        """Sync documents from SharePoint library with retry logic"""
        try:
            target_library = self.sharepoint_context.web.lists.get_by_title(library_name)
            items = target_library.items.get().execute_query()
            
            return [{
                'url': item.properties.get('FileRef'),
                'metadata': {
                    'created': item.properties.get('Created'),
                    'modified': item.properties.get('Modified'),
                    'author': item.properties.get('Author'),
                    'size': item.properties.get('Size')
                }
            } for item in items if item.properties.get('FileRef')]
            
        except Exception as e:
            self.logger.error(f"SharePoint sync error: {str(e)}")
            raise
            
    async def sync_with_salesforce(self, object_name: str, query_fields: List[str]) -> List[Dict]:
        """Sync documents from Salesforce"""
        try:
            fields = ', '.join(query_fields)
            query = f"SELECT {fields} FROM {object_name}"
            return self.sf.query(query)['records']
        except Exception as e:
            self.logger.error(f"Salesforce sync error: {str(e)}")
            return []
            
    async def export_to_data_warehouse(self, data: List[Dict], table_name: str):
        """Export processed data to BigQuery"""
        try:
            dataset_ref = self.bigquery_client.dataset(self.config['bigquery_dataset'])
            table_ref = dataset_ref.table(table_name)
            
            job_config = bigquery.LoadJobConfig()
            job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            job_config.schema_update_options = [
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]
            
            job = self.bigquery_client.load_table_from_json(
                data,
                table_ref,
                job_config=job_config
            )
            job.result()
            
        except Exception as e:
            self.logger.error(f"BigQuery export error: {str(e)}")
            raise
            
    async def webhook_notify(self, event: str, payload: Dict):
        """Notify external systems via webhook"""
        webhook_url = self.config['webhook_url']
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(webhook_url, json={
                    'event': event,
                    'payload': payload,
                    'timestamp': datetime.now().isoformat()
                }) as response:
                    return await response.json()
            except Exception as e:
                self.logger.error(f"Webhook notification error: {str(e)}")
                raise
            
    async def process_with_llm(self, content: str, llm_name: str, model: str = None) -> Dict:
        """Process content with specified LLM"""
        try:
            if llm_name == "claude":
                response = await self.claude.messages.create(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": content}]
                )
                return {
                    "text": response.content[0].text,
                    "model": "claude-3-sonnet",
                    "provider": "anthropic"
                }
            
            elif llm_name == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=model or "gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": content}]
                )
                return {
                    "text": response.choices[0].message.content,
                    "model": model or "gpt-4-turbo-preview",
                    "provider": "openai"
                }
            
            elif llm_name == "groq":
                response = await self.groq_client.chat.completions.create(
                    model=model or "mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": content}]
                )
                return {
                    "text": response.choices[0].message.content,
                    "model": model or "mixtral-8x7b-32768",
                    "provider": "groq"
                }
            
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_name}")
            
        except Exception as e:
            self.logger.error(f"Error processing with {llm_name}: {str(e)}")
            raise