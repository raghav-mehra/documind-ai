
import os
import json
import time
import logging
import queue
import threading
from Engine.ai_analyser import AnalysisType
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import google.generativeai as genai

class BatchStrategy(Enum):
    TIME_BASED = "time_based"      # Send batch when time threshold reached
    COUNT_BASED = "count_based"    # Send batch when page count reached
    HYBRID = "hybrid"              # Use both time and count thresholds

@dataclass
class BatchAnalysisResult:
    """Data class for batch analysis results"""
    file_path: str
    page_range: Tuple[int, int]  # (start_page, end_page)
    analysis_type: AnalysisType
    result: Dict
    processing_time: float
    timestamp: float
    batch_size: int

@dataclass
class DocumentAnalysisState:
    """Track analysis state for each document"""
    file_path: str
    total_pages: int
    processed_pages: int
    pending_pages: List[int]
    page_contents: Dict[int, str]  # page_num -> text
    analysis_start_time: float
    last_batch_time: float
    batches_sent: int

class SmartBatchAnalyzer:
    """
    Smart batch processor for AI analysis that optimizes API usage
    and handles variable OCR speeds
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gemini-2.5-flash",
                 batch_strategy: BatchStrategy = BatchStrategy.HYBRID,
                 max_pages_per_batch: int = 10,
                 max_chars_per_batch: int = 20000,
                 time_threshold_seconds: int = 10,
                 min_pages_for_batch: int = 3,
                 max_batches_per_document: int = 5,
                 analysis_categories: List[str] = None):
        """
        Initialize smart batch analyzer
        
        Args:
            api_key: Google AI API key
            model: Gemini model to use
            batch_strategy: Strategy for sending batches
            max_pages_per_batch: Maximum pages per API call
            max_chars_per_batch: Maximum characters per batch
            time_threshold_seconds: Send batch if this much time passed
            min_pages_for_batch: Minimum pages to wait for before sending
            max_batches_per_document: Maximum batches per document to avoid too many requests
            analysis_categories: Document categories for classification
        """
        
        # Configuration
        self.batch_strategy = batch_strategy
        self.max_pages_per_batch = max_pages_per_batch
        self.max_chars_per_batch = max_chars_per_batch
        self.time_threshold_seconds = time_threshold_seconds
        self.min_pages_for_batch = min_pages_for_batch
        self.max_batches_per_document = max_batches_per_document
        self.analysis_categories = analysis_categories or [
            "Financial", "Legal", "Medical", "Educational", 
            "Personal", "Business", "Technical", "Government"
        ]
        
        # AI client
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.client = self._initialize_google_ai_client()
        
        # Document tracking
        self.document_states: Dict[str, DocumentAnalysisState] = {}
        self.batch_queue = queue.Queue()
        self.final_analysis_queue = queue.Queue()
        
        # Callbacks
        self.batch_analysis_callback: Optional[Callable] = None
        self.final_analysis_callback: Optional[Callable] = None
        self.analysis_error_callback: Optional[Callable] = None
        
        # Monitoring
        self.monitor_thread = None
        self.is_monitoring = False
        
        # Statistics
        self.stats = {
            'total_batches_sent': 0,
            'total_pages_processed': 0,
            'api_requests_saved': 0,
            'total_processing_time': 0.0,
            'documents_completed': 0
        }
        
        self.setup_logging()
        self._start_processing_workers()
    
    def _initialize_google_ai_client(self):
        """Initialize Google Generative AI client"""
        if not self.api_key:
            raise ValueError("Google API key not provided")
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        
        print(f"Google AI client initialized with batch strategy: {self.batch_strategy.value}")
        return model
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _start_processing_workers(self):
        """Start background workers for batch processing"""
        # Batch processing worker
        self.batch_worker = threading.Thread(
            target=self._batch_processing_worker,
            daemon=True,
            name="Batch-Processor"
        )
        self.batch_worker.start()
        
        # Final analysis worker
        self.final_worker = threading.Thread(
            target=self._final_analysis_worker,
            daemon=True,
            name="Final-Analyzer"
        )
        self.final_worker.start()
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_documents,
            daemon=True,
            name="Document-Monitor"
        )
        self.monitor_thread.start()
        
        self.logger.info("All processing workers started")
    
    def register_document(self, file_path: str, total_pages: int):
        """
        Register a new document for analysis
        """
        if file_path in self.document_states:
            self.logger.warning(f"Document already registered: {file_path}")
            return
        
        document_state = DocumentAnalysisState(
            file_path=file_path,
            total_pages=total_pages,
            processed_pages=0,
            pending_pages=[],
            page_contents={},
            analysis_start_time=time.time(),
            last_batch_time=time.time(),
            batches_sent=0
        )
        
        self.document_states[file_path] = document_state
        self.logger.info(f"Registered document: {file_path} with {total_pages} pages")
    
    def add_page_result(self, file_path: str, page_num: int, ocr_result):
        """
        Add a page OCR result for analysis
        Triggers batch processing based on strategy
        """
        if file_path not in self.document_states:
            self.logger.error(f"Document not registered: {file_path}")
            return
        
        state = self.document_states[file_path]
        # Store page content
        state.page_contents[page_num] = ocr_result.text
        state.pending_pages.append(page_num)
        state.processed_pages += 1
        
        self.logger.debug(f"Added page {page_num} for {file_path}. "
                         f"Pending: {len(state.pending_pages)}, "
                         f"Processed: {state.processed_pages}/{state.total_pages}")
        
        # Check if we should send a batch
        should_send_batch = self._should_send_batch(state)
        
        if should_send_batch:
            self._prepare_and_send_batch(file_path, state)
    
    def _should_send_batch(self, state: DocumentAnalysisState) -> bool:
        """
        Determine if we should send a batch based on strategy
        """
        pending_count = len(state.pending_pages)
        time_since_last_batch = time.time() - state.last_batch_time
        
        if self.batch_strategy == BatchStrategy.TIME_BASED:
            return (time_since_last_batch >= self.time_threshold_seconds and 
                   pending_count >= 1)
        
        elif self.batch_strategy == BatchStrategy.COUNT_BASED:
            return pending_count >= self.min_pages_for_batch
        
        elif self.batch_strategy == BatchStrategy.HYBRID:
            # Send if we have enough pages OR if too much time passed
            count_ready = pending_count >= self.min_pages_for_batch
            time_ready = (time_since_last_batch >= self.time_threshold_seconds and 
                         pending_count >= 1)
            return count_ready or time_ready
        
        return False
    
    def _prepare_and_send_batch(self, file_path: str, state: DocumentAnalysisState):
        """
        Prepare a batch of pages and send for analysis
        """
        if not state.pending_pages:
            return
        
        # Sort pending pages and take up to max_pages_per_batch
        state.pending_pages.sort()
        batch_pages = state.pending_pages[:self.max_pages_per_batch]
        
        # Check character limit
        batch_text = self._combine_page_texts(state.page_contents, batch_pages)
        if len(batch_text) > self.max_chars_per_batch:
            # Reduce batch size to fit character limit
            batch_pages = self._adjust_batch_for_size(state.page_contents, batch_pages)
            batch_text = self._combine_page_texts(state.page_contents, batch_pages)
        
        # Remove these pages from pending
        for page in batch_pages:
            state.pending_pages.remove(page)
        
        # Update state
        state.last_batch_time = time.time()
        state.batches_sent += 1
        
        # Add to batch queue
        batch_data = {
            'file_path': file_path,
            'pages': batch_pages,
            'text': batch_text,
            'batch_number': state.batches_sent
        }
        
        self.batch_queue.put(batch_data)
        self.logger.info(f"Queued batch {state.batches_sent} for {file_path} "
                        f"with {len(batch_pages)} pages")
    
    def _combine_page_texts(self, page_contents: Dict[int, str], pages: List[int]) -> str:
        """Combine multiple page texts into a single string"""
        combined = []
        for page_num in pages:
            if page_num in page_contents:
                combined.append(f"--- Page {page_num} ---\n{page_contents[page_num]}")
        return "\n\n".join(combined)
    
    def _adjust_batch_for_size(self, page_contents: Dict[int, str], pages: List[int]) -> List[int]:
        """Reduce batch size to fit within character limit"""
        current_size = 0
        adjusted_pages = []
        
        for page_num in pages:
            page_text = page_contents.get(page_num, "")
            if current_size + len(page_text) <= self.max_chars_per_batch:
                adjusted_pages.append(page_num)
                current_size += len(page_text)
            else:
                break
        
        return adjusted_pages if adjusted_pages else pages[:1]  # At least one page
    
    def _batch_processing_worker(self):
        """Process batches from the queue"""
        while True:
            try:
                batch_data = self.batch_queue.get()
                if batch_data is None:  # Shutdown signal
                    break
                
                self._process_batch(batch_data)
                self.batch_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in batch processing worker: {e}")
    
    def _process_batch(self, batch_data: Dict):
        """Process a single batch of pages"""
        start_time = time.time()
        
        try:
            file_path = batch_data['file_path']
            pages = batch_data['pages']
            batch_text = batch_data['text']
            batch_number = batch_data['batch_number']
            
            self.logger.info(f"Processing batch {batch_number} for {file_path} "
                           f"(pages {min(pages)}-{max(pages)})")
            
            # Perform batch analysis
            analysis_result = self._analyze_batch(batch_text, pages, file_path)
            
            if analysis_result:
                processing_time = time.time() - start_time
                
                batch_result = BatchAnalysisResult(
                    file_path=file_path,
                    page_range=(min(pages), max(pages)),
                    analysis_type=AnalysisType.CATEGORIZATION,
                    result=analysis_result,
                    processing_time=processing_time,
                    timestamp=time.time(),
                    batch_size=len(pages)
                )
                
                # Update statistics
                self.stats['total_batches_sent'] += 1
                self.stats['total_pages_processed'] += len(pages)
                self.stats['total_processing_time'] += processing_time
                
                # Call batch callback
                if self.batch_analysis_callback:
                    self.batch_analysis_callback(batch_result)
                
                self.logger.info(f"Batch {batch_number} completed in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            if self.analysis_error_callback:
                self.analysis_error_callback(file_path, f"Batch error: {str(e)}")
    
    def _analyze_batch(self, batch_text: str, pages: List[int], file_path: str) -> Optional[Dict]:
        """Analyze a batch of pages using AI"""
        try:
            prompt = self._generate_batch_prompt(batch_text, pages)
            
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.3
                )
            )
            
            analysis_text = response.text.strip()
            return self._parse_batch_response(analysis_text, pages, file_path)
            
        except Exception as e:
            self.logger.error(f"AI analysis failed for batch: {e}")
            return None
    
    def _generate_batch_prompt(self, batch_text: str, pages: List[int]) -> str:
        """Generate prompt for batch analysis"""
        page_range = f"{min(pages)} to {max(pages)}"
        
        return f"""
        Analyze the following document pages ({page_range}) and provide insights.
        
        Document Pages Content:
        {batch_text}
        
        Provide a comprehensive analysis including:
        1. Overall category and subcategory
        2. Key themes across these pages
        3. Important entities mentioned
        4. Any patterns or notable information
        
        Return your response as a JSON object with this structure:
        {{
            "category": "main_category",
            "subcategory": "specific_type",
            "key_themes": ["theme1", "theme2", "theme3"],
            "important_entities": {{
                "people": ["name1", "name2"],
                "organizations": ["org1", "org2"],
                "locations": ["loc1", "loc2"]
            }},
            "notable_patterns": ["pattern1", "pattern2"],
            "pages_analyzed": {pages},
            "summary": "brief summary of these pages"
        }}
        
        Return only the JSON object.
        """
    
    def _parse_batch_response(self, response_text: str, pages: List[int], file_path: str) -> Dict:
        """Parse batch analysis response"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                result['pages_analyzed'] = pages  # Ensure pages are included
                return result
            else:
                return {"raw_response": response_text, "pages_analyzed": pages}
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed for batch: {e}")
            return {"raw_response": response_text, "pages_analyzed": pages, "parse_error": str(e)}
    
    def _monitor_documents(self):
        """Monitor documents and trigger final analysis when complete"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                documents_to_remove = []
                
                for file_path, state in self.document_states.items():
                    # Check if document processing is complete
                    if state.processed_pages >= state.total_pages and not state.pending_pages:
                        # Send any remaining pages as final batch
                        if state.page_contents:
                            self._send_final_analysis(file_path, state)
                        documents_to_remove.append(file_path)
                    
                    # Check for stalled documents (no new pages in long time)
                    time_since_last_page = current_time - state.last_batch_time
                    if (time_since_last_page > self.time_threshold_seconds * 2 and 
                        state.pending_pages and state.processed_pages < state.total_pages):
                        self.logger.warning(f"Document seems stalled: {file_path}")
                        self._prepare_and_send_batch(file_path, state)
                
                # Remove completed documents
                for file_path in documents_to_remove:
                    del self.document_states[file_path]
                    self.stats['documents_completed'] += 1
                    self.logger.info(f"Document analysis completed: {file_path}")
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in document monitor: {e}")
                time.sleep(5)
    
    def _send_final_analysis(self, file_path: str, state: DocumentAnalysisState):
        """Send final comprehensive analysis for entire document"""
        try:
            # Combine all pages for final analysis
            all_pages = list(state.page_contents.keys())
            all_pages.sort()
            full_text = self._combine_page_texts(state.page_contents, all_pages)
            
            # Limit text size for final analysis
            if len(full_text) > 30000:
                full_text = full_text[:30000] + "\n\n[Document truncated due to length]"
            
            final_data = {
                'file_path': file_path,
                'all_pages': all_pages,
                'full_text': full_text,
                'total_batches': state.batches_sent,
                'total_pages': state.total_pages
            }
            
            self.final_analysis_queue.put(final_data)
            self.logger.info(f"Queued final analysis for {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error preparing final analysis: {e}")
    
    def _final_analysis_worker(self):
        """Process final comprehensive analyses"""
        while True:
            try:
                final_data = self.final_analysis_queue.get()
                if final_data is None:
                    break
                
                self._process_final_analysis(final_data)
                self.final_analysis_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in final analysis worker: {e}")
    
    def _process_final_analysis(self, final_data: Dict):
        """Process final comprehensive document analysis"""
        start_time = time.time()
        
        try:
            file_path = final_data['file_path']
            all_pages = final_data['all_pages']
            full_text = final_data['full_text']
            
            self.logger.info(f"Starting final analysis for {file_path} "
                           f"({len(all_pages)} pages total)")
            
            prompt = self._generate_final_analysis_prompt(full_text, all_pages)
            
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.3
                )
            )
            
            analysis_text = response.text.strip()
            final_result = self._parse_final_response(analysis_text, all_pages, file_path)
            
            processing_time = time.time() - start_time
            
            # Create final result
            final_analysis = BatchAnalysisResult(
                file_path=file_path,
                page_range=(1, len(all_pages)),
                analysis_type=AnalysisType.CATEGORIZATION,
                result=final_result,
                processing_time=processing_time,
                timestamp=time.time(),
                batch_size=len(all_pages)
            )
            
            # Call final analysis callback
            if self.final_analysis_callback:
                self.final_analysis_callback(final_analysis)
            
            self.logger.info(f"Final analysis completed for {file_path} in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Final analysis failed for {file_path}: {e}")
            if self.analysis_error_callback:
                self.analysis_error_callback(file_path, f"Final analysis error: {str(e)}")
    
    def _generate_final_analysis_prompt(self, full_text: str, all_pages: List[int]) -> str:
        """Generate prompt for final comprehensive analysis"""
        return f"""
        Provide a COMPREHENSIVE analysis of the ENTIRE document (pages {min(all_pages)}-{max(all_pages)}).
        
        FULL DOCUMENT CONTENT:
        {full_text}
        
        Analyze and provide:
        1. Overall document category and type
        2. Executive summary
        3. Key findings and conclusions
        4. Important recommendations (if any)
        5. Document structure and organization
        6. Target audience and purpose
        
        Return as JSON with this structure:
        {{
            "document_category": "category_name",
            "document_type": "specific_type",
            "executive_summary": "comprehensive summary",
            "key_findings": ["finding1", "finding2", "finding3"],
            "recommendations": ["rec1", "rec2"],
            "document_structure": "description of organization",
            "target_audience": "intended readers",
            "confidence_score": 0.95,
            "total_pages_analyzed": {len(all_pages)}
        }}
        
        Return only the JSON object.
        """
    
    def _parse_final_response(self, response_text: str, pages: List[int], file_path: str) -> Dict:
        """Parse final analysis response"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"raw_response": response_text, "total_pages": len(pages)}
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed for final analysis: {e}")
            return {"raw_response": response_text, "total_pages": len(pages), "parse_error": str(e)}
    
    # Callback setters
    def set_batch_analysis_callback(self, callback: Callable):
        self.batch_analysis_callback = callback
    
    def set_final_analysis_callback(self, callback: Callable):
        self.final_analysis_callback = callback
    
    def set_analysis_error_callback(self, callback: Callable):
        self.analysis_error_callback = callback
    
    def get_statistics(self) -> Dict:
        """Get current processing statistics"""
        active_documents = len(self.document_states)
        pending_batches = self.batch_queue.qsize()
        pending_final = self.final_analysis_queue.qsize()
        
        return {
            **self.stats,
            'active_documents': active_documents,
            'pending_batches': pending_batches,
            'pending_final_analyses': pending_final,
            'average_batch_size': (self.stats['total_pages_processed'] / 
                                  max(self.stats['total_batches_sent'], 1))
        }
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def shutdown(self):
        """Graceful shutdown"""
        self.stop_monitoring()
        self.batch_queue.put(None)
        self.final_analysis_queue.put(None)