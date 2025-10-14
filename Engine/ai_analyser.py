# ai_analyzer_google.py
import os
import json
import time
import logging
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

class AnalysisType(Enum):
    CATEGORIZATION = "categorization"
    SUMMARIZATION = "summarization"
    KEY_POINTS = "key_points"
    SENTIMENT = "sentiment"
    ENTITY_EXTRACTION = "entity_extraction"
    CUSTOM_QUERY = "custom_query"

@dataclass
class AnalysisResult:
    """Data class to store AI analysis results"""
    file_path: str
    page_number: int
    analysis_type: AnalysisType
    result: Dict
    confidence: float
    processing_time: float
    timestamp: float
    model_used: str

class DocumentAnalyzer:
    """
    AI Document Analyzer using Google Generative AI (Gemini)
    Handles real-time analysis of OCR results
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gemini-2.5-flash",
                 analysis_categories: List[str] = None,
                 cache_results: bool = True):
        """
        Initialize AI Analyzer with Google Generative AI
        
        Args:
            api_key: Google AI Studio API key
            model: Gemini model to use
            analysis_categories: Predefined document categories
            cache_results: Whether to cache analysis results
        """
        
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        # API configuration
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.cache_results = cache_results
        
        # Analysis configuration
        self.analysis_categories = analysis_categories or [
            "Financial", "Legal", "Medical", "Educational", 
            "Personal", "Business", "Technical", "Government",
            "Academic", "Contract", "Invoice", "Report", "Letter"
        ]
        
        # Initialize Google AI client
        self.client = self._initialize_google_ai_client()
        
        # Analysis state
        self.analysis_queue = queue.Queue()
        self.active_analyses: Dict[str, threading.Event] = {}
        self.results_cache: Dict[str, AnalysisResult] = {}
        
        # Callbacks
        self.analysis_complete_callback: Optional[Callable] = None
        self.analysis_error_callback: Optional[Callable] = None
        
        # Statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0.0
        }
        
        self.setup_logging()
        self._start_analysis_worker()
    
    def _initialize_google_ai_client(self):
        """Initialize Google Generative AI client with error handling"""
        if not self.api_key:
            raise ValueError(
                "Google AI API key not provided. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter. "
                "Get free API key from: https://aistudio.google.com/app/apikey"
            )
        
        try:
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            model = genai.GenerativeModel(self.model)
            
            # Test the connection with a simple request
            test_response = model.generate_content("Hello")

            print(f"Google Generative AI client initialized successfully with model: {self.model}")
            
          #  self.logger.info(f"Google Generative AI client initialized successfully with model: {self.model}")
            return model
            
        except Exception as e:
            print(f"Failed to initialize Google AI client: {e}")
           # self.logger.error(f"Failed to initialize Google AI client: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _start_analysis_worker(self):
        """Start background worker for processing analysis requests"""
        self.worker_thread = threading.Thread(
            target=self._analysis_worker,
            daemon=True,
            name="AI-Analysis-Worker"
        )
        self.worker_thread.start()
        self.logger.info("AI analysis worker thread started")
    
    def _analysis_worker(self):
        """Background worker that processes analysis requests"""
        while True:
            try:
                # Get analysis request from queue
                analysis_request = self.analysis_queue.get()
                
                if analysis_request is None:  # Shutdown signal
                    break
                
                file_path, page_num, ocr_result, analysis_type, custom_query = analysis_request
                
                # Generate unique key for caching
                cache_key = f"{file_path}_page{page_num}_{analysis_type.value}"
                if custom_query:
                    cache_key += f"_{hash(custom_query)}"
                
                # Check cache
                if self.cache_results and cache_key in self.results_cache:
                    self.logger.info(f"Using cached analysis for {cache_key}")
                    cached_result = self.results_cache[cache_key]
                    if self.analysis_complete_callback:
                        self.analysis_complete_callback(cached_result)
                    continue
                
                # Perform analysis
                analysis_result = self._perform_analysis(
                    file_path, page_num, ocr_result, analysis_type, custom_query
                )
                
                if analysis_result:
                    # Cache the result
                    if self.cache_results:
                        self.results_cache[cache_key] = analysis_result
                    
                    # Update statistics
                    self.analysis_stats['total_analyses'] += 1
                    self.analysis_stats['successful_analyses'] += 1
                    self.analysis_stats['total_processing_time'] += analysis_result.processing_time
                    
                    # Call completion callback
                    if self.analysis_complete_callback:
                        self.analysis_complete_callback(analysis_result)
                
                self.analysis_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in analysis worker: {e}")
                self.analysis_stats['failed_analyses'] += 1
                
                if self.analysis_error_callback:
                    self.analysis_error_callback(file_path, page_num, str(e))
    
    def analyze_document_async(self, 
                             file_path: str,
                             page_num: int,
                             ocr_result,
                             analysis_type: AnalysisType = AnalysisType.CATEGORIZATION,
                             custom_query: Optional[str] = None) -> str:
        """
        Queue document for AI analysis (non-blocking)
        """
        analysis_id = f"{file_path}_page{page_num}_{analysis_type.value}"
        
        # Create stop event for this analysis
        stop_event = threading.Event()
        self.active_analyses[analysis_id] = stop_event
        
        # Add to queue
        self.analysis_queue.put((file_path, page_num, ocr_result, analysis_type, custom_query))
        
        self.logger.info(f"Queued analysis {analysis_id} for processing")
        return analysis_id
    
    def _perform_analysis(self,
                         file_path: str,
                         page_num: int,
                         ocr_result,
                         analysis_type: AnalysisType,
                         custom_query: Optional[str] = None) -> Optional[AnalysisResult]:
        """
        Perform AI analysis on OCR result using Google Generative AI
        """
        start_time = time.time()
        
        try:
            # Prepare text for analysis (Google has larger context windows)
            text_to_analyze = ocr_result.text[:8000]  # Increased limit for Gemini
            
            if not text_to_analyze.strip():
                self.logger.warning(f"No text to analyze for {file_path} page {page_num}")
                return None
            
            # Generate prompt based on analysis type
            prompt = self._generate_analysis_prompt(
                text_to_analyze, analysis_type, custom_query
            )
            
            # Call Google Generative AI API
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.3
                )
            )
            
            # Parse response
            analysis_text = response.text.strip()
            parsed_result = self._parse_analysis_response(
                analysis_text, analysis_type, file_path, page_num
            )
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                file_path=file_path,
                page_number=page_num,
                analysis_type=analysis_type,
                result=parsed_result,
                confidence=self._calculate_confidence(parsed_result, analysis_type),
                processing_time=processing_time,
                timestamp=time.time(),
                model_used=self.model
            )
            
        except Exception as e:
            self.logger.error(f"AI analysis failed for {file_path} page {page_num}: {e}")
            return None
    
    def _generate_analysis_prompt(self, 
                                text: str, 
                                analysis_type: AnalysisType,
                                custom_query: Optional[str] = None) -> str:
        """Generate appropriate prompt for analysis type for Google AI"""
        
        base_prompt = f"Analyze the following document text and provide your response as a valid JSON object only:\n\nDocument Text: {text}\n\n"
        
        if analysis_type == AnalysisType.CATEGORIZATION:
            categories_str = ", ".join(self.analysis_categories)
            return base_prompt + f"""
            Task: Categorize this document into one of these categories: {categories_str}
            
            Required JSON structure:
            {{
                "category": "category_name",
                "confidence": 0.95,
                "subcategory": "optional_subcategory",
                "tags": ["tag1", "tag2", "tag3"],
                "reasoning": "brief explanation of why this category was chosen"
            }}
            
            Return only the JSON object, no additional text.
            """
        
        elif analysis_type == AnalysisType.SUMMARIZATION:
            return base_prompt + """
            Task: Provide a concise summary of this document.
            
            Required JSON structure:
            {
                "summary": "concise summary here",
                "key_themes": ["theme1", "theme2", "theme3"],
                "length_estimate": "short/medium/long"
            }
            
            Return only the JSON object, no additional text.
            """
        
        elif analysis_type == AnalysisType.KEY_POINTS:
            return base_prompt + """
            Task: Extract the key points and important information from this document.
            
            Required JSON structure:
            {
                "key_points": ["point1", "point2", "point3", "point4", "point5"],
                "important_entities": ["entity1", "entity2", "entity3"],
                "action_items": ["action1", "action2"]
            }
            
            Return only the JSON object, no additional text.
            """
        
        elif analysis_type == AnalysisType.SENTIMENT:
            return base_prompt + """
            Task: Analyze the sentiment and tone of this document.
            
            Required JSON structure:
            {
                "sentiment": "positive/negative/neutral",
                "confidence": 0.95,
                "tone": ["formal", "urgent", "informative", etc.],
                "emotional_tones": ["emotion1", "emotion2"]
            }
            
            Return only the JSON object, no additional text.
            """
        
        elif analysis_type == AnalysisType.ENTITY_EXTRACTION:
            return base_prompt + """
            Task: Extract important entities, names, dates, amounts, and other key information.
            
            Required JSON structure:
            {
                "people": ["name1", "name2"],
                "organizations": ["org1", "org2"],
                "dates": ["date1", "date2"],
                "amounts": ["amount1", "amount2"],
                "locations": ["location1", "location2"],
                "key_terms": ["term1", "term2", "term3"]
            }
            
            Return only the JSON object, no additional text.
            """
        
        elif analysis_type == AnalysisType.CUSTOM_QUERY and custom_query:
            return base_prompt + f"""
            Task: {custom_query}
            
            Return your response as a valid JSON object. Structure it appropriately for the query.
            Return only the JSON object, no additional text.
            """
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def _parse_analysis_response(self, 
                               response_text: str, 
                               analysis_type: AnalysisType,
                               file_path: str,
                               page_num: int) -> Dict:
        """Parse Google AI response into structured data"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: return as text
                self.logger.warning(f"Could not parse JSON from response for {file_path} page {page_num}")
                return {"raw_response": response_text}
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed for {file_path} page {page_num}: {e}")
            return {"raw_response": response_text, "parse_error": str(e)}
    
    def _calculate_confidence(self, result: Dict, analysis_type: AnalysisType) -> float:
        """Calculate confidence score for analysis result"""
        if analysis_type == AnalysisType.CATEGORIZATION:
            return result.get('confidence', 0.8)
        elif analysis_type == AnalysisType.SENTIMENT:
            return result.get('confidence', 0.8)
        else:
            return 0.9
    
    # The following methods remain exactly the same:
    # set_analysis_complete_callback, set_analysis_error_callback,
    # stop_analysis, get_analysis_stats, clear_cache, wait_for_completion