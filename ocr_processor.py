# ocr_processor.py
import os
import tempfile
import time
import logging
import threading
import queue
import json
from typing import Dict, List, Optional, Union, Callable
from pdf2image import convert_from_path
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import numpy as np

try:
    from paddleocr import PaddleOCR
    import cv2
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available - using mock implementation")

@dataclass
class OCRResult:
    """Data class to store OCR results in structured format"""
    text: str
    confidence: float
    bounding_boxes: List[List[List[int]]]
    words: List[Dict[str, Union[str, float, List]]]
    image_path: str
    processing_time: float
    page_number: int = 1

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    file_name: str
    file_path: str
    file_size: int
    created_time: float
    modified_time: float
    file_stem: str
    parent_directory: str
    pages: int = 1
    file_type: str = ""
    dimensions: Optional[tuple] = None

class OCRProcessor:
    """
    Main OCR Processor using PaddleOCR
    Handles document/image processing with error handling and performance tracking
    """
    
    def __init__(self,  
                 lang: str = 'en',
                 use_doc_orientation_classify=False,
                 use_doc_unwarping=False,
                 use_textline_orientation=False,
                 pdf_dpi: int = 300,
                 poppler_path = None,
                 show_log: bool = False,
                 enable_realtime_analysis: bool = True):
        """
        Initialize PaddleOCR engine
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language for OCR ('en', 'ch', 'fr', etc.)
            show_log: Whether to show PaddleOCR logs
            gpu_available: Whether to use GPU acceleration
        """
        self.lang = lang
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.pdf_dpi = pdf_dpi
        self.show_log = show_log
        self.poppler_path = poppler_path
        self.enable_realtime_analysis = enable_realtime_analysis
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}

        # Multi-threading and process control
        self.active_processes: Dict[str, threading.Event] = {}  # file_path -> stop_event
        self.result_queue = queue.Queue()
        self.analysis_callback: Optional[Callable] = None
        
        # Initialize OCR engine
        self.ocr_engine = self._initialize_ocr_engine()
        self.setup_logging()
        
    def _initialize_ocr_engine(self) -> Optional[PaddleOCR]:
        """Initialize PaddleOCR engine with error handling"""
        if not PADDLEOCR_AVAILABLE:
            logging.warning("PaddleOCR not available. Using mock implementation.")
            return None
            
        try:
            ocr = PaddleOCR(
                lang=self.lang,
                use_gpu=True,
                use_textline_orientation=self.use_textline_orientation,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
            )
            
            logging.info("PaddleOCR engine initialized successfully")
            return ocr
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")
            return None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ocr_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_image_dimensions(self, image_path: str) -> Optional[tuple]:
      """Get dimensions of an image file"""
      try:
          with Image.open(image_path) as img:
              return img.size
      except Exception as e:
          self.logger.warning(f"Could not get image dimensions for {image_path}: {e}")
          return None
      
    
    def process_document_async(self, 
                             file_path: str,
                             analysis_callback: Optional[Callable] = None) -> str:
        """
        Process document asynchronously with real-time analysis
        
        Args:
            file_path: Path to document
            analysis_callback: Function to call when each page is processed
            
        Returns:
            Process ID (file path)
        """
        # Create stop event for this process
        stop_event = threading.Event()
        self.active_processes[file_path] = stop_event
        
        # Store analysis callback
        if analysis_callback:
            self.analysis_callback = analysis_callback
        
        # Start processing in separate thread
        thread = threading.Thread(
            target=self._process_document_thread,
            args=(file_path, stop_event),
            daemon=True
        )
        thread.start()
        
        self.logger.info(f"Started async processing for: {file_path}")
        return file_path

    
    def _process_document_thread(self, file_path: str, stop_event: threading.Event) -> Optional[Union[OCRResult, Dict[int, OCRResult]]]:
        """
        Main method to process a single document/image
        
        Args:
            file_path: Path to the document/image file
            
        Returns:
            OCRResult object containing extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Validate file
            if not self._validate_file(file_path):
                self.result_queue.put((file_path, None, "Validation failed"))
                return None
            
            # Get document metadata
            metadata = self._get_document_metadata(file_path)
            
            # Process based on file type
            if file_path.lower().endswith('.pdf'):
                result = self._process_pdf(file_path, metadata, stop_event)
            else:
                result = self._process_image(file_path, metadata, stop_event)

            # Check if process was stopped
            if stop_event.is_set():
                self.logger.info(f"Processing stopped for: {file_path}")
                self.result_queue.put((file_path, None, "Stopped by user"))
            else:
                self.result_queue.put((file_path, result, "Completed"))
        
            
            if result:
                # Handle different return types
                if isinstance(result, dict):
                    # PDF result - add processing time to each page
                    processing_time = time.time() - start_time
                    for page_result in result.values():
                        page_result.processing_time = processing_time / len(result)
                else:
                    # Single image result
                    result.processing_time = time.time() - start_time
            
            self.logger.info(f"Processed {metadata.file_name} in {time.time() - start_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
        finally:
            # Clean up
            if file_path in self.active_processes:
                del self.active_processes[file_path]
    
    def process_batch(self, file_paths: List[str]) -> Dict[str, Optional[OCRResult]]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary mapping file paths to OCRResult objects
        """
        results = {}
        
        for file_path in file_paths:
            result = self._process_document_thread(file_path)
            results[file_path] = result
            
        return results
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is supported"""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            self.logger.error(f"Unsupported file format: {file_ext}")
            return False
        
        return True
    

    def _get_document_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract document metadata"""
        file_path_obj = Path(file_path)
        file_size = os.path.getsize(file_path)
        file_type = file_path_obj.suffix.lower()
        file_name = file_path_obj.name
        file_stem = file_path_obj.stem
        created_time = file_path_obj.stat().st_ctime
        modified_time = file_path_obj.stat().st_mtime
        parent_directory = file_path_obj.parent.name
        dimensions = None
        pages = 1
        # Get image dimensions for image files
        dimensions = None
        if not file_path.lower().endswith('.pdf'):
            try:
                with Image.open(file_path) as img:
                    dimensions = img.size
            except Exception as e:
                self.logger.warning(f"Could not get image dimensions: {e}")
        
        return DocumentMetadata(
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
            dimensions=dimensions,
            file_type=file_type,
            pages=pages,
            created_time=created_time,
            modified_time=modified_time,
            file_stem=file_stem,
            parent_directory=parent_directory
        )
    

    def _save_ocr_to_json(self, metadata: DocumentMetadata, ocr_data: List):
        
        """Save the OCR Reading Result to the Json file"""

        directory_name = f"OCR_readings/{metadata.parent_directory}"
        file_name = f"{metadata.file_stem}.json"
        full_path = os.path.join(directory_name, file_name)

        os.makedirs(directory_name, exist_ok=True)

       # Save the Json file to the CWD (Current Working Directory)
  
        with open(full_path, "w") as json_file:
           json.dump(ocr_data, json_file, indent=4) 

    def _trigger_analysis(self, file_path: str, page_num: int, result: OCRResult):
        """
        Trigger AI analysis in separate thread
        """
        if not self.analysis_callback:
            return
        
        # Start analysis in separate thread to avoid blocking OCR
        analysis_thread = threading.Thread(
            target=self._run_analysis,
            args=(file_path, page_num, result),
            daemon=True
        )
        analysis_thread.start()

    
    def _run_analysis(self, file_path: str, page_num: int, result: OCRResult):
        """
        Run AI analysis on the page result
        """
        try:
            print(f"Starting AI analysis for {file_path} page {page_num}")
            
            # Call the analysis callback
            if self.analysis_callback:
                self.analysis_callback(file_path, page_num, result)
                
        except Exception as e:
            self.logger.error(f"Analysis failed for {file_path} page {page_num}: {e}")


    def _process_image(self, file_path: str, metadata: DocumentMetadata,stop_event: threading.Event) -> Optional[OCRResult]:
        """Process single image file"""

        if stop_event.is_set():
            return None
        
        if self.ocr_engine is None:
            return self._mock_ocr_processing(file_path, metadata)
        
        try:
            # Perform OCR
            result = self.ocr_engine.ocr(file_path,cls=False)
            
            result_file_path = "D:\\DocScanner_cum_Analysis\\output\\" + metadata.file_stem + "_res.json"
            data = {}

            self._save_ocr_to_json(metadata, ocr_data=result)
            # for res in result:
            #    res.save_to_json("output")

            # with open(result_file_path, 'r') as file:
            #    data = json.load(file)
            
            if not result:
                self.logger.warning(f"No text detected in {file_path}")
                return self._create_empty_result(file_path, metadata)
            
            if not stop_event.is_set():
                if self.enable_realtime_analysis and self.analysis_callback:
                   self._trigger_analysis(file_path, 1, result)
            
            return self._parse_ocr_result(result, file_path, metadata)
            
        except Exception as e:
            self.logger.error(f"OCR processing failed for {file_path}: {e}")
            return None
    
    def _process_pdf(self, file_path: str, metadata: DocumentMetadata, stop_event: threading.Event) -> Optional[Dict[int, OCRResult]]:
      """
      Process PDF file by converting each page to image and performing OCR
    
      Args:
          file_path: Path to the PDF file
          metadata: Document metadata
        
      Returns:
          Dictionary mapping page numbers to OCRResult objects
      """
    
      # if not PDF2IMAGE_AVAILABLE:
      #     self.logger.error("pdf2image library not available. Cannot process PDF files.")
      #     return self._process_pdf_fallback(file_path, metadata)
    
      try:
          self.logger.info(f"Starting PDF processing for {metadata.file_name}")
        
          # Create temporary directory for storing converted images
          with tempfile.TemporaryDirectory() as temp_dir:
              # Convert PDF to images
              images = self._convert_pdf_to_images(file_path, temp_dir, metadata)
            
              if not images:
                  self.logger.error(f"Failed to convert PDF to images: {metadata.file_name}")
                  return None
            
              # Process each page image
              results = {}
              total_pages = len(images)
            
              for page_num, image_path in enumerate(images, 1):
                  # Check if process should stop
                  if stop_event.is_set():
                        self.logger.info(f"Stopping PDF processing at page {page_num}")
                        break
                  
                  self.logger.info(f"Processing page {page_num}/{total_pages} of {metadata.file_name}")
                
                  # Process the page image
                  page_result = self._process_pdf_page(image_path, page_num, metadata)
                
                  if page_result:
                      results[page_num] = page_result
                      # Trigger real-time analysis
                      if self.enable_realtime_analysis and self.analysis_callback:
                            self._trigger_analysis(file_path, page_num, page_result)
                  else:
                      self.logger.warning(f"Failed to process page {page_num} of {metadata.file_name}")
                  
            
              self.logger.info(f"Completed PDF processing: {len(results)}/{total_pages} pages successful")
              return results
            
      except Exception as e:
          self.logger.error(f"Error processing PDF {metadata.file_name}: {e}")
          return None
    
    
    def _process_pdf_page(self, image_path: str, page_num: int, metadata: DocumentMetadata) -> Optional[OCRResult]:
      """
    Process a single PDF page image
    
    Args:
        image_path: Path to the page image
        page_num: Page number
        metadata: Original PDF metadata
        
    Returns:
        OCRResult for the page
    """
    
      try:
          page_metadata = DocumentMetadata(
            file_path=image_path,
            file_name=f"{metadata.file_stem}_page_{page_num}.png",
            file_size=os.path.getsize(image_path),
            dimensions=self._get_image_dimensions(image_path),
            file_type='.png',
            pages=1,  # Single page image
            created_time=metadata.created_time,
            modified_time=metadata.modified_time,
            file_stem=f"{metadata.file_stem}_page_{page_num:03d}",
            parent_directory=metadata.parent_directory
         )
        
          # Perform OCR on the page image
          if self.ocr_engine is None:
              return self._mock_ocr_processing(image_path, page_metadata, page_num)
        
          # Perform actual OCR
          result = self.ocr_engine.ocr(image_path)

          self._save_ocr_to_json(page_metadata, ocr_data=result)

        #   for res in result:
        #        res.save_to_json("output")

        #   with open(result_file_path, 'r') as file:
        #        data = json.load(file)
        
          if not result:
              self.logger.warning(f"No text detected in page {page_num}")
              return self._create_empty_result(image_path, page_metadata, page_num)
   
        
          # Parse OCR result
          return self._parse_ocr_result(result, image_path, page_metadata, page_num)
        
      except Exception as e:
          self.logger.error(f"Error processing PDF page {page_num}: {e}")
          return None

    
    def _convert_pdf_to_images(self, file_path: str, temp_dir: str, metadata: DocumentMetadata) -> List[str]:
      """
      Convert PDF pages to images using pdf2image
    
      Args:
        file_path: Path to PDF file
        temp_dir: Temporary directory to store images
        metadata: Document metadata
        
      Returns:
          List of paths to converted image files
      """
    
      try:
        # PDF conversion settings
          dpi = self.pdf_dpi  # Resolution for OCR (higher = better accuracy but slower)
          fmt = 'png'  # Output format
          thread_count = 4  # Number of threads for conversion
        
          self.logger.info(f"Converting PDF to images with DPI: {dpi}")
        
        # Convert PDF to images
          images = convert_from_path(
            file_path,
            dpi=self.pdf_dpi,
            output_folder=temp_dir,
            first_page=1,  # Start from page 1
            last_page=None,  # Process all pages
            fmt=fmt,
            thread_count=thread_count,
            poppler_path=self.poppler_path
        )
        
        # Save images to files and get paths
          image_paths = []
          for i, image in enumerate(images):
              page_num = i + 1
              image_filename = f"{metadata.file_stem}_page_{page_num:03d}.{fmt}"
              image_path = os.path.join(temp_dir, image_filename)
            
            # Save image
              image.save(image_path, fmt.upper())
              image_paths.append(image_path)
            
              self.logger.debug(f"Saved page {page_num} as {image_filename}")
        
          self.logger.info(f"Successfully converted {len(image_paths)} PDF pages to images")
          return image_paths
        
      except Exception as e:
          self.logger.error(f"Error converting PDF to images: {e}")
          return []
    
    def _parse_ocr_result(self, ocr_data: List, file_path: str, metadata: DocumentMetadata, page_num: int = 1) -> OCRResult:
        """Parse PaddleOCR result into structured format"""
        full_text = ""
        confidence_scores = []
        bounding_boxes = []
        words = []
        
        for line in ocr_data:
            if line:
                # Extract text and confidence
                text = line[1][0]
                confidence = line[1][1]
                
                # Extract bounding box coordinates
                bbox = [[int(coord[0]), int(coord[1])] for coord in line[0]]
                
                full_text += text + "\n"
                confidence_scores.append(confidence)
                bounding_boxes.append(bbox)
                words.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return OCRResult(
            text=full_text.strip(),
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            words=words,
            image_path=file_path,
            processing_time=0.0,  # Will be set by parent method
            page_number=page_num
        )
    
    def _parse_ocr_result_dict(self, ocr_data_dict: Dict, file_path: str, metadata: DocumentMetadata, page_num: int = 1) -> OCRResult:
      """Parse PaddleOCR JSON result into structured format"""
    
      full_text = ""
      confidence_scores = []
      bounding_boxes = []
      words = []
    
      try:
          # Extract the arrays from the OCR result dictionary
          rec_texts = ocr_data_dict.get('rec_texts', [])  # List of recognized texts
          rec_scores = ocr_data_dict.get('rec_scores', [])  # List of confidence scores
          dt_polys = ocr_data_dict.get('dt_polys', [])  # List of bounding box coordinates
        
          # Validate that we have data to process
          if not rec_texts:
              self.logger.warning(f"No text detected in OCR result for {file_path}")
              return self._create_empty_result(file_path, metadata)
        
          # Process each text detection
          for i, text in enumerate(rec_texts):
            if text and text.strip():  # Only process non-empty text
                # Get confidence score for this text (handle index bounds)
                confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                
                # Get bounding box coordinates for this text
                bbox = dt_polys[i] if i < len(dt_polys) else []
                
                # Convert coordinates to integers and ensure proper format
                normalized_bbox = []
                if bbox:
                    try:
                        # Convert each coordinate pair to integers
                        normalized_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
                    except (TypeError, IndexError, ValueError) as e:
                        self.logger.warning(f"Error processing bbox at index {i}: {e}")
                        normalized_bbox = []
                
                # Build the full text
                full_text += text.strip() + "\n"
                confidence_scores.append(confidence)
                bounding_boxes.append(normalized_bbox)
                
                # Store individual word information
                words.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': normalized_bbox,
                    'word_index': i
                })
        
        # Calculate average confidence
          avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
          return OCRResult(
            text=full_text.strip(),
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            words=words,
            image_path=file_path,
            processing_time=0.0,  # Will be set by parent method
            page_number=page_num
          )
        
      except Exception as e:
          self.logger.error(f"Error parsing OCR result for {file_path}: {e}")
          return self._create_empty_result(file_path, metadata)
    
    def _create_empty_result(self, file_path: str, metadata: DocumentMetadata) -> OCRResult:
        """Create empty result for documents with no text detected"""
        return OCRResult(
            text="",
            confidence=0.0,
            bounding_boxes=[],
            words=[],
            image_path=file_path,
            processing_time=0.0,
            page_number=1
        )
    
    def stop_processing(self, file_path: str) -> bool:
        """
        Stop processing for a specific document
        
        Args:
            file_path: Path of document to stop
            
        Returns:
            True if stopped successfully, False if not found
        """
        if file_path in self.active_processes:
            self.active_processes[file_path].set()
            self.logger.info(f"Stop signal sent for: {file_path}")
            return True
        else:
            self.logger.warning(f"No active process found for: {file_path}")
            return False
    
    def stop_all_processing(self):
        """Stop all active document processing"""
        for file_path, stop_event in self.active_processes.items():
            stop_event.set()
            self.logger.info(f"Stop signal sent for: {file_path}")
        
        self.logger.info("Stop signals sent for all active processes")
    

    def get_processing_status(self, file_path: str) -> Optional[str]:
        """
        Get status of a document processing
        
        Returns:
            Status string or None if not found
        """
        if file_path in self.active_processes:
            stop_event = self.active_processes[file_path]
            if stop_event.is_set():
                return "stopping"
            else:
                return "processing"
        return None
    
    def get_active_processes(self) -> List[str]:
        """Get list of currently active processes"""
        return list(self.active_processes.keys())
    
    def wait_for_completion(self, file_path: str, timeout: Optional[float] = None) -> bool:
        """
        Wait for a specific document to complete processing
        
        Returns:
            True if completed, False if timeout or error
        """
        start_time = time.time()
        
        while True:
            # Check if process is still active
            if file_path not in self.active_processes:
                return True  # Completed
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
    
    def get_next_result(self, timeout: Optional[float] = None) -> Optional[tuple]:
        """
        Get next completed result from queue (non-blocking)
        
        Returns:
            Tuple (file_path, result, status) or None if no results
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_analysis_callback(self, callback: Callable):
        """
        Set callback function for real-time analysis
        
        Callback signature: callback(file_path: str, page_num: int, result: OCRResult)
        """
        self.analysis_callback = callback

    
    def _mock_ocr_processing(self, file_path: str, metadata: DocumentMetadata) -> OCRResult:
        """Mock OCR processing for testing when PaddleOCR is not available"""
        # Simulate different types of documents based on filename
        filename = Path(file_path).stem.lower()
        
        mock_texts = {
            'invoice': """INVOICE
Invoice No: INV-2023-001
Date: January 15, 2023
Bill To: ABC Company
Amount: $1,250.00
Status: Paid""",
            
            'contract': """CONTRACT AGREEMENT
This agreement is made between Party A and Party B.
Effective Date: March 1, 2023
Term: 12 Months
Signatures: _________________""",
            
            'medical': """MEDICAL REPORT
Patient: John Smith
Date of Birth: 1985-05-15
Diagnosis: Healthy
Recommendation: Annual checkup""",
            
            'default': """DOCUMENT TITLE
This is a sample document containing text extracted using OCR.
The quick brown fox jumps over the lazy dog.
Artificial Intelligence is transforming document processing."""
        }
        
        text = mock_texts['default']
        for key in mock_texts:
            if key in filename:
                text = mock_texts[key]
                break
        
        return OCRResult(
            text=text,
            confidence=0.95,  # Mock confidence
            bounding_boxes=[[[0, 0], [100, 0], [100, 50], [0, 50]]],
            words=[{'text': word, 'confidence': 0.95, 'bbox': [[0, 0], [50, 0], [50, 20], [0, 20]]} 
                  for word in text.split()[:5]],
            image_path=file_path,
            processing_time=0.5,  # Mock processing time
            page_number=1
        )
    
    def get_supported_formats(self) -> set:
        """Get supported file formats"""
        return self.supported_formats
    
    def get_engine_info(self) -> Dict:
        """Get OCR engine information"""
        return {
            'engine': 'PaddleOCR' if self.ocr_engine else 'MockOCR',
            'language': self.lang,
            'supported_formats': list(self.supported_formats)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize OCR processor
    poppler_path = r"C:\poppler-windows\poppler-25.07.0\Library\bin"
    
    ocr_processor = OCRProcessor(lang='en',show_log=True,poppler_path=poppler_path)
    
    
    # Test with a sample (you'll need actual image files for real testing)
    print("OCR Processor Test")
    print("=" * 50)
    
    # Get engine info
    engine_info = ocr_processor.get_engine_info()
    print(f"Engine: {engine_info['engine']}")
    print(f"Language: {engine_info['language']}")
    print(f"Supported formats: {engine_info['supported_formats']}")
    
    # Test with mock files
    test_files = [
              "C:\\Users\\ragha\Documents\\My docs\\PAN.pdf"
                    # These would be real files in actual usage
    ]
    
    # Filter existing files
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("\nNo test files found. Creating mock processing demonstration...")
        # Create a temporary test file
        test_file = "test_document.txt"
        with open(test_file, 'w') as f:
            f.write("This is a test file for OCR demonstration.")
        
        # Process the file
        result = ocr_processor._process_document_thread(test_file)
        
        if result:
            print(f"\nProcessed: {result.image_path}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Extracted Text:\n{result.text}")
            
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
    else:
        # Process existing files
        for file_path in existing_files:
            print(f"\nProcessing: {file_path}")
            result = ocr_processor._process_document_thread(file_path, threading.Event())            

            if isinstance(result, dict):
               # PDF
               print(f"PDF with {len(result)} pages")
               for page_num, page_result in result.items():
                  text_preview = page_result.text[:100] + "..." if len(page_result.text) > 100 else page_result.text
                  print(f"Confidence: {page_result.confidence:.2f}")
                  print(f"  Page {page_num}: {text_preview}")
            
            else:
                # Single document
                text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Text Preview: {result.text[:100]}...")