import os
import time
import numpy as np
from PIL import Image
import cv2
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class PaddleOCRProcessor:
    """
    PaddleOCR processor with redundant verification
    """
    def __init__(self, lang='en', use_gpu=False):
        """
        Initialize PaddleOCR with specified language and GPU settings
        """
        # Initialize three separate instances for verification
        self.ocr_engines = [
            PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, det_model_dir=None, rec_model_dir=None,
         cls_model_dir=None, use_space_char=False)),
            PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, det_model_dir=None, rec_model_dir=None,
         cls_model_dir=None, use_space_char=False)),
            PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, det_model_dir=None, rec_model_dir=None,
         cls_model_dir=None, use_space_char=False))
        ]
        logger.info(f"Initialized PaddleOCR with lang={lang}, GPU={use_gpu}")

    def preprocess_for_paddle(self, img_array):
        """
        Preprocess image specifically for PaddleOCR
        """
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Additional preprocessing specific for PaddleOCR
        # Apply adaptive thresholding for better contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to RGB
        processed = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)
        return processed

    def process_image(self, image, data_type='text'):
        """
        Process an image with PaddleOCR with 3x redundant verification
        """
        start_time = time.time()
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Preprocess for PaddleOCR
        processed_img = self.preprocess_for_paddle(img_array)
        
        results = []
        confidences = []
        
        # Run prediction with all three engines
        for i, engine in enumerate(self.ocr_engines):
            try:
                logger.info(f"Running PaddleOCR engine {i+1}")
                paddle_result = engine.ocr(processed_img, cls=True)
                
                # Process results based on data type
                if data_type == 'text':
                    # Extract text from all detected regions
                    engine_text = ""
                    engine_conf = 0.0
                    
                    if paddle_result and len(paddle_result) > 0:
                        for line in paddle_result:
                            for item in line:
                                text = item[1][0]  # Get the text
                                conf = float(item[1][1])  # Get the confidence
                                engine_text += text + " "
                                engine_conf += conf
                        
                        if len(paddle_result) > 0:
                            engine_conf /= len(paddle_result)  # Average confidence
                    
                    results.append(engine_text.strip())
                    confidences.append(engine_conf)
                
                elif data_type == 'minimal_character':
                    # For single character, get highest confidence character
                    engine_text = ""
                    engine_conf = 0.0
                    
                    if paddle_result and len(paddle_result) > 0:
                        # Find the character with highest confidence
                        max_conf = 0.0
                        for line in paddle_result:
                            for item in line:
                                text = item[1][0]  # Get the text
                                conf = float(item[1][1])  # Get the confidence
                                
                                if conf > max_conf and len(text.strip()) > 0:
                                    engine_text = text.strip()[0]  # Take first character
                                    max_conf = conf
                    
                    results.append(engine_text)
                    confidences.append(max_conf)
                
                elif data_type == 'checkbox':
                    # For checkbox, detect if there's any content
                    has_content = False
                    engine_conf = 0.0
                    
                    if paddle_result and len(paddle_result) > 0:
                        has_content = True
                        engine_conf = 0.9  # High confidence if anything detected
                    
                    results.append("1" if has_content else "")
                    confidences.append(engine_conf)
                
            except Exception as e:
                logger.error(f"Error in PaddleOCR engine {i+1}: {str(e)}")
                results.append("")
                confidences.append(0.0)
        
        # Compare results from all three engines for verification
        final_result, confidence = self.verify_results(results, confidences)
        
        processing_time = time.time() - start_time
        logger.info(f"PaddleOCR processing completed in {processing_time:.2f}s with result: '{final_result}'")
        
        return final_result, confidence
    
    def verify_results(self, results, confidences):
        """
        Verify results from multiple OCR engines and return the most reliable one
        """
        # Filter out empty results
        valid_results = [(res, conf) for res, conf in zip(results, confidences) if res]
        
        if not valid_results:
            return "", 0.0
        
        # Check for exact matches
        result_counts = {}
        for res, conf in valid_results:
            if res in result_counts:
                result_counts[res]["count"] += 1
                result_counts[res]["total_conf"] += conf
            else:
                result_counts[res] = {"count": 1, "total_conf": conf}
        
        # Find the most common result
        max_count = 0
        best_result = ""
        best_confidence = 0.0
        
        for res, data in result_counts.items():
            if data["count"] > max_count:
                max_count = data["count"]
                best_result = res
                best_confidence = data["total_conf"] / data["count"]
            elif data["count"] == max_count and data["total_conf"] / data["count"] > best_confidence:
                # If tie, use the one with higher confidence
                best_result = res
                best_confidence = data["total_conf"] / data["count"]
        
        # Scale confidence based on agreement level
        agreement_factor = max_count / len(self.ocr_engines)
        adjusted_confidence = best_confidence * agreement_factor * 100  # Convert to percentage
        
        return best_result, min(adjusted_confidence, 100.0)  # Cap at 100%


# Add to main.py
def process_with_paddle_ocr(image, data_type, region_name):
    """Process a region using PaddleOCR with verification."""
    try:
        # Initialize PaddleOCR if not already done
        global paddle_processor
        if 'paddle_processor' not in globals():
            paddle_processor = PaddleOCRProcessor(lang='en', use_gpu=False)
        
        # Process with PaddleOCR
        text, confidence = paddle_processor.process_image(image, data_type)
        
        logger.info(f"PaddleOCR result for {region_name}: '{text}' (confidence: {confidence:.2f}%)")
        return text, confidence
        
    except Exception as e:
        logger.error(f"PaddleOCR error for {region_name}: {str(e)}")
        return "", 0.0