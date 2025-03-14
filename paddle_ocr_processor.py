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
            PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, rec_char_dict_path=None, 
              det_db_thresh=0.3, det_db_box_thresh=0.3, rec_thresh=0.3),
            PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, rec_char_dict_path=None, 
              det_db_thresh=0.5, det_db_box_thresh=0.3, rec_thresh=0.3),
            PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, rec_char_dict_path=None, 
              det_db_thresh=0.4, det_db_box_thresh=0.3, rec_thresh=0.3)
        ]
        logger.info(f"Initialized PaddleOCR with lang={lang}, GPU={use_gpu}")

    def preprocess_for_paddle(self, img_array):
        """
        Preprocess image specifically for PaddleOCR
        """
        # Initialize scale_factor with default value to avoid 'not associated with a value' error
        scale_factor = 1.0
        
        # Check if image is small and needs scaling
        if img_array.shape[0] < 50 or img_array.shape[1] < 50:  # Small image
            scale_factor = 2.0

        # Apply scaling if needed
        if scale_factor != 1.0:
            img_array = cv2.resize(img_array, (0, 0), fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_CUBIC)

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
                        # Special handling for digits
                        engine_text = ""
                        engine_conf = 0.0
                        
                        if paddle_result and len(paddle_result) > 0:
                            # For minimal_character, take ANY detected text with reasonable confidence
                            all_texts = []
                            all_confs = []
                            
                            for line in paddle_result:
                                for item in line:
                                    text = item[1][0].strip()  # Get the text
                                    conf = float(item[1][1])   # Get the confidence
                                    
                                    if text and len(text) > 0:
                                        all_texts.append(text[0])  # Take first character
                                        all_confs.append(conf)
                            
                            # If any text was found, use the highest confidence one
                            if all_texts:
                                best_idx = all_confs.index(max(all_confs))
                                engine_text = all_texts[best_idx]
                                engine_conf = all_confs[best_idx]
                
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
    
    def verify_results(self, results, confidences, data_type='text'):
        """
        Verify results from multiple OCR engines and return the most reliable one
        with enhanced handling for different data types
        """
        # For minimal_character, we want to keep any detected character
        if data_type == 'minimal_character':
            # Filter out empty results but keep all valid ones regardless of confidence
            valid_results = [(res, conf) for res, conf in zip(results, confidences) if res]
            
            # If all results are empty, return empty
            if not valid_results:
                return "", 0.0
            
            # For single character recognition, prefer results that agree
            result_counts = {}
            for res, conf in valid_results:
                # Take only first character for consistency
                if len(res) > 0:
                    char = res[0]
                    if char in result_counts:
                        result_counts[char]["count"] += 1
                        result_counts[char]["total_conf"] += conf
                    else:
                        result_counts[char] = {"count": 1, "total_conf": conf}
            
            # Sort by occurrence count first, then by confidence
            sorted_results = sorted(
                result_counts.items(),
                key=lambda x: (x[1]["count"], x[1]["total_conf"]),
                reverse=True
            )
            
            if sorted_results:
                best_char, data = sorted_results[0]
                agreement_factor = data["count"] / len(self.ocr_engines)
                confidence = (data["total_conf"] / data["count"]) * agreement_factor * 100
                return best_char, min(confidence, 100.0)
            else:
                return "", 0.0
        
        else:
            # For regular text, use a lower confidence threshold
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