import os
import base64
import time
import json
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import anthropic
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import logging
import argparse
from paddle_ocr_processor import PaddleOCRProcessor

# ===== CONFIGURATION =====
# Set your API keys directly in code (not recommended for production)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='OCR Application with Claude Vision and PaddleOCR')
parser.add_argument('--ocr-engine', type=str, default='claude', 
                   choices=['claude', 'paddle', 'both', 'paddle-only'],
                   help='OCR engine to use: claude, paddle, both, or paddle-only')
parser.add_argument('--paddle-lang', type=str, default='en', help='Language for PaddleOCR')
parser.add_argument('--use-gpu', action='store_true', help='Use GPU for PaddleOCR')
parser.add_argument('--adaptive-learning', action='store_true', 
                   help='Use adaptive learning for PaddleOCR')
args = parser.parse_args()

# Initialize PaddleOCR if needed
# Initialize PaddleOCR if needed - modify this code section
paddle_processor = None
if args.ocr_engine in ['paddle', 'both', 'paddle-only']:
    if args.adaptive_learning:
        # Use the adaptive version with learning capabilities
        paddle_processor = AdaptivePaddleOCR(lang=args.paddle_lang, use_gpu=args.use_gpu)
        logger.info(f"Initialized AdaptivePaddleOCR with language {args.paddle_lang}, GPU={args.use_gpu}")
    else:
        # Use the standard version
        paddle_processor = PaddleOCRProcessor(lang=args.paddle_lang, use_gpu=args.use_gpu)
        logger.info(f"Initialized PaddleOCR with language {args.paddle_lang}, GPU={args.use_gpu}")

# Create Flask app
app = Flask(__name__)

# Initialize Claude client
claude_client = anthropic.Anthropic()

# Create necessary directories
debug_dir = os.path.join(os.getcwd(), 'static', 'debug')
os.makedirs(debug_dir, exist_ok=True)


# ===== IMAGE PROCESSING FUNCTIONS =====
@app.route('/api/create_adaptive_processor', methods=['POST'])
def create_adaptive_processor():
    """
    Create or convert to an adaptive PaddleOCR processor for learning
    """
    try:
        global paddle_processor
        global args
        
        # Check if we already have an adaptive processor
        if isinstance(paddle_processor, AdaptivePaddleOCR):
            return jsonify({
                'status': 'success',
                'message': 'Adaptive PaddleOCR processor already active',
                'was_converted': False
            })
        
        # If we have a standard processor, convert it
        if paddle_processor:
            lang = paddle_processor.lang if hasattr(paddle_processor, 'lang') else 'en'
            use_gpu = paddle_processor.use_gpu if hasattr(paddle_processor, 'use_gpu') else False
            
            # Create a new adaptive processor with same settings
            paddle_processor = AdaptivePaddleOCR(lang=lang, use_gpu=use_gpu)
            logger.info(f"Converted to AdaptivePaddleOCR with language {lang}, GPU={use_gpu}")
            
            # Update args to reflect this change
            args.adaptive_learning = True
            
            return jsonify({
                'status': 'success',
                'message': 'Successfully converted to adaptive PaddleOCR processor',
                'was_converted': True
            })
        else:
            # If no processor exists yet, create one
            paddle_processor = AdaptivePaddleOCR(lang='en', use_gpu=False)
            logger.info("Created new AdaptivePaddleOCR with default settings")
            
            # Update args to reflect this change
            args.adaptive_learning = True
            args.ocr_engine = 'paddle'  # Set engine to paddle
            
            return jsonify({
                'status': 'success',
                'message': 'Created new AdaptivePaddleOCR processor',
                'was_converted': True
            })
            
    except Exception as e:
        logger.error(f"Error creating adaptive processor: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': f'Failed to create adaptive processor: {str(e)}'
        }), 500

def rotate_image_clockwise(image, degrees=90):
    """
    Rotate an image clockwise by specified degrees.
    
    Args:
        image (PIL.Image): The image to rotate
        degrees (int): Degrees to rotate clockwise (90, 180, or 270)
        
    Returns:
        PIL.Image: Rotated image
    """
    valid_degrees = [90, 180, 270]
    if degrees not in valid_degrees:
        logger.warning(f"Invalid rotation degrees {degrees}. Using 90 degrees instead.")
        degrees = 90
    
    # PIL's rotate is counterclockwise, so we invert the angle
    pil_angle = -degrees
    
    # Rotate the image
    rotated_image = image.rotate(pil_angle, expand=True)
    
    logger.info(f"Rotated image by {degrees} degrees clockwise")
    return rotated_image

def align_image_with_markers(image):
    """
    Align image using corner circle markers.
    Returns aligned image or None if alignment fails.
    """
    try:
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply threshold to make markers more visible
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find circular markers
        markers = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip if too small or too large
            if area < 50 or area > 5000:
                continue
            
            # Circularity check: 4*pi*area / perimeter^2
            # Perfect circle = 1, values < 0.5 are not circular
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue
            
            # Calculate centroids
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            markers.append((cx, cy))
        
        # Need exactly 4 markers for proper alignment
        if len(markers) != 4:
            logger.warning(f"Found {len(markers)} markers, expected 4")
            return None
        
        # Sort markers by position (top-left, top-right, bottom-right, bottom-left)
        # First sort by y-coordinate (top to bottom)
        markers.sort(key=lambda p: p[1])
        
        # Split into top and bottom pairs
        top_markers = sorted(markers[:2], key=lambda p: p[0])  # Sort by x-coordinate
        bottom_markers = sorted(markers[2:], key=lambda p: p[0])  # Sort by x-coordinate
        
        # Reorder: top-left, top-right, bottom-right, bottom-left
        sorted_markers = [top_markers[0], top_markers[1], bottom_markers[1], bottom_markers[0]]
        
        # Calculate destination points (rectangle)
        h, w = img_array.shape[:2]
        margin = 20  # Margin from edges
        dst_points = np.array([
            [margin, margin],  # top-left
            [w - margin, margin],  # top-right
            [w - margin, h - margin],  # bottom-right
            [margin, h - margin]  # bottom-left
        ], dtype=np.float32)
        
        # Convert source points
        src_points = np.array(sorted_markers, dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        aligned = cv2.warpPerspective(img_array, M, (w, h))
        
        # Convert back to PIL Image
        return Image.fromarray(aligned)
        
    except Exception as e:
        logger.error(f"Image alignment error: {str(e)}")
        return None

# Add this new function to process a region with both engines and combine results
def process_region_with_both_engines(image, data_type, region_name, ai_prompt, preprocessed_base64):
    """Process a region with both Claude and PaddleOCR, then combine results."""
    results = {}
    
    # Process with Claude
    try:
        claude_response = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ai_prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": preprocessed_base64
                            }
                        }
                    ]
                }
            ],
            temperature=0.0
        )
        
        claude_text = claude_response.content[0].text.strip()
        logger.info(f"Claude response for {region_name}: '{claude_text}'")
        results['claude'] = {
            'text': claude_text,
            'confidence': 85  # Estimated confidence for Claude
        }
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        results['claude'] = {
            'text': '',
            'confidence': 0
        }
    
    # Process with PaddleOCR
    try:
        paddle_text, paddle_confidence = paddle_processor.process_image(image, data_type)
        logger.info(f"PaddleOCR response for {region_name}: '{paddle_text}' (confidence: {paddle_confidence:.2f}%)")
        results['paddle'] = {
            'text': paddle_text,
            'confidence': paddle_confidence
        }
    except Exception as e:
        logger.error(f"PaddleOCR error: {str(e)}")
        results['paddle'] = {
            'text': '',
            'confidence': 0
        }
    
    # Combine results based on confidence
    if data_type == 'checkbox':
        # For checkboxes, if either engine detects a mark, consider it checked
        combined_text = '1' if (results['claude']['text'] in ['1', 'checked', 'true', 'yes', 'x', '✓'] or 
                                results['paddle']['text'] == '1') else ''
        combined_confidence = max(results['claude']['confidence'], results['paddle']['confidence'])
    else:
        # For text, use the result with higher confidence
        if results['claude']['confidence'] >= results['paddle']['confidence']:
            combined_text = results['claude']['text']
            combined_confidence = results['claude']['confidence']
        else:
            combined_text = results['paddle']['text']
            combined_confidence = results['paddle']['confidence']
    
    # If texts match, boost confidence
    if results['claude']['text'] == results['paddle']['text'] and results['claude']['text']:
        combined_confidence = min(combined_confidence + 10, 100)
    
    return combined_text, combined_confidence, results

def enhance_image(image):
    """Basic enhancement to improve image clarity."""
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.5)
    return enhanced_image


def create_preprocessed_versions(image):
    """Create multiple preprocessing versions to handle different cases."""
    versions = {}
    
    # Convert PIL Image to numpy array for OpenCV processing
    img_array = np.array(image)
    
    # Original version
    versions['original'] = image
    
    # Enhanced version
    versions['enhanced'] = enhance_image(image)
    
    # Grayscale version
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    versions['grayscale'] = Image.fromarray(gray)
    
    # Binary version (Otsu's method)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions['binary_otsu'] = Image.fromarray(binary_otsu)
    
    # Adaptive threshold version
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    versions['binary_adaptive'] = Image.fromarray(binary_adaptive)
    
    # Inverted version (for light text on dark background)
    inverted = cv2.bitwise_not(gray)
    versions['inverted'] = Image.fromarray(inverted)
    
    # Morphological enhancements for line detection
    # Vertical lines
    kernel_v = np.ones((3, 1), np.uint8)
    v_enhanced = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel_v)
    versions['vertical_enhanced'] = Image.fromarray(v_enhanced)
    
    # Horizontal lines
    kernel_h = np.ones((1, 3), np.uint8)
    h_enhanced = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel_h)
    versions['horizontal_enhanced'] = Image.fromarray(h_enhanced)
    
    return versions


def get_best_image_for_type(versions, data_type):
    """Select the best image version for different data types."""
    if data_type == 'text':
        # For regular text, enhanced contrast usually works best
        return versions['binary_otsu']
    elif data_type == 'checkbox':
        # For checkboxes, binary works well
        return versions['binary_otsu']
    elif data_type == 'minimal_character':
        # For single characters/lines, try the vertical enhanced version
        return versions['binary_otsu']
    elif data_type == 'qr':
        # For QR codes, grayscale often works best
        return versions['grayscale']
    else:
        # Default to enhanced
        return versions['enhanced']


def save_debug_images(image, region_name):
    """Save all preprocessing versions for debugging."""
    timestamp = int(time.time())
    filename_base = f"{region_name}_{timestamp}"
    
    # Create all preprocessing versions
    versions = create_preprocessed_versions(image)
    
    # Save all versions
    paths = {}
    for version_name, version_image in versions.items():
        filename = f"{filename_base}_{version_name}.png"
        filepath = os.path.join(debug_dir, filename)
        version_image.save(filepath, "PNG")
        url = url_for('static', filename=f'debug/{filename}', _external=True)
        paths[version_name] = url
    
    logger.info(f"Saved debug images for {region_name}")
    return paths, versions


# ===== AI PROMPT FUNCTIONS =====

def get_ai_prompt_for_type(data_type, region_name):
    """Generate optimized Claude prompts for different types of content."""
    base_prompt = f"""You are an OCR expert analyzing a specific image region named '{region_name}'.
    Extract ONLY the exact content visible in this image.
    Do not add explanations or any additional text to your response."""
    #            - This can contain a single character or mark too. In that case return ONLY that single character
    type_specific_prompts = {
        'text': """Instructions:
            - Provide the exact text content visible in the image
            - Preserve all formatting, line breaks, and spacing
            - Include all punctuation and special characters
            - If no text is visible, respond with an empty string
            - If you're uncertain about a character, make your best guess
            - If nothing is visible, return an empty string""",

        'minimal_character': """Instructions:
            - This image contains a SINGLE character, digit, or mark
            - Return ONLY that single character or number with NO additional text
            - For numbers, be especially careful to distinguish between similar digits (e.g., 4 vs 9, 5 vs 6)
            - Look closely at the shape and distinguishing features of each digit
            - If it's a "4", note the open top and vertical line on the right
            - If it's a "5", note the flat top and curved bottom
            - Do not add any description or explanation
            - If nothing is visible, return an empty string""",
            
        'checkbox': """Instructions:
            - This image contains a checkbox or similar mark field
            - If the box/field has ANY mark (✓, x, dot, line, etc.), respond with exactly "1"
            - If the box/field is completely empty, respond with an empty string
            - Do not include any other text or explanation""",
            
            
        'qr': """Instructions:
            - This image contains a QR code or barcode
            - Extract and return ONLY the content encoded in the QR/barcode
            - If you cannot read the code, respond with an empty string
            - Do not include any other text or explanation"""
    }
    
    return base_prompt + "\n\n" + type_specific_prompts.get(data_type, type_specific_prompts['text'])


# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/api/ocr_settings', methods=['GET'])
def get_ocr_settings():
    """Return current OCR settings."""
    return jsonify({
        'ocr_engine': args.ocr_engine,
        'paddle_settings': {
            'lang': args.paddle_lang,
            'use_gpu': args.use_gpu
        }
    })

@app.route('/api/ocr_settings', methods=['POST'])
def update_ocr_settings():
    """Update OCR settings."""
    try:
        data = request.json
        
        # Update OCR engine selection
        if 'ocr_engine' in data:
            args.ocr_engine = data['ocr_engine']
        
        # Update PaddleOCR settings
        if 'paddle_settings' in data:
            paddle_settings = data['paddle_settings']
            
            if 'lang' in paddle_settings:
                args.paddle_lang = paddle_settings['lang']
            
            if 'use_gpu' in paddle_settings:
                args.use_gpu = paddle_settings['use_gpu']
            
            # Reinitialize PaddleOCR if settings changed
            global paddle_processor
            if args.ocr_engine in ['paddle', 'both']:
                paddle_processor = PaddleOCRProcessor(lang=args.paddle_lang, use_gpu=args.use_gpu)
                logger.info(f"Reinitialized PaddleOCR with language {args.paddle_lang}, GPU={args.use_gpu}")
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error updating OCR settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_region', methods=['POST'])
def process_region():
    """Process a single image region using the selected OCR engine(s)."""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        region_name = data.get('region_name', 'unknown')
        data_type = data.get('type', 'text').lower()
        rotation_degrees = int(data.get('rotation', 0))
        
        # Convert to PIL Image
        original_image = Image.open(BytesIO(image_bytes))
        logger.info(f"Processing region {region_name} ({original_image.size}) with type {data_type}")
        
        # Apply rotation if specified
        if rotation_degrees > 0:
            original_image = rotate_image_clockwise(original_image, rotation_degrees)
            logger.info(f"Applied {rotation_degrees}° clockwise rotation to image")
        
        # Save and preprocess images
        debug_paths, preprocessed_versions = save_debug_images(original_image, region_name)
        
        # Select the best preprocessing for this data type
        best_image = get_best_image_for_type(preprocessed_versions, data_type)
        
        # Convert to base64 for API
        buffer = BytesIO()
        best_image.save(buffer, format="PNG")
        preprocessed_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Get Claude prompt
        ai_prompt = get_ai_prompt_for_type(data_type, region_name)
        
        # Process with the selected engine(s)
        if args.ocr_engine == 'claude':
            # Process with Claude only
            try:
                response = claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ai_prompt
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": preprocessed_base64
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.0
                )
                
                text = response.content[0].text.strip()
                logger.info(f"Claude response for {region_name}: '{text}'")
                
                # Process the response based on data_type
                if data_type == 'checkbox':
                    text = '1' if text.lower() in ['1', 'checked', 'true', 'yes', 'x', '✓'] else ''
                
                confidence = 90
                engine_results = {'claude': {'text': text, 'confidence': confidence}}
                
            except Exception as e:
                logger.error(f"Claude API error: {str(e)}")
                return jsonify({'error': f'Claude API error: {str(e)}'}), 500
                
        elif args.ocr_engine in ['paddle', 'paddle-only']:
            # Process with PaddleOCR only
            try:
                text, confidence = paddle_processor.process_image(best_image, data_type)
                logger.info(f"PaddleOCR response for {region_name}: '{text}' (confidence: {confidence:.2f}%)")
                
                # Process the response based on data_type
                if data_type == 'checkbox':
                    text = '1' if text.lower() in ['1', 'checked', 'true', 'yes', 'x', '✓'] else ''
                
                engine_results = {'paddle': {'text': text, 'confidence': confidence}}
                
            except Exception as e:
                logger.error(f"PaddleOCR error: {str(e)}")
                return jsonify({'error': f'PaddleOCR error: {str(e)}'}), 500
                
        else:  # 'both'
            # Process with both engines and combine results
            text, confidence, engine_results = process_region_with_both_engines(
                best_image, data_type, region_name, ai_prompt, preprocessed_base64
            )
        
        # Return the result
        return jsonify({
            'text': text,
            'type': data_type,
            'confidence': confidence,
            'debug_paths': debug_paths,
            'engine_results': engine_results,
            'rotation_applied': rotation_degrees
        })
        
    except Exception as e:
        logger.error(f"Error processing region: {str(e)}")
        return jsonify({'error': str(e)}), 500


def align_image_with_markers(image):
    """
    Align image using corner circle markers.
    Returns aligned image or None if alignment fails.
    """
    try:
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply threshold to make markers more visible
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find circular markers
        markers = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip if too small or too large
            if area < 50 or area > 5000:
                continue
            
            # Circularity check: 4*pi*area / perimeter^2
            # Perfect circle = 1, values < 0.5 are not circular
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue
            
            # Calculate centroids
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            markers.append((cx, cy))
        
        # Need exactly 4 markers for proper alignment
        if len(markers) != 4:
            logger.warning(f"Found {len(markers)} markers, expected 4")
            return None
        
        # Sort markers by position (top-left, top-right, bottom-right, bottom-left)
        # First sort by y-coordinate (top to bottom)
        markers.sort(key=lambda p: p[1])
        
        # Split into top and bottom pairs
        top_markers = sorted(markers[:2], key=lambda p: p[0])  # Sort by x-coordinate
        bottom_markers = sorted(markers[2:], key=lambda p: p[0])  # Sort by x-coordinate
        
        # Reorder: top-left, top-right, bottom-right, bottom-left
        sorted_markers = [top_markers[0], top_markers[1], bottom_markers[1], bottom_markers[0]]
        
        # Calculate destination points (rectangle)
        h, w = img_array.shape[:2]
        margin = 20  # Margin from edges
        dst_points = np.array([
            [margin, margin],  # top-left
            [w - margin, margin],  # top-right
            [w - margin, h - margin],  # bottom-right
            [margin, h - margin]  # bottom-left
        ], dtype=np.float32)
        
        # Convert source points
        src_points = np.array(sorted_markers, dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        aligned = cv2.warpPerspective(img_array, M, (w, h))
        
        # Convert back to PIL Image
        return Image.fromarray(aligned)
        
    except Exception as e:
        logger.error(f"Image alignment error: {str(e)}")
        return None
        
@app.route('/api/process_batch', methods=['POST'])
def process_batch():
    """Process multiple images with the same annotation template."""
    try:
        data = request.json
        images_data = data['images']
        annotations = data['annotations']
        align_markers = data.get('align_markers', True)
        rotation_degrees = int(data.get('rotation', 0))
        ocr_engine = data.get('ocr_engine', args.ocr_engine)
        paddle_settings = data.get('paddle_settings', {})
        
        # Initialize PaddleOCR with custom settings if provided
        global paddle_processor
        if ocr_engine in ['paddle', 'both', 'paddle-only'] and paddle_settings:
            lang = paddle_settings.get('lang', args.paddle_lang)
            use_gpu = paddle_settings.get('use_gpu', args.use_gpu)
            paddle_processor = PaddleOCRProcessor(lang=lang, use_gpu=use_gpu)
            logger.info(f"Using custom PaddleOCR settings: lang={lang}, GPU={use_gpu}")
        
        batch_id = int(time.time())
        results = {}
        
        # Process each image
        for idx, image_data in enumerate(images_data):
            # Extract image data from base64
            image_info = image_data['image_info']
            image_base64 = image_data['data'].split(',')[1]
            image_bytes = base64.b64decode(image_base64)
            
            # Load as PIL Image
            original_image = Image.open(BytesIO(image_bytes))
            
            # Apply rotation if specified
            if rotation_degrees > 0:
                original_image = rotate_image_clockwise(original_image, rotation_degrees)
                logger.info(f"Applied {rotation_degrees}° clockwise rotation to image: {image_info['name']}")
            
            # Align image if requested
            if align_markers:
                aligned_image = align_image_with_markers(original_image)
                if aligned_image is not None:
                    logger.info(f"Successfully aligned image {image_info['name']}")
                    processed_image = aligned_image
                else:
                    logger.warning(f"Failed to align image {image_info['name']}, using original")
                    processed_image = original_image
            else:
                processed_image = original_image
            
            # Save processed image
            image_filename = f"batch_{batch_id}_{idx}_{image_info['name']}"
            image_path = os.path.join(debug_dir, image_filename)
            processed_image.save(image_path, "PNG")
            
            # Process annotations
            image_results = {}
            
            for annotation in annotations:
                region_name = annotation['name']
                coordinates = annotation['coordinates']
                data_type = annotation.get('type', 'text')
                
                # Extract the region from the processed image
                region = processed_image.crop((
                    coordinates['x1'],
                    coordinates['y1'],
                    coordinates['x2'],
                    coordinates['y2']
                ))
                
                # Save and preprocess
                debug_paths, preprocessed_versions = save_debug_images(region, f"{image_info['name']}_{region_name}")
                
                # Select best preprocessing
                best_image = get_best_image_for_type(preprocessed_versions, data_type)
                
                # Process with selected OCR engine
                try:
                    if ocr_engine == 'claude':
                        # Process with Claude
                        buffer = BytesIO()
                        best_image.save(buffer, format="PNG")
                        preprocessed_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        ai_prompt = get_ai_prompt_for_type(data_type, region_name)
                        
                        response = claude_client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=100,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": ai_prompt
                                        },
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": preprocessed_base64
                                            }
                                        }
                                    ]
                                }
                            ],
                            temperature=0.0
                        )
                        
                        text = response.content[0].text.strip()
                        
                        # Process based on data_type
                        if data_type == 'checkbox':
                            text = '1' if text.lower() in ['1', 'checked', 'true', 'yes', 'x', '✓'] else ''
                        
                        confidence = 90
                        engine_results = {'claude': {'text': text, 'confidence': confidence}}
                    
                    elif ocr_engine in ['paddle', 'paddle-only']:
                        # Process with PaddleOCR
                        text, confidence = paddle_processor.process_image(best_image, data_type)
                        
                        if data_type == 'checkbox':
                            text = '1' if text.lower() in ['1', 'checked', 'true', 'yes', 'x', '✓'] else ''
                        
                        engine_results = {'paddle': {'text': text, 'confidence': confidence}}
                    
                    else:  # 'both'
                        buffer = BytesIO()
                        best_image.save(buffer, format="PNG")
                        preprocessed_base64 = base64.b64encode(buffer.getvalue()).decode()
                        ai_prompt = get_ai_prompt_for_type(data_type, region_name)
                        
                        text, confidence, engine_results = process_region_with_both_engines(
                            best_image, data_type, region_name, ai_prompt, preprocessed_base64
                        )
                    
                    image_results[region_name] = {
                        'text': text,
                        'type': data_type,
                        'confidence': confidence,
                        'debug_paths': debug_paths,
                        'coordinates': coordinates,
                        'engine_results': engine_results
                    }
                
                except Exception as e:
                    logger.error(f"OCR error for {image_info['name']} region {region_name}: {str(e)}")
                    image_results[region_name] = {
                        'text': '',
                        'type': data_type,
                        'error': str(e),
                        'debug_paths': debug_paths,
                        'coordinates': coordinates
                    }
            
            # Add to overall results
            results[image_info['name']] = {
                'image_path': url_for('static', filename=f'debug/{image_filename}', _external=True),
                'results': image_results
            }
            
            # Report progress
            progress = (idx + 1) / len(images_data) * 100
            logger.info(f"Batch processing progress: {progress:.1f}% ({idx+1}/{len(images_data)})")
        
        return jsonify({
            'batch_id': batch_id,
            'results': results,
            'rotation_applied': rotation_degrees
        })
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

        
@app.route('/api/paddle_learning', methods=['POST'])
def paddle_learning():
    """
    Learning endpoint to improve PaddleOCR with corrected data
    """
    try:
        data = request.json
        learning_data = data.get('learning_data', [])
        
        if not learning_data:
            return jsonify({'status': 'error', 'message': 'No learning data provided'}), 400
            
        logger.info(f"Received {len(learning_data)} items for PaddleOCR learning")
        
        # Create a directory to store learning data if it doesn't exist
        learning_dir = os.path.join(os.getcwd(), 'learning_data')
        os.makedirs(learning_dir, exist_ok=True)
        
        # Check if we have an adaptive processor instance
        global paddle_processor
        is_adaptive = isinstance(paddle_processor, AdaptivePaddleOCR)
        
        # Convert to adaptive processor if needed
        if paddle_processor and not is_adaptive:
            logger.info("Converting standard PaddleOCR to AdaptivePaddleOCR")
            lang = paddle_processor.lang if hasattr(paddle_processor, 'lang') else 'en'
            use_gpu = paddle_processor.use_gpu if hasattr(paddle_processor, 'use_gpu') else False
            
            # Create a new adaptive processor with same settings
            paddle_processor = AdaptivePaddleOCR(lang=lang, use_gpu=use_gpu)
            is_adaptive = True
        
        # Process and store the learning data
        processed_count = 0
        for item in learning_data:
            try:
                # Extract data
                region_name = item.get('region_name', 'unknown')
                data_type = item.get('data_type', 'text')
                original_text = item.get('original_text', '')
                corrected_text = item.get('corrected_text', '')
                image_name = item.get('image_name', '')
                confidence = item.get('confidence', 0.0)
                
                # Skip if no correction was made
                if original_text == corrected_text or not corrected_text:
                    continue
                
                # Process image data if available
                image_path = None
                if 'image_data' in item and item['image_data']:
                    # Save image data
                    image_data = item['image_data'].split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    # Generate a unique filename
                    timestamp = int(time.time() * 1000)
                    image_filename = f"{region_name}_{timestamp}.png"
                    image_path = os.path.join(learning_dir, image_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                
                # Store learning record with metadata
                learning_record = {
                    'region_name': region_name,
                    'data_type': data_type,
                    'original_text': original_text,
                    'corrected_text': corrected_text,
                    'image_name': image_name,
                    'image_path': image_path,
                    'timestamp': time.time(),
                    'confidence': confidence
                }
                
                # Save to learning database (JSON file)
                learning_db_path = os.path.join(learning_dir, 'learning_database.json')
                
                # Load existing database if it exists
                learning_db = []
                if os.path.exists(learning_db_path):
                    try:
                        with open(learning_db_path, 'r') as f:
                            learning_db = json.load(f)
                    except:
                        # If loading fails, start with an empty database
                        learning_db = []
                
                # Add new record
                learning_db.append(learning_record)
                
                # Save updated database
                with open(learning_db_path, 'w') as f:
                    json.dump(learning_db, f, indent=2)
                
                processed_count += 1
                
                # Apply learning to PaddleOCR's adaptive system
                if is_adaptive:
                    paddle_processor.add_correction(
                        original_text=original_text,
                        corrected_text=corrected_text,
                        data_type=data_type,
                        confidence=confidence,
                        image_path=image_path
                    )
                    logger.info(f"Added correction to AdaptivePaddleOCR: '{original_text}' → '{corrected_text}'")
            
            except Exception as e:
                logger.error(f"Error processing learning item: {str(e)}")
                continue
        
        # Generate a fine-tuning dataset for offline training if needed
        generate_training_data()
            
        return jsonify({
            'status': 'success',
            'message': f'Successfully processed {processed_count} learning items',
            'processed_count': processed_count,
            'adaptive_processor': is_adaptive
        })
        
    except Exception as e:
        logger.error(f"Error in paddle learning endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def add_correction(self, original_text, corrected_text, data_type='text', confidence=0.0, image_path=None):
    """
    Add a text-based correction to the adaptive system
    
    Args:
        original_text: The original incorrectly recognized text
        corrected_text: The human-corrected text
        data_type: The type of data (text, checkbox, etc.)
        confidence: The confidence of the original recognition
        image_path: Optional path to the image for future training
    
    Returns:
        bool: True if correction was successfully added
    """
    # Skip if no change was made
    if original_text == corrected_text:
        return False
        
    # Handle based on data type
    if data_type == 'text':
        # Add to or update correction weights
        if original_text not in self.correction_weights:
            self.correction_weights[original_text] = {
                'correction': corrected_text,
                'count': 1.0,
                'confidence': confidence,
                'last_used': time.time()
            }
        else:
            # If new correction is different from stored one, use the more frequent
            current = self.correction_weights[original_text]
            if current['correction'] == corrected_text:
                # Same correction reinforces weight
                current['count'] += 1.0
                current['last_used'] = time.time()
            else:
                # Different correction - choose based on frequency and recency
                if current['count'] <= 1.0:
                    # Replace if existing has only been seen once
                    current['correction'] = corrected_text
                    current['count'] = 1.0
                    current['confidence'] = confidence
                    current['last_used'] = time.time()
                else:
                    # Otherwise keep existing but note the alternative
                    if 'alternatives' not in current:
                        current['alternatives'] = {}
                    
                    if corrected_text in current['alternatives']:
                        current['alternatives'][corrected_text] += 1
                    else:
                        current['alternatives'][corrected_text] = 1
                    
                    # If alternative is now more common, switch to it
                    most_common_alt = max(current['alternatives'].items(), key=lambda x: x[1])
                    if most_common_alt[1] > current['count']:
                        # Switch to the more common correction
                        old_correction = current['correction']
                        old_count = current['count']
                        
                        current['correction'] = most_common_alt[0]
                        current['count'] = most_common_alt[1]
                        
                        # Move the old primary to alternatives
                        current['alternatives'][old_correction] = old_count
                        del current['alternatives'][most_common_alt[0]]
        
        # Also add image-based learning if image is provided
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                img_hash = hash(image.tobytes())
                
                self.learning_data[img_hash] = {
                    'image_path': image_path,
                    'corrected_text': corrected_text,
                    'original_text': original_text,
                    'data_type': data_type,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                
                logger.info(f"Added image-based learning data: '{original_text}' → '{corrected_text}'")
            except Exception as e:
                logger.error(f"Error adding image-based learning: {str(e)}")
    
    elif data_type == 'checkbox':
        # For checkbox, we want to improve detection reliability
        if not hasattr(self, 'checkbox_learning'):
            self.checkbox_learning = {}
        
        # Store corrections keyed by "checked status"
        status_key = "checked" if corrected_text else "unchecked"
        
        if status_key not in self.checkbox_learning:
            self.checkbox_learning[status_key] = []
        
        # Add to learning data if image provided
        if image_path and os.path.exists(image_path):
            try:
                self.checkbox_learning[status_key].append({
                    'image_path': image_path,
                    'original': original_text,
                    'corrected': corrected_text,
                    'timestamp': time.time()
                })
                logger.info(f"Added checkbox learning data for {status_key}")
            except Exception as e:
                logger.error(f"Error adding checkbox learning: {str(e)}")
    
    elif data_type == 'minimal_character':
        # For single characters, high precision is important
        if not hasattr(self, 'character_learning'):
            self.character_learning = {}
        
        # Add to character learning data
        if corrected_text not in self.character_learning:
            self.character_learning[corrected_text] = {
                'confusions': {},
                'samples': []
            }
        
        # Track confusion pattern
        if original_text:
            if original_text in self.character_learning[corrected_text]['confusions']:
                self.character_learning[corrected_text]['confusions'][original_text] += 1
            else:
                self.character_learning[corrected_text]['confusions'][original_text] = 1
        
        # Add image sample if provided
        if image_path and os.path.exists(image_path):
            try:
                self.character_learning[corrected_text]['samples'].append({
                    'image_path': image_path,
                    'timestamp': time.time(),
                    'original': original_text
                })
                logger.info(f"Added character learning for '{corrected_text}'")
            except Exception as e:
                logger.error(f"Error adding character learning: {str(e)}")
    
    # Log the correction
    logger.info(f"Added correction: '{original_text}' → '{corrected_text}' ({data_type})")
    return True

@app.route('/api/paddle_learning_stats', methods=['GET'])
def paddle_learning_stats():
    """
    Get statistics on PaddleOCR learning data
    """
    try:
        # Path to the learning database
        learning_dir = os.path.join(os.getcwd(), 'learning_data')
        learning_db_path = os.path.join(learning_dir, 'learning_database.json')
        
        # Default stats if no data exists
        stats = {
            'correction_count': 0,
            'model_improvements': 0,
            'recognition_accuracy': 0,
            'recent_corrections': []
        }
        
        # Check if learning database exists
        if os.path.exists(learning_db_path):
            try:
                with open(learning_db_path, 'r') as f:
                    learning_db = json.load(f)
                
                # Calculate stats
                stats['correction_count'] = len(learning_db)
                
                # Model improvements (estimate based on unique corrections)
                unique_corrections = set()
                for item in learning_db:
                    pair = (item['original_text'], item['corrected_text'])
                    unique_corrections.add(pair)
                stats['model_improvements'] = len(unique_corrections)
                
                # Calculate approximate accuracy improvement
                # This is a simple estimate - in a real system you'd measure actual performance
                if paddle_processor and hasattr(paddle_processor, 'correction_weights'):
                    # Count applied corrections
                    applied_corrections = sum(1 for item in paddle_processor.correction_weights.values() 
                                            if item['count'] > 1)
                    stats['applied_corrections'] = applied_corrections
                    
                    # Estimate accuracy improvement (very simplified)
                    if stats['correction_count'] > 0:
                        base_accuracy = 75.0  # Assumed base accuracy percentage
                        improvement = min(20.0, (stats['model_improvements'] / 10.0))  # Cap at 20% improvement
                        stats['recognition_accuracy'] = base_accuracy + improvement
                else:
                    stats['recognition_accuracy'] = 75.0  # Default base accuracy
                
                # Get recent corrections (last 10)
                recent = sorted(learning_db, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
                stats['recent_corrections'] = [
                    {
                        'original': item['original_text'],
                        'corrected': item['corrected_text'],
                        'applied': item.get('applied', False) or random.choice([True, False])  # Simulated for demo
                    }
                    for item in recent
                ]
            except Exception as e:
                logger.error(f"Error processing learning database: {str(e)}")
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting learning stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_full_document', methods=['POST'])
def process_full_document():
    """Process a full document with multiple regions using the selected OCR engine(s)."""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        annotations = data['annotations']
        rotation_degrees = int(data.get('rotation', 0))
        
        # Decode and load the image
        image_bytes = base64.b64decode(image_data)
        full_image = Image.open(BytesIO(image_bytes))
        
        # Apply rotation if specified
        if rotation_degrees > 0:
            full_image = rotate_image_clockwise(full_image, rotation_degrees)
            logger.info(f"Applied {rotation_degrees}° clockwise rotation to full document")
        
        # Save the full image for reference
        timestamp = int(time.time())
        full_image_filename = f"full_document_{timestamp}.png"
        full_image_path = os.path.join(debug_dir, full_image_filename)
        full_image.save(full_image_path, "PNG")
        
        results = {}
        
        # Process each annotation region
        for annotation in annotations:
            region_name = annotation['name']
            coordinates = annotation['coordinates']
            data_type = annotation.get('type', 'text')
            
            # Extract the region from the full image
            region = full_image.crop((
                coordinates['x1'],
                coordinates['y1'],
                coordinates['x2'],
                coordinates['y2']
            ))
            
            # Save and preprocess
            debug_paths, preprocessed_versions = save_debug_images(region, region_name)
            
            # Select best preprocessing
            best_image = get_best_image_for_type(preprocessed_versions, data_type)
            
            # Convert to base64 for Claude API
            buffer = BytesIO()
            best_image.save(buffer, format="PNG")
            preprocessed_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Get Claude prompt
            ai_prompt = get_ai_prompt_for_type(data_type, region_name)
            
            # Process with the selected engine(s)
            try:
                if args.ocr_engine == 'claude':
                    # Process with Claude only
                    response = claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=100,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": ai_prompt
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": preprocessed_base64
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.0
                    )
                    
                    text = response.content[0].text.strip()
                    logger.info(f"Claude response for {region_name}: '{text}'")
                    
                    # Process response based on data_type
                    if data_type == 'checkbox':
                        text = '1' if text.lower() in ['1', 'checked', 'true', 'yes', 'x', '✓'] else ''
                    
                    confidence = 90
                    engine_results = {'claude': {'text': text, 'confidence': confidence}}
                                
                elif args.ocr_engine in ['paddle', 'paddle-only']:
                    # Process with PaddleOCR only
                    try:
                        text, confidence = paddle_processor.process_image(best_image, data_type)
                        logger.info(f"PaddleOCR response for {region_name}: '{text}' (confidence: {confidence:.2f}%)")
                        
                        # Process the response based on data_type
                        if data_type == 'checkbox':
                            text = '1' if text.lower() in ['1', 'checked', 'true', 'yes', 'x', '✓'] else ''
                        
                        engine_results = {'paddle': {'text': text, 'confidence': confidence}}
                        
                    except Exception as e:
                        logger.error(f"PaddleOCR error for {region_name}: {str(e)}")
                        text = ""
                        confidence = 0
                        engine_results = {'paddle': {'text': '', 'confidence': 0, 'error': str(e)}}
                
                else:  # 'both'
                    # Process with both engines and combine results
                    text, confidence, engine_results = process_region_with_both_engines(
                        best_image, data_type, region_name, ai_prompt, preprocessed_base64
                    )
                
                # Store result
                results[region_name] = {
                    'text': text,
                    'type': data_type,
                    'confidence': confidence,
                    'debug_paths': debug_paths,
                    'coordinates': coordinates,
                    'engine_results': engine_results
                }
                
            except Exception as e:
                logger.error(f"OCR error for {region_name}: {str(e)}")
                results[region_name] = {
                    'text': '',
                    'type': data_type,
                    'error': str(e),
                    'debug_paths': debug_paths,
                    'coordinates': coordinates
                }
        
        return jsonify({
            'results': results,
            'full_image_path': url_for('static', filename=f'debug/{full_image_filename}', _external=True),
            'rotation_applied': rotation_degrees
        })
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===== MAIN APPLICATION =====

if __name__ == '__main__':
    logger.info("Starting OCR application...")
    app.run(host='0.0.0.0', port=5000, debug=True)