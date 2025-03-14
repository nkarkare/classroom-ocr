/**
 * Advanced OCR Application
 * 
 * This JavaScript file handles all frontend functionality including:
 * - Image loading and display
 * - Annotation creation and management
 * - API communication with the backend
 * - Results display
 */

// Main application class
class OCRApp {
    constructor() {
        this.image = null;
        this.annotations = [];
        this.results = null;
        this.drawingAnnotation = false;
        this.currentAnnotation = null;
        this.startX = 0;
        this.startY = 0;
        this.colorMap = {
            'text': '#198754',
            'checkbox': '#0d6efd',
            'minimal_character': '#6f42c1',
            'qr': '#fd7e14'
        };
        
        // For batch processing
        this.batchFiles = [];
        this.batchResults = null;
        this.batchId = null;
        this.currentBatchIndex = 0;
        this.isBatchMode = false;
        
        // Initialize the application
        this.init();
    }
       
    init() {
        // Bind event listeners
        this.bindEventListeners();
        
        // Initialize Bootstrap modal
        this.annotationModal = new bootstrap.Modal(document.getElementById('annotationModal'));
        
        // Initialize batch processing data
        this.batchFiles = [];
        this.batchResults = null;
        this.batchId = null;
    }
    
    bindEventListeners() {
        // Mode selection
        document.getElementById('singleMode').addEventListener('change', this.toggleProcessingMode.bind(this));
        document.getElementById('batchMode').addEventListener('change', this.toggleProcessingMode.bind(this));
        
        // File uploads
        document.getElementById('imageInput').addEventListener('change', this.handleImageUpload.bind(this));
        document.getElementById('jsonInput').addEventListener('change', this.handleJsonUpload.bind(this));
        document.getElementById('folderInput').addEventListener('change', this.handleFolderSelection.bind(this));
        document.getElementById('batchJsonInput').addEventListener('change', this.handleJsonUpload.bind(this));
        document.getElementById('refreshFolderBtn').addEventListener('click', () => {
            document.getElementById('folderInput').click();
        });
        
        // Buttons
        document.getElementById('processButton').addEventListener('click', this.processDocument.bind(this));
        document.getElementById('processFolderBtn').addEventListener('click', this.processBatch.bind(this));
        document.getElementById('downloadButton').addEventListener('click', this.downloadResults.bind(this));
        document.getElementById('downloadLearningButton').addEventListener('click', 
            this.isBatchMode ? this.downloadBatchLearningData.bind(this) : this.downloadLearningData.bind(this));
        document.getElementById('createAnnotationsBtn').addEventListener('click', this.openAnnotationModal.bind(this));
        document.getElementById('addRegionBtn').addEventListener('click', this.addAnnotation.bind(this));
        document.getElementById('saveAnnotationsBtn').addEventListener('click', this.saveAnnotations.bind(this));
        
        // Batch annotation buttons
        document.getElementById('batchCreateAnnotationsBtn').addEventListener('click', this.openAnnotationModal.bind(this));
        document.getElementById('batchAddRegionBtn').addEventListener('click', this.addAnnotation.bind(this));
        document.getElementById('batchSaveAnnotationsBtn').addEventListener('click', this.saveAnnotations.bind(this));
        
        // Batch navigation buttons
        document.getElementById('prevImageBtn').addEventListener('click', () => this.navigateBatchImage('prev'));
        document.getElementById('nextImageBtn').addEventListener('click', () => this.navigateBatchImage('next'));
        
        // Modal buttons
        document.getElementById('modalAddRegionBtn').addEventListener('click', this.addModalAnnotation.bind(this));
        document.getElementById('saveModalAnnotationsBtn').addEventListener('click', this.saveModalAnnotations.bind(this));
        
        // Canvas drawing events for annotation modal
        const annotationCanvas = document.getElementById('annotationCanvas');
        annotationCanvas.addEventListener('mousedown', this.startDrawing.bind(this));
        annotationCanvas.addEventListener('mousemove', this.drawAnnotation.bind(this));
        annotationCanvas.addEventListener('mouseup', this.finishDrawing.bind(this));
        annotationCanvas.addEventListener('mouseleave', this.cancelDrawing.bind(this));
        
        // Toggle switches
        document.getElementById('showDebugImages').addEventListener('change', this.toggleDebugImages.bind(this));
        document.getElementById('showBoundingBoxes').addEventListener('change', this.toggleBoundingBoxes.bind(this));
    }
    
    // File handling
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            // Create an image and strip EXIF data by drawing to a canvas
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                
                // Draw image on canvas, which strips EXIF metadata
                ctx.drawImage(img, 0, 0);
                
                // Convert canvas to data URL (removes EXIF)
                this.image = new Image();
                this.image.onload = () => {
                    this.displayImagePreview();
                    this.showStatus('Image loaded successfully');
                };
                this.image.src = canvas.toDataURL('image/png');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    handleJsonUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                this.annotations = JSON.parse(e.target.result);
                this.showStatus(`Loaded ${this.annotations.length} annotations`);
                this.updateAnnotationsList();
                
                // If in batch mode, enable the process button if we have files
                if (this.isBatchMode && this.batchFiles && this.batchFiles.length > 0) {
                    document.getElementById('processFolderBtn').disabled = false;
                }
                
                // If image is already loaded in single mode, display with annotations
                if (!this.isBatchMode && this.image) {
                    this.displayImagePreview();
                }
            } catch (error) {
                this.showError('Invalid JSON file: ' + error.message);
            }
        };
        reader.readAsText(file);
    }
        
    // UI display functions
    displayImagePreview() {
        const previewImg = document.getElementById('imagePreview');
        previewImg.src = this.image.src;
        document.getElementById('imagePreviewContainer').style.display = 'block';
        
        // Also draw on main canvas if available
        this.drawImageWithAnnotations();
    }
    
    drawImageWithAnnotations() {
        const canvas = document.getElementById('outputCanvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas dimensions to match image
        canvas.width = this.image.width;
        canvas.height = this.image.height;
        
        // Draw image
        ctx.drawImage(this.image, 0, 0);
        
        // Draw annotations if they exist and checkbox is checked
        if (this.annotations.length > 0 && document.getElementById('showBoundingBoxes').checked) {
            this.drawAnnotationOverlay();
        }
    }
    
    drawAnnotationOverlay() {
        const overlay = document.getElementById('annotationOverlay');
        overlay.innerHTML = '';
        overlay.style.width = this.image.width + 'px';
        overlay.style.height = this.image.height + 'px';
        
        this.annotations.forEach((annotation, index) => {
            const { name, coordinates, type = 'text' } = annotation;
            const { x1, y1, x2, y2 } = coordinates;
            
            // Create annotation box
            const box = document.createElement('div');
            box.className = 'annotation-box';
            box.style.left = x1 + 'px';
            box.style.top = y1 + 'px';
            box.style.width = (x2 - x1) + 'px';
            box.style.height = (y2 - y1) + 'px';
            box.style.borderColor = this.colorMap[type] || '#198754';
            
            // Create annotation label
            const label = document.createElement('div');
            label.className = 'annotation-label';
            label.style.left = x1 + 'px';
            label.style.top = (y1 - 20) + 'px';
            label.style.backgroundColor = this.colorMap[type] || '#198754';
            
            // Set label text based on results if available
            if (this.results && this.results[name]) {
                const resultText = this.results[name].text;
                label.textContent = `${name}: ${resultText || '[empty]'}`;
            } else {
                label.textContent = `${name} (${type})`;
            }
            
            overlay.appendChild(box);
            overlay.appendChild(label);
        });
    }
    
    updateAnnotationsList() {
        const listContainer = document.getElementById('annotationsList');
        listContainer.innerHTML = '';
        
        this.annotations.forEach((annotation, index) => {
            const item = document.createElement('div');
            item.className = 'annotation-list-item';
            item.innerHTML = `
                <span>${annotation.name} (${annotation.type || 'text'})</span>
                <div class="controls">
                    <button class="btn btn-sm btn-outline-danger" data-index="${index}">Remove</button>
                </div>
            `;
            
            // Add event listener for remove button
            item.querySelector('button').addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                this.annotations.splice(index, 1);
                this.updateAnnotationsList();
            });
            
            listContainer.appendChild(item);
        });
    }
    
    updateModalAnnotationsList() {
        const listContainer = document.getElementById('modalAnnotationsList');
        listContainer.innerHTML = '';
        
        this.annotations.forEach((annotation, index) => {
            const item = document.createElement('div');
            item.className = 'annotation-list-item';
            item.innerHTML = `
                <span>${annotation.name} (${annotation.type || 'text'}) - [${annotation.coordinates.x1},${annotation.coordinates.y1},${annotation.coordinates.x2},${annotation.coordinates.y2}]</span>
                <div class="controls">
                    <button class="btn btn-sm btn-outline-danger" data-index="${index}">Remove</button>
                </div>
            `;
            
            // Add event listener for remove button
            item.querySelector('button').addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                this.annotations.splice(index, 1);
                this.updateModalAnnotationsList();
                this.drawAnnotationsOnModalCanvas();
            });
            
            listContainer.appendChild(item);
        });
    }
    
    // Annotation creation
    openAnnotationModal() {
        if (!this.image) {
            this.showError('Please upload an image first');
            return;
        }
        
        const canvas = document.getElementById('annotationCanvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match image (or container if image is larger)
        const container = document.querySelector('.annotation-canvas-container');
        const containerWidth = container.clientWidth;
        
        // Calculate scale to fit image in container
        const scale = containerWidth / this.image.width;
        canvas.width = containerWidth;
        canvas.height = this.image.height * scale;
        
        // Store scale for coordinate translation
        this.annotationScale = scale;
        
        // Draw image
        ctx.drawImage(this.image, 0, 0, canvas.width, canvas.height);
        
        // Draw existing annotations
        this.drawAnnotationsOnModalCanvas();
        
        // Show modal
        this.annotationModal.show();
    }
    
    drawAnnotationsOnModalCanvas() {
        const canvas = document.getElementById('annotationCanvas');
        const ctx = canvas.getContext('2d');
        
        // Clear and redraw image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(this.image, 0, 0, canvas.width, canvas.height);
        
        // Draw existing annotations
        this.annotations.forEach(annotation => {
            const { coordinates, type = 'text' } = annotation;
            const { x1, y1, x2, y2 } = coordinates;
            
            // Scale coordinates
            const scaledX1 = x1 * this.annotationScale;
            const scaledY1 = y1 * this.annotationScale;
            const scaledX2 = x2 * this.annotationScale;
            const scaledY2 = y2 * this.annotationScale;
            
            // Draw rectangle
            ctx.strokeStyle = this.colorMap[type] || '#198754';
            ctx.lineWidth = 2;
            ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
            
            // Draw label
            ctx.fillStyle = this.colorMap[type] || '#198754';
            ctx.fillRect(scaledX1, scaledY1 - 20, 80, 20);
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.fillText(annotation.name, scaledX1 + 5, scaledY1 - 5);
        });
    }
    
    startDrawing(event) {
        if (!this.image) return;
        
        this.drawingAnnotation = true;
        
        // Get canvas coordinates
        const canvas = document.getElementById('annotationCanvas');
        const rect = canvas.getBoundingClientRect();
        this.startX = event.clientX - rect.left;
        this.startY = event.clientY - rect.top;
        
        // Create new annotation
        this.currentAnnotation = {
            x1: this.startX,
            y1: this.startY,
            x2: this.startX,
            y2: this.startY
        };
    }
    
    drawAnnotation(event) {
        if (!this.drawingAnnotation || !this.currentAnnotation) return;
        
        const canvas = document.getElementById('annotationCanvas');
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        
        // Update end coordinates
        this.currentAnnotation.x2 = event.clientX - rect.left;
        this.currentAnnotation.y2 = event.clientY - rect.top;
        
        // Redraw canvas
        this.drawAnnotationsOnModalCanvas();
        
        // Draw current annotation
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            this.currentAnnotation.x1,
            this.currentAnnotation.y1,
            this.currentAnnotation.x2 - this.currentAnnotation.x1,
            this.currentAnnotation.y2 - this.currentAnnotation.y1
        );
    }
    
    finishDrawing() {
        if (!this.drawingAnnotation || !this.currentAnnotation) return;
        
        this.drawingAnnotation = false;
        
        // Ensure coordinates are ordered (x1 < x2, y1 < y2)
        const { x1, y1, x2, y2 } = this.currentAnnotation;
        this.currentAnnotation = {
            x1: Math.min(x1, x2),
            y1: Math.min(y1, y2),
            x2: Math.max(x1, x2),
            y2: Math.max(y1, y2)
        };
        
        // Convert coordinates back to original image scale
        document.getElementById('modalRegionName').focus();
    }
    
    cancelDrawing() {
        this.drawingAnnotation = false;
        this.currentAnnotation = null;
    }
    
    addModalAnnotation() {
        if (!this.currentAnnotation) {
            this.showError('Please draw a region first');
            return;
        }
        
        const name = document.getElementById('modalRegionName').value.trim();
        const type = document.getElementById('modalRegionType').value;
        
        if (!name) {
            this.showError('Please enter a region name');
            return;
        }
        
        // Convert coordinates back to original image scale
        const { x1, y1, x2, y2 } = this.currentAnnotation;
        
        const annotation = {
            name: name,
            type: type,
            coordinates: {
                x1: Math.round(x1 / this.annotationScale),
                y1: Math.round(y1 / this.annotationScale),
                x2: Math.round(x2 / this.annotationScale),
                y2: Math.round(y2 / this.annotationScale)
            }
        };
        
        // Add to annotations array
        this.annotations.push(annotation);
        
        // Update UI
        this.updateModalAnnotationsList();
        this.drawAnnotationsOnModalCanvas();
        
        // Reset
        this.currentAnnotation = null;
        document.getElementById('modalRegionName').value = '';
    }
    
    saveModalAnnotations() {
        this.annotationModal.hide();
        this.updateAnnotationsList();
        
        // Show annotation editor
        document.getElementById('annotationEditor').style.display = 'block';
        
        // Draw on main canvas
        if (this.image) {
            this.drawImageWithAnnotations();
        }
        
        this.showStatus(`Saved ${this.annotations.length} annotations`);
    }
    
    addAnnotation() {
        const name = document.getElementById('regionName').value.trim();
        const type = document.getElementById('regionType').value;
        
        if (!name) {
            this.showError('Please enter a region name');
            return;
        }
        
        const annotation = {
            name: name,
            type: type,
            coordinates: {
                x1: 0,
                y1: 0,
                x2: 100,
                y2: 100
            }
        };
        
        // Add to annotations array
        this.annotations.push(annotation);
        
        // Update UI
        this.updateAnnotationsList();
        document.getElementById('regionName').value = '';
    }
    
    saveAnnotations() {
        // Create JSON file and trigger download
        const json = JSON.stringify(this.annotations, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'annotations.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // API communication
    // Add these methods to the OCRApp class

    // Modify the processDocument method to include OCR engine selection
    async processDocument() {
        if (!this.image || this.annotations.length === 0) {
            this.showError('Please upload both an image and annotations');
            return;
        }
        
        // Show loading indicator
        this.showLoading('Processing document...');
        
        try {
            // Get selected OCR engine
            const engineElement = document.querySelector('input[name="ocrEngine"]:checked');
            const ocrEngine = engineElement ? engineElement.value : 'claude';
            
            // Get PaddleOCR settings if applicable
            let paddleSettings = {};
            if (ocrEngine !== 'claude') {
                const langElement = document.getElementById('paddleLang');
                const useGpuElement = document.getElementById('useGpu');
                paddleSettings = {
                    lang: langElement ? langElement.value : 'en',
                    useGpu: useGpuElement ? useGpuElement.checked : false
                };
            }
            
            // Prepare data for API
            const data = {
                image: this.image.src,
                annotations: this.annotations,
                ocrEngine: ocrEngine,
                paddleSettings: paddleSettings
            };
            
            // Send to API
            const response = await fetch('/api/process_full_document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Store results
            this.results = result.results;
            
            // Update UI
            this.hideLoading();
            this.drawImageWithAnnotations();
            this.displayResults();
            
            // Enable download button
            document.getElementById('downloadButton').disabled = false;
            
            this.showStatus('Processing complete! Review the results below.');
        } catch (error) {
            this.hideLoading();
            this.showError('Processing failed: ' + error.message);
        }
    }

    // Modify the displayResultsTable method to show engine-specific results
    displayResultsTable() {
        const tableBody = document.getElementById('resultsTableBody');
        tableBody.innerHTML = '';
        
        // Create a row for each result
        for (const [name, result] of Object.entries(this.results)) {
            const row = document.createElement('tr');
            
            // Region name
            const nameCell = document.createElement('td');
            nameCell.textContent = name;
            
            // Type
            const typeCell = document.createElement('td');
            typeCell.textContent = result.type;
            
            // Recognized text
            const textCell = document.createElement('td');
            textCell.className = 'text-result';
            textCell.textContent = result.text || '(empty)';
            
            // Confidence
            const confidenceCell = document.createElement('td');
            confidenceCell.textContent = `${Math.round(result.confidence)}%`;
            
            // Engine details
            const engineCell = document.createElement('td');
            if (result.engine_results) {
                let engineHtml = '<small class="text-muted">';
                
                if (result.engine_results.claude) {
                    engineHtml += `Claude: "${result.engine_results.claude.text}" (${Math.round(result.engine_results.claude.confidence)}%)<br>`;
                }
                
                if (result.engine_results.paddle) {
                    engineHtml += `PaddleOCR: "${result.engine_results.paddle.text}" (${Math.round(result.engine_results.paddle.confidence)}%)`;
                }
                
                engineHtml += '</small>';
                engineCell.innerHTML = engineHtml;
            } else {
                engineCell.textContent = 'N/A';
            }
            
            row.appendChild(nameCell);
            row.appendChild(typeCell);
            row.appendChild(textCell);
            row.appendChild(confidenceCell);
            row.appendChild(engineCell);
            
            tableBody.appendChild(row);
        }
        
        // Show results table section
        document.getElementById('resultsTableSection').style.display = 'block';
    }    

    // Results display
    displayResults() {
        // Display debug images if enabled
        if (document.getElementById('showDebugImages').checked) {
            this.displayDebugImages();
        }
        
        // Display results table
        this.displayResultsTable();
    }
    
    updateResultsTableRow(regionName) {
        const tableBody = document.getElementById('resultsTableBody');
        if (!tableBody) return;
        
        // Find the row for this region
        const rows = tableBody.querySelectorAll('tr');
        for (const row of rows) {
            const nameCell = row.cells[0];
            if (nameCell && nameCell.textContent === regionName) {
                // Update text cell
                const textCell = row.cells[2];
                if (textCell) {
                    textCell.textContent = this.results[regionName].text || '(empty)';
                    
                    // Add "edited" indicator if manually edited
                    if (this.results[regionName].manuallyEdited) {
                        if (!textCell.querySelector('.edited-indicator')) {
                            const indicator = document.createElement('span');
                            indicator.className = 'badge bg-warning ms-2 edited-indicator';
                            indicator.textContent = 'Edited';
                            textCell.appendChild(indicator);
                        }
                    }
                }
                break;
            }
        }
    }

    saveTextEdit(event) {
        const input = event.target;
        const regionName = input.dataset.regionName;
        const newText = input.value;
        
        // Update results in memory
        if (this.results && this.results[regionName]) {
            // Store the original text for learning data if it's the first edit
            if (!this.results[regionName].originalText) {
                this.results[regionName].originalText = this.results[regionName].text;
            }
            
            // Update the text
            this.results[regionName].text = newText;
            
            // Mark as manually edited
            this.results[regionName].manuallyEdited = true;
            
            // Update the results table
            this.updateResultsTableRow(regionName);
            
            // Update the annotation overlay to show the new text
            this.drawImageWithAnnotations();
            
            // Show status message
            this.showStatus(`Updated text for region "${regionName}"`);
        }
    }

    displayDebugImages() {
        const container = document.getElementById('debugImagesContainer');
        container.innerHTML = '';
        
        // Create a row for each region
        for (const [name, result] of Object.entries(this.results)) {
            if (!result.debug_paths) continue;
            
            // Create a card for each debug image version
            const row = document.createElement('div');
            row.className = 'col-md-4 mb-3';
            
            const card = document.createElement('div');
            card.className = 'card debug-image-card';
            
            // Card header with region name
            const header = document.createElement('div');
            header.className = 'card-header';
            header.textContent = `Region: ${name} (${result.type})`;
            
            // Card body with images
            const body = document.createElement('div');
            body.className = 'card-body';
            
            // Add each version as a tab
            const versions = Object.entries(result.debug_paths).slice(0, 3); // Limit to 3 versions
            
            versions.forEach(([version, url]) => {
                const imgWrapper = document.createElement('div');
                imgWrapper.className = 'debug-image-wrapper mb-2';
                
                const img = document.createElement('img');
                img.src = url;
                img.className = 'debug-image';
                img.alt = `${name} (${version})`;
                
                const caption = document.createElement('div');
                caption.className = 'text-center small mt-1';
                caption.textContent = version;
                
                imgWrapper.appendChild(img);
                imgWrapper.appendChild(caption);
                body.appendChild(imgWrapper);
            });
            
            // Add editable recognized text
            const textDiv = document.createElement('div');
            textDiv.className = 'mt-2 border-top pt-2';
            
            const textLabel = document.createElement('strong');
            textLabel.textContent = 'Recognized: ';
            textDiv.appendChild(textLabel);
            
            // Create editable input for the recognized text
            const textInput = document.createElement('input');
            textInput.type = 'text';
            textInput.className = 'form-control form-control-sm recognized-text-input';
            textInput.value = result.text || '';
            textInput.dataset.regionName = name;
            
            // Add event listeners for saving changes
            textInput.addEventListener('blur', (e) => this.saveTextEdit(e));
            textInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.target.blur();
                }
            });
            
            textDiv.appendChild(textInput);
            
            // Add confidence display
            const confidenceDiv = document.createElement('div');
            confidenceDiv.className = 'small text-muted mt-1';
            confidenceDiv.textContent = `Confidence: ${Math.round(result.confidence)}%`;
            textDiv.appendChild(confidenceDiv);
            
            body.appendChild(textDiv);
            card.appendChild(header);
            card.appendChild(body);
            row.appendChild(card);
            container.appendChild(row);
        }
        
        // Show debug images section
        document.getElementById('debugImagesSection').style.display = 'block';
    }
    
    // Toggle functions
    toggleDebugImages(event) {
        const show = event.target.checked;
        document.getElementById('debugImagesSection').style.display = show ? 'block' : 'none';
    }
    
    toggleBoundingBoxes(event) {
        const show = event.target.checked;
        if (this.image) {
            this.drawImageWithAnnotations();
        }
    }
    
    // Utility functions
    showError(message) {
        const errorContainer = document.getElementById('errorContainer');
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
        
        // Hide status
        document.getElementById('statusContainer').style.display = 'none';
        
        // Automatically hide after 5 seconds
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 5000);
    }
    
    showStatus(message) {
        const statusContainer = document.getElementById('statusContainer');
        statusContainer.textContent = message;
        statusContainer.style.display = 'block';
        
        // Hide error
        document.getElementById('errorContainer').style.display = 'none';
    }
    
    showLoading(message) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        document.getElementById('loadingMessage').textContent = message;
        loadingIndicator.style.display = 'flex';
        
        // Reset progress
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('progressText').textContent = '0%';
    }
    
    hideLoading() {
        document.getElementById('loadingIndicator').style.display = 'none';
    }
    
    updateProgress(message, progress) {
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('progressBar').style.width = `${progress}%`;
        document.getElementById('progressText').textContent = `${Math.round(progress)}%`;
    }
    
    downloadResults() {
        if (!this.results) return;
        
        // Combine annotations with results
        const downloadData = this.annotations.map(annotation => {
            const result = this.results[annotation.name] || {
                text: '',
                type: annotation.type,
                confidence: 0
            };
            
            return {
                ...annotation,
                recognized: {
                    text: result.text,
                    confidence: result.confidence
                }
            };
        });
        
        // Create JSON file and trigger download
        const json = JSON.stringify(downloadData, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ocr_results.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    downloadLearningData() {
        if (!this.results) return;
        
        // Create learning dataset from edited results
        const learningData = [];
        
        for (const [name, result] of Object.entries(this.results)) {
            if (result.manuallyEdited && result.originalText !== undefined) {
                learningData.push({
                    region_name: name,
                    data_type: result.type,
                    original_text: result.originalText,
                    corrected_text: result.text,
                    confidence: result.confidence,
                    engine_results: result.engine_results
                });
            }
        }
        
        if (learningData.length === 0) {
            this.showError('No edited data found. Edit some recognized text first.');
            return;
        }
        
        // Create JSON file and trigger download
        const json = JSON.stringify(learningData, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'paddle_learning_data.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showStatus(`Downloaded learning dataset with ${learningData.length} entries`);
    }

    handleFolderSelection(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        
        // Filter for image files
        this.batchFiles = Array.from(files).filter(file => {
            const ext = file.name.toLowerCase().split('.').pop();
            return ['jpg', 'jpeg', 'png'].includes(ext);
        });
        
        if (this.batchFiles.length === 0) {
            this.showError('No image files found in the selected folder');
            document.getElementById('processFolderBtn').disabled = true;
            document.getElementById('folderStats').style.display = 'none';
            return;
        }
        
        // Enable the process button if we have both files and annotations
        const processFolderBtn = document.getElementById('processFolderBtn');
        processFolderBtn.disabled = this.annotations.length === 0;
        
        // Update folder stats
        const imageCountBadge = document.getElementById('imageCountBadge');
        imageCountBadge.textContent = `${this.batchFiles.length} images`;
        document.getElementById('folderStats').style.display = 'block';
        
        this.showStatus(`Found ${this.batchFiles.length} image files ready for batch processing. ${
            this.annotations.length ? 'Ready to process.' : 'Please upload or create annotations.'
        }`);
        
        // Show a preview of the first image
        if (this.batchFiles.length > 0) {
            const reader = new FileReader();
            reader.onload = (e) => {
                // Create an image preview (optional)
                const img = new Image();
                img.onload = () => {
                    const canvas = document.getElementById('outputCanvas');
                    const ctx = canvas.getContext('2d');
                    
                    // Set canvas dimensions to match image
                    canvas.width = img.width;
                    canvas.height = img.height;
                    
                    // Draw image
                    ctx.drawImage(img, 0, 0);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(this.batchFiles[0]);
        }
    }

    async processBatch() {
        if (!this.batchFiles || this.batchFiles.length === 0) {
            this.showError('Please select a folder with images');
            return;
        }
        
        if (this.annotations.length === 0) {
            this.showError('Please upload or create annotations first');
            return;
        }
        
        // Show loading indicator
        this.showLoading(`Processing batch of ${this.batchFiles.length} images...`);
        
        try {
            // Read all image files and convert to base64
            const imagesData = [];
            
            for (let i = 0; i < this.batchFiles.length; i++) {
                const file = this.batchFiles[i];
                
                // Update progress
                this.updateProgress(`Reading file ${i+1}/${this.batchFiles.length}: ${file.name}`, 
                                   (i / this.batchFiles.length) * 50);
                
                // Read file as base64
                const base64 = await this.readFileAsBase64(file);
                
                imagesData.push({
                    image_info: {
                        name: file.name,
                        size: file.size
                    },
                    data: base64
                });
            }
            
            // Get alignment option
            const useMarkerAlignment = document.getElementById('useMarkerAlignment').checked;
            
            // Get selected OCR engine
            const engineElement = document.querySelector('input[name="ocrEngine"]:checked');
            const ocrEngine = engineElement ? engineElement.value : 'claude';
            
            // Get PaddleOCR settings if applicable
            let paddleSettings = {};
            if (ocrEngine !== 'claude') {
                const langElement = document.getElementById('paddleLang');
                const useGpuElement = document.getElementById('useGpu');
                paddleSettings = {
                    lang: langElement ? langElement.value : 'en',
                    use_gpu: useGpuElement ? useGpuElement.checked : false
                };
            }
            
            // Prepare data for API
            const data = {
                images: imagesData,
                annotations: this.annotations,
                align_markers: useMarkerAlignment,
                ocr_engine: ocrEngine,
                paddle_settings: paddleSettings
            };
            
            // Update progress
            this.updateProgress(`Sending batch to server for processing...`, 50);
            
            // Send to API
            const response = await fetch('/api/process_batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            // Update progress
            this.updateProgress(`Processing complete, receiving results...`, 90);
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Store batch results
            this.batchResults = result.results;
            this.batchId = result.batch_id;
            
            // Update UI
            this.hideLoading();
            this.displayBatchResults();
            
            this.showStatus(`Batch processing complete! Processed ${this.batchFiles.length} images.`);
        } catch (error) {
            this.hideLoading();
            this.showError('Batch processing failed: ' + error.message);
        }
    }

    readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    displayBatchResults() {
        // Create tabs for each image
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '';
        
        // Create tabs navigation
        const tabsNav = document.createElement('ul');
        tabsNav.className = 'nav nav-tabs';
        
        // Create tabs content
        const tabsContent = document.createElement('div');
        tabsContent.className = 'tab-content';
        
        // Add tabs for each image
        let isFirst = true;
        for (const [imageName, imageResults] of Object.entries(this.batchResults)) {
            // Tab nav
            const tabNav = document.createElement('li');
            tabNav.className = 'nav-item';
            
            const tabLink = document.createElement('a');
            tabLink.className = `nav-link ${isFirst ? 'active' : ''}`;
            tabLink.href = `#tab-${imageName.replace(/\./g, '-')}`;
            tabLink.dataset.bsToggle = 'tab';
            tabLink.textContent = imageName;
            
            tabNav.appendChild(tabLink);
            tabsNav.appendChild(tabNav);
            
            // Tab content
            const tabContent = document.createElement('div');
            tabContent.className = `tab-pane fade ${isFirst ? 'show active' : ''}`;
            tabContent.id = `tab-${imageName.replace(/\./g, '-')}`;
            
            // Image display
            const imgContainer = document.createElement('div');
            imgContainer.className = 'text-center mb-3';
            
            const img = document.createElement('img');
            img.src = imageResults.image_path;
            img.className = 'img-fluid border';
            img.style.maxHeight = '500px';
            
            imgContainer.appendChild(img);
            tabContent.appendChild(imgContainer);
            
            // Results table
            const table = document.createElement('table');
            table.className = 'table table-hover table-striped';
            
            const tableHead = document.createElement('thead');
            tableHead.innerHTML = `
                <tr>
                    <th>Region</th>
                    <th>Type</th>
                    <th>Recognized Text</th>
                    <th>Confidence</th>
                    <th>Actions</th>
                </tr>
            `;
            
            const tableBody = document.createElement('tbody');
            
            // Add rows for each region
            for (const [regionName, result] of Object.entries(imageResults.results)) {
                const row = document.createElement('tr');
                
                // Region name
                const nameCell = document.createElement('td');
                nameCell.textContent = regionName;
                
                // Type
                const typeCell = document.createElement('td');
                typeCell.textContent = result.type;
                
                // Recognized text
                const textCell = document.createElement('td');
                textCell.className = 'text-result';
                
                // Create editable input
                const textInput = document.createElement('input');
                textInput.type = 'text';
                textInput.className = 'form-control form-control-sm';
                textInput.value = result.text || '';
                textInput.dataset.imageName = imageName;
                textInput.dataset.regionName = regionName;
                
                // Add event listener for saving changes
                textInput.addEventListener('blur', (e) => this.saveBatchTextEdit(e));
                textInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.target.blur();
                    }
                });
                
                textCell.appendChild(textInput);
                
                // Confidence
                const confidenceCell = document.createElement('td');
                confidenceCell.textContent = `${Math.round(result.confidence)}%`;
                
                // Actions
                const actionsCell = document.createElement('td');
                
                // View debug images button
                const viewDebugBtn = document.createElement('button');
                viewDebugBtn.className = 'btn btn-sm btn-outline-secondary me-2';
                viewDebugBtn.textContent = 'Debug Images';
                viewDebugBtn.addEventListener('click', () => this.showBatchDebugImages(imageName, regionName));
                actionsCell.appendChild(viewDebugBtn);
                
                row.appendChild(nameCell);
                row.appendChild(typeCell);
                row.appendChild(textCell);
                row.appendChild(confidenceCell);
                row.appendChild(actionsCell);
                tableBody.appendChild(row);
            }
            
            table.appendChild(tableHead);
            table.appendChild(tableBody);
            tabContent.appendChild(table);
            tabsContent.appendChild(tabContent);
            
            isFirst = false;
        }
        
        // Add tabs to container
        container.appendChild(tabsNav);
        container.appendChild(tabsContent);
        
        // Initialize bootstrap tabs
        const tabElms = document.querySelectorAll('[data-bs-toggle="tab"]');
        tabElms.forEach(tabEl => {
            new bootstrap.Tab(tabEl);
        });
    }

    saveBatchTextEdit(event) {
        const input = event.target;
        const imageName = input.dataset.imageName;
        const regionName = input.dataset.regionName;
        const newText = input.value;
        
        // Update results in memory
        if (this.batchResults && this.batchResults[imageName] && this.batchResults[imageName].results[regionName]) {
            const result = this.batchResults[imageName].results[regionName];
            
            // Store the original text for learning data if it's the first edit
            if (!result.originalText) {
                result.originalText = result.text;
            }
            
            // Update the text
            result.text = newText;
            
            // Mark as manually edited
            result.manuallyEdited = true;
            
            // Show status message
            this.showStatus(`Updated text for ${imageName}, region "${regionName}"`);
        }
    }

    showBatchDebugImages(imageName, regionName) {
        if (!this.batchResults || !this.batchResults[imageName] || !this.batchResults[imageName].results[regionName]) {
            return;
        }
        
        const result = this.batchResults[imageName].results[regionName];
        
        // Create modal for debug images
        const modalId = 'debugImagesModal';
        let modal = document.getElementById(modalId);
        
        if (!modal) {
            // Create modal if it doesn't exist
            modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.id = modalId;
            modal.tabIndex = -1;
            modal.setAttribute('aria-hidden', 'true');
            
            modal.innerHTML = `
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Debug Images</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="debugImagesModalContent" class="row"></div>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            this.debugImagesModal = new bootstrap.Modal(modal);
        }
        
        // Populate modal with debug images
        const container = document.getElementById('debugImagesModalContent');
        container.innerHTML = '';
        
        // Add title with image and region info
        const titleDiv = document.createElement('div');
        titleDiv.className = 'col-12 mb-3';
        titleDiv.innerHTML = `<h5>${imageName} - ${regionName} (${result.type})</h5>`;
        container.appendChild(titleDiv);
        
        // Add each debug image version
        if (result.debug_paths) {
            for (const [version, url] of Object.entries(result.debug_paths)) {
                const col = document.createElement('div');
                col.className = 'col-md-4 mb-3';
                
                const card = document.createElement('div');
                card.className = 'card h-100';
                
                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';
                
                const img = document.createElement('img');
                img.src = url;
                img.className = 'img-fluid';
                img.alt = version;
                
                const caption = document.createElement('p');
                caption.className = 'text-center mt-2 mb-0';
                caption.textContent = version;
                
                cardBody.appendChild(img);
                cardBody.appendChild(caption);
                card.appendChild(cardBody);
                col.appendChild(card);
                container.appendChild(col);
            }
        } else {
            const noImagesDiv = document.createElement('div');
            noImagesDiv.className = 'col-12';
            noImagesDiv.textContent = 'No debug images available for this region.';
            container.appendChild(noImagesDiv);
        }
        
        // Show modal
        this.debugImagesModal.show();
    }

    downloadBatchLearningData() {
        if (!this.batchResults) return;
        
        // Create learning dataset from all edited results
        const learningData = [];
        
        for (const [imageName, imageResults] of Object.entries(this.batchResults)) {
            for (const [regionName, result] of Object.entries(imageResults.results)) {
                if (result.manuallyEdited && result.originalText !== undefined) {
                    learningData.push({
                        image_name: imageName,
                        region_name: regionName,
                        data_type: result.type,
                        original_text: result.originalText,
                        corrected_text: result.text,
                        confidence: result.confidence,
                        engine_results: result.engine_results
                    });
                }
            }
        }
        
        if (learningData.length === 0) {
            this.showError('No edited data found. Edit some recognized text first.');
            return;
        }
        
        // Create JSON file and trigger download
        const json = JSON.stringify(learningData, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `paddle_learning_data_batch_${this.batchId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showStatus(`Downloaded learning dataset with ${learningData.length} entries`);
    }

    downloadBatchResults() {
        if (!this.batchResults) return;
        
        // Prepare results for download
        const downloadData = {};
        
        for (const [imageName, imageResults] of Object.entries(this.batchResults)) {
            downloadData[imageName] = {};
            
            for (const [regionName, result] of Object.entries(imageResults.results)) {
                downloadData[imageName][regionName] = {
                    text: result.text || '',
                    type: result.type,
                    confidence: result.confidence,
                    manually_edited: result.manuallyEdited || false
                };
            }
        }
        
        // Create JSON file and trigger download
        const json = JSON.stringify(downloadData, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `ocr_batch_results_${this.batchId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }



    // Add these methods to OCRApp class for batch navigation
    toggleProcessingMode() {
        const singleModeRadio = document.getElementById('singleMode');
        this.isBatchMode = !singleModeRadio.checked;
        
        // Clear any existing results when switching modes
        this.results = null;
        this.batchResults = null;
        this.batchFiles = [];
        this.currentBatchIndex = 0;
        
        // Hide results sections
        document.getElementById('debugImagesSection').style.display = 'none';
        document.getElementById('resultsTableSection').style.display = 'none';
        document.getElementById('batchNavigation').classList.add('d-none');
        
        // Disable download buttons
        document.getElementById('downloadButton').disabled = true;
        document.getElementById('downloadLearningButton').disabled = true;
    }

    navigateBatchImage(direction) {
        if (!this.batchResults || Object.keys(this.batchResults).length === 0) return;
        
        const imageNames = Object.keys(this.batchResults);
        
        if (direction === 'next') {
            this.currentBatchIndex = (this.currentBatchIndex + 1) % imageNames.length;
        } else {
            this.currentBatchIndex = (this.currentBatchIndex - 1 + imageNames.length) % imageNames.length;
        }
        
        this.displayCurrentBatchImage();
    }

    displayCurrentBatchImage() {
        if (!this.batchResults || Object.keys(this.batchResults).length === 0) return;
        
        const imageNames = Object.keys(this.batchResults);
        const currentImageName = imageNames[this.currentBatchIndex];
        const imageData = this.batchResults[currentImageName];
        
        // Update batch counter
        const batchCounter = document.getElementById('batchCounter');
        batchCounter.textContent = `Image ${this.currentBatchIndex + 1} of ${imageNames.length}: ${currentImageName}`;
        
        // Load the image
        const img = new Image();
        img.onload = () => {
            // Display image
            const canvas = document.getElementById('outputCanvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // Draw annotations
            this.drawBatchAnnotations(currentImageName);
            
            // Update results table
            this.displayBatchResultsTable(currentImageName);
            
            // Update debug images
            this.displayBatchDebugImages(currentImageName);
        };
        img.src = imageData.image_path;
    }

    drawBatchAnnotations(imageName) {
        if (!this.batchResults || !this.batchResults[imageName]) return;
        
        const imageData = this.batchResults[imageName];
        const overlay = document.getElementById('annotationOverlay');
        overlay.innerHTML = '';
        
        const canvas = document.getElementById('outputCanvas');
        overlay.style.width = canvas.width + 'px';
        overlay.style.height = canvas.height + 'px';
        
        // Draw each region
        for (const [regionName, result] of Object.entries(imageData.results)) {
            const coordinates = result.coordinates;
            const type = result.type || 'text';
            
            // Create annotation box
            const box = document.createElement('div');
            box.className = 'annotation-box';
            box.style.left = coordinates.x1 + 'px';
            box.style.top = coordinates.y1 + 'px';
            box.style.width = (coordinates.x2 - coordinates.x1) + 'px';
            box.style.height = (coordinates.y2 - coordinates.y1) + 'px';
            box.style.borderColor = this.colorMap[type] || '#198754';
            
            // Create annotation label
            const label = document.createElement('div');
            label.className = 'annotation-label';
            label.style.left = coordinates.x1 + 'px';
            label.style.top = (coordinates.y1 - 20) + 'px';
            label.style.backgroundColor = this.colorMap[type] || '#198754';
            label.textContent = `${regionName}: ${result.text || '[empty]'}`;
            
            overlay.appendChild(box);
            overlay.appendChild(label);
        }
    }

    displayBatchResultsTable(imageName) {
        if (!this.batchResults || !this.batchResults[imageName]) return;
        
        const imageData = this.batchResults[imageName];
        const tableBody = document.getElementById('resultsTableBody');
        tableBody.innerHTML = '';
        
        // Create a row for each region
        for (const [regionName, result] of Object.entries(imageData.results)) {
            const row = document.createElement('tr');
            
            // Region name
            const nameCell = document.createElement('td');
            nameCell.textContent = regionName;
            
            // Type
            const typeCell = document.createElement('td');
            typeCell.textContent = result.type;
            
            // Recognized text (editable)
            const textCell = document.createElement('td');
            textCell.className = 'text-result';
            
            const textInput = document.createElement('input');
            textInput.type = 'text';
            textInput.className = 'form-control form-control-sm';
            textInput.value = result.text || '';
            textInput.dataset.imageName = imageName;
            textInput.dataset.regionName = regionName;
            
            // Add event listeners for saving changes
            textInput.addEventListener('blur', (e) => this.saveBatchTextEdit(e));
            textInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.target.blur();
                }
            });
            
            textCell.appendChild(textInput);
            
            // If manually edited, add indicator
            if (result.manuallyEdited) {
                const indicator = document.createElement('span');
                indicator.className = 'badge bg-warning ms-2';
                indicator.textContent = 'Edited';
                textCell.appendChild(indicator);
            }
            
            // Confidence
            const confidenceCell = document.createElement('td');
            confidenceCell.textContent = `${Math.round(result.confidence)}%`;
            
            // Engine details
            const engineCell = document.createElement('td');
            if (result.engine_results) {
                let engineHtml = '<small class="text-muted">';
                
                if (result.engine_results.claude) {
                    engineHtml += `Claude: "${result.engine_results.claude.text}" (${Math.round(result.engine_results.claude.confidence)}%)<br>`;
                }
                
                if (result.engine_results.paddle) {
                    engineHtml += `PaddleOCR: "${result.engine_results.paddle.text}" (${Math.round(result.engine_results.paddle.confidence)}%)`;
                }
                
                engineHtml += '</small>';
                engineCell.innerHTML = engineHtml;
            } else {
                engineCell.textContent = 'N/A';
            }
            
            row.appendChild(nameCell);
            row.appendChild(typeCell);
            row.appendChild(textCell);
            row.appendChild(confidenceCell);
            row.appendChild(engineCell);
            
            tableBody.appendChild(row);
        }
        
        // Show results table section
        document.getElementById('resultsTableSection').style.display = 'block';
    }

    displayBatchResults() {
        if (!this.batchResults || Object.keys(this.batchResults).length === 0) {
            this.showError('No batch results to display');
            return;
        }
        
        // Show batch navigation controls
        document.getElementById('batchNavigation').classList.remove('d-none');
        
        // Display first image by default
        this.currentBatchIndex = 0;
        this.displayCurrentBatchImage();
        
        // Show debug images section if enabled
        if (document.getElementById('showDebugImages').checked) {
            document.getElementById('debugImagesSection').style.display = 'block';
        }
        
        // Enable download buttons
        document.getElementById('downloadButton').disabled = false;
        document.getElementById('downloadLearningButton').disabled = false;
        
        this.showStatus(`Displaying batch results with ${Object.keys(this.batchResults).length} images.`);
    }

    displayBatchDebugImages(imageName) {
        if (!this.batchResults || !this.batchResults[imageName]) return;
        
        const imageData = this.batchResults[imageName];
        const container = document.getElementById('debugImagesContainer');
        container.innerHTML = '';
        
        // Create a row for each region
        for (const [regionName, result] of Object.entries(imageData.results)) {
            if (!result.debug_paths) continue;
            
            // Create a card for each debug image version
            const row = document.createElement('div');
            row.className = 'col-md-4 mb-3';
            
            const card = document.createElement('div');
            card.className = 'card debug-image-card';
            
            // Card header with region name
            const header = document.createElement('div');
            header.className = 'card-header';
            header.textContent = `Region: ${regionName} (${result.type})`;
            
            // Card body with images
            const body = document.createElement('div');
            body.className = 'card-body';
            
            // Add each version as a tab
            const versions = Object.entries(result.debug_paths).slice(0, 3); // Limit to 3 versions
            
            versions.forEach(([version, url]) => {
                const imgWrapper = document.createElement('div');
                imgWrapper.className = 'debug-image-wrapper mb-2';
                
                const img = document.createElement('img');
                img.src = url;
                img.className = 'debug-image';
                img.alt = `${regionName} (${version})`;
                
                const caption = document.createElement('div');
                caption.className = 'text-center small mt-1';
                caption.textContent = version;
                
                imgWrapper.appendChild(img);
                imgWrapper.appendChild(caption);
                body.appendChild(imgWrapper);
            });
            
            // Add editable recognized text
            const textDiv = document.createElement('div');
            textDiv.className = 'mt-2 border-top pt-2';
            
            const textLabel = document.createElement('strong');
            textLabel.textContent = 'Recognized: ';
            textDiv.appendChild(textLabel);
            
            // Create editable input for the recognized text
            const textInput = document.createElement('input');
            textInput.type = 'text';
            textInput.className = 'form-control form-control-sm recognized-text-input';
            textInput.value = result.text || '';
            textInput.dataset.imageName = imageName;
            textInput.dataset.regionName = regionName;
            
            // Add event listeners for saving changes
            textInput.addEventListener('blur', (e) => this.saveBatchTextEdit(e));
            textInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.target.blur();
                }
            });
            
            textDiv.appendChild(textInput);
            
            // Add confidence display
            const confidenceDiv = document.createElement('div');
            confidenceDiv.className = 'small text-muted mt-1';
            confidenceDiv.textContent = `Confidence: ${Math.round(result.confidence)}%`;
            textDiv.appendChild(confidenceDiv);
            
            body.appendChild(textDiv);
            card.appendChild(header);
            card.appendChild(body);
            row.appendChild(card);
            container.appendChild(row);
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OCRApp();
});