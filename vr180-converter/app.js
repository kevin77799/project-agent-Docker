// VR 180 Converter Application Logic
class VR180Converter {
    constructor() {
        this.currentFile = null;
        this.isProcessing = false;
        this.processInterval = null;
        this.currentStep = 0;
        this.backendUrl = 'http://127.0.0.1:8000'; // Backend API URL
        
        this.processingSteps = [
            { name: 'Analyzing', duration: 2000, description: 'Analyzing video properties and preparing for processing' },
            { name: 'Depth Estimation', duration: 8000, description: 'Generating depth maps using AI models' },
            { name: 'Stereoscopic Rendering', duration: 6000, description: 'Creating left and right eye views' },
            { name: 'Encoding', duration: 4000, description: 'Encoding to VR 180 format with proper metadata' },
            { name: 'Complete', duration: 500, description: 'Processing complete, file ready for download' }
        ];
        
        this.supportedFormats = ['mp4', 'avi', 'mov'];
        this.maxFileSize = 100 * 1024 * 1024; // 100MB in bytes
        
        this.initializeEventListeners();
        this.initializeSliders();
    }
    
    initializeEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        uploadArea.addEventListener('click', () => fileInput.click());
        
        document.getElementById('startProcessing').addEventListener('click', () => this.startRealProcessing());
        document.getElementById('cancelProcessing').addEventListener('click', () => this.cancelProcessing());
        
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadFile());
        document.getElementById('previewBtn').addEventListener('click', () => this.previewFile());
        
        document.getElementById('depthModel').addEventListener('change', () => this.updateProcessingTime());
        document.getElementById('qualitySettings').addEventListener('change', () => this.updateOutputInfo());
    }
    
    initializeSliders() {
        const depthSlider = document.getElementById('depthIntensity');
        const eyeSlider = document.getElementById('eyeSeparation');
        
        depthSlider.addEventListener('input', (e) => {
            document.getElementById('depthValue').textContent = e.target.value;
        });
        
        eyeSlider.addEventListener('input', (e) => {
            document.getElementById('eyeValue').textContent = e.target.value;
        });
    }
    
    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.validateAndProcessFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.validateAndProcessFile(file);
        }
    }
    
    validateAndProcessFile(file) {
        const errorElement = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        errorElement.style.display = 'none';
        
        const fileExtension = file.name.split('.').pop().toLowerCase();
        if (!this.supportedFormats.includes(fileExtension)) {
            this.showError(`Unsupported file format. Please use ${this.supportedFormats.join(', ').toUpperCase()} files.`);
            return;
        }
        
        if (file.size > this.maxFileSize) {
            this.showError(`File size too large. Maximum size is ${this.maxFileSize / (1024 * 1024)}MB.`);
            return;
        }
        
        this.currentFile = file;
        this.showFilePreview(file);
        this.enableProcessingSection();
    }
    
    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        errorText.textContent = message;
        errorElement.style.display = 'flex';
        setTimeout(() => { errorElement.style.display = 'none'; }, 5000);
    }
    
    showFilePreview(file) {
        const preview = document.getElementById('filePreview');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileDuration = document.getElementById('fileDuration');
        const fileResolution = document.getElementById('fileResolution');
        const videoPreview = document.getElementById('videoPreview');
        
        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        
        const url = URL.createObjectURL(file);
        videoPreview.src = url;
        
        videoPreview.addEventListener('loadedmetadata', () => {
            const duration = this.formatDuration(videoPreview.duration);
            const resolution = `${videoPreview.videoWidth}x${videoPreview.videoHeight}`;
            fileDuration.textContent = duration;
            fileResolution.textContent = resolution;
        });
        
        preview.style.display = 'block';
    }
    
    enableProcessingSection() {
        const processingSection = document.getElementById('processingSection');
        const uploadNotice = document.getElementById('uploadNotice');
        const startButton = document.getElementById('startProcessing');
        const depthModel = document.getElementById('depthModel');
        const qualitySettings = document.getElementById('qualitySettings');
        const depthIntensity = document.getElementById('depthIntensity');
        const eyeSeparation = document.getElementById('eyeSeparation');
        
        processingSection.classList.remove('disabled');
        uploadNotice.classList.add('hidden');
        startButton.disabled = false;
        depthModel.disabled = false;
        qualitySettings.disabled = false;
        depthIntensity.disabled = false;
        eyeSeparation.disabled = false;
        processingSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    disableProcessingSection() {
        const processingSection = document.getElementById('processingSection');
        const uploadNotice = document.getElementById('uploadNotice');
        const startButton = document.getElementById('startProcessing');
        const depthModel = document.getElementById('depthModel');
        const qualitySettings = document.getElementById('qualitySettings');
        const depthIntensity = document.getElementById('depthIntensity');
        const eyeSeparation = document.getElementById('eyeSeparation');
        
        processingSection.classList.add('disabled');
        uploadNotice.classList.remove('hidden');
        startButton.disabled = true;
        depthModel.disabled = true;
        qualitySettings.disabled = true;
        depthIntensity.disabled = true;
        eyeSeparation.disabled = true;
    }
    
    async startRealProcessing() {
        if (this.isProcessing || !this.currentFile) return;
        
        this.isProcessing = true;
        document.getElementById('startProcessing').style.display = 'none';
        document.getElementById('progressContainer').style.display = 'block';
        this.resetProgress();
        
        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);
            
            // Get selected model and send to backend
            const modelSelect = document.getElementById('depthModel');
            formData.append('model_name', modelSelect.value);
            
            console.log('Starting processing with model:', modelSelect.value);
            this.startProgressSimulation();
            
            const response = await fetch(`${this.backendUrl}/convert`, {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                // Try to get error message from JSON body
                try {
                    const errorResult = await response.json();
                    throw new Error(errorResult.error || `Server error: ${response.statusText}`);
                } catch (e) {
                    // If response is not JSON, use status text
                    throw new Error(`Server error: ${response.statusText}`);
                }
            }
            
            // Handle successful file download response
            const blob = await response.blob();
            this.downloadedBlob = blob; // Store the blob for later
            
            this.stopProgressSimulation();
            this.completeProcessing();
            
        } catch (error) {
            this.stopProgressSimulation();
            this.showError(`Processing failed: ${error.message}`);
            this.resetProcessingUI();
        }
    }
    
    startProgressSimulation() {
        this.currentStep = 0;
        this.processNextStep();
    }
    
    stopProgressSimulation() {
        if (this.processInterval) clearInterval(this.processInterval);
    }
    
    resetProcessingUI() {
        document.getElementById('progressContainer').style.display = 'none';
        document.getElementById('startProcessing').style.display = 'block';
        this.isProcessing = false;
    }
    
    resetProgress() {
        document.querySelectorAll('.step').forEach((el, index) => {
            el.classList.remove('active', 'completed');
            const stepNumber = el.querySelector('.step-number');
            stepNumber.textContent = index + 1;
        });
        this.updateProgressBar(0);
        document.getElementById('timeRemaining').textContent = 'Calculating...';
        document.getElementById('step1').classList.add('active');
    }
    
    processNextStep() {
        if (this.currentStep >= this.processingSteps.length) {
            return; // This is now handled by the successful fetch response
        }
        
        const step = this.processingSteps[this.currentStep];
        this.updateProgressUI(step);
        let progress = 0;
        const stepDuration = step.duration;
        const updateInterval = 100;
        const progressPerUpdate = (100 / (stepDuration / updateInterval)) / this.processingSteps.length;
        
        const stepInterval = setInterval(() => {
            progress += progressPerUpdate;
            const totalProgress = (this.currentStep / this.processingSteps.length) * 100 +
                                  (progress / this.processingSteps.length);
            
            this.updateProgressBar(Math.min(totalProgress, 100));
            this.updateTimeRemaining(totalProgress);
            
            if (progress >= 100 / this.processingSteps.length) {
                clearInterval(stepInterval);
                this.completeCurrentStep();
                this.currentStep++;
                // Stop simulation before the last step, backend call takes over
                if (this.currentStep < this.processingSteps.length - 1) {
                    setTimeout(() => this.processNextStep(), 200);
                }
            }
        }, updateInterval);
        
        this.processInterval = stepInterval;
    }
    
    updateProgressUI(step) {
        document.querySelectorAll('.step').forEach((el, index) => {
            el.classList.remove('active', 'completed');
            if (index < this.currentStep) el.classList.add('completed');
            else if (index === this.currentStep) el.classList.add('active');
        });
        
        const activeStep = document.getElementById(`step${this.currentStep + 1}`);
        if (activeStep) {
            const description = activeStep.querySelector('.step-description');
            description.textContent = step.description;
        }
    }
    
    completeCurrentStep() {
        const stepElement = document.getElementById(`step${this.currentStep + 1}`);
        if (stepElement) {
            stepElement.classList.remove('active');
            stepElement.classList.add('completed');
            const stepNumber = stepElement.querySelector('.step-number');
            stepNumber.innerHTML = '<i class="fas fa-check"></i>';
        }
    }
    
    completeProcessing() {
        this.isProcessing = false;
        this.currentStep = this.processingSteps.length; // Ensure all steps are visually complete
        this.updateProgressUI(this.processingSteps[this.processingSteps.length - 1]);
        this.completeCurrentStep();
        this.updateProgressBar(100);
        this.updateTimeRemaining(100);
        setTimeout(() => {
            this.showDownloadSection();
        }, 1000);
    }
    
    showDownloadSection() {
        const downloadSection = document.getElementById('downloadSection');
        this.updateOutputInfo();
        downloadSection.style.display = 'block';
        downloadSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    updateOutputInfo() {
        if (!this.currentFile) return;
        
        const quality = document.getElementById('qualitySettings').value;
        const outputResolution = document.getElementById('outputResolution');
        const outputSize = document.getElementById('outputSize');
        const outputDuration = document.getElementById('outputDuration');
        
        const resolutions = {
            '720p': '1280x720',
            '1080p': '1920x1080',
            '4k': '3840x2160'
        };
        
        outputResolution.textContent = resolutions[quality];
        
        // Use blob size if available, otherwise estimate
        const finalSize = this.downloadedBlob ? this.downloadedBlob.size : this.currentFile.size * 1.5;
        outputSize.textContent = `~${this.formatFileSize(finalSize)}`;
        
        const videoPreview = document.getElementById('videoPreview');
        if (videoPreview.duration) outputDuration.textContent = this.formatDuration(videoPreview.duration);
    }
    
    cancelProcessing() {
        if (!this.isProcessing) return;
        
        this.isProcessing = false;
        if (this.processInterval) clearInterval(this.processInterval);
        document.getElementById('progressContainer').style.display = 'none';
        document.getElementById('startProcessing').style.display = 'block';
        this.resetProgress();
        this.showError('Processing cancelled by user.');
    }
    
    downloadFile() {
        if (!this.downloadedBlob) {
            this.showError("No file to download. Please process a video first.");
            return;
        }
        
        // Create a URL for the blob
        const url = window.URL.createObjectURL(this.downloadedBlob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        // Create a filename
        const originalName = this.currentFile.name.replace(/\.[^/.]+$/, '');
        a.download = `${originalName}_VR180.mp4`;
        document.body.appendChild(a);
        a.click();
        // Clean up
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    
    previewFile() {
        if (!this.downloadedBlob) {
            this.showError("No file to preview. Please process a video first.");
            return;
        }
        alert('VR Preview would open in a new window. This is a demo simulation.');
    }
    
    clearFile() {
        this.currentFile = null;
        this.downloadedBlob = null;
        const fileInput = document.getElementById('fileInput');
        fileInput.value = '';
        document.getElementById('filePreview').style.display = 'none';
        document.getElementById('downloadSection').style.display = 'none';
        
        // Reset the video preview to free up memory
        const videoPreview = document.getElementById('videoPreview');
        const oldSrc = videoPreview.src;
        videoPreview.src = '';
        if (oldSrc) {
            URL.revokeObjectURL(oldSrc);
        }
        
        this.disableProcessingSection();
        this.resetProcessingUI();
    }
    
    // --- UTILITY FUNCTIONS ---
    updateProgressBar(progress) {
        const fill = document.getElementById('progressFill');
        const percent = document.getElementById('progressPercent');
        if (fill && percent) {
            fill.style.width = `${progress}%`;
            percent.textContent = `${Math.round(progress)}%`;
        }
    }
    
    updateProcessingTime() {
        const model = document.getElementById('depthModel').value;
        const depthStep = this.processingSteps.find(step => step.name === 'Depth Estimation');
        if (depthStep) {
            switch (model) {
                case 'MiDaS_small':
                    depthStep.duration = 4000; // Fast
                    break;
                case 'DPT_Large':
                    depthStep.duration = 12000; // Slower, higher quality
                    break;
                case 'DPT_Hybrid':
                    depthStep.duration = 8000; // Reliable
                    break;
                default:
                    depthStep.duration = 8000;
            }
        }
    }
    
    updateTimeRemaining(progress) {
        const timeRemainingElement = document.getElementById('timeRemaining');
        const totalDuration = this.processingSteps.reduce((acc, step) => acc + step.duration, 0);
        const elapsed = (totalDuration * progress) / 100;
        const remaining = totalDuration - elapsed;
        const remainingSeconds = Math.round(remaining / 1000);
        
        if (progress >= 100) {
            timeRemainingElement.textContent = 'Finalizing...';
        } else if (remainingSeconds < 60) {
            timeRemainingElement.textContent = `~${remainingSeconds}s remaining`;
        } else {
            const minutes = Math.floor(remainingSeconds / 60);
            const seconds = remainingSeconds % 60;
            timeRemainingElement.textContent = `~${minutes}m ${seconds}s remaining`;
        }
    }
    
    // Utility functions
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new VR180Converter();
    
    document.querySelector('.header-actions .btn').addEventListener('click', () => {
        alert('VR 180 Converter Help:\n\n1. Upload a 2D video (MP4, AVI, or MOV)\n2. Configure processing settings\n3. Start conversion\n4. Download your VR 180 video\n\nFor best results, use high-resolution videos with good depth variation.');
    });
});
