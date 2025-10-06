// JavaScript for AI SpillGuard Application

let currentResults = null;
let processingStartTime = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    loadHistory();
    loadStats();
    
    // Refresh stats every 30 seconds
    setInterval(loadStats, 30000);
});

function initializeApp() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processImage(files[0]);
    }
}

async function processImage(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showAlert('Please select a valid image file.', 'error');
        return;
    }
    
    // Show loading state
    showLoading(true);
    processingStartTime = Date.now();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentResults = result;
            displayResults(result);
            loadHistory(); // Refresh history
            loadStats(); // Refresh stats
        } else {
            throw new Error('Detection failed');
        }
        
    } catch (error) {
        console.error('Error processing image:', error);
        showAlert('Error processing image. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(result) {
    const processingTime = ((Date.now() - processingStartTime) / 1000).toFixed(1);
    
    // Update metrics
    document.getElementById('coveragePercentage').textContent = 
        result.metrics.coverage_percentage.toFixed(2) + '%';
    document.getElementById('severityLevel').textContent = result.metrics.severity;
    document.getElementById('affectedPixels').textContent = 
        result.metrics.oil_spill_pixels.toLocaleString();
    document.getElementById('processingTime').textContent = processingTime + 's';
    
    // Update severity styling
    const severityElement = document.getElementById('severityLevel');
    severityElement.className = `severity-${result.metrics.severity.toLowerCase()}`;
    
    // Display images
    document.getElementById('originalImage').src = result.images.original;
    document.getElementById('maskImage').src = result.images.mask;
    document.getElementById('overlayImage').src = result.images.overlay;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').classList.add('fade-in-up');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    
    if (show) {
        spinner.style.display = 'block';
        resultsSection.style.display = 'none';
    } else {
        spinner.style.display = 'none';
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        
        const tbody = document.getElementById('historyTableBody');
        
        if (data.history && data.history.length > 0) {
            tbody.innerHTML = data.history.map(item => `
                <tr>
                    <td>${formatTimestamp(item.timestamp)}</td>
                    <td>${item.filename}</td>
                    <td>${item.metrics.coverage_percentage.toFixed(2)}%</td>
                    <td><span class="severity-${item.metrics.severity.toLowerCase()}">${item.metrics.severity}</span></td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="viewDetails('${item.timestamp}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No detection history available</td></tr>';
        }
        
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('historyTableBody').innerHTML = 
            '<tr><td colspan="5" class="text-center text-danger">Error loading history</td></tr>';
    }
}

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        document.getElementById('totalDetections').textContent = data.total_detections;
        document.getElementById('avgCoverage').textContent = data.average_coverage + '%';
        document.getElementById('modelStatus').textContent = data.model_status;
        document.getElementById('lastUpdated').textContent = formatTimestamp(data.last_updated);
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function downloadResults() {
    if (!currentResults) {
        showAlert('No results to download', 'warning');
        return;
    }
    
    // Create download links for each image
    const images = ['original', 'mask', 'overlay'];
    images.forEach(type => {
        const link = document.createElement('a');
        link.href = currentResults.images[type];
        link.download = `${currentResults.filename}_${type}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
    
    showAlert('Results downloaded successfully!', 'success');
}

function generateReport() {
    if (!currentResults) {
        showAlert('No results to generate report', 'warning');
        return;
    }
    
    const report = {
        timestamp: currentResults.timestamp,
        filename: currentResults.filename,
        metrics: currentResults.metrics,
        analysis: {
            risk_level: currentResults.metrics.severity,
            recommendation: getRecommendation(currentResults.metrics.severity),
            environmental_impact: getEnvironmentalImpact(currentResults.metrics.coverage_percentage)
        }
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `oil_spill_report_${currentResults.timestamp}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showAlert('Report generated successfully!', 'success');
}

function resetDetection() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('fileInput').value = '';
    currentResults = null;
    
    // Scroll back to upload area
    scrollToSection('detection');
}

function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function viewDetails(timestamp) {
    showAlert(`Viewing details for detection: ${timestamp}`, 'info');
    // This could open a modal or navigate to a detailed view
}

function formatTimestamp(timestamp) {
    if (timestamp.includes('T')) {
        return new Date(timestamp).toLocaleString();
    }
    // Handle custom timestamp format (YYYYMMDD_HHMMSS)
    const year = timestamp.substring(0, 4);
    const month = timestamp.substring(4, 6);
    const day = timestamp.substring(6, 8);
    const hour = timestamp.substring(9, 11);
    const minute = timestamp.substring(11, 13);
    const second = timestamp.substring(13, 15);
    
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function getRecommendation(severity) {
    switch (severity) {
        case 'High':
            return 'Immediate response required. Deploy cleanup crews and containment measures.';
        case 'Medium':
            return 'Monitor closely and prepare response teams. Consider preventive measures.';
        case 'Low':
            return 'Continue monitoring. Document for environmental assessment.';
        default:
            return 'Continue regular monitoring protocols.';
    }
}

function getEnvironmentalImpact(coverage) {
    if (coverage > 10) {
        return 'Severe environmental impact expected. Marine life and coastal areas at high risk.';
    } else if (coverage > 5) {
        return 'Moderate environmental impact. Local ecosystem may be affected.';
    } else {
        return 'Limited environmental impact. Localized effects possible.';
    }
}

function showAlert(message, type) {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    alert.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alert);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.parentNode.removeChild(alert);
        }
    }, 5000);
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});