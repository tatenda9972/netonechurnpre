/**
 * Main JavaScript functionality for NetOne Churn Prediction System
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Handle file input change to show file name
    const fileInput = document.querySelector('.custom-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0].name;
            const nextSibling = e.target.nextElementSibling;
            nextSibling.innerText = fileName;
        });
    }
    
    // Handle confirmation dialogs
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm(this.dataset.confirm)) {
                e.preventDefault();
            }
        });
    });
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
    
    // Toggle password visibility
    const togglePassword = document.querySelectorAll('.toggle-password');
    togglePassword.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const passwordField = document.querySelector(this.dataset.target);
            const type = passwordField.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordField.setAttribute('type', type);
            
            // Toggle icon
            this.classList.toggle('fa-eye');
            this.classList.toggle('fa-eye-slash');
        });
    });
    
    // Handle CSV file validation
    const csvForm = document.querySelector('#prediction-form');
    if (csvForm) {
        csvForm.addEventListener('submit', function(e) {
            const fileInput = this.querySelector('input[type="file"]');
            
            if (fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a CSV file for prediction.');
                return false;
            }
            
            const fileName = fileInput.files[0].name;
            if (!fileName.toLowerCase().endsWith('.csv')) {
                e.preventDefault();
                alert('Please upload a valid CSV file.');
                return false;
            }
            
            // Show loading spinner
            const submitBtn = this.querySelector('button[type="submit"]');
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            submitBtn.disabled = true;
            
            return true;
        });
    }
    
    // Initialize charts if they exist on the page
    initializeCharts();
});

// Function to initialize charts on the page
function initializeCharts() {
    // Check if we're on the insights or prediction result page
    if (document.getElementById('churn-pie-chart')) {
        // Get chart data from the page
        const chartData = document.getElementById('chart-data');
        
        if (chartData) {
            try {
                const data = JSON.parse(chartData.textContent);
                
                // Create churn pie chart
                if (data.churnDistribution && document.getElementById('churn-pie-chart')) {
                    createChurnPieChart(
                        'churn-pie-chart', 
                        data.churnDistribution.churned, 
                        data.churnDistribution.retained
                    );
                }
                
                // Create feature importance chart
                if (data.featureImportance && document.getElementById('feature-importance-chart')) {
                    createFeatureImportanceChart(
                        'feature-importance-chart',
                        data.featureImportance.features,
                        data.featureImportance.importance
                    );
                }
                
                // Create confusion matrix
                if (data.confusionMatrix && document.getElementById('confusion-matrix')) {
                    createConfusionMatrix('confusion-matrix', data.confusionMatrix);
                }
                
                // Create churn timeline chart
                if (data.churnTimeline && document.getElementById('churn-timeline-chart')) {
                    createChurnTimeline(
                        'churn-timeline-chart',
                        data.churnTimeline.dates,
                        data.churnTimeline.rates
                    );
                }
                
                // Create demographic charts
                if (data.demographics) {
                    if (data.demographics.gender && document.getElementById('gender-chart')) {
                        createDemographicChart(
                            'gender-chart',
                            data.demographics.gender.labels,
                            data.demographics.gender.values,
                            'Gender Distribution'
                        );
                    }
                    
                    if (data.demographics.location && document.getElementById('location-chart')) {
                        createDemographicChart(
                            'location-chart',
                            data.demographics.location.labels,
                            data.demographics.location.values,
                            'Location Distribution'
                        );
                    }
                    
                    if (data.demographics.planType && document.getElementById('plan-type-chart')) {
                        createDemographicChart(
                            'plan-type-chart',
                            data.demographics.planType.labels,
                            data.demographics.planType.values,
                            'Plan Type Distribution'
                        );
                    }
                }
                
                // Create histograms for numeric features
                if (data.histograms) {
                    const features = ['Age', 'Tenure_Months', 'Data_Usage_GB', 'Monthly_Bill'];
                    
                    features.forEach(feature => {
                        const elementId = feature.toLowerCase().replace('_', '-') + '-histogram';
                        if (data.histograms[feature] && document.getElementById(elementId)) {
                            createHistogram(
                                elementId,
                                data.histograms[feature],
                                feature
                            );
                        }
                    });
                }
            } catch (error) {
                console.error('Error initializing charts:', error);
            }
        }
    }
}

// Function to handle deletion confirmation
function confirmDelete(message, formId) {
    if (confirm(message)) {
        document.getElementById(formId).submit();
    }
    return false;
}

// Function to copy text to clipboard
function copyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    
    // Show a temporary message
    const message = document.createElement('div');
    message.className = 'alert alert-success position-fixed bottom-0 end-0 m-3';
    message.innerHTML = 'Copied to clipboard!';
    document.body.appendChild(message);
    
    setTimeout(() => {
        message.remove();
    }, 2000);
}
