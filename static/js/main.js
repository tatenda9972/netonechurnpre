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
    
    // Initialize customer data table with pagination and filtering
    initializeCustomerTable();
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
                    console.log("Histogram data available:", data.histograms);
                    const features = ['Age', 'Tenure_Months', 'Data_Usage_GB', 'Monthly_Bill'];
                    
                    features.forEach(feature => {
                        // Replace all underscores, not just the first one
                        const elementId = feature.toLowerCase().replaceAll('_', '-') + '-histogram';
                        console.log(`Looking for element with ID: ${elementId}, exists:`, !!document.getElementById(elementId));
                        if (data.histograms[feature] && document.getElementById(elementId)) {
                            console.log(`Creating histogram for ${feature} with data:`, data.histograms[feature]);
                            try {
                                createHistogram(
                                    elementId,
                                    data.histograms[feature],
                                    feature
                                );
                                console.log(`Histogram for ${feature} created successfully`);
                            } catch (err) {
                                console.error(`Error creating histogram for ${feature}:`, err);
                            }
                        } else {
                            console.warn(`Cannot create histogram for ${feature}: element or data missing`);
                        }
                    });
                } else {
                    console.warn("No histogram data available in the chart data");
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

// Function to initialize customer data table with pagination and filtering
function initializeCustomerTable() {
    const customerTable = document.getElementById('customerTable');
    if (!customerTable) return;
    
    const allCustomerData = document.getElementById('allCustomerData');
    const customerRows = allCustomerData ? allCustomerData.querySelectorAll('.customer-row') : [];
    const customerTableBody = document.getElementById('customerTableBody');
    const loadMoreBtn = document.getElementById('loadMoreBtn');
    const churnFilterToggle = document.getElementById('churnFilterToggle');
    const rowsToDisplay = document.getElementById('rowsToDisplay');
    const visibleRowCount = document.getElementById('visibleRowCount');
    const totalRowCount = document.getElementById('totalRowCount');
    
    let displayedRows = 10;
    let isChurnFilterOn = false;
    let maxRows = 10;
    
    // Format bill value with dollar sign
    function formatBill(bill) {
        if (bill === 'N/A' || bill === null || bill === undefined) return 'N/A';
        return '$' + parseFloat(bill).toFixed(2);
    }
    
    // Format probability value with percentage
    function formatProbability(probability) {
        if (probability === 'N/A' || probability === null || probability === undefined) return 'N/A';
        return parseFloat(probability).toFixed(1) + '%';
    }
    
    // Create table row HTML
    function createRowHTML(rowData) {
        const isChurning = rowData.getAttribute('data-is-churning') === '1';
        const trClass = isChurning ? 'table-danger' : '';
        const predictionBadge = isChurning ? 
            '<span class="badge bg-danger">Likely to Churn</span>' : 
            '<span class="badge bg-success">Likely to Stay</span>';
        
        return `
            <tr class="${trClass}" data-is-churning="${isChurning ? '1' : '0'}">
                <td>${rowData.getAttribute('data-id')}</td>
                <td>${rowData.getAttribute('data-age')}</td>
                <td>${rowData.getAttribute('data-gender')}</td>
                <td>${rowData.getAttribute('data-location')}</td>
                <td>${rowData.getAttribute('data-tenure')}</td>
                <td>${formatBill(rowData.getAttribute('data-bill'))}</td>
                <td>${predictionBadge}</td>
                <td>${formatProbability(rowData.getAttribute('data-probability'))}</td>
            </tr>
        `;
    }
    
    // Update the displayed row counter
    function updateRowCount(displayedRowCount, totalAvailableRows) {
        if (visibleRowCount) visibleRowCount.textContent = displayedRowCount;
        if (totalRowCount) totalRowCount.textContent = totalAvailableRows;
    }
    
    // Update the table with filtered and limited rows
    function updateTable() {
        if (!customerTableBody || !customerRows.length) return;
        
        // Clear current table
        customerTableBody.innerHTML = '';
        
        // Count total available rows after applying filter
        let totalAvailableRows = 0;
        let rowsAdded = 0;
        
        // Add rows to the table up to the limit
        for (let i = 0; i < customerRows.length; i++) {
            const row = customerRows[i];
            const isChurning = row.getAttribute('data-is-churning') === '1';
            
            // Apply churn filter if active
            if (isChurnFilterOn && !isChurning) continue;
            
            totalAvailableRows++;
            
            // Only add rows up to current display limit
            if (rowsAdded < displayedRows) {
                customerTableBody.innerHTML += createRowHTML(row);
                rowsAdded++;
            }
        }
        
        // Update displayed row count
        updateRowCount(rowsAdded, totalAvailableRows);
        
        // Update load more button visibility
        if (loadMoreBtn) {
            if (rowsAdded >= totalAvailableRows) {
                loadMoreBtn.disabled = true;
                loadMoreBtn.classList.add('disabled');
            } else {
                loadMoreBtn.disabled = false;
                loadMoreBtn.classList.remove('disabled');
            }
        }
    }
    
    // Load more rows when the button is clicked
    if (loadMoreBtn) {
        loadMoreBtn.addEventListener('click', function() {
            // Increase displayed rows by the current increment
            displayedRows += maxRows;
            updateTable();
        });
    }
    
    // Toggle filter to show only customers likely to churn
    if (churnFilterToggle) {
        churnFilterToggle.addEventListener('change', function() {
            isChurnFilterOn = this.checked;
            // Reset displayed rows when filter changes
            displayedRows = maxRows;
            updateTable();
        });
    }
    
    // Change the number of rows to display
    if (rowsToDisplay) {
        rowsToDisplay.addEventListener('change', function() {
            const value = this.value;
            if (value === 'all') {
                displayedRows = 9999; // Effectively show all rows
            } else {
                maxRows = parseInt(value, 10);
                displayedRows = maxRows;
            }
            updateTable();
        });
    }
    
    // Initialize the table on page load
    updateTable();
}
