/**
 * Charts.js - Handles chart rendering for NetOne Churn Prediction System
 * Uses Chart.js library for creating interactive and responsive charts
 */

// Function to create a pie chart for churn distribution
function createChurnPieChart(elementId, churned, retained) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.churnPieChart) {
        window.churnPieChart.destroy();
    }
    
    window.churnPieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Churned', 'Retained'],
            datasets: [{
                data: [churned, retained],
                backgroundColor: ['#dc3545', '#28a745'],
                borderColor: ['#dc3545', '#28a745'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = churned + retained;
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Customer Churn Distribution',
                    font: {
                        size: 16
                    }
                }
            }
        }
    });
}

// Function to create feature importance chart
function createFeatureImportanceChart(elementId, features, importances) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Sort features by importance (descending)
    const indices = Array.from(Array(features.length).keys())
        .sort((a, b) => importances[b] - importances[a]);
    
    const sortedFeatures = indices.map(i => features[i]);
    const sortedImportances = indices.map(i => importances[i]);
    
    // Limit to top 10 features for better visualization
    const topFeatures = sortedFeatures.slice(0, 10);
    const topImportances = sortedImportances.slice(0, 10);
    
    // Destroy existing chart if it exists
    if (window.featureChart) {
        window.featureChart.destroy();
    }
    
    window.featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topFeatures,
            datasets: [{
                label: 'Feature Importance',
                data: topImportances,
                backgroundColor: '#0046b8',
                borderColor: '#0046b8',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `Importance: ${value.toFixed(4)}`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Top Feature Importance',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Features'
                    }
                }
            }
        }
    });
}

// Function to create confusion matrix visualization
function createConfusionMatrix(elementId, matrix) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Calculate derived values
    const trueNegative = matrix[0][0];
    const falsePositive = matrix[0][1];
    const falseNegative = matrix[1][0];
    const truePositive = matrix[1][1];
    
    // Destroy existing chart if it exists
    if (window.confusionMatrix) {
        window.confusionMatrix.destroy();
    }
    
    window.confusionMatrix = new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                data: [
                    { x: 'Predicted Non-Churn', y: 'Actual Non-Churn', v: trueNegative },
                    { x: 'Predicted Churn', y: 'Actual Non-Churn', v: falsePositive },
                    { x: 'Predicted Non-Churn', y: 'Actual Churn', v: falseNegative },
                    { x: 'Predicted Churn', y: 'Actual Churn', v: truePositive }
                ],
                backgroundColor(ctx) {
                    const value = ctx.dataset.data[ctx.dataIndex].v;
                    const x = ctx.dataset.data[ctx.dataIndex].x;
                    const y = ctx.dataset.data[ctx.dataIndex].y;
                    
                    if ((x === 'Predicted Non-Churn' && y === 'Actual Non-Churn') || 
                        (x === 'Predicted Churn' && y === 'Actual Churn')) {
                        return `rgba(40, 167, 69, ${0.5 + 0.5 * (value / Math.max(trueNegative, truePositive))})`;
                    }
                    
                    return `rgba(220, 53, 69, ${0.5 + 0.5 * (value / Math.max(falsePositive, falseNegative))})`;
                },
                borderColor: '#fff',
                borderWidth: 1,
                width: 30,
                height: 30
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        title() {
                            return '';
                        },
                        label(context) {
                            const v = context.dataset.data[context.dataIndex];
                            return [`${v.y} | ${v.x}`, `Count: ${v.v}`];
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Confusion Matrix',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Predicted'
                    },
                    ticks: {
                        stepSize: 1
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Actual'
                    },
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
    
    // If the matrix chart type isn't supported, fallback to a simpler visualization
    if (!window.confusionMatrix.canvas) {
        const container = document.getElementById(elementId).parentNode;
        
        // Create a simple HTML table for the confusion matrix
        const table = document.createElement('table');
        table.className = 'table table-bordered confusion-matrix-table';
        
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th></th>
                <th>Predicted Non-Churn</th>
                <th>Predicted Churn</th>
            </tr>
        `;
        
        const tbody = document.createElement('tbody');
        tbody.innerHTML = `
            <tr>
                <th>Actual Non-Churn</th>
                <td class="bg-success-light">${trueNegative}</td>
                <td class="bg-danger-light">${falsePositive}</td>
            </tr>
            <tr>
                <th>Actual Churn</th>
                <td class="bg-danger-light">${falseNegative}</td>
                <td class="bg-success-light">${truePositive}</td>
            </tr>
        `;
        
        table.appendChild(thead);
        table.appendChild(tbody);
        
        // Replace canvas with table
        container.innerHTML = '';
        container.appendChild(table);
    }
}

// Function to create historical churn rate timeline chart
function createChurnTimeline(elementId, dates, rates) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.timelineChart) {
        window.timelineChart.destroy();
    }
    
    window.timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Churn Rate (%)',
                data: rates,
                fill: false,
                borderColor: '#ff6c00',
                tension: 0.1,
                pointBackgroundColor: '#0046b8',
                pointBorderColor: '#0046b8',
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Churn Rate: ${context.raw.toFixed(2)}%`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Churn Rate Over Time',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Churn Rate (%)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Function to create demographic distribution chart
function createDemographicChart(elementId, labels, values, title) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Generate a range of colors
    const colors = [
        '#0046b8', '#1a56c0', '#3367c8', '#4d77d0', '#6688d8', 
        '#8099e0', '#99aae8', '#b3bbef', '#cccdf7', '#e6deff'
    ];
    
    // Ensure we have enough colors
    while (colors.length < labels.length) {
        colors.push(...colors);
    }
    
    // Create chart ID based on element
    const chartId = elementId + 'Chart';
    
    // Destroy existing chart if it exists
    if (window[chartId]) {
        window[chartId].destroy();
    }
    
    window[chartId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: colors.slice(0, labels.length),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                }
            }
        }
    });
}

// Function to create histogram for numeric features
function createHistogram(elementId, data, feature, bins = 10) {
    // Calculate bins and frequencies
    const values = data.map(item => parseFloat(item[feature])).filter(v => !isNaN(v));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins;
    
    const binCounts = Array(bins).fill(0);
    const binLabels = [];
    
    // Create bin labels
    for (let i = 0; i < bins; i++) {
        const lowerBound = min + i * binWidth;
        const upperBound = min + (i + 1) * binWidth;
        binLabels.push(`${lowerBound.toFixed(2)} - ${upperBound.toFixed(2)}`);
    }
    
    // Count values in each bin
    values.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
        binCounts[binIndex]++;
    });
    
    // Create chart
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Create chart ID based on element
    const chartId = elementId + 'Histogram';
    
    // Destroy existing chart if it exists
    if (window[chartId]) {
        window[chartId].destroy();
    }
    
    window[chartId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: feature,
                data: binCounts,
                backgroundColor: '#0046b8',
                borderColor: '#0046b8',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: `Distribution of ${feature}`,
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: feature
                    }
                }
            }
        }
    });
}
