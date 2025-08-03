document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        this.classList.add('active');
        document.getElementById(this.dataset.tab).classList.add('active');
    });
});

// Data storage (in-memory for demo)
let platformSetup = null;
let inspections = [];
let aiStatus = null;
let alerts = [];

// Helper functions to get platform setup and inspections
function getPlatformSetup() {
    return platformSetup;
}

function getInspections() {
    return inspections;
}

function addAlert(title, description) {
    const now = new Date();
    const alert = {
        id: Date.now() + Math.random(),
        title,
        description,
        time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    alerts.unshift(alert); // Add to top
    updateAlertsTab();
    setTimeout(() => {
        alerts = alerts.filter(a => a.id !== alert.id);
        updateAlertsTab();
    }, 180000); // 3 minutes
}

// Setup Platform form
const setupForm = document.getElementById('setup-form');
if (setupForm) {
    setupForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = setupForm['site-specs'];
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a CSV file');
            return;
        }
        
        if (!file.name.toLowerCase().endsWith('.csv')) {
            alert('Please select a CSV file');
            return;
        }
        
        // Show loading state
        const submitButton = setupForm.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        submitButton.textContent = 'Uploading...';
        submitButton.disabled = true;
        
        try {
            // Upload the CSV file
            const formData = new FormData();
            formData.append('file', file);
            
            const uploadResponse = await fetch('/api/upload-csv', {
                method: 'POST',
                body: formData
            });
            
            const uploadResult = await uploadResponse.json();
            
            if (uploadResult.success) {
                platformSetup = {
                    siteType: setupForm['site-type'].value,
                    siteSpecs: file.name,
                    csvData: uploadResult
                };
                
                const saveMsgSetup = document.getElementById('saveMessage-setup');
                saveMsgSetup.classList.add('show');
                setTimeout(() => {
                    saveMsgSetup.classList.remove('show');
                }, 2000);
                
                addAlert('Platform Setup Saved', `Site type: ${platformSetup.siteType || 'N/A'}<br>CSV file: ${file.name}<br>Rows: ${uploadResult.rows}<br>Columns: ${uploadResult.columns.length}`);
            } else {
                alert(`Upload failed: ${uploadResult.error}`);
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload file. Please try again.');
        } finally {
            submitButton.textContent = originalText;
            submitButton.disabled = false;
        }
    });
}

// Insert Inspection Data form
const inspectionForm = document.getElementById('inspection-form');
if (inspectionForm) {
    inspectionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const inspection = {
            date: inspectionForm['inspection-date'].value,
            notes: inspectionForm['inspection-notes'].value,
            status: inspectionForm['inspection-status'].value
        };
        inspections.push(inspection);
        const saveMsgInspection = document.getElementById('saveMessage-inspection');
saveMsgInspection.classList.add('show');
setTimeout(() => {
    saveMsgInspection.classList.remove('show');
}, 2000);

        addAlert('Inspection Submitted', `Date: ${inspection.date}<br>Status: ${inspection.status}<br>Notes: ${inspection.notes}`);
    });
}

// AI Analysis logic
const runAIButton = document.getElementById('run-ai-analysis');
if (runAIButton) {
    runAIButton.addEventListener('click', async function() {
        if (!platformSetup) {
            document.getElementById('ai-result').innerHTML = '<span class="status-critical">Please complete platform setup first.</span>';
            return;
        }
        if (inspections.length === 0) {
            document.getElementById('ai-result').innerHTML = '<span class="status-critical">Please insert inspection data first.</span>';
            return;
        }
        
        // Show loading state
        const aiResult = document.getElementById('ai-result');
        aiResult.innerHTML = '<div style="text-align: center; padding: 2rem;"><div class="loading-spinner"></div><p>Running AI Analysis...</p><p style="font-size: 0.9rem; color: #666;">This may take 10-30 seconds for full analysis with charts</p></div>';
        
        try {
            // Debug: Test server connectivity first
            console.log('Testing server connectivity...');
            const testResponse = await fetch('/api/test');
            console.log('Test response status:', testResponse.status);
            
            // Call the Flask API
            console.log('Calling AI analysis endpoint...');
            const response = await fetch('/api/run-ai-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    platformSetup: platformSetup,
                    inspections: inspections,
                    lightweight: document.getElementById('lightweight-toggle').checked  // Use UI toggle
                })
            });
            
            console.log('AI analysis response status:', response.status);
            
            // Check if response is ok
            if (!response.ok) {
                if (response.status === 405) {
                    throw new Error('Method Not Allowed (405): The server does not support POST requests to this endpoint. Check if the Flask server is running correctly.');
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // Check if response has content
            const responseText = await response.text();
            if (!responseText) {
                throw new Error('Empty response from server');
            }
            
            let result;
            try {
                result = JSON.parse(responseText);
                console.log('Parsed result:', result);
                
                // Validate response structure
                if (!result || typeof result !== 'object') {
                    throw new Error('Invalid response structure: not an object');
                }
                
                if (!result.hasOwnProperty('success')) {
                    throw new Error('Invalid response structure: missing success property');
                }
                
                if (result.success && !result.hasOwnProperty('summary')) {
                    throw new Error('Invalid response structure: missing summary property');
                }
                
            } catch (parseError) {
                console.error('Response text:', responseText);
                throw new Error(`Invalid JSON response: ${parseError.message}`);
            }
            
            if (result.success) {
                // Debug logging
                console.log('AI Analysis Result:', result);
                console.log('Stats object:', result.stats);
                
                // Display the AI summary
                let resultHtml = `
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
                        <h3 style="margin-top: 0; color: #1a1a1a;">AI Analysis Summary</h3>
                        <div style="white-space: pre-wrap; line-height: 1.6;">${result.summary || 'No summary available'}</div>
                    </div>
                    <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px;">
                        <h4 style="margin-top: 0; color: #1a1a1a;">Key Statistics</h4>
                        <ul style="margin: 0; padding-left: 1.5rem;">
                            <li>Site Type: ${platformSetup.siteType || 'Not specified'}</li>
                            <li>Total Inspections: ${inspections.length}</li>
                            <li>Critical Issues: ${inspections.filter(i => i.status === 'critical').length}</li>
                            <li>Concerns: ${inspections.filter(i => i.status === 'concern-single' || i.status === 'concern-multiple').length}</li>
                            <li>Normal Status: ${inspections.filter(i => i.status === 'normal').length}</li>
                        </ul>
                    </div>
                    <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #856404;">User Configuration</h4>
                        <p><strong>Site Type:</strong> ${platformSetup.siteType || 'Not specified'}</p>
                        <p><strong>Site Specs:</strong> ${platformSetup.siteSpecs || 'Not specified'}</p>
                        <p><strong>Inspections:</strong> ${inspections.length} inspection(s) recorded</p>
                        <p><strong>CSV Data:</strong> ${platformSetup.csvData ? `${platformSetup.csvData.rows} rows, ${platformSetup.csvData.columns.length} columns` : 'No CSV data uploaded'}</p>
                    </div>
                `;
                
                // Add CSV data analysis section
                if (result.csv_stats && Object.keys(result.csv_stats).length > 0) {
                    resultHtml += `
                        <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h4 style="margin-top: 0; color: #1a1a1a;">CSV Data Analysis</h4>
                            <ul style="margin: 0; padding-left: 1.5rem;">
                                <li><strong>Data Points:</strong> ${result.csv_stats.data_points || 'N/A'}</li>
                                <li><strong>Features:</strong> ${result.csv_stats.features || 'N/A'}</li>
                                <li><strong>Target Variable:</strong> ${result.csv_stats.target_column || 'N/A'}</li>
                                <li><strong>Date Range:</strong> ${result.csv_stats.date_range ? `${result.csv_stats.date_range.start} to ${result.csv_stats.date_range.end}` : 'N/A'}</li>
                            </ul>
                        </div>
                    `;
                }
                
                // Add charts section if charts are available
                if (result.charts && result.charts.length > 0) {
                    resultHtml += `
                        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                            <h3 style="margin-top: 0; color: #1a1a1a;">Data Analysis Charts</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    `;
                    
                    result.charts.forEach(chartPath => {
                        const chartName = chartPath.split('/').pop().replace('.png', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                        resultHtml += `
                            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h4 style="margin-top: 0; color: #1a1a1a; font-size: 1.1rem;">${chartName}</h4>
                                <img src="${chartPath}" alt="${chartName}" style="width: 100%; height: auto; border-radius: 4px;" />
                            </div>
                        `;
                    });
                    
                    resultHtml += `
                            </div>
                        </div>
                    `;
                } else if (result.lightweight_mode) {
                    resultHtml += `
                        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h4 style="margin-top: 0; color: #856404;">Lightweight Mode</h4>
                            <p>Charts were not generated to save memory and processing time. The analysis was performed on your uploaded data.</p>
                        </div>
                    `;
                }
                
                // Add CSV analysis results if available
                if (result.csv_stats) {
                    resultHtml += `
                        <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h4 style="margin-top: 0; color: #1a1a1a;">CSV Data Analysis</h4>
                            <ul style="margin: 0; padding-left: 1.5rem;">
                                <li>Data Points: ${result.csv_stats.data_points || 'N/A'}</li>
                                <li>Features: ${result.csv_stats.features || 'N/A'}</li>
                                <li>Target Variable: ${result.csv_stats.target_column || 'N/A'}</li>
                                <li>Date Range: ${result.csv_stats.date_range ? `${result.csv_stats.date_range.start} to ${result.csv_stats.date_range.end}` : 'N/A'}</li>
                            </ul>
                        </div>
                    `;
                }
                
                aiResult.innerHTML = resultHtml;
                aiStatus = 'analysis-complete';
                addAlert('AI Analysis Complete', 'Comprehensive analysis generated using Qwen AI model');
                
                // Add PDF download button
                const pdfButton = document.createElement('button');
                pdfButton.textContent = '📄 Download PDF Report';
                pdfButton.style.cssText = `
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    margin-top: 1rem;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                `;
                pdfButton.addEventListener('click', () => downloadPDFReport(result, platformSetup));
                aiResult.appendChild(pdfButton);
            } else {
                aiResult.innerHTML = `<span class="status-critical">AI Analysis Failed: ${result.error}</span>`;
                aiStatus = 'analysis-failed';
                addAlert('AI Analysis Failed', result.error);
            }
        } catch (error) {
            console.error('Full error details:', error);
            console.error('Error stack:', error.stack);
            aiResult.innerHTML = `<span class="status-critical">Connection Error: ${error.message}</span>`;
            aiStatus = 'connection-error';
            addAlert('Connection Error', 'Failed to connect to AI analysis service');
        }
        
        updateStatusTab();
    });
}

// Add event listener for quick analysis button
document.getElementById('quick-ai-analysis').addEventListener('click', async function() {
    const aiResult = document.getElementById('ai-result');
    aiResult.innerHTML = '<div style="text-align: center; padding: 2rem;"><div class="loading-spinner"></div><p>Running quick analysis...</p></div>';
    
    try {
        // Check if platform setup and inspections are available
        if (!platformSetup) {
            aiResult.innerHTML = '<span class="status-critical">Please complete platform setup first.</span>';
            return;
        }
        if (inspections.length === 0) {
            aiResult.innerHTML = '<span class="status-critical">Please insert inspection data first.</span>';
            return;
        }
        
        console.log('Calling quick AI analysis endpoint...');
        
        const response = await fetch('/api/quick-ai-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                platformSetup: platformSetup,
                inspections: inspections,
                lightweight: false  // Quick analysis doesn't use lightweight mode
            })
        });
        
        console.log('Quick AI analysis response status:', response.status);
        
        // Check if response is ok
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Check if response has content
        const responseText = await response.text();
        if (!responseText) {
            throw new Error('Empty response from server');
        }
        
        let result;
        try {
            result = JSON.parse(responseText);
            console.log('Parsed quick analysis result:', result);
            
            // Validate response structure
            if (!result || typeof result !== 'object') {
                throw new Error('Invalid response structure: not an object');
            }
            
            if (!result.hasOwnProperty('success')) {
                throw new Error('Invalid response structure: missing success property');
            }
            
        } catch (parseError) {
            console.error('Response text:', responseText);
            throw new Error(`Invalid JSON response: ${parseError.message}`);
        }
        
        if (result.success) {
            // Display the quick analysis results
            let resultHtml = `
                <div style="background: #e8f5e8; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
                    <h3 style="margin-top: 0; color: #1a1a1a;">⚡ Quick Analysis Results</h3>
                    <p style="color: #28a745; font-weight: bold;">Analysis completed in ${result.elapsed_time ? result.elapsed_time.toFixed(2) : 'N/A'} seconds</p>
                    <div style="white-space: pre-wrap; line-height: 1.6;">${result.summary || 'No summary available'}</div>
                </div>
                <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #1a1a1a;">Key Statistics</h4>
                    <ul style="margin: 0; padding-left: 1.5rem;">
                        <li>Site Type: ${(result.stats && result.stats.site_type) ? result.stats.site_type : 'Not specified'}</li>
                        <li>Total Inspections: ${(result.stats && result.stats.inspections_count) ? result.stats.inspections_count : 0}</li>
                        <li>Critical Issues: ${(result.stats && result.stats.critical_inspections) ? result.stats.critical_inspections : 0}</li>
                        <li>Concerns: ${(result.stats && result.stats.concern_inspections) ? result.stats.concern_inspections : 0}</li>
                        <li>Normal Status: ${(result.stats && result.stats.normal_inspections) ? result.stats.normal_inspections : 0}</li>
                    </ul>
                </div>
                <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #856404;">User Configuration</h4>
                    <p><strong>Site Type:</strong> ${platformSetup.siteType || 'Not specified'}</p>
                    <p><strong>Site Specs:</strong> ${platformSetup.siteSpecs || 'Not specified'}</p>
                    <p><strong>Inspections:</strong> ${inspections.length} inspection(s) recorded</p>
                    <p><strong>CSV Data:</strong> ${platformSetup.csvData ? `${platformSetup.csvData.rows} rows, ${platformSetup.csvData.columns.length} columns` : 'No CSV data uploaded'}</p>
                </div>
            `;
            
            // Add CSV data analysis section
            if (result.csv_stats && Object.keys(result.csv_stats).length > 0) {
                resultHtml += `
                    <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #1a1a1a;">CSV Data Analysis</h4>
                        <ul style="margin: 0; padding-left: 1.5rem;">
                            <li><strong>Data Points:</strong> ${result.csv_stats.data_points || 'N/A'}</li>
                            <li><strong>Features:</strong> ${result.csv_stats.features || 'N/A'}</li>
                            <li><strong>Target Variable:</strong> ${result.csv_stats.target_variable || 'N/A'}</li>
                            <li><strong>Date Range:</strong> ${result.csv_stats.date_range || 'N/A'}</li>
                            <li><strong>File:</strong> ${result.stats.uploaded_filename || 'N/A'}</li>
                        </ul>
                    </div>
                `;
            }
            
            // Add note about quick mode
            resultHtml += `
                <div style="background: #d1ecf1; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #0c5460;">Quick Analysis Mode</h4>
                    <p>This was a fast analysis without chart generation. For detailed charts and PDF reports, use the "Run AI Analysis" button.</p>
                    <button onclick="runFullAnalysis()" style="background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 0.5rem;">
                        🎨 Run Full Analysis with Charts
                    </button>
                </div>
            `;
            
            aiResult.innerHTML = resultHtml;
            aiStatus = 'analysis-complete';
            addAlert('Quick Analysis Complete', 'Fast analysis completed successfully');
            
        } else {
            aiResult.innerHTML = `<span class="status-critical">Quick Analysis Failed: ${result.error}</span>`;
            aiStatus = 'analysis-failed';
            addAlert('Quick Analysis Failed', result.error);
        }
        
    } catch (error) {
        console.error('Quick analysis error:', error);
        aiResult.innerHTML = `<span class="status-critical">Connection Error: ${error.message}</span>`;
        aiStatus = 'connection-error';
        addAlert('Connection Error', 'Failed to connect to quick analysis service');
    }
    
    updateStatusTab();
});

// Function to run full analysis (called from quick analysis results)
function runFullAnalysis() {
    document.getElementById('run-ai-analysis').click();
}

function updateAlertsTab() {
    const alertsList = document.getElementById('alerts-list');
    if (!alertsList) return;
    if (alerts.length === 0) {
        alertsList.innerHTML = '<span class="status-normal">No alerts. System normal.</span>';
        return;
    }
    alertsList.innerHTML = alerts.map(alert => `
        <div class="alert-card">
            <div class="alert-title">${alert.title}</div>
            <div class="alert-desc">${alert.description}</div>
            <div class="alert-time">${alert.time}</div>
        </div>
    `).join('');
}

// Test function to check server connectivity
async function testServerConnection() {
    try {
        const response = await fetch('/api/test');
        const result = await response.json();
        console.log('Server test result:', result);
        return result;
    } catch (error) {
        console.error('Server test failed:', error);
        return null;
    }
}

function updateStatusTab() {
    const statusSummary = document.getElementById('status-summary');
    if (!statusSummary) return;
    let html = '';
    if (!aiStatus) {
        html = '<span>No analysis run yet.</span>';
    } else if (aiStatus === 'normal') {
        html = '<span class="status-normal">Normal Status: No Pings</span>';
    } else if (aiStatus === 'concern-single') {
        html = '<span class="status-concern">Sign of Concern (Single): Repair scheduled.</span>';
    } else if (aiStatus === 'concern-multiple') {
        html = '<span class="status-concern">Signs of Concern (Multiple): Repairs scheduled, priority ranked.</span>';
    } else if (aiStatus === 'critical') {
        html = '<span class="status-critical">Critical: Immediate repair scheduled!</span>';
    } else if (aiStatus === 'analysis-complete') {
        html = '<span class="status-normal">AI Analysis Complete</span>';
    } else if (aiStatus === 'analysis-failed') {
        html = '<span class="status-critical">AI Analysis Failed</span>';
    } else if (aiStatus === 'connection-error') {
        html = '<span class="status-critical">Connection Error</span>';
    }
    if (alerts.length > 0) {
        html += '<ul style="margin-top:1rem;">' + alerts.map(a => `<li>${a.title}: ${a.description}</li>`).join('') + '</ul>';
    }
    statusSummary.innerHTML = html;
}

// Function to download PDF report
async function downloadPDFReport(analysisResult, platformSetup) {
    try {
        // Show loading state
        const button = event.target;
        const originalText = button.textContent;
        button.textContent = '📄 Generating PDF...';
        button.disabled = true;
        
        // Prepare data for PDF generation
        const pdfData = {
            summary: analysisResult.summary,
            stats: analysisResult.stats,
            charts: analysisResult.charts || [],
            site_name: platformSetup.siteType || 'Energy Site'
        };
        
        // Call PDF generation endpoint
        const response = await fetch('/api/generate-pdf-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(pdfData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Convert base64 to blob and download
            const pdfBlob = base64ToBlob(result.pdf_data, 'application/pdf');
            const url = URL.createObjectURL(pdfBlob);
            
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = result.filename;
            downloadLink.style.display = 'none';
            
            // Trigger download
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            // Clean up
            URL.revokeObjectURL(url);
            
            addAlert('PDF Report Downloaded', `Report saved as: ${result.filename}`);
        } else {
            alert(`PDF generation failed: ${result.error}`);
        }
        
    } catch (error) {
        console.error('PDF download error:', error);
        alert('Failed to generate PDF report. Please try again.');
    } finally {
        // Restore button state
        const button = event.target;
        button.textContent = originalText;
        button.disabled = false;
    }
}

// Helper function to convert base64 to blob
function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
} 