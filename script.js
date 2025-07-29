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
    setupForm.addEventListener('submit', function(e) {
        e.preventDefault();
        platformSetup = {
            siteType: setupForm['site-type'].value,
            siteSpecs: setupForm['site-specs'].value
        };
        const saveMsgSetup = document.getElementById('saveMessage-setup');
saveMsgSetup.classList.add('show');
setTimeout(() => {
    saveMsgSetup.classList.remove('show');
}, 2000);


        addAlert('Platform Setup Saved', `Site type: ${platformSetup.siteType || 'N/A'}<br>Specs: ${platformSetup.siteSpecs || 'N/A'}`);
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
        aiResult.innerHTML = '<div style="text-align: center; padding: 2rem;"><div style="display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #1a1a1a; border-radius: 50%; animation: spin 1s linear infinite;"></div><br><p>Running AI Analysis...</p></div>';
        
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
                    inspections: inspections
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
                    </div>
                `;
                
                aiResult.innerHTML = resultHtml;
                aiStatus = 'analysis-complete';
                addAlert('AI Analysis Complete', 'Comprehensive analysis generated using Qwen AI model');
            } else {
                aiResult.innerHTML = `<span class="status-critical">AI Analysis Failed: ${result.error}</span>`;
                aiStatus = 'analysis-failed';
                addAlert('AI Analysis Failed', result.error);
            }
        } catch (error) {
            console.error('Full error details:', error);
            console.error('Error stack:', error.stack);
            console.error('Response data:', result);
            aiResult.innerHTML = `<span class="status-critical">Connection Error: ${error.message}</span>`;
            aiStatus = 'connection-error';
            addAlert('Connection Error', 'Failed to connect to AI analysis service');
        }
        
        updateStatusTab();
    });
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