// --- Tab Switching ---
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        this.classList.add('active');
        document.getElementById(this.dataset.tab).classList.add('active');
    });
});

// --- Data Storage ---
let platformSetup = null;
let inspections = [];
let aiStatus = null;
let alerts = [];

// --- Helpers ---
function getPlatformSetup() { return platformSetup; }
function getInspections() { return inspections; }

function addAlert(title, description) {
    const alert = {
        id: Date.now() + Math.random(),
        title,
        description,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    alerts.unshift(alert);
    updateAlertsTab();

    // Auto-remove after 3 minutes
    setTimeout(() => {
        alerts = alerts.filter(a => a.id !== alert.id);
        updateAlertsTab();
    }, 180000);
}

// --- Setup Platform Form ---
const setupForm = document.getElementById('setup-form');
if (setupForm) {
    setupForm.addEventListener('submit', async e => {
        e.preventDefault();

        const file = setupForm['site-specs'].files[0];
        if (!file || !file.name.toLowerCase().endsWith('.csv')) {
            alert('Please select a valid CSV file');
            return;
        }

        const submitButton = setupForm.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        submitButton.textContent = 'Uploading...';
        submitButton.disabled = true;

        try {
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

                const saveMsg = document.getElementById('saveMessage-setup');
                saveMsg.classList.add('show');
                setTimeout(() => saveMsg.classList.remove('show'), 2000);

                addAlert('Platform Setup Saved', `Site type: ${platformSetup.siteType}<br>CSV file: ${file.name}<br>Rows: ${uploadResult.rows}<br>Columns: ${uploadResult.columns.length}`);
            } else {
                alert(`Upload failed: ${uploadResult.error}`);
            }
        } catch {
            alert('Failed to upload file. Please try again.');
        } finally {
            submitButton.textContent = originalText;
            submitButton.disabled = false;
        }
    });
}

// --- Inspection Data Form ---
const inspectionForm = document.getElementById('inspection-form');
if (inspectionForm) {
    inspectionForm.addEventListener('submit', e => {
        e.preventDefault();

        inspections.push({
            date: inspectionForm['inspection-date'].value,
            notes: inspectionForm['inspection-notes'].value,
            status: inspectionForm['inspection-status'].value
        });

        const saveMsg = document.getElementById('saveMessage-inspection');
        saveMsg.classList.add('show');
        setTimeout(() => saveMsg.classList.remove('show'), 2000);

        addAlert('Inspection Submitted', `Date: ${inspectionForm['inspection-date'].value}<br>Status: ${inspectionForm['inspection-status'].value}<br>Notes: ${inspectionForm['inspection-notes'].value}`);
    });
}

// --- AI Analysis (Full) ---
const runAIButton = document.getElementById('run-ai-analysis');
if (runAIButton) {
    runAIButton.addEventListener('click', async () => {
        if (!platformSetup) return showAIError('Please complete platform setup first.');
        if (inspections.length === 0) return showAIError('Please insert inspection data first.');

        const aiResult = document.getElementById('ai-result');
        aiResult.innerHTML = loadingTemplate('Running AI Analysis...', 'This may take 10-30 seconds for full analysis with charts');

        try {
            const response = await fetch('/api/run-ai-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    platformSetup,
                    inspections,
                    lightweight: document.getElementById('lightweight-toggle').checked
                })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

            const result = await response.json();
            if (!result.success) throw new Error(result.error);

            renderAIResult(result, aiResult);
            aiStatus = 'analysis-complete';
            addAlert('AI Analysis Complete', 'Comprehensive analysis generated using Qwen AI model');

            if (result.charts?.length) addAlert('Graph Generated Successfully', 'Check graph for anomalies');
        } catch (err) {
            showAIError(`Connection Error: ${err.message}`);
            aiStatus = 'connection-error';
        }

        updateStatusTab();
    });
}
// --- Utility Functions ---
function showAIError(message) {
    document.getElementById('ai-result').innerHTML = `<span class="status-critical">${message}</span>`;
    addAlert('Error', message);
}

function loadingTemplate(title, subtitle = '') {
    return `
        <div style="text-align:center; padding:2rem;">
            <div class="loading-spinner"></div>
            <p>${title}</p>
            ${subtitle ? `<p style="font-size:0.9rem; color:#666;">${subtitle}</p>` : ''}
        </div>`;
}

function renderAIResult(result, aiResult) {
    let html = `
        <div style="background:#f8f9fa;padding:1.5rem;border-radius:8px;margin-bottom:1rem;">
            <h3>AI Analysis Summary</h3>
            <div style="white-space:pre-wrap; line-height:1.6;">${result.summary || 'No summary available'}</div>
        </div>
        <div style="background:#e8f4fd;padding:1rem;border-radius:8px;">
            <h4>Key Statistics</h4>
            <ul style="margin:0;padding-left:1.5rem;">
                <li>Site Type: ${platformSetup.siteType}</li>
                <li>Total Inspections: ${inspections.length}</li>
                <li>Critical Issues: ${inspections.filter(i => i.status === 'critical').length}</li>
                <li>Concerns: ${inspections.filter(i => i.status.includes('concern')).length}</li>
                <li>Normal Status: ${inspections.filter(i => i.status === 'normal').length}</li>
            </ul>
        </div>`;
    aiResult.innerHTML = html;

    // PDF download button
    const pdfButton = document.createElement('button');
    pdfButton.textContent = '📄 Download PDF Report';
    pdfButton.style.cssText = `background:#007bff;color:white;padding:12px 24px;border-radius:6px;margin-top:1rem;cursor:pointer;`;
    pdfButton.addEventListener('click', () => downloadPDFReport(result, platformSetup));
    aiResult.appendChild(pdfButton);
}

function renderQuickAIResult(result, aiResult) {
    aiResult.innerHTML = `
        <div style="background:#e8f5e8;padding:1.5rem;border-radius:8px;margin-bottom:1rem;">
            <h3>⚡ Quick Analysis Results</h3>
            <p style="color:#28a745;font-weight:bold;">
                Analysis completed in ${result.elapsed_time?.toFixed(2) || 'N/A'} seconds
            </p>
            <div style="white-space:pre-wrap;line-height:1.6;">${result.summary || 'No summary available'}</div>
        </div>
        <div style="background:#d1ecf1;padding:1rem;border-radius:8px;margin-top:1rem;">
            <h4>Quick Analysis Mode</h4>
            <p>This was a fast analysis without chart generation. For detailed charts, use "Run AI Analysis".</p>
            <button onclick="runFullAnalysis()" style="background:#007bff;color:white;padding:8px 16px;border-radius:4px;margin-top:0.5rem;">
                🎨 Run Full Analysis
            </button>
        </div>`;
}

function runFullAnalysis() {
    document.getElementById('run-ai-analysis').click();
}

function updateAlertsTab() {
    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = alerts.length
        ? alerts.map(a => `
            <div class="alert-card">
                <div class="alert-title">${a.title}</div>
                <div class="alert-desc">${a.description}</div>
                <div class="alert-time">${a.time}</div>
            </div>`).join('')
        : '<span class="status-normal">No alerts. System normal.</span>';
}

function updateStatusTab() {
    const statusSummary = document.getElementById('status-summary');
    if (!statusSummary) return;

    let html = aiStatus
        ? `<span class="status-${aiStatus.includes('critical') ? 'critical' : 'normal'}">${aiStatus.replace('-', ' ')}</span>`
        : '<span>No analysis run yet.</span>';

    if (alerts.length > 0) {
        html += '<ul style="margin-top:1rem;">' + alerts.map(a => `<li>${a.title}: ${a.description}</li>`).join('') + '</ul>';
    }

    statusSummary.innerHTML = html;
}

// --- PDF Generation ---
async function downloadPDFReport(analysisResult, platformSetup) {
    try {
        const button = event.target;
        const originalText = button.textContent;
        button.textContent = '📄 Generating PDF...';
        button.disabled = true;

        const response = await fetch('/api/generate-pdf-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                summary: analysisResult.summary,
                stats: analysisResult.stats,
                charts: analysisResult.charts || [],
                site_name: platformSetup.siteType || 'Energy Site'
            })
        });

        const result = await response.json();
        if (result.success) {
            const pdfBlob = base64ToBlob(result.pdf_data, 'application/pdf');
            const url = URL.createObjectURL(pdfBlob);
            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = result.filename;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            URL.revokeObjectURL(url);

            addAlert('PDF Report Downloaded', `Report saved as: ${result.filename}`);
        } else {
            alert(`PDF generation failed: ${result.error}`);
        }

        button.textContent = originalText;
        button.disabled = false;
    } catch {
        alert('Failed to generate PDF report. Please try again.');
    }
}

function base64ToBlob(base64, mimeType) {
    const byteArray = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
    return new Blob([byteArray], { type: mimeType });
}
