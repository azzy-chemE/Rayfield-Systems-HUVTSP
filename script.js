// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(tab.dataset.tab).classList.add('active');
    });
  });
  
  // In-memory data
  let platformSetup = null;
  let inspections = [];
  let alerts = [];
  let aiStatus = null;
  
  // Alerts handling
  function addAlert(title, description) {
    const now = new Date();
    const alert = {
      id: Date.now() + Math.random(),
      title,
      description,
      time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    alerts.unshift(alert);
    updateAlertsTab();
    setTimeout(() => {
      alerts = alerts.filter(a => a.id !== alert.id);
      updateAlertsTab();
    }, 180000);
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
  
  // Status overview
  function updateStatusTab() {
    const statusSummary = document.getElementById('status-summary');
    if (!statusSummary) return;
    let html = '';
    if (!aiStatus) {
      html = '<span>No analysis run yet.</span>';
    } else if (aiStatus === 'analysis-complete') {
      html = '<span class="status-normal">AI Analysis Complete</span>';
    } else {
      html = '<span class="status-critical">Error occurred</span>';
    }
    statusSummary.innerHTML = html;
  }
  
  // Platform setup form
  const setupForm = document.getElementById('setup-form');
  if (setupForm) {
    setupForm.addEventListener('submit', async e => {
      e.preventDefault();
      const fileInput = setupForm['site-specs'];
      const file = fileInput.files[0];
      if (!file || !file.name.toLowerCase().endsWith('.csv')) {
        alert('Please select a CSV file');
        return;
      }
      const button = setupForm.querySelector('button[type="submit"]');
      const originalText = button.textContent;
      button.textContent = 'Uploading...';
      button.disabled = true;
      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/upload-csv', { method: 'POST', body: formData });
        const result = await response.json();
        if (result.success) {
          platformSetup = {
            siteType: setupForm['site-type'].value,
            siteSpecs: file.name,
            csvData: result
          };
          document.getElementById('saveMessage-setup').classList.add('show');
          setTimeout(() => document.getElementById('saveMessage-setup').classList.remove('show'), 2000);
          addAlert(
            'Platform Setup Saved',
            `Site type: ${platformSetup.siteType}<br>CSV file: ${file.name}<br>Rows: ${result.rows}<br>Columns: ${result.columns.length}`
          );
        } else {
          alert(`Upload failed: ${result.error}`);
        }
      } catch (err) {
        console.error('Upload error:', err);
        alert('Failed to upload file. Please try again.');
      } finally {
        button.textContent = originalText;
        button.disabled = false;
      }
    });
  }
  
  // Inspection form
  const inspectionForm = document.getElementById('inspection-form');
  if (inspectionForm) {
    inspectionForm.addEventListener('submit', e => {
      e.preventDefault();
      const inspection = {
        date: inspectionForm['inspection-date'].value,
        notes: inspectionForm['inspection-notes'].value,
        status: inspectionForm['inspection-status'].value
      };
      inspections.push(inspection);
      document.getElementById('saveMessage-inspection').classList.add('show');
      setTimeout(() => document.getElementById('saveMessage-inspection').classList.remove('show'), 2000);
      addAlert(
        'Inspection Submitted',
        `Date: ${inspection.date}<br>Status: ${inspection.status}<br>Notes: ${inspection.notes}`
      );
    });
  }
  
  // AI Analysis
  const runAIButton = document.getElementById('run-ai-analysis');
  if (runAIButton) {
    runAIButton.addEventListener('click', async () => {
      const aiResult = document.getElementById('ai-result');
      if (!platformSetup) {
        aiResult.innerHTML = '<span class="status-critical">Please complete platform setup first.</span>';
        return;
      }
      if (inspections.length === 0) {
        aiResult.innerHTML = '<span class="status-critical">Please insert inspection data first.</span>';
        return;
      }
      aiResult.innerHTML = '<div class="loading-spinner"></div><p>Running AI Analysis...</p>';
      try {
        const response = await fetch('/api/run-ai-analysis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ platformSetup, inspections })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        const text = await response.text();
        if (!text) throw new Error('Empty response');
        const result = JSON.parse(text);
        if (result.success) {
          aiStatus = 'analysis-complete';
          let html = `
            <div class="analysis-summary">
              <h3>AI Analysis Summary</h3>
              <p>${result.summary || 'No summary available'}</p>
            </div>
            <div class="analysis-stats">
              <h4>Key Statistics</h4>
              <ul>
                <li>Total Inspections: ${inspections.length}</li>
                <li>Critical Issues: ${inspections.filter(i => i.status === 'critical').length}</li>
                <li>Concerns: ${inspections.filter(i => i.status.startsWith('concern')).length}</li>
                <li>Normal: ${inspections.filter(i => i.status === 'normal').length}</li>
              </ul>
            </div>
          `;
          if (result.charts && result.charts.length) {
            html += '<div class="analysis-charts">';
            result.charts.forEach(path => {
              html += `<div class="chart-card"><img src="${path}" alt="chart" /></div>`;
            });
            html += '</div>';
          }
          html += '<button id="download-pdf">📄 Download PDF Report</button>';
          aiResult.innerHTML = html;
          document.getElementById('download-pdf').addEventListener('click', () =>
            downloadPDFReport(result, platformSetup)
          );
          addAlert('AI Analysis Complete', 'Comprehensive analysis generated.');
        } else {
          aiStatus = 'analysis-failed';
          aiResult.innerHTML = `<span class="status-critical">AI Analysis Failed: ${result.error}</span>`;
          addAlert('AI Analysis Failed', result.error);
        }
      } catch (e) {
        aiStatus = 'connection-error';
        aiResult.innerHTML = `<span class="status-critical">Error: ${e.message}</span>`;
        addAlert('Connection Error', e.message);
        console.error(e);
      } finally {
        updateStatusTab();
      }
    });
  }
  
  // PDF download
  async function downloadPDFReport(analysisResult, platformSetup) {
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = 'Generating PDF...';
    btn.disabled = true;
    try {
      const response = await fetch('/api/generate-pdf-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...analysisResult, site_name: platformSetup.siteType })
      });
      const result = await response.json();
      if (result.success) {
        const blob = base64ToBlob(result.pdf_data, 'application/pdf');
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.filename;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
        addAlert('PDF Report Downloaded', `Saved as ${result.filename}`);
      } else {
        alert(`PDF generation failed: ${result.error}`);
      }
    } catch (err) {
      console.error('PDF download error:', err);
      alert('Failed to generate PDF.');
    } finally {
      btn.textContent = orig;
      btn.disabled = false;
    }
  }
  
  function base64ToBlob(base64, mimeType) {
    const bytes = atob(base64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
      arr[i] = bytes.charCodeAt(i);
    }
    return new Blob([arr], { type: mimeType });
  }
  