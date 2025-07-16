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

// Setup Platform form
const setupForm = document.getElementById('setup-form');
if (setupForm) {
    setupForm.addEventListener('submit', function(e) {
        e.preventDefault();
        platformSetup = {
            siteType: setupForm['site-type'].value,
            siteSpecs: setupForm['site-specs'].value
        };
        alert('Platform setup saved!');
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
        alert('Inspection data submitted!');
    });
}

// AI Analysis logic
const runAIButton = document.getElementById('run-ai-analysis');
if (runAIButton) {
    runAIButton.addEventListener('click', function() {
        if (!platformSetup) {
            document.getElementById('ai-result').innerHTML = '<span class="status-critical">Please complete platform setup first.</span>';
            return;
        }
        if (inspections.length === 0) {
            document.getElementById('ai-result').innerHTML = '<span class="status-critical">Please insert inspection data first.</span>';
            return;
        }
        // Analyze the latest inspection
        const latest = inspections[inspections.length - 1];
        let resultHtml = '';
        if (latest.status === 'normal') {
            aiStatus = 'normal';
            resultHtml = '<span class="status-normal">Normal Status: No Pings</span>';
            alerts = [];
        } else if (latest.status === 'concern-single') {
            aiStatus = 'concern-single';
            resultHtml = '<span class="status-concern">Sign of Concern (Single): Schedule repair with an alert.</span>';
            alerts = [{type: 'concern', level: 'single', message: 'Schedule repair with an alert.'}];
        } else if (latest.status === 'concern-multiple') {
            aiStatus = 'concern-multiple';
            resultHtml = '<span class="status-concern">Signs of Concern (Multiple): AI ranks priority, schedules accordingly, pings an alert.</span>';
            alerts = [{type: 'concern', level: 'multiple', message: 'AI ranks priority, schedules accordingly, pings an alert.'}];
        } else if (latest.status === 'critical') {
            aiStatus = 'critical';
            resultHtml = '<span class="status-critical">Critical: Immediate alert and schedule for immediate repair!</span>';
            alerts = [{type: 'critical', message: 'Immediate alert and schedule for immediate repair!'}];
        }
        document.getElementById('ai-result').innerHTML = resultHtml;
        updateAlertsTab();
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
    alertsList.innerHTML = alerts.map(alert => {
        if (alert.type === 'critical') {
            return `<div class="status-critical">${alert.message}</div>`;
        } else if (alert.type === 'concern' && alert.level === 'multiple') {
            return `<div class="status-concern">${alert.message}</div>`;
        } else if (alert.type === 'concern' && alert.level === 'single') {
            return `<div class="status-concern">${alert.message}</div>`;
        }
        return '';
    }).join('');
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
    }
    if (alerts.length > 0) {
        html += '<ul style="margin-top:1rem;">' + alerts.map(a => `<li>${a.message}</li>`).join('') + '</ul>';
    }
    statusSummary.innerHTML = html;
} 