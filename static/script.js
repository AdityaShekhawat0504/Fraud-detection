const resultEl = document.getElementById('result');
const batchEl = document.getElementById('batch-result');
const analyzeBtn = document.getElementById('analyze-btn');
const batchBtn = document.getElementById('batch-btn');

// ── Single Transaction ──────────────────────────────────────────────────────

document.getElementById('insight-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  resultEl.innerHTML = '';
  analyzeBtn.disabled = true;

  try {
    const data = {
      time:   parseFloat(document.getElementById('time').value),
      amount: parseFloat(document.getElementById('amount').value),
      v1: parseFloat(document.getElementById('v1').value) || 0,
      v2: parseFloat(document.getElementById('v2').value) || 0,
      v3: parseFloat(document.getElementById('v3').value) || 0,
    };

    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || 'Prediction failed');

    resultEl.innerHTML = buildSingleResult(payload);
  } catch (err) {
    resultEl.innerHTML = `<p class="error-msg">Error: ${err.message}</p>`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

function buildSingleResult({ is_anomaly, risk_score, risk_level, reasons }) {
  const statusClass = is_anomaly ? 'anomaly' : 'normal';
  const statusIcon  = is_anomaly ? '🚨' : '✅';
  const statusText  = is_anomaly ? 'Anomaly Detected' : 'Normal Transaction';

  const reasonsHtml = reasons && reasons.length
    ? `<ul class="reasons-list">${reasons.map(r => `<li>${r}</li>`).join('')}</ul>`
    : '';

  return `
    <div class="result-card ${statusClass}">
      <div class="result-header">
        <span class="result-icon">${statusIcon}</span>
        <span class="result-title">${statusText}</span>
        <span class="risk-badge risk-${risk_level.toLowerCase()}">${risk_level} Risk</span>
      </div>
      <div class="risk-bar-wrapper">
        <div class="risk-bar-label">Risk Score: <strong>${risk_score}/100</strong></div>
        <div class="risk-bar-track">
          <div class="risk-bar-fill" style="width:${risk_score}%"></div>
        </div>
      </div>
      ${is_anomaly ? `<div class="reasons"><strong>Why flagged:</strong>${reasonsHtml}</div>` : ''}
    </div>`;
}

// ── Batch Upload ────────────────────────────────────────────────────────────

async function uploadCSV() {
  batchEl.innerHTML = '';
  batchBtn.disabled = true;

  try {
    const file = document.getElementById('csv-upload').files[0];
    if (!file) throw new Error('Please choose a CSV file.');

    const formData = new FormData();
    formData.append('file', file);

    const res  = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Batch analysis failed');

    const { total, anomalies, data: rows } = data;
    const normal = total - anomalies;

    batchEl.innerHTML = `
      <p>Analyzed <strong>${total}</strong> rows &mdash;
         Normal: <strong>${normal}</strong> &nbsp;|&nbsp;
         Anomalies: <strong class="anomaly-count">${anomalies}</strong></p>`;

    updateChart([normal, anomalies]);
    renderTable(rows);
  } catch (err) {
    batchEl.innerHTML = `<p class="error-msg">Error: ${err.message}</p>`;
  } finally {
    batchBtn.disabled = false;
  }
}

// ── Chart ───────────────────────────────────────────────────────────────────

function updateChart([normal, anomalies]) {
  const ctx = document.getElementById('cluster-chart').getContext('2d');
  if (window.myChart) window.myChart.destroy();
  window.myChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Normal', 'Anomalies'],
      datasets: [{
        label: 'Transactions',
        data: [normal, anomalies],
        backgroundColor: ['#28a74588', '#dc354588'],
        borderColor:     ['#28a745',   '#dc3545'],
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
    },
  });
}

// ── Batch Results Table ─────────────────────────────────────────────────────

function renderTable(rows) {
  const section = document.getElementById('table-section');
  const tbody   = document.getElementById('results-tbody');
  tbody.innerHTML = '';

  rows.forEach((row, i) => {
    const levelClass = `risk-${(row.risk_level || 'low').toLowerCase()}`;
    const statusBadge = row.is_anomaly
      ? '<span class="badge badge-anomaly">Anomaly</span>'
      : '<span class="badge badge-normal">Normal</span>';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>$${(row.Amount || 0).toFixed(2)}</td>
      <td>
        <div class="mini-bar-track">
          <div class="mini-bar-fill" style="width:${row.risk_score}%"></div>
        </div>
        <span class="mini-score">${row.risk_score}</span>
      </td>
      <td><span class="risk-badge ${levelClass}">${row.risk_level}</span></td>
      <td>${statusBadge}</td>
      <td class="reason-cell">${row.top_reason || '—'}</td>`;
    tbody.appendChild(tr);
  });

  section.hidden = false;
}

// Initial empty chart
updateChart([0, 0]);
