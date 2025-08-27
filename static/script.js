const resultEl = document.getElementById('result');
const batchEl = document.getElementById('batch-result');
const analyzeBtn = document.getElementById('analyze-btn');
const batchBtn = document.getElementById('batch-btn');

document.getElementById('insight-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  resultEl.textContent = '';
  analyzeBtn.disabled = true;

  try {
    const data = {
      time: parseFloat(document.getElementById('time').value),
      amount: parseFloat(document.getElementById('amount').value),
      v1: parseFloat(document.getElementById('v1').value),
      v2: parseFloat(document.getElementById('v2').value),
      v3: parseFloat(document.getElementById('v3').value)
    };

    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.error || 'Prediction failed');
    }

    resultEl.innerHTML = payload.is_anomaly
      ? '<p style="color:red;">🚨 Anomaly Detected!</p>'
      : '<p style="color:green;">✅ Normal Transaction</p>';
  } catch (err) {
    resultEl.innerHTML = `<p style="color:#b00020;">Error: ${err.message}</p>`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

async function uploadCSV() {
  batchEl.textContent = '';
  batchBtn.disabled = true;

  try {
    const file = document.getElementById('csv-upload').files[0];
    if (!file) {
      throw new Error('Please choose a CSV file.');
    }

    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Batch analysis failed');
    }

    const anomalies = data.anomalies;
    const total = data.total;
    const normal = total - anomalies;

    batchEl.innerHTML = `<p>Analyzed <strong>${total}</strong> rows → Normal: <strong>${normal}</strong>, Anomalies: <strong>${anomalies}</strong></p>`;
    updateChart([normal, anomalies]);
  } catch (err) {
    batchEl.innerHTML = `<p style="color:#b00020;">Error: ${err.message}</p>`;
  } finally {
    batchBtn.disabled = false;
  }
}

function updateChart(data) {
  const ctx = document.getElementById('cluster-chart').getContext('2d');
  if (window.myChart) window.myChart.destroy();
  window.myChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Normal', 'Anomalies'],
      datasets: [{ label: 'Transactions', data }]
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } }
  });
}

// Initial chart seed
updateChart([0, 0]);
