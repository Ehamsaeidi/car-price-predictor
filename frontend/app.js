const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:5000'
  : (window.location.origin.replace(':8080','') || ''); // fallback for same-network

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const payload = {};
  for (const [k, v] of formData.entries()) payload[k] = isFinite(v) && v !== '' ? Number(v) : v;

  const out = document.getElementById('output');
  out.textContent = 'Predicting...';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: payload }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    const price = Number(data.prediction);
    const formatted = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(price);
    out.textContent = `Estimated price: ${formatted}`;
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
  }
});
