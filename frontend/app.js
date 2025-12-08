// ---------- API base (Local vs. Railway) ----------
// When running on your laptop use the local Flask port,
// otherwise use the public Railway backend URL.
const RAILWAY_BASE = 'https://car-price-predictor-production-c712.up.railway.app';
const API_BASE =
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:5000'
    : RAILWAY_BASE;

// Small helper to avoid double slashes in URLs
const join = (base, path) => `${base.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;

// ---------- Form submit handler ----------
document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();

  // Collect the form fields and coerce numeric values
  const formData = new FormData(e.target);
  const features = {};
  for (const [k, v] of formData.entries()) {
    features[k] = (v !== '' && isFinite(v)) ? Number(v) : v;
  }

  const out = document.getElementById('output');
  out.textContent = 'Predicting...';

  try {
    const res = await fetch(join(API_BASE, '/predict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });

    // Parse response
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Request failed');

    // Pretty-print currency
    const price = Number(data.prediction);
    const formatted = new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency: 'USD',
    }).format(price);

    out.textContent = `Estimated price: ${formatted}`;
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
  }
});
