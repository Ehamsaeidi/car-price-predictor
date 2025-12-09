// =========================
// # CONFIGURATION
// =========================

// # The public backend URL deployed on Railway (BACKEND)
const BACKEND_URL = "https://selfless-grace-production.up.railway.app";

// # Clean and validate the backend URL
function getApiBase() {
  if (BACKEND_URL && BACKEND_URL !== "") {
    return BACKEND_URL.replace(/\/+$/, ""); // # Remove trailing slashes
  }
  return window.location.origin.replace(/\/+$/, ""); // # Fallback: current domain
}

// # Final API base URL used for sending requests
const API_BASE = getApiBase();

// =========================
// # FORM SUBMISSION HANDLER
// =========================

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault(); // # Prevent page refresh

  const submitBtn = e.target.querySelector('[type="submit"]');
  const out = document.getElementById("output");
  const formData = new FormData(e.target);

  // =========================
  // # BUILDING PAYLOAD SAFELY
  // =========================

  const payload = {};

  for (const [k, v] of formData.entries()) {
    const raw = String(v).trim();

    // # Attempt to convert to valid number
    const maybeNum = Number(raw);
    const isNumeric =
      raw !== "" &&
      Number.isFinite(maybeNum) &&
      /^-?\d+(\.\d+)?$/.test(raw);

    payload[k] = isNumeric ? maybeNum : raw; // # Numeric OR string
  }

  out.textContent = "Predicting...";
  if (submitBtn) submitBtn.disabled = true; // # Disable submit button

  // =========================
  // # CALLING BACKEND API
  // =========================

  try {
    const url = new URL("/predict", API_BASE).toString(); // # Build final API URL

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: payload }), // # Send features object
    });

    const data = await res.json().catch(() => ({})); // # Safe JSON parse

    // # Handle failed API response
    if (!res.ok) {
      const msg = data?.error || `${res.status} ${res.statusText}`;
      throw new Error(msg);
    }

    // =========================
    // # VALIDATE PREDICTION
    // =========================

    const price = Number(data.prediction);
    if (!Number.isFinite(price)) {
      throw new
