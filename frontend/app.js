// app.js

// API endpoint — automatically uses the same origin as the frontend + /api
// If you manually set window.BACKEND_BASE in index.html, it will override this.
const API = window.BACKEND_BASE || (window.location.origin.replace(/\/$/, '') + "/api");

// Helper function: read the value of each input field
// - If the value is numeric, convert it to a Number
// - Otherwise return it as a string
function val(id) {
    const v = document.getElementById(id).value.trim();
    return /^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v;
}

// Handle Predict button click
document.getElementById("predictBtn").onclick = async () => {

    // Collect all form fields in a single object
    const features = {
        "Brand": val("Brand"),
        "Model": val("Model"),
        "Year": val("Year"),
        "Engine Size": val("Engine Size"),
        "Fuel Type": val("Fuel Type"),
        "Transmission": val("Transmission"),
        "Mileage": val("Mileage"),
        "Condition": val("Condition")
    };

    console.log("Sending data to backend:", features);

    // Send data to backend API
    const res = await fetch(API + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
    });

    const data = await res.json();

    // Show prediction result or error
    document.getElementById("result").textContent =
        data.predicted_price !== undefined
            ? ("Predicted price: " + data.predicted_price)
            : ("Error: " + (data.error || "Unknown error"));
};
