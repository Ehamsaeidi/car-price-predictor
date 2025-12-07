const API = "http://localhost:5000";

const $ = (id) => document.getElementById(id);
const n = (id) => {
  const v = $(id).value.trim();
  return v === "" ? null : Number(v);
};
const t = (id) => $(id).value.trim();
const money = (x) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(x);

const btn = $("predictBtn");
const result = $("result");

btn.addEventListener("click", async () => {
  // حداقل اعتبارسنجی
  const year = n("year");
  const mileage = n("mileage");
  if (!year || !mileage) {
    result.textContent = "Please enter valid Year and Mileage.";
    result.classList.remove("ok");
    return;
  }

  // داده‌ها (کلیدها با UI فعلی هماهنگ هستند)
  const payload = {
    brand: t("brand"),
    model: t("model"),
    year,
    engine_size: n("engine_size"),
    fuel_type: t("fuel_type"),
    transmission: t("transmission"),
    mileage,
    condition: t("condition"),
  };

  // حالت Loading روی دکمه
  btn.disabled = true;
  const prev = btn.textContent;
  btn.classList.add("loading");
  btn.textContent = "Predicting…";
  result.textContent = "";

  try {
    const res = await fetch(API + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    // پشتیبانی از هر دو فیلد پاسخ: price یا predicted_price
    const price = Number(
      data.price !== undefined ? data.price : data.predicted_price
    );

    if (res.ok && !Number.isNaN(price)) {
      result.textContent = "Predicted price: " + money(price);
      result.classList.add("ok");
    } else {
      result.textContent = "Prediction failed.";
      result.classList.remove("ok");
    }
  } catch (e) {
    result.textContent = "Network error. Please try again.";
    result.classList.remove("ok");
  } finally {
    btn.textContent = prev;
    btn.classList.remove("loading");
    btn.disabled = false;
  }
});
