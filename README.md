# CMX Airport — Live Flight Status

SAT4520/5520 Course Project

**Live site:** (fill in after deployment)
**Team members:** (fill in)

---

## Quick Start

1. Open `index.html` in your browser to see a demo
2. Get a free AviationStack key at https://aviationstack.com/signup/free
3. Enter your key in the banner on the page
4. Deploy to GitHub Pages (see below)

---

## Deploying to GitHub Pages (free public URL)

```bash
# 1. Create a new repo on github.com, then:
git init
git add .
git commit -m "Initial CMX flight status site"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cmx-flights.git
git push -u origin main

# 2. On GitHub: Settings → Pages → Source: Deploy from main branch → Save
# 3. Your URL: https://YOUR_USERNAME.github.io/cmx-flights
```

---

## APIs Used

| API | Used For | Key Required | Cost |
|-----|----------|-------------|------|
| AviationStack | Flight data (arrivals/departures) | Yes | Free (500 req/mo) |
| Open-Meteo | Weather — current & forecast | No | Free |
| FAA ASOS | Airport webcam image | No | Free |

---

## Project Structure

```
cmx-flights/
├── index.html       ← entire frontend (HTML + CSS + JS in one file)
├── README.md        ← this file
└── ai/              ← Python AI module (SAT5520 / bonus)
    ├── collect_data.py
    ├── train_model.py
    └── requirements.txt
```

---

## AI Module (SAT4520 Bonus / SAT5520 Required)

See `ai/` folder. Run in order:
1. `python ai/collect_data.py`   ← downloads BTS + historical weather
2. `python ai/train_model.py`    ← trains & evaluates the model

To serve predictions from your own model, deploy the Flask server and
update the `runAIPredictions()` function in `index.html` to call your endpoint.
