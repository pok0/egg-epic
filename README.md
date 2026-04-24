# 🥚 Scale-Free Thai Chicken Egg Sorter

A Streamlit web app for Thai egg analysis using image processing and mathematical modeling.

## Features
- **Mode 1** – Upload egg photo + Thai coin → auto-detects dimensions via OpenCV
- **Mode 2** – Upload egg photo + select known grade → proportional scaling
- **Mode 3** – Manual input of `a` (half-length) and `b_max` (max radius)

All modes compute: Volume, Surface Area, Arc Length, Shell Volume, Egg White/Yolk amounts, and nutritional components (albumin, folic acid, iron, choline, zinc, vitamin B12).

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set `app.py` as the main file
4. Deploy — Streamlit Cloud reads `requirements.txt` automatically

## Repo structure
```
├── app.py            ← main app
├── requirements.txt  ← dependencies
└── README.md
```

## Thai Egg Grades (TFDA Standard)
| Grade | Weight    |
|-------|-----------|
| 0     | ≥ 65 g    |
| 1     | 60–65 g   |
| 2     | 55–60 g   |
| 3     | < 55 g    |

## Math Model
Based on Narushin-type ovoid parametric equations with:
- `a` = semi-major axis (half the length, tip to tip)
- `b_max` = maximum equatorial radius
- `g`, `t`, `k`, `b` = grade-specific shape parameters
