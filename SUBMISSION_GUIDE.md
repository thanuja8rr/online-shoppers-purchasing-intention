# Submission Guide — BITS WILP ML Assignment 2 (15 Marks)

Follow these steps **in order** to score full marks.

---

## 1) Push to GitHub (required)

1. Create a **new public** GitHub repository (do **not** fork an existing assignment repo).
2. From this project folder:

```bash
git init
git add .
git commit -m "BITS WILP ML Assignment-2: shopper intent classifiers + Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

3. Copy the repository URL into:
   - `README.md` section **c. GitHub Repository Link**
   - Your submission PDF (item 1)

---

## 2) Deploy Streamlit Community Cloud (required)

1. Open https://streamlit.io/cloud and sign in with GitHub.
2. Click **New App**.
3. Select your repository, branch `main`, main file `app.py`.
4. Click **Deploy**.
5. Wait until the app opens successfully.
6. Put the live app URL into:
   - `README.md`
   - Your submission PDF (item 2)

App checklist after deploy:
- Upload CSV works (`test_data.csv` from repo)
- Model dropdown lists all 5 models
- Metrics (Accuracy, AUC, Precision, Recall, F1, MCC) appear
- Confusion matrix + classification report appear

---

## 3) BITS Virtual Lab screenshot (1 mark)

1. Open **BITS Virtual Lab**.
2. Clone/upload this project and run:

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

3. Take **ONE** clear screenshot showing execution on BITS Virtual Lab.
4. Include that screenshot in the submission PDF (item 3).

---

## 4) Build the submission PDF (single file, exact order)

Create **one PDF** containing, in this order:

1. GitHub Repository Link  
2. Live Streamlit App Link  
3. BITS Virtual Lab screenshot  
4. Full README content (sections a–d, including both tables)

Submit the PDF before **18-Aug-2026, 23:59**.  
Do **not** leave the submission in DRAFT.

---

## Marks map

| Component | Marks |
|---|---|
| Model implementation + GitHub (metrics for each model, observations, dataset, repo) | 10 |
| Streamlit app features | 4 |
| BITS Virtual Lab screenshot | 1 |
| **Total** | **15** |
