# 📊 HDB Resale Market Analytics Platform & AI Value Predictor

A production-grade, full-screen Streamlit application designed for real estate data transparency in Singapore. This platform combines cross-sectional business intelligence with an advanced machine learning regression pipeline to decode HDB resale transaction data, map neighborhood-level velocity, and predict property values.

Live Web Application: [Insert Your Streamlit Cloud Link Here]

---

## 🚀 Key Architectural Features

### 1. Interactive Market Analytics Kiosk (Tab 1)
* **Dynamic Executive Summaries:** Real-time summary blocks that automatically calculate total resale volume, average prices, and peak valuation outliers based on multi-select sidebar inputs.
* **High-Performance Geospatial Intelligence:** A lightweight Plotly Mapbox engine that utilizes custom street-level coordinate hashing to map neighborhood transaction clusters instantly without layout loops.
* **Statistical Distributions:** Interactive, high-contrast Plotly box plots and scatter correlations tracking pricing spreads across towns and floor areas.

### 2. Machine Learning Value Estimation Engine (Tab 2)
* **Algorithmic Model:** Powered by an optimized Ridge Regression pipeline yielding an interactive diagnostic $R^2$ accuracy score.
* **Advanced Feature Engineering:** * **Ordinal Encoding:** Applied to `storey_range` to natively preserve the structural premium that higher floor levels command in Singapore real estate.
    * **One-Hot Encoding:** Applied to nominal categorical variables like `town` and `flat_type` via a unified `ColumnTransformer`.
* **User Interface Protection:** Dynamically bounds dimensional floor area sliders based on the specific `flat_type` selected, ensuring zero out-of-bounds user inputs.

---

## 🛠️ Tech Stack & Dependencies

* **Core UI Engine:** Streamlit (Wide layout orientation, customized `.streamlit/config.toml` corporate branding)
* **Data Science Ecosystem:** Pandas, NumPy
* **Data Visualization:** Plotly Express (High-contrast, accessible color vision palettes)
* **Machine Learning Suite:** Scikit-Learn (Pipelines, ColumnTransformers, Ridge Regression)

---

## 💻 Local Installation & Deployment

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/cassngo86-crypto/hdb-analysis.git](https://github.com/cassngo86-crypto/hdb-analysis.git)
   cd hdb-analysis

