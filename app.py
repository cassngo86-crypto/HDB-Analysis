# =============================================================================
# 1. GLOBAL CORE IMPORTS & CONFIGURATION (MUST BE TOP FIRST)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

st.set_page_config(
    page_title="HDB Resale Analytics Kiosk",
    layout="wide",                  # Stretches the app to use the full screen layout
    initial_sidebar_state="expanded"
)

ACCESSIBLE_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

# =============================================================================
# 2. CACHED DATA PIPELINE (SAFE FROM MUTATION)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading HDB transaction dataset...")
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    # Read raw data frame
    base_df = pd.read_csv(file_path)
    
    # Create an explicit deep copy to avoid editing global cache vectors
    df = base_df.copy()
    
    for col in ['town', 'flat_type', 'flat_model', 'storey_range', 'street_name']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
            
    if 'month' in df.columns:
        df['year'] = df['month'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else int(x))
        df = df.sort_values('month').reset_index(drop=True)

    # Street-level neighborhood jitter (Created safely on the copied data frame)
    if 'town_lat' in df.columns and 'town_lon' in df.columns and 'street_name' in df.columns:
        df['street_hash'] = df['street_name'].apply(lambda x: int(abs(hash(x))) % 1000 / 1000.0)
        df['town_lat'] = df['town_lat'] + (df['street_hash'] - 0.5) * 0.007
        df['town_lon'] = df['town_lon'] + (df['street_hash'] - 0.5) * 0.007
        df = df.drop(columns=['street_hash'])
        
    return df


@st.cache_resource(show_spinner="Training predictive model with ordinal vertical metrics...")
def train_prediction_model(df):
    features = ['town', 'flat_type', 'floor_area_sqm', 'lease_commence_date', 'storey_range']
    if not all(col in df.columns for col in features + ['resale_price']):
        return None, None
    
    model_df = df[features + ['resale_price']].dropna()
    X = model_df[features]
    y = model_df['resale_price']
    
    unique_storeys = sorted(list(df['storey_range'].dropna().unique()))
    nominal_categorical = ['town', 'flat_type']
    ordinal_categorical = ['storey_range']
    numeric_features = ['floor_area_sqm', 'lease_commence_date']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('nom_cat', OneHotEncoder(handle_unknown='ignore'), nominal_categorical),
            ('ord_cat', OrdinalEncoder(categories=[unique_storeys], handle_unknown='use_encoded_value', unknown_value=-1), ordinal_categorical)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    pipeline.fit(X, y)
    r2_score = pipeline.score(X, y)
    return pipeline, r2_score


# =============================================================================
# 3. RUN PIPELINES & SIDEBAR FILTER CONTROLS
# =============================================================================
DATA_FILE = "hdb_resale_flats.csv" 
raw_df = load_data(DATA_FILE)

if raw_df is not None:
    # Always keep model training isolated on the raw copy
    model_pipeline, model_r2 = train_prediction_model(raw_df)
    
    st.sidebar.header("📊 Filter Options")
    
    selected_towns = st.sidebar.multiselect(
        "Select Towns:", 
        options=sorted(raw_df['town'].unique()), 
        default=sorted(raw_df['town'].unique())[:3]
    )
    selected_flat_types = st.sidebar.multiselect(
        "Select Flat Types:", 
        options=sorted(raw_df['flat_type'].unique()), 
        default=sorted(raw_df['flat_type'].unique())
    )
    
    # CRITICAL LOOP FIX: Create an explicit isolated copy of the filtered dataframe!
    # This guarantees that working with data in the tabs won't trigger re-execution loops.
    filtered_df = raw_df[
        (raw_df['town'].isin(selected_towns)) & 
        (raw_df['flat_type'].isin(selected_flat_types))
    ].copy()
    
    # =========================================================================
    # STEP 4: GLOBAL SIDEBAR ENGINE & FILTER ASSIGNMENT (WITH UNIQUE KEYS)
    # =========================================================================
    st.sidebar.header("📊 Filter Options")
    
    selected_towns = st.sidebar.multiselect(
        "Select Towns:", 
        options=sorted(raw_df['town'].unique()), 
        default=sorted(raw_df['town'].unique())[:3],
        key="global_town_filter"  # <-- ADD THIS UNIQUE KEY
    )
    selected_flat_types = st.sidebar.multiselect(
        "Select Flat Types:", 
        options=sorted(raw_df['flat_type'].unique()), 
        default=sorted(raw_df['flat_type'].unique()),
        key="global_flat_filter"  # <-- ADD THIS UNIQUE KEY
    )
    
    # Building filtered_df globally right here
    filtered_df = raw_df[
        (raw_df['town'].isin(selected_towns)) & 
        (raw_df['flat_type'].isin(selected_flat_types))
    ].copy()
    
    # =========================================================================
    # STEP 5: TABS NAVIGATION LAYER
    # =========================================================================
    tab1, tab2 = st.tabs(["📊 Market Analytics Dashboard", "🔮 AI Price Predictor"])
    
    # --- TAB 1 LAYOUT ---
    with tab1:
        st.subheader("📊 Cross-Sectional Market Performance Summary")
        
        if not filtered_df.empty:
            total_volume = len(filtered_df)
            avg_price = filtered_df['resale_price'].mean()
            avg_psf = (filtered_df['resale_price'] / filtered_df['floor_area_sqm']).mean()
            
            top_row = filtered_df.loc[filtered_df['resale_price'].idxmax()]
            top_town = top_row['town']
            top_val = top_row['resale_price']
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Resale Volume", f"{total_volume:,} Units")
            m_col2.metric("Average Resale Price", f"${avg_price:,.2f} SGD")
            m_col3.metric("Estimated Mean PSF", f"${avg_psf:,.2f} / m²")
            
            st.markdown(f"""
            > 💡 **Executive Summary Trend:** Within your active filtered criteria, a total of **{total_volume:,}** HDB resale transactions were processed. 
            > Properties are trading at an average valuation threshold of **${avg_price:,.2f} SGD**. 
            > The most capital-intensive single transaction recorded sits in **{top_town}** commanding a peak market evaluation of **${top_val:,.2f} SGD**.
            """)
            
            # --- MAP ---
            # --- MAP PORTAL WITH BOUNDARY LOCK ---
            # --- DIAGNOSTIC NATIVE MAP PORTAL ---
            st.write("---")
            st.subheader("🗺️ Geospatial Market Distribution Map")
            
            if 'town_lat' in filtered_df.columns and 'town_lon' in filtered_df.columns:
                # Isolate map metrics coordinates cleanly
                map_df = filtered_df[['town_lat', 'town_lon']].dropna().rename(
                    columns={'town_lat': 'latitude', 'town_lon': 'longitude'}
                )
                
                if not map_df.empty:
                    # Native Streamlit map uses your browser's default mapbox engine safely
                    st.map(map_df, use_container_width=True)
                else:
                    st.warning("⚠️ No valid geographical coordinates available for the selected data slice.")
            
    # --- TAB 2 LAYOUT ---
    with tab2:
        st.subheader("🔮 Machine Learning Value Estimation")
        if model_pipeline is not None:
            st.info(f"📈 **Model Diagnostic Stat:** The Ridge regression model is active with an $R^2$ of **{model_r2:.4f}**.")
            
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                pred_town = st.selectbox("Target Town Location:", options=sorted(raw_df['town'].unique()))
                pred_flat_type = st.selectbox("Target Flat Configuration:", options=sorted(raw_df['flat_type'].unique()))
                pred_storey = st.selectbox("Storey/Floor Level Range:", options=sorted(raw_df['storey_range'].unique()))
                
            with col_in2:
                HDB_DIMENSIONS = {
                    '1 ROOM': (30, 40, 35), '2 ROOM': (35, 55, 45), '3 ROOM': (55, 75, 68),
                    '4 ROOM': (80, 105, 92), '5 ROOM': (110, 125, 115), 'EXECUTIVE': (130, 160, 142),
                    'MULTI-GENERATION': (150, 180, 165)
                }
                lookup_key = str(pred_flat_type).strip().upper()
                min_area, max_area, mean_area = HDB_DIMENSIONS.get(lookup_key, (30, 150, 90))
                
                pred_area = st.slider(f"Floor Area Range ({pred_flat_type}):", min_value=int(min_area), max_value=int(max_area), value=int(mean_area))
                pred_lease = st.slider("Lease Commencement Year:", min_value=1966, max_value=2026, value=2010)
                
            input_data = pd.DataFrame([{
                'town': pred_town, 'flat_type': pred_flat_type, 'floor_area_sqm': pred_area,
                'lease_commence_date': pred_lease, 'storey_range': pred_storey
            }])
            
            predicted_price = model_pipeline.predict(input_data)[0]
            st.markdown("<h3 style='text-align: center;'>🔮 Estimated Resale Evaluation</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: #0072B2;'>${max(0.0, predicted_price):,.2f} SGD</h1>", unsafe_allow_html=True)
else:
    st.error("❌ Could not connect to data source. Check your filepath specifications.")