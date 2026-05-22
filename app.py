import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ACCESSIBILITY SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Singapore HDB Resale Analytics & AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom accessible color palette
ACCESSIBLE_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

st.title("🏢 Singapore HDB Resale Price Analytics & Prediction")

# -----------------------------------------------------------------------------
# 2. OPTIMIZED DATA LOADING PIPELINE (Step 1: Define the function)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading HDB transaction dataset...")
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # Standard string format transformations
    for col in ['town', 'flat_type', 'flat_model', 'storey_range']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
            
    # Parse transaction year safely
    if 'month' in df.columns:
        df['year'] = df['month'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else int(x))
        df = df.sort_values('month').reset_index(drop=True)
    
    # --- AUTOMATIC REMAINING LEASE FALLBACK GENERATOR ---
    if 'remaining_lease' in df.columns:
        df['remaining_lease_clean'] = df['remaining_lease'].astype(str).str.upper().str.strip()
        is_missing = df['remaining_lease_clean'].isna() | (df['remaining_lease_clean'] == 'NONE') | (df['remaining_lease_clean'] == 'NAN')
        
        if is_missing.any() and 'year' in df.columns and 'lease_commence_date' in df.columns:
            df['remaining_lease'] = df.apply(
                lambda row: f"{int(99 - (row['year'] - row['lease_commence_date']))} years" 
                if str(row['remaining_lease']).strip().upper() in ['NONE', 'NAN', ''] 
                else row['remaining_lease'], 
                axis=1
            )
        df = df.drop(columns=['remaining_lease_clean'])
        
    return df

# (Step 2: Actually create the raw_df variable so it exists)
CSV_FILENAME = "hdb_resale_flats.csv"
raw_df = load_data(CSV_FILENAME)

if raw_df is None:
    st.error(f"❌ Critical Error: File '{CSV_FILENAME}' not found at root layout level.")
    st.stop()

# Dynamic Date Coverage Banner Output
if 'month' in raw_df.columns and not raw_df.empty:
    min_month = raw_df['month'].min()
    max_month = raw_df['month'].max()
    st.info(f"🗓️ **Dataset Coverage Period:** {min_month} to {max_month}")
else:
    st.info("🗓️ **Dataset Coverage Period:** Timeline data processing...")

st.markdown("""
This interactive system provides historical HDB transaction insights along with a live-trained 
Machine Learning regression model to estimate current valuation parameters.
""")
st.write("---")

# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING MODEL TRAINING PIPELINE (Step 3: Define the ML function)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training predictive regression model...")
def train_prediction_model(df):
    features = ['town', 'flat_type', 'floor_area_sqm', 'lease_commence_date']
    if not all(col in df.columns for col in features + ['resale_price']):
        return None, None
    
    model_df = df[features + ['resale_price']].dropna()
    X = model_df[features]
    y = model_df['resale_price']
    
    categorical_features = ['town', 'flat_type']
    numeric_features = ['floor_area_sqm', 'lease_commence_date']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    pipeline.fit(X, y)
    r2_score = pipeline.score(X, y)
    return pipeline, r2_score

# (Step 4: Now that raw_df AND the function exist, we can call it safely!)
model_pipeline, model_r2 = train_prediction_model(raw_df)

# -----------------------------------------------------------------------------
# 4. TAB NAVIGATION WORKSPACE (Your tabs start right here...)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Market Analytics Dashboard", "🔮 AI Price Predictor (Regression Model)"])
# ... rest of your code remains exactly the same below this line
# -----------------------------------------------------------------------------
# TAB 1: HISTORICAL DATA MARKET ANALYSIS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("🔍 Filter Options")
    
    all_towns = sorted(raw_df['town'].unique()) if 'town' in raw_df.columns else []
    all_flat_types = sorted(raw_df['flat_type'].unique()) if 'flat_type' in raw_df.columns else []
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_towns = st.multiselect("Select Towns for Analysis:", options=all_towns, default=all_towns[:3] if len(all_towns) >= 3 else all_towns)
    with col_f2:
        selected_flat_types = st.multiselect("Select Flat Types for Analysis:", options=all_flat_types, default=all_flat_types)
        
    filtered_df = raw_df.copy()
    if selected_towns:
        filtered_df = filtered_df[filtered_df['town'].isin(selected_towns)]
    if selected_flat_types:
        filtered_df = filtered_df[filtered_df['flat_type'].isin(selected_flat_types)]
        
    # -------------------------------------------------------------------------
    # SECTION 4: EXECUTIVE SUMMARY METRICS BANNER (UPDATED)
    # -------------------------------------------------------------------------
    st.write("---")
    if not filtered_df.empty:
        total_transactions = len(filtered_df)
        avg_price = filtered_df['resale_price'].mean() if 'resale_price' in filtered_df.columns else 0.0
        avg_psf = (filtered_df['resale_price'] / filtered_df['floor_area_sqm']).mean() if 'floor_area_sqm' in filtered_df.columns and 'resale_price' in filtered_df.columns else 0.0
        
        # Displaying the 3-Column Calculation Grid
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Total Transacted Units", value=f"{total_transactions:,}")
        c2.metric(label="Average Resale Price", value=f"${avg_price:,.2f}")
        c3.metric(label="Average Price Per SQM", value=f"${avg_psf:,.2f} / sqm")
        
        # ---------------------------------------------------------------------
        # 5. DATA VISUALIZATION PORTALS
        # ---------------------------------------------------------------------
        st.write("---")
        l_col1, l_col2 = st.columns(2)
        
        with l_col1:
            st.subheader("📈 Resale Price Distribution by Town")
            fig_box = px.box(filtered_df, x='town', y='resale_price', color='town', color_discrete_sequence=ACCESSIBLE_PALETTE, labels={'town': 'Town', 'resale_price': 'Price (SGD)'})
            fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
            
        with l_col2:
            st.subheader("📐 Floor Area vs Resale Price Correlation")
            fig_scatter = px.scatter(filtered_df, x='floor_area_sqm', y='resale_price', color='flat_type', color_discrete_sequence=ACCESSIBLE_PALETTE, opacity=0.6)
            fig_scatter.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        # ---------------------------------------------------------------------
        # 6. RAW COMPREHENSIVE DATA ENGINE VIEW
        # ---------------------------------------------------------------------
        st.write("---")
        st.subheader("📋 Filtered Transaction Ledger Data")
        st.dataframe(
            filtered_df,
            column_config={
                "resale_price": st.column_config.NumberColumn("Resale Price (SGD)", format="$%d"),
                "floor_area_sqm": st.column_config.NumberColumn("Floor Area (Sqm)", format="%d m²"),
                "lease_commence_date": st.column_config.NumberColumn("Lease Commence Year", format="%d")
            },
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("⚠️ No records match selection filters. Please adjust your criteria.")

# -----------------------------------------------------------------------------
# ---------------------------------------------------------------------
        # TAB 2: DYNAMIC PARAMETER SELECTION
        # ---------------------------------------------------------------------
        st.markdown("### 🔧 Input Target Flat Specifications")
        
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            pred_town = st.selectbox("Target Town Location:", options=sorted(raw_df['town'].unique()))
            pred_flat_type = st.selectbox("Target Flat Configuration:", options=sorted(raw_df['flat_type'].unique()))
            
        with col_in2:
            # Filter dataset to extract ranges ONLY for the selected flat type
            type_specific_df = raw_df[raw_df['flat_type'] == pred_flat_type]
            
            if not type_specific_df.empty:
                min_area = int(type_specific_df['floor_area_sqm'].min())
                max_area = int(type_specific_df['floor_area_sqm'].max())
                mean_area = int(type_specific_df['floor_area_sqm'].mean())
            else:
                # Safe fallback if empty
                min_area, max_area, mean_area = 30, 200, 90
            
            # If min and max happen to be identical, pad them to prevent Streamlit slider crash
            if min_area == max_area:
                max_area += 5
                
            # Dynamic Slider: Updates its range automatically when the flat type changes!
            pred_area = st.slider(
                f"Floor Area for {pred_flat_type} (Square Meters):", 
                min_value=min_area, 
                max_value=max_area, 
                value=mean_area
            )
            
            # Lease slider constraints remain standard
            min_lease = int(raw_df['lease_commence_date'].min()) if 'lease_commence_date' in raw_df.columns else 1966
            max_lease = int(raw_df['lease_commence_date'].max()) if 'lease_commence_date' in raw_df.columns else 2026
            
            pred_lease = st.slider("Lease Commencement Year:", min_value=min_lease, max_value=max_lease, value=max_lease-10)
        st.write("---")
        
        # Build DataFrame for prediction matching pipeline schema requirements
        input_data = pd.DataFrame([{
            'town': pred_town,
            'flat_type': pred_flat_type,
            'floor_area_sqm': pred_area,
            'lease_commence_date': pred_lease
        }])
        
        # Calculate Prediction
        try:
            predicted_price = model_pipeline.predict(input_data)[0]
            predicted_price = max(0.0, predicted_price)  # Logical lower-bound floor
            
            st.markdown("<h3 style='text-align: center;'>🔮 Estimated Resale Evaluation Valuation</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: #0072B2;'>${predicted_price:,.2f} SGD</h1>", unsafe_allow_html=True)
            
            # Contextual data baseline helper
            baseline_match = raw_df[(raw_df['town'] == pred_town) & (raw_df['flat_type'] == pred_flat_type)]
            if not baseline_match.empty:
                historical_avg = baseline_match['resale_price'].mean()
                st.markdown(f"<p style='text-align: center; color: gray;'>Historical empirical base average for a {pred_flat_type} in {pred_town} is <b>${historical_avg:,.2f} SGD</b></p>", unsafe_allow_html=True)
        except Exception as pred_err:
            st.error(f"Prediction Pipeline Exception: {pred_err}")