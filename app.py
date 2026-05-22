import streamlit as st
import os
import plotly.express as px
import pandas as pd
from engine import RealEstateEngine
# Add these to your imports at the very top of app.py

# --- CORRECTED MACHINE LEARNING IMPORTS --- 
# Change the import at the very top of app.py to this:
from sklearn.ensemble import RandomForestRegressor  # Swapped from linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Page Configuration
st.set_page_config(page_title="HDB Asset Valuation Engine", layout="wide")

st.title("🏠 Real Estate Asset Valuation & Lease Decay Engine")
st.markdown("Automated algorithmic scoping for Singapore HDB resale trends.")


import os
import streamlit as st
# Make sure your engine import is at the top
from engine import RealEstateEngine  

# 1. Dynamically find the absolute path to your folder on the Streamlit server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_NAME = "hdb_resale_flats.csv"
FULL_DATA_PATH = os.path.join(BASE_DIR, CSV_FILE_NAME)

# 2. Setup the Caching Engine Safely
@st.cache_resource
def load_hdb_engine():
    # Double check that the server can see the file before feeding it to the engine
    if not os.path.exists(FULL_DATA_PATH):
        st.error(f"⚠️ Server path mismatch! Engine cannot find: {FULL_DATA_PATH}")
        st.stop()
        
    # Initialize your engine using the verified path
    engine = RealEstateEngine(FULL_DATA_PATH)
    engine.load_and_clean_data()
    return engine

# 3. Initialize your app variables safely
try:
    engine = load_hdb_engine()
    # Safely retrieve your dataframe from the cached engine
    if hasattr(engine, 'df'):
        cleaned_df = engine.df
    else:
        cleaned_df = engine.load_and_clean_data()
except Exception as e:
    st.error(f"❌ RealEstateEngine Initialization Failed: {str(e)}")
    st.info("💡 If this is a parsing error, ensure engine.py reads the path variable correctly.")
    st.stop()

# --- Your remaining sidebar and dashboard layout code continues below ---

# --- NEW: ADVANCED FEATURE ENGINEERING DIRECTLY IN APP ---
# Helper function for floor midpoint calculation
def clean_storey(storey_str):
    if pd.isna(storey_str):
        return 1
    try:
        parts = str(storey_str).upper().split(' TO ')
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
        return int(parts[0])
    except Exception:
        return 1

# Inject the new columns safely into the active dataframe
if 'floor_midpoint' not in cleaned_df.columns:
    cleaned_df['floor_midpoint'] = cleaned_df['storey_range'].apply(clean_storey)

if 'balas_retention_pct' not in cleaned_df.columns:
    # Industry standard geometric formula for remaining lease retention
    cleaned_df['balas_retention_pct'] = 100 * (1 - (0.992 ** cleaned_df['remaining_lease_years']))
# --- 2. SIDEBAR INTERACTION CONTROLS ---

st.sidebar.header("Filter Options")

selected_town = st.sidebar.selectbox("Select Town:", options=sorted(cleaned_df['town'].unique()))

# Filter out '1 ROOM' from the options list dynamically
flat_type_options = [x for x in sorted(cleaned_df['flat_type'].unique()) if x != '1 ROOM']

selected_flat_type = st.sidebar.selectbox("Select Flat Type:", options=flat_type_options)

# --- 3. FILTER DATASET BASED ON SELECTIONS ---
# Changed 'df' to 'cleaned_df' to match your data loader variable!
subset = cleaned_df[(cleaned_df['town'] == selected_town) & (cleaned_df['flat_type'] == selected_flat_type)]

# --- INTERACTIVE LEASE SLIDER (BULLETPROOF VERSION) ---
if subset.empty or subset['remaining_lease_years'].isna().all():
    min_lease = 1
    max_lease = 99
    default_lease = 50
else:
    valid_leases = subset['remaining_lease_years'].dropna()
    if len(valid_leases) == 0:
        min_lease, max_lease, default_lease = 1, 99, 50
    else:
        min_lease = int(float(valid_leases.min()))
        max_lease = int(float(valid_leases.max()))
        default_lease = int(float(valid_leases.median()))

if min_lease == max_lease:
    min_lease = max(1, min_lease - 1)

target_lease = st.sidebar.slider(
    "Target Remaining Lease (Years):", 
    min_value=min_lease, 
    max_value=max_lease, 
    value=default_lease
)
# --- NEW: INTERACTIVE FLOOR SLIDER ---
if subset.empty:
    min_floor, max_floor = 1, 30
else:
    min_floor = int(subset['floor_midpoint'].min())
    max_floor = int(subset['floor_midpoint'].max())

if min_floor == max_floor:
    min_floor = max(1, min_floor - 1)

target_floor = st.sidebar.slider(
    "Minimum Floor Level (Midpoint):",
    min_value=min_floor,
    max_value=max_floor,
    value=min_floor,
    step=1
)

# Apply the floor level adjustment to the final subset
subset = subset[subset['floor_midpoint'] >= target_floor]

# --- 4. HIGH-IMPACT KPI METRIC LAYER ---
st.markdown("### 📊 Market Snapshot Matrix")

if not subset.empty:
    # 1. Calculate Average Price Per SQM
    avg_ppsm = subset['price_per_sqm'].mean()
    overall_avg_ppsm = cleaned_df['price_per_sqm'].mean()
    ppsm_delta = avg_ppsm - overall_avg_ppsm
    
    # 2. Max Price Transaction
    max_resale_price = subset['resale_price'].max()
    
    # 3. Average Bala's Capital Retention
    avg_retention = subset['balas_retention_pct'].mean()

    # Layout metrics side-by-side in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Avg Price / SQM",
            value=f"${avg_ppsm:,.2f}",
            delta=f"${abs(ppsm_delta):,.2f} vs SG Avg",
            delta_color="normal" if ppsm_delta >= 0 else "inverse",
            help="Average resale price divided by floor area in square meters."
        )
        
    with col2:
        st.metric(
            label="Record High Transaction",
            value=f"${max_resale_price:,.0f}",
            help="The highest historical transacted resale price for this selection."
        )
        
    with col3:
        st.metric(
            label="Capital Retention (Bala's)",
            value=f"{avg_retention:.1f}%",
            help="SLA institutional standard for remaining leasehold value asset retention."
        )
else:
    st.warning("No data metrics available for this specific filtered subset.")
    
# --- 5. DATA VISUALIZATION: LEASE DECAY CURVE ---
st.markdown("---")
st.subheader("📉 Structural Lease Decay Curve vs Asset Valuation")

if not subset.empty:
    # Build an accessible, high-contrast scatter plot
    fig = px.scatter(
        subset, 
        x="remaining_lease_years", 
        y="price_per_sqm",
        color="flat_model",
        labels={
            "remaining_lease_years": "Remaining Lease (Years)",
            "price_per_sqm": "Price per SQM ($)",
            "flat_model": "Flat Model Architecture"
        },
        hover_data={
            "month": True,
            "resale_price": ":$,.0f",
            "floor_midpoint": True,
            "remaining_lease_years": ".1f"
        },
        color_discrete_sequence=px.colors.qualitative.Safe, # Color-vision accessible palette
        template="plotly_white"
    )
    
    # Add a clean aesthetic vertical line showing where the user's slider target is
    fig.add_vline(
        x=target_lease, 
        line_width=2, 
        line_dash="dash", 
        line_color="#E040FB", # Vivid purple for high visual contrast
        annotation_text=f" Target: {target_lease} Yrs",
        annotation_position="top right"
    )
    
    # Polish up layout dimensions and fonts
    fig.update_layout(
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please adjust filters to populate the decay graph matrix.")
    
# --- 7. AUTOMATED MACHINE LEARNING VALUATION ENGINE ---
st.markdown("---")
st.subheader("🤖 AI-Powered Fair Market Valuation Estimator")

# We train the model on the selected Town and Flat Type to capture hyper-local pricing
if len(subset) >= 10: # We need at least a few historical records to train accurately
    st.write("Configure the asset features below to calculate an instant AI price prediction:")
    
    # Create input fields for the user to define their target property
    val_col1, val_col2 = st.columns(2)
    with val_col1:
        input_lease = st.number_input("Target Remaining Lease (Years):", min_value=1.0, max_value=99.0, value=float(target_lease), step=1.0)
    with val_col2:
        input_floor = st.number_input("Target Floor Level (Midpoint):", min_value=1.0, max_value=50.0, value=float(target_floor), step=1.0)
        
    # Prepare features (X) and target variable (y)
    # We predict price_per_sqm because it scales naturally, then multiply by average flat size
  # Prepare features (X) and target variable (y)
    X_train = subset[['remaining_lease_years', 'floor_midpoint']]
    y_train = subset['price_per_sqm']
    
    # Initialize and train a Random Forest model (Bulletproof bounds)
    model_pipeline = Pipeline(steps=[
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    model_pipeline.fit(X_train, y_train)
    
    # Create a matching DataFrame for the prediction input
    input_data = pd.DataFrame(
        [[input_lease, input_floor]], 
        columns=['remaining_lease_years', 'floor_midpoint']
    )
    
    # Predict the Price per SQM safely
    predicted_ppsm = model_pipeline.predict(input_data)[0]
    
    # Calculate the total asset value based on average area size
    avg_area = subset['floor_area_sqm'].mean()
    predicted_total_price = predicted_ppsm * avg_area
    
    # Display the ML Valuation output in a premium summary banner
    st.markdown(f"""
    <div style="background-color:#1E1E2E; padding: 20px; border-radius: 10px; border-left: 5px solid #E040FB; margin-top: 15px;">
        <h4 style="color:#FFFFFF; margin:0;">🎯 Estimated Fair Market Valuation</h4>
        <h2 style="color:#E040FB; margin: 10px 0 5px 0;">${predicted_total_price:,.2f}</h2>
        <p style="color:#A9B1D6; margin:0; font-size: 0.9rem;">
            Based on a predicted benchmark of <strong>${predicted_ppsm:,.2f}/SQM</strong> evaluated against an average size of <strong>{avg_area:.1f} SQM</strong> for this property configuration.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.info("💡 Not enough localized data points available to train the AI valuation engine for this specific selection. Try broadening your filter options.")