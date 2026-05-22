import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ACCESSIBILITY SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Singapore HDB Resale Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom accessible color palette (Color-vision deficiency friendly)
ACCESSIBLE_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

st.title("🏢 Singapore HDB Resale Price Analytics")
st.markdown("""
This interactive dashboard analyzes historical HDB resale transactions in Singapore. 
Use the sidebar filters to customize the data slice.
""")
st.write("---")

# -----------------------------------------------------------------------------
# 2. OPTIMIZED DATA LOADING PIPELINE (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading HDB transaction dataset...")
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    # Read dataset
    df = pd.read_csv(file_path)
    
    # Ensure standard string format transformations for categorical filters
    for col in ['town', 'flat_type', 'flat_model', 'storey_range']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
            
    # Parse chronological markers safely if present
    if 'month' in df.columns:
        df['year'] = df['month'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else x)
        
    return df

# Initialize data fetch
CSV_FILENAME = "hdb_resale_flats.csv"
raw_df = load_data(CSV_FILENAME)

# Fallback mechanism if dataset is missing
if raw_df is None:
    st.error(f"❌ Critical Error: Dynamic path verification failed. '{CSV_FILENAME}' not found at root layout level.")
    st.info("💡 Diagnostic Hint: Double-check your GitHub repository root folder to ensure the CSV file is uploaded alongside app.py.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & DYNAMIC FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Selection Panel")

# Dynamic Town Filter
all_towns = sorted(raw_df['town'].unique()) if 'town' in raw_df.columns else []
selected_towns = st.sidebar.multiselect(
    "Select Towns:", 
    options=all_towns, 
    default=all_towns[:3] if len(all_towns) >= 3 else all_towns,
    help="Select one or more HDB towns to analyze."
)

# Dynamic Flat Type Filter
all_flat_types = sorted(raw_df['flat_type'].unique()) if 'flat_type' in raw_df.columns else []
selected_flat_types = st.sidebar.multiselect(
    "Select Flat Types:", 
    options=all_flat_types, 
    default=all_flat_types,
    help="Filter by room size configuration."
)

# Filter Application Logic
filtered_df = raw_df.copy()
if selected_towns:
    filtered_df = filtered_df[filtered_df['town'].isin(selected_towns)]
if selected_flat_types:
    filtered_df = filtered_df[filtered_df['flat_type'].isin(selected_flat_types)]

# -----------------------------------------------------------------------------
# 4. EXECUTIVE SUMMARY METRICS BANNER
# -----------------------------------------------------------------------------
if not filtered_df.empty:
    total_transactions = len(filtered_df)
    avg_price = filtered_df['resale_price'].mean() if 'resale_price' in filtered_df.columns else 0.0
    avg_psf = (filtered_df['resale_price'] / filtered_df['floor_area_sqm']).mean() if 'floor_area_sqm' in filtered_df.columns and 'resale_price' in filtered_df.columns else 0.0
    
    # Layout 3-Column Metrics Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Transacted Units", value=f"{total_transactions:,}")
    with col2:
        st.metric(label="Average Resale Price", value=f"${avg_price:,.2f}")
    with col3:
        st.metric(label="Average Price Per SQM", value=f"${avg_psf:,.2f} / sqm")
else:
    st.warning("⚠️ No records match the selected filter combinations. Please widen your selection panel inputs.")
    st.stop()

st.write("---")

# -----------------------------------------------------------------------------
# 5. DATA VISUALIZATION PORTALS
# -----------------------------------------------------------------------------
layout_col1, layout_col2 = st.columns(2)

with layout_col1:
    st.subheader("📈 Resale Price Distribution by Town")
    if 'town' in filtered_df.columns and 'resale_price' in filtered_df.columns:
        fig_box = px.box(
            filtered_df,
            x='town',
            y='resale_price',
            color='town',
            color_discrete_sequence=ACCESSIBLE_PALETTE,
            title="Price Spread Across Selected Towns",
            labels={'town': 'HDB Town Location', 'resale_price': 'Resale Price (SGD)'}
        )
        fig_box.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(l=40, r=40, t=40, b=80)
        )
        st.plotly_chart(fig_box, use_container_width=True)

with layout_col2:
    st.subheader("📐 Floor Area vs Resale Price Correlation")
    if 'floor_area_sqm' in filtered_df.columns and 'resale_price' in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x='floor_area_sqm',
            y='resale_price',
            color='flat_type',
            color_discrete_sequence=ACCESSIBLE_PALETTE,
            opacity=0.6,
            title="Price vs. Floor Area (sqm) Trend",
            labels={'floor_area_sqm': 'Floor Area (Square Meters)', 'resale_price': 'Resale Price (SGD)', 'flat_type': 'Flat Type'}
        )
        fig_scatter.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. RAW COMPREHENSIVE DATA ENGINE VIEW
# -----------------------------------------------------------------------------
st.write("---")
st.subheader("📋 Filtered Transaction Ledger Data")
st.markdown("Below is the filtered subset data matrix based on your active sidebar filter selections.")

# Display clean interactive dataframe
st.dataframe(
    filtered_df,
    column_config={
        "resale_price": st.column_config.NumberColumn("Resale Price (SGD)", format="$%d"),
        "floor_area_sqm": st.column_config.NumberColumn("Floor Area (Sqm)", format="%d m²"),
        "lease_commence_date": st.column_config.NumberColumn("Lease Commence Year", format="%d")
    },
    use_container_width=True,
    hide_index=True
)