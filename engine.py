# engine.py
import pandas as pd
import numpy as np

class RealEstateEngine:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_and_clean_data(self) -> pd.DataFrame:
        """Loads HDB resale data, fixes missing values, and engineers metrics."""
        df = pd.read_csv(self.filepath)
        
        # 1. Extract transaction year from 'YYYY-MM' format
        df['transaction_year'] = df['month'].str.split('-').str[0].astype(int)
        
        # 2. Reconstruct remaining lease column accurately
        def calculate_remaining_lease(row):
            if pd.notna(row['remaining_lease']):
                val = str(row['remaining_lease'])
                if 'years' in val:
                    return float(val.split('years')[0].strip())
                try:
                    return float(val)
                except ValueError:
                    pass
            # Mathematical fallback calculation
            return float(99 - (row['transaction_year'] - row['lease_commence_date']))

        df['remaining_lease_years'] = df.apply(calculate_remaining_lease, axis=1)
        
        # 3. Calculate Normalized Valuation Metric
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
        
        self.df = df
        return df

    def generate_valuation_matrix(self, town: str, flat_type: str) -> pd.DataFrame:
        """Generates statistical valuation baselines for predictive indexing."""
        filtered = self.df[(self.df['town'] == town) & (self.df['flat_type'] == flat_type)]
        return filtered
    # --- STEP 1: EXTRACT FLOOR MIDPOINT ---
    def clean_storey(storey_str):
        if pd.isna(storey_str):
            return 1  # Default fallback
        try:
            # Split "01 TO 03" into ['01', '03']
            parts = str(storey_str).upper().split(' TO ')
            if len(parts) == 2:
                low = int(parts[0])
                high = int(parts[1])
                return (low + high) / 2
            return int(parts[0])
        except Exception:
            return 1

        # Apply the floor conversion to the dataframe
        df['floor_midpoint'] = df['storey_range'].apply(clean_storey)
        
        # --- STEP 2: CALCULATE BALA'S CURVE RETENTION PERCENTAGE ---
        # Calculates how much of the original 99-year asset value is theoretically retained
        df['balas_retention_pct'] = 100 * (1 - (0.992 ** df['remaining_lease_years']))

        # Calculate baseline "Fresh Lease" Price per SQM (adjusted for lease age)
        # This lets you see what the flat would be worth if it had a fresh 99-year lease
        df['lease_adjusted_ppsm'] = df['price_per_sqm'] / (df['balas_retention_pct'] / 100) 