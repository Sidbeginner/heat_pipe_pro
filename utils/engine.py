import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class HeatPipeRecommendationEngine:
    def __init__(self, data_path='heat_pipe_dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.model_resistance = None
        self.model_max_temp = None
        self.nn_density = None
        self.design_catalog = None
        self.cat_encoders = {}
        
        # Operational and Design columns
        self.operational_cols = ['Heat Load (W)', 'Heat Flux (W/cm2)', 'Ambient Temperature', 'Cooling Method', 'Orientation']
        self.design_cols = ['Heat Pipe Length', 'Heat Pipe Diameter', 'Heat Pipe Width', 'Heat Pipe Thickness', 
                             'Evaporator Length', 'Condenser Length', 'Adiabatic Length', 'Working Fluid', 
                             'Heat Pipe Material', 'Wick Structure', 'Wick Material', 'Fill Ratio']

    def load_and_train(self):
        """Loads dataset, prepares data pipelines, and trains surrogate models."""
        print("[1/3] Loading and cleaning dataset...")
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.replace('Â²', '2').str.strip()
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Handle structural missingness in Flat vs Cylindrical geometries
        df['Heat Pipe Width'] = df['Heat Pipe Width'].fillna(0)
        df['Heat Pipe Thickness'] = df['Heat Pipe Thickness'].fillna(0)
        
        # Text normalization for categorical columns
        cat_cols = ['Working Fluid', 'Heat Pipe Material', 'Wick Structure', 'Wick Material', 'Orientation', 'Cooling Method']
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.title()
            # Fit an encoder per categorical column
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[col + '_enc'] = le.fit_transform(df[[col]])
            self.cat_encoders[col] = le
            
        self.df = df
        
        # Extract unique, valid physical designs to act as our product catalog
        print("[2/3] Extracting unique physical design catalog...")
        self.design_catalog = df[self.design_cols].drop_duplicates().reset_index(drop=True)
        
        # Prepare training data matrices for Surrogate Models
        features = [c + '_enc' if c in self.cat_encoders else c for c in self.operational_cols + self.design_cols]
        X = df[features]
        y_res = df['Thermal Resistance']
        y_temp = df['Maximum Temperature']
        
        print("[3/3] Training Machine Learning Surrogate Models (Digital Twins)...")
        # Random Forests handle non-linear thermodynamic interactions beautifully
        self.model_resistance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model_max_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.model_resistance.fit(X, y_res)
        self.model_max_temp.fit(X, y_temp)
        
        # Train a density estimator based on operational bounds to calculate an authentic Confidence Score
        # If user queries a heat load/flux far outside historical bounds, confidence drops.
        op_num_features = ['Heat Load (W)', 'Heat Flux (W/cm2)', 'Ambient Temperature']
        self.nn_density = NearestNeighbors(n_neighbors=5)
        self.nn_density.fit(df[op_num_features])
        self.max_historical_dist = np.max(self.nn_density.kneighbors(df[op_num_features])[0])
        
        print("Engine is fully compiled and ready for recommendation queries.\n")

    def _calculate_confidence(self, heat_load, heat_flux, ambient_temp):
        """Calculates distance from historical training envelope to generate a confidence score."""
        query = np.array([[heat_load, heat_flux, ambient_temp]])
        dist, _ = self.nn_density.kneighbors(query)
        mean_dist = np.mean(dist)
        
        # Map distance to a 0-100 percentage metric
        confidence = 100 * (1 - (mean_dist / (self.max_historical_dist * 1.5)))
        return round(max(min(confidence, 99.5), 5.0), 1)

    def recommend(self, heat_load, heat_source_area, ambient_temp, cooling_method, orientation, max_length, max_diameter):
        """Filters catalog space, simulates thermal performance, and returns the top 3 configurations."""
        heat_flux = heat_load / heat_source_area
        cooling_method = cooling_method.strip().title()
        orientation = orientation.strip().title()
        print("Available columns:", self.design_catalog.columns)
        # Keep the existing line 92 below this print
        # 1. Filter structural space constraints
        candidates = self.design_catalog[
            (self.design_catalog['Heat Pipe Length'] <= max_length) & 
            (self.design_catalog['Heat Pipe Diameter'] <= max_diameter)
        ].copy().reset_index(drop=True)
        
        # Fallback if space constraints are too restrictive
        if len(candidates) == 0:
            return "No configurations found within those maximum spatial dimensions. Please increase available length/diameter bounds."
            
        # 2. Re-synthesize matrix: Pair user environmental inputs with every valid candidate design
        for col in self.operational_cols:
            if col == 'Heat Load (W)': candidates[col] = heat_load
            elif col == 'Heat Flux (W/cm2)': candidates[col] = heat_flux
            elif col == 'Ambient Temperature': candidates[col] = ambient_temp
            elif col == 'Cooling Method': candidates[col] = cooling_method
            elif col == 'Orientation': candidates[col] = orientation

        # Encode input matrices for ML evaluation
        eval_candidates = candidates.copy()
        for col, encoder in self.cat_encoders.items():
            eval_candidates[col + '_enc'] = encoder.transform(eval_candidates[[col]])
            
        features_order = [c + '_enc' if c in self.cat_encoders else c for c in self.operational_cols + self.design_cols]
        X_eval = eval_candidates[features_order]
        
        # 3. Deploy digital twin models for performance predictions
        candidates['Predicted_Thermal_Resistance'] = self.model_resistance.predict(X_eval)
        candidates['Predicted_Max_Temperature'] = self.model_max_temp.predict(X_eval)
        
        # Calculate context-aware confidence score
        confidence_score = self._calculate_confidence(heat_load, heat_flux, ambient_temp)
        candidates['Confidence_Score'] = f"{confidence_score}%"
        
        # Determine product type tag based on physical profile properties
        candidates['Heat Pipe Type'] = np.where(candidates['Heat Pipe Width'] > 0, 'Flat Heat Pipe', 'Cylindrical Heat Pipe')
        
        # 4. Thermodynamic Ranking Matrix: Lower Resistance and lower delta temperatures are superior
        # Sorting priority: Minimizing thermal resistance is the gold standard for heat pipe performance
        candidates = candidates.sort_values(by=['Predicted_Thermal_Resistance', 'Predicted_Max_Temperature'], ascending=[True, True])
        
        # Extract Top 3 Unique Recommendations
        top_3 = candidates.head(3).copy().reset_index(drop=True)
        
        # Formatted Output Presentation Generation
        results = []
        for idx, row in top_3.iterrows():
            rec_dict = {
                "Rank": idx + 1,
                "Heat Pipe Type": row['Heat Pipe Type'],
                "Length (mm)": round(row['Heat Pipe Length'], 1),
                "Diameter (mm)": round(row['Heat Pipe Diameter'], 1),
                "Material": row['Heat Pipe Material'],
                "Working Fluid": row['Working Fluid'],
                "Wick Structure": row['Wick Structure'],
                "Fill Ratio (%)": f"{round(row['Fill Ratio'], 1)}%",
                "Predicted Max Temp (°C)": round(row['Predicted_Max_Temperature'], 2),
                "Predicted Thermal Resistance (°C/W)": round(row['Predicted_Thermal_Resistance'], 4),
                "Confidence Score": row['Confidence_Score']
            }
            results.append(rec_dict)
            
        return results

# --- EXECUTION & TESTING EXAMPLE ---
if __name__ == "__main__":
    # Initialize engine object
    engine = HeatPipeRecommendationEngine()
    
    # 1. Train internal surrogate frameworks using the cleaned workspace file
    # (Assuming the workspace file is available in your active working directory)
    try:
        engine.load_and_train()
        
        # 2. Simulate User Query Constraints
        recommendations = engine.recommend(
            heat_load=120.0,            # 120 Watts
            heat_source_area=6.0,       # 6.0 cm² area (Flux = 20 W/cm²)
            ambient_temp=25.0,          # 25°C Ambient Air
            cooling_method='Forced Convection',
            orientation='Horizontal',
            max_length=200.0,           # Max allowable length 200mm
            max_diameter=8.0            # Max allowable diameter 8mm
        )
        
        # 3. Print out configurations cleanly
        print("="*60)
        print("          TOP 3 RECOMMENDED HEAT PIPE CONFIGURATIONS        ")
        print("="*60)
        for rec in recommendations:
            print(f"\n[RECOMMENDATION RANK #{rec['Rank']}] - {rec['Heat Pipe Type']}")
            print(f"  • Dimensions: {rec['Length (mm)']} mm Length × {rec['Diameter (mm)']} mm Diameter")
            print(f"  • Core Design: {rec['Material']} Casing | {rec['Working Fluid']} Fluid | {rec['Wick Structure']} Wick")
            print(f"  • Charge Ratio: {rec['Fill Ratio (%)']}")
            print(f"  • Performance: Predicted Resistance = {rec['Predicted Thermal Resistance (°C/W)']} °C/W")
            print(f"  • Thermal Envelope: Expected Max Junction Temp = {rec['Predicted Max Temp (°C)']} °C")
            print(f"  • Recommendation Confidence: {rec['Confidence Score']}")
            print("-" * 50)
            
    except FileNotFoundError:
        print("Error: Please make sure 'heat_pipe_dataset.csv' is present in the working directory before initializing.")
