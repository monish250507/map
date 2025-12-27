import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from .train_utils import load_dataset, save_model, load_model
import logging

logger = logging.getLogger(__name__)



class RouteRiskDetector:
    def __init__(self):
        self.model_filename = "route_risk_isoforest.pkl"
        self.model = None

    def preprocess(self, df_routes: pd.DataFrame, df_events: pd.DataFrame = None) -> pd.DataFrame:
        """
        Derive risk features per route.
        """
        # If we have event logs, we can compute variance etc.
        # Assuming df_routes has summary stats already or we compute them.
        
        # Features needed: travel_time_variance, failure_events, congestion_spikes
        # We might need to construct these if not present.
        
        features = ['travel_time_variance', 'failure_count', 'avg_delay_min', 'congestion_level']
        
        # Ensure cols exist
        for f in features:
            if f not in df_routes.columns:
                df_routes[f] = 0.0 # Default to low risk
                
        # Fill NA
        df_routes = df_routes.fillna(0)
        
        return df_routes[features]

    def train(self):
        logger.info("Starting Route Risk Detector training...")
        
        df = load_dataset("main_agent_route_manager_india.xls")
        if df.empty:
            logger.error("Route risk training failed: no data.")
            return

        # Preprocess
        X = self.preprocess(df)
        
        # Isolation Forest
        # contamination='auto' or low value depending on anomaly rate
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X)
        
        self.model = iso
        save_model(iso, self.model_filename)
        logger.info("Route Risk Detector trained.")

    def detect_risk(self, route_stats: dict) -> float:
        """
        Returns an anomaly score between 0 and 1.
        Higher is more anomalous/risky.
        """
        if self.model is None:
            self.model = load_model(self.model_filename)
            if self.model is None:
                return 0.0

        try:
            df = pd.DataFrame([route_stats])
            X = self.preprocess(df)
            
            # decision_function returns score where lower is more abnormal (negative)
            # We want to map this to [0, 1] probability-like score
            score = self.model.decision_function(X)[0]
            
            # Map: score > 0 -> normal, score < 0 -> abnormal
            # We invert and normalize roughly
            # Typical range might be [-0.5, 0.5]
            risk = 0.5 - score 
            return float(np.clip(risk, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Risk detection failed: {e}")
            return 0.0

if __name__ == "__main__":
    detector = RouteRiskDetector()
    detector.train()
