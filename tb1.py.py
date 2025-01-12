import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle

# --- Data Classes and Constants ---
class NAT2Variants:
    # Key NAT2 SNPs
    SNPs = ['rs1799929', 'rs1799930', 'rs1799931']
    
    # NAT2 haplotype definitions
    HAPLOTYPES = {
        'NAT2*4': {'rs1799929': 'CC', 'rs1799930': 'GG', 'rs1799931': 'GG'},  # Wild type - Rapid
        'NAT2*5': {'rs1799929': 'TT', 'rs1799930': 'GG', 'rs1799931': 'GG'},  # Slow
        'NAT2*6': {'rs1799929': 'CC', 'rs1799930': 'AA', 'rs1799931': 'GG'},  # Slow
        'NAT2*7': {'rs1799929': 'CC', 'rs1799930': 'GG', 'rs1799931': 'AA'}   # Slow
    }

class DoseGuidelines:
    RECOMMENDATIONS = {
        'Rapid': {
            'dose': 'Standard dosing (5 mg/kg/day)',
            'risk': 'Low risk of toxicity',
            'monitoring': 'Regular monitoring'
        },
        'Intermediate': {
            'dose': 'Standard dosing (5 mg/kg/day)',
            'risk': 'Moderate risk - monitor closely',
            'monitoring': 'Enhanced monitoring'
        },
        'Slow': {
            'dose': 'Reduced dosing (3-4 mg/kg/day)',
            'risk': 'High risk of toxicity',
            'monitoring': 'Frequent monitoring required'
        }
    }

# --- Data Processing Functions ---
def create_sample_data():
    """Create sample datasets for demonstration"""
    # NAT2 variant data
    nat2_variants = pd.DataFrame({
        'sample_id': range(1, 11),
        'rs1799929': ['CC', 'CT', 'TT', 'CC', 'CT', 'CC', 'TT', 'CT', 'CC', 'CT'],
        'rs1799930': ['GG', 'GA', 'AA', 'GG', 'GA', 'GG', 'AA', 'GA', 'GG', 'GA'],
        'rs1799931': ['GG', 'GA', 'GG', 'GA', 'GG', 'GA', 'GG', 'GA', 'GG', 'GA']
    })
    
    # Phenotype data
    phenotype_data = pd.DataFrame({
        'sample_id': range(1, 11),
        'acetylator_status': ['Rapid', 'Intermediate', 'Slow', 'Rapid', 'Intermediate',
                             'Rapid', 'Slow', 'Intermediate', 'Rapid', 'Intermediate'],
        'drug_response': ['Normal', 'Normal', 'Adverse', 'Normal', 'Normal',
                         'Normal', 'Adverse', 'Normal', 'Normal', 'Normal']
    })
    
    return nat2_variants, phenotype_data

def encode_variants(variants_df):
    """Convert genotype data to numerical features"""
    encoded_df = pd.DataFrame()
    
    for snp in NAT2Variants.SNPs:
        # Create dummy variables for each genotype
        dummies = pd.get_dummies(variants_df[snp], prefix=snp)
        encoded_df = pd.concat([encoded_df, dummies], axis=1)
    
    return encoded_df

def predict_acetylator_status(variants):
    """Predict acetylator status based on NAT2 variants"""
    # Simple rule-based prediction
    slow_variants = 0
    
    if variants['rs1799929'] in ['CT', 'TT']:
        slow_variants += 1
    if variants['rs1799930'] in ['GA', 'AA']:
        slow_variants += 1
    if variants['rs1799931'] in ['GA', 'AA']:
        slow_variants += 1
    
    if slow_variants >= 2:
        return 'Slow'
    elif slow_variants == 1:
        return 'Intermediate'
    else:
        return 'Rapid'

def get_dosing_recommendation(acetylator_status):
    """Get dosing recommendations based on acetylator status"""
    return DoseGuidelines.RECOMMENDATIONS[acetylator_status]

# --- Machine Learning Model ---
def train_model(X, y):
    """Train a Random Forest classifier"""
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

# --- Streamlit Web Interface ---
def create_web_interface():
    st.title("TB Drug Response Predictor")
    st.write("Analysis of NAT2 variants for Isoniazid response")
    
    # Create tabs for different functions
    tab1, tab2 = st.tabs(["Single Patient Analysis", "Batch Analysis"])
    
    with tab1:
        st.header("Individual Patient Analysis")
        
        # Input fields for SNPs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rs1799929 = st.selectbox(
                "rs1799929 genotype",
                options=['CC', 'CT', 'TT']
            )
        
        with col2:
            rs1799930 = st.selectbox(
                "rs1799930 genotype",
                options=['GG', 'GA', 'AA']
            )
        
        with col3:
            rs1799931 = st.selectbox(
                "rs1799931 genotype",
                options=['GG', 'GA', 'AA']
            )
        
        if st.button("Analyze"):
            # Create patient data
            patient_variants = {
                'rs1799929': rs1799929,
                'rs1799930': rs1799930,
                'rs1799931': rs1799931
            }
            
            # Predict acetylator status
            status = predict_acetylator_status(patient_variants)
            
            # Get recommendations
            recommendations = get_dosing_recommendation(status)
            
            # Display results
            st.subheader("Results:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Predicted Acetylator Status: {status}")
                st.write(f"Recommended Dose: {recommendations['dose']}")
            
            with col2:
                st.warning(f"Risk Assessment: {recommendations['risk']}")
                st.write(f"Monitoring: {recommendations['monitoring']}")
    
    with tab2:
        st.header("Batch Analysis")
        uploaded_file = st.file_uploader("Upload CSV file with variant data", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if st.button("Analyze Batch"):
                results = []
                for _, row in df.iterrows():
                    status = predict_acetylator_status(row)
                    recommendations = get_dosing_recommendation(status)
                    results.append({
                        'Sample_ID': row.get('sample_id', 'Unknown'),
                        'Acetylator_Status': status,
                        'Recommended_Dose': recommendations['dose'],
                        'Risk_Level': recommendations['risk']
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="batch_analysis_results.csv",
                    mime="text/csv"
                )

def main():
    # Create sample data
    variants_df, phenotype_df = create_sample_data()
    
    # Save sample data
    variants_df.to_csv('nat2_variants.csv', index=False)
    phenotype_df.to_csv('phenotype_data.csv', index=False)
    
    # Launch web interface
    create_web_interface()

if __name__ == "__main__":
    main()
