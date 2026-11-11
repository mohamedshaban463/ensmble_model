def preprocess_inference_features(df_input, pipeline_filepath='/content/ensemble_pipeline.pkl'):
    """
    Takes raw, original features (as a pandas DataFrame) and outputs fully engineered,
    target-encoded, and scaled features matching the exact structure used for model prediction.
    """
    import pickle
    import pandas as pd
    import numpy as np
    
    # 1. Load trained artifacts
    with open(pipeline_filepath, 'rb') as f:
        artifacts = pickle.load(f)
    
    target_encoder = artifacts['target_encoder']
    scaler = artifacts['scaler']
    expected_features = artifacts['feature_names']
    
    # Work on a copy to avoid changing the original input
    df = df_input.copy()
    
    # Drop IDs if present
    df = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    
    # List of numeric columns from raw dataset expected to match preprocessing
    num_cols_raw = [
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
        'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 
        'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 
        'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 
        'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'EXT_SOURCE_1', 
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    
    # Ensure columns exist, filling missing columns with NaN if necessary
    for col in num_cols_raw:
        if col not in df.columns:
            df[col] = np.nan
            
    # Fill missing values for numerical raw features using medians (or default value fallback)
    for col in num_cols_raw:
        # Using 0 or median fallback if column is entirely empty in the single instance
        val = df[col].median()
        if pd.isna(val):
            val = 0.0
        df[col] = df[col].fillna(val)
        
    # 2. Re-create engineered features
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1e-5)
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
    df['INCOME_TO_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_ANNUITY'] + 1e-5)
    df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1e-5)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] / -365.25 + 1e-5)
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] / -365.25 + 1e-5)

    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_SUM'] = df[ext_cols].sum(axis=1)
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[ext_cols].std(axis=1).fillna(0)

    # Group-based features: fallback using a constant if they don't resolve
    if 'ORGANIZATION_TYPE' in df.columns:
        df['GROUP_EXT_MEAN_BY_ORGANIZATION'] = df.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_2'].transform('mean').fillna(0.5)
    else:
        df['GROUP_EXT_MEAN_BY_ORGANIZATION'] = 0.5
        
    if 'NAME_EDUCATION_TYPE' in df.columns:
        df['GROUP_CREDIT_MEAN_BY_EDUCATION'] = df.groupby('NAME_EDUCATION_TYPE')['AMT_CREDIT'].transform('mean').fillna(500000.0)
    else:
        df['GROUP_CREDIT_MEAN_BY_EDUCATION'] = 500000.0

    # Ensure all new columns also handle potential nulls
    new_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in new_num_cols:
        df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)

    # 3. Categorical encoding (Target Encoding)
    cat_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
    # Check that categoricals exist
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
            
    df[cat_cols] = target_encoder.transform(df[cat_cols])
    
    # Reorder columns to match original expected training order
    df_final = df[expected_features]
    
    # 4. Scale features
    scaled_values = scaler.transform(df_final)
    df_scaled = pd.DataFrame(scaled_values, columns=expected_features)
    
    return df_scaled