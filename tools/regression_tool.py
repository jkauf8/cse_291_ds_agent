import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def run_regression(df: pd.DataFrame, target_column: str, feature_columns: list = None):
    """
    Runs a Random Forest Regression on the given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target variable column.
        feature_columns (list, optional): A list of feature column names. 
                                          If None, all columns except the target are used. 
                                          Defaults to None.

    Returns:
        dict: A dictionary containing the R2 score, feature importances,
              and a brief summary of the results.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])

    if feature_columns:
        if not all(col in df.columns for col in feature_columns):
            raise ValueError("One or more feature columns are not in the DataFrame.")
        X = df[feature_columns]
    else:
        X = df.drop(columns=[target_column])

    y = df[target_column]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the regression pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)

    # Get feature importances
    try:
        # Get feature names after preprocessing
        importances = model.named_steps['regressor'].feature_importances_
        
        # Build feature names list
        all_feature_names = []
        
        # Add numerical feature names
        all_feature_names.extend(numerical_features.tolist())
        
        # Add categorical feature names (after one-hot encoding)
        if len(categorical_features) > 0:
            ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_feature_names.extend(ohe_feature_names.tolist())
        
        # Create dictionary of feature importances with actual names
        feature_importance_dict = dict(zip(all_feature_names, importances))
        
        # Sort feature importances
        sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    except Exception as e:
        # Fallback if getting feature names fails - use original feature names
        importances = model.named_steps['regressor'].feature_importances_
        # Use the original X column names as fallback
        feature_names = X.columns.tolist()
        if len(feature_names) == len(importances):
            feature_importance_dict = dict(zip(feature_names, importances))
        else:
            feature_importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)


    summary = (
        f"Random Forest Regression Results:\n"
        f"R^2 Score: {r2:.4f}\n\n"
        f"Top 5 Most Important Features:\n"
    )
    for feature, importance in sorted_feature_importance[:5]:
        summary += f"- {feature}: {importance:.4f}\n"


    return {
        "r2_score": r2,
        "feature_importances": sorted_feature_importance,
        "summary": summary
    }
