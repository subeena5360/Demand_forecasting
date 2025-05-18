import streamlit as st
import pandas as pd
import numpy as np
import joblib,pickle
import traceback
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(df, encoder=None, feature_selector=None, fit_models=False):
    """Complete preprocessing with datetime preservation"""
    try:
        # Remove unnecessary columns (preserve Year and Month)
        cols_to_drop = [
            'Delivery Status', 'Late_delivery_risk', 'shipping date (DateOrders)',
            'Benefit per order', 'Sales per customer', 'Category Id', 'Order Profit Per Order',
            'Order Item Discount', 'Order Item Total', 'Order Status', 'Customer Email',
            'Customer Password', 'Latitude', 'Longitude', 'Product Description',
            'Product Image', 'Customer Fname', 'Customer Id', 'Customer Lname',
            'Department Id', 'Order Customer Id', 'Order Item Cardprod Id',
            'Order Item Id', 'Product Card Id', 'Product Category Id', 'Order Id',
            'Customer Street', 'Customer Zipcode', 'Order Zipcode',
            'Order Item Product Price', 'Order Item Profit Ratio'
        ]
        data = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Convert and extract datetime features
        if 'order date (DateOrders)' in data.columns:
            data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])
            data = data.rename(columns={"order date (DateOrders)": "order_date"})
            data['Year'] = data['order_date'].dt.year
            data['Month'] = data['order_date'].dt.month
            data.sort_values(by='order_date', inplace=True)
            data.drop(columns=['order_date'], inplace=True)

        # Ensure Year and Month exist
        if 'Year' not in data.columns:
            raise ValueError("Year column is missing in input data")
        if 'Month' not in data.columns:
            raise ValueError("Month column is missing in input data")

        # Grouping and aggregation - preserve Year and Month
        group_cols = ['Year', 'Market', 'Month', 'Order Country', 'Order Region',
                     'Order State', 'Product Name', 'Category Name', 'Customer Segment']
        grouped_data = data.groupby([col for col in group_cols if col in data.columns], as_index=False)
        
        grouped_data_df = grouped_data["Order Item Quantity"].sum()
        for col, agg_func in [("Sales", "sum"), ("Days for shipping (real)", "mean"),
                      ("Product Price", "mean"), ("Product Status", "mean")]:
            if col in data.columns:
                grouped_agg = grouped_data.agg({col: agg_func})
                grouped_data_df = grouped_data_df.merge(grouped_agg, on=group_cols, how='left')

        # One-hot encoding
        categorical_cols = ['Market', 'Order Country', 'Order Region', 'Order State',
                          'Product Name', 'Category Name', 'Customer Segment']
        categorical_cols = [col for col in categorical_cols if col in grouped_data_df.columns]
        
        for col in categorical_cols:
            grouped_data_df[col] = grouped_data_df[col].astype(str)

        if fit_models:
            encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            encoder.fit(grouped_data_df[categorical_cols])
            encoded_data = encoder.transform(grouped_data_df[categorical_cols])
        else:
            encoded_data = encoder.transform(grouped_data_df[categorical_cols])
        
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=grouped_data_df.index
        )

        # Feature selection only during training
        if fit_models:
            feature_selector = RandomForestRegressor(random_state=42)
            feature_selector.fit(encoded_df, grouped_data_df["Order Item Quantity"])
            selected_features = get_selected_features_columns(encoded_df, feature_selector)
            encoded_df = encoded_df[selected_features]

        # Combine features while preserving Year and Month
        final_df = pd.concat([
            grouped_data_df[['Year', 'Month'] + [col for col in grouped_data_df.columns 
                                               if col not in categorical_cols and col != 'Order Item Quantity']],
            encoded_df
        ], axis=1)

        return final_df, encoder, feature_selector

    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        print(traceback.format_exc())
        return None, None, None
def get_selected_features_columns(X, feature_selector, top_k=1400):
    """Safe feature selection with bounds checking"""
    feature_importances = feature_selector.feature_importances_
    
    selected_indices = np.argsort(feature_importances)[1400:]
    return X.columns[selected_indices]

def predict_quantity(df, scaler, encoder, feature_selector, prediction_model, model_feature_names):
    """Final robust prediction function"""
    try:
        processed_data, _, _ = preprocess_data(
            df,
            encoder=encoder,
            feature_selector=None,
            fit_models=False
        )
        
        # Feature alignment
        
        with open("artifacts/data_transformation/scaler.pkl", 'rb') as f:
            scaler_data = pickle.load(f)

        
        # Ensure all features exist in processed_data, add any missing as zeros
        target_col = "Order Item Quantity"  # or whatever your target column name is
        if target_col in processed_data.columns:
            print("hi")
            processed_data = processed_data.drop(columns=[target_col])
        scaler = scaler_data['scaler']
        scaler_features = scaler_data['features']
        for feature in scaler_features:
            if feature not in processed_data.columns:
                processed_data[feature] = 0

# Keep only scaler_features columns and order correctly
        print(processed_data.columns[processed_data.columns.duplicated()])
        processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]
        processed_data = processed_data.reindex(columns=scaler_features, fill_value=0)
        print("Scaler expected features:",len(scaler_features))
        print("Processed data columns before selection:",len(processed_data.columns.tolist()))
        extra = set(processed_data.columns) - set(scaler_features)
        missing = set(scaler_features) - set(processed_data.columns)
        print("Extra columns in processed_data:", extra)
        print("Missing columns in processed_data:", missing)

        X_scaled = scaler.transform(processed_data)
        feature_to_index = {name: i for i, name in enumerate(scaler_features)}
        model_feature_indices = [feature_to_index[name] for name in model_feature_names 
                               if name in feature_to_index]
        X_final = X_scaled[:, model_feature_indices]
        y_pred = prediction_model.predict(X_final)
        
        # Safely create datetime
        datetime_data = pd.DataFrame({
            'Year': processed_data['Year'] if 'Year' in processed_data.columns else df['Year'],
            'Month': processed_data['Month'] if 'Month' in processed_data.columns else df['Month']
        }).assign(DAY=1)
        
        result_df = pd.DataFrame({
            "datetime": pd.to_datetime(datetime_data).dt.strftime('%Y-%m'),
            "predicted": y_pred
        })
        
        return result_df.groupby("datetime").sum().reset_index()
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return None

# [Rest of the code (get_selected_features_columns, main) remains the same]

def main():
    # Configure page
    st.set_page_config(page_title="Demand Forecast", layout="wide")
    st.title("🚀 Supply Chain Demand Forecasting")
    
    # File upload section
    with st.sidebar:
        st.header("📤 Data Upload")
        upload_file = st.file_uploader(
            "Upload your sales data (CSV or Excel)",
            type=['csv', 'xlsx'],
            help="File should contain historical order data"
        )
    
    if upload_file is not None:
        try:
            # Load data
            with st.spinner("Loading data..."):
                try:
                    if upload_file.name.endswith(".csv"):
                        raw_data = pd.read_csv(upload_file, encoding="ISO-8859-1")
                    else:
                        raw_data = pd.read_excel(upload_file)
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to read file: {str(e)}")
                    return

            # Load models
            with st.spinner("Loading models..."):
                try:
                    encoder = joblib.load("artifacts/data_transformation/onehot_encoder.pkl")
                    scaler = joblib.load("artifacts/data_transformation/scaler.pkl")
                    feature_selector = joblib.load("artifacts/data_transformation/rfr.pkl")
                    model_data = joblib.load("artifacts/model_trainer/model.joblib")
                    model = model_data['model']
                    feature_names = model_data['feature_names']
                    st.success("Models loaded successfully!")
                except Exception as e:
                    st.error(f"Model loading failed: {str(e)}")
                    return

            # Make predictions
            with st.spinner("Processing predictions..."):
                result_df = predict_quantity(
                    raw_data,
                    scaler,
                    encoder,
                    feature_selector,
                    model,
                    feature_names
                )
                
                if result_df is None:
                    st.error("Prediction failed")
                    return
                
                st.success("Predictions completed!")

            # Display results
            st.header("📊 Forecast Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Monthly Summary")
                st.dataframe(result_df.style.format({"predicted": "{:.0f}"}))
            
            with col2:
                st.subheader("Trend Visualization")
                fig = px.line(
                    result_df, 
                    x="datetime", 
                    y="predicted",
                    title="Monthly Demand Forecast",
                    labels={"predicted": "Predicted Quantity", "datetime": "Month"}
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()