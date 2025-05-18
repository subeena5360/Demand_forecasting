from src import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from src.entity.config_entity import DataTransformationConfig
import os
import pickle


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self):
        df = pd.read_csv(self.config.data_dir,encoding = "ISO-8859-1")
        # removing the unnecessary columns
        data = df.drop(columns=[
                                  'Delivery Status', 'Late_delivery_risk', 
                                  'shipping date (DateOrders)', 'Benefit per order', 'Sales per customer', 'Category Id',
                                  'Order Profit Per Order', 'Order Item Discount', 'Order Item Total', 'Order Status', 
                                  'Customer Email', 'Customer Password', 'Latitude', 'Longitude', 'Product Description', 'Product Image',
                                  'Customer Fname', 'Customer Id', 'Customer Lname', 'Department Id',
                                  'Order Customer Id', 'Order Item Cardprod Id', 'Order Item Id',
                                  'Product Card Id', 'Product Category Id', 'Order Id', 'Customer Street',
                                  'Customer Zipcode', 'Order Zipcode', 'Order Item Product Price',
                                   'Order Item Profit Ratio'])
        # converting into date time
        data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])
        data = data.rename(columns={"order date (DateOrders)":"order_date"})
        # manipulated year and month column
        data['Year'] = data['order_date'].dt.year
        data['Month'] = data['order_date'].dt.month
        data.sort_values(by='order_date', inplace=True)
        data.drop(columns=['order_date'], inplace=True)
        # grouping the data 
        grouped_data = data.groupby(['Year', 'Market', 'Month', 'Order Country',
                                        'Order Region', 'Order State', 'Product Name', 
                                         'Category Name','Customer Segment'])
        grouped_data_1 = grouped_data["Order Item Quantity"].sum().reset_index()
        grouped_data_2 = grouped_data["Sales"].sum().reset_index()
        grouped_data_3 = grouped_data["Days for shipping (real)"].mean().reset_index()
        grouped_data_4 = grouped_data["Product Price"].mean().reset_index()
        grouped_data_5 = grouped_data["Product Status"].mean().reset_index()
        grouped_data_df = grouped_data_1.copy()
        grouped_data_df["Sales"] = grouped_data_2["Sales"]
        grouped_data_df["Days for shipping (real)"] = grouped_data_3["Days for shipping (real)"]
        grouped_data_df["Product Price"] = grouped_data_4["Product Price"]
        grouped_data_df["Product Status"] = grouped_data_5["Product Status"]
        print(grouped_data_df[['Market', 'Order Country', 'Order Region', 'Order State', 'Product Name', 'Category Name', 'Customer Segment']].dtypes)
        categorical_cols=['Market', 'Order Country', 'Order Region', 'Order State', 'Product Name', 'Category Name', 'Customer Segment']
        for col in categorical_cols:
            grouped_data_df[col] = grouped_data_df[col].astype(str)
        # one hot encoding
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoder.fit(grouped_data_df[categorical_cols])

        # Transform categorical data
        encoded_data = encoder.transform(grouped_data_df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_cols) ) # Get proper column names
        numerical_data = grouped_data_df.drop(columns=categorical_cols)

        # Combine encoded + numerical data
        final_df = pd.concat([numerical_data, encoded_df], axis=1)
        # grouped_data_df = pd.get_dummies(grouped_data_df, columns = ['Market', 'Order Country',
        #                                 'Order Region', 'Order State', 'Product Name', 
        #                                  'Category Name','Customer Segment'],drop_first=True, dtype=int)
        final_df.to_csv(self.config.transformed_data, index=False)
        logger.info("grouping of data ........")
        with open(self.config.one_hot_encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        logger.info("saving one hot model ........")
        print(final_df.shape)
    
    def one_hot_preprocess(self):
        df = pd.read_csv(self.config.transformed_data, encoding="ISO-8859-1")
        x = df.drop(columns=['Order Item Quantity'])
        y = df['Order Item Quantity']

        # Save the feature list before scaling
        original_features = x.columns.tolist()

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(x)
        scaled_X = pd.DataFrame(scaled_X, columns=original_features)

        # Drop all-zero columns
        scaled_X = scaled_X.loc[:, (scaled_X != 0).any(axis=0)]

        logger.info("One-hot encoding and scaling completed.")

        # Feature selection with RandomForest
        rfr = RandomForestRegressor(random_state=42)
        rfr.fit(scaled_X, y)
        selected_ind = np.argsort(rfr.feature_importances_)[1400:]  # select top features
        selected_features = scaled_X.columns[selected_ind]
        X_selected = scaled_X[selected_features]
        X_selected['Order Item Quantity'] = y.values

        X_selected.to_csv(self.config.preprocessed_dir, index=False)
        print(X_selected.shape)

        # Save scaler + feature list
        with open(self.config.scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'features': original_features}, f)
        logger.info("Scaler and feature list saved.")

        # Save the RandomForest model
        with open(self.config.model_feature_path, 'wb') as f:
            pickle.dump(rfr, f)
        logger.info("Random forest model for feature importance saved.")

 


    def train_test_spliting(self):
        data = pd.read_csv(self.config.preprocessed_dir)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)