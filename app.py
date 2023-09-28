import streamlit as st
import pandas as pd
from SimpleAutoML import SimpleAutoML  # Import your SimpleAutoML class from the previous code

# Define the Streamlit app
def main():
    st.title("SimpleAutoML with Streamlit")

    # Upload a dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(data.head())

        # Allow the user to select the target column
        target_column = st.selectbox("Select the target column:", data.columns)

        # Initialize SimpleAutoML
        automl = SimpleAutoML(data)

        # Load data and perform EDA
        automl.load_data(target_column)
        automl.handle_missing_values()
        automl.preprocess_data()
        automl.encode_data()
        automl.encode_target()

        # Visualize numerical and categorical columns
        automl.visualize_numerical()
        automl.visualize_categorical()

        # Split data into training and testing sets
        automl.split_data()

        # Allow the user to select the ML task (classification or regression)
        task = st.radio("Select the ML task:", ["classification", "regression"])

        # Automatically select appropriate models based on the task
        automl.select_models(task)

        # Train the selected models
        automl.train_models()

        # Evaluate the models
        results = automl.evaluate_models()

        # Allow the user to select and display model results
        selected_model = st.selectbox("Select a model:", list(results.keys()))
        st.subheader(f"{selected_model} Performance:")
        selected_scores = results[selected_model]
        for metric, score in selected_scores.items():
            st.write(f"{metric}: {score}")

if __name__ == "__main__":
    main()
