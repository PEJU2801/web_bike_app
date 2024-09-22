import datetime
import streamlit as st
from scripts.model_per_counter import (
    get_train_data,
    get_test_data,
    encode_train_data,
    display_predictions,
    Feature_Engineering,
    render_counters,
)


# Function to display the welcome page
def welcome_page():
    st.markdown(
        """
        <h1 style='text-align: center; font-weight: bold;'>Paris on Wheels: Analyzing and Predicting Bike Traffic Trends</h1>
        """,
        unsafe_allow_html=True,
    )
    st.image("scripts/images/bike_picture.jpg", use_column_width=True)

    st.markdown(
        """
        This project aims to **predict the number of bikes** passing through a specific counter at a given time in Paris using historical data and machine learning techniques.        
        - **Map Page**: Here, you can see the **Paris map** and analyze the affluences at different counters during different hours of the day.
        - **Predictions Page**: On this page, you can find the predicted number of bikers versus the actual number of bikes.
    """
    )


# Main app logic
def main():
    # Read train & test data
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()

    # Encode train data
    X_train, y_train, le = encode_train_data(X_train, y_train)

    # Get counter names
    counters = [*le.classes_]

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Welcome", "Map", "Predictions"])

    if page == "Welcome":
        welcome_page()

    elif page == "Predictions":
        # Sidebar for counter selection
        st.sidebar.header("Filter by Counter")
        selected_counter = st.sidebar.selectbox("Select a Counter", counters)

        # Display predictions for the selected counter
        display_predictions(X_test, y_test, le, selected_counter)

    elif page == "Map":
        st.sidebar.header("Filter Map by:")
        st.markdown(
            """
        <h1 style='text-align: center; font-weight: bold;'>Paris Interactive Map</h1>
        """,
            unsafe_allow_html=True,
        )

        selected_date = st.sidebar.date_input(
            "Select Date",
            value=datetime.date(2020, 9, 2),  # Default value
            min_value=datetime.date(2020, 9, 2),  # Minimum allowed date
            max_value=datetime.date(2021, 8, 9),  # Maximum allowed date
        )
        selected_hour = st.sidebar.slider("Select Hour of Day", 0, 23, step=1)

        # Render the filtered map
        render_counters(selected_date, selected_hour)


if __name__ == "__main__":
    main()
