from datetime import datetime
import folium
from folium.plugins import MarkerCluster
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    SplineTransformer,
)
from streamlit_folium import folium_static


FORMAT_DATE = "%Y-%m-%d %H:%M:%S"
DATA_PATH = "data/"
DATA_PATH_COVID = "data/indicateurs_covid.csv"


class Feature_Engineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed = self._select_columns(X_transformed)
        X_transformed = self._encode_datetime(X_transformed)
        X_transformed = self._add_meteo(X_transformed)
        X_transformed = self._add_direction(X_transformed)
        X_transformed = self._is_weekend(X_transformed)
        X_transformed = self._add_bank_holidays(X_transformed)
        X_transformed = self._add_lockdown(X_transformed)
        X_transformed = self._add_daylight(X_transformed)
        X_transformed = self._one_hot_encode_month(X_transformed)
        X_transformed = self._add_season(X_transformed)
        X_transformed = self._add_holidays(X_transformed)
        return X_transformed

    def _select_columns(self, df):
        df["date"] = pd.to_datetime(df["date"], format=FORMAT_DATE)
        df = df[["date", "counter_name", "site_name"]]
        return df

    def _encode_datetime(self, df):
        """
        We want to include the cyclical continuity present in the date.
        For instance, we want:
        - Hour of Day: 12am to be close to 1pm but far to 10pm
        - Day of week: Sunday to be close to Monday but far from
        - Month: January to be close to February and far from July
        Hence, using sine/cosine transformations, we can model that cyclicity.
        """
        # Day of Week
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)

        df["day_of_year"] = df["date"].dt.dayofyear
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)

        # Hour of Day
        df["hour_sin"] = np.sin(2 * np.pi * df["date"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["date"].dt.hour / 24)
        df["hour"] = df["date"].dt.hour
        df["hour_cos_squared"] = np.cos(4 * np.pi * df["date"].dt.hour / 24)
        df["hour_sin_squared"] = np.sin(4 * np.pi * df["date"].dt.hour / 24)

        # Month
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
        df["month"] = df["date"].dt.month

        # Year
        df["year"] = df["date"].dt.year

        return df

    def _is_weekend(self, df):
        df["is_saturday"] = df["day_of_week"] == 5
        df["is_sunday"] = df["day_of_week"] == 6
        df["is_day_of_week"] = df["day_of_week"] < 5
        return df

    def _one_hot_encode_month(self, df):
        # Create 12 columns for each month with values 1 or 0
        for month in range(1, 13):
            column_name = f"month_{month}"
            df[column_name] = 0
            df.loc[df["date"].dt.month == month, column_name] = 1

        return df

    def _add_meteo(self, df):
        """
        Adding meteo informations to our data:
        -
        """
        weather = pd.read_csv(
            f"{DATA_PATH}external_data.csv",
            usecols=["date", "ff", "t", "etat_sol", "rr1", "rr3", "u"],
        ).dropna()
        weather = weather.rename(
            columns={
                "ff": "wind_power",
                "t": "temperature",
                "rr1": "last_hour_precip_mm",
                "rr3": "last_3hours_precip_mm",
                "u": "humidity",
            }
        )
        # Modify column date to a date matching the format of the ain data.
        weather["date"] = pd.to_datetime(weather["date"], format=FORMAT_DATE)

        # Create a DataFrame with a row per hour from 2020-09-01-2021-10-21
        all_hours = pd.date_range(
            start=min(weather["date"]), end=max(weather["date"]), freq="h"
        )
        df_hours = pd.DataFrame({"date": all_hours})

        # Merge the DataFrames to have NaN values for rows with missing hour.
        df_all_weather = pd.merge(df_hours, weather, on="date", how="left")

        # Drop duplicates (2020-11-20 18:00:00 is duplicated for some reason)
        df_all_weather.drop_duplicates(inplace=True)

        # Set "date" column as the index for time-weighted interpolation
        df_all_weather.set_index("date", inplace=True)

        # Perform time-weighted interpolation
        df_all_weather = df_all_weather.interpolate(method="time").round(2)

        # Reset index to make "date" a regular column again
        df_all_weather.reset_index(inplace=True)

        # Merge the weather information with the main data
        df = pd.merge(df, df_all_weather, on="date", how="left")
        df["temperature2"] = df["temperature"] ** 2
        return df

    def _add_direction(self, df):
        """
        This function will add a column with the counter's direction among:
        - E-O and O-E
        - NE-SO and SO-NE
        - N-S and S-N
        - SE-NO and NO-SE
        """

        df["direction"] = df["counter_name"].str.extract(r"([A-Z]+-[A-Z]+)$")
        return df

    def _add_bank_holidays(self, df):
        """
        This function adds a column to our main data indicating whether
        the given date is a day off or not.
        """
        bank_holidays = pd.read_csv(f"{DATA_PATH}holidays.csv", parse_dates=["date"])
        # Create a new column 'is_dayoff' indicating whether it's a workday
        is_holiday = df["date"].isin(bank_holidays["date"])
        df["is_dayoff"] = (is_holiday).astype(int)
        return df

    def _add_lockdown(self, df):
        def is_lockdown(d):
            # First lockdown from March 17 to May 10
            if datetime(2020, 3, 17) <= d <= datetime(2020, 5, 10, 23, 59):
                return 1

            # Curfew from October 17 to October 29 (9pm to 6am)
            if datetime(2020, 10, 17) <= d <= datetime(2020, 10, 29, 23, 59):
                return int(d.hour >= 21 or d.hour < 6)

            # Lockdown from October 30 to December 14 (all day long)
            if datetime(2020, 10, 30) <= d <= datetime(2020, 12, 14, 23, 59):
                return 1

            # Curfew from December 15 to January 15 (8pm to 6am)
            elif datetime(2020, 12, 15) <= d <= datetime(2021, 1, 15, 23, 59):
                return int(d.hour >= 20 or d.hour < 6)

            # Curfew from January 16 to March 19 (6pm to 6am)
            elif datetime(2021, 1, 16) <= d <= datetime(2021, 3, 19, 23, 59):
                return int(d.hour >= 18 or d.hour < 6)

            # Curfew from March 20 to May 18 (7pm to 6am)
            elif datetime(2021, 3, 20) <= d <= datetime(2021, 5, 18, 23, 59):
                return int(d.hour >= 19 or d.hour < 6)

            # Curfew from May 19 to June 8 (9pm to 6am)
            elif datetime(2021, 5, 19) <= d <= datetime(2021, 6, 8, 23, 59):
                return int(d.hour >= 21 or d.hour < 6)

            # Curfew from June 9 to June 19 (11pm to 6am)
            elif datetime(2021, 6, 9) <= d <= datetime(2021, 6, 19, 23, 59):
                return int(d.hour >= 23 or d.hour < 6)

            # No curfew
            else:
                return 0

        # Apply the is_lockdown function to the "date" column
        df["is_lockdown"] = df["date"].apply(is_lockdown)
        return df

    def _add_daylight(self, df):
        def _is_daylight(r):
            return int(r["sunrise_hour"] <= r["date"] <= r["sunset_hour"])

        # Import data about sunrises and sunsets
        df_sun = pd.read_csv(f"{DATA_PATH}ephemerides.csv", sep=",")
        df_sun["date"] = pd.to_datetime(df_sun["date"])
        df_sun.rename(columns={"date": "date_merge"}, inplace=True)
        df_sun["sunrise_hour"] = pd.to_datetime(
            df_sun["sunrise_hour"], format=FORMAT_DATE
        )
        df_sun["sunset_hour"] = pd.to_datetime(
            df_sun["sunset_hour"], format=FORMAT_DATE
        )

        # Create a date column with less granularity
        df["date_merge"] = df["date"].dt.floor("d")

        # Perform the join on the common date column
        result_df = pd.merge(df, df_sun, on="date_merge", how="left")

        # Apply _is_daylight function to each row
        result_df["is_daylight"] = result_df.apply(_is_daylight, axis=1)

        # Drop the temporary "date_merge" column
        result_df = result_df.drop(
            columns=["date_merge", "sunrise_hour", "sunset_hour"]
        )

        return result_df

    def _add_season(self, df):
        def get_season(month, day):
            if (
                (month == 12 and day >= 21)
                or any([month == m for m in [1, 2]])
                or (month == 3 and day <= 20)
            ):
                return "Winter"
            elif (
                (month == 3 and day >= 21)
                or any([month == m for m in [4, 5]])
                or (month == 6 and day <= 20)
            ):
                return "Spring"
            elif (
                (month == 6 and day >= 21)
                or any([month == m for m in [7, 8]])
                or (month == 9 and day <= 20)
            ):
                return "Summer"
            else:
                return "Fall"

        df["season"] = df.apply(
            lambda row: get_season(row["date"].month, row["date"].day), axis=1
        )
        return df

    def _add_holidays(self, df):
        # Preparing df
        df["day"] = df["date"].dt.day
        df["is_holiday"] = 0

        # Loading vacs data and ensuring datetime has a proper format
        vacs = pd.read_csv(f"{DATA_PATH}vacs_scolaires.csv")
        vacs["start_date"] = pd.to_datetime(vacs["start_date"], format=FORMAT_DATE)
        vacs["end_date"] = pd.to_datetime(vacs["end_date"], format=FORMAT_DATE)

        # Set is_holiday to 1 if date is in the range
        for _, holiday_row in vacs.iterrows():
            start_year, start_month, start_day = (
                holiday_row["start_date"].year,
                holiday_row["start_date"].month,
                holiday_row["start_date"].day,
            )
            end_month, end_day = (
                holiday_row["end_date"].month,
                holiday_row["end_date"].day,
            )

            # Set is_holiday to 1 for rows within the holiday range
            df.loc[
                (df["year"] == start_year)
                & (df["month"] >= start_month)
                & (df["month"] <= end_month)
                & (df["day"] >= start_day)
                & (df["day"] <= end_day),
                "is_holiday",
            ] = 1
            # df.drop(columns=["day"], inplace=True)
        return df

    def _add_covid(self, df):
        covid = pd.read_csv(DATA_PATH_COVID, usecols=["date", "dep", "hosp"])
        covid["d"] = pd.to_datetime(covid["date"]).dt.date
        covid = covid[covid["dep"] == 75]
        covid.drop(columns=["date", "dep"], inplace=True)
        df["d"] = df["date"].dt.date
        df = pd.merge(left=df, right=covid, how="left", on="d")
        df.drop(columns=["d"], inplace=True)
        return df


def preprocessing():
    def periodic_spline(period, n_splines=None, degree=2):
        if n_splines is None:
            n_splines = period
        n_knots = n_splines + 1
        return SplineTransformer(
            degree=degree,
            n_knots=n_knots,
            knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
            extrapolation="periodic",
            include_bias=True,
        )

    # Specify the columns to scale
    numeric_features = [
        "wind_power",
        "temperature",
        "last_3hours_precip_mm",
        "humidity",
    ]

    # Specify the columns to one-hot encode
    categorical_features = ["season", "hour", "year"]

    # Specify the columns to be used without any preprocessing
    passthrough_features = [
        "etat_sol",
        "is_lockdown",
        "is_holiday",
        "is_sunday",
        "is_saturday",
        "is_day_of_week",
        "is_dayoff",
        "is_daylight",
        "month_1",
        "month_2",
        "month_3",
        "month_4",
        "month_5",
        "month_6",
        "month_7",
        "month_8",
        "month_9",
        "month_10",
        "month_11",
        "month_12",
    ]

    # Define the column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cyclic_month", periodic_spline(12, n_splines=3), ["month"]),
            ("cyclic_hour", periodic_spline(24, n_splines=5), ["hour"]),
            (
                "spline_weekdays",
                periodic_spline(24 * 5, n_splines=5),
                ["hour"],
            ),
            (
                "spline_weekends",
                periodic_spline(24 * 2, n_splines=2),
                ["hour"],
            ),
            ("cat", OneHotEncoder(), categorical_features),
            ("passthrough", "passthrough", passthrough_features),
        ],
    )

    return preprocessor


def get_train_data():
    f_name = f"{DATA_PATH}train.parquet"
    _target_column_name = "log_bike_count"
    data = pd.read_parquet(f_name)

    # Removing anormal 0 values from our model for better training.
    data["to_remove"] = (
        (data["log_bike_count"] == 0)
        & (data["date"].dt.hour >= 6)
        & (data["date"].dt.hour <= 20)
    )
    data = data[data["to_remove"] == 0]
    data = data.drop(columns=["to_remove"])

    # Order the dataframes in case we use CV or plot visuals
    # data = data.sort_values(["date", "counter_name"])
    y_df = data[["counter_name", _target_column_name]]
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_df


def get_test_data():
    f_name = f"{DATA_PATH}test.parquet"
    data = pd.read_parquet(f_name)
    X_test = data.drop(columns=["bike_count", "log_bike_count"]).copy()
    y_test = data[["counter_name", "log_bike_count", "date"]].copy()

    return X_test, y_test


def encode_train_data(X_train, y_train):
    le = LabelEncoder()
    X_train["counter_name_encoded"] = le.fit_transform(X_train["counter_name"])
    y_train["counter_name_encoded"] = le.transform(y_train["counter_name"])
    return X_train, y_train, le


def train_model(X_train, y_train):
    preprocessor = preprocessing()
    X_train, y_train, le = encode_train_data(X_train, y_train)

    for i in range(len(le.classes_)):
        sub_X_train = X_train[X_train["counter_name_encoded"] == i]
        sub_y_train = y_train[y_train["counter_name_encoded"] == i]
        alphas = np.logspace(-3, 3, 20)
        regressor = RidgeCV(alphas=alphas, cv=7)

        pipe = Pipeline(
            steps=[
                ("Feature Engineering", Feature_Engineering()),
                ("preprocessor", preprocessor),
                (
                    "kernel expansion",
                    Nystroem(
                        kernel="rbf",
                        degree=3,
                        n_components=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
                ("regressor", regressor),
            ]
        )

        pipe.fit(sub_X_train, sub_y_train["log_bike_count"])

        # Make predictions
        pred = pipe.predict(sub_X_train)

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(sub_y_train["log_bike_count"], pred))
        print(f"RMSE for counter {le.classes_[i]}: {rmse:.4f}")

        # Save the model
        dump(pipe, f"model_{i}.joblib")

    return le


def fit_test(X_test, y_test, le):
    X_test["counter_name_encoded"] = le.transform(X_test["counter_name"])
    y_test["counter_name_encoded"] = le.transform(y_test["counter_name"])
    rmses = []

    for i in range(len(le.classes_)):
        sub_X_test = X_test[X_test["counter_name_encoded"] == i]
        sub_y_test = y_test[y_test["counter_name_encoded"] == i]
        pipe = load(f"scripts/fitted_models/model_{i}.joblib")
        y_pred = pipe.predict(sub_X_test)

        # Calculate RMSE for the model on the test set
        rmse = np.sqrt(mean_squared_error(sub_y_test["log_bike_count"], y_pred))
        print(f"On test set counter {le.classes_[i]} has an RMSE of {rmse: .4f}.")
        rmses.append(rmse)

    return rmses


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import load

# Caching the prediction data
@st.cache_data
def get_model_predictions(X, y, _le, counter):
    # Encode the counter names
    X["counter_name_encoded"] = _le.transform(X["counter_name"])
    y["counter_name_encoded"] = _le.transform(y["counter_name"])

    # Load pre-trained model for the selected counter
    pipe = load(f"scripts/fitted_models/model_{_le.transform([counter])[0]}.joblib")

    # Select data for the current counter and sort by dates
    sub_X = X[X["counter_name_encoded"] == _le.transform([counter])[0]].sort_values("date")
    sub_y = y[y["counter_name_encoded"] == _le.transform([counter])[0]].sort_values("date")

    # Make predictions on the selected data
    train_predictions = pipe.predict(sub_X)

    return sub_X, sub_y, train_predictions


# Function to plot the predictions with caching
def plot_predictions(X, y, le, counter):
    # Set plot style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Get cached model predictions
    sub_X, sub_y, train_predictions = get_model_predictions(X, y, le, counter)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))

    # Extract dates, times, and days for plotting
    train_dates = sub_X["date"]
    train_times = sub_X["date"].dt.hour + sub_X["date"].dt.minute / 60
    train_days = sub_X["date"].dt.day_of_week

    # Plot 1: Actual vs predicted values by date
    axs[0].scatter(train_dates, sub_y["log_bike_count"].apply(lambda x: np.exp(x)), label="Actual Values", color="coral", alpha=0.7, s=50)
    axs[0].plot(train_dates, np.exp(train_predictions), label="Predicted Values", color="royalblue", alpha=0.9, linewidth=2)
    axs[0].set_xlabel("Date", fontsize=14)
    axs[0].set_ylabel("Number of bikes", fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    # Plot 2: Actual vs predicted values by time of day
    axs[1].scatter(train_times, sub_y["log_bike_count"].apply(lambda x: np.exp(x)), label="Actual Values", color="coral", alpha=0.7, s=50)
    axs[1].scatter(train_times + 0.25, np.exp(train_predictions), label="Predicted Values", color="royalblue", alpha=0.9, s=50)
    axs[1].set_xlabel("Time of Day (Hour)", fontsize=14)
    axs[1].set_ylabel("Number of bikes", fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    # Plot 3: Actual vs predicted values by day of the week
    axs[2].scatter(train_days, sub_y["log_bike_count"].apply(lambda x: np.exp(x)), label="Actual Values", color="coral", alpha=0.7, s=50)
    axs[2].scatter(train_days + 0.25, np.exp(train_predictions), label="Predicted Values", color="royalblue", alpha=0.9, s=50)
    axs[2].set_xlabel("Day of the Week", fontsize=14)
    axs[2].set_ylabel("Number of bikes", fontsize=14)
    axs[2].legend(fontsize=12)
    axs[2].grid(True)

    # Add a title and adjust layout
    fig.suptitle(f"Train Set Predictions vs Actuals for Counter: {counter}", fontsize=20, weight="bold", y=1.02)
    plt.tight_layout(pad=3)

    # Return the figure (so that each subplot can be handled individually if needed)
    return fig


# Use Streamlit to display the plot
def display_predictions(X, y, le, counter):
    fig = plot_predictions(X, y, le, counter)
    st.pyplot(fig)



def render_counters(selected_date, selected_hour):
    # Load the data
    df_counters = pd.read_parquet(f"{DATA_PATH}train.parquet")[
        ["counter_name", "longitude", "latitude", "date", "bike_count"]
    ]

    # Convert date column to datetime
    df_counters["date"] = pd.to_datetime(df_counters["date"])
    df_counters["hour"] = df_counters["date"].dt.hour

    # Drop rows with missing latitude or longitude
    df_counters.dropna(subset=["longitude", "latitude"], inplace=True)

    # Apply filters based on user selection
    if selected_date is not None:
        df_counters = df_counters[df_counters["date"].dt.date == selected_date]

    if selected_hour is not None:
        df_counters = df_counters[df_counters["hour"] == selected_hour]

    # Check if there are any counters left after filtering
    if df_counters.empty:
        st.warning("No data available for the selected filters.")
        return

    # Create a Folium map centered at the mean of the coordinates
    m = folium.Map(
        location=[df_counters["latitude"].mean(), df_counters["longitude"].mean()],
        zoom_start=12.2,
    )

    # Add circles to represent the number of bikes
    for index, row in df_counters.iterrows():
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=row["bike_count"],  # Adjust the divisor to control circle size
            color="navy",
            fill=True,
            fill_color="royalblue",
            fill_opacity=0.8,
            weight=0,
            popup=f"{row['counter_name']}<br>Bike Count: {row['bike_count']}",
        ).add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=True).add_to(m)

    # Use folium_static to render the map in Streamlit
    folium_static(m)
