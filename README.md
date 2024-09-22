# Paris on Wheels: Analyzing and Predicting Bike Traffic Trends

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-brightgreen.svg)

This project aims to predict the number of bikes passing through a specific counter at a given time in Paris using historical data and machine learning techniques. The prediction model is implemented in Python and leverages popular libraries such as Pandas, Scikit-learn, and Matplotlib. 


## 🚀 Quick Start

To get started and being able to use this application on your machine, please follow these instructions.

### 1. Clone the repository:

```bash
git clone https://github.com/PEJU2801/web_bike_app.git
cd web_bike_app
```

### 2. Build the docker image:

```bash
docker build -t web_bike_app .
```

### 3. Run the Docker container:

```bash
docker run -p 8501:8501 web_bike_app
```

### 4. Access the app:

Once the container is running, you'll be able to naivigate through the app on a web page at the following address: http://localhost:8501.
