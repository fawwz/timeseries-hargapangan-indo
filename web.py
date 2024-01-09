import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyFTS.partitioners import Grid
from pyFTS.models import chen
from pyFTS.benchmarks import Measures
from plotly.subplots import make_subplots

st.set_option('deprecation.showPyplotGlobalUse', False)

st.info(
    "Created by:\n" 

    "Faiq Riestiansyah (3321600014) - Applied Data Science (D4) PENS 2021\n"

    )
st.title("Analisis Fuzzy Time Series Terhadap Harga Makanan di Indonesia")

# set browser tab title
st.markdown("## Dataset Keseluruhan")

# Load data
df = pd.read_csv(
    'dataset.csv'
    )
selected_data = df.copy()
selected_data = selected_data[[
    'Region','date','commodity','price']]


region_data = selected_data['Region'].unique()
commodity_data = selected_data['commodity'].unique()


st.dataframe(selected_data,width=1400, height=500)


with st.sidebar:
  st.markdown("### Configuration")
  region = st.selectbox('Select Region', region_data)
  commodity = st.selectbox(
      'Select Commodity',
      commodity_data)

# Filter dataframe based on user selection
filtered_data = selected_data[(selected_data['Region'] == region) & (selected_data['commodity'] == commodity)]

st.markdown("## Filtered Dataset")

st.dataframe(filtered_data,width=1400, height=780)

# Visualizations
st.markdown("## Visualisasi Data")

st.markdown("### Line Chart")

# Filtered dataset line chart
fig = px.line(filtered_data, x='date', y='price', title=f'{commodity} Prices in {region}')
st.plotly_chart(fig)


# Proses FTS
with st.sidebar:
  n_part = st.slider('Insert npart for Grid Partitioner', 1, 45, 12)

st.markdown("## Fuzzy Time Series Model")

fs = Grid.GridPartitioner(
    data=selected_data[(selected_data['Region'] == region) & (selected_data['commodity'] == commodity)].price, npart=n_part)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])

ax.set_ylim(-0.1, 0.1)
ax.set_xlim(
    0, len(selected_data[(selected_data['Region'] == region) & (selected_data['commodity'] == commodity)].price))
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Membership', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

fs.plot(ax)
ax.set_title(f"Prices of {commodity} in {region} Fuzzy Membership", fontsize=30)
st.pyplot(plt.show())

data_model = selected_data[(selected_data['Region'] == region) & (selected_data['commodity'] == commodity)].price
data_model = data_model.tolist()
model = chen.ConventionalFTS(partitioner=fs)

model.fit(data_model)

prediction = model.predict(data_model)
actual = data_model
fts_dates = selected_data[selected_data['Region'] == region]['date'].values


st.markdown("### Fuzzy Logical Relationship")
st.code(model)


st.markdown("### Model Performance Measurement")
# Model performance
st.code(f" \
        RMSE: {Measures.rmse(actual, prediction)} \
        MAPE: {Measures.mape(actual, prediction)} \
        ")

st.markdown("### Prediction")
fig, ax = plt.subplots(figsize=[15, 5])
prediction.insert(0, None)

orig, = ax.plot(actual, label="Original data")
pred, = ax.plot(prediction, label="Forecasts")

ax.legend(handles=[orig, pred])

# Set axis labels and title
ax.set_xlabel('Index')
ax.set_ylabel('Prices')
ax.set_title(f"{commodity} Prices Prediction in {region}")

# Show the plot
st.pyplot(fig)

# st.markdown("### Forecasting Ahead")
# # Forecasting 7 steps ahead
# forecasting = model.forecast(data_model, steps=7)

# # Generate forecasting dates
# last_date = pd.to_datetime(filtered_data['date'].iloc[-1])
# forecasting_dates = pd.date_range(start=last_date, periods=7, freq='M').strftime("%Y-%m-%d").tolist()

# # Plot original data and forecasts
# fig, ax = plt.subplots(figsize=[15, 5])

# # Plot original data
# ax.plot(data_model, label="Original data")

# # Plot forecasts
# ax.plot(range(len(data_model), len(data_model) + len(forecasting)), forecasting, label="Forecasts")

# # Set axis labels and title
# ax.set_xlabel('Index')
# ax.set_ylabel('Prices')
# ax.set_title("Prediction")

# # Show the plot
# st.pyplot(fig)