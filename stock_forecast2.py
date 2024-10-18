import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.dates as mdates
from datetime import date, datetime, timedelta
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
from functools import reduce
from datetime import timedelta
import statsmodels.api as sm
import itertools
import streamlit as st


# Function to load stock data
def load_stock_data(symbol):
    Bank_stock = f'Dataset CSV/{symbol}_monthly_stock_data.csv'
    data = pd.read_csv(Bank_stock)
    data['date'] = pd.to_datetime(data['date'])
   # data['% change']=data['Adjusted_close']/data['Adjusted_close'].shift(1)-1
    return data[['date', 'Adjusted_close']]
#st.write()

# Function to load and preprocess economic data
def load_and_preprocess_economic_data():
    # Load CPI data
    CPI_data = pd.read_csv('Dataset CSV/CPI_data.csv')
    CPI_data['date'] = pd.to_datetime(CPI_data['date']) + pd.offsets.MonthEnd(0)
    CPI_data.rename(columns={'value': 'cpi_value'}, inplace=True)

    # Load unemployment data
    unemployment_data = pd.read_csv('Dataset CSV/unemployment_data.csv')
    unemployment_data['value'] = pd.to_numeric(unemployment_data['value'])
    unemployment_data['date'] = pd.to_datetime(unemployment_data['date']) + pd.offsets.MonthEnd(0)
    unemployment_data.rename(columns={'value': 'unemployment'}, inplace=True)

    # Load retail data
    retail_data = pd.read_csv('Dataset CSV/retail_data.csv')
    retail_data['value'] = pd.to_numeric(retail_data['value'])
    retail_data['date'] = pd.to_datetime(retail_data['date']) + pd.offsets.MonthEnd(0)
    retail_data.rename(columns={'value': 'Retail'}, inplace=True)

    # Load federal fund (interest) rate data
    interest_data = pd.read_csv('Dataset CSV/Interest_data.csv')
    interest_data['value'] = pd.to_numeric(interest_data['value'])
    interest_data['date'] = pd.to_datetime(interest_data['date']) + pd.offsets.MonthEnd(0)
    interest_data.rename(columns={'value': 'Interest'}, inplace=True)

    # Load GDP data
    gdp_data = pd.read_csv('Dataset CSV/gdp_data.csv')
    gdp_data['value'] = pd.to_numeric(gdp_data['value'], errors='coerce')
    gdp_data['date'] = pd.to_datetime(gdp_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    gdp_data.rename(columns={'value': 'GDP'}, inplace=True)
    gdp_data.set_index('date', inplace=True)
    rgdp_df = gdp_data.resample('M').interpolate(method='linear')
    gdp_sorted = rgdp_df.sort_index(ascending=False)
    r_gdp = gdp_sorted.reset_index()
    r_gdp['GDP'] = r_gdp['GDP'].round(2)

    # Load treasury yield data
    yield_data = pd.read_csv('Dataset CSV/yield_data.csv')
    yield_data['value'] = pd.to_numeric(yield_data['value'], errors='coerce')
    yield_data['date'] = pd.to_datetime(yield_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    yield_data.rename(columns={'value': 'treasure_yield'}, inplace=True)
    yield_data=yield_data.drop (columns=['Unnamed: 0'])

    # Load durables data
    durables_data = pd.read_csv('Dataset CSV/durables_data.csv')
    durables_data['value'] = pd.to_numeric(durables_data['value'], errors='coerce')
    durables_data['date'] = pd.to_datetime(durables_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    durables_data.rename(columns={'value': 'durables'}, inplace=True)
    durables_data=durables_data.drop(columns=['Unnamed: 0'])

    # Load payroll data
    payroll_data = pd.read_csv('Dataset CSV/payroll_data.csv')
    payroll_data['value'] = pd.to_numeric(payroll_data['value'], errors='coerce')
    payroll_data['date'] = pd.to_datetime(payroll_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    payroll_data.rename(columns={'value': 'payroll'}, inplace=True)
    payroll_data=payroll_data.drop(columns=['Unnamed: 0'])

    # Merge all economic data
    dataframe = [CPI_data, unemployment_data, retail_data, interest_data, r_gdp, yield_data, durables_data, payroll_data]
    economic_data = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dataframe)
    return economic_data


# Setting up the initial dates and periods
START = '1999-11-30'
END = '2024-04-30'


# Streamlit App
st.title('Stock Price Analysis and Forecasting')

# Sidebar for stock selection
stock_symbol = st.sidebar.selectbox("Select a stock symbol:", ['JPM', 'MS', 'GS'])

# Slider to select prediction years
n_years = st.sidebar.slider('Years of prediction:', 1, 3)
period = n_years * 12

# Load stock data
stock_data = load_stock_data(stock_symbol)

# Load economic data
economic_data = load_and_preprocess_economic_data()

# Merge stock and economic data
data = pd.merge(stock_data, economic_data, on='date')
data_cleaned = data.dropna(subset=['GDP', 'durables'])
data_cleaned.set_index('date', inplace=True)

st.write(f'{stock_symbol} Stock data with economic indicators')
st.write(data_cleaned)

# Sort by 'date' in ascending order and reset the index
data_sorted = data.sort_values(by='date', ascending=True).reset_index(drop=True)

# Set 'date' as the index
data_sorted.set_index('date', inplace=True)


# Plot stock prices
st.subheader(f'{stock_symbol} Stock Price')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_cleaned.index, y=data_sorted['Adjusted_close'], mode='lines', name='Stock Price'))
fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# annual_returns= data_cleaned['% change'].mean()*12
# st.write('annual return is',annual_returns, '%' )
# stdev=np.std(data_cleaned['% change'])*np.sqrt(12)
# st.write('standard deviation is', stdev*100, '%')
# st.write('risk adjusted return is', annual_returns/ (stdev*100))



# visualising effect some major events on stock prices
economic_events = [
    {"label": "Dot-Com Bubble Burst", "start": "2000-03-10", "end": "2002-12-31", "color": "blue"},
    {"label": "9/11 Attacks", "start": "2001-09-11", "end": "2001-12-31", "color": "purple"},
    {"label": "Global Financial Crisis", "start": "2008-09-15", "end": "2009-12-31", "color": "red"},
    {"label": "European Debt Crisis", "start": "2010-05-09", "end": "2012-12-31", "color": "green"},
    {"label": "U.S. Debt Ceiling Crisis", "start": "2011-08-02", "end": "2011-12-31", "color": "orange"},
    {"label": "Oil Price Collapse", "start": "2014-06-30", "end": "2016-06-30", "color": "brown"},
    {"label": "Brexit Vote", "start": "2016-06-23", "end": "2017-12-31", "color": "cyan"},
    {"label": "U.S.-China Trade War", "start": "2018-03-22", "end": "2020-12-31", "color": "pink"},
    {"label": "COVID-19 Pandemic", "start": "2020-03-11", "end": "2021-12-31", "color": "orange"},
    {"label": "Russian Invasion of Ukraine", "start": "2022-02-24", "end": "2023-12-31", "color": "black"},
    {"label": "Global Inflation Surge", "start": "2022-01-01", "end": "2023-12-31", "color": "gray"},
    {"label": "Silicon Valley Bank Collapse", "start": "2023-03-10", "end": "2023-12-31", "color": "yellow"}
]

# Visualising the effect of major economic events on stock prices
st.subheader(f'Effects of Major Economic Events on {stock_symbol} Stock Price')

plt.figure(figsize=(14, 8))

# Plot the adjusted close price
plt.plot(data['date'], data['Adjusted_close'], label=f'{stock_symbol} Adjusted Close Price', color='b', linestyle='-', marker='o')

# Highlight major economic events using axvspan
for event in economic_events:
    start_date = pd.to_datetime(event["start"])
    end_date = pd.to_datetime(event["end"])
    plt.axvspan(start_date, end_date, color=event["color"], alpha=0.3, label=event["label"])

# Formatting the plot
plt.title(f'{stock_symbol} Adjusted Close Price with Major Economic Events', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Adjusted Close Price', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)

# Formatting x-axis for date clarity
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Ticks at the start of each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Year format for dates
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

# Adjust layout for better fit
plt.tight_layout()

# Show the plot in Streamlit
st.pyplot(plt)

# Skewness transformation function
def safe_log1p(series):
    return np.log1p(series[series > 0])

def safe_sqrt(series):
    return np.sqrt(series[series >= 0])

def apply_skewness_transformations(data):
    transformed_data = pd.DataFrame(index=data.index)
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            skewness = skew(data[col].dropna())
            if skewness > 1:  
                transformed_data[col + '_log'] = safe_log1p(data[col])
            elif 0.5 < skewness <= 1:
                transformed_data[col + '_sqrt'] = safe_sqrt(data[col])
            else:
                transformed_data[col] = data[col]
        else:
            transformed_data[col] = data[col]
    return transformed_data.fillna(0)

# Apply transformations
data_transformed = apply_skewness_transformations(data_sorted)
st.subheader('transformed data')
st.write(data_transformed)

# Visualization of original and transformed data
st.subheader(f'Distribution and Skewness Visualization for {stock_symbol}')

# Select a column to visualize
column_to_visualize = st.selectbox("Select a column to visualize:", data_sorted.columns)

# Plot original and transformed distribution side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Original data distribution
sns.histplot(data_sorted[column_to_visualize].dropna(), bins=30, kde=True, ax=ax[0], color='blue')
ax[0].set_title(f'Original Distribution of {column_to_visualize}')
ax[0].set_xlabel(column_to_visualize)

# Check if the column was transformed and plot transformed distribution
if column_to_visualize + '_log' in data_transformed.columns:
    transformed_col = column_to_visualize + '_log'
elif column_to_visualize + '_sqrt' in data_transformed.columns:
    transformed_col = column_to_visualize + '_sqrt'
else:
    transformed_col = column_to_visualize

# Transformed data distribution
sns.histplot(data_transformed[transformed_col].dropna(), bins=30, kde=True, ax=ax[1], color='green')
ax[1].set_title(f'Transformed Distribution of {transformed_col}')
ax[1].set_xlabel(transformed_col)

# Display the plots
st.pyplot(fig)

# Correlation Heatmap
st.subheader('Correlation Matrix')
corr_matrix = data_transformed.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
st.pyplot(fig)

# Feature Importance (Random Forest)
st.subheader('Feature Importance using Random Forest')

# #lets use the correct target column based on transformation
# if stock_symbol in ['BAC']:
#     target_column=['Adjusted_close'] #for BAC
# elif stock_data=='BAC':
#     target_column = 'Adjusted_close' #no
# else:
#     target_column=['Adjusted_close_log']#for other stock

features = data_transformed.drop(columns=['Adjusted_close_log'])
target = data_transformed['Adjusted_close_log']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(features, target)
importances = rf.feature_importances_
indices = np.argsort(importances)
fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], color='b', align='center')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features.columns[i] for i in indices])
ax.set_xlabel('Relative Importance')
st.pyplot(fig)

# Prophet Forecasting
st.subheader(f'{stock_symbol} Stock Price Forecast Prophet')
# Reset the index to access the 'date' column
data_transformed_reset = data_transformed.reset_index()

columns_to_select = ['Adjusted_close_log', 'date']
columns_to_select.extend(features.columns[indices[:6]])
data_ready = data_transformed_reset[columns_to_select]
data_ready.rename(columns={'date': 'ds', 'Adjusted_close_log': 'y'}, inplace=True)

model = Prophet()
for feature in features.columns[indices[:6]]:
    model.add_regressor(feature)
model.fit(data_ready)

# Getting Future forecast
future = model.make_future_dataframe(periods=period, freq='M')
for feature in features.columns[indices[:6]]:
    future[feature] = np.append(data_ready[feature].values, [data_ready[feature].values[-1]] * (len(future) - len(data_ready)))

forecast = model.predict(future)
fig = model.plot(forecast)
st.pyplot(fig)

data_transformed_reset = data_transformed.reset_index()

# Calculate RMSE and MAE for Prophet
y_true = data_transformed_reset['Adjusted_close_log']  # Actual values
y_pred_prophet = forecast['yhat'].iloc[:len(y_true)]   # Predicted values from Prophet

# RMSE for Prophet
rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
# MAE for Prophet
mae_prophet = mean_absolute_error(y_true, y_pred_prophet)
r2_prophet = r2_score(y_true, y_pred_prophet)

#visualise in streamlit
st.write(f'Prophet Model RMSE: {rmse_prophet}')
st.write(f'Prophet Model MAE: {mae_prophet}')
st.write(f'Prophet Model R2: {r2_prophet}')


# --- ARIMA MODEL ---
# ARIMA Model setup and training
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
exog_vars = data_transformed[['GDP', 'unemployment_log', 'cpi_value', 'Retail_sqrt']]
aic_values = []
st.write(exog_vars)

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data_transformed['Adjusted_close_log'],
                                            exog=exog_vars,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=False)
            aic_values.append((param, param_seasonal, results.aic))
        except Exception:
            continue

best_pdq, best_seasonal_pdq, best_aic = sorted(aic_values, key=lambda x: x[2])[0]

best_model = sm.tsa.statespace.SARIMAX(data_transformed['Adjusted_close_log'],
                                       exog=exog_vars,
                                       order=best_pdq,
                                       seasonal_order=best_seasonal_pdq,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
best_results = best_model.fit(disp=False)

# Start forecasting from '2020-03-31'
start_date = pd.to_datetime('2020-03-31')

# Generate predictions starting from '2020-03-31'
pred = results.get_prediction(start=start_date, dynamic=False)

# Get the predicted mean values
predicted_mean = pred.predicted_mean

# Extracting  the confidence intervals
pred_ci = pred.conf_int()

# to get the actual data from the forecast start date for comparison
actual_data =data_transformed.loc[start_date:, 'Adjusted_close_log']

st.subheader(f'{stock_symbol} forward forecast ARIMA')
fig, ax= plt.subplots()
# lets Plot the observed data and predicted values
ax = data_transformed['Adjusted_close_log'].plot(label='Observed', figsize=(14, 7))
predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.7)

# Plot the confidence intervals
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.2)

# we Add labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing stock')
plt.legend()
st.pyplot(fig)

# Calculate the Mean Squared Error (MSE) for the forecasted period
mse_Arima = mean_squared_error(actual_data, predicted_mean)
Mae_Arima = mean_absolute_error (actual_data, predicted_mean)
r2_Arima = r2_score(actual_data, predicted_mean)
st.write(f'Mean Squared Error (MSE) from {start_date.date()}: {mse_Arima}')
st.write(f'Mean absolute Error (Mae) from {start_date.date()}: {Mae_Arima}')
st.write(f'R2 from {start_date.date()}: {r2_Arima}')


# lets get the last known values for each exogenous variable (from data2_reset)
last_known_exog = data_transformed[['GDP', 'unemployment_log', 'cpi_value', 'Retail_sqrt']].iloc[-1]

# Create a DataFrame of future exog values by repeating the last known values for 45 months
future_exog = pd.DataFrame([last_known_exog] * period, columns=['GDP', 'unemployment_log', 'cpi_value', 'Retail_sqrt'])

# Forecast with the repeated future exog values
pred_uc = results.get_forecast(steps=period, exog=future_exog)

# Get the confidence intervals of the forecast
pred_ci = pred_uc.conf_int()

# Plot the historical data and forecasted values
st.subheader(f'{stock_symbol} Stock Price Forecast ARIMA')
fig, ax = plt.subplots()
ax = data_transformed['Adjusted_close_log'].plot(label='Historic', figsize=(14, 7))

# Plot the forecasted mean
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

# Plot the confidence intervals
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

# Add labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted_close')
plt.legend()
st.pyplot(fig)

# Sidebar for selecting banks to compare
st.sidebar.header("Select Banks for Ratio Comparison")
selected_banks = st.sidebar.multiselect("Select one or more banks:", ['JPM', 'MS', 'GS'])
# Define a function to calculate the average of a column in a DataFrame
def calculate_average(data, column_name):
    return data[column_name].mean()

#Lets create tabs for the various ratios and news
PE_Ratio, ROE_Data, CET1_Ratio,  News=st.tabs(['PE_Ratio', 'ROE', 'CET1_Ratio', 'top 10 news'])
#lets load the PE data
with PE_Ratio:
    def load_PE_Ratio(symbol):#define a function to load data
        PE = f'Dataset xlsx/{symbol}_PE Ratio.xlsx'
        data = pd.read_excel(PE)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        return data
    
    # Comparison of P/E Ratios
    st.subheader("P/E Ratio Comparison")
    
    if selected_banks:
        fig = go.Figure()
        for bank in selected_banks:
            PE_data = load_PE_Ratio(bank)
            avg_pe_ratio = calculate_average(PE_data, 'PE ratio')
            st.metric(label=f'Average P/E Ratio for {bank}', value=f"{avg_pe_ratio:.2f}")
            fig.add_trace(go.Scatter(x=PE_data.index, y=PE_data['PE ratio'], mode='lines', name=f'{bank} P/E Ratio'))

        fig.layout.update(title_text='P/E Ratio Comparison', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.write("Please select at least one bank for comparison.")

    #displaying the data on streamlit
    # st.subheader(f'{stock_symbol} PE Ratio' )
    PE_data= load_PE_Ratio(stock_symbol)
    # st.write (PE_data)

    #calculate and display the avg P/E ratio in a card
    # avg_pe_ratio=calculate_average (PE_data, 'PE ratio')
    # st.metric (label='Average P/E Ratio', value=f"{avg_pe_ratio:.2f}")
    

    
    # st.subheader(f'{stock_symbol} PE Ratio')
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=PE_data.index, y=PE_data['PE ratio'], mode='lines', name='PE Ratio'))
    # fig.layout.update(title_text='PE ratio data', xaxis_rangeslider_visible=True)
    
    # st.plotly_chart(fig)

    #lets write a function to determine the type of transformation based on how skewed the data is
    def safe_log1p(series):
        return np.log1p(series[series > 0])#highly skewed
    def safe_sqrt(series):
        return np.sqrt(series[series >= 0]) #moderately skewed
    
    #lets write a function to transform data based on skeness value
    def apply_skewness_transformations(data, stock_symbol):
        transformed_PEdata = pd.DataFrame(index=data.index)
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64'] and col != 'date':  # Skip 'date' column:
                skewness = skew(data[col].dropna())
                if stock_symbol in ['MS', 'GS']:   
                    if skewness > 1:
                        transformed_PEdata[col + '_log'] = safe_log1p(data[col])
                    elif 0.5 < skewness <= 1:
                        transformed_PEdata[col + '_sqrt'] = safe_sqrt(data[col])
                    else:
                        transformed_PEdata[col] = data[col]
                else:# if JPM, n transformation
                    transformed_PEdata[col] = data[col]
        return transformed_PEdata.fillna(0)

    # Apply transformations
    PEdata_transformed = apply_skewness_transformations(PE_data, stock_symbol)
    st.subheader(f'transformed PE data for {stock_symbol}')
    st.write(PEdata_transformed)




    #st.write(PE_data.dtypes)
   # st.write(features)
    #lets merge normalised econoic data with PE ratio
    #featurePe=features.drop(columns=['% change_log'])
    features_quarterly= features.resample('Q').mean()#resampling, so that feature data match PE data
    features_quarterly.reset_index(inplace=True)

    PEdata_transformed.reset_index(inplace=True)
    # Adjust the PE column based on the selected stock
    if stock_symbol == 'JPM':
        mergedPE_data = pd.merge(features_quarterly, PEdata_transformed[['date', 'PE ratio']], on='date', how='inner')
        targetPE = mergedPE_data['PE ratio']  # Use the non-transformed PE ratio
    else:
        mergedPE_data = pd.merge(features_quarterly, PEdata_transformed[['date', 'PE ratio_log']], on='date', how='inner')
        targetPE = mergedPE_data['PE ratio_log']  # Use the log-transformed PE for MS and GS

    # Ensure 'date' is in the index before merging
    mergedPE_data.dropna(inplace=True)
   # st.write(mergedPE_data)

    # mergedPE_data=pd.merge(features_quarterly, PE_data, on='date', how='inner')
    # mergedPE_data.dropna(inplace=True)
    # st.write(mergedPE_data)

    featureSPE = mergedPE_data.drop(columns=[targetPE.name, 'date'])  # Dynamically drop the correct PE column
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(featureSPE, targetPE)
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    # Plot feature importance
    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices], color='b', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([featureSPE.columns[i] for i in indices])
    ax.set_xlabel('Relative Importance')
    st.pyplot(fig)
    
    period2 = n_years * 4
    st.subheader(f'{stock_symbol} Stock Price Forecast Prophet')
    # Reset the index to access the 'date' column
    #mergedPE_data_reset = mergedPE_data.reset_index()
    columns_to_select = [targetPE.name, 'date']
    columns_to_select.extend(featureSPE.columns[indices[:8]])
    dataPE_ready = mergedPE_data[columns_to_select]
    dataPE_ready.rename(columns={'date': 'ds', targetPE.name: 'y'}, inplace=True)
    st.write(dataPE_ready)
    
    model = Prophet()
    for feature in featureSPE.columns[indices[:8]]:
        model.add_regressor(feature)
    model.fit(dataPE_ready)
    # Getting Future forecast
    future = model.make_future_dataframe(periods=period2, freq='Q')
    for feature in featureSPE.columns[indices[:8]]:
        future[feature] = np.append(dataPE_ready[feature].values, [dataPE_ready[feature].values[-1]] * (len(future) - len(dataPE_ready)))
    
    forecast = model.predict(future)
    fig = model.plot(forecast)
    st.pyplot(fig)

    dataPE_reset = mergedPE_data.reset_index()
    st.write(dataPE_reset)
    # Calculate RMSE and MAE for Prophet
    y_true = dataPE_reset[targetPE.name]  # Actual values
    y_pred_prophet = forecast['yhat'].iloc[:len(y_true)]   # Predicted values from Prophet
    # RMSE for Prophet
    rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
    # MAE for Prophet
    mae_prophet = mean_absolute_error(y_true, y_pred_prophet)
    #R2 for Prophet
    r2_prophet = r2_score(y_true, y_pred_prophet)
    #visualise in streamlit
    st.write(f'PE Prophet Model RMSE: {rmse_prophet}')
    st.write(f'PE Prophet Model MAE: {mae_prophet}')
    st.write(f'Prophet Model R2: {r2_prophet}')

    


    # --- ARIMA MODEL ---
    # # ARIMA Model setup and training
    mergedPE_data.set_index('date', inplace=True)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 4) for x in pdq]
    exog_varPE =mergedPE_data.drop(columns=[targetPE.name])
    #exog_varPE =mergedPE_data[['GDP','Interest_sqrt', 'unemployment_log', 'cpi_value', 'Retail_sqrt', 'durables', 'treasure_yield', 'payroll_sqrt']]
    st.write(exog_varPE)
    aic_values = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(mergedPE_data[targetPE.name],
                                                exog=exog_varPE,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                aic_values.append((param, param_seasonal, results.aic))
            except Exception:
                continue

    best_pdq, best_seasonal_pdq, best_aic = sorted(aic_values, key=lambda x: x[2])[0]

    best_model = sm.tsa.statespace.SARIMAX(mergedPE_data[targetPE.name],
                                        exog=exog_varPE,
                                        order=best_pdq,
                                        seasonal_order=best_seasonal_pdq,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
    best_results = best_model.fit(disp=False)

    # Start forecasting from '2020-03-31'
    start_date = pd.to_datetime('2021-03-31')

    # Generate predictions starting from '2020-03-31'
    pred = results.get_prediction(start=start_date, dynamic=False)

    # Get the predicted mean values
    predicted_mean = pred.predicted_mean

    # Extracting  the confidence intervals
    pred_ci = pred.conf_int()

    # to get the actual data from the forecast start date for comparison
    actualPE_data =mergedPE_data.loc[start_date:, targetPE.name]

    st.subheader(f'{stock_symbol} forward forecast ARIMA')
    fig, ax= plt.subplots()
    # lets Plot the observed data and predicted values
    ax = mergedPE_data[targetPE.name].plot(label='Observed', figsize=(14, 7))
    predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.7)

    # Plot the confidence intervals
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=0.2)

        # we Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('PE ratio')
    plt.legend()
    st.pyplot(fig)

    # Calculate the Mean Squared Error (MSE) for the forecasted period
    mse_Arima = mean_squared_error(actualPE_data , predicted_mean)
    Mae_Arima = mean_absolute_error (actualPE_data , predicted_mean)
    r2_Arima=r2_score(actualPE_data , predicted_mean)
    st.write(f'Mean Squared Error (MSE) from {start_date.date()}: {mse_Arima}')
    st.write(f'Mean absolute Error (Mae) from {start_date.date()}: {Mae_Arima}')
    st.write(f'R2 From {start_date.date()}: {r2_Arima}')

    # lets get the last known values for each exogenous variable (from data2_reset)
    last_known_exog = exog_varPE.iloc[-1]

   # st.write(last_known_exog)

  

    # Dynamically get the columns from the last known exogenous variables
    columns = last_known_exog.index.tolist()

    #st.write(last_known_exog)

    # Create a DataFrame of future exog values by repeating the last known values for 45 months
    #future_exog = pd.DataFrame([last_known_exog] * period, columns=['GDP', 'interest_sqrt', 'unemployment_log', 'cpi_value', 'Retail_sqrt', 'treasure_yield', 'durables', 'payroll_sqrt'])

    # Create a DataFrame of future exog values by repeating the last known values for the forecast period
    # period is the number of months to forecast
    #future_exog = pd.DataFrame([last_known_exog.values] * period, columns=columns)
   
    #we need to keep columns dynamic and use period2 instead, to accounts for quaterly diplay
    future_exog = pd.DataFrame(
    [last_known_exog.values] * period2,  # Repeating the last known exogenous values
    index=pd.date_range(start=last_known_exog.name, periods=period2, freq='Q'),  # Quarterly frequency
    columns=columns  # Dynamic columns from last_known_exog 
    )
    #st.write(future_exog)

    # Forecast with the repeated future exog values
    pred_uc = results.get_forecast(steps=period2, exog=future_exog)

    # Get the confidence intervals of the forecast
    pred_ci = pred_uc.conf_int()

    # Plot the historical data and forecasted values
    st.subheader(f'{stock_symbol} Stock Price Forecast ARIMA')
    fig, ax = plt.subplots()
    ax = mergedPE_data[targetPE.name].plot(label='Historic', figsize=(14, 7))

    # Plot the forecasted mean
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

    # Plot the confidence intervals
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)

    # Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('PE Ratio')
    plt.legend()
    st.pyplot(fig)


#lets load the ROE data
with ROE_Data: 
    def load_ROE_Data(symbol):#define a function to load data
        ROE = f'Dataset xlsx/{symbol}_ROE.xlsx'
        data = pd.read_excel(ROE)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        return data
    
    # Comparison of ROE Ratios

    st.subheader("ROE Comparison")
    
    if selected_banks:
        fig = go.Figure()
        for bank in selected_banks:#to display plot as well as avg for selected banks
            ROE_data = load_ROE_Data(bank)
            avg_roe = calculate_average(ROE_data, 'ROE')
            st.metric(label=f'Average ROE for {bank}', value=f"{avg_roe:.2f}%")
            fig.add_trace(go.Scatter(x=ROE_data.index, y=ROE_data['ROE'], mode='lines', name=f'{bank} ROE'))

        fig.layout.update(title_text='ROE Comparison', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.write("Please select at least one bank for comparison.")

    # #displaying the data on streamlit
    # st.subheader(f'{stock_symbol} ROE data' )
    ROE_data= load_ROE_Data(stock_symbol)
    # st.write (ROE_data)
    
    # st.subheader(f'{stock_symbol} ROE  data')
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=ROE_data.index, y= ROE_data['ROE'], mode='lines', name='ROE Ratio'))
    # fig.layout.update(title_text='ROE data', xaxis_rangeslider_visible=True)
   
    # st.plotly_chart(fig)

#lets write a function to determine the type of transformation based on how skewed the data is
    def safe_log1p(series):
        return np.log1p(series[series > 0])#highly skewed
    def safe_sqrt(series):
        return np.sqrt(series[series >= 0]) #moderately skewed
    
    #lets write a function to transform data based on skeness value
    def apply_skewness_transformations(data, stock_symbol):
        transformed_ROEdata = pd.DataFrame(index=data.index)
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64'] and col != 'date':  # Skip 'date' column:
                skewness = skew(data[col].dropna())
                if stock_symbol in ['MS', 'GS']:   
                    if skewness > 1:
                        transformed_ROEdata[col + '_log'] = safe_log1p(data[col])
                    elif 0.5 < skewness <= 1:
                        transformed_ROEdata[col + '_sqrt'] = safe_sqrt(data[col])
                    else:
                        transformed_ROEdata[col] = data[col]
                else:# if JPM, n transformation
                    transformed_ROEdata[col] = data[col]
        return transformed_ROEdata.fillna(0)

    # Apply transformations
    ROEdata_transformed = apply_skewness_transformations(ROE_data, stock_symbol)
    st.subheader(f'transformed PE data for {stock_symbol}')
    st.write(ROEdata_transformed)


    # # #lets merge normalised economic data with ROE ratio
    # features_quarterly= features.resample('Q').mean()#resampling, so that feature data match PE data
    # features_quarterly.reset_index(inplace=True)

    # ROEdata_transformed.reset_index(inplace=True)
    ROEdata_transformed.reset_index(inplace=True)
    mergedROE_data = pd.merge(features_quarterly, ROEdata_transformed[['date', 'ROE']], on='date', how='inner')
    targetROE = mergedROE_data['ROE']  

     # Ensure 'date' is in the index before merging
    mergedROE_data.dropna(inplace=True)

    st.write(mergedROE_data)

    featureROE = mergedROE_data.drop(columns=['ROE', 'date'])
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(featureROE, targetROE )
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices], color='b', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([featureROE.columns[i] for i in indices])
    ax.set_xlabel('Relative Importance')
    st.pyplot(fig)
    
    st.subheader(f'{stock_symbol} Stock Price Forecast Prophet')
    # Reset the index to access the 'date' column
    mergedROE_data_reset = mergedROE_data.reset_index()
    columns_to_select = ['ROE', 'date']
    columns_to_select.extend(featureROE.columns[indices[:8]])
    dataROE_ready = mergedROE_data_reset[columns_to_select]
    dataROE_ready.rename(columns={'date': 'ds', 'ROE': 'y'}, inplace=True)
    st.write( dataROE_ready)

    model = Prophet()
    for feature in featureROE.columns[indices[:8]]:
        model.add_regressor(feature)
    model.fit(dataROE_ready)
    # Getting Future forecast
    future = model.make_future_dataframe(periods=period2, freq='Q')
    for feature in featureROE.columns[indices[:8]]:
        future[feature] = np.append(dataROE_ready[feature].values, [dataROE_ready[feature].values[-1]] * (len(future) - len(dataROE_ready)))
    
    forecast = model.predict(future)
    fig = model.plot(forecast)
    st.pyplot(fig)

    dataROE_reset = mergedROE_data.reset_index()
    # Calculate RMSE and MAE for Prophet
    y_true = dataROE_reset['ROE']  # Actual values
    y_pred_prophet = forecast['yhat'].iloc[:len(y_true)]   # Predicted values from Prophet
    # RMSE for Prophet
    rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
    # MAE for Prophet
    mae_prophet = mean_absolute_error(y_true, y_pred_prophet)
    r2_prophet = r2_score(y_true, y_pred_prophet)

    #visualise in streamlit
    st.write(f'ROE Prophet Model RMSE: {rmse_prophet}')
    st.write(f'ROE Prophet Model MAE: {mae_prophet}')
    st.write(f'ROE Prophet Model R2: {r2_prophet}')
    

   # --- ARIMA MODEL ---
    # # ARIMA Model setup and training
    mergedROE_data.set_index('date', inplace=True)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 4) for x in pdq]
    exog_varROE =mergedROE_data.drop(columns=[targetROE.name])
   
    st.write(exog_varROE)
    aic_values = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(mergedROE_data[targetROE.name],
                                                exog=exog_varROE,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                aic_values.append((param, param_seasonal, results.aic))
            except Exception:
                continue

    best_pdq, best_seasonal_pdq, best_aic = sorted(aic_values, key=lambda x: x[2])[0]

    best_model = sm.tsa.statespace.SARIMAX(mergedROE_data[targetROE.name],
                                        exog=exog_varROE,
                                        order=best_pdq,
                                        seasonal_order=best_seasonal_pdq,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
    best_results = best_model.fit(disp=False)

    # Start forecasting from '2020-03-31'
    start_date = pd.to_datetime('2021-03-31')

    # Generate predictions starting from '2020-03-31'
    pred = results.get_prediction(start=start_date, dynamic=False)

    # Get the predicted mean values
    predicted_mean = pred.predicted_mean

    # Extracting  the confidence intervals
    pred_ci = pred.conf_int()

    # to get the actual data from the forecast start date for comparison
    actualROE_data =mergedROE_data.loc[start_date:, targetROE.name]

    st.subheader(f'{stock_symbol} forward forecast ARIMA')
    fig, ax= plt.subplots()
    # lets Plot the observed data and predicted values
    ax = mergedROE_data[targetROE.name].plot(label='Observed', figsize=(14, 7))
    predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.7)

    # Plot the confidence intervals
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=0.2)

        # we Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('ROE')
    plt.legend()
    st.pyplot(fig)

    # Calculate the Mean Squared Error (MSE) for the forecasted period
    mse_Arima = mean_squared_error(actualROE_data , predicted_mean)
    Mae_Arima = mean_absolute_error (actualROE_data , predicted_mean)
    r2_Arima=r2_score(actualROE_data , predicted_mean)
    st.write(f'Mean Squared Error (MSE) from {start_date.date()}: {mse_Arima}')
    st.write(f'Mean absolute Error (Mae) from {start_date.date()}: {Mae_Arima}')
    st.write(f'R2 From {start_date.date()}: {r2_Arima}')

    # lets get the last known values for each exogenous variable (from data2_reset)
    last_known_exog = exog_varROE.iloc[-1]

 
    # Dynamically get the columns from the last known exogenous variables
    columns = last_known_exog.index.tolist()

    #st.write(last_known_exog)

   
    # Create a DataFrame of future exog values by repeating the last known values for the forecast period, we need to keep columns dynamic and use period2 instead, to accounts for quaterly diplay
    future_exog = pd.DataFrame(
    [last_known_exog.values] * period2,  # Repeating the last known exogenous values
    index=pd.date_range(start=last_known_exog.name, periods=period2, freq='Q'),  # Quarterly frequency
    columns=columns  # Dynamic columns from last_known_exog 
    )
    #st.write(future_exog)

    # Forecast with the repeated future exog values
    pred_uc = results.get_forecast(steps=period2, exog=future_exog)

    # Get the confidence intervals of the forecast
    pred_ci = pred_uc.conf_int()

    # Plot the historical data and forecasted values
    st.subheader(f'{stock_symbol} Stock Price Forecast ARIMA')
    fig, ax = plt.subplots()
    ax = mergedROE_data[targetROE.name].plot(label='Historic', figsize=(14, 7))

    # Plot the forecasted mean
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

    # Plot the confidence intervals
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)

    # Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('PE Ratio')
    plt.legend()
    st.pyplot(fig)

with CET1_Ratio: 
    def load_CET1_Ratio(symbol):#define a function to load data
       CET1_ratio = f'Dataset xlsx/{symbol}_CET1 ratio.xlsx'
       data = pd.read_excel(CET1_ratio)
       data['date'] = pd.to_datetime(data['date'])
       data.set_index('date', inplace=True)

       return data
       
    # #displaying the data on streamlit
    # st.subheader(f'{stock_symbol} CET1_Ratio')
    #CET1_data= load_CET1_Ratio(stock_symbol)
    # st.write (CET1_data)
    
    # st.subheader(f'{stock_symbol} CET1 atio')
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x= CET1_data.index, y=  CET1_data['CET1'], mode='lines', name='CET1 Ratio'))
    # fig.layout.update(title_text='CET1 ratio', xaxis_rangeslider_visible=True)
   
    # st.plotly_chart(fig)

# with PE_Ratio:
#     def load_PE_Ratio(symbol):#define a function to load data
#         PE = f'{symbol}_PE Ratio.xlsx'
#         data = pd.read_excel(PE)
#         data['date'] = pd.to_datetime(data['date'])
#         data.set_index('date', inplace=True)

#         return data
#     #displaying the data on streamlit
#     st.subheader(f'{stock_symbol} PE Ratio' )
#     PE_data= load_PE_Ratio(stock_symbol)
#     st.write (PE_data)

    
#     st.subheader(f'{stock_symbol} PE Ratio')
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=PE_data.index, y=PE_data['PE ratio'], mode='lines', name='PE Ratio'))
#     fig.layout.update(title_text='PE ratio data', xaxis_rangeslider_visible=True)
    
#     st.plotly_chart(fig)

    st.subheader("CET1 Ratio Comparison")
    
    if selected_banks:
        fig = go.Figure()
        for bank in selected_banks:
            CET1_data = load_CET1_Ratio(bank)
            avg_cet1 = calculate_average(CET1_data, 'CET1')
            st.metric(label=f'Average CET1 Ratio for {bank}', value=f"{avg_cet1:.2f}%")
            fig.add_trace(go.Scatter(x=CET1_data.index, y=CET1_data['CET1'], mode='lines', name=f'{bank} CET1 Ratio'))

        fig.layout.update(title_text='CET1 Ratio Comparison', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.write("Please select at least one bank for comparison.")

    # with News: 
    #     st.write('News')