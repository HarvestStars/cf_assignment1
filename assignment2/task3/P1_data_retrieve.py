import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import matplotlib.pyplot as plt

# Set up caching and retry logic
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define parameters
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude": 52.37,         # Amsterdam
    "longitude": 4.89,
    "start_date": "2020-08-10",
    "end_date": "2024-08-23",
    "daily": "temperature_2m_mean"
}

# Make the API request
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# Extract daily temperature data
daily = response.Daily()
temperatures = daily.Variables(0).ValuesAsNumpy()

# Construct DataFrame
dates = pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
)

df = pd.DataFrame({
    "date": dates,
    "temperature_2m_mean": temperatures
})
df = df.set_index("date")

# Optional: handle missing data
df = df.interpolate(method="time").dropna()

# Save to CSV
df.to_csv("amsterdam_daily_temperature.csv")

# Optional: quick plot
df.plot(title="Daily Mean Temperature in Amsterdam (2m)", figsize=(12, 4))
plt.ylabel("Temperature (°C)", fontsize=14)
plt.xlabel("Date", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/amsterdam_daily_temperature.png", dpi=300, bbox_inches="tight")
plt.show()

# Show first few rows
print(df.head())
