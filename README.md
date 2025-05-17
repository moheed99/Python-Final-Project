# NeoFinance Dashboard

A futuristic-themed financial dashboard with ML-powered analytics, real-time stock data, and interactive visualizations.

## Features

- Real-time stock data visualization using Yahoo Finance API
- Machine Learning models for financial prediction
- Interactive financial data analysis
- Futuristic neon themed UI
- Multiple visualization types for financial analysis

## Deployment Instructions for Streamlit Cloud

1. **Create a GitHub repository** and upload all the files from this project.

2. **Important Files**:
   - `app.py`: Main application file
   - `streamlit_requirements.txt`: Rename this to `requirements.txt` in your GitHub repository
   - `.streamlit/config.toml`: Streamlit configuration
   - `.streamlit/secrets.toml`: For storing any API keys (create securely in Streamlit Cloud)

3. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Set the main file path to `app.py`
   - Deploy your app

## Local Development

To run this application locally:

1. Install requirements:
   ```
   pip install -r streamlit_requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## Technologies Used

- Streamlit
- Pandas
- NumPy
- Yahoo Finance API (yfinance)
- scikit-learn
- Matplotlib
- Plotly

## Note

This application uses the Yahoo Finance API for real-time stock data. No API key is required for basic usage.