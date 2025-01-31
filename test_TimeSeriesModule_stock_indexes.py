# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:47:56 2025

@author: Nicola
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from TimeSeriesModule import GARCH, beta_tGARCH, beta_tEGARCH
from datetime import datetime, timedelta
from scipy.stats import norm
import scipy.stats
from time import time

def get_last_working_date():
    
    today = datetime.now()
    last_working_day = today

    # Step back until a weekday (Monday to Friday) is found
    while last_working_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        last_working_day -= timedelta(days=1)
    
    return last_working_day


if __name__=='__main__':
    
    import warnings
    warnings.filterwarnings("ignore")

    # Step 1: Download data from yahoo finance API 
    
    # List of major stock index tickers
    indexes = {
        "S&P 500 (US)": "^GSPC",
        "Dow Jones (US)": "^DJI",
        "Nasdaq (US)": "^IXIC",
        "Russell 2000 (US)": "^RUT",
        "FTSE 100 (UK)": "^FTSE",
        "DAX (Germany)": "^GDAXI",
        "CAC 40 (France)": "^FCHI",
        "Euro Stoxx 50 (Europe)": "^STOXX50E",
        "IBEX 35 (Spain)": "^IBEX",
        "FTSE MIB (Italy)": "FTSEMIB.MI",
        "AEX (Netherlands)": "^AEX"
    }
    
    # Define the date range
    start_date = "2010-01-04"
    end_date = get_last_working_date().strftime("%Y-%m-%d")
    
    # Loop through the tickers and add adjusted close data to the combined DataFrame
    
    combined_data = pd.DataFrame()
    
    for name, ticker in indexes.items():
        print(f"Processing {name} ({ticker})...")
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        if not data.empty:
            combined_data[name] = data["Adj Close"]
        else:
            print(f"Warning: No data found for {name} ({ticker}). Skipping.")
    
    combined_data = combined_data.interpolate(method='linear')
    
    # # Save combined data to a CSV
    # combined_data.to_csv("all_indexes_data_combined.csv")
    # print("Combined index data saved to 'all_indexes_data_combined.csv'.")
    # print()
    
    fit_garch_results = {}
    fit_tgarch_results = {}
    
    fit_beta_tgarch_results = {}
    fit_beta_tegarch_results = {}
    
    x_gauss = np.linspace(-5, 5, len(combined_data)-1)
    gauss_pdf = norm.pdf(x_gauss, loc=0, scale=1)
    
    trading_dates = list(combined_data.index)[1:]
    
    garch11_model = GARCH(1, 1, density='normal')
    tgarch11_model = GARCH(1, 1, density='tstudent')
    
    beta_tgarch11_model = beta_tGARCH(1, 1)
    beta_tegarch11_model = beta_tEGARCH(1, 1)
    
    global_start_time = time()
    
    for stock_index in combined_data.columns:
        
        stock_start_time = time()
        
        yt = np.log(combined_data[stock_index]).diff().dropna().values
                
        # Fitting GARCH(1,1)
        
        params_garch11 = np.array([0, .00007, 0.1, 0.1, .000015]) # Gaussian case
        params_garch11[1:] = np.log(params_garch11[1:])
        
        res_garch11 = garch11_model.fit(params_garch11, yt)
        
        fit_garch_results[stock_index] = res_garch11
        
        # Fitting tGARCH(1,1)
        
        params_tgarch11 = np.array([0, .00007, 0.1, 0.1, 5, .000015]) # Gaussian case
        params_tgarch11[1:] = np.log(params_tgarch11[1:])
        
        res_tgarch11 = tgarch11_model.fit(params_tgarch11, yt)
        
        fit_tgarch_results[stock_index] = res_tgarch11
        
        DoF = res_tgarch11.params.loc['DoF',0]
        
        tstudent_pdf = scipy.stats.t.pdf(x_gauss, DoF, loc=0, scale=1)
        
        # # Beta_tGARCH(1,1)
        
        # params_beta_tgarch11 = np.array([0, .0001, 0.1, 0.1, 4, .0001]) 
        # # params_beta_tgarch11[1:] = np.log(params_beta_tgarch11[1:])
        
        # res_beta_tgarch11 = beta_tgarch11_model.fit(params_beta_tgarch11, yt, 1, 1)
        
        # fit_beta_tgarch_results[stock_index] = res_beta_tgarch11
        
        # # Beta_tEGARCH(1,1)
        
        # params_beta_tegarch11 = np.array([0, .00001, 0.01, 0.01, 5, .000001]) 
                
        # res_beta_tegarch11 = beta_tegarch11_model.fit(params_beta_tgarch11, yt)
        
        # fit_beta_tegarch_results[stock_index] = res_beta_tegarch11
        
        # Plots
        
        plt.hist(res_garch11.resid, density=True, bins=100, label='GARCH Residuals')
        # plt.hist(res_beta_tgarch11.resid, density=True, bins=100, label='BetaTGARCH Residuals')
        plt.plot(x_gauss, gauss_pdf, color='r', label='std Gauss')
        plt.grid()
        plt.legend()
        plt.title(stock_index)
        plt.show()
        
        plt.hist(res_tgarch11.resid, density=True, bins=100, label='tGARCH Residuals')
        # plt.hist(res_beta_tgarch11.resid, density=True, bins=100, label='BetaTGARCH Residuals')
        plt.plot(x_gauss, tstudent_pdf, color='r', label='std t-Student')
        plt.grid()
        plt.legend()
        plt.title(stock_index)
        plt.show()
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        
        # Top-left plot
        axes[0].plot(trading_dates, yt, label='returns', color='blue')
        axes[0].set_title(stock_index)
        axes[0].legend()
        axes[0].grid()
        
        # Bottom-left plot
        axes[1].plot(trading_dates, np.sqrt(res_garch11.hist_vol), label='garch vol', color='red')
        axes[1].plot(trading_dates, np.sqrt(res_tgarch11.hist_vol), label='tgarch vol', color='green')
        # axes[1].plot(trading_dates, np.sqrt(res_beta_tgarch11.hist_vol), label='beta-tgarch vol', color='green')
        # axes[1].plot(trading_dates, np.sqrt(res_beta_tegarch11.hist_vol), label='beta-tegarch vol', color='black')
        axes[1].set_title(" ")
        axes[1].legend()
        axes[1].grid()
        
        # Adjust layout and show
        plt.tight_layout()
        plt.show()
        
        stock_end_time = time()
        
        print(stock_index + ' enlapsed time : ' + str(stock_end_time - stock_start_time))
        
        # break
    
    global_end_time = time()
    print()
    print('Total Enlapsed time : ' + str(global_end_time - global_start_time))
    
        
        
        
        
        
    
    
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    