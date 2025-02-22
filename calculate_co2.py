import random
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime, timedelta
from pyvis.network import Network
import webbrowser
import os
from pathlib import Path

# Emission factors in kg CO2e per euro spent
CO2_EMISSION_FACTORS = {
    # ----- EU-Specific Categories -----
    ('Food', 'Groceries'): 0.5,         # Average for EU food supply chain
    ('Food', 'Restaurant'): 0.7,        # Includes energy for cooking and service
    ('Food', 'Coffee Shop'): 0.4,        # Lower than restaurants due to smaller portions
    ('Transportation', 'Public Transit'): 0.06,  # EU average for public transport
    ('Transportation', 'Fuel'): 1.28,    # Based on €1.80/L petrol price (2.3kg CO2/L)
    ('Transportation', 'Ride Sharing'): 0.85,    # Similar to personal vehicles
    ('Housing', 'Rent'): 0.08,           # Building maintenance and heating
    ('Housing', 'Utilities'): 0.35,      # Average for EU household utilities
    ('Healthcare', 'Pharmacy'): 0.15,     # Pharmaceutical production and distribution
    ('Healthcare', 'Dental'): 0.1,       # Medical equipment energy use
    ('Electronics', 'TV'): 0.25,         # Manufacturing and transportation
    ('Electronics', 'Smartphone'): 0.18, # Manufacturing emissions
    ('Travel', 'Hotels'): 0.28,          # Hotel operations energy use
    ('Travel', 'Airfare'): 0.82,         # EU flight emissions average
    ('Education', 'Tuition'): 0.04,      # Institutional energy use
    ('Education', 'Textbooks'): 0.12,    # Paper production and distribution
    ('Subscriptions', 'Streaming'): 0.03, # Data center energy use
    ('Software', 'Apps'): 0.02,          # Software development and maintenance
    ('Finance', 'Wire Transfer'): 0.01,  # Banking infrastructure
    ('Finance', 'ATM Withdrawal'): 0.01, # ATM operation energy
    ('Shopping', 'Clothing'): 0.35,      # Fast fashion environmental impact
    ('Shopping', 'Furniture'): 0.22,     # Manufacturing and logistics
    ('Cryptocurrency', 'Trading'): 0.75, # Blockchain energy consumption estimate
    ('Sustainability', 'Carbon Offsets'): -55.0,  # Based on €18/ton market price

    # ----- Global Categories -----
    ('Healthcare', 'Medicines'): 0.12,
    ('Healthcare', 'Consultation'): 0.08,
    ('Transportation', 'Public'): 0.06,
    ('Housing', 'Mortgage'): 0.07,
    ('Housing', 'Property Tax'): 0.0,
    ('Housing', 'Home Insurance'): 0.0,
    ('Transportation', 'Car Payment'): 0.05,
    ('Transportation', 'Auto Insurance'): 0.01,
    ('Food', 'Food Delivery'): 0.65,
    ('Food', 'Bulk Stores'): 0.32,
    ('Entertainment', 'Gaming'): 0.15,
    ('Entertainment', 'Concerts'): 0.25,
    ('Personal Finance', 'ATM Fee'): 0.0,
    ('Shopping', 'Electronics'): 0.2,
    ('Shopping', 'Cosmetics'): 0.28,
    ('Childcare', 'Daycare'): 0.12,
    ('Pets', 'Veterinary'): 0.1,
    ('Utilities', 'Electric'): 1.55,     # EU average grid intensity (0.3kg/kWh @ €0.20/kWh)
    ('Utilities', 'Internet'): 0.04,
    ('Fitness', 'Gym Membership'): 0.15,
    ('Gifts', 'Birthday'): 0.3,
    ('Investments', 'Stocks'): 0.06,
    ('Taxes', 'Income Tax'): 0.0,
    ('Business', 'Advertising'): 0.12,
    ('Legal', 'Consultation'): 0.05,
    ('Charity', 'Donations'): 0.0,
    ('Insurance', 'Life'): 0.0,
    ('Repairs', 'Home'): 0.18,
    ('Hobbies', 'Crafts'): 0.12,
    ('Alcohol', 'Liquor Store'): 0.42,
    ('Memberships', 'Clubs'): 0.1,
    ('Fees', 'Late Payment'): 0.0,
    ('Beauty', 'Spa'): 0.25,
    ('Storage', 'Units'): 0.15,
    ('Public Transit', 'Passes'): 0.06,
    ('Laundry', 'Services'): 0.35,
    ('Postage', 'Shipping'): 0.52,
    ('Office', 'Supplies'): 0.1,
    ('Photography', 'Services'): 0.12,
    ('Music', 'Instruments'): 0.22,
    ('Books', 'Ebooks'): 0.03,
    ('Security', 'Systems'): 0.15,
    ('Cleaning', 'Services'): 0.2,
    ('Maintenance', 'Vehicle'): 0.25,
    ('VPN', 'Services'): 0.03,
    ('Cloud', 'Storage'): 0.06,
    ('Farming', 'Supplies'): 0.38,
    ('Camping', 'Gear'): 0.25,
    ('Meditation', 'Classes'): 0.06,
    ('Therapy', 'Sessions'): 0.07,
    ('Pest Control', 'Services'): 0.32,
    ('Recycling', 'Fees'): -0.15,
    ('EV Charging', 'Stations'): 1.55
}

def add_co2_emissions(df):
    """
    Adds CO2_Emission column to dataframe based on category/subcategory emission factors
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'Category', 'Subcategory', and 'Amount' columns
        
    Returns:
        pd.DataFrame: Dataframe with added CO2_Emission column
    """
    def calculate_co2(row):
        key = (row['Category'], row['Subcategory'])
        factor = CO2_EMISSION_FACTORS.get(key, 0)
        co2e = row['Amount'] * factor
        co2e = co2e if co2e >= 0 else 0
        return co2e
    
    df['CO2_Emission'] = df.apply(calculate_co2, axis=1)
    return df

if __name__ == "__main__":
    df = pd.read_csv("transactions_generated.csv")

    add_co2_emissions(df)

    # Save df
    df.to_csv("transactions_generated.csv", index=False)
