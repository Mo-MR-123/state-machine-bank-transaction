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

# Housing ranges based on national average rent/mortgage data 
# Food categories using USDA food spending reports 
# Healthcare costs aligned with CMS national health expenditure data
# Transportation ranges from AAA driving cost calculations
# Entertainment values using Bureau of Labor Statistics consumer expenditure surveys
# Technology ranges based on average SaaS subscription costs
# Education amounts from College Board tuition trend reports
# Business expenses using SBA small business cost guidelines

# Adjustments based on European country statistics:
# Food & Transport ranges refined using ECB's SPACE study on POS payment behaviors, showing 52% of sub-€50 transactions are cash-based
# Housing costs aligned with Euro Area Report's analysis of regional disparities
# Travel ranges reflect €60.5B surplus in EU services trade (transport + accommodation)
# Education costs adjusted for EU's mixed public/private tuition models
# Digital services calibrated to ECB's finding that 75% of subscriptions are under €15/mo
CAT_AND_SUBCATS_WITH_RANGES = {
    # ----- EU-Specific Ranges (Priority) -----
    ('Food', 'Groceries'): (10.0, 150.0),
    ('Food', 'Restaurant'): (15.0, 100.0),
    ('Food', 'Coffee Shop'): (2.5, 12.0),
    ('Transportation', 'Public Transit'): (1.5, 25.0),
    ('Transportation', 'Fuel'): (30.0, 120.0),
    ('Transportation', 'Ride Sharing'): (8.0, 65.0),
    ('Housing', 'Rent'): (500.0, 3500.0),
    ('Housing', 'Utilities'): (80.0, 400.0),
    ('Healthcare', 'Pharmacy'): (5.0, 100.0),
    ('Healthcare', 'Dental'): (50.0, 1200.0),
    ('Electronics', 'TV'): (300.0, 5000.0),
    ('Electronics', 'Smartphone'): (200.0, 1500.0),
    ('Travel', 'Hotels'): (80.0, 400.0),
    ('Travel', 'Airfare'): (100.0, 1500.0),
    ('Education', 'Tuition'): (300.0, 10000.0),
    ('Education', 'Textbooks'): (30.0, 300.0),
    ('Subscriptions', 'Streaming'): (5.0, 25.0),
    ('Software', 'Apps'): (0.99, 150.0),
    ('Finance', 'Wire Transfer'): (15.0, 25000.0),
    ('Finance', 'ATM Withdrawal'): (20.0, 400.0),
    ('Shopping', 'Clothing'): (20.0, 500.0),
    ('Shopping', 'Furniture'): (200.0, 4000.0),
    ('Cryptocurrency', 'Trading'): (10.0, 10000.0),
    ('Sustainability', 'Carbon Offsets'): (5.0, 500.0),

    # ----- Global Ranges (Non-Conflicting) -----
    ('Healthcare', 'Medicines'): (5.0, 50.0),
    ('Healthcare', 'Consultation'): (30.0, 200.0),
    ('Transportation', 'Public'): (2.0, 20.0),
    ('Housing', 'Mortgage'): (1200.0, 3500.0),
    ('Housing', 'Property Tax'): (200.0, 2500.0),
    ('Housing', 'Home Insurance'): (50.0, 400.0),
    ('Transportation', 'Car Payment'): (200.0, 800.0),
    ('Transportation', 'Auto Insurance'): (80.0, 300.0),
    ('Food', 'Food Delivery'): (15.0, 150.0),
    ('Food', 'Bulk Stores'): (50.0, 400.0),
    ('Entertainment', 'Gaming'): (10.0, 100.0),
    ('Entertainment', 'Concerts'): (50.0, 500.0),
    ('Personal Finance', 'ATM Fee'): (2.0, 5.0),
    ('Shopping', 'Electronics'): (50.0, 5000.0),
    ('Shopping', 'Cosmetics'): (10.0, 200.0),
    ('Childcare', 'Daycare'): (400.0, 2000.0),
    ('Pets', 'Veterinary'): (50.0, 1500.0),
    ('Utilities', 'Electric'): (50.0, 400.0),
    ('Utilities', 'Internet'): (40.0, 150.0),
    ('Fitness', 'Gym Membership'): (20.0, 200.0),
    ('Gifts', 'Birthday'): (15.0, 500.0),
    ('Investments', 'Stocks'): (50.0, 10000.0),
    ('Taxes', 'Income Tax'): (500.0, 20000.0),
    ('Business', 'Advertising'): (50.0, 5000.0),
    ('Legal', 'Consultation'): (100.0, 1000.0),
    ('Charity', 'Donations'): (10.0, 10000.0),
    ('Insurance', 'Life'): (50.0, 500.0),
    ('Repairs', 'Home'): (75.0, 5000.0),
    ('Hobbies', 'Crafts'): (10.0, 300.0),
    ('Alcohol', 'Liquor Store'): (15.0, 200.0),
    ('Memberships', 'Clubs'): (25.0, 500.0),
    ('Fees', 'Late Payment'): (25.0, 50.0),
    ('Beauty', 'Spa'): (50.0, 500.0),
    ('Storage', 'Units'): (50.0, 300.0),
    ('Public Transit', 'Passes'): (20.0, 300.0),
    ('Laundry', 'Services'): (5.0, 50.0),
    ('Postage', 'Shipping'): (3.0, 50.0),
    ('Office', 'Supplies'): (10.0, 500.0),
    ('Photography', 'Services'): (50.0, 2000.0),
    ('Music', 'Instruments'): (100.0, 5000.0),
    ('Books', 'Ebooks'): (5.0, 50.0),
    ('Security', 'Systems'): (100.0, 5000.0),
    ('Cleaning', 'Services'): (80.0, 400.0),
    ('Maintenance', 'Vehicle'): (50.0, 2000.0),
    ('VPN', 'Services'): (5.0, 15.0),
    ('Cloud', 'Storage'): (2.0, 200.0),
    ('Farming', 'Supplies'): (50.0, 5000.0),
    ('Camping', 'Gear'): (30.0, 1500.0),
    ('Meditation', 'Classes'): (15.0, 200.0),
    ('Therapy', 'Sessions'): (80.0, 250.0),
    ('Pest Control', 'Services'): (75.0, 400.0),
    ('Recycling', 'Fees'): (10.0, 100.0),
    ('EV Charging', 'Stations'): (5.0, 50.0)
}

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

COUNTRY_WEIGHTS = {
    # Median liquid assets by country (€)
    'GER': 6300,   # Germany
    'FR': 5800,    # France
    'IT': 4100,    # Italy
    'ES': 3500,    # Spain
    'NL': 12500,   # Netherlands
    'BE': 7500,    # Belgium
    'CH': 24500,   # Switzerland (non-EU but included for completeness)
    'SE': 10500,   # Sweden
    'PL': 2200     # Poland
}

# INITIAL_BALANCE = (1_000, 1_000_000)
START_DATE = "20-10-2020"
NUM_TRANSACTIONS_TO_GENERATE = 150_000
NUM_UNIQUE_IBANS = 5_000

# Global log to record transitions
transition_log = []
logging_graph_done = False

# Decorator that wraps a node's next() method to log transitions.
def log_transition(func):
    def wrapper(self, context, *args, **kwargs):
        result = func(self, context, *args, **kwargs)
        from_node = type(self).__name__
        to_node = type(result).__name__ if result is not None else None
        transition_log.append((from_node, to_node))
        return result
    return wrapper

# Metaclass that automatically wraps the 'next' method of any Node subclass.
class NodeMeta(ABCMeta):
    def __new__(cls, name, bases, attrs):
        # Check if a next method is defined and wrap it.
        if 'next' in attrs and callable(attrs['next']):
            attrs['next'] = log_transition(attrs['next'])
        return super().__new__(cls, name, bases, attrs)

# -------------------------------
# Abstract Base Class for Nodes
# -------------------------------
class Node(ABC, metaclass=NodeMeta):
    @abstractmethod
    def execute(self, context):
        """Process current state and update context."""
        pass

    @abstractmethod
    def next(self, context):
        """Determine and return the next node based on context."""
        pass

# -------------------------------
# Concrete Node Classes for Transfer Transaction
# -------------------------------

class CategorySelectionNode(Node):
    def execute(self, context):
        # Select a random (Category, Subcategory) pair from the provided list
        selected_pair = random_cat_and_subcat()
        context["category"] = selected_pair[0]
        context["subcategory"] = selected_pair[1]
    
    def next(self, context):
        return TransferAmountNode()

class TransferAmountNode(Node):
    def execute(self, context):
        sender_balance = context["sender_info"]["balance"]
        category = context["category"]
        subcategory = context["subcategory"]
        min_amount, max_amount = get_amount_range(category, subcategory)
        
        # If sender_balance is too low for the minimum required, set amount to 0
        if sender_balance < min_amount:
            context["amount"] = 0
        else:
            # Determine the upper limit as the minimum of max_amount and sender's balance
            upper_limit = min(max_amount, sender_balance)
            context["amount"] = random.uniform(min_amount, upper_limit)
    def next(self, context):
        return BalanceUpdateNode()

class BalanceUpdateNode(Node):
    def execute(self, context):
        amount = context["amount"]
        # Update sender balance
        sender_prev = context["sender_info"]["balance"]
        sender_new = sender_prev - amount
        if sender_new < 0:
            raise ValueError("Sender balance cannot be negative!")
        context["sender_info"]["new_balance"] = sender_new
    def next(self, context):
        return DateTransactionNode()

class DateTransactionNode(Node):
    def execute(self, context):
        # Use the global transaction counter and a start_date provided in the context.
        counter = context["global_counter"]
        start_date = context["start_date"]
        # Every 100 transactions represent one day
        current_date = start_date + timedelta(days=(counter // context["transactions_per_day"]))
        context["transaction_date"] = current_date.strftime("%Y-%m-%d")
        # Generate a unique transaction ID (e.g., TXN00000001) possible values: 10^5
        context["transaction_id"] = f"TXN{counter:08d}"
    def next(self, context):
        return EndNode()

class EndNode(Node):
    def execute(self, context):
        # Terminal state; nothing further to process.
        pass
    def next(self, context):
        return None

# -------------------------------
# IBAN Generation with Fixed Demographics
# -------------------------------
def random_cat_and_subcat():
    global CAT_AND_SUBCATS_WITH_RANGES
    return random.choice(list(CAT_AND_SUBCATS_WITH_RANGES.keys()))

def get_amount_range(category, subcategory):
    global CAT_AND_SUBCATS_WITH_RANGES
    return CAT_AND_SUBCATS_WITH_RANGES.get((category, subcategory), (1.0, 1000.0))

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
        return row['Amount'] * factor
    
    df['CO2_Emission'] = df.apply(calculate_co2, axis=1)
    return df

def generate_initial_balance(age, country):
    global COUNTRY_WEIGHTS
    """Generate realistic bank balances based on EU financial statistics"""
    # Base parameters from ECB Household Finance and Consumption Survey (HFCS 2020)
    # country_weights = {
    #     # Median liquid assets by country (€)
    #     'GER': 6300,   # Germany
    #     'FR': 5800,    # France
    #     'IT': 4100,    # Italy
    #     'ES': 3500,    # Spain
    #     'NL': 12500,   # Netherlands
    #     'BE': 7500,    # Belgium
    #     'CH': 24500,   # Switzerland (non-EU but included for completeness)
    #     'SE': 10500,   # Sweden
    #     'PL': 2200     # Poland
    # }
    
    # Age adjustment factors (ECB age-based liquidity patterns)
    age_factor = np.clip(age/30, 0.5, 2.5)  # Peaks around 55-65
    
    # Base distribution using EU wealth stratification:
    # 40%: Low liquidity (under €3,000)
    # 35%: Medium liquidity (€3,000-€25,000)
    # 20%: High liquidity (€25,000-€100,000)
    # 5%: Very high liquidity (>€100,000)
    
    base_balance = 0
    rand = np.random.random()
    
    if rand < 0.40:  # Low liquidity group
        base_balance = np.random.gamma(shape=1.5, scale=800)
    elif rand < 0.75:  # Medium liquidity
        base_balance = np.random.lognormal(mean=np.log(8500), sigma=0.7)
    elif rand < 0.95:  # High liquidity
        base_balance = np.random.pareto(2.5) * 25000
    else:  # Very high liquidity
        base_balance = np.abs(np.random.normal(loc=150000, scale=80000))
    
    # Apply country weighting and age factor
    country_base = COUNTRY_WEIGHTS.get(country, None)
    balance = base_balance * (country_base / 6300) * age_factor
    
    # Final adjustments and constraints
    return np.clip(balance, 1_000, 750_000)  # Min €500, max €750.000

def generate_ibans(n):
    global COUNTRY_WEIGHTS
    """
    Generate n IBANs with an initial random balance and fixed demographic details.
    Each IBAN is stored in a dictionary with the following keys:
      - balance: current account balance.
      - country: fixed country code.
      - age: fixed age.
      - gender: fixed gender.
    """
    ibans = {}
    # countries = ["NL", "GER", "FR", "IT", "ES", "UK", "BE", "CH", "SE", "NO"]
    for i in range(n):
        iban = f"IBAN{i:05d}"
        # initial_balance = random.uniform(*INITIAL_BALANCE)
        country = random.choice(list(COUNTRY_WEIGHTS.keys()))
        age = int(np.clip(np.random.normal(40, 15), 18, 90))
        gender = random.choices(["male", "female"], weights=[50, 50])[0]
        ibans[iban] = {
            "balance": generate_initial_balance(age, country),
            "country": country,
            "age": age,
            "gender": gender,
        }
    return ibans

# -------------------------------
# Run State Machine for a Single Transfer Transaction
# -------------------------------
def run_state_machine_for_transaction(sender_iban, ibans, global_counter, start_date):
    """
    Runs the node chain for a single transfer transaction.
    The context contains sender and receiver info (demographics and balance).
    """
    context = {
        "sender_IBAN": sender_iban,
        "sender_info": ibans[sender_iban].copy(),
        "amount": 0,
        "transaction_date": None,
        "transaction_id": None,
        "global_counter": global_counter,
        "transactions_per_day": 100,
        "start_date": start_date,
        "category": None,
        "subcategory": None,
    }
    
    current_node = CategorySelectionNode()
    while current_node is not None:
        current_node.execute(context)
        current_node = current_node.next(context)
    
    return context

# -------------------------------
# Main State Machine Loop for Multiple Transactions
# -------------------------------
def state_machine(ibans, num_transactions, start_date_str):
    global logging_graph_done

    """
    For each transaction:
      - Select a sender IBAN (with sufficient balance) and a distinct receiver IBAN.
      - Run the state machine chain to process the transfer (amount, balance updates, date, transaction id).
      - Update the global IBAN balances.
      - Record all details into a transaction record.
    """
    transaction_data = []
    start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
    global_counter = 0
    attempts = 0
    max_attempts = num_transactions * 5  # Prevent infinite loops
    ibans_keys = list(ibans.keys())
    
    while len(transaction_data) < num_transactions and attempts < max_attempts:
        attempts += 1
        # Select sender: must have balance greater than 1 to allow a nonzero transfer.
        valid_senders = [iban for iban, info in ibans.items() if info["balance"] > 1]
        if not valid_senders:
            # If no valid sender exists, break out of the loop.
            break
        sender_iban = random.choice(valid_senders)
        
        # Run the state machine for this transaction.
        context = run_state_machine_for_transaction(sender_iban, ibans, global_counter, start_date)

        # Skip transactions that are 0. 
        if abs(context["amount"]) < 1e-6:
            continue

        if not logging_graph_done:
            # Display the final context of the transaction.
            print("Context:")
            for key, value in context.items():
                print(f"{key}: {value}")
            
            # Display the dynamically inferred transitions.
            print("\nTransition Log (from_node -> to_node):")
            for from_node, to_node in transition_log:
                print(f"{from_node} -> {to_node}")
            
            logging_graph_done = True
        
        # Update the global IBAN balances.
        ibans[sender_iban]["balance"] = context["sender_info"]["new_balance"]
        
        # Build the complete transaction record.
        record = {
            "TransactionID": context["transaction_id"],
            "TransactionDate": context["transaction_date"],
            "IBAN": sender_iban,
            "PreviousBalance": context["sender_info"]["balance"],
            "NewBalance": context["sender_info"]["new_balance"],
            "Country": context["sender_info"]["country"],
            "Age": context["sender_info"]["age"],
            "Gender": context["sender_info"]["gender"],
            "Amount": context["amount"],
            "Category": context["category"],
            "Subcategory": context["subcategory"],
        }
        transaction_data.append(record)
        global_counter += 1
        
    df = pd.DataFrame(transaction_data)
    return df

def show_state_machine_networkx():
    # Create a Pyvis directed network
    net = Network(height="750px", width="100%", directed=True)
    
    # -------------------------------
    # Original Bank Transaction State Machine
    # -------------------------------
    original_nodes = [
        "CategorySelectionNode",
        "TransferAmountNode",
        "BalanceUpdateNode",
        "DateTransactionNode",
        "EndNode"
    ]
    
    for node in original_nodes:
        net.add_node(node, label=node, color="lightgreen")
    
    net.add_edge("CategorySelectionNode", "TransferAmountNode")
    net.add_edge("TransferAmountNode", "BalanceUpdateNode")
    net.add_edge("BalanceUpdateNode", "DateTransactionNode")
    net.add_edge("DateTransactionNode", "EndNode")
    
    # Optionally, add physics configuration buttons.
    net.show_buttons(filter_=['physics'])
    
    # Save the graph to an HTML file.
    output_file = "state_machine.html"
    net.save_graph(output_file)
    
    # Open the generated file in your default web browser.
    abs_path = os.path.abspath(output_file)
    webbrowser.open(f"file://{abs_path}")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Generate 50 IBANs with initial balances and fixed demographics.
    ibans = generate_ibans(NUM_UNIQUE_IBANS)
    
    # Run the state machine for a specified number of transactions.
    # Here, every 100 transactions correspond to one day starting from 2020-10-20.
    df = state_machine(
        ibans, 
        num_transactions=NUM_TRANSACTIONS_TO_GENERATE, 
        start_date_str=START_DATE,
    )

    add_co2_emissions(df)

    # Save df
    df.to_csv("transactions_generated.csv", index=False)

    # show_state_machine_networkx()

