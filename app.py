import requests
import os
import pandas as pd
import json
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from flask import Flask, request, jsonify 

app = Flask(__name__)

url = "https://api.rootfi.dev/v3/accounting/bill_payments"
url_1 = "https://api.rootfi.dev/v3/accounting/bill_credit_notes"
url_2 = "https://api.rootfi.dev/v3/accounting/Accounts"
url_a = "https://api.rootfi.dev/v3/accounting/Invoice_Payments"
url_b = "https://api.rootfi.dev/v3/accounting/Invoice_Credit_Notes"
url_c = "https://api.rootfi.dev/v3/accounting/Invoices"

llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0.5,
    max_tokens=1024,
)

llm_2 = ChatOpenAI(
    model="gpt-4o-mini", temperature=0
)

def fetch_all_bill_payments(api_key, company_id, base_url, limit=1000):
    headers = {"api_key": api_key}
    all_bill_payments = []
    next_cursor = None

    while True:
        params = {
            "limit": limit,
            "rootfi_company_id[eq]": company_id
        }
        if next_cursor:
            params["next"] = next_cursor

        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        all_bill_payments.extend(data["data"])

        if not data.get("next"):
            break

        next_cursor = data["next"]

    return all_bill_payments

df = pd.json_normalize(fetch_all_bill_payments(os.environ["ROOTIFY_API"], os.environ["COMPANY_ID"],url))
df_2 = pd.json_normalize(fetch_all_bill_payments(os.environ["ROOTIFY_API"], os.environ["COMPANY_ID"],url_1))
df_3 = pd.json_normalize(fetch_all_bill_payments(os.environ["ROOTIFY_API"], os.environ["COMPANY_ID"],url_2))
df_a = pd.json_normalize(fetch_all_bill_payments(os.environ["ROOTIFY_API"], os.environ["COMPANY_ID"],url_a))
df_b = pd.json_normalize(fetch_all_bill_payments(os.environ["ROOTIFY_API"], os.environ["COMPANY_ID"],url_b))
df_c = pd.json_normalize(fetch_all_bill_payments(os.environ["ROOTIFY_API"], os.environ["COMPANY_ID"],url_c))

columns_to_drop = [
    'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
    'rootfi_company_id', 'platform_id', 'platform_unique_id', 'bill_id',
    'credit_note_id', 'updated_at', 'document_number', 'currency_rate',
    'custom_fields', 'currency_id', 'payment_mode'
]

# Drop the columns from the DataFrame
df = df.drop(columns=columns_to_drop)

rename_mapping = {
    'payment_id': 'Bill_Payment_ID',
    'account_id': 'ID',
    'contact_id': 'Vendor_Contact_ID',
    'amount': 'Payment_Amount',
    'memo': 'Payment_Memo',
    'payment_date': 'Bill_Payment_Date'
}

# Rename columns in the DataFrame
df = df.rename(columns=rename_mapping)

columns_to_drop_2 = [
    'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
    'rootfi_company_id', 'platform_id', 'platform_unique_id','currency_id',
    'total_discount' ,	'tax_amount', 'document_number',	'remaining_credit',
    'custom_fields',	'updated_at', 	'bill_ids'
]

# Drop the columns from the DataFrame
df_2 = df_2.drop(columns=columns_to_drop_2)

rename_mapping_2 = {
    'contact_id': 'Vendor_Contact_ID',
    'total_amount': 'Credit_Note_Total_Amount',
    'posted_date': 'Credit_Note_Posted_Date',
    'memo': 'Credit_Note_Memo',
    'status': 'Credit_Note_Status'
}

# Rename columns in the DataFrame
df_2 = df_2.rename(columns=rename_mapping_2)

columns_to_drop_a = [
    'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
    'rootfi_company_id', 'platform_id', 'platform_unique_id','invoice_id',
    'credit_note_id', 'updated_at', 'document_number', 'currency_rate',
    'custom_fields', 'currency_id', 'payment_mode'
]

# Drop the columns from the DataFrame
df_a = df_a.drop(columns=columns_to_drop_a)

rename_mapping = {
    'payment_id': 'Invoice_Payment_ID',
    'account_id': 'ID',
    'contact_id': 'Vendor_Contact_ID',
    'amount': 'Payment_Amount',
    'memo': 'Payment_Memo',
    'payment_date': 'Invoice_Payment_Date'
}

# Rename columns in the DataFrame
df_a = df_a.rename(columns=rename_mapping)

columns_to_drop_b = [
    'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
    'rootfi_company_id', 'platform_id', 'platform_unique_id','currency_id',
     'document_number', 'custom_fields',	'tax_amount', 	'total_discount', 'remaining_credit', 'status',
       'updated_at','invoice_ids'
]

# Drop the columns from the DataFrame
df_b = df_b.drop(columns=columns_to_drop_b)

rename_mapping_b = {
    'payment_id': 'Invoice_Payment_ID',
    'contact_id':'Vendor_Contact_ID',
    'total_amount' : 'Payment_Amount',
    'memo': 'Payment_Memo',
    'posted_date': 'Invoice_Payment_Date'
}

# Rename columns in the DataFrame
df_b = df_b.rename(columns=rename_mapping_b)

columns_to_drop_c = [
    'rootfi_id',	'rootfi_created_at',	'rootfi_updated_at',	'rootfi_deleted_at',
    'rootfi_company_id',	'platform_id',	'platform_unique_id',	'currency_id',		'custom_fields',
  	'updated_at', 'currency_rate', 'document_number', 'total_discount','sub_total', 'tax_amount', 'amount_due', 'due_date', 'status', 'currency_rate', 'sales_order_ids']

# Drop the columns from the DataFrame
df_c = df_c.drop(columns=columns_to_drop_c)

rename_mapping = {
    'contact_id': 'Vendor_Contact_ID',
    'total_amount': 'Payment_Amount',
    'memo': 'Payment_Memo',
    'posted_date': 'Invoice_Payment_Date'
}

# Rename columns in the DataFrame
df_c = df_c.rename(columns=rename_mapping)

columns_to_drop_3 = [
    'rootfi_id',	'rootfi_created_at',	'rootfi_updated_at',	'rootfi_deleted_at',	'rootfi_company_id',	'platform_id',	'platform_unique_id',	'currency_id',
    'parent_account_id',	'custom_fields',	'nominal_code',	'description',	'current_balance',	'updated_at',	'status'	,'category'
]

# Drop the columns from the DataFrame
df_3 = df_3.drop(columns=columns_to_drop_3)


rename_mapping = {
    'name': 'ID',
    'sub_category': 'Expense_Category'
}

# Rename columns in the DataFrame
df_3 = df_3.rename(columns=rename_mapping)

df = df.merge(
    df_3,
    on='ID',
    how='left'
)

create_table_query_1 = """
CREATE TABLE IF NOT EXISTS Bill_Payments (
    Bill_Payment_ID VARCHAR(255) ,
    ID VARCHAR(255),
    Vendor_Contact_ID VARCHAR(255),
    Payment_Amount DECIMAL(15, 2),
    Payment_Memo TEXT,
    Bill_Payment_Date VARCHAR(50),
    Expense_Category VARCHAR(255)
);
"""

create_table_query_2 = """
CREATE TABLE IF NOT EXISTS Bill_Credit_Notes (
    Vendor_Contact_ID VARCHAR(255),
    Credit_Note_Total_Amount DECIMAL(15, 2),
    Credit_Note_Posted_Date VARCHAR(50),
    Credit_Note_Memo TEXT,
    Credit_Note_Status VARCHAR(50)
);

"""

create_table_query_c = """
CREATE TABLE IF NOT EXISTS Invoices (
    Vendor_Contact_ID VARCHAR(255),
    Payment_Amount DECIMAL(15, 2),
    Payment_Memo TEXT,
    Invoice_Payment_Date VARCHAR(50)
);
"""

create_table_query_a = """
CREATE TABLE IF NOT EXISTS Invoice_Payments (
    Invoice_Payment_ID VARCHAR(255) ,
    ID VARCHAR(255),
    Vendor_Contact_ID VARCHAR(255),
    Payment_Amount DECIMAL(15, 2),
    Payment_Memo TEXT,
    Invoice_Payment_Date VARCHAR(50)
);
"""

create_table_query_b = """
CREATE TABLE IF NOT EXISTS Invoice_Credit_Notes (
    Vendor_Contact_ID VARCHAR(255),
    Payment_Amount DECIMAL(15, 2),
    Payment_Memo TEXT,
    Invoice_Payment_Date VARCHAR(50)
);
"""

conn = sqlite3.connect("Bills.db")
cursor = conn.cursor()
cursor.execute(create_table_query_1)
cursor.execute(create_table_query_2)
cursor.execute(create_table_query_a)
cursor.execute(create_table_query_b)
cursor.execute(create_table_query_c)
# Step 5: Insert Data from DataFrame to SQLite Table
# Using 'if_exists="replace"' to overwrite any existing table
df.to_sql("Bill_Payments", conn, if_exists="append", index=False)
df_2.to_sql("Bill_Credit_Notes", conn, if_exists="append", index=False)
df_a.to_sql("Invoice_Payments", conn, if_exists="append", index=False)
df_b.to_sql("Invoice_Credit_Notes", conn, if_exists="append", index=False)
df_c.to_sql("Invoices", conn, if_exists="append", index=False)

# Commit the changes and close the connection
conn.commit()
conn.close()
print("Data successfully inserted into the Bill DB.")

data_base = SQLDatabase.from_uri("sqlite:///Bills.db")

def get_average_cash_flows(inputs: str) -> str:
    """
    Calculate average cash inflows and outflows for a selected duration.
    Accepts input as a JSON-like string containing 'start_date' and 'end_date'.
    """
    import json
    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        start_date = input_dict["start_date"]
        end_date = input_dict["end_date"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    inflow_query = """
        SELECT AVG(Payment_Amount) as avg_inflow
        FROM Invoice_Payments
        WHERE Invoice_Payment_Date BETWEEN ? AND ?
    """

    outflow_query = """
        SELECT AVG(Payment_Amount) as avg_outflow
        FROM Bill_Payments
        WHERE Bill_Payment_Date BETWEEN ? AND ?
    """

    cursor.execute(inflow_query, (start_date, end_date))
    avg_inflow = cursor.fetchone()[0] or 0

    cursor.execute(outflow_query, (start_date, end_date))
    avg_outflow = cursor.fetchone()[0] or 0

    conn.close()

    return f"Average Inflow: {avg_inflow:.2f}, Average Outflow: {avg_outflow:.2f}"

def analyze_monthly_trends(year: str) -> str:
    """
    Analyze monthly cash flow trends for a specific year.
    Args:
        year (str): Year to analyze
    Returns:
        String containing monthly trend analysis
    """
    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    monthly_query = """
        WITH monthly_inflows AS (
    SELECT
        strftime('%Y-%m', Invoice_Payment_Date) as month,
        SUM(Payment_Amount) as total_inflow
    FROM Invoice_Payments
    WHERE strftime('%Y', Invoice_Payment_Date) = ?
    GROUP BY month
),
monthly_outflows AS (
    SELECT
        strftime('%Y-%m', Bill_Payment_Date) as month,
        SUM(Payment_Amount) as total_outflow
    FROM Bill_Payments
    WHERE strftime('%Y', Bill_Payment_Date) = ?
    GROUP BY month
),
combined_data AS (
    SELECT
        mi.month as month,
        mi.total_inflow as total_inflow,
        mo.total_outflow as total_outflow
    FROM monthly_inflows mi
    LEFT JOIN monthly_outflows mo ON mi.month = mo.month
    UNION
    SELECT
        mo.month as month,
        mi.total_inflow as total_inflow,
        mo.total_outflow as total_outflow
    FROM monthly_outflows mo
    LEFT JOIN monthly_inflows mi ON mo.month = mi.month
)
SELECT
    month,
    COALESCE(total_inflow, 0) as total_inflow,
    COALESCE(total_outflow, 0) as total_outflow,
    COALESCE(total_inflow, 0) - COALESCE(total_outflow, 0) as net_flow
FROM combined_data
ORDER BY month;
    """

    cursor.execute(monthly_query, (year, year))
    results = cursor.fetchall()
    conn.close()

    if not results:
        return f"No data found for year {year}"

    analysis = f"Monthly Cash Flow Analysis for {year}:\n"
    for row in results:
        analysis += f"\nMonth {row[0]}:"
        analysis += f"\n  Inflow: {row[1]:.2f}"
        analysis += f"\n  Outflow: {row[2]:.2f}"
        analysis += f"\n  Net Flow: {row[3]:.2f}"

    return analysis

def analyze_revenue_streams(inputs: str) -> str:
    """
    Analyze revenue streams and their reliability.
    Accepts input as a JSON-like string containing 'period_start' and 'period_end'.
    Args:
        period_start (str): Start date in YYYY-MM-DD format
        period_end (str): End date in YYYY-MM-DD format
    Returns:
        String containing revenue stream analysis
    """
    import json
    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        period_start = input_dict["period_start"]
        period_end = input_dict["period_end"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    revenue_query = """
        WITH revenue_data AS (
            SELECT
                Vendor_Contact_ID,
                SUM(Payment_Amount) as total_amount,
                COUNT(*) as transaction_count,
                AVG(Payment_Amount) as avg_payment
            FROM Invoice_Payments
            WHERE Invoice_Payment_Date BETWEEN ? AND ?
            GROUP BY Vendor_Contact_ID
        )
        SELECT
            Vendor_Contact_ID,
            total_amount,
            transaction_count,
            avg_payment,
            (total_amount * 1.0 / (SELECT SUM(total_amount) FROM revenue_data) * 100) as percentage
        FROM revenue_data
        ORDER BY total_amount DESC
        LIMIT 10
    """

    cursor.execute(revenue_query, (period_start, period_end))
    results = cursor.fetchall()
    conn.close()

    analysis = "Top Revenue Streams Analysis:\n"
    for row in results:
        analysis += f"\nVendor: {row[0]}"
        analysis += f"\n  Total Amount: {row[1]:.2f}"
        analysis += f"\n  Transaction Count: {row[2]}"
        analysis += f"\n  Average Payment: {row[3]:.2f}"
        analysis += f"\n  Revenue Percentage: {row[4]:.2f}%"

    return analysis

def analyze_expenses(inputs: str) -> str:
    """
    Get comprehensive expense analysis including categories and patterns.
    Accepts input as a JSON-like string containing 'period_start' and 'period_end'.
    Args:
        period_start (str): Start date in YYYY-MM-DD format
        period_end (str): End date in YYYY-MM-DD format
    Returns:
        String containing detailed expense analysis
    """
    import json
    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        period_start = input_dict["period_start"]
        period_end = input_dict["period_end"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    query = """
        WITH expense_analysis AS (
            SELECT
                COALESCE(Expense_Category, 'Uncategorized') as category,
                COUNT(*) as transaction_count,
                SUM(Payment_Amount) as total_amount,
                AVG(Payment_Amount) as avg_amount,
                MIN(Payment_Amount) as min_amount,
                MAX(Payment_Amount) as max_amount
            FROM Bill_Payments
            WHERE Bill_Payment_Date BETWEEN ? AND ?
            GROUP BY Expense_Category
        )
        SELECT
            category,
            transaction_count,
            total_amount,
            avg_amount,
            min_amount,
            max_amount,
            (total_amount / (SELECT SUM(total_amount) FROM expense_analysis) * 100) as percentage
        FROM expense_analysis
        ORDER BY total_amount DESC
    """

    cursor.execute(query, (period_start, period_end))
    results = cursor.fetchall()
    conn.close()

    analysis = "Comprehensive Expense Analysis:\n"
    total_expenses = sum(row[2] for row in results)

    for row in results:
        analysis += f"\nCategory: {row[0]}"
        analysis += f"\n  Transaction Count: {row[1]}"
        analysis += f"\n  Total Amount: {row[2]:.2f}"
        analysis += f"\n  Average Amount: {row[3]:.2f}"
        analysis += f"\n  Range: {row[4]:.2f} to {row[5]:.2f}"
        analysis += f"\n  Percentage of Total: {row[6]:.2f}%"

    return analysis

def analyze_receivables_collection(inputs: str) -> str:
    """
    Analyze receivables collection patterns and their impact on cash flow.
    Args:
        period_start (str): Start date in YYYY-MM-DD format
        period_end (str): End date in YYYY-MM-DD format
    Returns:
        String containing receivables analysis
    """
    import json
    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        period_start = input_dict["period_start"]
        period_end = input_dict["period_end"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    try:
        # Get overall collection statistics
        collection_query = """
        SELECT
            COUNT(DISTINCT Vendor_Contact_ID) as unique_customers,
            COUNT(*) as total_payments,
            SUM(Payment_Amount) as total_collected,
            AVG(Payment_Amount) as avg_payment
        FROM Invoice_Payments
        WHERE Invoice_Payment_Date BETWEEN ? AND ?
        """

        cursor.execute(collection_query, (period_start, period_end))
        overall_stats = cursor.fetchone()

        # Get customer-wise collection patterns
        customer_query = """
        SELECT
            Vendor_Contact_ID,
            COUNT(*) as payment_count,
            SUM(Payment_Amount) as total_amount,
            MIN(Invoice_Payment_Date) as first_payment,
            MAX(Invoice_Payment_Date) as last_payment,
            AVG(Payment_Amount) as avg_amount
        FROM Invoice_Payments
        WHERE Invoice_Payment_Date BETWEEN ? AND ?
        GROUP BY Vendor_Contact_ID
        ORDER BY total_amount DESC
        LIMIT 5
        """

        cursor.execute(customer_query, (period_start, period_end))
        customer_patterns = cursor.fetchall()

        # Format analysis
        analysis = "Receivables Collection Analysis:\n\n"

        if overall_stats[0]:  # if we have data
            analysis += f"Overall Collection Statistics ({period_start} to {period_end}):\n"
            analysis += f"• Number of Unique Customers: {overall_stats[0]}\n"
            analysis += f"• Total Payments Received: {overall_stats[1]}\n"
            analysis += f"• Total Amount Collected: {overall_stats[2]:,.2f}\n"
            analysis += f"• Average Payment Amount: {overall_stats[3]:,.2f}\n"

            # Calculate daily collection rate
            days_query = """
            SELECT COUNT(DISTINCT date(Invoice_Payment_Date))
            FROM Invoice_Payments
            WHERE Invoice_Payment_Date BETWEEN ? AND ?
            """
            cursor.execute(days_query, (period_start, period_end))
            active_days = cursor.fetchone()[0]

            daily_collection = overall_stats[2] / active_days if active_days > 0 else 0
            analysis += f"• Average Daily Collection: {daily_collection:,.2f}\n"

            # Top customer analysis
            analysis += "\nTop 5 Customers by Collection Amount:\n"
            for customer in customer_patterns:
                analysis += f"\nCustomer: {customer[0]}"
                analysis += f"\n• Number of Payments: {customer[1]}"
                analysis += f"\n• Total Amount: {customer[2]:,.2f}"
                analysis += f"\n• Average Payment: {customer[5]:,.2f}"
                analysis += f"\n• First Payment: {customer[3]}"
                analysis += f"\n• Last Payment: {customer[4]}"

            # Cash Flow Impact Analysis
            analysis += "\n\nCash Flow Impact:\n"
            total_amount = overall_stats[2]
            if total_amount > 0:
                analysis += f"• Daily Cash Inflow Rate: {daily_collection:,.2f}\n"

                # Calculate collection concentration
                top_customer_amount = customer_patterns[0][2] if customer_patterns else 0
                concentration = (top_customer_amount / total_amount * 100) if total_amount > 0 else 0
                analysis += f"• Collection Concentration: {concentration:.1f}% from top customer\n"

        else:
            analysis += "No collection data found for the specified period."

    except Exception as e:
        analysis = f"Error analyzing receivables collection: {str(e)}"
    finally:
        conn.close()

    return analysis
#q7
def analyze_vendor_payments(inputs: str) -> str:
    """
    Analyze average time to pay vendors and its impact on cash flow health.
    Accepts input as a JSON-like string containing 'period_start' and 'period_end'.
    Args:
        inputs (str): JSON string containing 'period_start' and 'period_end'
    Returns:
        String containing vendor payment analysis
    """
    import json
    import sqlite3

    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        period_start = input_dict["period_start"]
        period_end = input_dict["period_end"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    # Query to calculate vendor payment patterns
    vendor_query = """
        SELECT
            Vendor_Contact_ID,
            AVG(JULIANDAY(Bill_Payment_Date) - JULIANDAY(?)) AS avg_payment_time,
            SUM(Payment_Amount) AS total_paid,
            COUNT(*) AS payment_count
        FROM Bill_Payments
        WHERE Bill_Payment_Date BETWEEN ? AND ?
        GROUP BY Vendor_Contact_ID
        ORDER BY avg_payment_time DESC
    """

    cursor.execute(vendor_query, (period_start, period_start, period_end))
    vendor_results = cursor.fetchall()

    # Query to calculate cash flow impact
    cash_flow_query = """
        SELECT
            SUM(Payment_Amount) AS total_outflow,
            COUNT(DISTINCT Vendor_Contact_ID) AS vendor_count
        FROM Bill_Payments
        WHERE Bill_Payment_Date BETWEEN ? AND ?
    """

    cursor.execute(cash_flow_query, (period_start, period_end))
    cash_flow_results = cursor.fetchone()

    # Include credit notes impact from Bill_Credit_Notes
    credit_note_query = """
        SELECT
            SUM(Credit_Note_Total_Amount) AS total_credit_notes,
            COUNT(*) AS credit_note_count
        FROM Bill_Credit_Notes
        WHERE Credit_Note_Posted_Date BETWEEN ? AND ?
    """

    cursor.execute(credit_note_query, (period_start, period_end))
    credit_note_results = cursor.fetchone()

    conn.close()

    total_credit_notes = credit_note_results[0] if credit_note_results[0] is not None else 0.0
    credit_note_count = credit_note_results[1] if credit_note_results[1] is not None else 0

    # Prepare the analysis output
    analysis = f"""Vendor Payment Analysis:
    Total Vendors: {cash_flow_results[1]}
    Total Outflow (Payments): {cash_flow_results[0]:.2f}
    Total Credit Notes Applied: {total_credit_notes:.2f} (from {credit_note_count} credit notes)

    Vendor Payment Patterns:"""

    for row in vendor_results[:10]:  # Limit to the top 10 vendors
        analysis += f"""
        Vendor: {row[0]}
        Average Payment Time: {row[1]:.2f} days (from {period_start})
        Total Paid: {row[2]:.2f}
        Payment Count: {row[3]}
        """

    return analysis


#q8 What were our top 10 largest cash outflow transactions last year, and what were they for?
def analyze_large_outflows(year: str) -> str:
    """
    Identify and analyze top 10 largest cash outflow transactions.
    Args:
        year (str): Year to analyze in YYYY format
    Returns:
        String containing large outflow analysis
    """
    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    outflow_query = """
        SELECT
            Bill_Payment_Date,
            Vendor_Contact_ID,
            Payment_Amount,
            Payment_Memo,
            Expense_Category
        FROM Bill_Payments
        WHERE strftime('%Y', Bill_Payment_Date) = ?
        ORDER BY Payment_Amount DESC
        LIMIT 10
    """

    cursor.execute(outflow_query, (year,))
    results = cursor.fetchall()
    conn.close()

    analysis = f"Top 10 Largest Cash Outflows for {year}:\n"
    for row in results:
        analysis += f"""
        Date: {row[0]}
        Vendor: {row[1]}
        Amount: {row[2]:.2f}
        Purpose: {row[3]}
        Category: {row[4] or 'Uncategorized'}
        """

    return analysis

#q9  What is our cash conversion cycle, and are there areas where we can reduce the time it takes to turn cash invested in operations back into cash inflows?

def analyze_cash_conversion_cycle(inputs: str) -> str:
    """
    Analyze cash conversion cycle and identify areas for improvement.
    Args:
        period_start (str): Start date in YYYY-MM-DD format
        period_end (str): End date in YYYY-MM-DD format
    Returns:
        String containing cash conversion cycle analysis
    """
    import json
    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        period_start = input_dict["period_start"]
        period_end = input_dict["period_end"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()
    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    try:
        # Analyze Invoice Payment Patterns (Cash Inflows)
        inflow_query = """
        SELECT
            COUNT(*) as total_transactions,
            SUM(Payment_Amount) as total_amount,
            AVG(Payment_Amount) as avg_amount,
            MIN(Invoice_Payment_Date) as earliest_date,
            MAX(Invoice_Payment_Date) as latest_date
        FROM Invoice_Payments
        WHERE Invoice_Payment_Date BETWEEN ? AND ?
        """

        # Analyze Bill Payment Patterns (Cash Outflows)
        outflow_query = """
        SELECT
            COUNT(*) as total_transactions,
            SUM(Payment_Amount) as total_amount,
            AVG(Payment_Amount) as avg_amount,
            MIN(Bill_Payment_Date) as earliest_date,
            MAX(Bill_Payment_Date) as latest_date
        FROM Bill_Payments
        WHERE Bill_Payment_Date BETWEEN ? AND ?
        """

        cursor.execute(inflow_query, (period_start, period_end))
        inflow_stats = cursor.fetchone()

        cursor.execute(outflow_query, (period_start, period_end))
        outflow_stats = cursor.fetchone()

        analysis = "Cash Conversion Cycle Analysis:\n\n"

        if inflow_stats and outflow_stats:
            # Calculate key metrics
            inflow_days = len(set(row[0] for row in cursor.execute(
                "SELECT Invoice_Payment_Date FROM Invoice_Payments WHERE Invoice_Payment_Date BETWEEN ? AND ?",
                (period_start, period_end)
            ).fetchall()))

            outflow_days = len(set(row[0] for row in cursor.execute(
                "SELECT Bill_Payment_Date FROM Bill_Payments WHERE Bill_Payment_Date BETWEEN ? AND ?",
                (period_start, period_end)
            ).fetchall()))

            # Cash Flow Patterns
            analysis += "Cash Flow Patterns:\n"
            analysis += f"1. Cash Inflows (Collections):\n"
            analysis += f"   • Total Collections: {inflow_stats[1]:,.2f}\n"
            analysis += f"   • Number of Collection Days: {inflow_days}\n"
            analysis += f"   • Average Daily Collection: {(inflow_stats[1]/inflow_days if inflow_days else 0):,.2f}\n\n"

            analysis += f"2. Cash Outflows (Payments):\n"
            analysis += f"   • Total Payments: {outflow_stats[1]:,.2f}\n"
            analysis += f"   • Number of Payment Days: {outflow_days}\n"
            analysis += f"   • Average Daily Payment: {(outflow_stats[1]/outflow_days if outflow_days else 0):,.2f}\n\n"

            # Net Cash Flow
            net_cash_flow = inflow_stats[1] - outflow_stats[1]
            analysis += f"3. Net Cash Position:\n"
            analysis += f"   • Net Cash Flow: {net_cash_flow:,.2f}\n"

            # Optimization Recommendations
            analysis += "\nOptimization Opportunities:\n"

            # Collection Improvements
            if inflow_stats[1] > 0:
                avg_transaction_size = inflow_stats[1] / inflow_stats[0]
                analysis += "1. Collection Process:\n"
                if avg_transaction_size > 100000:
                    analysis += "   • Consider implementing progressive billing for large transactions\n"
                analysis += "   • Implement early payment incentives for faster collections\n"

            # Payment Optimization
            if outflow_stats[1] > 0:
                analysis += "\n2. Payment Process:\n"
                analysis += "   • Review payment scheduling to optimize cash retention\n"
                analysis += "   • Consider vendor payment terms negotiation\n"

            # Cash Flow Timing
            analysis += "\n3. Cash Flow Timing:\n"
            if net_cash_flow < 0:
                analysis += "   • Priority: Accelerate collections to improve cash position\n"
            else:
                analysis += "   • Maintain current collection efficiency\n"

            analysis += "   • Schedule major payments to align with expected collections\n"

        else:
            analysis += "Insufficient data for cash conversion cycle analysis."

    except Exception as e:
        analysis = f"Error analyzing cash conversion cycle: {str(e)}"
    finally:
        conn.close()

    return analysis

#q6
def analyze_customer_delays(inputs: str) -> str:
    """
    Identify customers with longest payment delays and their outstanding amounts.
    Args:
        period_start (str): Start date in YYYY-MM-DD format
        period_end (str): End date in YYYY-MM-DD format
    Returns:
        String containing customer payment delay analysis
    """
    import json
    try:
        # Parse the input string into a dictionary
        input_dict = json.loads(inputs)
        period_start = input_dict["period_start"]
        period_end = input_dict["period_end"]
    except (ValueError, KeyError) as e:
        return f"Invalid input format: {e}"

    conn = sqlite3.connect("Bills.db")
    cursor = conn.cursor()

    # Query to analyze payment patterns by customer
    delay_query = """
    WITH CustomerPayments AS (
        SELECT
            Vendor_Contact_ID,
            Invoice_Payment_Date,
            Payment_Amount,
            STRFTIME('%Y-%m-%d', Invoice_Payment_Date) as payment_date
        FROM Invoice_Payments
        WHERE Invoice_Payment_Date BETWEEN ? AND ?
    ),
    CustomerStats AS (
        SELECT
            Vendor_Contact_ID,
            COUNT(*) as payment_count,
            SUM(Payment_Amount) as total_amount,
            MIN(payment_date) as first_payment,
            MAX(payment_date) as last_payment,
            AVG(Payment_Amount) as avg_payment
        FROM CustomerPayments
        GROUP BY Vendor_Contact_ID
        HAVING payment_count > 1
    )
    SELECT
        cs.Vendor_Contact_ID,
        cs.payment_count,
        cs.total_amount,
        cs.first_payment,
        cs.last_payment,
        cs.avg_payment,
        JULIANDAY(cs.last_payment) - JULIANDAY(cs.first_payment) as date_range
    FROM CustomerStats cs
    ORDER BY total_amount DESC
    LIMIT 10
    """

    try:
        cursor.execute(delay_query, (period_start, period_end))
        results = cursor.fetchall()

        if not results:
            return "No payment data found for the specified period."

        analysis = "Top 10 Customers by Payment Analysis:\n\n"

        for row in results:
            vendor_id, payment_count, total_amount, first_payment, last_payment, avg_payment, date_range = row

            # Calculate average days between payments
            avg_days_between = date_range / payment_count if payment_count > 0 else 0

            analysis += f"""Customer: {vendor_id}
                        - Total Payments: {payment_count}
                        - Total Amount: {total_amount:.2f}
                        - Average Payment: {avg_payment:.2f}
                        - First Payment: {first_payment}
                        - Last Payment: {last_payment}
                        - Average Days Between Payments: {avg_days_between:.1f}

                        """

        # Add summary statistics
        total_volume = sum(row[2] for row in results)
        avg_payment_size = sum(row[5] for row in results) / len(results)

        analysis += f"\nSummary Statistics:"
        analysis += f"\n• Total Payment Volume: {total_volume:.2f}"
        analysis += f"\n• Average Payment Size: {avg_payment_size:.2f}"

    except Exception as e:
        analysis = f"Error analyzing customer delays: {str(e)}"
    finally:
        conn.close()

    return analysis


tools = [
    Tool(
        name="get_average_cash_flows",
        func= get_average_cash_flows,
        description="Calculate average cash inflows and outflows for a selected duration. Requires a dictionary input with keys 'start_date' and 'end_date'. and provide output without any currency symbols",
    ),
    Tool(
        name="analyze_monthly_trends",
        func=analyze_monthly_trends,
        description="Analyze monthly cash flow trends for a specific year. Requires a year as input.",
    ),
    Tool(
        name="analyze_revenue_streams",
        func=analyze_revenue_streams,
        description="Analyze revenue streams and their reliability. Requires a dictionary input with keys 'period_start' and 'period_end'.",
    ),
    Tool(
        name="analyze_expenses",
        func=analyze_expenses,
        description="Get comprehensive expense analysis including categories and patterns.Requires a dictionary input with keys 'period_start' and 'period_end'.",
    ),
    Tool(
        name="Analyze Receivables Collection",  # Tool name
        func=analyze_receivables_collection,  # Function that the tool will execute
        description="Analyze accounts receivable collection time and its impact on cash flow. Requires a dictionary input with keys 'period_start' and 'period_end'.",  # Description of the tool
    ),
    Tool(
        name="Analyze Delayed Payments",  # Tool name
        func=analyze_customer_delays,  # Function that the tool will execute
        description="Identify customers with significant payment delays and outstanding amounts",  # Description of the tool
    ),
    Tool(
        name="Analyze Vendor Payment Time",  # Tool name
        func=analyze_vendor_payments,  # Function that the tool will execute
        description="Analyze vendor payment timing and its impact on cash flow. Requires a dictionary input with keys 'period_start' and 'period_end'.",  # Description of the tool
    ),
    Tool(
        name="Get Largest Outflows",  # Tool name
        func=analyze_large_outflows,  # Function that the tool will execute
        description="Identify the largest cash outflow transactions for a specific year",  # Description of the tool
    ),
    Tool(
        name="Analyze Cash Conversion Cycle",  # Tool name
        func=analyze_cash_conversion_cycle,  # Function that the tool will execute
        description="Calculate and analyze the cash conversion cycle. Requires a dictionary input with keys 'period_start' and 'period_end'.",  # Description of the tool
    )
    # Tool(
    #     name="Analyze Fixed Variable Expenses",  # Tool name
    #     func=analyze_fixed_variable_expenses,  # Function that the tool will execute
    #     description="Analyze the proportion of fixed versus variable expenses. Requires a dictionary input with keys 'period_start' and 'period_end'.",  # Description of the tool
    # ),
]

template = """ Answer the following questions as best you can. You have access to the following tools:

{tools}

You are a **cash flow analyst agent** with expertise in analyzing financial data, trends, and expenses. Your goal is to interpret user queries, select the appropriate tool, and provide accurate insights.

### Tools Available:
1. **`get_average_cash_flows`**: Calculate average inflows and outflows for a date range.
2. **`analyze_monthly_trends`**: Analyze cash flow trends for a specific year.
3. **`analyze_revenue_streams`**: Analyze top revenue streams for a date range.
4. **`analyze_expenses`**: Perform detailed expense analysis for a date range.
5. **`list_tables`**: Retrieve available database tables.
6. **`tables_schema`**: Get schema and sample rows for tables.
7. **`execute_sql`**: Execute a SQL query and return results.
8. **`check_sql`**: Validate a SQL query before execution.

### Guidelines:
- Use the right tool based on the query.
- If inputs (e.g., dates) are missing, ask the user.
- Always validate SQL queries with `check_sql` before execution.
- Respond in a clear and professional manner.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.Convert the action input in suitable format matching with the function's Input
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}  """

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=llm_2,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    )

# inputs = {"input": "What is the average cash inflow and outflow for the selected duration  of year 2023?"}
# #q1
# response = agent_executor.invoke(inputs)

company_id = os.environ["COMPANY_ID"]
@app.route('/<int:company_id>/cashflow_query', methods=['POST'])
def execute_query(company_id):
    try:
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({"error": "No query provided"}), 400

        inputs = {"input": user_input}
        result = agent_executor.invoke(inputs)
        
        # Extract just the final answer 
        final_answer = result.get('output', '')

        # Return the response in a simple format
        return jsonify({
            "status": "success",
            "answer": final_answer
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Render sets the PORT environment variable automatically
    port = int(os.getenv('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
