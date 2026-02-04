"""
Domain knowledge for Text-to-SQL: four types of knowledge injected before LLM generation.
1. Structural (schema)  2. Relational (FKs, joins)  3. Semantic (meaning, rules)  4. Usage constraints
"""

# ---------------------------------------------------------------------------
# 3. SEMANTIC KNOWLEDGE (domain meaning - what tables/columns represent)
# ---------------------------------------------------------------------------

TABLE_SEMANTICS = {
    "Clients": "Business clients/customers. Each row is one client.",
    "Invoices": "Invoices issued to clients. Each row is one invoice. Links to Clients.",
    "InvoiceLineItems": "Line items within an invoice (services, quantities, rates). Each row is one line. Links to Invoices.",
}

# ---------------------------------------------------------------------------
# VALUE-DOMAIN KNOWLEDGE (concept -> actual values in data)
# ---------------------------------------------------------------------------

COUNTRY_REGIONS = {
    "European": [
        "UK", "United Kingdom", "Germany", "France", "Spain", "Italy",
        "Netherlands", "Belgium", "Portugal", "Poland", "Sweden", "Austria",
        "Ireland", "Denmark", "Finland", "Greece", "Romania", "Czech Republic",
        "Czechia", "Hungary", "Switzerland", "Norway", "Ukraine", "Bulgaria",
        "Croatia", "Slovakia", "Slovenia", "Lithuania", "Latvia", "Estonia",
        "Luxembourg", "Malta", "Cyprus", "Iceland", "Serbia",
        "Bosnia and Herzegovina", "Albania", "North Macedonia", "Montenegro",
        "Kosovo", "Moldova", "Belarus", "Russia", "Andorra", "Monaco",
        "San Marino", "Vatican City",
    ],
    "Americas": [
        "US", "USA", "United States", "Canada", "Mexico", "Brazil",
        "Argentina", "Chile", "Colombia", "Peru", "Venezuela", "Ecuador",
    ],
    "Asia": [
        "Japan", "China", "India", "South Korea", "Singapore", "Hong Kong",
        "Thailand", "Vietnam", "Indonesia", "Malaysia", "Philippines",
        "Taiwan", "Israel", "UAE", "United Arab Emirates", "Saudi Arabia",
    ],
    "Oceania": ["Australia", "New Zealand"],
    "Africa": [
        "South Africa", "Egypt", "Nigeria", "Kenya", "Morocco", "Ghana",
    ],
}

COLUMN_SEMANTICS = {
    # Clients
    "client_id": "Client identifier (PK in Clients, FK in Invoices)",
    "id": "Primary key (client_id or id)",
    "name": "Client company or person name",
    "industry": "Industry sector (e.g. Legal, Tech)",
    "country": "Client country. Use COUNTRY_REGIONS for 'European', 'Americas', etc.",
    # Invoices
    "invoice_id": "Invoice identifier (e.g. I1001). FK in InvoiceLineItems",
    "invoice_date": "Date the invoice was issued",
    "due_date": "Payment due date",
    "status": "Invoice status: Paid, Pending, Overdue",
    "amount": "Invoice total amount",
    "currency": "Currency code (e.g. USD, GBP)",
    # InvoiceLineItems
    "line_item_id": "Unique line item identifier",
    "service_name": "Name of the service (e.g. Contract Review, Consulting)",
    "quantity": "Number of units",
    "unit_price": "Price per unit",
    "tax_rate": "Percentage of tax applied to the line item (e.g., 0.20 for 20% VAT"
}

# ---------------------------------------------------------------------------
# 4. USAGE CONSTRAINTS (allowed joins, aggregation, time semantics)
# ---------------------------------------------------------------------------

ALLOWED_JOIN_PATHS = [
    "Invoices.client_id = Clients.client_id (or Clients.id) - to get client info for an invoice",
    "InvoiceLineItems.invoice_id = Invoices.invoice_id - to get line items for an invoice",
    "Full path: Clients <- Invoices <- InvoiceLineItems (client -> invoices -> line items)",
]

TIME_SEMANTICS = [
    "invoice_date: when invoice was issued. Use date(invoice_date) or strftime('%Y-%m-%d', invoice_date)",
    "due_date: payment due. Overdue if due_date < current date and status != 'Paid'",
    "Filter by year: strftime('%Y', invoice_date) = '2024'",
    "Filter by month: strftime('%Y-%m', invoice_date) = '2024-09'",
    "Date range: invoice_date BETWEEN '2024-01-01' AND '2024-12-31'",
]

# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES (question -> SQL) for in-prompt in-context learning
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
  {"question": "List all countries from clients.", "sql": "SELECT DISTINCT country FROM Clients ORDER BY country;"},
  {"question": "Count of invoices per status.", "sql": "SELECT status, COUNT(*) AS cnt FROM Invoices GROUP BY status;"}
]
