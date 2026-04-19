import os
from fpdf import FPDF

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Global Finance Bank - Account Statement', border=0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)

def generate_pdf(filename, client_name, limit_bal, bills, pays, delays):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # Client Info
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Client Demographics", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    
    demographics = [
        f"Name: {client_name}",
        "Age: 32",
        "Gender: Female (2)",
        "Education: University (2)",
        "Marital Status: Single (2)"
    ]
    for demo in demographics:
        pdf.cell(0, 8, demo, border=0, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(5)
    
    # Account Info
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Account Information", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Credit Limit (LIMIT_BAL): ${limit_bal}", border=0, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    
    # Billing Info
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Billing History (Last 6 Months)", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    
    for i, b in enumerate(bills):
        pdf.cell(0, 6, f"Month {i+1} Bill Amount: ${b}", border=0, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(5)
    
    # Payment Info
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Payment History & Delays (Last 6 Months)", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    
    for i, p in enumerate(pays):
        pdf.cell(0, 6, f"Month {i+1} Payment Made: ${p} (Delay: {delays[i]} months)", border=0, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, "This is a machine-generated statement for AI Underwriting Copilot demonstration.", border=0, new_x="LMARGIN", new_y="NEXT", align='C')
    
    # Save the pdf
    out_path = os.path.join(SAMPLES_DIR, filename)
    pdf.output(out_path)
    print(f"Generated sample statement at: {out_path}")

def generate_all_samples():
    # 1. Low Risk: High limit, low bills, full payments on time
    generate_pdf(
        filename="sample_low_risk.pdf",
        client_name="Sarah Johnson (Low Risk)",
        limit_bal=200000,
        bills=[500, 400, 600, 300, 200, 100],
        pays=[500, 400, 600, 300, 200, 100],
        delays=[-1, -1, -1, -1, -1, -1]
    )
    
    # 2. Grey Zone: Moderate limit, moderate usage, some late payments
    generate_pdf(
        filename="sample_grey_zone.pdf",
        client_name="Michael Smith (Medium Risk)",
        limit_bal=50000,
        bills=[45000, 44000, 42000, 40000, 39000, 38000],
        pays=[2000, 2000, 2000, 2000, 2000, 2000],
        delays=[1, 2, 0, 1, 0, 0]
    )
    
    # 3. High Risk: Low limit, maxed out, severe late payments
    generate_pdf(
        filename="sample_high_risk.pdf",
        client_name="Dave Miller (High Risk)",
        limit_bal=10000,
        bills=[10500, 10200, 9800, 9500, 9000, 8500],
        pays=[0, 0, 500, 0, 200, 0],
        delays=[4, 3, 2, 2, 1, 1]
    )

if __name__ == "__main__":
    try:
        generate_all_samples()
    except Exception as e:
        print("Failed to generate PDF:", e)
