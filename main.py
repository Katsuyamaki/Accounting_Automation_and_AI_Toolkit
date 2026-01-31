import os
import io
import re
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from google.cloud import secretmanager
from google import genai

app = FastAPI()

# --- SOC 2 & ENTERPRISE CONTROLS ---

def get_gemini_key():
    """Accesses the 'Vault' to retrieve the API key at runtime."""
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/305499564828/secrets/GEMINI_API_KEY/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def sanitize_description(desc):
    """Clean descriptions to improve matching accuracy."""
    desc = str(desc).upper()
    desc = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', desc) 
    desc = re.sub(r'CHECK\s*#?\d+', '', desc) 
    desc = re.sub(r'\b\d{6,}\b', '', desc)
    desc = desc.replace("PAYMENT ID", "").strip()
    return desc

# --- CORE ACCOUNTING LOGIC (PORTED FROM CLASSIFY.PY) ---

def generate_je_smart(row, category):
    """Generates a standard or intercompany journal entry based on [Target] tags."""
    amt = row['Amount']
    abs_amt = abs(amt)
    payer_entity = "ENTITY_A"  # Default for this engine
    
    # Check for Intercompany Tags: e.g., "Repairs [ENTITY_B]"
    match = re.search(r'\[(.*?)\]', str(category))
    if match:
        target_entity = match.group(1).upper()
        clean_cat = str(category).replace(f'[{target_entity}]', '').strip()
        
        # Two-way Entry (SOC 2 Processing Integrity)
        return [
            {"Entity": payer_entity, "Amount": abs_amt, "Debit": f"Notes - {target_entity}", "Credit": "Cash", "Memo": f"Paid for {target_entity}"},
            {"Entity": target_entity, "Amount": abs_amt, "Debit": clean_cat, "Credit": f"Notes - {payer_entity}", "Memo": f"Paid by {payer_entity}"}
        ]
    
    # Standard Entry
    return [{"Entity": payer_entity, "Amount": abs_amt, "Debit": category, "Credit": "Cash" if amt < 0 else "Revenue"}]

# --- THE API ENDPOINT ---

@app.post("/classify")
async def cloud_waterfall(file: UploadFile = File(...)):
    # 1. ZERO-PERSISTENCE: Read into RAM
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    final_output = []
    
    # 2. THE WATERFALL (Simplified for Cloud Deployment)
    for _, row in df.iterrows():
        # A. Deterministic Check (Example Rule)
        desc = sanitize_description(row['Description'])
        category = "Uncategorized"
        
        if "PEPCO" in desc: category = "Utility Expenses"
        elif "WAWA" in desc: category = "Travel/Gas"
        
        # B. Generate Journal Entries
        jes = generate_je_smart(row, category)
        final_output.extend(jes)
        
    return {
        "status": "success",
        "security": "Zero-Persistence Active",
        "engine": "Hybrid-Waterfall-v1",
        "journal_entries": final_output
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
