
import os
import io
import json
import re
import random
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from google.cloud import secretmanager, storage
from google import genai

# --- CONFIGURATION ---
RULES_BUCKET = os.environ.get("RULES_BUCKET", "accounting-rules-bucket")
MODEL_ROSTER = ["models/gemini-2.0-flash", "models/gemini-2.5-flash"]
GCP_PROJECT = "305499564828"

app = FastAPI(title="Accounting Classification API", version="2.0")


# --- SECURITY LAYER ---

def _get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{GCP_PROJECT}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


async def verify_client(
    x_api_key: str = Header(...),
    x_client_id: str = Header(...),
) -> str:
    expected_key = _get_secret(f"CLIENT_KEY_{x_client_id}")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_client_id


# --- GCS HELPERS ---

def _stream_csv_from_gcs(bucket_name: str, blob_path: str) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))

# --- 1. SETUP ---


def load_rules_from_gcs(client_id: str):
    prefix = f"{client_id}/"

    rules_dict = {}
    try:
        rules_df = _stream_csv_from_gcs(RULES_BUCKET, f"{prefix}transaction_rules.csv")
        if not rules_df.empty:
            rules_df = rules_df[~rules_df['Keyword'].astype(str).str.strip().str.startswith('#')]
        rules_dict = dict(zip(rules_df['Keyword'].str.upper(), rules_df['Category']))
    except: rules_dict = {}

    logic_df = pd.DataFrame()
    try:
        logic_df = _stream_csv_from_gcs(RULES_BUCKET, f"{prefix}logic_rules.csv")
        if not logic_df.empty:
            logic_df = logic_df[~logic_df['Rule_Name'].astype(str).str.strip().str.startswith('#')]
    except: logic_df = pd.DataFrame()

    history_dict = {}
    try:
        hist_df = _stream_csv_from_gcs(RULES_BUCKET, f"{prefix}correction_history.csv")
        for _, row in hist_df.iterrows():
            key = f"{row['Date']}|{float(row['Amount'])}|{str(row['Original_Description']).strip()}"
            history_dict[key] = {'Category': row['Category'], 'Memo': row.get('Memo')}
    except: pass

    system_toggles = {'INTERCOMPANY_AUTO': False, 'SPLIT_DISTRIBUTION': False}
    if not logic_df.empty:
        ic = logic_df[(logic_df['Action'] == 'INTERCOMPANY_AUTO') & (logic_df['Trigger_Value'] == 'ON')]
        if not ic.empty: system_toggles['INTERCOMPANY_AUTO'] = True
        dist = logic_df[(logic_df['Action'] == 'SPLIT_DISTRIBUTION') & (logic_df['Trigger_Value'] == 'ON')]
        if not dist.empty: system_toggles['SPLIT_DISTRIBUTION'] = True

    return rules_dict, logic_df, system_toggles, history_dict


def sanitize_description(desc):
    desc = str(desc).upper()
    desc = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', desc) 
    desc = re.sub(r'CHECK\s*#?\d+', '', desc) 
    desc = re.sub(r'\b\d{6,}\b', '', desc)
    desc = re.sub(r'BBT\d+', '', desc)
    desc = desc.replace("PAYMENT ID", "").replace("BUS ONLINE BILL PAYMENT", "").replace("ZELLE BUSINESS PAYMENT TO", "")
    desc = re.sub(r'\s+', ' ', desc).strip()
    return desc

# --- SMART JE GENERATOR (Handles [Target] Tags) ---

# Updated to accept custom_memo argument
def generate_je_smart(row, category, custom_memo=None):
    amt = row['Amount']
    abs_amt = abs(amt)
    payer_entity = row['Entity']
    account = str(row['Account'])
    desc = row['Description']
    
    # Use Custom Memo if provided (from History)
    final_memo = desc
    if custom_memo and str(custom_memo).strip() != '' and str(custom_memo).lower() != 'nan':
        final_memo = str(custom_memo).strip()
    
    if pd.isna(category) or category == "None" or category == "": category = "Uncategorized"

    target_entity = None
    clean_category = category
    match = re.search(r'\[(.*?)\]', category)
    if match:
        tag = match.group(1).upper()
        clean_category = category.replace(f'[{match.group(1)}]', '').strip()
        if "BB" in tag: target_entity = "ENTITY_B"
        elif "CC" in tag: target_entity = "ENTITY_C"
        elif "CC" in tag: target_entity = "ENTITY_B"
    
    if not target_entity or target_entity == payer_entity:
        debit, credit = "Uncategorized", "Uncategorized"
        if amt < 0: 
            credit = "Cash - BB&T"
            if "Visa" in account: credit = "Credit - BB&T"; debit = clean_category
            else:
                 if clean_category == "Internal Transfer": debit = "Cash - BB&T"
                 else: debit = clean_category 
        else: 
            debit = "Cash - BB&T"
            if "Visa" in account: debit = "Credit - BB&T"; credit = clean_category
            else:
                 if clean_category == "Rental Revenue": credit = "Rental Revenue"
                 else: credit = clean_category
        # Use final_memo here
        return [{"Date": row['Date'], "Entity": payer_entity, "JE Amount": abs_amt, "Debit": debit, "Credit": credit, "Memo": final_memo, "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}]

    if amt < 0:
        # Use final_memo here if desired, or keep automated intercompany memos
        entry1 = {"Date": row['Date'], "Entity": payer_entity, "JE Amount": abs_amt, "Debit": f"Short Term Notes - {target_entity}", "Credit": "Cash - BB&T", "Memo": f"Paid on behalf of {target_entity}", "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}
        entry2 = {"Date": row['Date'], "Entity": target_entity, "JE Amount": abs_amt, "Debit": clean_category, "Credit": f"Short Term Notes - {payer_entity}", "Memo": f"Paid by {payer_entity}", "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}
        return [entry1, entry2]
    
    return generate_je_smart(row, clean_category, custom_memo)

    
    # Fallback for deposits
    return generate_je_smart(row, clean_category)



# --- 2. LOGIC ENGINE ---
def apply_special_logic(row, logic_df, system_toggles):
    desc = str(row['Description']).upper()
    amt = row['Amount']
    abs_amt = abs(amt)
    entity = row['Entity']
    account = str(row['Account'])
    
    # A. TOGGLES
    if system_toggles['INTERCOMPANY_AUTO']:
        target_map = {"6666": "ENTITY_A", "77777": "ENTITY_C", "8888": "ENTITY_B", "9999": "VISA_CARD"}
        found_target = None
        for code, target in target_map.items():
            if code in desc: found_target = target; break
        
        if found_target:
            # 1. RESTORED: Visa Payment Logic
            if found_target == "VISA_CARD" and amt < 0:
                 return "CC_PMT_AUTO", [{"Date": row['Date'], "Entity": entity, "JE Amount": abs_amt, "Debit": "Credit - BB&T", "Credit": "Cash - BB&T", "Memo": "CC Payment (Detected 6660)", "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}]
            
            # 2. Intercompany Transfers
            if found_target != "VISA_CARD":
                if amt < 0:
                    return "INTERCOMPANY_AUTO", [{"Date": row['Date'], "Entity": entity, "JE Amount": abs_amt, "Debit": f"Short Term Notes - {found_target}", "Credit": "Cash - BB&T", "Memo": f"Intercompany Out", "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}]
                else:
                    return "INTERCOMPANY_AUTO", [{"Date": row['Date'], "Entity": entity, "JE Amount": abs_amt, "Debit": "Cash - BB&T", "Credit": f"Short Term Notes - {found_target}", "Memo": f"Intercompany In", "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}]

    if system_toggles['SPLIT_DISTRIBUTION']:
        if entity == "ENTITY_B" and amt < 0 and ("CHRISTOPHER" in desc or "CHUCK" in desc):
            partner_name = "Christopher" if "CHRISTOPHER" in desc else "Chuck"
            capital_account = f"Partner's Capital - {partner_name}"
            split_rows = [
                    {"Entity": entity, "JE Amount": abs_amt/2, "Debit": "Short Term Notes - ENTITY_B", "Credit": "Cash - BB&T", "Memo": f"Distro Split (ENTITY_B) - {partner_name}", "Date": row['Date'], "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']},
                {"Entity": entity, "JE Amount": abs_amt/2, "Debit": "Short Term Notes - ENTITY_C", "Credit": "Cash - BB&T", "Memo": f"Distro Split (ENTITY_C) - {partner_name}", "Date": row['Date'], "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']},
                {"Entity": "ENTITY_B", "JE Amount": abs_amt/2, "Debit": capital_account, "Credit": "Short Term Notes - ENTITY_B", "Memo": f"Mirror Distro - {partner_name}", "Date": row['Date'], "Original_Account": "Non-Cash", "Original_Amount": 0, "Original_Description": "Mirror Entry"},
                {"Entity": "ENTITY_C", "JE Amount": abs_amt/2, "Debit": capital_account, "Credit": "Short Term Notes - ENTITY_B", "Memo": f"Mirror Distro - {partner_name}", "Date": row['Date'], "Original_Account": "Non-Cash", "Original_Amount": 0, "Original_Description": "Mirror Entry"}
            ]
            return "SPLIT_TRANSACTION", split_rows

    # B. STANDARD CSV LOGIC
    if not logic_df.empty:
        for _, rule in logic_df.iterrows():
            if rule['Trigger_Type'] == 'SYSTEM_TOGGLE': continue
            if rule['Entity_Filter'] != "ALL" and rule['Entity_Filter'] != "VISA_ONLY" and rule['Entity_Filter'] != entity: continue
            if rule['Entity_Filter'] == "VISA_ONLY" and "Visa" not in account: continue
            
            match = False
            if rule['Trigger_Type'] == "AMOUNT_EQUALS":
                if float(rule['Trigger_Value']) == amt: match = True
            elif rule['Trigger_Type'] == "KEYWORD":
                if str(rule['Trigger_Value']).upper() in desc: match = True
            
            if match:
                if rule['Action'] == "STANDARD":
                    jes = generate_je_smart(row, rule['Debit_Account'])
                    if len(jes) == 1: jes[0]['Credit'] = rule['Credit_Account']
                    return "CSV_RULE", jes
                elif rule['Action'] == "ZERO_OUT":
                     return "CSV_RULE", [{"Entity": entity, "JE Amount": abs_amt, "Debit": "Cash - BB&T", "Credit": "Cash - BB&T", "Memo": rule['Memo_Tag'], "Date": row['Date'], "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}]
                elif rule['Action'] == "FORCE_CREDIT": 
                    if amt > 0: return rule['Credit_Account'], None
                elif rule['Action'] == "CC_PAYMENT":
                    return "CC_PMT_AUTO", [{"Entity": entity, "JE Amount": abs_amt, "Debit": rule['Debit_Account'], "Credit": rule['Credit_Account'], "Memo": rule['Memo_Tag'], "Date": row['Date'], "Original_Account": row['Account'], "Original_Amount": row['Amount'], "Original_Description": row['Description']}]

    return None, None

# --- 3. AI ENGINE (RESTORED) ---
def clean_and_parse_json(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1: text = text[start:end+1]
    text = re.sub(r'\}\s*\{', ', ', text)
    try: return json.loads(text)
    except json.JSONDecodeError: return {}

def get_ai_category(client, batch_data, rules_dict):
    if not batch_data: return {}
    history_examples = ""
    if rules_dict:
        sample_keys = random.sample(list(rules_dict.keys()), min(10, len(rules_dict)))
        history_examples = "\n".join([f"- {k}: {rules_dict[k]}" for k in sample_keys])

    system_instruction = f"""
    You are an Accounting Classifier for ENTITY_A. Map transactions to ONE Category.
    
    IMPORTANT: Return a SINGLE JSON object where keys are IDs and values are Categories.
    
    ### APPROVED ACCOUNTS:
    - Utility Expenses, Repairs and Maintenance, General and Administrative
    - Tax Expenses, Rental Revenue, Interest Expense, Bank Fees, Legal Expenses
    - Intercompany Transfer Out, Intercompany Transfer In, Internal Transfer
    - Loan to Partner - Christopher

    ### CONTEXT:
    {history_examples}
    
    EXAMPLE OUTPUT:
    {{"10": "General and Administrative", "11": "Utility Expenses"}}
    """
    prompt = "Classify:\n" + "\n".join([f"ID {item['id']}: {item['desc']}" for item in batch_data])
    
    for model in MODEL_ROSTER:
        try:
            resp = client.models.generate_content(
                model=model,
                config={'temperature': 0.0, 'response_mime_type': 'application/json', 'system_instruction': system_instruction},
                contents=prompt
            )
            return clean_and_parse_json(resp.text)
        except Exception as e: 
            print(f"   ⚠️ AI Error: {str(e)[:50]}")
            continue
    return {}

# --- REVIEW JE GENERATOR (merged from confirm_review_and_update.py) ---

def generate_corrected_je_smart(row_data, category, custom_memo=None):
    amt = row_data.get('Original_Amount', 0)
    if amt == 0: amt = row_data.get('JE Amount', 0)
    abs_amt = abs(amt)
    payer_entity = row_data['Entity']
    account = str(row_data.get('Original_Account', ''))
    desc = row_data.get('Original_Description', '')

    final_memo = desc
    if custom_memo and str(custom_memo).strip() != '' and str(custom_memo).lower() != 'nan':
        final_memo = str(custom_memo).strip()

    if pd.isna(category) or category in ("None", ""): category = "Uncategorized"

    target_entity = None
    clean_category = category
    match = re.search(r'\[(.*?)\]', category)
    if match:
        tag = match.group(1).upper()
        clean_category = category.replace(f'[{match.group(1)}]', '').strip()
        if "BB" in tag: target_entity = "ENTITY_B"
        elif "CC" in tag: target_entity = "ENTITY_C"
        elif "AA" in tag: target_entity = "ENTITY_A"

    if not target_entity or target_entity == payer_entity:
        debit, credit = "Uncategorized", "Uncategorized"
        if amt < 0:
            credit = "Cash - BB&T"
            if "Visa" in account: credit = "Credit - BB&T"; debit = clean_category
            else:
                if clean_category == "Internal Transfer": debit = "Cash - BB&T"
                else: debit = clean_category
        else:
            debit = "Cash - BB&T"
            if "Visa" in account: debit = "Credit - BB&T"; credit = clean_category
            else:
                if clean_category == "Rental Revenue": credit = "Rental Revenue"
                else: credit = clean_category
        return [{"Date": row_data['Date'], "Entity": payer_entity, "JE Amount": abs_amt, "Debit": debit, "Credit": credit, "Memo": final_memo, "Original_Account": row_data.get('Original_Account', ''), "Original_Amount": amt, "Original_Description": desc}]

    if amt < 0:
        entry1 = {"Date": row_data['Date'], "Entity": payer_entity, "JE Amount": abs_amt, "Debit": f"Short Term Notes - {target_entity}", "Credit": "Cash - BB&T", "Memo": f"Paid on behalf of {target_entity}", "Original_Account": row_data.get('Original_Account', ''), "Original_Amount": amt, "Original_Description": desc}
        entry2 = {"Date": row_data['Date'], "Entity": target_entity, "JE Amount": abs_amt, "Debit": clean_category, "Credit": f"Short Term Notes - {payer_entity}", "Memo": f"Paid by {payer_entity}", "Original_Account": row_data.get('Original_Account', ''), "Original_Amount": amt, "Original_Description": desc}
        return [entry1, entry2]

    return generate_corrected_je_smart(row_data, clean_category, custom_memo)


# --- MAIN ---

@app.post("/classify")
async def classify_transactions(
    file: UploadFile = File(...),
    client_id: str = Depends(verify_client),
):
    # 1. ZERO-PERSISTENCE: Read uploaded CSV into RAM only (SOC 2 Confidentiality)
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # 2. Load rules from GCS
    rules_dict, logic_df, system_toggles, history_dict = load_rules_from_gcs(client_id)

    # 3. Derive entity from account
    def get_entity(acc):
        if "BB" in str(acc).upper(): return "ENTITY_B"
        if "CC" in str(acc).upper(): return "ENTITY_C"
        return "AA"
    df['Entity'] = df['Account'].apply(get_entity)
    df['Category'] = None

    final_entries = []
    ai_batch = []
    review_items = []

    # --- WATERFALL: History -> Special Logic -> Keyword Rules -> AI Fallback ---
    for idx, row in df.iterrows():
        # A. CHECK HISTORY FIRST (The Golden Record)
        hist_key = f"{row['Date']}|{float(row['Amount'])}|{str(row['Description']).strip()}"
        if hist_key in history_dict:
            data = history_dict[hist_key]
            if isinstance(data, str):
                cat, memo = data, None
            else:
                cat, memo = data['Category'], data.get('Memo')
            df.at[idx, 'Category'] = cat
            final_entries.extend(generate_je_smart(row, cat, custom_memo=memo))
            continue

        # B. LOGIC & RULES
        special_cat, special_rows = apply_special_logic(row, logic_df, system_toggles)
        if special_rows: final_entries.extend(special_rows); continue
        if special_cat: df.at[idx, 'Category'] = special_cat; continue

        # C. KEYWORD RULES
        sanitized = sanitize_description(row['Description'])
        if sanitized and sanitized in rules_dict:
            df.at[idx, 'Category'] = rules_dict[sanitized]
        else:
            found_partial = False
            for key in rules_dict:
                if key in sanitized: df.at[idx, 'Category'] = rules_dict[key]; found_partial = True; break
            if not found_partial: ai_batch.append({"id": idx, "desc": row['Description']})

    # --- AI FALLBACK ---
    new_mappings_details = []
    successful_ai_ids = set()

    if ai_batch:
        api_key = _get_secret("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        chunk_size = 20
        for i in range(0, len(ai_batch), chunk_size):
            chunk = ai_batch[i:i+chunk_size]
            results = get_ai_category(client, chunk, rules_dict)

            if results:
                seen_ids_in_batch = set()
                for key_id, cat in results.items():
                    try:
                        idx = int(re.sub(r'\D', '', str(key_id)))
                        if idx in seen_ids_in_batch: continue
                        seen_ids_in_batch.add(idx)

                        if idx in df.index:
                            df.at[idx, 'Category'] = cat
                            successful_ai_ids.add(idx)

                            clean = sanitize_description(df.at[idx, 'Description'])
                            if not clean: clean = str(df.at[idx, 'Description']).strip()

                            if clean and clean not in rules_dict:
                                new_mappings_details.append({
                                    'Keyword': clean,
                                    'Category': cat,
                                    'Source_Idx': idx,
                                    'Amount': df.at[idx, 'Amount']
                                })
                    except: continue

    # --- Generate remaining JEs ---
    ai_ids = {item['id'] for item in ai_batch}
    for idx, row in df.iterrows():
        if pd.isna(row.get('Category')) and idx not in ai_ids: continue
        cat = df.at[idx, 'Category']
        entries = generate_je_smart(row, cat)
        final_entries.extend(entries)

    # --- Build review_items for front-end feedback loop (merged from confirm_review_and_update.py) ---
    if new_mappings_details:
        out_df = pd.DataFrame(final_entries)
        for mapping in new_mappings_details:
            source_row = df.loc[mapping['Source_Idx']]
            matches = out_df[
                (out_df['Original_Description'] == source_row['Description']) &
                (abs(out_df['Original_Amount'] - source_row['Amount']) < 0.01)
            ]
            if not matches.empty:
                je_row = matches.iloc[0]
                review_items.append({
                    "needs_review": True,
                    "keyword": mapping['Keyword'],
                    "ai_category": mapping['Category'],
                    "date": je_row['Date'],
                    "entity": je_row['Entity'],
                    "je_amount": je_row['JE Amount'],
                    "debit": je_row['Debit'],
                    "credit": je_row['Credit'],
                    "memo": je_row['Memo'],
                    "original_description": je_row['Original_Description'],
                    "original_amount": float(source_row['Amount']),
                })

    return {
        "status": "success",
        "security": "Zero-Persistence Active",
        "engine": "Hybrid-Waterfall-v2",
        "journal_entries": final_entries,
        "review_items": review_items,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
