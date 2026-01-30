
import pandas as pd
import os
import time
import json
import re
import random
from google import genai

# --- CONFIGURATION ---
INPUT_FILE = '../1_import_raw_transactions/Data_in.csv'
RULES_FILE = '../1_import_raw_transactions/transaction_rules.csv'
LOGIC_FILE = '../1_import_raw_transactions/logic_rules.csv'
HISTORY_FILE = '../1_import_raw_transactions/correction_history.csv' # <--- NEW
OUTPUT_FILE = '../3_script_outputs/categorized_journal_entries.csv'
NEW_MAPPINGS_FILE = '../3_script_outputs/new_mappings_to_review.csv'
API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_ROSTER = ["models/gemini-2.0-flash", "models/gemini-2.5-flash"]

# --- 1. SETUP ---


def load_files():
    try:
        rules_df = pd.read_csv(RULES_FILE)
        if not rules_df.empty:
            rules_df = rules_df[~rules_df['Keyword'].astype(str).str.strip().str.startswith('#')]
        rules_dict = dict(zip(rules_df['Keyword'].str.upper(), rules_df['Category']))
    except: rules_dict = {}
    
    try:
        logic_df = pd.read_csv(LOGIC_FILE)
        if not logic_df.empty:
            logic_df = logic_df[~logic_df['Rule_Name'].astype(str).str.strip().str.startswith('#')]
    except: logic_df = pd.DataFrame()

    # NEW: Load Correction History
    history_dict = {}
    try:
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            # Create a unique key for every corrected transaction
            for _, row in hist_df.iterrows():
                # Signature: Date + Signed Amount + Description
                key = f"{row['Date']}|{float(row['Amount'])}|{str(row['Original_Description']).strip()}"
                # Save both Category and Memo
                history_dict[key] = {'Category': row['Category'], 'Memo': row.get('Memo')}
            print(f"🧠 Loaded {len(history_dict)} past corrections from memory.")
    except Exception as e:
        print(f"⚠️ Could not load history: {e}")
        
    explode_keywords = set()
    try:
        if os.path.exists(NEW_MAPPINGS_FILE):
            prev_review = pd.read_csv(NEW_MAPPINGS_FILE)
            if 'Explode? (Y/N)' in prev_review.columns:
                to_explode = prev_review[prev_review['Explode? (Y/N)'].str.upper() == 'Y']
                explode_keywords = set(to_explode['Keyword'].str.upper().tolist())
                if explode_keywords: print(f"💥 Exploding {len(explode_keywords)} keywords.")
    except: pass

    system_toggles = {'INTERCOMPANY_AUTO': False, 'SPLIT_DISTRIBUTION': False}
    if not logic_df.empty:
        ic = logic_df[(logic_df['Action'] == 'INTERCOMPANY_AUTO') & (logic_df['Trigger_Value'] == 'ON')]
        if not ic.empty: system_toggles['INTERCOMPANY_AUTO'] = True
        dist = logic_df[(logic_df['Action'] == 'SPLIT_DISTRIBUTION') & (logic_df['Trigger_Value'] == 'ON')]
        if not dist.empty: system_toggles['SPLIT_DISTRIBUTION'] = True
        
    return rules_dict, logic_df, explode_keywords, system_toggles, history_dict


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

# --- MAIN ---

def main():
    print("--- ENTITY_A: Persistent Memory ---")
    if not API_KEY: print("❌ No API Key"); return
    try: df = pd.read_csv(INPUT_FILE)
    except: print("❌ No Input File"); return
    
    df['Category'] = None
    rules_dict, logic_df, explode_keywords, system_toggles, history_dict = load_files()
    
    def get_entity(acc):
        if "BB" in str(acc).upper(): return "ENTITY_B"
        if "CC" in str(acc).upper(): return "ENTITY_C"
        return "AA"
    df['Entity'] = df['Account'].apply(get_entity)

    final_entries = []
    ai_batch = []
    
    # 1. Processing
    for idx, row in df.iterrows():
        # A. CHECK HISTORY FIRST (The Golden Record)
        hist_key = f"{row['Date']}|{float(row['Amount'])}|{str(row['Description']).strip()}"

        if hist_key in history_dict:
            # Handle Dictionary format (Cat + Memo)
            data = history_dict[hist_key]
            if isinstance(data, str): # Backward compatibility
                cat = data
                memo = None
            else:
                cat = data['Category']
                memo = data.get('Memo')

            df.at[idx, 'Category'] = cat
            # Pass the saved memo to the generator
            entries = generate_je_smart(row, cat, custom_memo=memo)
            final_entries.extend(entries)
            continue 


        # B. LOGIC & RULES
        special_cat, special_rows = apply_special_logic(row, logic_df, system_toggles)
        if special_rows: final_entries.extend(special_rows); continue
        if special_cat: df.at[idx, 'Category'] = special_cat; continue

        sanitized = sanitize_description(row['Description'])
        if sanitized and sanitized in rules_dict: 
            df.at[idx, 'Category'] = rules_dict[sanitized]
        else:
            found_partial = False
            for key in rules_dict:
                if key in sanitized: df.at[idx, 'Category'] = rules_dict[key]; found_partial = True; break
            if not found_partial: ai_batch.append({"id": idx, "desc": row['Description']})


    # 2. AI
    new_mappings_details = [] 
    successful_ai_ids = set() # Track what the AI actually fixes

    if ai_batch:
        print(f"🤖 AI Processing {len(ai_batch)} items...")
        client = genai.Client(api_key=API_KEY)
        chunk_size = 20
        for i in range(0, len(ai_batch), chunk_size):
            chunk = ai_batch[i:i+chunk_size]
            results = get_ai_category(client, chunk, rules_dict)

            if results:
                seen_ids_in_batch = set() # NEW: Prevent double-adding the same ID
                
                for key_id, cat in results.items():
                    try:
                        idx = int(re.sub(r'\D', '', str(key_id)))
                        
                        # FAILSAFE: If AI returns same ID twice, ignore the second one
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


        # --- NEW: VISUAL FEEDBACK FOR FAILURES ---
        failed_count = len(ai_batch) - len(successful_ai_ids)
        if failed_count > 0:
            print(f"\n⚠️  AI failed to classify {failed_count} items (Remaining 'Uncategorized'):")
            print(f"{'Date':<12} | {'Amount':>10} | {'Description'}")
            print("-" * 80)
            for item in ai_batch:
                if item['id'] not in successful_ai_ids:
                    row = df.loc[item['id']]
                    print(f"{str(row['Date']):<12} | {row['Amount']:>10.2f} | {str(row['Description'])[:50]}")
            print("-" * 80)
            print("   (Tip: Copy these keywords into 'transaction_rules.csv' manually to fix)\n")


    # 3. Generate JEs (Using Smart Generator)
    print("🐍 Generating Journal Entries...")
    for idx, row in df.iterrows():
        if pd.isna(row.get('Category')) and idx not in [item['id'] for item in ai_batch]: continue
        cat = df.at[idx, 'Category']
        entries = generate_je_smart(row, cat)
        final_entries.extend(entries)

    out_df = pd.DataFrame(final_entries)
    cols = ['Date', 'Entity', 'JE Amount', 'Debit', 'Credit', 'Memo', 'Original_Account', 'Original_Amount', 'Original_Description']
    out_df = out_df[cols]
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    # 5. Create Enhanced Review File (RESTORED)
    if new_mappings_details:
        review_df = pd.DataFrame(new_mappings_details)
        review_df['Sign'] = review_df['Amount'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
        review_df['Count'] = review_df.groupby(['Keyword', 'Sign'])['Keyword'].transform('count')
        
        exploded_rows = review_df[review_df['Keyword'].str.upper().isin(explode_keywords)]
        grouped_rows = review_df[~review_df['Keyword'].str.upper().isin(explode_keywords)].drop_duplicates(subset=['Keyword', 'Sign'])
        final_review_df = pd.concat([exploded_rows, grouped_rows]).sort_values(by=['Keyword'])
        
        final_review_rows = []

        for _, item in final_review_df.iterrows():
            source_row = df.loc[item['Source_Idx']]
            
            # OLD BUGGY LINE:
            # matches = out_df[out_df['Original_Description'] == source_row['Description']]
            
            # NEW FIXED LINE: Match Description AND Amount to prevent mix-ups
            matches = out_df[
                (out_df['Original_Description'] == source_row['Description']) & 
                (abs(out_df['Original_Amount'] - source_row['Amount']) < 0.01) # Use tolerance for float
            ]
            
            if not matches.empty:
                je_row = matches.iloc[0]
                # ... rest of the code ...
                final_review_rows.append({
                    'Approve/Correct (A/C)': '',
                    'Correction': '',
                    'Add Rule? (Y/N)': '', 
                    'Explode? (Y/N)': 'Y' if item['Keyword'].upper() in explode_keywords else '', 
                    'Count': item['Count'],
                    'Keyword': item['Keyword'],
                    'Category': item['Category'],
                    'Date': je_row['Date'],
                    'Entity': je_row['Entity'],
                    'JE Amount': je_row['JE Amount'],
                    'Debit': je_row['Debit'],
                    'Credit': je_row['Credit'],
                    'Memo': je_row['Memo'],
                    'Original_Description': je_row['Original_Description'],
                    'Original_Amount': source_row['Amount']
                })
        
        final_review_df = pd.DataFrame(final_review_rows)
        review_cols = ['Approve/Correct (A/C)', 'Correction', 'Add Rule? (Y/N)', 'Explode? (Y/N)', 'Count', 'Keyword', 'Category', 
                       'Date', 'Entity', 'JE Amount', 'Debit', 'Credit', 'Memo', 'Original_Description', 'Original_Amount']
        final_review_df = final_review_df[review_cols]
        final_review_df.to_csv(NEW_MAPPINGS_FILE, index=False)
        print(f"⚠️ Created {len(final_review_df)} review items.")
    else:
        print("ℹ️ No new rules created.")

    print(f"🚀 DONE! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
