import pandas as pd
import os
import re

# --- CONFIGURATION ---
RULES_FILE = '../1_import_raw_transactions/transaction_rules.csv'
REVIEW_FILE = '../3_script_outputs/new_mappings_to_review.csv'
JE_FILE = '../3_script_outputs/categorized_journal_entries.csv'
HISTORY_FILE = '../1_import_raw_transactions/correction_history.csv'


# --- SMART JE GENERATOR ---
def generate_corrected_je_smart(row_data, category, custom_memo=None):
    amt = row_data.get('Original_Amount', 0)
    if amt == 0: amt = row_data.get('JE Amount', 0) 
    abs_amt = abs(amt)
    payer_entity = row_data['Entity']
    account = str(row_data['Original_Account'])
    desc = row_data['Original_Description']
    
    # Use Custom Memo if provided, otherwise default to Description
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
                 
        return [{
            "Date": row_data['Date'], "Entity": payer_entity, "JE Amount": abs_amt, 
            "Debit": debit, "Credit": credit, "Memo": final_memo, 
            "Original_Account": row_data['Original_Account'], "Original_Amount": amt, "Original_Description": desc
        }]

    if amt < 0:
        entry1 = {"Date": row_data['Date'], "Entity": payer_entity, "JE Amount": abs_amt, "Debit": f"Short Term Notes - {target_entity}", "Credit": "Cash - BB&T", "Memo": f"Paid on behalf of {target_entity}", "Original_Account": row_data['Original_Account'], "Original_Amount": amt, "Original_Description": desc}
        entry2 = {"Date": row_data['Date'], "Entity": target_entity, "JE Amount": abs_amt, "Debit": clean_category, "Credit": f"Short Term Notes - {payer_entity}", "Memo": f"Paid by {payer_entity}", "Original_Account": row_data['Original_Amount'], "Original_Amount": amt, "Original_Description": desc}
        return [entry1, entry2]
    
    return generate_corrected_je_smart(row_data, clean_category, custom_memo)


def main():
    print("--- Confirm Review (Persistent Memory) ---")
    
    try:
        review_df = pd.read_csv(REVIEW_FILE)
        je_df = pd.read_csv(JE_FILE)
        rules_df = pd.read_csv(RULES_FILE)
        
        # Load or Init History
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
        else:
            history_df = pd.DataFrame(columns=['Date', 'Amount', 'Original_Description', 'Category'])
            
        print(f"Loaded {len(review_df)} review items.")
    except Exception as e: print(f"❌ Error: {e}"); return

    new_rules = []
    rows_to_drop = []
    new_je_rows = []
    new_history = []


    print("Processing reviews...")
    actions_count = 0
    
    for idx, row in review_df.iterrows():
        action = str(row['Approve/Correct (A/C)']).strip().upper()
        add_rule_flag = str(row.get('Add Rule? (Y/N)', '')).strip().upper()
        explode_flag = str(row.get('Explode? (Y/N)', '')).strip().upper()
        keyword = row['Keyword']
        
        # FEEDBACK: Explicitly acknowledge Explode requests
        if explode_flag == 'Y': 
            print(f"💥 Explode requested for: {keyword} (Skipping Approval/Correction to allow re-run)")
            continue

        if action == 'A':
            actions_count += 1
            cat = row['Category']
            if add_rule_flag == 'Y':
                new_rules.append({'Keyword': keyword, 'Category': cat})
                print(f"✅ Approved (Rule Added): {keyword} -> {cat}")
            else:
                print(f"✅ Approved (Saved to History): {keyword}")
                new_history.append({
                    'Date': row['Date'],
                    'Amount': row['Original_Amount'],
                    'Original_Description': row['Original_Description'],
                    'Category': cat,
                    'Memo': row.get('Memo') 
                })
            
        elif action == 'C':
            actions_count += 1
            correction = str(row['Correction']).strip()
            if not correction or correction == 'nan': continue
                
            if add_rule_flag == 'Y':
                new_rules.append({'Keyword': keyword, 'Category': correction})
                print(f"🔧 Correcting (Rule Added): {keyword} -> {correction}")
            else:
                print(f"🔧 Correcting (Saved to History): {keyword} -> {correction}")
                new_history.append({
                    'Date': row['Date'],
                    'Amount': row['Original_Amount'],
                    'Original_Description': row['Original_Description'],
                    'Category': correction,
                    'Memo': row.get('Memo')
                })
            
            mask = (je_df['Original_Description'] == row['Original_Description']) & \
                   (je_df['JE Amount'] == row['JE Amount'])
            matches = je_df[mask]
            
            if not matches.empty:
                target_idx = None
                for candidate in matches.index:
                    if candidate not in rows_to_drop:
                        target_idx = candidate
                        break
                
                if target_idx is not None:
                    rows_to_drop.append(target_idx)
                    original_data = matches.loc[target_idx].to_dict()
                    new_jes = generate_corrected_je_smart(original_data, correction, custom_memo=row.get('Memo'))
                    new_je_rows.extend(new_jes)

    if actions_count == 0 and not new_rules and not new_history:
        print("ℹ️  No actions taken. (Check if 'Explode' column is blocking Approvals)")


    # --- APPLY ---
    if new_rules:
        updated_rules = pd.concat([rules_df, pd.DataFrame(new_rules)], ignore_index=True)
        updated_rules = updated_rules.drop_duplicates(subset=['Keyword'], keep='last')
        updated_rules.to_csv(RULES_FILE, index=False)
        print(f"💾 Updated Rules File.")
        
    if new_history:
        updated_history = pd.concat([history_df, pd.DataFrame(new_history)], ignore_index=True)
        # Deduplicate History (Latest fix wins)
        updated_history = updated_history.drop_duplicates(subset=['Date', 'Amount', 'Original_Description'], keep='last')
        updated_history.to_csv(HISTORY_FILE, index=False)
        print(f"🧠 Updated History File with {len(new_history)} new memories.")
    
    if rows_to_drop or new_je_rows:
        je_df_clean = je_df.drop(rows_to_drop)
        if new_je_rows:
            je_df_new = pd.DataFrame(new_je_rows)
            je_df_new = je_df_new[je_df.columns]
            je_df_final = pd.concat([je_df_clean, je_df_new], ignore_index=True)
        else: je_df_final = je_df_clean
        je_df_final.to_csv(JE_FILE, index=False)
        print(f"💾 Updated JE File.")

    print("🚀 All Done!")

if __name__ == "__main__":
    main()

