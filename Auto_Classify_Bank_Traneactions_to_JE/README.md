
# 🧠 Levelup AI Accountant: Assisted Automation System

## 🌟 Executive Summary
This system is a **hybrid "Learning Autocorrect" engine** for accounting. It automates the classification of bank transactions and generates complex Journal Entries by combining **Deterministic Python Logic** with **Probabilistic AI**.

**Why this is superior to pure AI or pure Python:**
* **vs. Pure Python:** Pure scripts are rigid. They break when a vendor changes their name slightly (e.g., "Home Depot" vs "The Home Depot"). Our AI layer handles this fuzziness effortlessly.
* **vs. Pure AI:** AI can hallucinate or be inconsistent. It doesn't know your specific business logic (e.g., "Always split distributions 50/50"). Our Logic Rules layer enforces these accounting absolutes.
* **The "Learning" Effect:** The system possesses **Long-Term Memory**.
    * **Global Rules:** "Always map PEPCO to Utilities."
    * **Specific Memory:** "That one time I corrected an Amazon purchase to 'Personal' on Jan 12th, remember that forever."

Every time you review and correct a transaction, the system saves that decision. Over time, the AI has to do less and less work, and the system becomes tailored specifically to your business.

---

## ⚙️ How the Engine Works (The "Waterfall")

When a transaction enters the system, it passes through **4 Logic Layers**. It only moves to the next layer if the previous one didn't find a match.

1.  **🧠 History (Golden Record):**
    * *Check:* "Have I seen this exact transaction (Date + Amount + Desc) before?"
    * *Action:* If yes, apply the **exact manual fix** you made last time. (Bypasses everything else).
2.  **⚡ Special Logic (The Enforcer):**
    * *Check:* Does this match a complex accounting rule? (e.g., "Intercompany Transfer," "Partner Distro Split," "Loan Zero-Out").
    * *Action:* Apply the complex 2-line or 4-line journal entry logic defined in `logic_rules.csv`.
3.  **📖 Transaction Rules (The Dictionary):**
    * *Check:* Does the description contain a known keyword? (e.g., "PEPCO", "WAWA", "VERIZON").
    * *Action:* Apply the mapped category from `transaction_rules.csv`.
4.  **🤖 The AI Waterfall (The Safety Net):**
    * *Check:* "I have no idea what this is."
    * *Action:* Send a batch of unknown items to Google Gemini.
    * *Waterfall Strategy:* The system tries the **Fastest Model** first. If it fails or hallucinates invalid JSON, it automatically falls back to the **Stronger/Slower Model** to ensure accuracy.

---

## 📂 File Structure

* **`1_import_raw_transactions/`** (Inputs)
    * `Data_in.csv`: Your raw bank export (Checking + Credit Card combined).
    * `transaction_rules.csv`: Global keyword rules (AI learns these).
    * `logic_rules.csv`: Advanced logic (Zero-outs, Splits).
    * `correction_history.csv`: **Persistent Memory** of your manual overrides.
* **`2_python_scripts/`** (The Engine)
    * `classify.py`: **The Generator.** Reads inputs -> Generates output.
    * `confirm_review_and_update.py`: **The Patcher.** Reads review -> Updates memory & output.
* **`3_script_outputs/`** (Outputs)
    * `new_mappings_to_review.csv`: Your review dashboard.
    * `categorized_journal_entries.csv`: Final import-ready file.

---

## 🚀 The Workflow

### Phase 1: Global Rules & "Easy Stuff"
*Goal: Teach the system permanent rules and handle the bulk of transactions.*

1.  **Run Generator:** `python classify.py`
2.  **Review (`new_mappings_to_review.csv`):**
    * **Recurring Vendors:** Mark `A` (Approve) and set **Add Rule?** to `Y`.
    * **Messy Vendors (Amazon/Home Depot):** Set **Explode?** to `Y` (Do not approve yet).
3.  **Run Patcher:** `python confirm_review_and_update.py`
    * *Result:* Saves new Global Rules.

### Phase 2: The Deep Dive (Exploded View)
*Goal: Handle complex, mixed-use vendors individually.*

4.  **Run Generator (Again):** `python classify.py`
    * *Result:* Rules are applied. "Exploded" vendors are expanded into individual lines.
5.  **Review (`new_mappings_to_review.csv`):**
    * **Specific Fixes:** Mark `C` (Correct) and enter the Category.
    * **Add Rule?**: Generally leave **Blank**. (This saves it to History, not Global Rules).
    * **Intercompany:** Use `Category [Entity]` syntax (e.g., `Repairs [ENTITY_B]`) to trigger automatic intercompany splits.
6.  **Run Patcher:** `python confirm_review_and_update.py`
    * *Result:* Updates the Journal Entries file and saves your fixes to `correction_history.csv`.

### Phase 3: Completion 🛑
*Goal: Final check.*

7.  **STOP.**
    * **DO NOT** run `classify` again immediately.
    * *Why?* You are done. Your file is ready.
8.  **Verify:** Open `categorized_journal_entries.csv`. This is your final file.

*(Note: If you do run Classify again later, don't worry—the `correction_history.csv` will remember your Phase 2 fixes and re-apply them automatically.)*

---

## 💡 Smart Features Cheat Sheet

| Feature | How to Trigger | What it Does |
| :--- | :--- | :--- |
| **Global Rule** | Review File: **Add Rule? = Y** | "Always map 'STARBUCKS' to 'Meals' forever." |
| **One-Off Fix** | Review File: **Add Rule? = Blank** | "Fix this specific $12.50 Starbucks charge, but don't change the rule." (Saved to History) |
| **Explode Mode** | Review File: **Explode? = Y** | "Show me every single Home Depot transaction so I can classify them one by one." |
| **Intercompany Tag** | Category: **Repairs [ENTITY_B]** | Automatically creates a 2-way entry: <br>1. Levelup pays on behalf of ENTITY_B (Loan).<br>2. ENTITY_B recognizes the expense (Repairs). |
| **System Toggle** | `logic_rules.csv` | Turn features like "Auto Intercompany" on/off without touching code. |
