from mistralai import Mistral
import os
import google.generativeai as genai
import pandas as pd

def fraud_text(row):

     return (
        f"Example of FRAUD transaction:\n"
        f"- Amount: {row['amount']} TRY\n"
        f"- Withdrawable balance: {row['withdrawable_cash']} TRY\n"
        f"- Customer profile: {row['yas']} years old, nationality: {row['uyruk']}, occupation: {row['meslek']}\n"
        f"- Account type: {row['hesap_acilis_tipi']}\n"
        f"- Trading activity: US market volume {row['us_borsasi_usd_cinsinden_hacim']} USD, total volume {row['usd_toplam_islem_hacmi']} USD\n"
        f"- Location: {row['ikamet_ili'] if pd.notnull(row['ikamet_ili']) else 'unspecified'}\n"
        f"- External deposits: {row['farkli_kisi_deposit_amount_try'] if pd.notnull(row['farkli_kisi_deposit_amount_try']) else 0} TRY from {row['farkli_kisi_sayisi'] if pd.notnull(row['farkli_kisi_sayisi']) else 0} different individuals\n"
        f"- Timestamp: Day {row['day']:02d}, {row['hour']:02d}:{row['minute']:02d}:{row['second']:02d}\n"
    )

def non_fraud_text(row):
    return (
        f"Example of LEGITIMATE transaction:\n"
        f"- Amount: {row['amount']} TRY\n"
        f"- Withdrawable balance: {row['withdrawable_cash']} TRY\n"
        f"- Customer profile: {row['yas']} years old, nationality: {row['uyruk']}, occupation: {row['meslek']}\n"
        f"- Account type: {row['hesap_acilis_tipi']}\n"
        f"- Trading activity: US market volume {row['us_borsasi_usd_cinsinden_hacim']} USD, total volume {row['usd_toplam_islem_hacmi']} USD\n"
        f"- Location: {row['ikamet_ili'] if pd.notnull(row['ikamet_ili']) else 'unspecified'}\n"
        f"- External deposits: {row['farkli_kisi_deposit_amount_try'] if pd.notnull(row['farkli_kisi_deposit_amount_try']) else 0} TRY from {row['farkli_kisi_sayisi'] if pd.notnull(row['farkli_kisi_sayisi']) else 0} different individuals\n"
        f"- Timestamp: Day {row['day']:02d}, {row['hour']:02d}:{row['minute']:02d}:{row['second']:02d}\n"
    )

def row_to_text(row):
    return (
        f"Analyze this transaction for potential fraud:\n"
        f"- Amount: {row['amount']} TRY\n"
        f"- Withdrawable balance: {row['withdrawable_cash']} TRY\n"
        f"- Customer profile: {row['yas']} years old, nationality: {row['uyruk']}, occupation: {row['meslek']}\n"
        f"- Account type: {row['hesap_acilis_tipi']}\n"
        f"- Trading activity: US market volume {row['us_borsasi_usd_cinsinden_hacim']} USD, total volume {row['usd_toplam_islem_hacmi']} USD\n"
        f"- Location: {row['ikamet_ili'] if pd.notnull(row['ikamet_ili']) else 'unspecified'}\n"
        f"- External deposits: {row['farkli_kisi_deposit_amount_try'] if pd.notnull(row['farkli_kisi_deposit_amount_try']) else 0} TRY from {row['farkli_kisi_sayisi'] if pd.notnull(row['farkli_kisi_sayisi']) else 0} different individuals\n"
        f"- Timestamp: Day {row['day']:02d}, {row['hour']:02d}:{row['minute']:02d}:{row['second']:02d}\n"
        f"\nBased on the patterns in the previous examples, is this transaction fraudulent? Respond only with:\n"
        f"'Yes' or 'No'\n"
        f"Followed by a very short explanation of the key factors that led to your decision only in one sentence"
    )

def final_prompt(fraud_rows, non_fraud_rows, test_row):
    f_text = ""
    for index, f in fraud_rows.iterrows():
        f_text +=  fraud_text(f) + "\n"
    
    n_fraud_text = ""
    for index, f in non_fraud_rows.iterrows():
        n_fraud_text +=  non_fraud_text(f) + "\n"
        
    test = row_to_text(test_row)
    return (
        f"You are a fraud detection expert. Analyze transactions based on historical patterns.\n\n"
        f"Known fraud patterns:\n"
        f"{f_text}\n\n"
        f"Known legitimate patterns:\n"
        f"{n_fraud_text}\n\n"
        f"New transaction to analyze:\n"
        f"{test}"
    )

def run_mistral(message):
    api_key = ""

    client = Mistral(api_key=api_key)

    model = "mistral-large-2411"
    messages = [
        {
            "role": "user",
            "content": f"{message}",
        },
    ]

    response = client.chat.complete(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content

def run_gemini(message):
    genai.configure(api_key="")

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
    history=[
    ]
    )

    response = chat_session.send_message(f"{message}")

    return response.text
