from prepareData import get_labeled_data
from randomForest import train_random_forest
from llm import row_to_text, run_mistral, final_prompt, run_gemini
import time

def random_forest(train_data, test_data):
    train_random_forest(train_data, test_data)

def train_model():
    aug_data, sep_data = get_labeled_data()
    random_forest(aug_data, sep_data)

def process_llm(train_data, test_data):
    fraud = train_data[train_data["fraud_label"] == 1]
    fraud.reset_index(drop=True, inplace=True)
    non_fraud = train_data[train_data["fraud_label"] == 0]
    non_fraud.reset_index(drop=True, inplace=True)
    
    sep_fraud = test_data[test_data["fraud_label"] == 0]
    sep_fraud = sep_fraud.iloc[1:100]

    for index, f in sep_fraud.iterrows():
        test_size = 100
        model = "gemini"
        message = final_prompt(fraud.head(test_size), non_fraud.head(test_size), f)

        result = run_gemini(message) if model == "gemini" else run_mistral(message)
        with open(f"llm-outputs/{model}_non_fraud_v1.txt", 'a') as f:
            line = "-" * 100
            f.write(result + f"\n{line}\n")

        time.sleep(6.5)

if __name__ == '__main__':
    train_data, test_data = get_labeled_data()

    process_llm(train_data, test_data)