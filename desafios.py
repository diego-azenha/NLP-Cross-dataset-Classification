# desafios.py
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the data directory
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"  # Directory where IMDB and Yelp files are stored

# Function to read JSON line by line
def _read_json_lines(fp: Path) -> pd.DataFrame:
    """Reads a JSON file line by line and returns a DataFrame"""
    for enc in ("utf-8", "utf-8-sig"):
        try:
            rows = []
            with open(fp, "r", encoding=enc) as f:
                for line in f:
                    s = line.strip()
                    if not s: continue
                    rows.append(json.loads(s))
            if rows: return pd.DataFrame(rows)
        except json.JSONDecodeError:
            pass
    # Attempt to read as a complete JSON array
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(fp, "r", encoding=enc) as f:
                obj = json.load(f)
            return pd.DataFrame(obj if isinstance(obj, list) else [obj])
        except Exception:
            continue
    raise SystemExit(f"Unable to interpret {fp.name} as JSON.")

# Import functions from main.py
from main import read_imdb, read_yelp, model, run, plot  # Reusing functions from main.py

# Function to run experiments with different data sizes
def run_with_different_sizes(X, y, X_test, y_test, sizes=[0.1, 0.5, 1.0]):
    """Runs tests for different data sizes, ensuring the number of samples is consistent"""
    results = []
    for size in sizes:
        # Calculate the fraction corresponding to the desired size
        train_size = size if size < 1 else (len(X) - 1) / len(X)  # Avoid training size equal to the total number of samples
        
        # Split the data for training
        X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=42)
        result = run(X_train, y_train, X_test, y_test, f"Size-{size}", "Test")
        results.append(result)
    return results

# Function to combine IMDB and Yelp and test with more data
def run_with_all_data(Xtr_imdb, ytr_imdb, Xtr_yelp, ytr_yelp, Xte_imdb, yte_imdb):
    """Tests using the combination of IMDB and Yelp data"""
    Xtr_combined = np.concatenate([Xtr_imdb, Xtr_yelp])
    ytr_combined = np.concatenate([ytr_imdb, ytr_yelp])
    return run(Xtr_combined, ytr_combined, Xte_imdb, yte_imdb, "IMDB+Yelp", "Test")

# Function to run the "Very Hard" experiment - Train with all data except the test dataset
def run_with_all_except_one(Xtr_imdb, ytr_imdb, Xte_imdb, yte_imdb, Xtr_yelp, ytr_yelp, Xte_yelp, yte_yelp):
    """Train with all data except the test dataset, and repeat for each dataset"""
    results = []
    
    # Train with IMDB + Yelp and test on Yelp
    X_train_all = np.concatenate([Xtr_imdb, Xtr_yelp])  # Training with IMDB + Yelp
    y_train_all = np.concatenate([ytr_imdb, ytr_yelp])
    result_train_all_test_yelp = run(X_train_all, y_train_all, Xte_yelp, yte_yelp, "IMDB+Yelp", "Yelp")
    results.append(result_train_all_test_yelp)
    
    # Train with IMDB + Yelp and test on IMDB
    result_train_all_test_imdb = run(X_train_all, y_train_all, Xte_imdb, yte_imdb, "IMDB+Yelp", "IMDB")
    results.append(result_train_all_test_imdb)

    return results

# Function to plot the results
def plot_results(results, title):
    """Generates graphs for the experiment results"""
    # Ensure the index is correct
    df = pd.DataFrame(results)
    
    # Ensure the keys are present in "setup"
    df['setup'] = df['train_on'] + " → " + df['test_on']
    
    # Custom sorting, now with verification
    order = ['IMDB→IMDB', 'Yelp→Yelp', 'IMDB→Yelp', 'Yelp→IMDB']
    
    # Checking if all necessary values are present
    if not all(val in df['setup'].values for val in order):
        order = df['setup'].unique().tolist()

    df = df.set_index("setup").loc[order].reset_index()

    # Create bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['setup'], df['accuracy'], color='royalblue')
    plt.ylim(0, 1)
    plt.xlabel('Experiments')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Experiment Performance')
    plt.xticks(rotation=45, ha='right')
    
    # Adding values to the bars
    for bar, acc in zip(bars, df['accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # Save the graph
    plt.tight_layout()
    plt.savefig(f'./results/{title.replace(" ", "_")}_results.png')
    plt.show()

# Function to read and preprocess Yelp data (adjusted to treat 3 stars as neutral)
def read_yelp():
    tr = _read_json_lines(DATA / "yelp_train.json")
    te = _read_json_lines(DATA / "yelp_test.json")
    
    for need in ("text", "stars"):
        if need not in tr or need not in te: 
            raise SystemExit("Yelp requires 'text' and 'stars' columns.")
    
    def _prep(df):
        # Create the 'label' column and assign 1 for positive, 0 for negative, and -1 for neutral (3 stars)
        df = df[["text", "stars"]].copy()
        df["label"] = np.where(df["stars"] >= 4, 1, np.where(df["stars"] <= 2, 0, -1))  # 3 stars as neutral
        
        # Remove samples with 'label' equal to -1 (neutral)
        df = df[df["label"] != -1]
        
        return df["text"].astype(str), df["label"].astype(int)
    
    Xtr, ytr = _prep(tr)
    Xte, yte = _prep(te)
    
    if len(Xtr) == 0 or len(Xte) == 0: 
        raise SystemExit("After filtering stars (≥4=pos, ≤2=neg), Yelp became empty.")
    
    return Xtr, ytr, Xte, yte

# Main function to run the experiments
def main():
    # Read the datasets
    Xtr_imdb, ytr_imdb, Xte_imdb, yte_imdb = read_imdb()
    Xtr_yelp, ytr_yelp, Xte_yelp, yte_yelp = read_yelp()

    # Challenge 1: Test with different data sizes
    sizes = [0.001, 0.05, 0.1, 0.5, 1.0]  # Adjusted sizes (now including 0.01 and 0.05)
    results_size_imdb = run_with_different_sizes(Xtr_imdb, ytr_imdb, Xte_imdb, yte_imdb, sizes=sizes)
    results_size_yelp = run_with_different_sizes(Xtr_yelp, ytr_yelp, Xte_yelp, yte_yelp, sizes=sizes)
    
    print("Results for different data sizes (IMDB):")
    for result in results_size_imdb:
        print(result)
    
    print("\nResults for different data sizes (Yelp):")
    for result in results_size_yelp:
        print(result)
    
    # Plot the graphs for Challenge 1
    plot_results(results_size_imdb, "Results IMDB - Different Data Sizes")
    plot_results(results_size_yelp, "Results Yelp - Different Data Sizes")

    # Challenge 2: Test with more data (IMDB + Yelp)
    results_all_data = run_with_all_data(Xtr_imdb, ytr_imdb, Xtr_yelp, ytr_yelp, Xte_imdb, yte_imdb)
    print("\nResults using more data (IMDB + Yelp):")
    print(results_all_data)
    
    # Plot the graph for Challenge 2
    plot_results(results_all_data, "Results IMDB+Yelp - More Data")

if __name__ == "__main__":
    main()