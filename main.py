# main.py
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore")

# Define the base and data directories
BASE   = Path(__file__).resolve().parent
DATA   = BASE / "data"
OUTDIR = BASE / "results"; OUTDIR.mkdir(exist_ok=True)  # Create results directory if it doesn't exist

# Function to read the IMDB dataset
def read_imdb():
    try:
        # Load train and test datasets for IMDB
        tr = pd.read_parquet(DATA / "imdb_train.parquet")
        te = pd.read_parquet(DATA / "imdb_test.parquet")
    except Exception as e:
        raise SystemExit("Failed to open Parquet files. Install pyarrow: pip install pyarrow\n" + str(e))
    
    # Check for necessary columns
    for need in ("text", "label"):
        if need not in tr or need not in te:
            raise SystemExit("IMDB needs 'text' and 'label' columns.")
    
    # Return the training and testing data
    return tr["text"].astype(str), tr["label"].astype(int), te["text"].astype(str), te["label"].astype(int)

# Function to read JSON lines from a file and return a DataFrame
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
    # Try reading as a full JSON array
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(fp, "r", encoding=enc) as f:
                obj = json.load(f)
            return pd.DataFrame(obj if isinstance(obj, list) else [obj])
        except Exception:
            continue
    raise SystemExit(f"Unable to interpret {fp.name} as JSON.")

# Function to read the Yelp dataset
def read_yelp():
    tr = _read_json_lines(DATA / "yelp_train.json")
    te = _read_json_lines(DATA / "yelp_test.json")
    
    # Check if necessary columns are present
    for need in ("text", "stars"):
        if need not in tr or need not in te: 
            raise SystemExit("Yelp requires 'text' and 'stars' columns.")
    
    # Prepare the dataset (labeling as positive, negative, or neutral)
    def _prep(df):
        df = df[["text", "stars"]].copy()
        # Label: 1 for positive, 0 for negative, -1 for neutral (3 stars)
        df["label"] = np.where(df["stars"] >= 4, 1, np.where(df["stars"] <= 2, 0, -1))  # 3 stars as neutral
        
        # Remove samples with 'label' equal to -1 (neutral)
        df = df[df["label"] != -1]
        
        return df["text"].astype(str), df["label"].astype(int)
    
    Xtr, ytr = _prep(tr)
    Xte, yte = _prep(te)
    
    if len(Xtr) == 0 or len(Xte) == 0: 
        raise SystemExit("After filtering stars (≥4=pos, ≤2=neg), Yelp became empty.")
    
    return Xtr, ytr, Xte, yte

# Function to define the model pipeline (TF-IDF + Logistic Regression)
def model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                  ngram_range=(1,2), min_df=2, max_features=100_000)),  # TF-IDF vectorizer
        ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42))  # Logistic Regression classifier
    ])

# Function to run the model on the dataset and return the results
def run(Xtr, ytr, Xte, yte, trn, tst):
    m = model()  # Create the model
    m.fit(Xtr, ytr)  # Train the model
    acc = accuracy_score(yte, m.predict(Xte))  # Get accuracy on the test set
    return {"train_on": trn, "test_on": tst, "n_train": len(Xtr), "n_test": len(Xte), "accuracy": float(acc)}

# Function to plot the results
def plot(df):
    df = df.assign(setup=df["train_on"] + " → " + df["test_on"])  # Create a new column for the experiment setup
    order = ["IMDB→IMDB", "Yelp→Yelp", "IMDB→Yelp", "Yelp→IMDB"]  # Define custom order for the experiments
    
    df = df.set_index("setup").loc[order].reset_index()  # Sort the dataframe
    
    plt.figure(figsize=(8.5, 6))  # Create a figure for the plot
    bars = plt.bar(df["setup"], df["accuracy"], color='royalblue')  # Create bars for each experiment result
    plt.ylim(0, 1)  # Set the y-axis limit
    plt.ylabel("Accuracy")
    plt.title("Same vs Cross-dataset Sentiment (TF-IDF + Logistic Regression)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)  # Add gridlines
    
    # Add accuracy values above each bar
    for b, v in zip(bars, df["accuracy"]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    
    # Add a caption for the plot
    caption = ("Bars show test accuracy. Same-dataset: IMDB→IMDB, Yelp→Yelp. "
               "Cross-dataset: IMDB→Yelp, Yelp→IMDB. Yelp labels: stars≥4 positive; ≤2 negative (3 dropped).")
    plt.gcf().text(0.5, -0.08, caption, ha="center", va="top", fontsize=9, wrap=True)
    plt.tight_layout()  # Adjust layout
    plt.savefig(OUTDIR / "cross_domain_results.png", bbox_inches="tight")  # Save the plot as PNG
    plt.close()  # Close the plot

# Main function to run the challenges
def main():
    # Read the datasets
    Xtr_i, ytr_i, Xte_i, yte_i = read_imdb()
    Xtr_y, ytr_y, Xte_y, yte_y = read_yelp()

    # Define the scenarios for the experiments
    scenarios = [("IMDB", "IMDB", Xtr_i, ytr_i, Xte_i, yte_i),
                 ("Yelp", "Yelp", Xtr_y, ytr_y, Xte_y, yte_y),
                 ("IMDB", "Yelp", Xtr_i, ytr_i, Xte_y, yte_y),
                 ("Yelp", "IMDB", Xtr_y, ytr_y, Xte_i, yte_i)]
    
    # Run the experiments
    results = [run(Xtr, ytr, Xte, yte, trn, tst) for trn, tst, Xtr, ytr, Xte, yte in scenarios]
    
    # Convert the results to a DataFrame
    df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    df.to_csv(OUTDIR / "cross_domain_results.csv", index=False)
    
    # Plot the results
    plot(df)
    
    # Print the results
    print(df[["train_on", "test_on", "n_train", "n_test", "accuracy"]])

# Run the main function if the script is executed
if __name__ == "__main__":
    main()