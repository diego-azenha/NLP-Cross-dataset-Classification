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

warnings.filterwarnings("ignore")
BASE   = Path(__file__).resolve().parent
DATA   = BASE / "data"
OUTDIR = BASE / "results"; OUTDIR.mkdir(exist_ok=True)

def read_imdb():
    print('Lendo IMDB')
    try:
        tr = pd.read_parquet(DATA / "imdb_train.parquet")
        te = pd.read_parquet(DATA / "imdb_test.parquet")
    except Exception as e:
        raise SystemExit("Falha ao abrir Parquet. Instale pyarrow: pip install pyarrow\n" + str(e))
    for need in ("text","label"):
        if need not in tr or need not in te: raise SystemExit("IMDB precisa de colunas 'text' e 'label'.")
    return tr["text"].astype(str), tr["label"].astype(int), te["text"].astype(str), te["label"].astype(int)

def _read_json_lines(fp: Path) -> pd.DataFrame:
    # Robusto: aceita JSONL, BOM, linhas vazias; se falhar, tenta array JSON único
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
    # Tenta JSON array completo
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(fp, "r", encoding=enc) as f:
                obj = json.load(f)
            return pd.DataFrame(obj if isinstance(obj, list) else [obj])
        except Exception:
            continue
    raise SystemExit(f"Não foi possível interpretar {fp.name} como JSON.")

def read_yelp():
    tqdm.write("Lendo Yelp…")
    tr = _read_json_lines(DATA / "yelp_train.json")
    te = _read_json_lines(DATA / "yelp_test.json")
    for need in ("text","stars"):
        if need not in tr or need not in te: raise SystemExit("Yelp precisa de colunas 'text' e 'stars'.")
    def _prep(df):
        df = df[["text","stars"]].copy()
        df["label"] = np.where(df["stars"]>=4,1, np.where(df["stars"]<=2,0, np.nan))
        df = df.dropna(subset=["label"])
        return df["text"].astype(str), df["label"].astype(int)
    Xtr, ytr = _prep(tr); Xte, yte = _prep(te)
    if len(Xtr)==0 or len(Xte)==0: raise SystemExit("Após filtrar estrelas (≥4=pos, ≤2=neg), Yelp ficou vazio.")
    return Xtr, ytr, Xte, yte

def model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                  ngram_range=(1,2), min_df=2, max_features=100_000)),
        ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42))
    ])

def run(Xtr,ytr,Xte,yte, trn, tst):
    m = model()
    p = tqdm(total=2, desc=f"{trn}→{tst}")
    m.fit(Xtr,ytr); p.update(1)
    acc = accuracy_score(yte, m.predict(Xte)); p.update(1); p.close()
    return {"train_on":trn,"test_on":tst,"n_train":len(Xtr),"n_test":len(Xte),"accuracy":float(acc)}

def plot(df):
    df = df.assign(setup=df["train_on"]+"→"+df["test_on"])
    order = ["IMDB→IMDB","Yelp→Yelp","IMDB→Yelp","Yelp→IMDB"]
    df = df.set_index("setup").loc[order].reset_index()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.5,6))
    bars = plt.bar(df["setup"], df["accuracy"])
    plt.ylim(0,1); plt.ylabel("Accuracy")
    plt.title("Same vs Cross-dataset Sentiment (TF-IDF + Logistic Regression)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b,v in zip(bars, df["accuracy"]):
        plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    caption = ("Bars show test accuracy. Same-dataset: IMDB→IMDB, Yelp→Yelp. "
               "Cross-dataset: IMDB→Yelp, Yelp→IMDB. Yelp labels: stars≥4 positive; ≤2 negative (3 dropped).")
    plt.gcf().text(0.5, -0.08, caption, ha="center", va="top", fontsize=9, wrap=True)
    plt.tight_layout(); plt.savefig(OUTDIR/"cross_domain_results.pdf", bbox_inches="tight"); plt.close()

def main():
    Xtr_i,ytr_i,Xte_i,yte_i = read_imdb()
    Xtr_y,ytr_y,Xte_y,yte_y = read_yelp()
    scenarios = [("IMDB","IMDB",Xtr_i,ytr_i,Xte_i,yte_i),
                 ("Yelp","Yelp",Xtr_y,ytr_y,Xte_y,yte_y),
                 ("IMDB","Yelp",Xtr_i,ytr_i,Xte_y,yte_y),
                 ("Yelp","IMDB",Xtr_y,ytr_y,Xte_i,yte_i)]
    results = [run(Xtr,ytr,Xte,yte,trn,tst) for trn,tst,Xtr,ytr,Xte,yte in scenarios]
    df = pd.DataFrame(results)
    df.to_csv(OUTDIR/"cross_domain_results.csv", index=False)
    plot(df)
    print(df[["train_on","test_on","n_train","n_test","accuracy"]])

if __name__ == "__main__":
    main()
