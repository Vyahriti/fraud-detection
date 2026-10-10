# Evaluating XGBoost v.s. SageGNN v.s. GCN

| Model | AUPRC (mean ± std, 3 seeds) | Lift over naive (3.4% fraud) |
|---|---|---|
| XGBoost | 0.517 (single run) | ~15x |
| GraphSAGE | 0.450 ± 0.004 | ~13x |
| MLP (same features, no edges) | 0.435 ± 0.003 | ~12.6x |
| GCN | 0.357 ± 0.002 | ~10.4x |

The goal of this project was to see which model performed best on the IEEE-CIS Fraud Detection dataset. XGBoost outperformed all the graph models, but within the graph models, GraphSAGE with hub-based edges outperformed MLP without edges.

## Data and split

- IEEE CIS Fraud Detection Dataset
- Using "TransactionDT", the dataset is cut at 80% of the earliest transactions for training, and the rest for testing

## Preprocessing

1. Grouped the "V" columns by how many NaNs they have: those that are missing on exactly the same rows likely come from the same source. 14 groups were made
2. Compute the spearman correlations for a correlation matrix. Use union-find to link any columns with more than 0.9 correlation and group them into clusters. Leave only the highest variance column in each cluster
   - 339 "V" columns -> 165
3. Step #2 repeated on "D" and "C" columns.
   - 15 "D" columns -> 11; 14 "C" columns -> 12
4. One-hot encoding or frequency encoding on rest of the columns

**Leakage fix:** The encoders, frequency maps and correlation pruning were originally fit on all of the data before the train/test split, which leaked test-period information into the features. This was caught and fixed: the split now happens first, and everything is fit on the training data only and then applied to the test data.

## Baseline Model

1. XGBClassifier with 300 trees
   - Used scale_pos_weight of non-fraud count/ fraud-count to give fraud a higher weightage in the loss
2. AUPRC of 0.517- accuracy isn't valuable when fraud is so rare
3. Checked feature importance: "V258" carries 23% importance
4. Compared how often a feature is split and how useful that split is to see this importance and rarity of features

## Graph Models

1. Created a hub node per attribute value to mitigate the effects of dominating attributes in a feature
   - Links every transactions with that value to the hub, in both directions
   - To scale down, dropped values that covered more than 1.5% of that columns' rows
2. Created node features. Gave hub nodes all-zero features plus an `is_hub` flag specifying it's a hub node (to not confuse with the 0s of transaction nodes)
3. Use labels 0/1 for fraud, and -1 for hubs. Ensures loss & metrics only use real transactions
4. Created evaluation graph where no test value creates a new hub- test transaction is only attached to hubs created from training
5. Standard scaling is applied because neural networks are sensitive to feature scale (trees aren't)
6. Used SAGEConv and GCNConv with 2 convolution layers to enable information travel from transaction -> hub -> transaction

## Results and Ablation

- No edge MLP: same model runs without the edges. Did slightly worse than SAGEConv but better than GCNConv
- Tested each on 3 seeds- reported AUPRC and std (see table at top)
- GCN underperformed even the no-edge MLP; a plausible explanation is that its shared weight blends each transaction with near-zero hub features. This was not isolated experimentally.

## Limitations

- NaNs were zero-filled before scaling, which conflates a missing value with a zero value
- The graph uses only 9 attribute columns: `["card1", "card2", "card3", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "dist1", "dist2"]`
- One temporal split, 3 seeds, no hyperparameter tuning. The goal was a fair baseline comparison rather than maximizing performance.
- XGBoost was run once with default settings so seed variance is expected to be small but it wasn't measured.

## Reproduction

**Requirements:** Python 3.14.2 and a Kaggle account. Developed and run on a machine with 32 GB of RAM (peak memory usage wasn't measured). Everything ran on CPU; the full notebook takes roughly 21 minutes (the 9-run seed loop is the slowest step)

1. **Environment:** `pip install -r requirements.txt` (PyTorch Geometric may need a platform-specific install; see https://pytorch-geometric.readthedocs.io)
2. **Data:** The data is not included in this repo. Join the competition at https://www.kaggle.com/c/ieee-fraud-detection (you must accept the rules), then authenticate with Kaggle credentials. The first notebook cell downloads `train_transaction.csv` via `kagglehub`. Only that file is used; `train_identity.csv` is not
3. **Run:** Open FraudDetection.ipynb and use Restart & Run All. Order matters: the split happens in cell 5, before any feature fitting, and the scaling cell must run only once per kernel session
4. **Notes:**
   - GNN runs are seeded (0, 1, 2); small differences in the third decimal are possible across hardware and library versions.
   - XGBoost was run once without a fixed seed.
