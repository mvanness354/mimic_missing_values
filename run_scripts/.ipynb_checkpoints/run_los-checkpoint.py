import os
import numpy as np
import pandas as pd
import argparse
import yaml
import itertools
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from hypopt import GridSearch
import keras_tuner

import sys
sys.path.append("../../")
from utils import get_tf_dataset
from tf_models import MLPClassifier, NeuMissMLP, MLPMIWAE, MLPNotMIWAE

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

# Model Map
model_map = {
    "mlp": MLPClassifier,
    "neumiss": NeuMissMLP,
    "supmiwae": MLPMIWAE,
    "supnotmiwae": MLPNotMIWAE,
    "gbt": HistGradientBoostingClassifier,
}

# Read in mortality data
train_data = pd.read_csv("../../mimic/processed_data/los_tab_train.csv")
train_X = train_data[[c for c in train_data.columns if c != "target"]].copy(deep=False)
train_X = train_X[[c for c in train_X.columns if "full" in c and "mean" in c]]
train_y = train_data["target"].copy(deep=False)

val_data = pd.read_csv("../../mimic/processed_data/los_tab_val.csv")
val_X = val_data[[c for c in val_data.columns if c != "target"]].copy(deep=False)
val_X = val_X[[c for c in val_X.columns if "full" in c and "mean" in c]]
val_y = val_data["target"].copy(deep=False)

test_data = pd.read_csv("../../mimic/processed_data/los_tab_test.csv")
test_X = test_data[[c for c in test_data.columns if c != "target"]].copy(deep=False)
test_X = test_X[[c for c in test_X.columns if "full" in c and "mean" in c]]
test_y = test_data["target"].copy(deep=False)

# Prepare dataset
preprocessor = make_pipeline(
    StandardScaler(),
)
train_X = preprocessor.fit_transform(train_X).clip(min=-10, max=10)
val_X = preprocessor.transform(val_X).clip(min=-10, max=10)
test_X = preprocessor.transform(test_X).clip(min=-10, max=10)

# Concat the mask if MLP
if args.model == "mlp":
    train_X = np.column_stack([train_X, np.isnan(train_X).astype(int)])
    val_X = np.column_stack([val_X, np.isnan(val_X).astype(int)])
    test_X = np.column_stack([test_X, np.isnan(test_X).astype(int)])
    
# Read in defeault task params
with open("../task_params.yaml", 'r') as f:
    model_params = yaml.safe_load(f)["los"][args.model]
    
# Read in best hpo params
with open(f"../hpo/los/best_configs/{args.model}_los_params.yaml", 'r') as f:
    hpo_params = yaml.safe_load(f)
    
# Combine model params
model_params = {**model_params, **hpo_params}
    
if args.model == "gbt":
    np.random.seed(args.seed)
    model_params["random_state"] = args.seed
    
    model = model_map[args.model](**model_params)
    model.fit(train_X, train_y)
    score = accuracy_score(test_y, model.predict(test_X))
    
    score_df = pd.DataFrame([[args.seed, args.model, score]], columns=["seed", "model", "score"])
    score_df.to_csv(f"../results/los.csv", mode='a', index=False, header=False)
    
else:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    batch_size = 128
    train_dataset = get_tf_dataset([train_X, train_y], batch_size)
    val_dataset = get_tf_dataset([val_X, val_y], batch_size)
    test_dataset = get_tf_dataset([test_X, test_y], batch_size)

    model = model_map[args.model](**model_params)

    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
    )


    checkpoint_path = "checkpoints/"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode="max", min_delta=1e-4, patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_acc',
            mode='max',
            save_best_only=True,
        )
    ]
    model.fit(
        train_dataset,
        epochs=100,
        validation_data = val_dataset,
        callbacks = callbacks,
    )
    
    model.load_weights(checkpoint_path)
    outputs = model.evaluate(test_dataset)
    score = dict(zip(model.metrics_names, outputs))["acc"]
    
    score_df = pd.DataFrame([[args.seed, args.model, score]], columns=["seed", "model", "score"])
    score_df.to_csv(f"../results/los.csv", mode='a', index=False, header=False)
    
    

