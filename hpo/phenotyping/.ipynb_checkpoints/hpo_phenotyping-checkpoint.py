import os
import numpy as np
import pandas as pd
import argparse
import yaml
import itertools
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from hypopt import GridSearch
import keras_tuner

import sys
sys.path.append("../../../")
from utils import get_tf_dataset
from tf_models import MLPMultiLabel, NeuMissMLP, MLPMIWAE, MLPNotMIWAE

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

# Model Map
model_map = {
    "mlp": MLPMultiLabel,
    "neumiss": NeuMissMLP,
    "supmiwae": MLPMIWAE,
    "supnotmiwae": MLPNotMIWAE,
    "gbt": HistGradientBoostingClassifier,
}

# Read in mortality data
train_X = pd.read_csv("../../../mimic/processed_data/phenotyping_tab_train.csv")
train_X = train_X[[c for c in train_X.columns if "full" in c and "mean" in c]]
train_y = pd.read_csv("../../../mimic/processed_data/phenotyping_train_labels.csv")

val_X = pd.read_csv("../../../mimic/processed_data/phenotyping_tab_val.csv")
val_X = val_X[[c for c in val_X.columns if "full" in c and "mean" in c]]
val_y = pd.read_csv("../../../mimic/processed_data/phenotyping_val_labels.csv")

test_X = pd.read_csv("../../../mimic/processed_data/phenotyping_tab_test.csv")
test_X = test_X[[c for c in test_X.columns if "full" in c and "mean" in c]]
test_y = pd.read_csv("../../../mimic/processed_data/phenotyping_test_labels.csv")

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

# Read in hpo grid
with open("../hpo_grid.yaml", 'r') as f:
    hyperparam_grid = yaml.safe_load(f)[args.model]
    
# Read in defeault task params
with open("../../task_params.yaml", 'r') as f:
    model_params = yaml.safe_load(f)["phenotyping"][args.model]
    
if args.model == "gbt":
    np.random.seed(args.seed)
    model_params["random_state"] = args.seed
    
    model = MultiOutputClassifier(
        model_map[args.model](**model_params)
    )
    
    for param, values in list(hyperparam_grid.items()):
        hyperparam_grid[f"estimator__{param}"] = values
        del hyperparam_grid[param]
        
    print(hyperparam_grid)
    
    def multilabel_auc(y, y_pred, **kwargs):
        return roc_auc_score(y, np.column_stack([probs[:, 1] for probs in y_pred]))
    
    model_grid = GridSearch(model, param_grid=hyperparam_grid, num_threads=1, parallelize=False)
    model_grid.fit(train_X, train_y, val_X, val_y, scoring=make_scorer(multilabel_auc, needs_proba=True))
    print(model_grid.get_best_params())
    
    with open(f"best_configs/{args.model}_phenotyping_params.yaml", 'w') as f:
        yaml.dump(model_grid.get_best_params(), f)
    
else:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    batch_size = 128
    train_dataset = get_tf_dataset([train_X, train_y], batch_size)
    val_dataset = get_tf_dataset([val_X, val_y], batch_size)
    test_dataset = get_tf_dataset([test_X, test_y], batch_size)
    
    def build_model(hp):
        for param, values in hyperparam_grid.items():
            model_params[param] = hp.Choice(param, values)

        model = model_map[args.model](**model_params)

        model.compile(
            optimizer = "adam",
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics = [tf.keras.metrics.AUC(name="auc", from_logits=False, multi_label=True, num_labels=25)],
        )

        return model

    max_trials = np.prod([
        len(values) for values in hyperparam_grid.values()
    ])
    tuner = keras_tuner.RandomSearch(
        hypermodel = build_model,
        objective = keras_tuner.Objective("val_auc", direction="max"),
        max_trials = max_trials,
        seed=args.seed,
        directory="checkpoints/",
        project_name=f"{args.model}_phenotyping_hpo_checkpoints",
    )

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode="max", min_delta=1e-4, patience=5)]
    tuner.search(
        train_dataset,
        epochs=100,
        validation_data = val_dataset,
        callbacks = callbacks,
    )
    best_hp = tuner.get_best_hyperparameters()[0]
    
    best_hp_dict = {
        param: best_hp.get(param) for param in hyperparam_grid
    }
    
    with open(f"best_configs/{args.model}_phenotyping_params.yaml", 'w') as f:
        yaml.dump(best_hp_dict, f)
    
    

