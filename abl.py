# Shadow-model ablation: K in {1, 3, 5, 10}, 3 seeds each
import numpy as np, torch
from preprocess_data import *
from Target_Models.target_model_1a import *
from Target_Models.target_model_2a import *
from Target_Models.target_model_2b import *
from Target_Models.target_model_2c import *
from Target_Models.target_model_1c import *
from Target_Models.target_model_3a import *
from Target_Models.target_model_3b import *
from Target_Models.target_model_3c import *
from Target_Models.target_model_3d import *
from MIA_Functions import train_shadow_models_and_attack_model, create_membership_dataframe, evaluate_attack_model_get_stats

target = torch.load('models/1a_dataset_target_model.pth')
retrain = torch.load('models/1a_dataset_retrain_baseline_model.pth')
results = []
for K in [1, 3, 5, 10]:
    for seed in [0, 1, 2]:
        np.random.seed(seed); torch.manual_seed(seed)
        attack = train_shadow_models_and_attack_model(
            TargetModel_1c, X_shadow, y_shadow, K, num_features, num_classes,
            batch_size=32, learning_rate=0.01, num_epochs=100)
        row = {'K': K, 'seed': seed}
        for name, model in [('target', target), ('retrain', retrain)]:
            for split, members in [('forget', forget_member_df), ('retain', retain_member_df)]:
                df = create_membership_dataframe(model, members, train_non_member_data)
                row[f'{name}_{split}'] = evaluate_attack_model_get_stats(df, attack)[0]
        results.append(row)
import pandas as pd
print(pd.DataFrame(results).groupby('K').agg(['mean', 'std']))