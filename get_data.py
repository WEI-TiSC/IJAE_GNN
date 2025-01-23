import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


USE_FEATS = ['Model Year', 'Uphill or Downhill', 'Related to Intersection', 'Clock-form Direction of force', 'Day in Week',
             'Race', 'Maneuver before collision', 'year', 'Number of lanes', 'Body Category', 'Alignment of Road',
             'Curb Weight', 'Age', 'Traffic Conrtol Functioning', 'weight', 'Pre-event Location', 'Lighting Condition',
             'Climate', 'Traffic Condition', 'Traffic Flow Situation', 'premovement before collision', 'month',
             'Alcohol Present', 'Surface Type', 'Crash Type', 'height', 'Sex', 'Surface Condition', 'Speed Limit',
             'Distracted in Driving', 'InjurySeverity']


def get_processed_data(input_df: str):
    """
    Clean & drop useless features
    """
    df_raw = pd.read_csv(input_df)
    df_raw = df_raw.replace(97, np.nan)
    df_raw = df_raw.replace(65536, np.nan)
    df_features = df_raw[USE_FEATS]
    df_features = df_features.reset_index(drop=True)

    for feat in df_features.columns:
        lack_count = df_features[feat].isnull().sum()
        if lack_count:
            print(f'缺失值:  {feat}, 其非缺失值比例： {round(1 - (lack_count / len(df_features)), 3)}')

    df_features = df_features.dropna(axis=0)

    feats, labels = df_features.drop(columns=['InjurySeverity']), df_features['InjurySeverity']
    feats = feats.astype('int')

    # z-score standardization
    scaler = StandardScaler()
    feats_normalized = scaler.fit_transform(feats)
    feats_normalized = pd.DataFrame(feats_normalized, columns=feats.columns)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(save_dir, exist_ok=True)
    feats.to_csv(os.path.join(save_dir, 'features.csv'), index=False)
    feats_normalized.to_csv(os.path.join(save_dir, 'features_norm.csv'), index=False)
    labels.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)

    return feats, feats_normalized, labels


if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'nhtsa.csv')
    get_processed_data(input_dir)
