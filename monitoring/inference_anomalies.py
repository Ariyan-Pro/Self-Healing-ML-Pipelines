import numpy as np

def anomaly_rate(predictions, z_thresh=3.0):
    z_scores = (predictions - predictions.mean()) / predictions.std()
    anomalies = np.abs(z_scores) > z_thresh
    return anomalies.mean()
