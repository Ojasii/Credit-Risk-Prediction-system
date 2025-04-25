
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Important Features')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
