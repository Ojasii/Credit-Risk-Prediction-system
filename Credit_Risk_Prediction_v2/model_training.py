from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    
    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return grid.best_estimator_, report
    
    
