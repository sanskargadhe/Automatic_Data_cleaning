from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

def run_model(df, target, model_name, problem_type):
    X = df.drop(target, axis=1) if target else df
    y = df[target] if target else None

    X = X.select_dtypes(include=['int64','float64'])

    if problem_type != "clustering":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if problem_type == "regression":
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)

    elif problem_type == "classification":
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    else:
        model = KMeans(n_clusters=3)
        model.fit(X)
        return "Clustering Done"

def compare_models(df, target, problem_type):
    results = {}

    if problem_type == "classification":
        models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    elif problem_type == "regression":
        models = ["Linear Regression", "Decision Tree", "Random Forest"]

    for m in models:
        score = run_model(df, target, m, problem_type)
        results[m] = score

    return results