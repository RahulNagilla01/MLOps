from classifier import load_data, get_features_and_target, train_and_save_best_model

df = load_data()
X, y = get_features_and_target(df)
print("Training and comparing models...")
model = train_and_save_best_model(X, y)
print("Training pipeline complete.")
