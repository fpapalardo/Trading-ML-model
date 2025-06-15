from sklearn.metrics import accuracy_score

def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    train_preds = model.predict(Xtr)
    test_preds = model.predict(Xte)

    train_acc = accuracy_score(ytr, train_preds)
    test_acc = accuracy_score(yte, test_preds)

    print(f"\n📊 {name} Classification Accuracy:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_preds