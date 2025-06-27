import pandas as pd

def inference(X_test, y_test,model):
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1,1)

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Real MAE (cycles): {mae:.2f}")

    # create dataframe
    df_y = pd.DataFrame({
        'y_real': y_test.flatten(),
        'y_pred': y_pred.flatten()
    })
    df_y.head()
    df_y.tail()
    return df_y