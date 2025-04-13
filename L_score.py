import pandas as pd

def L_score(y_true, y_pred):
    '''
    Funkcja oblicza L_score na podstawie danych y_true i y_pred.
    y_true powinno być DataFrame lub Series z kolumnami 'label' i 'clicked'.
    params:
    y_true: DataFrame lub Series z kolumnami 'label' i 'clicked'
    y_pred: DataFrame lub Series z kolumną 'y_pred'
    return:
    L_score: float
    '''
    
    if isinstance(y_true, pd.Series):
        y_true_df = y_true.to_frame()
    else:
        y_true_df = y_true.copy()
    
    y_true_df['y_pred'] = y_pred

    correct_preds = y_true_df[y_true_df['label'] == y_true_df['y_pred']]

    filtered = correct_preds[correct_preds['clicked'] == 1]

    score = filtered.shape[0] / y_true_df['clicked'].sum()

    return score