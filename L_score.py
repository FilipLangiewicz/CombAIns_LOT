import pandas as pd

def L_score(y_true, y_pred):
    # Przekształcamy y_pred do DataFrame, aby było kompatybilne z pandas merge
    y_pred_df = pd.DataFrame(y_pred, columns=['y'])
    
    # Zakładając, że y_true jest Series, konwertujemy ją na DataFrame
    y_true_df = pd.DataFrame(y_true)
    
    # Łączenie y_true z y_pred po kolumnie 'y'
    joined = y_true_df.merge(y_pred_df, left_index=True, right_index=True, how='inner')
    
    # Filtrowanie tylko tych, gdzie 'clicked' == 1
    filtered_data = joined[joined['clicked'] == 1]
    
    # Obliczanie L_score
    L_score = filtered_data.shape[0] / sum(y_true_df['clicked'])
    
    return L_score
