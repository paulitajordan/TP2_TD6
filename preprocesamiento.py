## aca voy a definir las funciones que voy a usar para preprocesar los datos


def process_list_columns(df, list_columns):
    for col in list_columns:
        df[f'{col}_len'] = df[col].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
        df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) and len(eval(x)) > 0 else 0)
        df = df.drop(columns=[col])
    return df

