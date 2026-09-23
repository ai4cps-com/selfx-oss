def expand_vector_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    result = df.copy()

    for column in columns:
        components = (
            result[column]
            .astype("string")
            .str.strip("<>")
            .str.split(r"\.\s*", n=2, expand=True)
        )

        if components.shape[1] != 3:
            raise ValueError(f"{column!r} does not consistently contain 3 values")

        components = components.apply(
            lambda values: pd.to_numeric(
                values.str.replace(",", ".", regex=False),
                errors="coerce",
            )
        ).astype("float64")

        components.columns = [
            f"{column}.X",
            f"{column}.Y",
            f"{column}.Z",
        ]

        result[components.columns] = components
    result = result.drop(columns=columns)
    return result


def process_json_text(text: str) -> pd.DataFrame:
    """Process JSON text and return a DataFrame."""
    text = text.strip()

    if not text:
        return pd.DataFrame()

    if text.startswith("["):
        samples = json.loads(text)
    elif text.startswith("{"):
        lines = text.splitlines()
        samples = [json.loads(line.strip()) for line in lines if line.strip()]
    else:
        raise ValueError("Unsupported JSON format.")

    return process_json_samples(samples)