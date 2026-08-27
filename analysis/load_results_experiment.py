"""Load an experiment results JSON file into a pandas DataFrame."""

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Union

import pandas as pd

if __package__:
    from util.parsing import compact_long_numeric_lists
else:
    # horrivel, mas resolve
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from util.parsing import compact_long_numeric_lists


PathLike = Union[str, Path]


def load_results(path: PathLike) -> pd.DataFrame:
    """Load the records in a results_experiment.json file.

    Nested result fields are flattened using dot-separated column names,
    while list-valued fields such as test results remain as list values.
    """
    with Path(path).open(encoding="utf-8") as results_file:
        payload = json.load(results_file)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("The JSON file must contain a 'results' list.")

    dataframe = pd.json_normalize(results, sep=".")
    dataframe.attrs["metadata"] = {
        key: copy.deepcopy(value) for key, value in payload.items() if key != "results"
    }
    return dataframe


def dataframe_to_json(dataframe: pd.DataFrame, path: PathLike | None = None) -> str:
    """Convert a DataFrame returned by :func:`load_results` to JSON text.

    The original top-level metadata is read from ``dataframe.attrs`` and
    flattened result columns are reconstructed as nested objects. If ``path``
    is provided, the JSON text is also written to that file.
    """
    metadata = dataframe.attrs.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("The DataFrame must be created by load_results().")

    payload = copy.deepcopy(metadata)
    payload["results"] = []

    for record in dataframe.to_dict(orient="records"):
        nested_record: dict = {}
        for column, value in record.items():
            if not isinstance(column, str):
                raise ValueError("DataFrame columns must be strings.")
            target = nested_record
            parts = column.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        payload["results"].append(nested_record)

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_text = compact_long_numeric_lists(json_text)
    if path is not None:
        Path(path).write_text(json_text, encoding="utf-8")
    return json_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load experiment results and print the resulting DataFrame."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, default=None)
    args = parser.parse_args()

    dataframe = load_results(args.input)
    print(dataframe.to_string(index=False))
    dataframe_to_json(dataframe, args.output)
        


if __name__ == "__main__":
    main()