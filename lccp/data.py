"""Read paired fixed documents and character-span annotations."""

import gzip
from itertools import zip_longest
import json
from pathlib import Path


def read_fixed_dataset(directory):
    directory = Path(directory)
    seen = set()
    with gzip.open(directory / "documents.jsonl.gz", "rt", encoding="utf-8") as documents, \
         gzip.open(directory / "annotations.jsonl.gz", "rt", encoding="utf-8") as annotations:
        for index, pair in enumerate(zip_longest(documents, annotations)):
            if None in pair:
                raise ValueError("Document and annotation counts differ")
            document, annotation = [json.loads(line) for line in pair]
            if document["id"] != annotation["id"] or annotation["index"] != index or document["id"] in seen:
                raise ValueError("Dataset order or sample IDs differ")
            for key in ["long_spans", "short_spans"]:
                for span in annotation[key]:
                    if len(span) != 2 or any(type(value) is not int for value in span) or not 0 <= span[0] <= span[1] <= len(document["text"]):
                        raise ValueError("Invalid character span")
            seen.add(document["id"])
            yield {"id": document["id"], "text": document["text"],
                   "long_spans": annotation["long_spans"], "short_spans": annotation["short_spans"]}
