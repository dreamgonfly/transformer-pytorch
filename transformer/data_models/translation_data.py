from dataclasses import dataclass
from pathlib import Path
from typing import List
import json


@dataclass
class TranslationDataPair:
    source: str
    target: str

    def to_json(self):
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(source=json_data["source"], target=json_data["target"])


@dataclass
class TranslationDataList:
    pairs: List[TranslationDataPair]

    def save(self, path: Path):
        with path.open("w") as file:
            for data_pair in self.pairs:
                line = json.dumps(data_pair.to_json(), ensure_ascii=False)
                file.write(line + "\n")

    @classmethod
    def load(cls, path: Path):
        data_pairs = []
        with path.open() as file:
            for line in file:
                line_data = json.loads(line)
                data_pair = TranslationDataPair.from_json(line_data)
                data_pairs.append(data_pair)
        return cls(pairs=data_pairs)
