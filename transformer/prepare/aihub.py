from pathlib import Path

from tqdm import tqdm
from typer import Argument

from transformer.data_models.translation_data import TranslationDataPair, TranslationDataList
import csv
import random


def prepare_aihub(
    data_dir: Path = Argument(..., help="Path to root data directory. (e.g. data/AIHub).")
):
    csv_paths = data_dir.joinpath("csv").glob("*.csv")

    prepared_train_data_list_path = data_dir.joinpath("train.json")
    prepared_val_data_list_path = data_dir.joinpath("val.json")

    val_ratio = 0.1

    pairs = []
    for csv_path in tqdm(csv_paths):
        with csv_path.open() as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                source = row["원문"]
                target = row["번역문"]
                data_pair = TranslationDataPair(source, target)
                pairs.append(data_pair)

    random.shuffle(pairs)

    data_size = len(pairs)
    val_size = int(data_size * val_ratio)

    train_pairs = pairs[:-val_size]
    val_pairs = pairs[-val_size:]

    train_data_list = TranslationDataList(pairs=train_pairs)
    val_data_list = TranslationDataList(pairs=val_pairs)

    train_data_list.save(prepared_train_data_list_path)
    val_data_list.save(prepared_val_data_list_path)

    print(
        f"Train data {len(train_data_list.pairs)} lines written to {prepared_train_data_list_path}"
    )
    print(f"Val data {len(val_data_list.pairs)} lines written to {prepared_val_data_list_path}")
