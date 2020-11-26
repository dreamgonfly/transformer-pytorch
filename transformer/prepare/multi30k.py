from pathlib import Path

from typer import Argument

from transformer.data_models.translation_data import TranslationDataPair, TranslationDataList


def prepare_multi30k(
    data_dir: Path = Argument(..., help="Path to root data directory. (e.g. data/Multi30k).")
):
    train_source_path = data_dir.joinpath("training/train.de")
    train_target_path = data_dir.joinpath("training/train.en")
    val_source_path = data_dir.joinpath("validation/val.de")
    val_target_path = data_dir.joinpath("validation/val.en")
    eval_source_path = data_dir.joinpath("mmt16_task1_test/test.de")
    eval_target_path = data_dir.joinpath("mmt16_task1_test/test.en")

    prepared_train_data_list_path = data_dir.joinpath("train.json")
    prepared_val_data_list_path = data_dir.joinpath("val.json")
    prepared_eval_data_list_path = data_dir.joinpath("eval.json")

    train_data_list = TranslationDataList(pairs=[])
    with train_source_path.open() as train_source:
        with train_target_path.open() as train_target:
            for source_line, target_line in zip(train_source.readlines(), train_target.readlines()):
                if not source_line.strip() or not target_line.strip():
                    continue
                data_pair = TranslationDataPair(source_line.strip(), target_line.strip())
                train_data_list.pairs.append(data_pair)

    val_data_list = TranslationDataList(pairs=[])
    with val_source_path.open() as val_source:
        with val_target_path.open() as val_target:
            for source_line, target_line in zip(val_source.readlines(), val_target.readlines()):
                if not source_line.strip() or not target_line.strip():
                    continue
                data_pair = TranslationDataPair(source_line.strip(), target_line.strip())
                val_data_list.pairs.append(data_pair)

    eval_data_list = TranslationDataList(pairs=[])
    with eval_source_path.open() as eval_source:
        with eval_target_path.open() as eval_target:
            for source_line, target_line in zip(eval_source.readlines(), eval_target.readlines()):
                if not source_line.strip() or not target_line.strip():
                    continue
                data_pair = TranslationDataPair(source_line.strip(), target_line.strip())
                eval_data_list.pairs.append(data_pair)

    train_data_list.save(prepared_train_data_list_path)
    val_data_list.save(prepared_val_data_list_path)
    eval_data_list.save(prepared_eval_data_list_path)

    print(
        f"Train data {len(train_data_list.pairs)} lines written to {prepared_train_data_list_path}"
    )
    print(f"Val data {len(val_data_list.pairs)} lines written to {prepared_val_data_list_path}")
    print(f"Eval data {len(eval_data_list.pairs)} lines written to {prepared_eval_data_list_path}")
