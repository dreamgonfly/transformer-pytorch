"""transformer-pytorch: A clean implementation of Transformer"""

import typer

from transformer.prepare.cli import app as prepare_app
from transformer.token_indexers.build_vocab import build_vocab
from transformer.train import train

app = typer.Typer()

v1_app = typer.Typer()
app.add_typer(v1_app, name="v1")

v1_app.add_typer(prepare_app, name="prepare")
v1_app.command()(build_vocab)
v1_app.command()(train)


if __name__ == "__main__":
    app()
