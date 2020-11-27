import typer
from .multi30k import prepare_multi30k
from .aihub import prepare_aihub

app = typer.Typer()
app.command(name="multi30k")(prepare_multi30k)
app.command(name="aihub")(prepare_aihub)
