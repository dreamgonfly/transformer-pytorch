import typer
from .multi30k import prepare_multi30k

app = typer.Typer()
app.command(name="multi30k")(prepare_multi30k)
