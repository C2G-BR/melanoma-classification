import typer

app = typer.Typer(name="Melanoma Classification")


@app.command()
def greet(name: str):
    """Say hello to NAME."""
    typer.echo(f"Hello {name}!")


if __name__ == "__main__":
    app()
