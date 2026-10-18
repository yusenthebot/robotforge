"""robotforge: a beginner-friendly robot playground."""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """The `robotforge` command. The command line is imported here, not with the package: every
    `import robotforge.<module>` runs this file first, and the command line brings the viewer's web server,
    the import doctor and the whole playground with it."""
    from robotforge.cli import main as cli_main

    return cli_main(argv)
