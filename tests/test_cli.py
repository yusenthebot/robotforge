import socket

import pytest

from robotforge import cli


def _listening_socket(*, reuse: bool) -> socket.socket:
    """A server socket the way uvicorn opens it (it sets SO_REUSEADDR on every OS)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if reuse:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s


@pytest.mark.parametrize("reuse", [True, False], ids=["reuseaddr", "plain"])
def test_port_in_use_is_not_free(reuse):
    with _listening_socket(reuse=reuse) as busy:
        port = busy.getsockname()[1]
        assert not cli.port_is_free(port)
        assert cli.find_free_port(port) != port
    assert cli.port_is_free(port)


def test_demo_refuses_a_busy_port(capsys):
    with _listening_socket(reuse=True) as busy:
        code = cli.main(["demo", "--no-browser", "--port", str(busy.getsockname()[1])])
    assert code == 2
    assert "already in use" in capsys.readouterr().err


def test_open_reports_a_missing_model(tmp_path, capsys):
    assert cli.main(["open", str(tmp_path / "nope.xml"), "--no-browser"]) == 2
    assert "not found" in capsys.readouterr().err
