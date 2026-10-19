import socket
import threading
import time

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

ARGS = ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


@pytest.fixture(scope="session")
def server(model):
    """The real FastAPI app on a free port, using a random-weight model so no gpt.pt is needed."""
    import uvicorn
    import app as A

    patch = pytest.MonkeyPatch()
    patch.setattr(A, "MODEL", model)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = uvicorn.Server(uvicorn.Config(A.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)
    patch.undo()


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", args=ARGS)  # the Chrome that is already installed
        except Exception:
            try:
                b = p.chromium.launch(args=ARGS)  # or the one from: playwright install chromium
            except Exception as e:
                pytest.skip(f"no browser available: {e}")
        yield b
        b.close()


@pytest.fixture()
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1500, "height": 1000})
    pg = ctx.new_page()
    pg.set_default_timeout(30000)
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and "favicon" not in m.location.get("url", "") and pg.errors.append(m.text))
    pg.base = server
    yield pg
    ctx.close()
