import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

SCENES = {"overview": "The whole GPT", "embed": "Embeddings", "norm": "Layer norm", "attn": "Attention",
          "resid": "Add", "ffn": "Feed forward", "output": "Output"}


def open_page(page, scene="overview"):
    page.goto(f"{page.base}/#{scene}")
    expect(page.locator(".node")).to_have_count(13)   # the flow chart appears once the model has answered


def fake_stream(page):
    """Answer chat requests instantly with one letter, so tests that only look at the page stay fast."""
    page.route("**/api/generate/stream", lambda route: route.fulfill(
        status=200, content_type="text/event-stream", body='data: "x"\n\ndata: [DONE]\n\n'))


def say(page, text):
    """Type a message in the chat and send it."""
    page.locator('[data-tab="chat"]').click()
    expect(page.locator("#csend")).to_be_enabled()
    page.fill("#cin", text)
    page.press("#cin", "Enter")


def typed_text(page):
    return page.evaluate("gptApp.state.valid")


def canvas_has_a_drawing(page):
    page.wait_for_timeout(1500)                        # let the camera settle
    return len(page.locator("#gl").screenshot()) > 12000   # a blank white canvas is only a few KB


def test_page_loads_without_errors(page):
    open_page(page)
    expect(page.locator("#info")).to_contain_text("The whole GPT")
    assert canvas_has_a_drawing(page)
    assert page.errors == []


@pytest.mark.parametrize("scene,crumb", SCENES.items())
def test_every_scene_renders(page, scene, crumb):
    open_page(page, scene)
    expect(page.locator("#crumbs")).to_contain_text(crumb)
    assert canvas_has_a_drawing(page)
    assert page.errors == []


def test_hovering_a_flow_chart_box_explains_it(page):
    open_page(page)
    page.locator('.node[data-part="attn"]').hover()
    expect(page.locator("#info")).to_contain_text("query")
    page.locator('.node[data-part="ffn"]').hover()
    expect(page.locator("#info")).to_contain_text("thinking")


def test_clicking_a_box_opens_its_dissection(page):
    open_page(page)
    page.locator('.node[data-part="attn"]').click()
    expect(page.locator("#crumbs")).to_contain_text("Attention")
    assert page.url.endswith("#attn")
    expect(page.locator("[data-head]")).to_have_count(6)
    expect(page.locator("[data-layer]")).to_have_count(6)
    page.locator("[data-home]").click()
    expect(page.locator("#crumbs")).to_contain_text("The whole GPT")


def test_head_and_block_selectors(page):
    open_page(page, "attn")
    expect(page.locator("[data-head]")).to_have_count(6)
    page.locator('[data-head="2"]').click()
    expect(page.locator('[data-head="2"]')).to_have_attribute("aria-pressed", "true")
    with page.expect_request(lambda r: r.url.endswith("/api/trace") and '"layers":[3]' in (r.post_data or "").replace(" ", "")):
        page.locator('[data-layer="3"]').click()
    expect(page.locator('[data-layer="3"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#info")).to_contain_text("block 4")


def test_chat_message_updates_the_visualizer(page):
    fake_stream(page)
    open_page(page)
    say(page, "ROMEO:")
    expect(page.locator('.node[data-part="text"] .tx')).to_contain_text("ROMEO:")


def test_chat_explains_skipped_and_cropped_text(page):
    fake_stream(page)
    open_page(page)
    say(page, "h\u00e9llo")
    expect(page.locator("#pstat")).to_contain_text("Skipped")
    say(page, "To be or not to be, that is")
    expect(page.locator("#pstat")).to_contain_text("last 12 of 27")
    say(page, "\u00e9")
    expect(page.locator("#pstat")).to_contain_text("at least 2")


def test_output_scene_can_pick_the_next_letter(page):
    open_page(page, "output")
    before = typed_text(page)
    page.locator("#sample").click()
    page.wait_for_function("(n) => gptApp.state.valid.length === n + 1", arg=len(before))
    assert typed_text(page).startswith(before)


def test_chat_streams_a_reply_and_visualizes_it(page):
    open_page(page)
    page.locator('[data-tab="chat"]').click()
    expect(page.locator("#chat")).to_contain_text("text completer")
    page.locator('[data-sugg="ROMEO:"]').click()
    expect(page.locator(".bub.me")).to_have_text("ROMEO:")
    page.locator(".bub .tools button").wait_for(timeout=120000)   # appears when the stream is done
    reply = page.locator(".bub.bot .gen").inner_text()
    assert len(reply) == 150
    page.locator(".bub .tools button").click()
    expect(page.locator("#crumbs")).to_contain_text("Attention")
    expect(page.locator("#info")).to_be_visible()
    assert typed_text(page) == "ROMEO:" + reply


def test_error_banner_and_retry_when_the_model_is_unreachable(page):
    open_page(page)
    page.route("**/api/trace", lambda route: route.abort())
    page.evaluate("gptApp.applyText('Hello there')")
    expect(page.locator("#err")).to_have_class("err on")
    expect(page.locator("#err")).to_contain_text("Can")
    page.unroute("**/api/trace")
    page.locator("#retry").click()
    expect(page.locator("#err")).not_to_have_class("err on")
    expect(page.locator('.node[data-part="text"] .tx')).to_contain_text("Hello")


def test_signature_links(page):
    open_page(page)
    sig = page.locator(".sig")
    expect(sig).to_contain_text("Chitraksh Suri")
    expect(sig.get_by_role("link", name="LinkedIn")).to_have_attribute("href", "https://linkedin.com/in/chitrakshsuri")
    expect(sig.get_by_role("link", name="GitHub")).to_have_attribute("href", "https://github.com/ChitrakshSuri")
