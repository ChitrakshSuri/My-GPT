import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

SCENES = {"overview": "The whole GPT", "embed": "Embeddings", "norm": "Layer norm", "attn": "Attention",
          "resid": "Add", "ffn": "Feed forward", "output": "Output"}


def open_page(page, scene="overview"):
    page.goto(f"{page.base}/#{scene}")
    expect(page.locator(".node")).to_have_count(13)   # the flow chart appears once the model has answered


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


def test_message_box_updates_the_flow_chart(page):
    open_page(page)
    page.fill("#prompt", "ROMEO:")
    expect(page.locator('.node[data-part="text"] .tx')).to_contain_text("ROMEO:")


def test_message_box_explains_skipped_and_cropped_text(page):
    open_page(page)
    page.fill("#prompt", "héllo")
    expect(page.locator("#pstat")).to_contain_text("Skipped")
    page.fill("#prompt", "To be or not to be, that is")
    expect(page.locator("#pstat")).to_contain_text("last 12 of 27")
    page.fill("#prompt", "é")
    expect(page.locator("#pstat")).to_contain_text("at least 2")


def test_example_buttons_fill_the_message_box(page):
    open_page(page)
    page.locator('[data-ex="First Citizen:"]').click()
    expect(page.locator("#prompt")).to_have_value("First Citizen:")
    expect(page.locator('.node[data-part="text"] .tx')).to_contain_text("Citizen")


def test_output_scene_can_pick_the_next_letter(page):
    open_page(page, "output")
    before = page.input_value("#prompt")
    page.locator("#sample").click()
    expect(page.locator("#prompt")).not_to_have_value(before)
    assert len(page.input_value("#prompt")) == len(before) + 1


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
    assert page.input_value("#prompt") == "ROMEO:" + reply


def test_error_banner_and_retry_when_the_model_is_unreachable(page):
    open_page(page)
    page.route("**/api/trace", lambda route: route.abort())
    page.fill("#prompt", "Hello there")
    expect(page.locator("#err")).to_have_class("err on")
    expect(page.locator("#err")).to_contain_text("Can")
    page.unroute("**/api/trace")
    page.locator("#retry").click()
    expect(page.locator("#err")).not_to_have_class("err on")


def test_signature_links(page):
    open_page(page)
    sig = page.locator(".sig")
    expect(sig).to_contain_text("Chitraksh Suri")
    expect(sig.get_by_role("link", name="LinkedIn")).to_have_attribute("href", "https://linkedin.com/in/chitrakshsuri")
    expect(sig.get_by_role("link", name="GitHub")).to_have_attribute("href", "https://github.com/ChitrakshSuri")
