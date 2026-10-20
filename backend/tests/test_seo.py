import json
import re

SITE = "https://gpt-anatomy.onrender.com/"


def page(client):
    return client.get("/").text


def meta(html, attr, name):
    m = re.search(rf'<meta {attr}="{re.escape(name)}" content="([^"]*)"', html)
    return m.group(1) if m else None


def test_title_and_description_are_the_right_length_and_use_the_keywords(client):
    html = page(client)
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    description = meta(html, "name", "description")
    assert "Transformer Visualization" in title and len(title) <= 65
    assert "transformer visualization" in description.lower() and len(description) <= 160


def test_the_page_has_one_h1_and_a_canonical_link(client):
    html = page(client)
    assert html.count("<h1") == 1
    assert f'<link rel="canonical" href="{SITE}">' in html
    assert '<html lang="en">' in html


def test_link_previews_use_an_absolute_image_that_exists(client):
    html = page(client)
    assert meta(html, "property", "og:image") == SITE + "social.png"
    assert meta(html, "name", "twitter:image") == SITE + "social.png"
    assert meta(html, "name", "twitter:card") == "summary_large_image"
    assert meta(html, "property", "og:title") and meta(html, "property", "og:description")
    image = client.get("/social.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_structured_data_is_valid_json(client):
    html = page(client)
    data = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1))
    assert data["@type"] == "WebApplication" and data["url"] == SITE
    assert data["author"]["name"] == "Chitraksh Suri"


def test_readable_text_for_search_engines_is_in_the_html_itself(client):
    html = page(client)   # not drawn by JavaScript
    assert 'id="about"' in html
    assert "transformer visualization" in html.lower() and "self-attention" in html.lower()
    assert "<h2" in html and "<h3" in html


def test_robots_txt_blocks_the_api_and_points_to_the_sitemap(client):
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "Disallow: /api/" in robots.text and "Disallow: /docs" in robots.text
    assert f"Sitemap: {SITE}sitemap.xml" in robots.text


def test_sitemap_lists_the_home_page(client):
    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200 and f"<loc>{SITE}</loc>" in sitemap.text


def test_favicon_is_served(client):
    assert client.get("/favicon.svg").status_code == 200
