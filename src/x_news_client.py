import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

import httpx

logger = logging.getLogger(__name__)

X_API_BASE = "https://api.x.com/2"
MAX_PAGE_TEXT_LENGTH = 1000


class _TextExtractor(HTMLParser):
    """Minimal HTML text extractor."""

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "nav", "footer", "header"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._text.append(text)

    def get_text(self) -> str:
        return " ".join(self._text)


def _extract_text_from_html(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return ""
    return parser.get_text()


def _fetch_page_text(url: str) -> str:
    """Fetch a URL and extract main text content."""
    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; NewsCuratorBot/1.0)"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return ""
            text = _extract_text_from_html(resp.text)
            return text[:MAX_PAGE_TEXT_LENGTH]
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return ""


class XNewsStory:
    """Represents a news story derived from an X tweet."""

    def __init__(self, tweet_id: str, text: str, urls: list[str], author: str = ""):
        self.tweet_id = tweet_id
        self.text = text
        self.urls = urls
        self.author = author
        self.page_texts: dict[str, str] = {}  # url -> extracted text

    @property
    def x_url(self) -> str:
        return f"https://x.com/i/web/status/{self.tweet_id}"

    @property
    def sources(self) -> list[dict]:
        return [{"title": "X", "uri": self.x_url}]

    def to_prompt_text(self, index: int) -> str:
        """Format story for inclusion in LLM prompt with index number."""
        author_tag = f" (@{self.author})" if self.author else ""
        lines = [f"[{index}]{author_tag} {self.text}"]
        for url, page_text in self.page_texts.items():
            if page_text:
                lines.append(f"  リンク先 ({url}): {page_text}")
        return "\n".join(lines)

    def fetch_linked_pages(self):
        """Fetch text content from linked URLs."""
        for url in self.urls:
            text = _fetch_page_text(url)
            if text:
                self.page_texts[url] = text


class XNewsClient:
    """Fetches recent tweets from X API v2 to use as news sources."""

    MIN_LIKES = 10  # public_metrics のいいね数フィルタ閾値

    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.headers = {"Authorization": f"Bearer {bearer_token}"}

    def search(self, query: str, max_results: int = 20, min_likes: int | None = None) -> list[XNewsStory]:
        """Search tweets matching a query using full-archive search.

        Uses `/2/tweets/search/all` with `sort_order=relevancy` to get
        high-engagement tweets. Results are filtered by `min_likes`.

        Args:
            query: Search query string (topic keywords).
            max_results: Number of tweets to request from API (10-100).
            min_likes: Minimum like count to include. Defaults to MIN_LIKES.

        Returns:
            List of XNewsStory objects with linked page content fetched.
        """
        min_likes = min_likes if min_likes is not None else self.MIN_LIKES
        full_query = f"({query}) -is:retweet"
        params = {
            "query": full_query,
            "max_results": max(10, min(max_results, 100)),
            "sort_order": "relevancy",
            "tweet.fields": "text,entities,created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username",
        }

        logger.info(f"Searching X tweets (full-archive): query={full_query!r}")

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"{X_API_BASE}/tweets/search/all",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (401, 403):
                raise RuntimeError(
                    f"X API authentication failed (HTTP {status}). "
                    "Ensure X_BEARER_TOKEN is valid and the App belongs to a Project "
                    "(X API v2 requires Apps within Projects). "
                    "See: https://developer.x.com/"
                ) from e
            logger.error(f"X API error: {status} {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"X API request failed: {e}")
            raise

        tweets = data.get("data", [])
        logger.info(f"X API returned {len(tweets)} tweets")

        # author_id → username のマッピングを構築
        users = {u["id"]: u["username"] for u in (data.get("includes") or {}).get("users", [])}

        # いいね数でフィルタ
        if min_likes > 0:
            filtered = [t for t in tweets if (t.get("public_metrics") or {}).get("like_count", 0) >= min_likes]
            logger.info(f"Filtered to {len(filtered)} tweets with >= {min_likes} likes (from {len(tweets)})")
            tweets = filtered

        # 同一著者の投稿を最大2件に制限（多様性確保）
        max_per_author = 2
        author_counts: dict[str, int] = {}
        diverse_tweets = []
        for t in tweets:
            aid = t.get("author_id", "")
            author_counts[aid] = author_counts.get(aid, 0) + 1
            if author_counts[aid] <= max_per_author:
                diverse_tweets.append(t)
        if len(diverse_tweets) < len(tweets):
            logger.info(f"Deduplicated to {len(diverse_tweets)} tweets (max {max_per_author}/author)")
        tweets = diverse_tweets

        stories = [self._tweet_to_story(t, users) for t in tweets]

        # リンク先ページを並列取得
        self._fetch_all_pages(stories)

        return stories

    def _tweet_to_story(self, tweet: dict, users: dict[str, str]) -> XNewsStory:
        """Convert a tweet dict to XNewsStory."""
        text = tweet.get("text", "")
        author_id = tweet.get("author_id", "")
        username = users.get(author_id, "")
        urls = []
        for url_entity in (tweet.get("entities") or {}).get("urls", []):
            expanded = url_entity.get("expanded_url", "")
            if not expanded:
                continue
            if any(domain in expanded for domain in ("twitter.com", "x.com", "t.co")):
                continue
            urls.append(expanded)
        return XNewsStory(
            tweet_id=tweet.get("id", ""),
            text=text,
            urls=urls,
            author=username,
        )

    def _fetch_all_pages(self, stories: list[XNewsStory]):
        """Fetch linked page content for all stories in parallel."""
        stories_with_urls = [s for s in stories if s.urls]
        if not stories_with_urls:
            return

        logger.info(f"Fetching linked pages for {len(stories_with_urls)} tweets...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(s.fetch_linked_pages): s for s in stories_with_urls}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"Page fetch error: {e}")

        fetched = sum(1 for s in stories if s.page_texts)
        logger.info(f"Successfully fetched page content for {fetched} tweets")
