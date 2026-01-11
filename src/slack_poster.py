import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .config import Config, TopicConfig

if TYPE_CHECKING:
    from .news_curator import NewsItem

logger = logging.getLogger(__name__)

MAX_BLOCK_TEXT_LENGTH = 3000
HISTORY_DAYS = 3


class SlackPoster:
    """Posts messages to Slack using Block Kit."""

    def __init__(self, config: Config, topic: TopicConfig):
        self.client = WebClient(token=config.slack_bot_token)
        self.channel_id = topic.channel_id
        self.header = topic.header
        self.model_name = config.model_name

    def post_news(self, items: list["NewsItem"]) -> bool:
        """Post news items to Slack using Block Kit.

        Args:
            items: List of NewsItem objects with text and sources.

        Returns:
            True if successful, False otherwise.
        """
        blocks = self._build_blocks(items)

        try:
            response = self.client.chat_postMessage(
                channel=self.channel_id,
                blocks=blocks,
                text=self.header,
                unfurl_links=False,
                unfurl_media=False,
            )
            logger.info(f"Message posted successfully: {response['ts']}")
            return True
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return False

    def _build_blocks(self, items: list["NewsItem"]) -> list[dict]:
        """Build Slack Block Kit blocks for the message."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y年%-m月%-d日")
        time_str = now.strftime("%H:%M UTC")

        blocks = [
            # ヘッダー
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": self.header,
                    "emoji": True,
                },
            },
            # 日付コンテキスト
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📅 {date_str}",
                    }
                ],
            },
        ]

        # 各ニュース項目をカードとして追加
        for item in items:
            # カード境界の divider
            blocks.append({"type": "divider"})

            # ニュース本文（タイトル + 説明 + 補足情報）
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": item.text,
                    },
                }
            )

            # ソースリンク（感想以外で、ソースがある場合）
            if not item.is_impression and item.sources:
                source_links = self._format_source_links(item.sources)
                if source_links:
                    blocks.append(
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"🔗 {source_links}",
                                }
                            ],
                        }
                    )

        # フッター
        blocks.extend(
            [
                {"type": "divider"},
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"⚡ {self.model_name} · {time_str}",
                        }
                    ],
                },
            ]
        )

        return blocks

    def _format_source_links(self, sources: list[dict]) -> str:
        """Format source links for display."""
        links = []
        for source in sources:
            title = source.get("title", "リンク")
            uri = source.get("uri", "")
            if uri:
                links.append(f"<{uri}|{title}>")
        return " · ".join(links)

    def fetch_recent_titles(self) -> list[str]:
        """Fetch news titles from recent messages in the channel.

        Returns:
            List of news titles from the past HISTORY_DAYS days.
        """
        oldest = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
        oldest_ts = str(oldest.timestamp())

        try:
            response = self.client.conversations_history(
                channel=self.channel_id,
                oldest=oldest_ts,
                limit=100,
            )

            titles = []
            # タイトルパターン: 行頭の *タイトル名* （番号付き/なし両対応）
            # 例: "*タイトル名*" or ":one: *タイトル名*" or "*1. タイトル名*"
            title_pattern = re.compile(r"(?:^|\n)(?::[a-zA-Z0-9_]+:\s*|\*\d+\.\s*)?\*([^*\n]+)\*")

            for message in response.get("messages", []):
                blocks = message.get("blocks", [])
                for block in blocks:
                    if block.get("type") == "section":
                        text = block.get("text", {}).get("text", "")
                        matches = title_pattern.findall(text)
                        titles.extend(matches)

            logger.info(f"Found {len(titles)} titles from past {HISTORY_DAYS} days")
            logger.debug(f"Recent titles: {titles}")
            return titles

        except SlackApiError as e:
            logger.warning(f"Failed to fetch conversation history: {e.response['error']}")
            return []
