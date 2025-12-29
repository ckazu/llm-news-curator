import logging

from google import genai
from google.genai import types

from .config import Config, TopicConfig

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """「{topic}」に関する過去24時間以内のニュースを検索し、「ずんだもん」というキャラクターとしてSlack mrkdwn形式で報告してください。

# ずんだもんの設定
- ずんだ餅の妖精
- 一人称は「ボク」
- 必ず語尾に「〜のだ」「〜なのだ」をつける（例:「わかったのだ」「すごいのだ」）
- フレンドリーかつ優しい言葉遣い
- 禁止: 「だよ。」「なのだよ。」「かな？」は使わない。「かな？」の代わりに「のだ？」を使う

# 出力形式（以下のフォーマットを厳守）

:one: *タイトル名*
ずんだもん口調で概要を1-2文で説明するのだ

:memo: ニュースに応じた補足情報1（例: *プラットフォーム:* 、*価格:* 、*発売日:* など）
:memo: ニュースに応じた補足情報2（例: *プラットフォーム:* 、*価格:* 、*発売日:* など）
:bulb: *ここがすごいのだ:* 注目ポイントを1文で

---

:two: *次のタイトル名*
（同様のフォーマットで続ける）

---

:zundamon: *ボクの感想なのだ*
これらのニュースから見えるトレンドについて、ずんだもん口調で2-3文でコメント。

# 注意事項
- 自己紹介や挨拶は含めず、いきなり1件目のニュースから始めること
- URLは含めないこと（参照元は自動追加されます）
- 各ニュースは `---` のみの行で区切る
- Markdown の ## や ** は使わず、Slack mrkdwn の *太字* を使用
- 過去24時間以内のニュースのみ対象
- 情報がない場合は「該当するニュースは見つからなかったのだ」と報告
- ニュースは3〜5件報告すること（最低3件、最大5件）
- すべての説明文でずんだもん口調を維持すること
{exclude_section}"""

EXCLUDE_SECTION_TEMPLATE = """
# 既報のため除外するニュース（以下と同じ内容は報告しないこと）
{titles}
"""


class NewsItem:
    """Represents a single news item with text and sources."""

    def __init__(self, text: str, sources: list[dict], is_impression: bool = False):
        self.text = text.strip()
        self.sources = sources
        self.is_impression = is_impression


class NewsCurator:
    """Curates news using Vertex AI with Google Search grounding."""

    SEPARATOR = "---"
    # 短すぎるセグメントテキストは複数パートに誤マッチする可能性があるため除外
    MIN_SEGMENT_TEXT_LENGTH = 10

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.gcp_location,
        )

    def fetch_news(
        self, topic: str, exclude_titles: list[str] | None = None
    ) -> list[NewsItem]:
        """Fetch news using Google Search grounding.

        Args:
            topic: The topic to search for news.
            exclude_titles: List of news titles to exclude (already reported).

        Returns:
            List of NewsItem objects with text and sources.
        """
        exclude_section = ""
        if exclude_titles:
            titles_text = "\n".join(f"- {title}" for title in exclude_titles)
            exclude_section = EXCLUDE_SECTION_TEMPLATE.format(titles=titles_text)

        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            exclude_section=exclude_section,
        )

        logger.info(f"Fetching news for topic: {topic}")
        logger.info(f"Using model: {self.config.model_name}")

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )

        logger.info("Successfully received response from Vertex AI")
        logger.debug(f"Response candidates: {response.candidates}")

        # grounding metadata から参照元を取得
        chunks, supports = self._extract_grounding_metadata(response)

        # 各ニュース項目を構造化
        items = self._parse_news_items(response.text, chunks, supports)

        return items

    def _parse_news_items(
        self, text: str, chunks: list[dict], supports: list[dict]
    ) -> list[NewsItem]:
        """Parse LLM output into structured NewsItem objects."""
        parts = text.split(self.SEPARATOR)

        if len(parts) <= 1:
            # 区切りがない場合は全体を1つの項目として扱う
            all_sources = self._dedupe_sources(chunks)
            return [NewsItem(text, all_sources)]

        items = []
        for i, part_text in enumerate(parts):
            part_text = part_text.strip()
            if not part_text:
                continue

            # 最後のパートは感想セクション
            is_last_part = i == len(parts) - 1
            is_impression = is_last_part

            # このパートに対応するソースを収集
            sources = self._find_sources_for_part(part_text, chunks, supports)

            # 感想セクションには参照元を追加しない
            if is_impression:
                sources = []

            items.append(NewsItem(part_text, sources, is_impression))

        return items

    def _find_sources_for_part(
        self, part_text: str, chunks: list[dict], supports: list[dict]
    ) -> list[dict]:
        """Find sources that match the given part text."""
        source_indices = set()
        for support in supports:
            segment = support.get("segment", {})
            seg_text = segment.get("text", "")

            # セグメントのテキストがこのパートに含まれるか確認
            if seg_text and len(seg_text) > self.MIN_SEGMENT_TEXT_LENGTH and seg_text in part_text:
                for idx in support.get("chunk_indices", []):
                    if idx < len(chunks):
                        source_indices.add(idx)

        # URIで重複排除
        seen_uris = set()
        sources = []
        for idx in sorted(source_indices):
            chunk = chunks[idx]
            uri = chunk.get("uri", "")
            if not uri or uri in seen_uris:
                continue
            seen_uris.add(uri)
            sources.append(chunk)

        return sources

    def _dedupe_sources(self, chunks: list[dict]) -> list[dict]:
        """Deduplicate sources by URI."""
        seen_uris = set()
        sources = []
        for chunk in chunks:
            uri = chunk.get("uri", "")
            if not uri or uri in seen_uris:
                continue
            seen_uris.add(uri)
            sources.append(chunk)
        return sources

    def _extract_grounding_metadata(self, response) -> tuple[list[dict], list[dict]]:
        """Extract grounding chunks and supports from response metadata."""
        chunks = []
        supports = []
        try:
            for candidate in response.candidates:
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata

                    # Extract grounding chunks
                    if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, "web") and chunk.web:
                                chunks.append({
                                    "title": getattr(chunk.web, "title", ""),
                                    "uri": getattr(chunk.web, "uri", ""),
                                })

                    # Extract grounding supports
                    if hasattr(metadata, "grounding_supports") and metadata.grounding_supports:
                        for support in metadata.grounding_supports:
                            segment = getattr(support, "segment", None)
                            support_data = {
                                "chunk_indices": getattr(support, "grounding_chunk_indices", []),
                                "confidence_scores": getattr(support, "confidence_scores", []),
                            }
                            if segment:
                                support_data["segment"] = {
                                    "start_index": getattr(segment, "start_index", 0),
                                    "end_index": getattr(segment, "end_index", 0),
                                    "text": getattr(segment, "text", ""),
                                }
                            supports.append(support_data)

                    logger.debug(f"Grounding chunks: {chunks}")
                    logger.debug(f"Grounding supports: {supports}")

        except Exception as e:
            logger.warning(f"Failed to extract grounding metadata: {e}")
        return chunks, supports
