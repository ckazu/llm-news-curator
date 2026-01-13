import logging
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

from .config import Config, TopicConfig

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """本日は {current_date} です。「{topic}」に関する過去24時間以内（{date_range_start} 以降）のニュースを検索し、「ずんだもん」と「あんこもん」の2人が議論する形式でSlack mrkdwn形式で報告してください。

# ずんだもんの設定
- ずんだ餅の妖精
- 一人称は「ボク」
- 語尾は「〜のだ」「〜なのだ」（例:「すごいのだ」「楽しみなのだ」）
- ポジティブ・期待・ワクワクする視点でコメント
- フレンドリーで優しい性格
- 禁止: 「だよ。」「なのだよ。」「かな？」は使わない

# あんこもんの設定
- あんこ餅の妖精（ずんだもんのライバル）
- 一人称は「あんこもん」（自分のことを名前で呼ぶ。例:「あんこもんは知ってるもん」）
- 語尾は「〜もん」（動詞・形容詞の後）または「〜だもん」（名詞の後）
- 例:「知らないもん」「そうだもん」「あんこもんの方が詳しいもん」
- 現実的・慎重な視点でコメント（建設的な批判）
- ツンデレで負けず嫌いだが、良いものは素直に認めることもある
- ずんだもんに対抗意識を持ちつつも、最終的にはフォローすることもある
- 禁止: 全否定や攻撃的な表現（「〜なんてない」「粗悪」「価値がない」「おもちゃ」など）
- 推奨: 「〜には注意が必要だもん」「〜は慎重に見た方がいいもん」「〜という懸念もあるもん」

# 出力形式（以下のフォーマットを厳守）

*1件目のタイトル名*（ずんだもん始まり）

{zundamon}: ニュースを紹介しつつ、ワクワクポイントを語るのだ！
{ankomon}: ツッコミや反論を入れるだもん。
{zundamon}: あんこもんの指摘に対して、さらにポジティブな反論や補足をするのだ！
{ankomon}: 最後に一言、皮肉や現実的なコメントで締めるだもん。

---

*2件目のタイトル名*（あんこもん始まり）

{ankomon}: ニュースを紹介しつつ、現実的な視点でコメントするだもん。
{zundamon}: ポジティブな補足や期待を語るのだ！
{ankomon}: ずんだもんの楽観に対してツッコミを入れるだもん。
{zundamon}: 最後はポジティブに締めくくるのだ！

---

*3件目以降*
（奇数件目はずんだもん始まり、偶数件目はあんこもん始まりで交互に続ける）

---

💭 *まとめ*
{zundamon}: 今日のニュースの総括を楽しげに語るのだ！
{ankomon}: ずんだもんに対抗して、クールに締めるだもん。
{zundamon}: 最後はポジティブに締めくくるのだ！

# 注意事項
- 自己紹介や挨拶は含めず、いきなり1件目のニュースから始めること
- URLは含めないこと（参照元は自動追加されます）
- 各ニュースは `---` のみの行で区切る
- Markdown の ## や ** は使わず、Slack mrkdwn の *太字* を使用
- 過去24時間以内のニュースのみ対象
- 情報がない場合は「該当するニュースは見つからなかったのだ」と報告
- ニュースは3〜5件報告すること（最低3件、最大5件）
- ずんだもんとあんこもんの口調を厳守すること
- 2人の意見は対照的になるようにすること（ポジティブ vs 現実的）
- 会話の順序を交互にすること（奇数件目はずんだもん始まり、偶数件目はあんこもん始まり）
{exclude_section}"""

EXCLUDE_SECTION_TEMPLATE = """
# 既報のため除外するニュース（以下と同一のURLの記事は報告しないこと）
{urls}
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
        self, topic: str, exclude_urls: list[str] | None = None
    ) -> list[NewsItem]:
        """Fetch news using Google Search grounding.

        Args:
            topic: The topic to search for news.
            exclude_urls: List of news URLs to exclude (already reported).

        Returns:
            List of NewsItem objects with text and sources.
        """
        exclude_section = ""
        if exclude_urls:
            urls_text = "\n".join(f"- {url}" for url in exclude_urls)
            exclude_section = EXCLUDE_SECTION_TEMPLATE.format(urls=urls_text)

        # キャラクター名の表示形式を設定に基づいて決定
        if self.config.use_emoji_names:
            zundamon = ":zundamon:"
            ankomon = ":ankomon:"
        else:
            zundamon = "ずんだもん"
            ankomon = "あんこもん"

        # 現在の日時と24時間前の日時を計算
        now = datetime.now(timezone.utc)
        date_range_start = now - timedelta(hours=24)
        current_date = now.strftime("%Y年%m月%d日 %H:%M UTC")
        date_range_start_str = date_range_start.strftime("%Y年%m月%d日 %H:%M UTC")

        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            exclude_section=exclude_section,
            zundamon=zundamon,
            ankomon=ankomon,
            current_date=current_date,
            date_range_start=date_range_start_str,
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

        # grounding metadata から参照元を取得（期間内のソースのみ）
        chunks, supports = self._extract_grounding_metadata(response, cutoff_time=date_range_start)

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

    def _extract_grounding_metadata(
        self, response, cutoff_time: datetime | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Extract grounding chunks and supports from response metadata.

        Args:
            response: The API response containing grounding metadata.
            cutoff_time: Optional datetime to filter out old sources.

        Returns:
            Tuple of (chunks, supports) lists.
        """
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
                                chunk_data = {
                                    "title": getattr(chunk.web, "title", ""),
                                    "uri": getattr(chunk.web, "uri", ""),
                                }
                                # Try to extract date if available
                                if hasattr(chunk.web, "date"):
                                    chunk_data["date"] = getattr(chunk.web, "date", None)
                                if hasattr(chunk.web, "published_date"):
                                    chunk_data["date"] = getattr(chunk.web, "published_date", None)

                                # Filter by cutoff_time if date is available
                                if cutoff_time and chunk_data.get("date"):
                                    try:
                                        chunk_date = chunk_data["date"]
                                        if isinstance(chunk_date, str):
                                            # Try to parse common date formats
                                            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                                                try:
                                                    chunk_date = datetime.strptime(chunk_date, fmt)
                                                    if chunk_date.tzinfo is None:
                                                        chunk_date = chunk_date.replace(tzinfo=timezone.utc)
                                                    break
                                                except ValueError:
                                                    continue
                                        if isinstance(chunk_date, datetime) and chunk_date < cutoff_time:
                                            logger.debug(f"Filtering old source: {chunk_data['uri']} (date: {chunk_date})")
                                            continue
                                    except Exception as e:
                                        logger.debug(f"Could not parse date for filtering: {e}")

                                chunks.append(chunk_data)

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
