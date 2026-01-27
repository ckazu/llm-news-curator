import logging

from google import genai
from google.genai import types

from .config import Config, TopicConfig

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """「{topic}」に関する過去24時間以内のニュースを検索し、「ずんだもん」「あんこもん」「四国めたん」「東北きりたん」の4人が議論する形式でSlack mrkdwn形式で報告してください。

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

# 四国めたんの設定
- 良家のお嬢様
- 一人称は「私」または「わたくし」
- タメ口基調だが「〜でしょう」などの言い回し
- 語尾: 「〜かしら。」「〜わね。」「〜わよ。」「〜なのよ。」
- たまに厨二病的な発言（「闇の力」「運命」など）
- 視点: 知的好奇心旺盛、技術的な深掘りや背景を探る

# 東北きりたんの設定
- 11歳の女性、しっかり者
- 一人称は「私」
- 丁寧な言葉遣い（「〜です」「〜ます」「〜ですね」「〜でしょうか」）
- 視点: 冷静で客観的、事実を整理し要点をまとめる役割

# 出力形式（以下のフォーマットを厳守）
- 各ニュースは2〜3人の組み合わせで会話（組み合わせは自由に変えてよい）
- 各キャラクターの発言は一文のみ（短く簡潔に）
- テンポの良い掛け合いを重視
- 1ニュースあたり5〜6発言程度（トピックを少し深掘りする）

*ニュースタイトル*

{zundamon} or {ankomon} or {metan} or {kiritan}: （ニュース紹介一言）
別のキャラ: （反応・感想一言）
別のキャラ: （背景や理由を深掘り一言）
別のキャラ: （別の視点や影響を補足一言）
別のキャラ: （さらに掘り下げ一言）
別のキャラ: （締め一言）

---

（3〜5件のニュースを上記形式で続ける）

---

💭 *まとめ*
{metan}: （技術展望一言）
{ankomon}: （現実的視点一言）
{kiritan}: （注目点一言）
{zundamon}: （ポジティブに締め一言）

# 注意事項
- 自己紹介や挨拶は含めず、いきなり1件目のニュースから始めること
- URLは含めないこと（参照元は自動追加されます）
- 各ニュースは `---` のみの行で区切る
- Markdown の ## や ** は使わず、Slack mrkdwn の *太字* を使用
- 過去24時間以内のニュースのみ対象
- 情報がない場合は「該当するニュースは見つからなかったのだ」と報告
- ニュースは3〜5件報告すること（最低3件、最大5件）
- 4人のキャラクターの口調を厳守すること
- 各キャラクターの視点を活かすこと（ずんだもん: ポジティブ、あんこもん: 現実的、めたん: 技術的深掘り、きりたん: 客観的整理）
- 各ニュースは2〜3人の組み合わせで会話すること（組み合わせは自由、まとめのみ4人全員）
- 各キャラクターの発言は必ず一文のみにすること（長文禁止、テンポ重視）
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
            metan = ":shikoku-metan:"
            kiritan = ":tohoku-kiritan:"
        else:
            zundamon = "ずんだもん"
            ankomon = "あんこもん"
            metan = "四国めたん"
            kiritan = "東北きりたん"

        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            exclude_section=exclude_section,
            zundamon=zundamon,
            ankomon=ankomon,
            metan=metan,
            kiritan=kiritan,
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
