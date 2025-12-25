import logging

from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """「{topic}」に関する過去24時間以内のニュースを検索し、「ずんだもん」というキャラクターとしてSlack mrkdwn形式で報告してください。

# ずんだもんの設定
- ずんだ餅の妖精
- 一人称は「ボク」
- 必ず語尾に「〜のだ」「〜なのだ」をつける（例:「わかったのだ」「すごいのだ」）
- フレンドリーかつ優しい言葉遣い
- 禁止: 「だよ。」「なのだよ。」「かな？」は使わない。「かな？」の代わりに「のだ？」を使う

# 出力形式（以下のフォーマットを厳守）

:zap: *1. タイトル名*
> ずんだもん口調で概要を1-2文で説明するのだ

:video_game: プラットフォーム: PC / Steam / iOS 等
:pushpin: ステータス: 発表 / リリース / 開発中 等
:bulb: ここがすごいのだ: 重要な特徴や注目点を1文で

───────────────────

:zap: *2. 次のタイトル名*
（同様のフォーマットで続ける）

───────────────────

:speech_balloon: *ボクの感想なのだ*
これらのニュースから見えるトレンドについて、ずんだもん口調で2-3文でコメント。

# 注意事項
- URLは含めないこと（参照元は自動追加されます）
- 各ニュースは上記フォーマットで統一し、───で区切る
- Markdown の ## や ** は使わず、Slack mrkdwn の *太字* を使用
- 過去24時間以内のニュースのみ対象
- 情報がない場合は「該当するニュースは見つからなかったのだ」と報告
- 検索で見つかった情報は可能な限り個別のニュースとして取り上げること（同じ情報の重複は除く）
- 最低でも3件以上のニュースを報告するよう努めること
- すべての説明文でずんだもん口調を維持すること
"""


class NewsCurator:
    """Curates news using Vertex AI with Google Search grounding."""

    SEPARATOR = "───────────────────"
    # 短すぎるセグメントテキストは複数パートに誤マッチする可能性があるため除外
    MIN_SEGMENT_TEXT_LENGTH = 10

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.gcp_location,
        )

    def fetch_news(self) -> str:
        """Fetch news using Google Search grounding.

        Returns:
            The response text from the LLM.
        """
        prompt = PROMPT_TEMPLATE.format(topic=self.config.curator_topic)

        logger.info(f"Fetching news for topic: {self.config.curator_topic}")
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

        # 各ニュース項目に参照元を挿入
        text = self._insert_sources_per_item(response.text, chunks, supports)

        return text

    def _insert_sources_per_item(
        self, text: str, chunks: list[dict], supports: list[dict]
    ) -> str:
        """Insert source links at the end of each news item based on grounding supports."""
        if not chunks or not supports:
            logger.debug("No chunks or supports available, returning original text")
            return text

        parts = text.split(self.SEPARATOR)

        if len(parts) <= 1:
            # 区切り線がない場合は最後に全ソースを追加
            return self._append_all_sources(text, chunks)

        # 各パートに対応するソースを特定（テキストマッチング方式）
        result_parts = []
        for i, part_text in enumerate(parts):
            # 最後のパート（感想セクション）には参照元を追加しない
            is_last_part = i == len(parts) - 1
            if is_last_part:
                result_parts.append(part_text)
                continue

            # このパートに対応するソースを収集
            source_indices = set()
            for support in supports:
                segment = support.get("segment", {})
                seg_text = segment.get("text", "")

                # セグメントのテキストがこのパートに含まれるか確認
                if seg_text and len(seg_text) > self.MIN_SEGMENT_TEXT_LENGTH and seg_text in part_text:
                    for idx in support.get("chunk_indices", []):
                        if idx < len(chunks):
                            source_indices.add(idx)

            # ソースリンクを追加（URIで重複排除）
            if source_indices:
                seen_uris = set()
                source_links = []
                for idx in sorted(source_indices):
                    chunk = chunks[idx]
                    uri = chunk.get("uri", "")
                    if not uri or uri in seen_uris:
                        continue
                    seen_uris.add(uri)
                    link = self._format_source_link(chunk)
                    if link:
                        source_links.append(link)
                if source_links:
                    sources_text = "\n:link: 参照元: " + " | ".join(source_links)
                    part_text = part_text.rstrip() + sources_text + "\n"

            result_parts.append(part_text)

        return self.SEPARATOR.join(result_parts)

    def _format_source_link(self, chunk: dict) -> str:
        """Format a single source link for Slack mrkdwn."""
        title = chunk.get("title")
        uri = chunk.get("uri")
        if not uri:
            if title:
                logger.warning(
                    "Grounding chunk has title but no URI; omitting source link: %s",
                    chunk,
                )
            return ""
        return f"<{uri}|{title or 'リンク'}>"

    def _append_all_sources(self, text: str, chunks: list[dict]) -> str:
        """Append all sources at the end of the text."""
        text += f"\n{self.SEPARATOR}\n\n:link: *参照元*\n"
        for chunk in chunks:
            link = self._format_source_link(chunk)
            if link:
                text += f"• {link}\n"
        return text

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
