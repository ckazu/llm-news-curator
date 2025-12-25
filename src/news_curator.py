import logging

from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """「{topic}」に関する過去24時間以内のニュースを検索し、Slack mrkdwn形式で報告してください。

重要: 前置きや挨拶なしで、ニュース内容のみを直接出力してください。

# 出力形式（以下のフォーマットを厳守）

:newspaper: *1. タイトル名*
> 概要を1-2文で簡潔に説明

:video_game: プラットフォーム: PC / Steam / iOS 等
:pushpin: ステータス: 発表 / リリース / 開発中 等
:bulb: キーポイント: 重要な特徴や注目点を1文で

───────────────────

:newspaper: *2. 次のタイトル名*
（同様のフォーマットで続ける）

───────────────────

:chart_with_upwards_trend: *トレンド分析*
これらのニュースから見える全体的なトレンドを2-3文で分析。

# 注意事項
- URLは含めないこと（参照元は自動追加されます）
- 各ニュースは上記フォーマットで統一し、───で区切る
- 絵文字は指定のものを使用
- Markdown の ## や ** は使わず、Slack mrkdwn の *太字* を使用
- 過去24時間以内のニュースのみ対象
- 情報がない場合は「該当するニュースは見つかりませんでした」と報告
- 検索で見つかった情報は可能な限り個別のニュースとして取り上げること（同じ情報の重複は除く）
- 最低でも3件以上のニュースを報告するよう努めること
"""


class NewsCurator:
    """Curates news using Vertex AI with Google Search grounding."""

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
        grounding_sources = self._extract_grounding_sources(response)
        if grounding_sources:
            logger.debug(f"Grounding sources: {grounding_sources}")

        # 本文に参照元を追加
        text = response.text
        if grounding_sources:
            text += "\n───────────────────\n\n:link: *参照元*\n"
            for source in grounding_sources:
                title = source.get("title", "リンク")
                uri = source.get("uri", "")
                if uri:
                    text += f"• <{uri}|{title}>\n"

        return text

    def _extract_grounding_sources(self, response) -> list[dict]:
        """Extract grounding sources from response metadata."""
        sources = []
        try:
            for candidate in response.candidates:
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, "grounding_chunks"):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, "web") and chunk.web:
                                sources.append({
                                    "title": getattr(chunk.web, "title", ""),
                                    "uri": getattr(chunk.web, "uri", ""),
                                })
        except Exception as e:
            logger.warning(f"Failed to extract grounding sources: {e}")
        return sources
