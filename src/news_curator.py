import logging

from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """あなたはニュースキュレーターです。
「{topic}」に関する過去24時間以内のニュースを検索し、以下の形式で報告してください。

## 出力形式

### ニュース一覧
各ニュースについて以下の情報を提供してください：
1. **タイトル/プロジェクト名**
2. **プラットフォーム**: (PC, Steam, iOS, Web等、該当する場合)
3. **ステータス**: (発表, リリース, ベータ, 開発中等)
4. **詳細**: (主な特徴、AIの活用方法など)
5. **キーポイント**: (重要な特徴や注目点)
6. **参照元URL**

### トレンド分析
最後に、これらのニュースから見える全体的なトレンドについて、2-3文で分析コメントを追加してください。

## 重要な注意
- 過去24時間以内のニュースのみを対象としてください
- 情報が見つからない場合は正直に「該当するニュースは見つかりませんでした」と報告してください
- 推測や古い情報は含めないでください
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

        return response.text
