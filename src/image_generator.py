import base64
import logging

from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

MANGA_STORY_PROMPT = """以下のニュースまとめを元に、ずんだもんとあんこもんが登場する4コマ漫画のストーリーを作成してください。

# ニュースまとめ
{summary}

# キャラクター
- ずんだもん: ずんだ餅の妖精。ポジティブで楽観的。緑色。
- あんこもん: あんこ餅の妖精。現実的で慎重派だが根は優しい。茶色。ずんだもんのライバル。

# 出力形式（厳守）
1コマ目: [シーン説明]
2コマ目: [シーン説明]
3コマ目: [シーン説明]
4コマ目: [シーン説明（オチ）]

# ルール
- 各コマは1行で簡潔に（30文字以内）
- ずんだもんとあんこもんの掛け合いを含める
- 最後のコマはオチや意外な展開で締める
- ニュースの内容を反映しつつ、ユーモラスに
- キャラクターの動作や表情も含める
"""

# Nano Banana ベストプラクティス: 物語的描写で詳細に指定
IMAGE_PROMPT_TEMPLATE = """Create a cute 4-panel vertical comic strip in Japanese yonkoma manga style.

The comic features two adorable round mochi fairy characters having a conversation about today's tech news:

Panel 1 (Top):
{panel1}

Panel 2:
{panel2}

Panel 3:
{panel3}

Panel 4 (Bottom - Punchline):
{panel4}

Character Descriptions:
- Zundamon (Green Mochi Fairy): A cheerful, round green mochi character with big sparkling eyes, small leaf-like decoration on head, always smiling and optimistic. Speaks with enthusiasm.
- Ankomon (Brown Mochi Fairy): A slightly grumpy but secretly caring round brown/red mochi character with smaller eyes and a subtle frown that sometimes turns into a hidden smile. Has a small red bean decoration.

Art Style Requirements:
- Soft, hand-drawn doodle aesthetic with gentle lines
- Pastel color palette (soft greens, warm browns, cream backgrounds)
- Chibi/kawaii Japanese illustration style
- Simple backgrounds with soft gradients
- Clear panel borders arranged vertically
- Expressive facial reactions (sparkly eyes, sweat drops, blush marks)
- Speech bubbles with simple text indicators
- Warm, friendly atmosphere

The overall mood should be heartwarming and slightly comedic, like a slice-of-life manga about two fairy friends discussing the day's events.
"""


class ImageGenerator:
    """Generates 4-panel manga illustrations using Gemini Nano Banana."""

    def __init__(self, config: Config):
        self.config = config
        self.image_model = config.image_model_name
        self.image_location = config.image_gcp_location
        # テキスト生成用クライアント (ストーリー生成)
        self.text_client = genai.Client(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.gcp_location,
        )
        # 画像生成用クライアント (リージョンが異なる場合がある)
        self.image_client = genai.Client(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.image_gcp_location,
        )

    def generate_manga_story(self, summary_text: str) -> dict | None:
        """Generate a 4-panel manga story from news summary.

        Args:
            summary_text: The news summary text (まとめ section).

        Returns:
            Dictionary with panel descriptions or None if generation fails.
        """
        prompt = MANGA_STORY_PROMPT.format(summary=summary_text)

        try:
            response = self.text_client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,  # More creative for manga
                ),
            )
            story_text = response.text.strip()
            logger.info("Generated 4-panel manga story")
            logger.debug(f"Story: {story_text}")

            # Parse the story into panels
            panels = self._parse_story_panels(story_text)
            return panels

        except Exception as e:
            logger.error(f"Failed to generate manga story: {e}")
            return None

    def _parse_story_panels(self, story_text: str) -> dict:
        """Parse story text into panel descriptions."""
        panels = {
            "panel1": "",
            "panel2": "",
            "panel3": "",
            "panel4": "",
        }

        lines = story_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("1コマ目:") or line.startswith("1コマ目："):
                panels["panel1"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            elif line.startswith("2コマ目:") or line.startswith("2コマ目："):
                panels["panel2"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            elif line.startswith("3コマ目:") or line.startswith("3コマ目："):
                panels["panel3"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            elif line.startswith("4コマ目:") or line.startswith("4コマ目："):
                panels["panel4"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()

        # Fallback if parsing failed
        if not any(panels.values()):
            panels = {
                "panel1": "Zundamon excitedly shares today's tech news",
                "panel2": "Ankomon responds with skeptical expression",
                "panel3": "They debate the implications together",
                "panel4": "Both smile, agreeing to watch how things develop",
            }

        return panels

    def generate_manga_image(self, panels: dict) -> bytes | None:
        """Generate a 4-panel manga image using Nano Banana.

        Args:
            panels: Dictionary with panel1-4 descriptions.

        Returns:
            PNG image bytes or None if generation fails.
        """
        logger.info(f"Using image model: {self.image_model} (location: {self.image_location})")
        prompt = IMAGE_PROMPT_TEMPLATE.format(
            panel1=panels.get("panel1", ""),
            panel2=panels.get("panel2", ""),
            panel3=panels.get("panel3", ""),
            panel4=panels.get("panel4", ""),
        )

        try:
            # Nano Banana: generate_content with IMAGE modality
            response = self.image_client.models.generate_content(
                model=self.image_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            # Extract image from response
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_data = part.inline_data.data
                        # Handle both base64 string and bytes
                        if isinstance(image_data, str):
                            image_bytes = base64.b64decode(image_data)
                        else:
                            image_bytes = image_data
                        logger.info("Generated 4-panel manga image with Nano Banana")
                        return image_bytes

            logger.warning("No image in response")
            return None

        except Exception as e:
            logger.error(f"Failed to generate manga image: {e}")
            return None

    def generate_manga(self, summary_text: str) -> bytes | None:
        """Generate a complete 4-panel manga from news summary.

        Args:
            summary_text: The news summary text (まとめ section).

        Returns:
            PNG image bytes or None if generation fails.
        """
        # Step 1: Generate story panels
        panels = self.generate_manga_story(summary_text)
        if not panels:
            return None

        # Step 2: Generate image with Nano Banana
        return self.generate_manga_image(panels)
