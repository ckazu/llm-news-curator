import logging
import os
import sys

from dotenv import load_dotenv

from .config import Config
from .image_generator import ImageGenerator
from .news_curator import NewsCurator
from .slack_poster import SlackPoster

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for the news curator."""
    load_dotenv()

    try:
        config = Config.from_env()
        logger.info("Configuration loaded successfully")
        logger.info(f"Found {len(config.topics)} topic(s) to process")

        curator = NewsCurator(config)
        image_generator = ImageGenerator(config) if config.generate_manga else None
        all_success = True

        for topic in config.topics:
            logger.info(f"Processing topic: {topic.name}")

            poster = SlackPoster(config, topic)

            # 過去の投稿からタイトルを取得して重複を避ける
            logger.info("Fetching recent titles from Slack...")
            exclude_titles = poster.fetch_recent_titles()

            logger.info("Fetching news with Google Search grounding...")
            items = curator.fetch_news(topic.name, exclude_titles=exclude_titles)
            logger.info(f"Received {len(items)} news items")
            for i, item in enumerate(items):
                logger.debug(f"Item {i + 1}: {item.text[:100]}...")
                logger.debug(f"  Sources: {len(item.sources)}, Impression: {item.is_impression}")

            # Generate 4-panel manga if enabled
            manga_image = None
            if image_generator and items:
                # Find the summary section (last item with is_impression=True)
                summary_item = next(
                    (item for item in reversed(items) if item.is_impression), None
                )
                if summary_item:
                    logger.info("Generating 4-panel manga...")
                    manga_image = image_generator.generate_manga(summary_item.text)
                    if manga_image:
                        logger.info("Manga generated successfully")
                    else:
                        logger.warning("Failed to generate manga, continuing without it")

            logger.info("Posting to Slack...")
            success = poster.post_news(items, manga_image=manga_image)

            if success:
                logger.info(f"News posted successfully for topic: {topic.name}")
            else:
                logger.error(f"Failed to post news for topic: {topic.name}")
                all_success = False

        return 0 if all_success else 1

    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
