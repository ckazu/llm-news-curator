import logging
import os
import sys

from dotenv import load_dotenv

from .config import Config
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
        all_success = True

        for topic in config.topics:
            logger.info(f"Processing topic: {topic.name}")

            poster = SlackPoster(config, topic)

            # 過去の投稿からURLを取得して重複を避ける
            logger.info("Fetching recent URLs from Slack...")
            exclude_urls = poster.fetch_recent_urls()

            logger.info("Fetching news with Google Search grounding...")
            items = curator.fetch_news(topic.name, exclude_urls=exclude_urls)
            logger.info(f"Received {len(items)} news items")
            for i, item in enumerate(items):
                logger.debug(f"Item {i + 1}: {item.text[:100]}...")
                logger.debug(f"  Sources: {len(item.sources)}, Impression: {item.is_impression}")

            logger.info("Posting to Slack...")
            success = poster.post_news(items)

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
