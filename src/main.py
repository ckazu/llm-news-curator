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
        logger.info(f"Topic: {config.curator_topic}")

        curator = NewsCurator(config)
        logger.info("Fetching news with Google Search grounding...")
        items = curator.fetch_news()
        logger.info(f"Received {len(items)} news items")
        for i, item in enumerate(items):
            logger.debug(f"Item {i + 1}: {item.text[:100]}...")
            logger.debug(f"  Sources: {len(item.sources)}, Impression: {item.is_impression}")

        poster = SlackPoster(config)
        logger.info("Posting to Slack...")
        success = poster.post_news(items)

        if success:
            logger.info("News posted successfully")
            return 0
        else:
            logger.error("Failed to post news to Slack")
            return 1

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
