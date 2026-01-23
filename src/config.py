import json
import os
from dataclasses import dataclass


def _parse_bool(value) -> bool:
    """Parse a boolean value from various types."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() == "true"


@dataclass
class TopicConfig:
    """Configuration for a single topic."""

    name: str
    channel_id: str
    header: str
    unfurl_links: bool = False
    unfurl_media: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "TopicConfig":
        """Create TopicConfig from a dictionary."""
        name = data.get("name", "")
        if not name:
            raise ValueError("Topic 'name' is required")

        channel_id = data.get("channel_id", "")
        if not channel_id:
            raise ValueError("Topic 'channel_id' is required")

        header = data.get("header") or f"{name} ニュース"
        unfurl_links = _parse_bool(data.get("unfurl_links", False))
        unfurl_media = _parse_bool(data.get("unfurl_media", False))
        return cls(
            name=name,
            channel_id=channel_id,
            header=header,
            unfurl_links=unfurl_links,
            unfurl_media=unfurl_media,
        )


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Vertex AI settings
    gcp_project_id: str
    gcp_location: str
    model_name: str

    # Slack settings
    slack_bot_token: str

    # Topics
    topics: list[TopicConfig]

    # Display settings
    use_emoji_names: bool

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        topics = cls._load_topics()

        return cls(
            gcp_project_id=os.environ["GCP_PROJECT_ID"],
            gcp_location=os.environ.get("GCP_LOCATION", "asia-northeast1"),
            model_name=os.environ.get("MODEL_NAME", "gemini-2.5-pro"),
            slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
            topics=topics,
            use_emoji_names=_parse_bool(os.environ.get("USE_EMOJI_NAMES", False)),
        )

    @classmethod
    def _load_topics(cls) -> list[TopicConfig]:
        """Load topics from TOPICS_CONFIG environment variable."""
        topics_json = os.environ.get("TOPICS_CONFIG", "")
        if not topics_json:
            raise ValueError("TOPICS_CONFIG environment variable is required")

        try:
            topics_data = json.loads(topics_json)
            if not isinstance(topics_data, list):
                raise ValueError("TOPICS_CONFIG must be a JSON array")
            return [TopicConfig.from_dict(t) for t in topics_data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid TOPICS_CONFIG JSON: {e}")
