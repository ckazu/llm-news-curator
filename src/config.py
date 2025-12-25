import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Vertex AI settings
    gcp_project_id: str
    gcp_location: str
    model_name: str

    # Slack settings
    slack_bot_token: str
    slack_channel_id: str

    # Curator settings
    curator_topic: str
    slack_header: str

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        topic = os.environ.get("CURATOR_TOPIC", "")
        if not topic:
            raise ValueError("CURATOR_TOPIC environment variable is required")

        slack_header = os.environ.get("SLACK_HEADER", "")
        if not slack_header:
            slack_header = f"{topic} ニュース"

        return cls(
            gcp_project_id=os.environ["GCP_PROJECT_ID"],
            gcp_location=os.environ.get("GCP_LOCATION", "asia-northeast1"),
            model_name=os.environ.get("MODEL_NAME", "gemini-2.5-pro"),
            slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
            slack_channel_id=os.environ["SLACK_CHANNEL_ID"],
            curator_topic=topic,
            slack_header=slack_header,
        )
