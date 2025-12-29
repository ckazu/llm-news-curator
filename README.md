# LLM News Curator

Vertex AI (Gemini 2.5 Pro) と Google Search グラウンディングを使用して、任意のテーマのニュースを自動収集し、Slackに投稿するシステム。

## 機能

- Google Search グラウンディングによるリアルタイムニュース検索
- テーマを設定するだけで様々なニュース収集に対応
- **複数トピック対応**: 異なるテーマを異なるチャンネルに投稿可能
- Slack Block Kit を使用した見やすいフォーマット
- 過去3日間の既報ニュースを自動除外（重複防止）
- GitHub Actions による毎日の自動実行

## セットアップ

### 1. Google Cloud セットアップ

```bash
# 0. gcloud 認証とプロジェクト設定
gcloud auth login
gcloud config set project PROJECT_ID

# 1. サービスアカウント作成
gcloud iam service-accounts create llm-news-curator \
  --display-name="LLM News Curator"

# 2. Vertex AI 権限を付与
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:llm-news-curator@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# 3. サービスアカウントキーを作成（GitHub Actions 用）
gcloud iam service-accounts keys create credentials.json \
  --iam-account=llm-news-curator@PROJECT_ID.iam.gserviceaccount.com
```

**注意**: `PROJECT_ID` を実際のプロジェクトIDに置き換えてください。

### 2. Slack セットアップ

1. [Slack Apps](https://api.slack.com/apps) で新しいアプリを作成
2. OAuth & Permissions で以下の Bot Token Scopes を追加:
   - `chat:write`
   - `channels:history`（重複防止機能に必要）
3. アプリをワークスペースにインストール
4. Bot User OAuth Token をコピー
5. 投稿先チャンネルの ID を取得

### 3. GitHub Secrets / Variables 設定

リポジトリの Settings > Secrets and variables > Actions で以下を設定:

#### Secrets（機密情報）

| 名前 | 説明 |
|------|------|
| `GCP_CREDENTIALS_JSON` | サービスアカウントキー JSON の内容をそのまま貼り付け |
| `GCP_PROJECT_ID` | GCP プロジェクト ID |
| `SLACK_BOT_TOKEN` | Slack Bot User OAuth Token (`xoxb-...`) |
| `SLACK_CHANNEL_ID` | 投稿先チャンネル ID（単一トピックモード時のみ） |

#### Variables（設定値）

| 名前 | 説明 | 例 |
|------|------|-----|
| `TOPICS_CONFIG` | 複数トピック設定（JSON配列） | 下記参照 |
| `CURATOR_TOPIC` | 収集するニュースのテーマ（単一トピックモード） | `AI生成ゲーム` |
| `SLACK_HEADER` | Slack メッセージのヘッダー（任意） | 空の場合は `{topic} ニュース` |
| `GCP_LOCATION` | Vertex AI リージョン（任意） | `asia-northeast1` |
| `MODEL_NAME` | 使用するモデル（任意） | `gemini-2.5-pro` |

### 4. ローカル開発

```bash
# Google Cloud 認証（初回のみ）
gcloud auth application-default login

# 依存関係のインストール
uv sync

# 環境変数の設定
cp .env.example .env
# .env を編集して GCP_PROJECT_ID, Slack設定, CURATOR_TOPIC を設定

# 実行
uv run python -m src.main
```

> **Note**: ローカル開発では `gcloud auth application-default login` で認証するため、サービスアカウントキーは不要です。

## トピック設定

### 複数トピックモード（推奨）

`TOPICS_CONFIG` 変数に JSON 配列を設定することで、複数のトピックを異なるチャンネルに投稿できます。

```json
[
  {
    "name": "生成AI",
    "channel_id": "C0123456789",
    "header": "🤖 生成AI ニュース"
  },
  {
    "name": "ゲーム",
    "channel_id": "C9876543210",
    "header": "🎮 ゲーム ニュース"
  }
]
```

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `name` | ✅ | 検索するトピック名 |
| `channel_id` | ✅ | 投稿先 Slack チャンネル ID |
| `header` | - | メッセージヘッダー（省略時: `{name} ニュース`） |

**設定方法**: GitHub リポジトリの Settings → Secrets and variables → Actions → Variables で `TOPICS_CONFIG` を作成し、上記 JSON を貼り付け。

### 単一トピックモード（レガシー）

`TOPICS_CONFIG` が未設定の場合、従来の環境変数を使用します：

```
CURATOR_TOPIC=AI生成ゲーム
SLACK_CHANNEL_ID=C0XXXXXXX
SLACK_HEADER=🎮 AI生成ゲーム ニュース（任意）
```

## 使用例

### 複数トピックの設定例

```json
[
  {"name": "生成AI LLM", "channel_id": "C111...", "header": "🤖 生成AI ニュース"},
  {"name": "スタートアップ 資金調達", "channel_id": "C222...", "header": "💰 スタートアップ ニュース"},
  {"name": "Web3 ブロックチェーン", "channel_id": "C333...", "header": "⛓️ Web3 ニュース"}
]
```

### 単一トピックの設定例

```
CURATOR_TOPIC=大規模言語モデル LLM
```

## アーキテクチャ

```
GitHub Actions (毎日 13:00 JST)
    │
    ▼
Python Script
    │
    ├─→ Slack API (履歴取得: 重複防止)
    │
    ├─→ Vertex AI (Gemini 2.5 Pro + Google Search)
    │       │
    │       └─→ リアルタイムニュース検索・要約
    │
    └─→ Slack Bot Token
            │
            └─→ 各チャンネルへ投稿
```

## ファイル構成

```
llm-news-curator/
├── .github/workflows/
│   └── daily-news-curator.yml   # GitHub Actions
├── src/
│   ├── __init__.py
│   ├── main.py                  # エントリーポイント
│   ├── news_curator.py          # Vertex AI 連携
│   ├── slack_poster.py          # Slack 投稿
│   └── config.py                # 設定管理
├── pyproject.toml               # uv 依存関係管理
├── uv.lock
├── .env.example
└── README.md
```

## 手動実行

GitHub Actions の Actions タブから workflow_dispatch で手動実行できます。

## ライセンス

MIT
