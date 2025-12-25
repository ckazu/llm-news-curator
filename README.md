# LLM News Curator

Vertex AI (Gemini 2.5 Pro) と Google Search グラウンディングを使用して、任意のテーマのニュースを自動収集し、Slackに投稿するシステム。

## 機能

- Google Search グラウンディングによるリアルタイムニュース検索
- テーマを設定するだけで様々なニュース収集に対応
- Slack Block Kit を使用した見やすいフォーマット
- GitHub Actions による毎日の自動実行

## セットアップ

### 1. Google Cloud セットアップ（Workload Identity Federation）

秘密鍵を使わない安全な認証方式です。

```bash
# 1. サービスアカウント作成
gcloud iam service-accounts create llm-news-curator \
  --display-name="LLM News Curator"

# 2. Vertex AI 権限を付与
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:llm-news-curator@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# 3. Workload Identity Pool 作成
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# 4. OIDC Provider 追加
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'YOUR_GITHUB_USERNAME'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 5. サービスアカウントに WIF 権限付与
gcloud iam service-accounts add-iam-policy-binding \
  "llm-news-curator@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_USERNAME/llm-news-curator"

# 6. プロバイダー名を取得（GitHub Secrets に設定）
gcloud iam workload-identity-pools providers describe "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"
```

**注意**: `PROJECT_ID`, `PROJECT_NUMBER`, `YOUR_GITHUB_USERNAME` を実際の値に置き換えてください。

### 2. Slack セットアップ

1. [Slack Apps](https://api.slack.com/apps) で新しいアプリを作成
2. OAuth & Permissions で以下の Bot Token Scopes を追加:
   - `chat:write`
3. アプリをワークスペースにインストール
4. Bot User OAuth Token をコピー
5. 投稿先チャンネルの ID を取得

### 3. GitHub Secrets / Variables 設定

リポジトリの Settings > Secrets and variables > Actions で以下を設定:

#### Secrets（機密情報）

| 名前 | 説明 |
|------|------|
| `GCP_PROJECT_ID` | GCP プロジェクト ID |
| `GCP_SERVICE_ACCOUNT` | サービスアカウントメール（例: `llm-news-curator@PROJECT_ID.iam.gserviceaccount.com`） |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | 手順6で取得したプロバイダー名 |
| `SLACK_BOT_TOKEN` | Slack Bot User OAuth Token (`xoxb-...`) |
| `SLACK_CHANNEL_ID` | 投稿先チャンネル ID (`C0XXXXXXX`) |

#### Variables（設定値）

| 名前 | 説明 | 例 |
|------|------|-----|
| `CURATOR_TOPIC` | 収集するニュースのテーマ | `AI生成ゲーム` |
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
# または
uv run llm-news-curator
```

> **Note**: ローカル開発では `gcloud auth application-default login` で認証するため、サービスアカウントキーは不要です。

## 使用例

### AI生成ゲームのニュース収集

```
CURATOR_TOPIC=AI生成ゲーム
```

### LLM関連ニュースの収集

```
CURATOR_TOPIC=大規模言語モデル LLM
```

### スタートアップ資金調達ニュースの収集

```
CURATOR_TOPIC=スタートアップ 資金調達
```

## アーキテクチャ

```
GitHub Actions (毎日 9:00 JST)
    │
    ▼
Python Script
    │
    ├─→ Vertex AI (Gemini 2.5 Pro + Google Search)
    │       │
    │       └─→ リアルタイムニュース検索・要約
    │
    └─→ Slack Bot Token
            │
            └─→ チャンネルへ投稿
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
