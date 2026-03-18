# @mcpware/ig-mcp

[![npm version](https://img.shields.io/npm/v/@mcpware/ig-mcp.svg)](https://www.npmjs.com/package/@mcpware/ig-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/node-%3E%3D20-brightgreen)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue)](https://www.typescriptlang.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io)

Instagram MCP server — **23 tools** for the Instagram Graph API via [Model Context Protocol](https://modelcontextprotocol.io).

Manage posts, comments, DMs, stories, hashtags, reels, carousels, and analytics from Claude Code, Cursor, or any MCP client.

## Quick Start

```bash
npx @mcpware/ig-mcp
```

### Claude Code / Cursor `.mcp.json`

```json
{
  "mcpServers": {
    "instagram": {
      "command": "npx",
      "args": ["-y", "@mcpware/ig-mcp"],
      "env": {
        "INSTAGRAM_ACCESS_TOKEN": "your-meta-long-lived-token",
        "INSTAGRAM_ACCOUNT_ID": "your-ig-business-account-id"
      }
    }
  }
}
```

## Prerequisites

- **Instagram Business or Creator account** (personal accounts are not supported by the Graph API)
- **Facebook Page** connected to your Instagram account
- **Meta long-lived access token** (see setup guide below)
- **Instagram Business Account ID** (obtained during setup)

## Setup Guide — Getting Your Access Token

Meta's token setup is a multi-step process. Follow these steps carefully.

### Step 1: Connect Instagram to a Facebook Page

Your IG Business/Creator account **must** be linked to a Facebook Page. Without this, the Graph API won't work.

1. Open Instagram app → Settings → Account → Sharing to other apps → Facebook
2. Select the Facebook Page to connect
3. If you don't have a Page, create one at [facebook.com/pages/creation](https://www.facebook.com/pages/creation/)

### Step 2: Create a Meta Developer App

1. Go to [developers.facebook.com](https://developers.facebook.com) → Log in
2. Click **"Create App"**
3. App name: anything (e.g. "My IG Tool") — **cannot contain** "IG", "Instagram", "Facebook", etc.
4. Use case: **"Manage Instagram content and messaging"** (under Content Management)
5. Skip business portfolio for now
6. Complete creation

### Step 3: Add Permissions

1. Go to your app → **Use Cases** → **Customize** → **API setup with Facebook login**
2. Click **"Add required content permissions"** (adds `instagram_basic`, `instagram_content_publish`, `pages_read_engagement`, `business_management`, `pages_show_list`)
3. Click **"Add required messaging permissions"** (adds `instagram_manage_messages`)

### Step 4: Generate Access Token

1. Go to [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
2. Select your app from the **"Meta App"** dropdown
3. Click **"Get Token"** → **"Get User Access Token"**
4. Add permissions: `instagram_basic`, `pages_show_list`, `pages_read_engagement`, `instagram_content_publish`, `instagram_manage_insights`, `instagram_manage_comments`
5. Click **"Generate Access Token"** → Authorize in the popup
6. Copy the token

> **Note:** The Graph API Explorer may show "No configuration available" if permissions aren't set up yet. Make sure Step 3 is done first.

### Step 5: Get Your Instagram Business Account ID

In the Graph API Explorer, run:

```
GET /me/accounts?fields=id,name,instagram_business_account
```

Find your Page in the response. The `instagram_business_account.id` is your `INSTAGRAM_ACCOUNT_ID`.

### Step 6: Exchange for Long-Lived Token (60 days)

The token from Step 4 expires in 1 hour. Exchange it:

```bash
curl "https://graph.facebook.com/v19.0/oauth/access_token?\
grant_type=fb_exchange_token&\
client_id=YOUR_APP_ID&\
client_secret=YOUR_APP_SECRET&\
fb_exchange_token=YOUR_SHORT_LIVED_TOKEN"
```

Find your App ID and App Secret in: App Dashboard → Settings → Basic.

The returned `access_token` is valid for 60 days.

### Step 7: Configure and Run

```bash
export INSTAGRAM_ACCESS_TOKEN="your-long-lived-token"
export INSTAGRAM_ACCOUNT_ID="your-ig-business-account-id"
npx @mcpware/ig-mcp
```

### Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `me/accounts` returns empty `[]` | IG not connected to a Facebook Page, or you're not Page admin | Do Step 1 |
| Graph API Explorer says "No configuration available" | Permissions not added to app | Do Step 3 |
| "Generate Access Token" is disabled | Need to select "Get User Access Token" first | Click "Get Token" dropdown |
| App name rejected (contains "IG", "Insta", etc.) | Meta blocks trademarked words | Use a generic name |
| Token expired | Short-lived tokens last 1 hour | Do Step 6 for 60-day token |
| `(#10) To use Instagram Graph API...` | IG account is Personal, not Business | Switch to Business/Creator in IG settings |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INSTAGRAM_ACCESS_TOKEN` | Yes | — | Meta long-lived access token |
| `INSTAGRAM_ACCOUNT_ID` | Yes | — | Instagram business account ID |
| `INSTAGRAM_API_VERSION` | No | `v19.0` | Graph API version |

## Tools (23)

### Profile & Account
| Tool | Description |
|------|-------------|
| `get_profile_info` | Get profile info (bio, followers, media count) |
| `get_account_pages` | List connected Facebook pages |
| `get_account_insights` | Account-level analytics (reach, profile views) |
| `validate_access_token` | Check if token is valid |

### Media & Publishing
| Tool | Description |
|------|-------------|
| `get_media_posts` | Get recent posts with engagement metrics |
| `get_media_insights` | Detailed analytics for a specific post |
| `publish_media` | Publish image or video |
| `publish_carousel` | Publish carousel (2-10 images/videos) |
| `publish_reel` | Publish a Reel |
| `get_content_publishing_limit` | Check daily publishing quota |

### Comments
| Tool | Description |
|------|-------------|
| `get_comments` | Get comments on a post |
| `post_comment` | Post a comment |
| `reply_to_comment` | Reply to a comment |
| `delete_comment` | Delete a comment |
| `hide_comment` | Hide/unhide a comment |

### Direct Messages
| Tool | Description |
|------|-------------|
| `get_conversations` | List DM conversations |
| `get_conversation_messages` | Read messages in a conversation |
| `send_dm` | Send a direct message |

### Discovery & Content
| Tool | Description |
|------|-------------|
| `search_hashtag` | Search for a hashtag ID |
| `get_hashtag_media` | Get top/recent media for a hashtag |
| `get_stories` | Get current active stories |
| `get_mentions` | Get posts you're tagged in |
| `business_discovery` | Look up another business account |

## Limitations

These are Instagram Graph API limitations, not this tool's:

- **Business/Creator accounts only** — personal accounts are not supported
- **Long-lived tokens expire after 60 days** — refresh before expiry
- **200 API calls per hour** rate limit
- **25 posts per day** publishing limit
- **DMs require Advanced Access** — Meta app review required
- **Hashtag search**: 30 unique hashtags per 7 days

## Credits

TypeScript rewrite of [jlbadano/ig-mcp](https://github.com/jlbadano/ig-mcp) (Python).

## License

MIT
