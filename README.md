# @mcpware/ig-mcp

Instagram MCP server — 23 tools for the Instagram Graph API via [Model Context Protocol](https://modelcontextprotocol.io).

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
- **Meta long-lived access token** with permissions:
  - `instagram_basic` — profile info, media
  - `instagram_content_publish` — publish posts, reels, carousels
  - `instagram_manage_comments` — read/write comments
  - `instagram_manage_insights` — analytics
  - `instagram_manage_messages` — DMs (requires Advanced Access from Meta)
  - `pages_show_list` — connected Facebook pages
- **Instagram Business Account ID** — found in Meta Business Suite or via `get_account_pages` tool

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
