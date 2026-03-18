#!/usr/bin/env node

/**
 * Instagram MCP Server — 23 tools for the Instagram Graph API.
 *
 * Manages posts, comments, DMs, stories, hashtags, reels, carousels,
 * and analytics for Instagram Business/Creator accounts.
 *
 * Environment variables:
 *   INSTAGRAM_ACCESS_TOKEN  — Meta long-lived access token (required)
 *   INSTAGRAM_ACCOUNT_ID    — Instagram business account ID (required)
 *   INSTAGRAM_API_VERSION   — Graph API version (default: v19.0)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { InstagramClient, InstagramAPIError } from "./client.js";

// ─── Config ──────────────────────────────────────────────────────────

function getClient(): InstagramClient {
  const token = process.env.INSTAGRAM_ACCESS_TOKEN;
  const accountId = process.env.INSTAGRAM_ACCOUNT_ID;

  if (!token || !accountId) {
    throw new Error(
      "Missing required environment variables: INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID"
    );
  }

  return new InstagramClient({
    accessToken: token,
    accountId,
    apiVersion: process.env.INSTAGRAM_API_VERSION,
  });
}

let client: InstagramClient | null = null;

function ig(): InstagramClient {
  if (!client) client = getClient();
  return client;
}

// ─── Tool Definitions ────────────────────────────────────────────────

const TOOLS = [
  // ── Profile ──
  {
    name: "get_profile_info",
    description:
      "Get Instagram business profile information including followers, bio, and account details",
    inputSchema: {
      type: "object" as const,
      properties: {
        account_id: {
          type: "string",
          description: "Instagram business account ID (optional, uses configured account if not provided)",
        },
      },
    },
  },
  // ── Media ──
  {
    name: "get_media_posts",
    description:
      "Get recent media posts from Instagram account with engagement metrics",
    inputSchema: {
      type: "object" as const,
      properties: {
        account_id: { type: "string", description: "Instagram business account ID (optional)" },
        limit: { type: "integer", description: "Number of posts to retrieve (max 100)", minimum: 1, maximum: 100, default: 25 },
        after: { type: "string", description: "Pagination cursor" },
      },
    },
  },
  {
    name: "get_media_insights",
    description:
      "Get detailed insights and analytics for a specific Instagram post",
    inputSchema: {
      type: "object" as const,
      properties: {
        media_id: { type: "string", description: "Instagram media ID" },
        metrics: {
          type: "array",
          items: { type: "string", enum: ["reach", "likes", "comments", "shares", "saved", "video_views"] },
          description: "Specific metrics to retrieve (optional, gets all if not specified). Note: video_views only works for video posts",
        },
      },
      required: ["media_id"],
    },
  },
  // ── Publishing ──
  {
    name: "publish_media",
    description:
      "Upload and publish an image or video to Instagram with caption and optional location",
    inputSchema: {
      type: "object" as const,
      properties: {
        image_url: { type: "string", format: "uri", description: "URL of the image to publish (must be publicly accessible)" },
        video_url: { type: "string", format: "uri", description: "URL of the video to publish (must be publicly accessible)" },
        caption: { type: "string", description: "Caption for the post (optional)" },
        location_id: { type: "string", description: "Facebook location ID for geotagging (optional)" },
      },
      anyOf: [{ required: ["image_url"] }, { required: ["video_url"] }],
    },
  },
  {
    name: "publish_carousel",
    description:
      "Publish a carousel (album) post with 2-10 images or videos. All media must be publicly accessible URLs.",
    inputSchema: {
      type: "object" as const,
      properties: {
        image_urls: { type: "array", items: { type: "string", format: "uri" }, description: "List of 2-10 image/video URLs", minItems: 2, maxItems: 10 },
        caption: { type: "string", description: "Caption for the carousel post (optional)" },
      },
      required: ["image_urls"],
    },
  },
  {
    name: "publish_reel",
    description:
      "Publish a Reel (short-form video) to Instagram. Video must be publicly accessible URL, MP4 format.",
    inputSchema: {
      type: "object" as const,
      properties: {
        video_url: { type: "string", format: "uri", description: "URL of the video to publish as Reel" },
        caption: { type: "string", description: "Caption for the Reel (optional)" },
        share_to_feed: { type: "boolean", description: "Also share to main feed (default: true)", default: true },
      },
      required: ["video_url"],
    },
  },
  {
    name: "get_content_publishing_limit",
    description:
      "Check how many posts you can still publish today. Instagram limits content publishing per 24-hour period.",
    inputSchema: { type: "object" as const, properties: {} },
  },
  // ── Account ──
  {
    name: "get_account_pages",
    description:
      "Get Facebook pages connected to the account and their Instagram business accounts",
    inputSchema: { type: "object" as const, properties: {} },
  },
  {
    name: "get_account_insights",
    description:
      "Get account-level insights and analytics for Instagram business account",
    inputSchema: {
      type: "object" as const,
      properties: {
        account_id: { type: "string", description: "Instagram business account ID (optional)" },
        metrics: {
          type: "array",
          items: { type: "string", enum: ["reach", "profile_views", "website_clicks", "accounts_engaged"] },
          description: "Specific metrics to retrieve",
        },
        period: { type: "string", enum: ["day", "lifetime"], description: "Time period for insights", default: "day" },
      },
    },
  },
  {
    name: "validate_access_token",
    description: "Validate the Instagram API access token and check permissions",
    inputSchema: { type: "object" as const, properties: {} },
  },
  // ── DMs ──
  {
    name: "get_conversations",
    description:
      "Get Instagram DM conversations. Requires instagram_manage_messages permission.",
    inputSchema: {
      type: "object" as const,
      properties: {
        page_id: { type: "string", description: "Facebook page ID (optional, auto-detected)" },
        limit: { type: "integer", description: "Number of conversations (max 100)", minimum: 1, maximum: 100, default: 25 },
      },
    },
  },
  {
    name: "get_conversation_messages",
    description:
      "Get messages from a specific Instagram DM conversation. Requires instagram_manage_messages permission.",
    inputSchema: {
      type: "object" as const,
      properties: {
        conversation_id: { type: "string", description: "Instagram conversation ID" },
        limit: { type: "integer", description: "Number of messages (max 100)", minimum: 1, maximum: 100, default: 25 },
      },
      required: ["conversation_id"],
    },
  },
  {
    name: "send_dm",
    description:
      "Send Instagram direct message. Requires instagram_manage_messages with Advanced Access. Can only reply within 24 hours.",
    inputSchema: {
      type: "object" as const,
      properties: {
        recipient_id: { type: "string", description: "Instagram Scoped User ID (IGSID) of recipient" },
        message: { type: "string", description: "Message text (max 1000 characters)", maxLength: 1000 },
      },
      required: ["recipient_id", "message"],
    },
  },
  // ── Comments ──
  {
    name: "get_comments",
    description:
      "Get comments on an Instagram post. Returns comment text, username, timestamp, and like count.",
    inputSchema: {
      type: "object" as const,
      properties: {
        media_id: { type: "string", description: "Instagram media ID" },
        limit: { type: "integer", description: "Number of comments (max 100)", minimum: 1, maximum: 100, default: 25 },
      },
      required: ["media_id"],
    },
  },
  {
    name: "post_comment",
    description:
      "Post a top-level comment on an Instagram post. Requires instagram_manage_comments permission.",
    inputSchema: {
      type: "object" as const,
      properties: {
        media_id: { type: "string", description: "Instagram media ID" },
        message: { type: "string", description: "Comment text (max 2200 characters)", maxLength: 2200 },
      },
      required: ["media_id", "message"],
    },
  },
  {
    name: "reply_to_comment",
    description:
      "Reply to a specific comment on an Instagram post. Requires instagram_manage_comments permission.",
    inputSchema: {
      type: "object" as const,
      properties: {
        comment_id: { type: "string", description: "Comment ID to reply to" },
        message: { type: "string", description: "Reply text (max 2200 characters)", maxLength: 2200 },
      },
      required: ["comment_id", "message"],
    },
  },
  {
    name: "delete_comment",
    description:
      "Delete a comment on your Instagram post. Can only delete comments on your own media.",
    inputSchema: {
      type: "object" as const,
      properties: {
        comment_id: { type: "string", description: "Comment ID to delete" },
      },
      required: ["comment_id"],
    },
  },
  {
    name: "hide_comment",
    description:
      "Hide or unhide a comment on your Instagram post. Hidden comments are not visible to the public.",
    inputSchema: {
      type: "object" as const,
      properties: {
        comment_id: { type: "string", description: "Comment ID to hide or unhide" },
        hide: { type: "boolean", description: "True to hide, False to unhide", default: true },
      },
      required: ["comment_id"],
    },
  },
  // ── Hashtags ──
  {
    name: "search_hashtag",
    description:
      "Search for an Instagram hashtag and get its ID. Use the returned ID with get_hashtag_media.",
    inputSchema: {
      type: "object" as const,
      properties: {
        hashtag_name: { type: "string", description: "Hashtag to search for (with or without #)" },
      },
      required: ["hashtag_name"],
    },
  },
  {
    name: "get_hashtag_media",
    description:
      "Get top or recent media for a hashtag. Use search_hashtag first to get the hashtag ID.",
    inputSchema: {
      type: "object" as const,
      properties: {
        hashtag_id: { type: "string", description: "Hashtag ID from search_hashtag" },
        media_type: { type: "string", enum: ["top", "recent"], description: "Get top or recent media", default: "top" },
        limit: { type: "integer", description: "Number of posts (max 50)", minimum: 1, maximum: 50, default: 25 },
      },
      required: ["hashtag_id"],
    },
  },
  // ── Stories ──
  {
    name: "get_stories",
    description:
      "Get current active stories on your Instagram account. Stories expire after 24 hours.",
    inputSchema: {
      type: "object" as const,
      properties: {
        account_id: { type: "string", description: "Instagram account ID (optional)" },
      },
    },
  },
  // ── Mentions ──
  {
    name: "get_mentions",
    description:
      "Get posts where your account has been tagged or @mentioned. Useful for tracking UGC.",
    inputSchema: {
      type: "object" as const,
      properties: {
        limit: { type: "integer", description: "Number of mentions (max 100)", minimum: 1, maximum: 100, default: 25 },
      },
    },
  },
  // ── Business Discovery ──
  {
    name: "business_discovery",
    description:
      "Look up another public Business or Creator account's profile. Returns bio, follower count, media count, etc.",
    inputSchema: {
      type: "object" as const,
      properties: {
        target_username: { type: "string", description: "Instagram username to look up (without @)" },
      },
      required: ["target_username"],
    },
  },
];

// ─── Tool Handlers ───────────────────────────────────────────────────

async function handleTool(name: string, args: any): Promise<string> {
  const c = ig();

  switch (name) {
    case "get_profile_info":
      return JSON.stringify(await c.getProfileInfo(args.account_id), null, 2);

    case "get_media_posts": {
      const posts = await c.getMediaPosts(args.limit ?? 25, args.after, args.account_id);
      return JSON.stringify({ posts, count: posts.length }, null, 2);
    }

    case "get_media_insights": {
      const insights = await c.getMediaInsights(args.media_id, args.metrics);
      return JSON.stringify({ media_id: args.media_id, insights }, null, 2);
    }

    case "publish_media": {
      const result = await c.publishMedia({
        imageUrl: args.image_url,
        videoUrl: args.video_url,
        caption: args.caption,
        locationId: args.location_id,
      });
      return JSON.stringify(result, null, 2);
    }

    case "publish_carousel": {
      const result = await c.publishCarousel(args.image_urls, args.caption);
      return JSON.stringify(result, null, 2);
    }

    case "publish_reel": {
      const result = await c.publishReel(args.video_url, args.caption, args.share_to_feed ?? true);
      return JSON.stringify(result, null, 2);
    }

    case "get_content_publishing_limit":
      return JSON.stringify(await c.getContentPublishingLimit(), null, 2);

    case "get_account_pages": {
      const pages = await c.getAccountPages();
      return JSON.stringify({ pages, count: pages.length }, null, 2);
    }

    case "get_account_insights": {
      const insights = await c.getAccountInsights(
        args.metrics,
        args.period ?? "day",
        args.account_id
      );
      return JSON.stringify({ insights, period: args.period ?? "day" }, null, 2);
    }

    case "validate_access_token": {
      const valid = await c.validateAccessToken();
      return JSON.stringify({ valid }, null, 2);
    }

    case "get_conversations": {
      const convos = await c.getConversations(args.page_id, args.limit ?? 25);
      return JSON.stringify({ conversations: convos, count: convos.length }, null, 2);
    }

    case "get_conversation_messages": {
      const messages = await c.getConversationMessages(args.conversation_id, args.limit ?? 25);
      return JSON.stringify({ conversation_id: args.conversation_id, messages, count: messages.length }, null, 2);
    }

    case "send_dm": {
      const result = await c.sendDM(args.recipient_id, args.message);
      return JSON.stringify(result, null, 2);
    }

    case "get_comments": {
      const comments = await c.getComments(args.media_id, args.limit ?? 25);
      return JSON.stringify({ media_id: args.media_id, comments, count: comments.length }, null, 2);
    }

    case "post_comment": {
      const comment = await c.postComment(args.media_id, args.message);
      return JSON.stringify(comment, null, 2);
    }

    case "reply_to_comment": {
      const reply = await c.replyToComment(args.comment_id, args.message);
      return JSON.stringify(reply, null, 2);
    }

    case "delete_comment":
      await c.deleteComment(args.comment_id);
      return JSON.stringify({ comment_id: args.comment_id, deleted: true }, null, 2);

    case "hide_comment":
      await c.hideComment(args.comment_id, args.hide ?? true);
      return JSON.stringify({ comment_id: args.comment_id, hidden: args.hide ?? true }, null, 2);

    case "search_hashtag":
      return JSON.stringify(await c.searchHashtag(args.hashtag_name), null, 2);

    case "get_hashtag_media": {
      const media = await c.getHashtagMedia(args.hashtag_id, args.media_type ?? "top", args.limit ?? 25);
      return JSON.stringify({ hashtag_id: args.hashtag_id, media, count: media.length }, null, 2);
    }

    case "get_stories": {
      const stories = await c.getStories(args.account_id);
      return JSON.stringify({ stories, count: stories.length }, null, 2);
    }

    case "get_mentions": {
      const mentions = await c.getMentions(args.limit ?? 25);
      return JSON.stringify({ mentions, count: mentions.length }, null, 2);
    }

    case "business_discovery":
      return JSON.stringify(await c.businessDiscovery(args.target_username), null, 2);

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ─── MCP Server ──────────────────────────────────────────────────────

const server = new Server(
  { name: "instagram-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS,
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const result = await handleTool(name, args ?? {});
    return { content: [{ type: "text" as const, text: result }] };
  } catch (error) {
    const message =
      error instanceof InstagramAPIError
        ? `Instagram API error: ${error.message}`
        : error instanceof Error
          ? error.message
          : String(error);
    return {
      content: [{ type: "text" as const, text: `Error: ${message}` }],
      isError: true,
    };
  }
});

// ─── Start ───────────────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("instagram-mcp server running");
}

main().catch((error) => {
  console.error("Fatal:", error);
  process.exit(1);
});
