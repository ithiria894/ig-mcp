/**
 * Instagram Graph API client.
 *
 * Handles HTTP calls, rate limiting, caching, and error handling.
 * Uses native fetch (Node 20+) — zero HTTP dependencies.
 */

// ─── Types ───────────────────────────────────────────────────────────

export interface IGProfile {
  id: string;
  username: string;
  name?: string;
  biography?: string;
  website?: string;
  followers_count?: number;
  follows_count?: number;
  media_count?: number;
  profile_picture_url?: string;
}

export interface IGMedia {
  id: string;
  media_type: string;
  media_url?: string;
  permalink?: string;
  thumbnail_url?: string;
  caption?: string;
  timestamp?: string;
  like_count?: number;
  comments_count?: number;
}

export interface IGComment {
  id: string;
  text?: string;
  timestamp?: string;
  username?: string;
  like_count?: number;
  hidden?: boolean;
}

export interface IGConversation {
  id: string;
  updated_time: string;
  message_count?: number;
}

export interface IGMessage {
  id: string;
  from: Record<string, string>;
  to: Record<string, string>[];
  message?: string;
  created_time: string;
  attachments?: any[];
}

export interface IGHashtag {
  id: string;
  name?: string;
}

export interface IGStory {
  id: string;
  media_type?: string;
  media_url?: string;
  timestamp?: string;
}

export interface IGMention {
  id: string;
  media_type?: string;
  media_url?: string;
  permalink?: string;
  caption?: string;
  timestamp?: string;
  username?: string;
}

export interface IGBusinessProfile {
  id: string;
  username?: string;
  name?: string;
  biography?: string;
  website?: string;
  followers_count?: number;
  follows_count?: number;
  media_count?: number;
  profile_picture_url?: string;
}

export interface IGPublishResult {
  id: string;
}

export interface IGPublishingLimit {
  quota_usage?: number;
  config?: Record<string, any>;
  quota_duration?: number;
}

export interface IGPage {
  id: string;
  name: string;
  instagram_business_account?: { id: string };
}

export interface IGInsight {
  name: string;
  period: string;
  values?: Record<string, any>[];
  total_value?: Record<string, any>;
  title?: string;
  description?: string;
  id?: string;
}

// ─── Error ───────────────────────────────────────────────────────────

export class InstagramAPIError extends Error {
  constructor(
    message: string,
    public errorCode?: number,
    public errorSubcode?: number
  ) {
    super(message);
    this.name = "InstagramAPIError";
  }
}

// ─── Cache ───────────────────────────────────────────────────────────

interface CacheEntry {
  data: any;
  expiresAt: number;
}

// ─── Client ──────────────────────────────────────────────────────────

export class InstagramClient {
  private baseUrl: string;
  private accessToken: string;
  private accountId: string;
  private cache = new Map<string, CacheEntry>();
  private cacheTtl: number;

  // Simple rate limiter: track call timestamps
  private callTimestamps: number[] = [];
  private rateLimit: number;

  constructor(opts: {
    accessToken: string;
    accountId: string;
    apiVersion?: string;
    cacheTtlSeconds?: number;
    rateLimitPerHour?: number;
  }) {
    this.accessToken = opts.accessToken;
    this.accountId = opts.accountId;
    this.baseUrl = `https://graph.facebook.com/${opts.apiVersion ?? "v19.0"}`;
    this.cacheTtl = (opts.cacheTtlSeconds ?? 300) * 1000;
    this.rateLimit = opts.rateLimitPerHour ?? 200;
  }

  // ── HTTP ─────────────────────────────────────────────────────────

  private async request(
    method: string,
    endpoint: string,
    opts?: {
      params?: Record<string, string>;
      body?: Record<string, any>;
      useCache?: boolean;
      useFacebookApi?: boolean;
    }
  ): Promise<any> {
    const params = new URLSearchParams(opts?.params ?? {});
    params.set("access_token", this.accessToken);

    // Cache check (GET only)
    const cacheKey = `${endpoint}?${params.toString()}`;
    if (method === "GET" && opts?.useCache !== false) {
      const cached = this.cache.get(cacheKey);
      if (cached && cached.expiresAt > Date.now()) {
        return cached.data;
      }
    }

    // Rate limiting
    await this.throttle();

    const base = opts?.useFacebookApi
      ? "https://graph.facebook.com/v22.0"
      : this.baseUrl;
    const url = `${base}/${endpoint}?${params.toString()}`;

    const fetchOpts: RequestInit = { method };
    if (opts?.body) {
      fetchOpts.headers = { "Content-Type": "application/json" };
      fetchOpts.body = JSON.stringify(opts.body);
    }

    const res = await fetch(url, fetchOpts);

    if (res.status === 429) {
      throw new InstagramAPIError("Rate limit exceeded");
    }

    const data = await res.json();

    if (data.error) {
      throw new InstagramAPIError(
        data.error.message ?? "Unknown error",
        data.error.code,
        data.error.error_subcode
      );
    }

    // Cache successful GET
    if (method === "GET" && opts?.useCache !== false) {
      this.cache.set(cacheKey, {
        data,
        expiresAt: Date.now() + this.cacheTtl,
      });
    }

    return data;
  }

  private async throttle(): Promise<void> {
    const now = Date.now();
    const oneHourAgo = now - 3600_000;
    this.callTimestamps = this.callTimestamps.filter((t) => t > oneHourAgo);
    if (this.callTimestamps.length >= this.rateLimit) {
      const waitMs = this.callTimestamps[0] - oneHourAgo + 100;
      await new Promise((r) => setTimeout(r, waitMs));
    }
    this.callTimestamps.push(Date.now());
  }

  private aid(): string {
    return this.accountId;
  }

  // ── Profile ──────────────────────────────────────────────────────

  async getProfileInfo(accountId?: string): Promise<IGProfile> {
    const id = accountId ?? this.aid();
    const fields =
      "id,username,name,biography,website,profile_picture_url,followers_count,follows_count,media_count";
    return this.request("GET", id, { params: { fields } });
  }

  // ── Media ────────────────────────────────────────────────────────

  async getMediaPosts(
    limit = 25,
    after?: string,
    accountId?: string
  ): Promise<IGMedia[]> {
    const id = accountId ?? this.aid();
    const fields =
      "id,media_type,media_url,permalink,thumbnail_url,caption,timestamp,like_count,comments_count";
    const params: Record<string, string> = {
      fields,
      limit: String(Math.min(limit, 100)),
    };
    if (after) params.after = after;
    const data = await this.request("GET", `${id}/media`, { params });
    return data.data ?? [];
  }

  async getMediaInsights(
    mediaId: string,
    metrics?: string[]
  ): Promise<IGInsight[]> {
    const defaultMetrics = ["reach", "likes", "comments", "shares", "saved"];
    const params = { metric: (metrics ?? defaultMetrics).join(",") };
    const data = await this.request("GET", `${mediaId}/insights`, { params });
    return data.data ?? [];
  }

  // ── Publishing ───────────────────────────────────────────────────

  async publishMedia(opts: {
    imageUrl?: string;
    videoUrl?: string;
    caption?: string;
    locationId?: string;
  }): Promise<IGPublishResult> {
    const id = this.aid();
    const body: Record<string, any> = {};
    if (opts.caption) body.caption = opts.caption;
    if (opts.imageUrl) body.image_url = opts.imageUrl;
    else if (opts.videoUrl) body.video_url = opts.videoUrl;
    else throw new InstagramAPIError("Either imageUrl or videoUrl is required");
    if (opts.locationId) body.location_id = opts.locationId;

    const container = await this.request("POST", `${id}/media`, { body });
    const result = await this.request("POST", `${id}/media_publish`, {
      body: { creation_id: container.id },
    });
    return { id: result.id };
  }

  async publishCarousel(
    imageUrls: string[],
    caption?: string
  ): Promise<IGPublishResult> {
    const id = this.aid();
    if (imageUrls.length < 2)
      throw new InstagramAPIError("Carousel requires at least 2 items");
    if (imageUrls.length > 10)
      throw new InstagramAPIError("Carousel supports maximum 10 items");

    const containerIds: string[] = [];
    for (const url of imageUrls) {
      const isVideo = /\.(mp4|mov)$/i.test(url);
      const body: Record<string, any> = { is_carousel_item: "true" };
      if (isVideo) {
        body.video_url = url;
        body.media_type = "VIDEO";
      } else {
        body.image_url = url;
      }
      const resp = await this.request("POST", `${id}/media`, { body });
      containerIds.push(resp.id);
    }

    const carouselBody: Record<string, any> = {
      media_type: "CAROUSEL",
      children: containerIds.join(","),
    };
    if (caption) carouselBody.caption = caption;
    const carousel = await this.request("POST", `${id}/media`, {
      body: carouselBody,
    });

    const result = await this.request("POST", `${id}/media_publish`, {
      body: { creation_id: carousel.id },
    });
    return { id: result.id };
  }

  async publishReel(
    videoUrl: string,
    caption?: string,
    shareToFeed = true
  ): Promise<IGPublishResult> {
    const id = this.aid();
    const body: Record<string, any> = {
      media_type: "REELS",
      video_url: videoUrl,
      share_to_feed: String(shareToFeed),
    };
    if (caption) body.caption = caption;

    const container = await this.request("POST", `${id}/media`, { body });
    const result = await this.request("POST", `${id}/media_publish`, {
      body: { creation_id: container.id },
    });
    return { id: result.id };
  }

  async getContentPublishingLimit(
    accountId?: string
  ): Promise<IGPublishingLimit> {
    const id = accountId ?? this.aid();
    const data = await this.request(
      "GET",
      `${id}/content_publishing_limit`,
      { params: { fields: "quota_usage,config,quota_duration" } }
    );
    return data.data?.[0] ?? {};
  }

  // ── Account ──────────────────────────────────────────────────────

  async getAccountPages(): Promise<IGPage[]> {
    const data = await this.request("GET", "me/accounts", {
      params: { fields: "id,name,instagram_business_account" },
    });
    return data.data ?? [];
  }

  async getAccountInsights(
    metrics?: string[],
    period = "day",
    accountId?: string
  ): Promise<IGInsight[]> {
    const id = accountId ?? this.aid();
    const defaultMetrics = ["reach", "profile_views", "website_clicks"];
    const data = await this.request("GET", `${id}/insights`, {
      params: {
        metric: (metrics ?? defaultMetrics).join(","),
        period,
        metric_type: "total_value",
      },
    });
    return data.data ?? [];
  }

  async validateAccessToken(): Promise<boolean> {
    try {
      await this.request("GET", "me", {
        params: { fields: "id" },
        useCache: false,
      });
      return true;
    } catch {
      return false;
    }
  }

  // ── DMs ──────────────────────────────────────────────────────────

  async getConversations(
    pageId?: string,
    limit = 25
  ): Promise<IGConversation[]> {
    let pid = pageId;
    if (!pid) {
      const pages = await this.getAccountPages();
      if (!pages.length)
        throw new InstagramAPIError("No Facebook pages found");
      pid = pages[0].id;
    }
    const data = await this.request("GET", `${pid}/conversations`, {
      params: {
        platform: "instagram",
        fields: "id,updated_time,message_count",
        limit: String(Math.min(limit, 100)),
      },
      useFacebookApi: true,
    });
    return data.data ?? [];
  }

  async getConversationMessages(
    conversationId: string,
    limit = 25
  ): Promise<IGMessage[]> {
    const fields = "id,from,to,message,created_time,attachments";
    const data = await this.request("GET", conversationId, {
      params: {
        fields: `messages{${fields}}`,
        limit: String(Math.min(limit, 100)),
      },
      useFacebookApi: true,
    });
    return data.messages?.data ?? [];
  }

  async sendDM(
    recipientId: string,
    message: string
  ): Promise<{ messageId: string }> {
    const data = await this.request("POST", "me/messages", {
      body: {
        recipient: { id: recipientId },
        message: { text: message },
      },
      useFacebookApi: true,
    });
    return { messageId: data.message_id ?? "" };
  }

  // ── Comments ─────────────────────────────────────────────────────

  async getComments(mediaId: string, limit = 25): Promise<IGComment[]> {
    const data = await this.request("GET", `${mediaId}/comments`, {
      params: {
        fields: "id,text,timestamp,username,like_count,hidden",
        limit: String(Math.min(limit, 100)),
      },
    });
    return data.data ?? [];
  }

  async postComment(mediaId: string, message: string): Promise<IGComment> {
    const data = await this.request("POST", `${mediaId}/comments`, {
      body: { message },
    });
    return { id: data.id, text: message };
  }

  async replyToComment(
    commentId: string,
    message: string
  ): Promise<IGComment> {
    const data = await this.request("POST", `${commentId}/replies`, {
      body: { message },
    });
    return { id: data.id, text: message };
  }

  async deleteComment(commentId: string): Promise<void> {
    await this.request("DELETE", commentId);
  }

  async hideComment(commentId: string, hide = true): Promise<void> {
    await this.request("POST", commentId, { body: { hide } });
  }

  // ── Hashtags ─────────────────────────────────────────────────────

  async searchHashtag(hashtagName: string): Promise<IGHashtag> {
    const data = await this.request("GET", "ig_hashtag_search", {
      params: {
        q: hashtagName.replace(/^#/, ""),
        user_id: this.aid(),
      },
    });
    const results = data.data ?? [];
    if (!results.length)
      throw new InstagramAPIError(`Hashtag '${hashtagName}' not found`);
    return results[0];
  }

  async getHashtagMedia(
    hashtagId: string,
    mediaType: "top" | "recent" = "top",
    limit = 25
  ): Promise<IGMedia[]> {
    const fields =
      "id,media_type,media_url,permalink,caption,timestamp,like_count,comments_count";
    const data = await this.request(
      "GET",
      `${hashtagId}/${mediaType}_media`,
      {
        params: {
          user_id: this.aid(),
          fields,
          limit: String(Math.min(limit, 50)),
        },
      }
    );
    return data.data ?? [];
  }

  // ── Stories ──────────────────────────────────────────────────────

  async getStories(accountId?: string): Promise<IGStory[]> {
    const id = accountId ?? this.aid();
    const data = await this.request("GET", `${id}/stories`, {
      params: { fields: "id,media_type,media_url,timestamp" },
    });
    return data.data ?? [];
  }

  // ── Mentions ─────────────────────────────────────────────────────

  async getMentions(limit = 25, accountId?: string): Promise<IGMention[]> {
    const id = accountId ?? this.aid();
    const data = await this.request("GET", `${id}/tags`, {
      params: {
        fields:
          "id,media_type,media_url,permalink,caption,timestamp,username",
        limit: String(Math.min(limit, 100)),
      },
    });
    return data.data ?? [];
  }

  // ── Business Discovery ───────────────────────────────────────────

  async businessDiscovery(
    targetUsername: string
  ): Promise<IGBusinessProfile> {
    const fields =
      "username,name,biography,website,followers_count,follows_count,media_count,profile_picture_url";
    const data = await this.request("GET", this.aid(), {
      params: { fields: `business_discovery.fields(${fields})` },
    });
    const bd = data.business_discovery;
    if (!bd)
      throw new InstagramAPIError(
        `Could not find business account: ${targetUsername}`
      );
    return bd;
  }
}
