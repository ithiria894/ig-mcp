"""
Instagram API client for MCP server.
"""

import json
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import structlog
from asyncio_throttle import Throttler
from PIL import Image

from .config import get_settings
from .models.instagram_models import (
    AccountInsight,
    BusinessDiscoveryProfile,
    ContentPublishingLimit,
    FacebookPage,
    HashtagMedia,
    InsightMetric,
    InsightPeriod,
    InstagramComment,
    InstagramConversation,
    InstagramHashtag,
    InstagramMedia,
    InstagramMention,
    InstagramMessage,
    InstagramProfile,
    InstagramStory,
    MediaInsight,
    PublishMediaRequest,
    PublishMediaResponse,
    RateLimitInfo,
    ReplyCommentRequest,
    SendDMRequest,
    SendDMResponse,
)

logger = structlog.get_logger(__name__)


class InstagramAPIError(Exception):
    """Custom exception for Instagram API errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        error_subcode: Optional[int] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.error_subcode = error_subcode
        super().__init__(self.message)


class RateLimitExceeded(InstagramAPIError):
    """Exception raised when rate limit is exceeded."""

    pass


class InstagramClient:
    """Instagram Graph API client with rate limiting and error handling."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.instagram_api_url
        self.access_token = self.settings.instagram_access_token

        # Rate limiting
        self.throttler = Throttler(
            rate_limit=self.settings.rate_limit_requests_per_hour, period=3600  # 1 hour
        )

        # HTTP client with timeout and retry configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        # Cache for storing responses
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Instagram client initialized",
            api_version=self.settings.instagram_api_version,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        param_str = urlencode(sorted(params.items()))
        return f"{endpoint}?{param_str}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if not self.settings.cache_enabled:
            return False

        expires_at = cache_entry.get("expires_at")
        if not expires_at:
            return False

        return datetime.fromisoformat(expires_at) > datetime.utcnow()

    def _cache_response(self, key: str, data: Any) -> None:
        """Cache API response."""
        if not self.settings.cache_enabled:
            return

        expires_at = datetime.utcnow() + timedelta(
            seconds=self.settings.cache_ttl_seconds
        )
        self._cache[key] = {
            "data": data,
            "expires_at": expires_at.isoformat(),
            "cached_at": datetime.utcnow().isoformat(),
        }

    async def _validate_image_aspect_ratio(self, image_url: str) -> None:
        """
        Validate image aspect ratio against Instagram requirements.
        Raises InstagramAPIError if ratio is not supported.

        Instagram accepts:
        - Portrait: 4:5 (ratio 0.8)
        - Square: 1:1 (ratio 1.0)
        - Landscape: 1.91:1 (ratio 1.91)
        """
        try:
            # Download image to get dimensions
            response = await self.client.get(image_url)
            response.raise_for_status()

            # Open image and get dimensions
            image = Image.open(BytesIO(response.content))
            width, height = image.size
            ratio = width / height

            # Instagram accepted ratios with tolerance
            ACCEPTED_RATIOS = {
                "4:5 (portrait)": (0.8, 0.78, 0.82),  # 4:5 with ±2% tolerance
                "1:1 (square)": (1.0, 0.98, 1.02),  # 1:1 with ±2% tolerance
                "1.91:1 (landscape)": (1.91, 1.89, 1.93),  # 1.91:1 with ±2% tolerance
            }

            # Check if ratio matches any accepted ratio
            for ratio_name, (target, min_ratio, max_ratio) in ACCEPTED_RATIOS.items():
                if min_ratio <= ratio <= max_ratio:
                    logger.debug(
                        "Image aspect ratio valid",
                        width=width,
                        height=height,
                        ratio=ratio,
                        accepted_as=ratio_name,
                    )
                    return

            # Ratio not accepted - build helpful error message
            error_msg = (
                f"Image aspect ratio {width}:{height} (ratio {ratio:.2f}) is not supported by Instagram.\n"
                f"Accepted ratios are:\n"
                f"  • 4:5 (portrait, ratio ~0.8)\n"
                f"  • 1:1 (square, ratio 1.0)\n"
                f"  • 1.91:1 (landscape, ratio ~1.91)\n"
                f"Please crop or resize your image to one of these ratios."
            )

            logger.warning(
                "Invalid image aspect ratio",
                width=width,
                height=height,
                ratio=ratio,
                url=image_url,
            )

            raise InstagramAPIError(error_msg)

        except httpx.HTTPError as e:
            logger.error(
                "Failed to download image for validation", error=str(e), url=image_url
            )
            raise InstagramAPIError(
                f"Failed to download image for validation: {str(e)}"
            )
        except Exception as e:
            if isinstance(e, InstagramAPIError):
                raise
            logger.error(
                "Failed to validate image aspect ratio", error=str(e), url=image_url
            )
            raise InstagramAPIError(f"Failed to validate image: {str(e)}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        use_facebook_api: bool = False,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Instagram API with rate limiting and error handling.

        Args:
            use_facebook_api: If True, use graph.facebook.com instead of graph.instagram.com.
                             Required for Instagram Direct Messaging (Messenger Platform API).
        """

        # Prepare request parameters
        if params is None:
            params = {}

        # Add access token to params
        params["access_token"] = self.access_token

        # Check cache first for GET requests
        cache_key = self._get_cache_key(endpoint, params)
        if method.upper() == "GET" and use_cache and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug("Cache hit", endpoint=endpoint)
                return cache_entry["data"]

        # Apply rate limiting
        async with self.throttler:
            # Choose base URL: Facebook for DMs, Instagram for everything else
            if use_facebook_api:
                base_url = "https://graph.facebook.com/v22.0"
            else:
                base_url = self.base_url

            url = f"{base_url}/{endpoint}"

            try:
                logger.debug(
                    "Making API request",
                    method=method,
                    endpoint=endpoint,
                    params=params,
                )

                if method.upper() == "GET":
                    response = await self.client.get(url, params=params)
                elif method.upper() == "POST":
                    if data:
                        response = await self.client.post(url, params=params, json=data)
                    else:
                        response = await self.client.post(url, params=params)
                elif method.upper() == "DELETE":
                    response = await self.client.delete(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check for rate limiting
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded", endpoint=endpoint)
                    raise RateLimitExceeded("Instagram API rate limit exceeded")

                # Parse response
                response_data = response.json()

                # Check for API errors
                if "error" in response_data:
                    error = response_data["error"]
                    error_msg = error.get("message", "Unknown error")
                    error_code = error.get("code")
                    error_subcode = error.get("error_subcode")

                    logger.error(
                        "Instagram API error",
                        error_message=error_msg,
                        error_code=error_code,
                        error_subcode=error_subcode,
                    )

                    raise InstagramAPIError(error_msg, error_code, error_subcode)

                # Cache successful GET responses
                if method.upper() == "GET" and use_cache:
                    self._cache_response(cache_key, response_data)

                logger.debug("API request successful", endpoint=endpoint)
                return response_data

            except httpx.RequestError as e:
                logger.error("HTTP request failed", error=str(e), endpoint=endpoint)
                raise InstagramAPIError(f"HTTP request failed: {str(e)}")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON response", error=str(e))
                raise InstagramAPIError(f"Invalid JSON response: {str(e)}")

    async def get_profile_info(
        self, account_id: Optional[str] = None
    ) -> InstagramProfile:
        """Get Instagram business profile information."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id

        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        fields = [
            "id",
            "username",
            "name",
            "biography",
            "website",
            "profile_picture_url",
            "followers_count",
            "follows_count",
            "media_count",
        ]

        params = {"fields": ",".join(fields)}

        try:
            data = await self._make_request("GET", account_id, params=params)
            return InstagramProfile(**data)
        except Exception as e:
            logger.error("Failed to get profile info", error=str(e))
            raise InstagramAPIError(f"Failed to get profile info: {str(e)}")

    async def get_media_posts(
        self,
        account_id: Optional[str] = None,
        limit: int = 25,
        after: Optional[str] = None,
    ) -> List[InstagramMedia]:
        """Get recent media posts from Instagram account."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id

        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        fields = [
            "id",
            "media_type",
            "media_url",
            "permalink",
            "thumbnail_url",
            "caption",
            "timestamp",
            "like_count",
            "comments_count",
        ]

        params = {
            "fields": ",".join(fields),
            "limit": min(limit, 100),  # Instagram API limit
        }

        if after:
            params["after"] = after

        try:
            data = await self._make_request("GET", f"{account_id}/media", params=params)
            media_list = []

            for item in data.get("data", []):
                # Convert timestamp to datetime
                if "timestamp" in item:
                    item["timestamp"] = datetime.fromisoformat(
                        item["timestamp"].replace("Z", "+00:00")
                    )

                media_list.append(InstagramMedia(**item))

            return media_list

        except Exception as e:
            logger.error("Failed to get media posts", error=str(e))
            raise InstagramAPIError(f"Failed to get media posts: {str(e)}")

    async def get_media_insights(
        self, media_id: str, metrics: Optional[List[InsightMetric]] = None
    ) -> List[MediaInsight]:
        """Get insights for a specific media post."""
        if not metrics:
            metrics = [
                InsightMetric.REACH,
                InsightMetric.LIKES,
                InsightMetric.COMMENTS,
                InsightMetric.SHARES,
                InsightMetric.SAVED,
            ]

        params = {"metric": ",".join([m.value for m in metrics])}

        try:
            data = await self._make_request(
                "GET", f"{media_id}/insights", params=params
            )
            insights = []

            for item in data.get("data", []):
                insights.append(MediaInsight(**item))

            return insights

        except Exception as e:
            logger.error(
                "Failed to get media insights", error=str(e), media_id=media_id
            )
            raise InstagramAPIError(f"Failed to get media insights: {str(e)}")

    async def publish_media(self, request: PublishMediaRequest) -> PublishMediaResponse:
        """Publish media to Instagram account."""
        account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        try:
            # Validate image aspect ratio before API call
            if request.image_url:
                await self._validate_image_aspect_ratio(str(request.image_url))
            # Note: Video validation would require different logic (not implemented yet)

            # Step 1: Create media container
            container_data = {
                "caption": request.caption or "",
            }

            if request.image_url:
                container_data["image_url"] = str(request.image_url)
            elif request.video_url:
                container_data["video_url"] = str(request.video_url)
            else:
                raise InstagramAPIError("Either image_url or video_url is required")

            if request.location_id:
                container_data["location_id"] = request.location_id

            container_response = await self._make_request(
                "POST", f"{account_id}/media", data=container_data
            )

            container_id = container_response["id"]

            # Step 2: Publish the media
            publish_data = {"creation_id": container_id}
            publish_response = await self._make_request(
                "POST", f"{account_id}/media_publish", data=publish_data
            )

            return PublishMediaResponse(id=publish_response["id"], success=True)

        except Exception as e:
            logger.error("Failed to publish media", error=str(e))
            raise InstagramAPIError(f"Failed to publish media: {str(e)}")

    async def get_account_pages(self) -> List[FacebookPage]:
        """Get Facebook pages connected to the account."""
        params = {"fields": "id,name,instagram_business_account"}

        try:
            data = await self._make_request("GET", "me/accounts", params=params)
            pages = []

            for item in data.get("data", []):
                pages.append(FacebookPage(**item))

            return pages

        except Exception as e:
            logger.error("Failed to get account pages", error=str(e))
            raise InstagramAPIError(f"Failed to get account pages: {str(e)}")

    async def get_account_insights(
        self,
        account_id: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        period: InsightPeriod = InsightPeriod.DAY,
    ) -> List[AccountInsight]:
        """Get account-level insights."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id

        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        if not metrics:
            metrics = ["reach", "profile_views", "website_clicks"]

        params = {
            "metric": ",".join(metrics),
            "period": period.value,
            "metric_type": "total_value",  # Required for action metrics like website_clicks
        }

        try:
            data = await self._make_request(
                "GET", f"{account_id}/insights", params=params
            )
            insights = []

            for item in data.get("data", []):
                insights.append(AccountInsight(**item))

            return insights

        except Exception as e:
            logger.error("Failed to get account insights", error=str(e))
            raise InstagramAPIError(f"Failed to get account insights: {str(e)}")

    async def validate_access_token(self) -> bool:
        """Validate the access token."""
        try:
            await self._make_request(
                "GET", "me", params={"fields": "id"}, use_cache=False
            )
            return True
        except InstagramAPIError:
            return False

    async def get_conversations(
        self, page_id: Optional[str] = None, limit: int = 25
    ) -> List[InstagramConversation]:
        """
        Get Instagram DM conversations for a Facebook page.

        Note: Requires instagram_manage_messages permission.
        """
        if not page_id:
            # Try to get page ID from connected pages
            pages = await self.get_account_pages()
            if not pages:
                raise InstagramAPIError(
                    "No Facebook pages found. Please connect a Facebook page to your Instagram account."
                )
            page_id = pages[0].id
            logger.info(f"Using page ID: {page_id}")

        fields = "id,updated_time,message_count"
        params = {"platform": "instagram", "fields": fields, "limit": min(limit, 100)}

        try:
            data = await self._make_request(
                "GET",
                f"{page_id}/conversations",
                params=params,
                use_facebook_api=True,  # DMs use graph.facebook.com
            )
            conversations = []

            for item in data.get("data", []):
                conversations.append(InstagramConversation(**item))

            logger.info(f"Retrieved {len(conversations)} conversations")
            return conversations

        except InstagramAPIError as e:
            logger.error("Failed to get conversations", error=str(e))
            error_msg = str(e)
            # Detect Advanced Access permission error
            if (
                "#2" in error_msg
                or "unavailable" in error_msg.lower()
                or "temporarily" in error_msg.lower()
            ):
                error_msg += (
                    "\n\n⚠️  This error indicates that instagram_manage_messages permission "
                    "requires Advanced Access from Meta via App Review. "
                    "\n📖 See INSTAGRAM_DM_SETUP.md for the complete approval process."
                )
            raise InstagramAPIError(error_msg)
        except Exception as e:
            logger.error("Failed to get conversations", error=str(e))
            raise InstagramAPIError(f"Failed to get conversations: {str(e)}")

    async def get_conversation_messages(
        self, conversation_id: str, limit: int = 25
    ) -> List[InstagramMessage]:
        """
        Get messages from a specific Instagram DM conversation.

        Note: Requires instagram_manage_messages permission.
        """
        fields = "id,from,to,message,created_time,attachments"
        params = {"fields": f"messages{{" + fields + "}}", "limit": min(limit, 100)}

        try:
            data = await self._make_request(
                "GET",
                conversation_id,
                params=params,
                use_facebook_api=True,  # DMs use graph.facebook.com
            )
            messages = []

            for item in data.get("messages", {}).get("data", []):
                messages.append(InstagramMessage(**item))

            logger.info(
                f"Retrieved {len(messages)} messages from conversation {conversation_id}"
            )
            return messages

        except InstagramAPIError as e:
            logger.error(
                "Failed to get conversation messages",
                error=str(e),
                conversation_id=conversation_id,
            )
            error_msg = str(e)
            # Detect Advanced Access permission error
            if (
                "#2" in error_msg
                or "unavailable" in error_msg.lower()
                or "temporarily" in error_msg.lower()
            ):
                error_msg += (
                    "\n\n⚠️  This error indicates that instagram_manage_messages permission "
                    "requires Advanced Access from Meta via App Review. "
                    "\n📖 See INSTAGRAM_DM_SETUP.md for the complete approval process."
                )
            raise InstagramAPIError(error_msg)
        except Exception as e:
            logger.error(
                "Failed to get conversation messages",
                error=str(e),
                conversation_id=conversation_id,
            )
            raise InstagramAPIError(f"Failed to get conversation messages: {str(e)}")

    async def send_dm(self, request: SendDMRequest) -> SendDMResponse:
        """
        Send Instagram direct message.

        IMPORTANT:
        - Requires instagram_manage_messages permission (Advanced Access)
        - Can only reply within 24 hours of user's last message
        - Recipient must have initiated conversation first

        Note: This method may fail if Advanced Access is not approved.
        """
        message_data = {
            "recipient": {"id": request.recipient_id},
            "message": {"text": request.message},
        }

        try:
            data = await self._make_request(
                "POST",
                "me/messages",
                data=message_data,
                use_facebook_api=True,  # DMs use graph.facebook.com
            )

            return SendDMResponse(
                message_id=data.get("message_id", ""),
                recipient_id=request.recipient_id,
                success=True,
            )

        except InstagramAPIError as e:
            logger.error(
                "Failed to send DM", error=str(e), recipient=request.recipient_id
            )
            error_msg = str(e)
            # Detect Advanced Access permission error
            if (
                "#2" in error_msg
                or "unavailable" in error_msg.lower()
                or "temporarily" in error_msg.lower()
                or "permissions" in error_msg.lower()
                or "access" in error_msg.lower()
            ):
                error_msg += (
                    "\n\n⚠️  This error indicates that instagram_manage_messages permission "
                    "requires Advanced Access from Meta via App Review. "
                    "\n📖 See INSTAGRAM_DM_SETUP.md for the complete approval process."
                )
            raise InstagramAPIError(error_msg)
        except Exception as e:
            logger.error(
                "Failed to send DM", error=str(e), recipient=request.recipient_id
            )
            raise InstagramAPIError(f"Failed to send DM: {str(e)}")

    # ── Comment Methods ───────────────────────────────────────────

    async def get_comments(
        self,
        media_id: str,
        limit: int = 25,
    ) -> List[InstagramComment]:
        """Get comments on a media post."""
        fields = "id,text,timestamp,username,like_count,hidden"
        params = {
            "fields": fields,
            "limit": min(limit, 100),
        }

        try:
            data = await self._make_request(
                "GET", f"{media_id}/comments", params=params
            )
            comments = []
            for item in data.get("data", []):
                comments.append(InstagramComment(**item))
            return comments
        except Exception as e:
            logger.error("Failed to get comments", error=str(e), media_id=media_id)
            raise InstagramAPIError(f"Failed to get comments: {str(e)}")

    async def reply_to_comment(
        self,
        comment_id: str,
        message: str,
    ) -> InstagramComment:
        """Reply to a specific comment."""
        params = {"message": message}

        try:
            data = await self._make_request(
                "POST", f"{comment_id}/replies", data=params
            )
            return InstagramComment(id=data["id"], text=message)
        except Exception as e:
            logger.error(
                "Failed to reply to comment", error=str(e), comment_id=comment_id
            )
            raise InstagramAPIError(f"Failed to reply to comment: {str(e)}")

    async def post_comment(
        self,
        media_id: str,
        message: str,
    ) -> InstagramComment:
        """Post a top-level comment on a media post."""
        params = {"message": message}

        try:
            data = await self._make_request("POST", f"{media_id}/comments", data=params)
            return InstagramComment(id=data["id"], text=message)
        except Exception as e:
            logger.error("Failed to post comment", error=str(e), media_id=media_id)
            raise InstagramAPIError(f"Failed to post comment: {str(e)}")

    async def delete_comment(self, comment_id: str) -> bool:
        """Delete a comment."""
        try:
            await self._make_request("DELETE", comment_id)
            return True
        except Exception as e:
            logger.error(
                "Failed to delete comment", error=str(e), comment_id=comment_id
            )
            raise InstagramAPIError(f"Failed to delete comment: {str(e)}")

    async def hide_comment(self, comment_id: str, hide: bool = True) -> bool:
        """Hide or unhide a comment."""
        try:
            await self._make_request("POST", comment_id, data={"hide": hide})
            return True
        except Exception as e:
            action = "hide" if hide else "unhide"
            logger.error(
                f"Failed to {action} comment", error=str(e), comment_id=comment_id
            )
            raise InstagramAPIError(f"Failed to {action} comment: {str(e)}")

    # ── Hashtag Methods ──────────────────────────────────────────

    async def search_hashtag(
        self,
        hashtag_name: str,
        account_id: Optional[str] = None,
    ) -> InstagramHashtag:
        """Search for a hashtag ID by name."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        params = {
            "q": hashtag_name.lstrip("#"),
            "user_id": account_id,
        }

        try:
            data = await self._make_request("GET", "ig_hashtag_search", params=params)
            results = data.get("data", [])
            if not results:
                raise InstagramAPIError(f"Hashtag '{hashtag_name}' not found")
            return InstagramHashtag(**results[0])
        except InstagramAPIError:
            raise
        except Exception as e:
            logger.error("Failed to search hashtag", error=str(e), hashtag=hashtag_name)
            raise InstagramAPIError(f"Failed to search hashtag: {str(e)}")

    async def get_hashtag_media(
        self,
        hashtag_id: str,
        media_type: str = "top",
        account_id: Optional[str] = None,
        limit: int = 25,
    ) -> List[HashtagMedia]:
        """Get top or recent media for a hashtag."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        if media_type not in ("top", "recent"):
            raise InstagramAPIError("media_type must be 'top' or 'recent'")

        fields = "id,media_type,media_url,permalink,caption,timestamp,like_count,comments_count"
        params = {
            "user_id": account_id,
            "fields": fields,
            "limit": min(limit, 50),
        }

        try:
            endpoint = f"{hashtag_id}/{media_type}_media"
            data = await self._make_request("GET", endpoint, params=params)
            media_list = []
            for item in data.get("data", []):
                media_list.append(HashtagMedia(**item))
            return media_list
        except Exception as e:
            logger.error(
                "Failed to get hashtag media", error=str(e), hashtag_id=hashtag_id
            )
            raise InstagramAPIError(f"Failed to get hashtag media: {str(e)}")

    # ── Story Methods ────────────────────────────────────────────

    async def get_stories(
        self,
        account_id: Optional[str] = None,
    ) -> List[InstagramStory]:
        """Get current stories for the account."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        fields = "id,media_type,media_url,timestamp"
        params = {"fields": fields}

        try:
            data = await self._make_request(
                "GET", f"{account_id}/stories", params=params
            )
            stories = []
            for item in data.get("data", []):
                stories.append(InstagramStory(**item))
            return stories
        except Exception as e:
            logger.error("Failed to get stories", error=str(e))
            raise InstagramAPIError(f"Failed to get stories: {str(e)}")

    # ── Mention Methods ──────────────────────────────────────────

    async def get_mentions(
        self,
        account_id: Optional[str] = None,
        limit: int = 25,
    ) -> List[InstagramMention]:
        """Get recent mentions (posts/comments that @mention you)."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        fields = "id,media_type,media_url,permalink,caption,timestamp,username"
        params = {
            "fields": fields,
            "limit": min(limit, 100),
        }

        try:
            data = await self._make_request("GET", f"{account_id}/tags", params=params)
            mentions = []
            for item in data.get("data", []):
                mentions.append(InstagramMention(**item))
            return mentions
        except Exception as e:
            logger.error("Failed to get mentions", error=str(e))
            raise InstagramAPIError(f"Failed to get mentions: {str(e)}")

    # ── Business Discovery Methods ───────────────────────────────

    async def business_discovery(
        self,
        target_username: str,
        account_id: Optional[str] = None,
    ) -> BusinessDiscoveryProfile:
        """Look up another business/creator account's public profile."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        discovery_fields = (
            "username,name,biography,website,"
            "followers_count,follows_count,media_count,profile_picture_url"
        )
        params = {
            "fields": f"business_discovery.fields({discovery_fields}){{username,name,biography,website,followers_count,follows_count,media_count,profile_picture_url}}",
        }

        try:
            # Use the simpler approach: fields=business_discovery.fields(...)
            simple_params = {
                "fields": f"business_discovery.fields({discovery_fields})",
            }
            data = await self._make_request("GET", account_id, params=simple_params)
            bd = data.get("business_discovery", {})
            if not bd:
                raise InstagramAPIError(
                    f"Could not find business account: {target_username}"
                )
            return BusinessDiscoveryProfile(**bd)
        except InstagramAPIError:
            raise
        except Exception as e:
            logger.error(
                "Failed business discovery", error=str(e), target=target_username
            )
            raise InstagramAPIError(f"Failed business discovery: {str(e)}")

    # ── Carousel / Reels Publishing ──────────────────────────────

    async def publish_carousel(
        self,
        image_urls: List[str],
        caption: Optional[str] = None,
    ) -> PublishMediaResponse:
        """Publish a carousel (album) post with multiple images/videos."""
        account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        if len(image_urls) < 2:
            raise InstagramAPIError("Carousel requires at least 2 items")
        if len(image_urls) > 10:
            raise InstagramAPIError("Carousel supports maximum 10 items")

        try:
            # Step 1: Create individual item containers
            container_ids = []
            for url in image_urls:
                is_video = any(url.lower().endswith(ext) for ext in (".mp4", ".mov"))
                container_data = {
                    "is_carousel_item": "true",
                }
                if is_video:
                    container_data["video_url"] = url
                    container_data["media_type"] = "VIDEO"
                else:
                    container_data["image_url"] = url

                resp = await self._make_request(
                    "POST", f"{account_id}/media", data=container_data
                )
                container_ids.append(resp["id"])

            # Step 2: Create carousel container
            carousel_data = {
                "media_type": "CAROUSEL",
                "children": ",".join(container_ids),
            }
            if caption:
                carousel_data["caption"] = caption

            carousel_resp = await self._make_request(
                "POST", f"{account_id}/media", data=carousel_data
            )

            # Step 3: Publish
            publish_resp = await self._make_request(
                "POST",
                f"{account_id}/media_publish",
                data={"creation_id": carousel_resp["id"]},
            )
            return PublishMediaResponse(id=publish_resp["id"], success=True)

        except Exception as e:
            logger.error("Failed to publish carousel", error=str(e))
            raise InstagramAPIError(f"Failed to publish carousel: {str(e)}")

    async def publish_reel(
        self,
        video_url: str,
        caption: Optional[str] = None,
        share_to_feed: bool = True,
    ) -> PublishMediaResponse:
        """Publish a Reel."""
        account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        try:
            # Step 1: Create reel container
            container_data = {
                "media_type": "REELS",
                "video_url": video_url,
                "share_to_feed": str(share_to_feed).lower(),
            }
            if caption:
                container_data["caption"] = caption

            container_resp = await self._make_request(
                "POST", f"{account_id}/media", data=container_data
            )

            # Step 2: Publish
            publish_resp = await self._make_request(
                "POST",
                f"{account_id}/media_publish",
                data={"creation_id": container_resp["id"]},
            )
            return PublishMediaResponse(id=publish_resp["id"], success=True)

        except Exception as e:
            logger.error("Failed to publish reel", error=str(e))
            raise InstagramAPIError(f"Failed to publish reel: {str(e)}")

    # ── Content Publishing Limit ─────────────────────────────────

    async def get_content_publishing_limit(
        self,
        account_id: Optional[str] = None,
    ) -> ContentPublishingLimit:
        """Check how many more posts you can publish today."""
        if not account_id:
            account_id = self.settings.instagram_business_account_id
        if not account_id:
            raise InstagramAPIError("Instagram business account ID not configured")

        params = {"fields": "quota_usage,config,quota_duration"}

        try:
            data = await self._make_request(
                "GET", f"{account_id}/content_publishing_limit", params=params
            )
            # API returns {"data": [{ ... }]}
            items = data.get("data", [])
            if items:
                return ContentPublishingLimit(**items[0])
            return ContentPublishingLimit()
        except Exception as e:
            logger.error("Failed to get publishing limit", error=str(e))
            raise InstagramAPIError(f"Failed to get publishing limit: {str(e)}")

    def get_rate_limit_info(self) -> RateLimitInfo:
        """Get current rate limit information."""
        # This is a simplified implementation
        # In a real implementation, you'd track actual usage
        return RateLimitInfo(
            app_id=self.settings.facebook_app_id,
            call_count=0,  # Would track actual calls
            total_cputime=0,
            total_time=0,
        )
