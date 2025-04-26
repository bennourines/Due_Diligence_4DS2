# filepath: DeployTrial2/generation/llm_clients.py
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# --- LLM Client Factory ---
# Store initialized clients to avoid re-creation on every request
_llm_clients = {}

def get_llm_client(provider: str = "openrouter") -> BaseChatModel:
    """
    Factory function to get an initialized LLM client based on configuration.
    Currently supports 'openrouter'. Caches client instances.

    Args:
        provider: The name of the LLM provider (e.g., "openrouter").

    Returns:
        An instance of BaseChatModel (e.g., ChatOpenAI).

    Raises:
        ValueError: If the provider is not supported.
        RuntimeError: If client initialization fails.
    """
    global _llm_clients

    # Use a key combining provider and relevant settings if config can change
    client_key = f"{provider}_{settings.LLM_MODEL_NAME}"

    if client_key in _llm_clients:
        logger.debug(f"Returning cached LLM client for key: {client_key}")
        return _llm_clients[client_key]

    logger.info(f"Initializing new LLM client for provider: {provider}")

    if provider.lower() == "openrouter":
        try:
            logger.info(f"Configuring ChatOpenAI for OpenRouter with model: {settings.LLM_MODEL_NAME}")
            llm = ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
                temperature=0.2, # Make configurable via settings? e.g., settings.LLM_TEMPERATURE
                max_tokens=1500, # Make configurable via settings? e.g., settings.LLM_MAX_TOKENS
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=settings.OPENROUTER_API_KEY,
                # Add streaming=True if needed later
                # timeout=settings.LLM_TIMEOUT # Add timeout from settings
            )
            # Optional: Test connection or basic call
            # llm.invoke("test")
            logger.info("OpenRouter LLM client initialized successfully.")
            _llm_clients[client_key] = llm # Cache the client
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter LLM client: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize OpenRouter LLM: {e}") from e

    # Add other providers here (e.g., OpenAI, Anthropic)
    # elif provider.lower() == "openai":
    #     try:
    #         from langchain_openai import ChatOpenAI as LangchainChatOpenAI # Avoid name clash
    #         logger.info(f"Configuring ChatOpenAI for OpenAI with model: {settings.OPENAI_MODEL_NAME}")
    #         llm = LangchainChatOpenAI(
    #             model=settings.OPENAI_MODEL_NAME,
    #             openai_api_key=settings.OPENAI_API_KEY,
    #             temperature=settings.LLM_TEMPERATURE,
    #             max_tokens=settings.LLM_MAX_TOKENS,
    #             # timeout=settings.LLM_TIMEOUT
    #         )
    #         logger.info("OpenAI LLM client initialized successfully.")
    #         _llm_clients[client_key] = llm
    #         return llm
    #     except Exception as e:
    #         logger.error(f"Failed to initialize OpenAI LLM client: {e}", exc_info=True)
    #         raise RuntimeError(f"Failed to initialize OpenAI LLM: {e}") from e

    else:
        logger.error(f"Unsupported LLM provider specified: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Convenience function to get the default OpenRouter client
def get_default_llm() -> BaseChatModel:
    """Gets the default LLM client (currently OpenRouter)."""
    return get_llm_client(provider="openrouter")
