import os
import sys
import asyncio
from dotenv import load_dotenv
from research_and_analyst.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from research_and_analyst.logger import GLOBAL_LOGGER as log
from research_and_analyst.exception.custom_exception import ResearchAnalystException


class ApiKeyManager:
    """
    Loads and manages only the Google API key.
    """

    def __init__(self):
        load_dotenv()
        self.api_keys = {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
        }

        log.info("Initializing ApiKeyManager (Google-only)")

        if not self.api_keys["GOOGLE_API_KEY"]:
            log.error("GOOGLE_API_KEY is missing in environment variables")
            raise ResearchAnalystException("Missing required API key: GOOGLE_API_KEY", sys)
        else:
            log.info("GOOGLE_API_KEY loaded successfully from environment")

    def get(self, key: str):
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads Google embedding models and Google LLMs dynamically based on configuration.
    """

    def __init__(self):
        try:
            self.api_key_mgr = ApiKeyManager()
            self.config = load_config()
            log.info("YAML configuration loaded successfully", config_keys=list(self.config.keys()))
        except Exception as e:
            log.error("Error initializing ModelLoader", error=str(e))
            raise ResearchAnalystException("Failed to initialize ModelLoader", sys)

    # ----------------------------------------------------------------------
    # Embedding Loader
    # ----------------------------------------------------------------------
    def load_embeddings(self):
        """
        Load and return a Google Generative AI embedding model.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            # Ensure event loop exists for gRPC-based embedding API
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"), # pyright: ignore[reportCallIssue]
            )

            log.info("Embedding model loaded successfully", model=model_name)
            return embeddings

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ResearchAnalystException("Failed to load embedding model", sys)

    # ----------------------------------------------------------------------
    # LLM Loader
    # ----------------------------------------------------------------------
    def load_llm(self):
        """
        Load and return a Google chat-based LLM.
        """
        try:
            llm_config = self.config["llm"]["google"]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading Google LLM", provider=provider, model=model_name)

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            log.info("Google LLM loaded successfully", provider=provider, model=model_name)
            return llm

        except Exception as e:
            log.error("Error loading Google LLM", error=str(e))
            raise ResearchAnalystException("Failed to load Google LLM", sys)


# ----------------------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        loader = ModelLoader()

        # Test LLM
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")
        result = llm.invoke("Hello, how are you?")
        print(f"LLM Result: {result.content[:200]}")

        log.info("Google-only ModelLoader test completed successfully")

    except ResearchAnalystException as e:
        log.error("Critical failure in ModelLoader test", error=str(e))