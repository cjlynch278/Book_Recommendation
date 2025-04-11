import os
from dotenv import load_dotenv
load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str  

    class Config:
        env_file = ".env"  # Load variables from a .env file if applicable

settings = Settings()
