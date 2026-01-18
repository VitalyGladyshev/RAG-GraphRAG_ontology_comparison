import openai
from openai import AsyncOpenAI
import logging
from typing import Optional

class UniversalLLMClient:
    """Универсальный клиент для работы с LLM через OpenAI-совместимый API."""
    
    def __init__(self,
                 model: str,
                 api_key: str,
                 api_base: str = "https://api.openai.com/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 timeout: int = 120):
        """
        Инициализация LLM клиента.
        
        Args:
            model: Название модели
            api_key: API ключ
            api_base: Базовый URL API
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            timeout: Таймаут запроса в секундах
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Создание синхронного клиента OpenAI
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=3
        )
        
        # Создание асинхронного клиента OpenAI
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=3,
            default_headers={
                "X-Max-Tokens": str(max_tokens)
            }
        )
        
        self.logger.info(f"LLM клиент инициализирован: {model} @ {api_base}")
        self.logger.info(f"Параметры: max_tokens={max_tokens}, temperature={temperature}")
    
    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Генерация текста на основе промпта.
        
        Args:
            prompt: Основной промпт пользователя
            system_prompt: Системный промпт (опционально)
            temperature: Температура генерации (опционально)
            max_tokens: Максимальное количество токенов (опционально)
        
        Returns:
            Сгенерированный текст
        
        Raises:
            Exception: При ошибках генерации
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации: {e}")
            raise