from typing import List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
)
from langchain.llms.fake import FakeListLLM


class AsyncFakeListLLM(FakeListLLM):
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        response = self.responses[self.i]
        self.i += 1
        return response
