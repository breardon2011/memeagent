from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, ToolMetadata, ToolSelection
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
import requests
from datetime import datetime


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class RouterInputEvent(Event):
    input: list[ChatMessage]

class Scrape(Event):
    input: list[ChatMessage]

class Save(Event): 
    input: list[ChatMessage]

class AgentFlow(Workflow):
    def __init__(self, llm, timeout=300):
        super().__init__(timeout=timeout)
        self.llm = llm
        self.memory = ChatMemoryBuffer(token_limit=1000).from_defaults(llm=llm)
        self.tools = []

    @step
    async def prepare_agent(self, ev: StartEvent) -> RouterInputEvent:
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        chat_history = self.memory.get()
        return RouterInputEvent(input=chat_history)
    


class ScrapeFlow(Workflow): 
    def __init__(self, llm, timeout=300):
        super().__init__(timeout=timeout)
        self.llm = llm
        self.memory = ChatMemoryBuffer(token_limit=1000).from_defaults(llm=llm)
        self.tools = []
        self.urls = []

    @step 
    async def scrape_memes(self, ev: Scrape) -> Save:
        # Process each URL in the list
        for url in self.urls:
            try:
                # Download image from URL
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Convert image to bytes
                    image_bytes = response.content
                    
                    
                    # Store the processed image data
                    self.meme_data.append({
                        'url': url,
                        'image_data': image_bytes,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                continue
        
        return Save(input=ev.input)
    


# ToolNode to save memes to RAG memory
