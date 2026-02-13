"""LLM interface for Ollama with GPU queue support and per-task model routing."""
import sys
import json
import logging
import requests
from typing import Optional, Dict, Any, List
from config import MODELS, OLLAMA_BASE_URL

# GPU Queue integration
try:
    sys.path.insert(0, "/home/andus/.openclaw/workspace/gpu-queue")
    from queue_manager import ollama_query, gpu_queue
    GPU_QUEUE_AVAILABLE = True
except ImportError:
    GPU_QUEUE_AVAILABLE = False
    logging.warning("GPU queue not available, falling back to direct Ollama calls")

logger = logging.getLogger(__name__)

# Valid task types for model routing
TASK_TYPES = ("planning", "conversation", "reflection", "importance", "default")

# Display callback — set by main.py to update UI with LLM status
_llm_status_callback = None

def set_llm_status_callback(callback):
    """Set a callback that receives (agent, task, model) on each LLM call."""
    global _llm_status_callback
    _llm_status_callback = callback

def _notify_llm_status(agent: str, task: str, model: str):
    """Notify the display about current LLM usage."""
    if _llm_status_callback:
        try:
            _llm_status_callback(agent, task, model)
        except Exception:
            pass

def get_model_for_task(task: str) -> str:
    """Get the appropriate model for a given task type."""
    return MODELS.get(task, MODELS["default"])

class OllamaClient:
    """Client for interacting with Ollama models with per-task routing."""
    
    def __init__(self, use_gpu_queue: bool = True):
        self.base_url = OLLAMA_BASE_URL
        self.use_gpu_queue = use_gpu_queue and GPU_QUEUE_AVAILABLE
        logger.info(f"Model routing: {', '.join(f'{k}={v}' for k, v in MODELS.items())}")
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.7, max_tokens: int = 512,
                      task: str = "default", agent_name: str = "") -> str:
        """Generate text using the task-appropriate model."""
        model = get_model_for_task(task)
        _notify_llm_status(agent_name, task, model)
        try:
            if self.use_gpu_queue:
                return await self._generate_with_queue(prompt, system_prompt, temperature, max_tokens, model)
            else:
                return await self._generate_direct(prompt, system_prompt, temperature, max_tokens, model)
        except Exception as e:
            logger.error(f"Error generating text ({task}/{model}): {e}")
            return ""
    
    async def _generate_with_queue(self, prompt: str, system_prompt: Optional[str],
                                  temperature: float, max_tokens: int, model: str) -> str:
        """Generate using GPU queue."""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = ollama_query(full_prompt, model=model)
            return response.strip()
        except Exception as e:
            logger.error(f"GPU queue generation failed ({model}): {e}")
            return await self._generate_direct(prompt, system_prompt, temperature, max_tokens, model)
    
    async def _generate_direct(self, prompt: str, system_prompt: Optional[str],
                              temperature: float, max_tokens: int, model: str) -> str:
        """Generate using direct Ollama API."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Direct Ollama generation failed ({model}): {e}")
            return ""
    
    async def score_importance(self, observation: str, agent_name: str = "") -> int:
        """Score the importance of an observation (1-10). Uses fast/small model."""
        prompt = f"""On a scale of 1-10 where 1 is mundane (e.g., brushing teeth) and 10 is life-changing (e.g., getting married), rate the importance of this observation:

{observation}

Respond with only a number from 1 to 10."""

        response = await self.generate(prompt, temperature=0.3, max_tokens=5, task="importance", agent_name=agent_name)
        try:
            score = int(response.strip())
            return max(1, min(10, score))  # Clamp to 1-10
        except ValueError:
            logger.warning(f"Failed to parse importance score: {response}")
            return 5  # Default moderate importance
    
    async def generate_reflection_questions(self, recent_memories: List[str], agent_name: str = "") -> List[str]:
        """Generate 3 salient high-level questions from recent memories."""
        memories_text = "\n".join(f"- {memory}" for memory in recent_memories)
        
        prompt = f"""Given only the information above, what are the 3 most salient high-level questions we can ask about the person's recent experiences and activities?

Recent observations:
{memories_text}

Respond with exactly 3 questions, one per line, starting each with a number (1., 2., 3.)."""

        response = await self.generate(prompt, temperature=0.8, max_tokens=200, task="reflection", agent_name=agent_name)
        
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                questions.append(line[2:].strip())
        
        return questions[:3]  # Ensure only 3 questions
    
    async def generate_reflection(self, question: str, relevant_memories: List[str], agent_name: str = "") -> str:
        """Generate a reflection based on a question and relevant memories."""
        memories_text = "\n".join(f"- {memory}" for memory in relevant_memories)
        
        prompt = f"""Based on the following memories, provide a thoughtful reflection addressing this question: {question}

Relevant memories:
{memories_text}

Write a 2-3 sentence reflection that synthesizes insights from these memories."""

        return await self.generate(prompt, temperature=0.7, max_tokens=150, task="reflection", agent_name=agent_name)
    
    async def generate_daily_plan(self, agent_name: str, agent_description: str, 
                                 current_date: str) -> str:
        """Generate a daily plan for an agent."""
        prompt = f"""{agent_description}

Here is {agent_name}'s plan for {current_date}:
1) wake up and complete the morning routine at 6:00 am
2) """

        response = await self.generate(prompt, temperature=0.8, max_tokens=300, task="planning")
        return response
    
    async def decompose_plan_item(self, agent_name: str, plan_item: str, 
                                 duration_minutes: int) -> List[str]:
        """Decompose a plan item into 5-15 minute actions."""
        prompt = f"""Break down this {duration_minutes}-minute activity for {agent_name} into specific 5-15 minute actions:

Activity: {plan_item}

List the specific actions in chronological order. Each action should be 5-15 minutes and include what {agent_name} is doing and where."""

        response = await self.generate(prompt, temperature=0.7, max_tokens=250, task="planning")
        
        actions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        any(line.startswith(f"{i})") for i in range(1, 20))):
                # Clean up the line
                if line.startswith(('-', '•')):
                    action = line[1:].strip()
                else:
                    action = line.split(')', 1)[1].strip() if ')' in line else line
                actions.append(action)
        
        return actions
    
    async def should_initiate_conversation(self, agent_name: str, other_agent: str,
                                          context: str, agent_memories: List[str]) -> bool:
        """Determine if an agent should initiate conversation."""
        memories_text = "\n".join(f"- {memory}" for memory in agent_memories[-5:])  # Recent memories
        
        prompt = f"""Should {agent_name} initiate a conversation with {other_agent}?

Context: {context}

{agent_name}'s recent memories:
{memories_text}

Consider: Are they likely to want to talk? Do they have something to discuss? Is it an appropriate time/place?

Respond with only YES or NO."""

        response = await self.generate(prompt, temperature=0.6, max_tokens=5, task="conversation", agent_name=agent_name)
        return response.strip().upper() == "YES"
    
    async def generate_conversation_response(self, agent_name: str, other_agent: str,
                                           conversation_history: List[str],
                                           agent_memories: List[str]) -> str:
        """Generate a conversation response."""
        history_text = "\n".join(conversation_history)
        memories_text = "\n".join(f"- {memory}" for memory in agent_memories[-10:])
        
        prompt = f"""You are {agent_name}. Continue this conversation with {other_agent}:

{history_text}

Your recent memories:
{memories_text}

Respond naturally as {agent_name} would. Keep it conversational and brief (1-2 sentences)."""

        return await self.generate(prompt, temperature=0.9, max_tokens=100, task="conversation", agent_name=agent_name)

# Global client instance
llm_client = OllamaClient()

async def get_llm_client(use_gpu_queue: bool = True) -> OllamaClient:
    """Get the global LLM client."""
    global llm_client
    if llm_client.use_gpu_queue != use_gpu_queue:
        llm_client = OllamaClient(use_gpu_queue=use_gpu_queue)
    return llm_client

def print_model_routing():
    """Print the current model routing table."""
    print("\n🧠 Model Routing:")
    print(f"  Planning:      {MODELS['planning']}")
    print(f"  Conversation:  {MODELS['conversation']}")
    print(f"  Reflection:    {MODELS['reflection']}")
    print(f"  Importance:    {MODELS['importance']}")
    print(f"  Default:       {MODELS['default']}")
    print()