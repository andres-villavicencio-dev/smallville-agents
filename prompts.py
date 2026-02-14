"""Prompt templates for generative agents, faithful to the original paper."""

IMPORTANCE_SCORING_PROMPT = """On a scale of 1-10 where 1 is mundane (e.g., brushing teeth) and 10 is life-changing (e.g., getting married, death of a family member), rate the importance of the following observation:

{observation}

Respond with only a number from 1 to 10."""

REFLECTION_QUESTIONS_PROMPT = """Given only the information above, what are the 3 most salient high-level questions we can ask about {agent_name}'s recent experiences and activities?

Recent observations:
{recent_memories}

Respond with exactly 3 questions, one per line, starting each with a number (1., 2., 3.)."""

REFLECTION_GENERATION_PROMPT = """Based on the following memories, provide a thoughtful reflection addressing this question: {question}

Relevant memories:
{relevant_memories}

Write a 2-3 sentence reflection that synthesizes insights from these memories. Start with "It seems like" or "I've been noticing that" or similar reflective language."""

DAILY_PLANNING_PROMPT = """{agent_description}

Here is {agent_name}'s plan for {date}:
1) wake up and complete the morning routine at 6:00 am
2) """

CONVERSATION_INITIATION_PROMPT = """Should {agent_name} initiate a conversation with {other_agent}?

Context: {context}

{agent_name}'s recent memories:
{agent_memories}

Consider:
- Are they likely to want to talk based on their personality and current situation?
- Do they have something specific to discuss?
- Is it an appropriate time and place for conversation?
- Would this interaction make sense given their relationship?

Respond with only YES or NO."""

CONVERSATION_RESPONSE_PROMPT = """You are {agent_name}. Continue this conversation with {other_agent}:

{conversation_history}

Your relevant memories:
{agent_memories}

Your personality: {agent_personality}

Respond naturally as {agent_name} would. Keep it conversational and brief (1-2 sentences). Stay in character and consider your relationship with {other_agent}."""

CONVERSATION_TOPIC_PROMPT = """What would {agent_name} want to talk about with {other_agent} given their current situation?

{agent_name}'s recent memories:
{agent_memories}

Current context: {context}

Suggest 1-2 specific topics {agent_name} might bring up. Consider their personality and what's currently important to them."""

CONVERSATION_ENDING_PROMPT = """Should this conversation between {agent1} and {agent2} end now?

Current conversation:
{conversation_history}

Consider:
- Natural conversational flow
- Whether the topic has been exhausted
- Social cues and appropriateness
- Other commitments or activities

Respond with YES or NO."""
