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

PLAN_DECOMPOSITION_PROMPT = """Break down this {duration_minutes}-minute activity for {agent_name} into specific 5-15 minute actions:

Activity: {plan_item}
Location: {location}

List the specific actions in chronological order. Each action should be 5-15 minutes and include what {agent_name} is doing and where they are. Format as a numbered list."""

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

OBSERVATION_IMPORTANCE_CONTEXT = """Consider the following factors when rating importance:
- Personal relevance to the agent
- Emotional impact  
- Social significance
- Practical implications
- Uniqueness or novelty
- Potential future consequences

Mundane activities (1-3): routine tasks, normal daily activities
Moderate events (4-6): interesting encounters, minor problems or achievements  
Important events (7-8): significant social interactions, major decisions, emotional moments
Life-changing events (9-10): major life changes, traumatic events, profound realizations"""

PLANNING_CONTEXT_PROMPT = """{agent_name} is planning their day. Consider:
- Their typical routine and habits
- Their goals and responsibilities  
- Social commitments and relationships
- Current projects or concerns
- Their personality and preferences

The plan should be realistic and reflect their character."""

MEMORY_QUERY_PROMPT = """Retrieve memories relevant to: {query}

Consider memories that relate to:
- Direct mentions of the topic
- Related concepts and themes
- Emotional associations
- Recent experiences that connect
- People involved in similar situations"""

AGENT_STATUS_PROMPT = """Based on {agent_name}'s recent activities and memories, what is their current:
1) Emotional state
2) Primary focus or concern
3) Social situation
4) Physical location and activity

Recent memories:
{recent_memories}

Provide a brief assessment (2-3 sentences)."""

RELATIONSHIP_ASSESSMENT_PROMPT = """Based on these interactions, how would you describe the relationship between {agent1} and {agent2}?

Shared memories and interactions:
{shared_memories}

Consider: closeness, trust, common interests, frequency of interaction, emotional tone.
Describe in 1-2 sentences."""

ACTION_FEASIBILITY_PROMPT = """Is this action feasible for {agent_name} at {current_time}?

Planned action: {action}
Current location: {current_location}  
Planned location: {planned_location}

Consider:
- Travel time between locations
- Whether the location is appropriate for the activity
- Time constraints
- Physical requirements

Respond with YES or NO and a brief reason."""

CONVERSATION_ENDING_PROMPT = """Should this conversation between {agent1} and {agent2} end now?

Current conversation:
{conversation_history}

Consider:
- Natural conversational flow
- Whether the topic has been exhausted
- Social cues and appropriateness
- Other commitments or activities

Respond with YES or NO."""

EMOTION_ASSESSMENT_PROMPT = """What emotion is {agent_name} likely feeling based on this situation?

Situation: {situation}
Recent memories: {recent_memories}
Personality: {personality}

Choose from: happy, sad, excited, anxious, angry, content, surprised, confused, determined, tired

Respond with only the emotion word."""

GOAL_EXTRACTION_PROMPT = """What are {agent_name}'s current goals or objectives based on their recent activities?

Recent memories:
{recent_memories}

List 2-3 specific, actionable goals that seem important to them right now."""

LOCATION_PREFERENCE_PROMPT = """Where would {agent_name} prefer to be for this activity: {activity}

Available locations: {available_locations}

Consider:
- The nature of the activity
- Their preferences and habits
- Social appropriateness
- Practical requirements

Respond with the most suitable location."""