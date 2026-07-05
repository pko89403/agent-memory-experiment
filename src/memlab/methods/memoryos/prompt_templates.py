"""MemoryOS 프롬프트 템플릿 — 순수 상수만 (로직은 llm_ops.py).

출처: 논문에는 프롬프트가 없어 원본 구현체에서 차용했다.
- 이어짐 판단·meta 요약: eval/dynamic_update.py
- 키워드·세그먼트 요약: eval/utils.py
- 90차원 프로필·지식 추출: memoryos-pypi/prompts.py
- 답변 생성: eval/main_loco_parse.py (간결·날짜 형식 규칙은 F1 채점과
  직결되므로 문구 유지. 원본의 오타만 교정)

차용분의 라이선스는 Apache-2.0 — 전문은 licenses/MemoryOS-Apache-2.0.txt.
"""

# ── STM: dialogue chain (논문 식 1) ─────────────────────────────────

CONTINUITY_SYSTEM = "You are a conversation continuity detector."

CONTINUITY_USER = """\
Determine if these two conversation pages are continuous \
(true continuation without topic shift).

Previous Page:
User: {prev_user}
Assistant: {prev_agent}

Current Page:
User: {curr_user}
Assistant: {curr_agent}

Respond in JSON: {{"continuous": true|false}}"""

CHAIN_SUMMARY_SYSTEM = (
    "You are a conversation meta-summary writer. Output ONLY the summary."
)

CHAIN_SUMMARY_USER = """\
Summarize the conversation flow so far in a concise meta-summary.

Guidelines:
1. Cover the whole conversation below
2. Keep it concise (1-2 sentences max)
3. Output ONLY the summary (no explanations)

Conversation:
{dialogue}

Meta-summary:"""

# ── MTM: 키워드(식 3의 K)와 세그먼트 요약 ──────────────────────────

KEYWORDS_SYSTEM = (
    "You are a keyword extraction expert. Please extract the keywords "
    "of the conversation topic."
)

KEYWORDS_USER = """\
Please extract the keywords of the conversation topic from the following \
dialogue, and do not exceed three.
Respond in JSON: {{"keywords": ["...", "..."]}}

{text}"""

SEGMENT_SUMMARY_SYSTEM = (
    "You are an expert in summarizing dialogue topics, please generate "
    "a concise and precise summary."
)

SEGMENT_SUMMARY_USER = """\
Please generate a topic summary based on the following conversation:
{dialogue}

Subject Summary:"""

# ── LPM: 90차원 프로필 (Li et al. 2025) & 지식 추출 ────────────────

PERSONALITY_DIMENSIONS = """\
[Psychological Model (Basic Needs & Personality)]
Extraversion: Preference for social activities.
Openness: Willingness to embrace new ideas and experiences.
Agreeableness: Tendency to be friendly and cooperative.
Conscientiousness: Responsibility and organizational ability.
Neuroticism: Emotional stability and sensitivity.
Physiological Needs: Concern for comfort and basic needs.
Need for Security: Emphasis on safety and stability.
Need for Belonging: Desire for group affiliation.
Need for Self-Esteem: Need for respect and recognition.
Cognitive Needs: Desire for knowledge and understanding.
Aesthetic Appreciation: Appreciation for beauty and art.
Self-Actualization: Pursuit of one's full potential.
Need for Order: Preference for cleanliness and organization.
Need for Autonomy: Preference for independent decision-making and action.
Need for Power: Desire to influence or control others.
Need for Achievement: Value placed on accomplishments.

[AI Alignment Dimensions]
Helpfulness: Whether the AI's response is practically useful to the user.
Honesty: Whether the AI's response is truthful.
Safety: Avoidance of sensitive or harmful content.
Instruction Compliance: Strict adherence to user instructions.
Truthfulness: Accuracy and authenticity of content.
Coherence: Clarity and logical consistency of expression.
Complexity: Preference for detailed and complex information.
Conciseness: Preference for brief and clear responses.

[Content Platform Interest Tags]
Science Interest: Interest in science topics.
Education Interest: Concern with education and learning.
Psychology Interest: Interest in psychology topics.
Family Concern: Interest in family and parenting.
Fashion Interest: Interest in fashion topics.
Art Interest: Engagement with or interest in art.
Health Concern: Concern with physical health and lifestyle.
Financial Management Interest: Interest in finance and budgeting.
Sports Interest: Interest in sports and physical activity.
Food Interest: Passion for cooking and cuisine.
Travel Interest: Interest in traveling and exploring new places.
Music Interest: Interest in music appreciation or creation.
Literature Interest: Interest in literature and reading.
Film Interest: Interest in movies and cinema.
Social Media Activity: Frequency and engagement with social media.
Tech Interest: Interest in technology and innovation.
Environmental Concern: Attention to environmental and sustainability issues.
History Interest: Interest in historical knowledge and topics.
Political Concern: Interest in political and social issues.
Religious Interest: Interest in religion and spirituality.
Gaming Interest: Enjoyment of video games or board games.
Animal Concern: Concern for animals or pets.
Emotional Expression: Preference for direct vs. restrained emotional expression.
Sense of Humor: Preference for humorous or serious communication style.
Information Density: Preference for detailed vs. concise information.
Language Style: Preference for formal vs. casual tone.
Practicality: Preference for practical advice vs. theoretical discussion."""

PERSONALITY_SYSTEM = (
    "You are a user profile analysis engine. Output only the user profile."
)

PERSONALITY_USER = """\
Please analyze the latest user-AI conversation below and update the user \
profile based on the 90 personality preference dimensions.

Here are the 90 dimensions and their explanations:

{dimensions}

**Task Instructions:**
1. Review the existing user profile below
2. Analyze the new conversation for evidence of the 90 dimensions above
3. Update and integrate the findings into a comprehensive user profile
4. For each dimension that can be identified, use the format: \
Dimension ( Level(High/Medium/Low) )
5. Include brief reasoning for each dimension when possible
6. Maintain existing insights from the old profile while incorporating new \
observations
7. If a dimension cannot be inferred from either the old profile or new \
conversation, do not include it
8. Keep the complete updated profile under 600 words — keep only the most \
strongly evidenced dimensions

**Existing User Profile:**
{existing_profile}

**Latest User-AI Conversation:**
{conversation}

**Updated User Profile:**"""

KNOWLEDGE_SYSTEM = (
    "You are a knowledge extraction assistant. Your task is to extract "
    "user private data and assistant knowledge from conversations. "
    "Be extremely concise and factual in your extractions."
)

KNOWLEDGE_USER = """\
Please extract user private data and assistant knowledge from the latest \
user-AI conversation below.

Latest User-AI Conversation:
{dialogue}

user_facts: personal information about the user. Be extremely concise - \
shortest possible phrases, each including entities and time.
assistant_knowledge: what the assistant demonstrated, format \
"Assistant [action] at [time]". Be extremely brief.
If nothing found, use an empty list.
Respond in JSON: {{"user_facts": ["..."], "assistant_knowledge": ["..."]}}"""

# ── 답변 생성 (논문 3.4: 세 tier 통합) ─────────────────────────────

ANSWER_SYSTEM = """\
You are role-playing as {speaker_b} in a conversation with {speaker_a}. \
Here are some of your character traits and knowledge:
{assistant_knowledge}

Any content referring to 'User' in the prompt refers to {speaker_a}'s \
content, and any content referring to 'AI' or 'assistant' refers to \
{speaker_b}'s content. \
Your task is to answer questions about {speaker_a} or {speaker_b} in an \
extremely concise manner.
When the question is: "What did the charity race raise awareness for?", \
you should not answer in the form of: "The charity race raised awareness \
for mental health." Instead, it should be: "mental health", as this is \
more concise."""

ANSWER_USER = """\
<CONTEXT>
Recent conversation between {speaker_a} and {speaker_b}:
{history}

<MEMORY>
Relevant past conversations:
{retrieval}

<CHARACTER TRAITS>
Characteristics of {speaker_a}:
{background}

the question is: {question}
Your task is to answer questions about {speaker_a} or {speaker_b} in an \
extremely concise manner.
Please only provide the content of the answer, without including 'answer:'
For questions that require answering a date or time, strictly follow the \
format "15 July 2023" and provide a specific date whenever possible. For \
example, if you need to answer "last year," give the specific year of last \
year rather than just saying "last year." Only provide one year, date, or \
time, without any extra responses.
If the question is about the duration, answer in the form of several years, \
months, or days.
Generate answers primarily composed of concrete entities, such as Mentoring \
program, school speech, etc"""
