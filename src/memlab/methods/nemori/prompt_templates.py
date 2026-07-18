"""Nemori 파이프라인의 프롬프트 — 순수 상수만.

출처 규칙: 논문이 명세서다. 논문 v4 Appendix D에 7종 전문이 실려 있어
전부 논문에서 차용한다 (원본 repo external/nemori는 MIT, Copyright
nemori-ai — 대조 참고용으로만 씀).

    상수            논문 출처                       호출 방식
    ─────────────  ─────────────────────────────  ─────────────────
    PARTITION       D.1.1 (Ppar)                   structured
    NARRATIVE       D.1.2 (Pnar)                   structured
    SELECT_TARGET   D.1.3 (Psel)                   structured
    INTEGRATE       D.1.4 (Pint)                   structured
    ANTICIPATE      D.1.5 (Pant)                   자유 텍스트
    DISTILL         D.1.6 (Pdis)                   structured
    DIRECT_DISTILL  D.2 (NEMORI-s 폴백)            structured
    ANSWER          D.3 (Pans)                     자유 텍스트

structured output 변형 (2026-07-17 합의 — "그대로 가져오되 structured
output 활용 부분은 과감하게 변형"):
- JSON 예시 블록·"Return only the JSON object" 강제문·마크다운 펜스
  대비 지시를 제거했다 — 형식은 llm_ops.py의 Pydantic json_schema가
  강제하고, 프롬프트에는 필드의 **의미** 서술만 남긴다. 로컬 소형
  모델에서 프롬프트 자체 형식 지시와 schema 강제가 중복되면 서로
  충돌하는 것을 MemoryOS·Zep 재현에서 실측했다.
- Psel: 원본 코드는 후보의 UUID 문자열 반환을 요구하지만 논문 형식화는
  idx ∈ {1,...,Ke} ∪ {-1} (§3.2.3) — 논문을 따라 후보 번호 선택으로
  바꿨다. 로컬 9B가 UUID를 오전사할 위험도 함께 제거된다.
- Pnar 출력 키: 논문 본문은 episodic_cue/narrative_episode, 논문 예시와
  원본 코드는 title/content로 서로 어긋난다 → 논문 본문 용어로 통일
  (Pint도 동일 키 사용, llm_ops의 응답 모델과 일치).
- ANTICIPATE·ANSWER는 자유 텍스트 유지 — 예측은 서사 산문이 목적이라
  schema가 오히려 방해 (원본 커밋 5a4ee4f "don't use json_object for
  prediction"과 같은 결론).

논문이 침묵하거나 결함이 있어 정한 것:
- DISTILL은 논문판을 쓴다. 원본 코드의 같은 단계 프롬프트는 "statement에
  날짜를 넣지 마라" + 7개 카테고리 체계가 붙은 다른 버전인데, 논문판
  예시는 날짜 포함을 허용한다 ("promoted to team lead in 2022") —
  temporal 질문은 날짜 있는 fact가 유리하므로 논문 우선이 실리와도 맞다.
- DIRECT_DISTILL(D.2)은 논문에선 ablation(NEMORI-s) 전용이지만, 원본
  코드는 semantic DB가 빈 cold start에서도 이 프롬프트로 분기한다 —
  빈 지식으로 예측하는 것은 무의미한 2배 비용이라 코드 배선을 차용
  (2026-07-17 합의, method.py에서 분기).
- 예시 속 "Caroline's favorite book is 'Becoming Nicole'..."은 LoCoMo
  화자·정답 자체다 (D.2 원문 그대로). 벤치마크 누출 소지가 있는 예시를
  논문이 싣고 있음을 기록해둔다 — 재현 충실성을 위해 자구 유지.
- 30분 갭(PARTITION)·1시간 갭(SELECT_TARGET) 상수는 프롬프트 자구에
  박힌 채 둔다 — 논문 자구 우선, Config로 빼지 않는다.
- PARTITION에 "Every message number from 1 to {count} must belong to
  exactly one episode" 한 줄을 신설했다 — §3.2.1 형식화는 partition이
  index 집합 전체를 커버할 것을 요구하는데 부록 프롬프트에는 그 조항이
  없다 (논문 내부 불일치). 프롬프트로 요구하고, 그래도 새는 index는
  method.py가 직전 그룹 편입으로 복구한다 (2026-07-17 합의).
- SELECT_TARGET에는 새 episode의 Title 슬롯이 없다 — §3.2.3 형식화는
  Psel 입력에 (c_j, N_j) 둘 다 넣지만 부록 D.1.3 프롬프트에는 Content
  (narrative)뿐이다 (논문 내부 불일치). 부록 자구를 따른다.
- INTEGRATE의 "Combined Event Details" 슬롯은 두 서사의 재복제다 —
  원본(merger.py:122)이 "Original: ...\n\nNew: ..."로 채우며 내용이
  프롬프트에 두 번 들어간다. 중복이지만 부록 D.1.4 자구라 유지한다.
- 시스템 메시지는 쓰지 않는다 (system="") — 논문 부록이 단일 프롬프트
  형태다. 원본 코드는 episode 생성에만 system을 넣는데 user 프롬프트
  첫 줄과 중복이라 채택하지 않는다.
- 논문 프롬프트의 줄바꿈은 2단 조판 아티팩트라 자연 줄로 정리했다.
  자구(단어·구두점)는 그대로다.
- placeholder 이름은 str.format() 관례로 통일. 값의 직렬화 형식
  (메시지 나열이 번호냐 줄글이냐)은 llm_ops.py 소관.
"""

# --- Local Message Partitioning (Sec 3.2.1 / Appendix D.1.1) ---

PARTITION = """\
You are an intelligent conversation segmentation expert. Your task is to analyze a batch of messages and group them into coherent episodes.

You will receive {count} messages numbered from 1 to {count}:

{messages}

## Your Task
Analyze these messages and group them into coherent episodes with **HIGH SENSITIVITY** to topic shifts. Be strict and create NEW episodes when detecting:

1. **Topic Change** (Highest Priority):
- Do the new messages introduce a completely different topic?
- Is there a shift from one specific event to another?
- Has the conversation moved from one question to an unrelated new question?

2. **Intent Transition**:
- Has the purpose of the conversation changed? (e.g., from casual chat to seeking help, from discussing work to discussing personal life)
- Has the core question or issue of the current topic been answered or fully discussed?

3. **Temporal Markers**:
- Are there temporal transition markers ("earlier", "before", "by the way", "oh right", "also", etc.)?
- Is the time gap between messages more than 30 minutes?

4. **Structural Signals**:
- Are there explicit topic transition phrases ("changing topics", "speaking of which", "quick question", etc.)?
- Are there concluding statements indicating the current topic is finished?

5. **Content Relevance**:
- How related is the new message to the previous discussion? (Consider splitting if relevance < 30%)
- Does it involve completely different people, places, or events?

Decision Principles:
- **Prioritize topic independence**: Each episode should revolve around one core topic or event
- **When in doubt, split**: When uncertain, lean towards starting a new episode
- **Maintain reasonable length**: A single episode typically shouldn't exceed 10-15 messages

For each episode, give:
- indices: the message numbers (1-based) belonging to this episode
- topic: a brief, specific description of what this episode is about

## Important Guidelines
- Episodes can have non-consecutive indices if messages are interleaved
- An episode should typically contain 2-15 messages
- Every message number from 1 to {count} must belong to exactly one episode
- Focus on topical coherence over strict chronological order
- When in doubt, prefer smaller, more focused episodes"""

# --- Narrative Episode Generation (Sec 3.2.2 / Appendix D.1.2) ---

NARRATIVE = """\
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation content:
{conversation}

Boundary detection reason:
{boundary_reason}

Please analyze the conversation to extract time information and generate a structured episodic memory with the following three fields:
- episodic_cue: A concise, descriptive title that accurately summarizes the theme (10-20 words)
- narrative_episode: A detailed description of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a coherent story so that the reader can clearly understand what happened. Ensure that time information is precise to the hour, including year, month, day, and hour.
- timestamp: YYYY-MM-DDTHH:MM:SS format timestamp representing when this episode occurred (analyze from message timestamps or content)

Time Analysis Instructions:
1. **Primary Source**: Look for explicit timestamps in the message metadata or content
2. **Secondary Source**: Analyze temporal references in the conversation content ("yesterday", "last week", "this morning", etc.)
3. **Fallback**: If no time information is available, use a reasonable estimate based on context
4. **Format**: Always return timestamp in ISO format: "2024-01-15T14:30:00"

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search.
7. Notice the time information, and write the time information in the content.
8. When relative times (e.g., last week, next month, etc.) are mentioned in the conversation, you need to convert them to absolute dates (year, month, day). Write the converted time in parentheses after the original time reference.
9. **IMPORTANT**: Analyze the actual time when the conversation happened from the message timestamps or content, not the current time.

Example:
If the conversation is about someone planning to go hiking and the messages have timestamps from March 14, 2024 at 3:00 PM:
- episodic_cue: "Weekend Hiking Plan March 16, 2024: Sunrise Trip to Mount Rainier"
- narrative_episode: "On March 14, 2024 at 3:00 PM, the user expressed interest in going hiking on the upcoming weekend (March 16, 2024) and sought advice. They particularly wanted to see the sunrise at Mount Rainier, having heard the scenery is beautiful. When asked about gear, they received suggestions including hiking boots, warm clothing (as it's cold at the summit), a flashlight, water, and high-energy food. The user decided to leave at 4:00 AM on Saturday, March 16, 2024 to catch the sunrise and planned to invite friends for the adventure. They were very excited about the trip, hoping to connect with nature."
- timestamp: "2024-03-14T15:00:00\""""

# --- Associative Memory Integration: 대상 선택 (Sec 3.2.3 / Appendix D.1.3) ---

SELECT_TARGET = """\
You are an episodic memory merge decision expert. Determine if a new episode should be merged with an existing similar episode.

## New Episode
Time Range: {new_time_range}
Content: {new_content}

## Candidate Episodes to Merge With
{candidates}

## Your Task
Decide whether the new episode should:
1. **merge**: Merge with one of the candidates (they describe the same event/topic)
2. **new**: Keep as a separate new episode (it's a distinct event)

## Merge Criteria
Merge ONLY if:
- Both episodes describe the SAME event or conversation session
- They have significant temporal overlap or are very close in time
- The content is clearly a continuation or different perspective of the same topic
- Merging would create a more complete picture without mixing different events

Do NOT merge if:
- They are different events/conversations even if on similar topics
- They are separated by significant time gaps (>1 hour)
- They involve different contexts or participants

If the decision is merge, give the number of the target candidate (1-based, as numbered above); otherwise give -1. Also give a brief reason for your decision."""

# --- Associative Memory Integration: 병합 서사 (Sec 3.2.3 / Appendix D.1.4) ---

INTEGRATE = """\
You are an episodic memory merge content generator. Combine two related episodes into a single, coherent episode.

## Original Episode
Time Range: {original_time_range}
Title: {original_title}
Content: {original_content}

## New Episode to Merge
Time Range: {new_time_range}
Title: {new_title}
Content: {new_content}

Combined Event Details: {combined_events}

## Your Task
Generate a merged episode that:
1. Combines information from both episodes without duplication
2. Maintains chronological flow of events
3. Preserves all important details from both episodes
4. Creates a coherent narrative

Give three fields:
- episodic_cue: merged episode title that captures the complete topic
- narrative_episode: detailed narrative combining both episodes chronologically. Include all participants, key decisions, emotions, and outcomes. Use third-person narrative style.
- timestamp: ISO format timestamp of when the merged episode occurred (use earliest time)

## Guidelines
- Integrate details naturally, don't just concatenate
- Eliminate redundancy while preserving unique information
- Maintain temporal coherence in the narrative
- Use specific details that aid searchability
- Write in third-person narrative style"""

# --- Anticipatory Schema Synthesis (Sec 3.3.1 / Appendix D.1.5) ---

ANTICIPATE = """\
You are a knowledge-based episode prediction system. Your task is to reconstruct a complete conversation episode based on limited clues and your knowledge base.

IMPORTANT: You are predicting the ACTUAL CONTENT and KNOWLEDGE of what happened, not the writing style or format.

## Input Information

Episodic Cue (Title/Summary): {episodic_cue}

Evoked Context (Prior Knowledge):
{evoked_context}

## Your Task

Based on the above clues, reconstruct what you believe happened in this episode. Focus on:
1. **Core Facts**: What specific information was discussed?
2. **Key Decisions**: What choices or conclusions were made?
3. **Knowledge Exchange**: What knowledge was shared or learned?
4. **Logical Flow**: How did the conversation progress?

## What to IGNORE
- Writing style or level of detail
- Specific formatting or structure
- Exact phrasing or word choices
- Whether timestamps are included in the text
- How formal or casual the language is

## Output Format

Generate a natural narrative that captures what you predict happened. Write it as if you're describing the episode to someone else. Focus on the SUBSTANCE, not the STYLE.

Your prediction:"""

# --- Prediction Error Distillation (Sec 3.3.2 / Appendix D.1.6) ---

DISTILL = """\
You are extracting valuable knowledge by comparing original conversation with predicted content.

Actual Episode (P_in - Ground Truth):
{original_messages}

Anticipatory Schema (predicted - Expectation):
{predicted_episode}

## Your Task:
Extract ONLY the valuable knowledge that exists in the original but is missing or misrepresented in the prediction.

## What to Extract:
Knowledge that is:
- Factual and will remain true over time
- Specific (names, titles, preferences, reasons)
- Useful for future interactions
- Not captured accurately in the prediction

## What to Ignore:
- Temporary states or emotions
- Conversational flow or style
- Information already well-represented in prediction
- Social pleasantries or reactions

## Examples:
Original: "I'm Alice, a senior engineer at Google. I switched from Java to Python last year because I wanted to work on ML projects."
Predicted: "Alice discussed their programming experience."
Extract:
- "Alice is a senior engineer at Google"
- "Alice switched from Java to Python for ML projects"

Original: "My favorite book is 'Deep Learning' by Goodfellow. I read it three times because the math explanations are so clear."
Predicted: "Alice mentioned liking technical books."
Extract:
- "Alice's favorite book is 'Deep Learning' by Goodfellow"
- "Alice values clear mathematical explanations in technical books"

Original: "I've been with Microsoft since 2019, started as a junior developer and got promoted to team lead in 2022. Planning to finish my online CS masters by December 2024."
Predicted: "Alice works at Microsoft and is studying."
Extract:
- "Alice has been at Microsoft since 2019 (5+ years)"
- "Alice was promoted from junior developer to team lead in 2022"
- "Alice is pursuing an online CS masters degree, expected completion December 2024"

Give the extracted statements. Important:
- Each statement should be self-contained and understandable without context
- Use present tense for persistent facts
- Include specific names, titles, and details
- Focus on quality over quantity - only extract truly valuable knowledge"""

# --- Direct Distillation — cold start 폴백 (Appendix D.2, NEMORI-s) ---

DIRECT_DISTILL = """\
You are an AI memory system. Extract HIGH-VALUE, PERSISTENT semantic memories from the following episodes.

CRITICAL: Focus on extracting LONG-TERM VALUABLE KNOWLEDGE, not temporary conversation details.

Episodes to analyze:
{episodes}

## HIGH-VALUE Knowledge Criteria

Extract ONLY knowledge that passes these tests:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs?
- **Independence Test**: Can be understood without conversation context?

## HIGH-VALUE Categories (FOCUS ON THESE):

1. **Identity & Professional**
- Names, titles, companies, roles
- Education, qualifications, skills

2. **Persistent Preferences**
- Favorite books, movies, music, tools
- Technology preferences with reasons
- Long-term likes and dislikes

3. **Technical Knowledge**
- Technologies used (with versions)
- Architectures, methodologies
- Technical decisions and rationales

4. **Relationships**
- Names of family, colleagues, friends
- Team structure, reporting lines
- Professional networks

5. **Goals & Plans**
- Career objectives
- Learning goals
- Project plans

6. **Patterns & Habits**
- Regular activities
- Workflows, schedules
- Recurring challenges

## Examples:

HIGH-VALUE (Extract these):
- "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
- "The user works at ByteDance as a senior ML engineer"
- "The user prefers PyTorch over TensorFlow for debugging"
- "The user's team lead is named Sarah"
- "The user is learning Rust for systems programming"
- "The user has been practicing yoga since March 2021"
- "The user joined Amazon in August 2020 as a data scientist"
- "The user plans to relocate to Seattle in January 2025"

LOW-VALUE (Skip these):
- "The user thanked the assistant"
- "The user was confused about X"
- "The user appreciated the help"
- "The conversation was productive"
- Any temporary emotions or reactions

Give the extracted statements. Quality over quantity - extract only knowledge that truly helps understand the user long-term."""

# --- Response Generation (Sec 3.4 / Appendix D.3) ---

ANSWER = """\
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Episodic Memories:
{episodic}

Semantic Memories:
{semantic}

Question: {question}

Answer:"""
