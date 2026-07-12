"""Zep 파이프라인의 프롬프트 — 순수 상수만.

출처 규칙: 논문이 명세서다. 논문 Appendix 6.1에 실린 프롬프트는 자구
그대로(verbatim) 쓰고, 논문이 침묵하는 단계만 원본 graphiti(v0.5.2 핀,
external/graphiti)에서 차용한다. 차용분은 Apache-2.0 (Copyright 2024,
Zep Software, Inc.).

    상수                        출처
    ─────────────────────────  ─────────────────────────────────────────
    ENTITY_EXTRACTION           논문 6.1.1 (guideline 6만 아래 참고)
    ENTITY_RESOLUTION           논문 6.1.2
    FACT_EXTRACTION             논문 6.1.3
    FACT_RESOLUTION             graphiti HEAD 526dcad7 dedupe_edges.py
                                resolve_edge() — 논문 6.1.4(dedup)와
                                INVALIDATION(v2)을 대체하는 병합 프롬프트
                                (아래 참고)
    TEMPORAL_EXTRACTION         논문 6.1.5
    CONTEXT_TEMPLATE            논문 Sec 3 "sample context string template"
                                (COMMUNITIES 섹션만 신설 — 아래 참고)
    ENTITY_REFLEXION            graphiti prompts/extract_nodes.py reflexion()
    ENTITY_SUMMARY              graphiti prompts/summarize_nodes.py summarize_context()
    SUMMARY_COMBINE             graphiti prompts/summarize_nodes.py summarize_pair()
                                — 원본은 community 요약과 entity 병합에 같은
                                프롬프트를 쓴다 (node_operations.py:300). N개
                                packing을 위해 "two"만 제거 (아래 참고)
    COMMUNITY_NAME              graphiti prompts/summarize_nodes.py summary_description()
    ANSWER_SYSTEM / ANSWER_USER 논문·graphiti 모두 부재 (Sec 4.1이 Appendix에
                                준다고 했으나 없음) → MemoryOS 재구현의 answer
                                프롬프트에서 채점 관련 지시(간결성 예시·날짜
                                형식·duration·구체 entity)를 그대로 가져옴 —
                                메소드 간 채점 조건 통일. role-play·persona
                                틀은 MemoryOS 고유 구조라 제외.
    *_SYSTEM (ANSWER 제외)      전부 graphiti — 논문은 user 프롬프트만 싣는다

논문이 침묵하거나 결함이 있어 정한 것:
- 논문 template에는 COMMUNITIES 섹션이 없다 — Sec 3 χ의 형식 정의(N_c →
  summary field)를 따라 같은 문체로 신설하고, 머리문장과 ANSWER_SYSTEM의
  memory 나열도 맞춰 넓혔다 (2026-07-10 결정). 근거는 search.py docstring.
- 출판본(PDF·HTML 동일)에서 6.1.1 guideline 6이 "DO NOT extract entities
  mentioned only"에서 잘려 있다 → graphiti extract_nodes.py:68의 원문
  "...only in PREVIOUS MESSAGES, those messages are only to provide
  context."로 보완.
- 논문 프롬프트의 줄바꿈은 2단 조판 아티팩트라 자연 줄로 정리했다.
  자구(단어·구두점)는 그대로다.
- edge용 reflexion은 채택하지 않는다 — 논문은 reflection을 entity
  extraction(Sec 2.2.1)에만 언급한다. 코드에만 있는 메커니즘은 논문
  것으로 착각하지 않는다 (graphiti extract_edges.py reflexion은 존재).
- graphiti의 dedupe_nodes user 프롬프트는 논문 6.1.2와 자구가 다르다 →
  논문 우선. edge 쪽(6.1.4)은 예외 — 속도 결정으로 병합 프롬프트를 차용해
  논문 프롬프트를 대체했다 (FACT_RESOLUTION 위 주석, 2026-07-11 합의).
- placeholder 이름은 str.format() 관례로 통일 ({previous_messages} 등).
  값의 직렬화 형식(JSON이냐 줄글이냐)은 llm_ops.py 소관.
"""

# --- entity extraction (Sec 2.2.1 / Appendix 6.1.1) ---

ENTITY_EXTRACTION_SYSTEM = (
    "You are an AI assistant that extracts entity nodes from conversational "
    "messages. Your primary task is to identify and extract the speaker and "
    "other significant entities mentioned in the conversation."
)

ENTITY_EXTRACTION = """\
<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

Given the above conversation, extract entity nodes from the CURRENT MESSAGE that are explicitly or implicitly mentioned:

Guidelines:
1. ALWAYS extract the speaker/actor as the first node. The speaker is the part before the colon in each line of dialogue.
2. Extract other significant entities, concepts, or actors mentioned in the CURRENT MESSAGE.
3. DO NOT create nodes for relationships or actions.
4. DO NOT create nodes for temporal information like dates, times or years (these will be added to edges later).
5. Be as explicit as possible in your node names, using full names.
6. DO NOT extract entities mentioned only in PREVIOUS MESSAGES, those messages are only to provide context."""

# --- entity reflexion (Sec 2.2.1이 언급, 프롬프트는 graphiti 차용) ---

ENTITY_REFLEXION_SYSTEM = (
    "You are an AI assistant that determines which entities have not been "
    "extracted from the given context"
)

ENTITY_REFLEXION = """\
<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{extracted_entities}
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any entities haven't been extracted."""

# --- entity resolution (Sec 2.2.1 / Appendix 6.1.2) ---

ENTITY_RESOLUTION_SYSTEM = (
    "You are a helpful assistant that de-duplicates nodes from node lists."
)

ENTITY_RESOLUTION = """\
<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
<EXISTING NODES>
{existing_nodes}
</EXISTING NODES>

Given the above EXISTING NODES, MESSAGE, and PREVIOUS MESSAGES. Determine if the NEW NODE extracted from the conversation is a duplicate entity of one of the EXISTING NODES.

<NEW NODE>
{new_node}
</NEW NODE>

Task:
1. If the New Node represents the same entity as any node in Existing Nodes, return 'is_duplicate: true' in the response. Otherwise, return 'is_duplicate: false'
2. If is_duplicate is true, also return the uuid of the existing node in the response
3. If is_duplicate is true, return a name for the node that is the most complete full name.

Guidelines:
1. Use both the name and summary of nodes to determine if the entities are duplicates, duplicate nodes may have different names"""

# --- entity summary (Sec 2.2.1이 언급, 프롬프트는 graphiti 차용) ---

ENTITY_SUMMARY_SYSTEM = (
    "You are a helpful assistant that combines summaries with new "
    "conversation context."
)

ENTITY_SUMMARY = """\
<MESSAGES>
{messages}
</MESSAGES>

Given the above MESSAGES and the following ENTITY name, create a summary for the ENTITY. Your summary must only use information from the provided MESSAGES. Your summary should also only contain information relevant to the provided ENTITY.

Summaries must be under 500 words.

<ENTITY>
{entity_name}
</ENTITY>"""

# --- fact extraction (Sec 2.2.2 / Appendix 6.1.3) ---

FACT_EXTRACTION_SYSTEM = (
    "You are an expert fact extractor that extracts fact triples from text."
)

FACT_EXTRACTION = """\
<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
<ENTITIES>
{entities}
</ENTITIES>

Given the above MESSAGES and ENTITIES, extract all facts pertaining to the listed ENTITIES from the CURRENT MESSAGE.

Guidelines:
1. Extract facts only between the provided entities.
2. Each fact should represent a clear relationship between two DISTINCT nodes.
3. The relation_type should be a concise, all-caps description of the fact (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR).
4. Provide a more detailed fact containing all relevant information.
5. Consider temporal aspects of relationships when relevant."""

# --- fact resolution: dedup(6.1.4) + invalidation 선별을 한 콜로 ---
# 논문 6.1.4·INVALIDATION(v2)의 개별 프롬프트를 버리고 현 upstream의 병합
# 프롬프트를 차용 — triple당 3콜을 2콜로 줄이는 속도 결정 (2026-07-11 합의,
# 출처: graphiti HEAD 526dcad7 prompts/dedupe_edges.py resolve_edge, v0.5.2엔
# 없음). 자구 변경 둘: idx 지시는 우리 alias 표기에 맞춤, FACT INVALIDATION
# CANDIDATES에 날짜 포함 — 날짜 없인 qwen이 모순에 무반응 (llm_ops 실측 참고).

FACT_RESOLUTION_SYSTEM = (
    "You are a fact deduplication assistant. "
    "NEVER mark facts with key differences as duplicates."
)

FACT_RESOLUTION = """\
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{existing_edges}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{invalidation_candidates}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{new_edge}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>"""

# --- temporal extraction (Sec 2.2.3 / Appendix 6.1.5) ---

TEMPORAL_EXTRACTION_SYSTEM = (
    "You are an AI assistant that extracts datetime information for graph "
    "edges, focusing only on dates directly related to the establishment or "
    "change of the relationship described in the edge fact."
)

TEMPORAL_EXTRACTION = """\
<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>
<REFERENCE TIMESTAMP>
{reference_timestamp}
</REFERENCE TIMESTAMP>
<FACT>
{fact}
</FACT>

IMPORTANT: Only extract time information if it is part of the provided fact. Otherwise ignore the time mentioned. Make sure to do your best to determine the dates if only the relative time is mentioned. (eg 10 years ago, 2 mins ago) based on the provided reference timestamp
If the relationship is not of spanning nature, but you are still able to determine the dates, set the valid_at only.

Definitions:
- valid_at: The date and time when the relationship described by the edge fact became true or was established.
- invalid_at: The date and time when the relationship described by the edge fact stopped being true or ended.

Task:
Analyze the conversation and determine if there are dates that are part of the edge fact. Only set dates if they explicitly relate to the formation or alteration of the relationship itself.

Guidelines:
1. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) for datetimes.
2. Use the reference timestamp as the current time when determining the valid_at and invalid_at dates.
3. If the fact is written in the present tense, use the Reference Timestamp for the valid_at date
4. If no temporal information is found that establishes or changes the relationship, leave the fields as null.
5. Do not infer dates from related events. Only use dates that are directly stated to establish or change the relationship.
6. For relative time mentions directly related to the relationship, calculate the actual datetime based on the reference timestamp.
7. If only a date is mentioned without a specific time, use 00:00:00 (midnight) for that date.
8. If only year is mentioned, use January 1st of that year at 00:00:00.
9. Always include the time zone offset (use Z for UTC if no specific time zone is mentioned)."""

# --- summary 결합 (Sec 2.3 community 요약 + Sec 2.2.1 entity 병합, graphiti 차용) ---
# 원본 문구의 "the following two summaries"에서 "two"만 제거 — packed
# map-reduce가 배치당 N개를 넣는다 (2026-07-10 합의, communities.py 참고).

SUMMARY_COMBINE_SYSTEM = "You are a helpful assistant that combines summaries."

SUMMARY_COMBINE = """\
Synthesize the information from the following summaries into a single succinct summary.

Summaries must be under 500 words.

Summaries:
{summaries}"""

COMMUNITY_NAME_SYSTEM = (
    "You are a helpful assistant that describes provided contents in a "
    "single sentence."
)

COMMUNITY_NAME = """\
Create a short one sentence description of the summary that explains what kind of information is summarized.
Summaries must be under 500 words.

Summary:
{summary}"""

# --- 최종 답변 생성 (실험 경로 — Sec 4의 chat agent 역할) ---

ANSWER_SYSTEM = """\
You are a helpful assistant that answers questions about a conversation \
between {speaker_a} and {speaker_b}. The FACTS, ENTITIES and COMMUNITIES \
provided by the user are your memory of that conversation.
Your task is to answer questions about {speaker_a} or {speaker_b} in an \
extremely concise manner.
When the question is: "What did the charity race raise awareness for?", \
you should not answer in the form of: "The charity race raised awareness \
for mental health." Instead, it should be: "mental health", as this is \
more concise."""

ANSWER_USER = """\
{context}

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

# --- context 조립 (Sec 3 "sample context string template", constructor χ.
#     COMMUNITIES 섹션만 신설 — 파일 docstring 참고) ---

CONTEXT_TEMPLATE = """\
FACTS, ENTITIES and COMMUNITIES represent relevant context to the current conversation.

These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

These are the most relevant entities
ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>

These are the most relevant community summaries
<COMMUNITIES>
{communities}
</COMMUNITIES>"""
