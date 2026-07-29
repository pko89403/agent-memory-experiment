"""A-MEM 파이프라인의 프롬프트 — 순수 상수만.

출처 규칙: 논문이 명세서이나 A-MEM은 예외적으로 **코드판을 차용**한다
(external/a-mem @ ceffb86, MIT). 근거 — 논문 부록 B가 결함투성이다:
B.2(Ps2)는 출력 형식조차 없는 불완전 프롬프트라 실행 가능한 해석이
없고(코드는 Ps2·Ps3를 한 콜로 병합 — 유일한 실행 해석), B.3(Ps3)의
JSON 예시에는 본문·코드 어디에도 구현이 없는 유령 액션(merge, prune)이
등장한다. 코드판 프롬프트는 자기일관적이다 (2026-07-25 합의: 원본에
실체 없으면 제외).

    상수          출처                              호출 방식
    ───────────  ────────────────────────────────  ──────────
    CONSTRUCT     코드 analyze_content              structured
                  (memory_system.py:176-203) = 논문
                  B.1과 사실상 동일 자구
    EVOLVE        코드 _evolution_system_prompt      structured
                  (memory_system.py:127-157) —
                  Ps2+Ps3 병합판, strengthen/
                  update_neighbor 두 액션만

structured output 변형 (nemori 때 확립한 방식):
- JSON 예시 블록·"Return your decision in JSON format" 강제문 제거 —
  형식은 llm_ops의 Pydantic json_schema가 강제한다. 단 JSON 블록의
  주석에 실려 있던 실질 지시(키워드 최소 3개, 화자·시간 제외, 중요도
  순 정렬 등)는 산문 필드 서술로 전부 보존했다.
- EVOLVE 끝에 필드 나열 한 줄을 신설했다 — 원 프롬프트는 JSON 예시로만
  필드명을 알려줬는데 그 예시를 제거했으므로 의미 서술이 필요하다.

논문·코드가 침묵하거나 결함이 있어 정한 것:
- suggested_connections는 이웃의 **번호(0-based index)**로 답하게 한다.
  원본은 이웃을 "memory index:{i}"로 나열하고도 응답을 note.links에
  문자열 그대로 extend해서, links에 uuid가 아닌 인덱스 문자열("0","1")이
  들어가 검색 시 절대 해소되지 않는 버그가 있다 (find_related_memories_raw의
  `if link_id in self.memories`가 항상 False). 우리는 번호 응답을
  llm_ops에서 실제 uuid로 매핑해 복원한다 — 버그 비복제 원칙 + zep의
  로컬 별칭 관례.
- "Analyze the the new memory" 등 원문의 오타·겹말은 자구 보존한다.
- 이웃 직렬화 형식("memory index:… talk start time:…")은 llm_ops 소관.
"""

# --- Note Construction (Sec 3.1 / 코드 analyze_content) ---

CONSTRUCT = """\
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Give:
- keywords: several specific, distinct keywords that capture key concepts and terminology. Order from most to least important. Don't include keywords that are the name of the speaker or time. At least three keywords, but don't be too redundant.
- context: one sentence summarizing the main topic/domain, key arguments/points, and intended audience/purpose.
- tags: several broad categories/themes for classification. Include domain, format, and type tags. At least three tags, but don't be too redundant.

Content for analysis:
{content}"""

# --- Memory Evolution (Sec 3.2-3.3 병합 / 코드 _evolution_system_prompt) ---

EVOLVE = """\
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

For your decision give: should_evolve, actions (any of "strengthen", "update_neighbor"), suggested_connections (the memory index numbers to connect to), tags_to_update, new_context_neighborhood, new_tags_neighborhood."""
