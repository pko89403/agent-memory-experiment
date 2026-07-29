"""A-MEM (arXiv 2502.12110, NeurIPS 2025) note-management 부품.

MemoryMethod가 아니다 — 단독 실행하지 않기로 합의(2026-07-25: plain
A-MEM baseline은 우리 연구 질문에 불요). Zettelkasten식 note 구성·링크·
evolution을 nemori의 ManagementSystem 자리에 꽂는 부품이며, 실행 가능한
메소드는 methods/nemori_amem이 조립한다.

논문이 명세서이나 프롬프트는 예외적으로 코드판 차용 (근거는
prompt_templates docstring — 부록 B의 결함). reference: external/a-mem
@ ceffb86 핀.
"""
from memlab.methods.amem.store import AmemNoteStore

__all__ = ["AmemNoteStore"]
