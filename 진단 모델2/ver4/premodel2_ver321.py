import json
import re
import os
import math
import warnings
from typing import List, Dict, Any, Union, Set
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=ResourceWarning)

# 임베딩 모델 및 Kiwi 초기화
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()


# -------------------------------------------------------------------------
# Dataclass 정의
# -------------------------------------------------------------------------
@dataclass
class TextRatioResult:
    final_score: float      # 최종 반영 점수
    raw_ratio: float        # 1. 이미지 대비 텍스트 비율 (len_body / total)
    weight: float           # 2. 적용된 가중치 W(N)
    len_body: int           # 3. 본문 텍스트 길이
    ratio_score: float      # 4. 임계값(0.1 ~ 0.4) 적용 후 점수


@dataclass
class HybridSearchResult:
    final_score: float
    avg_combined_score: float
    avg_lexical_overlap: float
    avg_cosine_sim_raw: float
    total_queries_count: int         # DB 전체 질문 수
    cat_filtered_count: int          # 1차 카테고리 필터링 통과 수
    must_have_filtered_count: int    # 2차 키워드 필터링 통과 수 (최종 평가 대상)
    query_details: List[Dict[str, Any]]


@dataclass
class KeywordStuffingResult:
    final_score: float
    is_stuffing: bool
    dup_noun: str
    noun_count: int
    sep_count: int
    pattern_count: int
    dup_penalty: float
    len_penalty: float
    delim_penalty: float
    pattern_penalty: float


@dataclass
class JsonLdEvaluationResult:
    final_score: float
    raw_score: float
    is_valid: bool
    parsing_score: float
    density_score: float
    clothing_score: float
    trust_score: float
    present_attrs_count: int
    trust_count: int


@dataclass
class SingleAltEvaluationResult:
    final_score: float
    has_alt_attribute: bool
    is_stuffing: bool
    clean_alt: str
    s1_presence_score: float
    s2_relevance_score: float
    s3_sentence_score: float


@dataclass
class ImageAltEvaluationResult:
    avg_score: float
    raw_avg_score: float
    is_valid: bool
    total_image_count: int
    stuffing_image_count: int
    detail_results: List[SingleAltEvaluationResult] = field(default_factory=list)


@dataclass
class GEOTotalEvaluationResult:
    total_score: float
    text_ratio: TextRatioResult
    hybrid_search: HybridSearchResult
    keyword_stuffing: KeywordStuffingResult
    json_ld: JsonLdEvaluationResult
    image_alt: ImageAltEvaluationResult

    @property
    def formatted_score(self) -> str:
        return f"{int(self.total_score)}점/100점 만점"

    def print_summary(self):
        """평가 결과를 가독성 있게 요약 출력합니다."""
        print("=" * 60)
        print(f" 📊 GEO 평가 종합 결과 : {self.formatted_score}")
        print("=" * 60)
        
        # 1. 텍스트 비율
        tr = self.text_ratio
        print(f"1. 텍스트 비율 평가")
        print(f"    ├─ 최종 점수: {tr.final_score * 100:.1f}점")
        print(f"    └─ 세부 지표: 원본 비율 {tr.raw_ratio:.2f} | 변환 점수 {tr.ratio_score:.2f} | 가중치 {tr.weight:.2f} | 본문 길이 {tr.len_body}자")
        
        # 2. 하이브리드 검색
        hs = self.hybrid_search
        print(f"\n2. 하이브리드 검색 평가")
        print(f"    ├─ 최종 점수: {hs.final_score * 100:.1f}점")
        print(f"    ├─ 검색 통과 수: 전체 {hs.total_queries_count}개 중 {hs.must_have_filtered_count}개 통과 (1차 카테고리: {hs.cat_filtered_count}개)")
        print(f"    └─ 세부 지표: 어휘 중복도 {hs.avg_lexical_overlap:.2f} | 코사인 유사도 {hs.avg_cosine_sim_raw:.2f}")

        # 3. 키워드 스터핑
        ks = self.keyword_stuffing
        ks_status = "⚠️ 감점 누적됨" if ks.is_stuffing else "✅ 정상"
        print(f"\n3. 키워드 스터핑 검사")
        print(f"    ├─ 상태: {ks_status}")
        print(f"    ├─ 감지된 중복 명사: {ks.dup_noun if ks.dup_noun else '없음'}")
        print(f"    └─ 세부 감점: 최종 {ks.final_score:.2f} (중복: -{ks.dup_penalty}, 명사수: -{ks.len_penalty}, 구분자: -{ks.delim_penalty}, 패턴: -{ks.pattern_penalty})")

        # 4. JSON-LD 평가
        jld = self.json_ld
        jld_status = "✅ Valid" if jld.is_valid else "❌ Invalid"
        print(f"\n4. JSON-LD 구조화 데이터")
        print(f"    ├─ 상태: {jld_status} (최종 점수: {jld.final_score * 100:.1f}점)")
        print(f"    └─ 항목 점수: 파싱 {jld.parsing_score} | 밀도 {jld.density_score} | 속성 {jld.clothing_score} | 신뢰도 {jld.trust_score}")

        # 5. 이미지 Alt 태그
        alt = self.image_alt
        print(f"\n5. 이미지 Alt 태그 평가")
        print(f"    ├─ 최종 점수: {alt.avg_score * 100:.1f}점 (유효성 통과 여부: {alt.is_valid})")
        print(f"    └─ 이미지 현황: 총 {alt.total_image_count}개 중 스터핑 감지 {alt.stuffing_image_count}개")
        print("=" * 60)


class GEOScorer:
    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.replace('\x00', '').replace('\u0000', '')
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text.strip()

    def _extract_nouns(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = kiwi.tokenize(text)
        return [t.form for t in tokens if t.tag.startswith('N')]

    def _extract_morphs(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = kiwi.tokenize(text)
        return [t.form for t in tokens]

    def _extract_pos(self, text: str) -> List[tuple]:
        if not text:
            return []
        tokens = kiwi.tokenize(text)
        return [(t.form, t.tag) for t in tokens]

    # =========================================================================
    # ① 본문 텍스트 비율
    # =========================================================================
    def calculate_text_ratio(self, body_text: str, image_text: str) -> TextRatioResult:
        clean_body = self._clean_text(body_text)
        clean_image = self._clean_text(image_text)
        
        len_body = len(clean_body)
        len_image = len(clean_image)
        total = len_body + len_image
        
        if total == 0:
            return TextRatioResult(0.0, 0.0, 0.0, 0, 0.0)
        
        raw_ratio = len_body / total

        if raw_ratio < 0.1:
            ratio_score = 0.0
        elif raw_ratio >= 0.4:
            ratio_score = 1.0
        else:
            ratio_score = (raw_ratio - 0.1) / 0.3

        weight = min(1.0, math.sqrt(len_body / 800.0))
        final_score = round(min(1.0, ratio_score * weight), 4)

        return TextRatioResult(
            final_score=final_score,
            raw_ratio=round(raw_ratio, 4),
            weight=round(weight, 4),
            len_body=len_body,
            ratio_score=round(ratio_score, 4)
        )

    # =========================================================================
    # ② 하이브리드 검색
    # =========================================================================
    def calculate_hybrid_search(
        self, 
        user_queries: List[Dict[str, Any]], 
        product_cat: str, 
        product_name: str, 
        page_text: str
    ) -> HybridSearchResult:
        clean_page = self._clean_text(page_text)
        clean_prod_name = self._clean_text(product_name)
        
        if not clean_page or not user_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, len(user_queries) if user_queries else 0, 0, 0, [])

        total_queries = len(user_queries)
        
        cat_filtered_queries = [
            q for q in user_queries 
            if q.get("category") == product_cat
        ]
        cat_count = len(cat_filtered_queries)

        if not cat_filtered_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, 0, 0, [])

        final_queries = []
        for q in cat_filtered_queries:
            must_have_list = q.get("must_have", [])
            if any(mh in clean_prod_name for mh in must_have_list if mh):
                final_queries.append(q)
                
        must_have_count = len(final_queries)

        if not final_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, 0, [])

        page_nouns = self._extract_nouns(clean_page)
        page_nouns_set = set(page_nouns) if page_nouns else set(self._extract_morphs(clean_page))

        valid_query_objs = []
        query_strs = []
        for q_obj in final_queries:
            q_text = self._clean_text(q_obj.get("query_text", ""))
            if q_text:
                valid_query_objs.append(q_obj)
                query_strs.append(q_text)

        if not query_strs:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, must_have_count, [])

        try:
            p_vec = embedding_model.encode([clean_page])
            q_vecs = embedding_model.encode(query_strs, show_progress_bar=False)
            cos_sims = cosine_similarity(q_vecs, p_vec).reshape(-1)
        except Exception:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, must_have_count, [])

        query_details, lexical_overlap_list, cos_raw_list, combined_score_list = [], [], [], []

        for idx, q_obj in enumerate(valid_query_objs):
            query_str = query_strs[idx]
            query_tokens = self._extract_nouns(query_str) or self._extract_morphs(query_str)

            matched_count = sum(1 for token in query_tokens if token in page_nouns_set) if query_tokens else 0
            lexical_overlap = matched_count / len(query_tokens) if query_tokens else 0.0

            raw_val = float(cos_sims[idx].item()) if hasattr(cos_sims[idx], 'item') else float(cos_sims[idx])
            cos_sim_raw = max(0.0, min(1.0, raw_val))
            
            query_hybrid_score = min(1.0, (lexical_overlap * 0.5) + (cos_sim_raw * 0.5))

            lexical_overlap_list.append(lexical_overlap)
            cos_raw_list.append(cos_sim_raw)
            combined_score_list.append(query_hybrid_score)

            query_details.append({
                "query": query_str,
                "category": q_obj.get("category"),
                "must_have": q_obj.get("must_have"),
                "lexical_overlap": round(lexical_overlap, 4),
                "cosine_sim_raw": round(cos_sim_raw, 4),
                "query_hybrid_score": round(query_hybrid_score, 4)
            })

        if not combined_score_list:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, must_have_count, [])

        n = len(combined_score_list)
        avg_combined = min(1.0, sum(combined_score_list) / n)

        return HybridSearchResult(
            final_score=round(avg_combined, 4),
            avg_combined_score=round(avg_combined, 4),
            avg_lexical_overlap=round(sum(lexical_overlap_list) / n, 4),
            avg_cosine_sim_raw=round(sum(cos_raw_list) / n, 4),
            total_queries_count=total_queries,
            cat_filtered_count=cat_count,
            must_have_filtered_count=must_have_count,
            query_details=query_details
        )

    # =========================================================================
    # ③ 상품명 키워드 스터핑
    # =========================================================================
    def calculate_title_keyword_stuffing(self, title_text: str) -> KeywordStuffingResult:
        clean_title = self._clean_text(title_text) if title_text else ""
        if not clean_title:
            return KeywordStuffingResult(0.0, True, "", 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

        pos_tags = self._extract_pos(clean_title)
        if not pos_tags:
            return KeywordStuffingResult(0.0, True, "", 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

        nouns = [word for word, tag in pos_tags if tag.startswith("N")]
        noun_count = len(nouns)

        noun_counts = Counter(nouns)
        dup_nouns = [word for word, count in noun_counts.items() if count >= 2]
        dup_noun = ", ".join(dup_nouns) if dup_nouns else ""

        max_duplicate = max(noun_counts.values()) if noun_counts else 0
        dup_penalty = 0.25 if max_duplicate >= 3 else (0.10 if max_duplicate == 2 else 0.0)
        len_penalty = 0.25 if noun_count >= 8 else (0.10 if noun_count >= 7 else 0.0)

        delimiters = re.findall(r"[/|\[\](),_\-~!+*#&]", clean_title)
        sep_count = len(delimiters)
        delim_penalty = 0.25 if sep_count >= 4 else (0.10 if sep_count >= 3 else 0.0)

        target_suffixes = ("핏", "룩", "템", "LOOK", "FIT", "ITEM")
        pattern_count = sum(1 for word, _ in pos_tags if str(word).upper().endswith(target_suffixes))
        pattern_penalty = 0.25 if pattern_count >= 3 else (0.10 if pattern_count == 2 else 0.0)

        total_penalty = dup_penalty + len_penalty + delim_penalty + pattern_penalty
        final_score = max(0.0, round(1.0 - total_penalty, 3))

        return KeywordStuffingResult(
            final_score=final_score,
            is_stuffing=final_score < 0.6,
            dup_noun=dup_noun,
            noun_count=noun_count,
            sep_count=sep_count,
            pattern_count=pattern_count,
            dup_penalty=dup_penalty,
            len_penalty=len_penalty,
            delim_penalty=delim_penalty,
            pattern_penalty=pattern_penalty,
        )

    # =========================================================================
    # ④ JSON-LD 구조화 데이터 평가 (구현 보완 및 순환 참조 방지)
    # =========================================================================
    def _flatten_json_items(self, items: Any) -> List[dict]:
        flat = []
        if isinstance(items, list):
            for item in items:
                flat.extend(self._flatten_json_items(item))
        elif isinstance(items, dict):
            flat.append(items)
        return flat

    def _extract_json_ld_attributes(self, data: Any) -> Set[str]:
        target_attrs = {"material", "color", "size", "category"}
        found_attrs = set()
        visited = set()

        def _search(d):
            if not isinstance(d, (dict, list)):
                return
            d_id = id(d)
            if d_id in visited:
                return
            visited.add(d_id)

            if isinstance(d, dict):
                for k, v in d.items():
                    if k in target_attrs and v:
                        found_attrs.add(k)
                    _search(v)
            elif isinstance(d, list):
                for item in d:
                    _search(item)

        _search(data)
        return found_attrs

    def calculate_json_ld_score(self, json_ld_input: Union[str, list, dict]) -> JsonLdEvaluationResult:
        if not json_ld_input:
            return JsonLdEvaluationResult(0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0, 0)

        try:
            if isinstance(json_ld_input, str):
                data = json.loads(json_ld_input)
            else:
                data = json_ld_input
            items = self._flatten_json_items(data)
            parsing_score = 0.25 if items else 0.0
        except Exception:
            return JsonLdEvaluationResult(0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0, 0)

        # 밀도 평가
        density_score = 0.25 if len(items) >= 1 else 0.0

        # 속성 추출 평가
        found_attrs = self._extract_json_ld_attributes(data)
        present_attrs_count = len(found_attrs)
        clothing_score = min(0.25, present_attrs_count * 0.0625)

        # 신뢰도 평가 (brand, offers, aggregateRating 등)
        trust_keys = {"brand", "offers", "aggregateRating", "review", "sku"}
        trust_count = sum(1 for item in items for k in trust_keys if k in item)
        trust_score = min(0.25, trust_count * 0.0833)

        raw_score = parsing_score + density_score + clothing_score + trust_score
        final_score = round(min(1.0, raw_score), 4)

        return JsonLdEvaluationResult(
            final_score=final_score,
            raw_score=round(raw_score, 4),
            is_valid=parsing_score > 0,
            parsing_score=round(parsing_score, 4),
            density_score=round(density_score, 4),
            clothing_score=round(clothing_score, 4),
            trust_score=round(trust_score, 4),
            present_attrs_count=present_attrs_count,
            trust_count=trust_count
        )

    # =========================================================================
    # ⑤ 이미지 Alt 태그 평가 (단일 및 전체 평균 구현 보완)
    # =========================================================================
    def evaluate_single_alt(self, img_obj: Union[dict, str]) -> SingleAltEvaluationResult:
        if isinstance(img_obj, dict):
            has_alt_attr = "alt" in img_obj
            raw_alt = img_obj.get("alt", "")
        else:
            has_alt_attr = img_obj is not None
            raw_alt = img_obj if img_obj else ""

        if not has_alt_attr:
            return SingleAltEvaluationResult(0.0, False, False, "", 0.0, 0.0, 0.0)

        s1 = 0.20
        clean_alt = self._clean_text(raw_alt) if raw_alt else ""

        if not clean_alt:
            return SingleAltEvaluationResult(0.20, True, False, "", s1, 0.0, 0.0)

        invalid_patterns = r'(\.(jpg|jpeg|png|gif|webp|bmp)$)|(^img[_-]?\d+$|^image[_-]?\d+$)'
        has_invalid_pattern = bool(re.search(invalid_patterns, clean_alt, re.IGNORECASE))
        s2 = 0.0 if has_invalid_pattern else 0.30

        pos_tags = self._extract_pos(clean_alt)

        continuous_noun_count = 0
        max_continuous_nouns = 0
        for _, pos in pos_tags:
            if pos.startswith('N'):
                continuous_noun_count += 1
                max_continuous_nouns = max(max_continuous_nouns, continuous_noun_count)
            else:
                continuous_noun_count = 0
        cond1 = max_continuous_nouns >= 4

        etm_continuous_count = 0
        for i in range(len(pos_tags) - 1):
            if pos_tags[i][1] == 'ETM' and pos_tags[i+1][1] != 'EC':
                etm_continuous_count += 1
        cond2 = etm_continuous_count >= 2

        is_stuffing = cond1 or cond2

        if is_stuffing:
            s3 = 0.0
        else:
            has_josa = any(p.startswith('J') for _, p in pos_tags)
            has_verb = any(p.startswith('V') for _, p in pos_tags)
            has_modifier = any(p.startswith('M') for _, p in pos_tags)

            element_count = sum([has_josa, has_verb, has_modifier])
            s3_map = {3: 0.50, 2: 0.40, 1: 0.30}
            s3 = s3_map.get(element_count, 0.20)

        final_score = round(min(1.0, s1 + s2 + s3), 3)

        return SingleAltEvaluationResult(
            final_score=final_score,
            has_alt_attribute=True,
            is_stuffing=is_stuffing,
            clean_alt=clean_alt,
            s1_presence_score=s1,
            s2_relevance_score=s2,
            s3_sentence_score=s3
        )

    def calculate_avg_alt_score(self, image_list: List[Dict[str, Any]]) -> ImageAltEvaluationResult:
        if not image_list:
            return ImageAltEvaluationResult(0.0, 0.0, False, 0, 0, [])

        details = [self.evaluate_single_alt(img) for img in image_list]
        total_count = len(details)
        stuffing_count = sum(1 for d in details if d.is_stuffing)
        
        scores = [d.final_score for d in details]
        raw_avg = sum(scores) / total_count if total_count > 0 else 0.0

        return ImageAltEvaluationResult(
            avg_score=round(raw_avg, 4),
            raw_avg_score=round(raw_avg, 4),
            is_valid=True,
            total_image_count=total_count,
            stuffing_image_count=stuffing_count,
            detail_results=details
        )

    # =========================================================================
    # Total 평가 함수
    # =========================================================================
    def evaluate_page(
        self,
        body_text: str,
        image_text: str,
        user_queries: List[Dict[str, Any]],
        product_cat: str,
        product_name: str,
        json_ld_str: Union[str, list, dict] = "",
        image_list: List[Dict[str, Any]] = None
    ) -> GEOTotalEvaluationResult:

        if image_list is None:
            image_list = []
        page_text = f"{body_text} {image_text}".strip()

        s1 = self.calculate_text_ratio(body_text=body_text, image_text=image_text)
        s2 = self.calculate_hybrid_search(
            user_queries=user_queries, 
            product_cat=product_cat, 
            product_name=product_name, 
            page_text=page_text
        )
        s3 = self.calculate_title_keyword_stuffing(title_text=product_name)
        s4 = self.calculate_json_ld_score(json_ld_input=json_ld_str)
        s5 = self.calculate_avg_alt_score(image_list=image_list)

        total_score = round(((s1.final_score + s2.final_score + s3.final_score + s4.final_score + s5.avg_score) / 5.0) * 100, 0)

        return GEOTotalEvaluationResult(
            total_score=total_score,
            text_ratio=s1,
            hybrid_search=s2,
            keyword_stuffing=s3,
            json_ld=s4,
            image_alt=s5
        )