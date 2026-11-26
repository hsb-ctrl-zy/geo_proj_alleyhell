import os
import json
from flask import Flask, jsonify, render_template, request
import pymysql
from dotenv import load_dotenv

# 1. 절대 경로 기반으로 상위 폴더의 project_db/geo.env 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '..', 'project_db', 'geo.env')
load_dotenv(dotenv_path=ENV_PATH)

app = Flask(__name__)

# 2. DB 접속 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 3. 테스트용 하드코딩 target page_id
# HARDCODED_PAGE_ID = 2900
# HARDCODED_QUERY_ID = 313
# 3. 시연영상 찍을 후보
HARDCODED_PAGE_ID = 5020
HARDCODED_QUERY_ID = 853
# HARDCODED_PAGE_ID = 3340
# HARDCODED_QUERY_ID = 981

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def build_accordion_html(spec_text: str, qa_list: list, title_text: str) -> str:
    qa_html = "".join([
        f'<p style="margin-bottom: 2px;"><strong>Q. {qa["q"]}</strong></p>'
        f'<p style="margin-bottom: 8px;">A. {qa["a"]}</p>'
        for qa in qa_list
    ])
    
    return f"""<details style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 8px; background-color: #0f172a; color: #ffffff; font-family: 'Pretendard', sans-serif;">
  <summary style="font-weight: bold; cursor: pointer; background-color: #0f172a; color: #ffffff;">
    🔍 제품 상세 설명 및 Q&A 보기 (클릭)
  </summary>
  <div style="margin-top: 12px; font-size: 14px; line-height: 1.6; border-top: 1px solid #f3f4f6; padding-top: 10px;">
    <p style="margin-bottom: 4px;"><strong>[제품 상세 설명]</strong></p>
    <p style="margin-bottom: 12px;">{spec_text}</p>
    
    <p style="margin-bottom: 4px;"><strong>[자주 묻는 질문 (Q&A)]</strong></p>
    {qa_html}
    <p style="margin-top: 12px; margin-bottom: 4px;"><strong>[상품명 추천]</strong></p>
    <p style="margin-bottom: 4px;">{title_text}</p>
  </div>
</details>"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    product_url = data.get('url')

    if not product_url:
        return jsonify({'error': 'URL을 입력해주세요.'}), 400

    # DB 연동용 기본 변수 선언
    diagnosis_results = []
    stats_result = []
    pre_score = 0
    post_score = 0
    product_name = "상품명 정보 없음"

    # tab2
    weaknesses = []
    warnings = []
    strengths = []

    # tab3
    title_dummy = ""
    spec_dummy = ""
    qa_dummy = []
    json_ld_schema = ""
    alt_text = []

    try:
        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 1. 진단 및 상품 기본 정보 조회
            sql = """
                SELECT 
                    d.page_id, 
                    d.is_corrected,
                    d.geo_score, 
                    r.product_name 
                FROM diagnosis_table d
                LEFT JOIN raw_data_table r ON d.page_id = r.page_id
                WHERE d.page_id = %s
            """
            cursor.execute(sql, (HARDCODED_PAGE_ID,))
            diagnosis_results = cursor.fetchall()

            # 2. 세션별 인용 통계 데이터 조회 (%%V2%% 로 escape 처리하여 파이싱 에러 방지)
            # 1. SQL 구문 정의 (변수가 들어갈 자리에 %s 배치)
            sql_stats = """
                SELECT
                    s.session_id,
                    s.is_corrected,
                    l.page_id,
                    ROUND(AVG(l.is_cited), 2) AS is_cited_avg,
                    COALESCE((
                        SELECT sub.trial_citation_rank
                        FROM geo_db.eval_log_table sub
                        WHERE sub.session_id = s.session_id 
                        AND sub.page_id = l.page_id
                        AND sub.trial_citation_rank > 0
                        GROUP BY sub.trial_citation_rank
                        ORDER BY COUNT(*) DESC
                        LIMIT 1
                    ), '-') AS cited_rank
                FROM geo_db.eval_session_table s
                LEFT JOIN geo_db.eval_log_table l
                    ON s.session_id = l.session_id
                WHERE l.page_id = %s
                AND s.session_id LIKE CONCAT('%%', %s, '%%V2%%')
                GROUP BY s.session_id, s.is_corrected, l.page_id
                ORDER BY s.session_id, l.page_id;
            """

            # 2. execute 실행 시 튜플로 순서대로 전달 (%s 순서와 동일해야 함)
            cursor.execute(sql_stats, (HARDCODED_PAGE_ID, HARDCODED_QUERY_ID))
            stats_result = cursor.fetchall()

            # 터미널 콘솔로 DB 조회 결과 확인
            print(f"=== DB 진단 조회 결과: {diagnosis_results} ===")
            print(f"=== DB 통계 조회 결과: {stats_result} ===")

            if diagnosis_results:
                if diagnosis_results[0].get('product_name'):
                    product_name = diagnosis_results[0]['product_name']

                for row in diagnosis_results:
                    corrected_type = row.get('is_corrected')
                    score = row.get('geo_score', 0)
                    
                    if corrected_type == 0:
                        pre_score = score
                    elif corrected_type == 1:
                        post_score = score
            else:
                print("⚠️ DB에서 진단 정보(geo_score)를 찾지 못해 0점으로 설정합니다.")
            # [Tab 2]
            # 3. 상세 진단 데이터 조회
            sql_detail = """
                SELECT *
                FROM geo_db.diagnosis_table
                WHERE page_id = %s
                AND model_ver = '3.2.3'
                AND is_corrected = 0
            """
            cursor.execute(sql_detail, (HARDCODED_PAGE_ID,))
            detail_result = cursor.fetchall()
            # [Tab 3]
            # 4. 최적화 산출물
            sql_correction = """
                SELECT
                    a.*,
                    b.json_ld_contents
                FROM geo_db.geo_correction_table a,
                    geo_db.raw_data_table b
                WHERE a.page_id = b.page_id
                AND a.page_id = %s
            """
            cursor.execute(sql_correction, (HARDCODED_PAGE_ID,))
            correction_result = cursor.fetchall()

            print(f"=== DB GEO 교정 결과: {correction_result} ===") 

    except Exception as e:
        print(f"DB 연결/조회 error: {e}")
    finally:
        if 'conn' in locals() and conn and conn.open:
            conn.close()

    # is_corrected 값을 Key로 하는 딕셔너리로 변환
    stats_dict = {row['is_corrected']: row for row in stats_result}

    # is_corrected가 0인 데이터 (교정 전)
    pre_data = stats_dict.get(0, {})
    pre_cited_ratio = pre_data.get('is_cited_avg', 0)
    pre_cited_rank = pre_data.get('cited_rank', None)

    # is_corrected가 1인 데이터 (교정 후)
    post_data = stats_dict.get(1, {})
    post_cited_ratio = post_data.get('is_cited_avg', 0)
    post_cited_rank = post_data.get('cited_rank', None)

    # 1. 인용률 차이 계산
    if post_cited_ratio is not None and pre_cited_ratio is not None:
        growth_percent = round((float(post_cited_ratio) - float(pre_cited_ratio)) * 100, 1)
    else:
        growth_percent = None

    # 2. 순위 변동 계산
    if pre_cited_rank is None and post_cited_rank is not None:
        rank_up = None
    elif pre_cited_rank is not None and post_cited_rank is not None:
        try:
            rank_up = round(float(pre_cited_rank) - float(post_cited_rank), 1)
        except (ValueError, TypeError):
            rank_up = None
    else:
        rank_up = None
    # [Tab 2]
    # 3. 약점 / 주의 / 강점 계산
    if detail_result:
        detail_row = detail_result[0]
        detail_json = detail_row.get('raw_details')

        if detail_json:

            if isinstance(detail_json, str):
                detail_data = json.loads(detail_json)
            else:
                detail_data = detail_json

            details = detail_data.get('details', {})

            text_ratio = details.get('text_ratio', {})
            hybrid_search = details.get('hybrid_search', {})
            keyword_stuffing = details.get('keyword_stuffing', {})
            json_ld = details.get('json_ld', {})
            image_alt = details.get('image_alt', {})

            # image_alt 추가 계산
            detail_results = image_alt.get('detail_results', [])

            if len(detail_results) > 0:
                s2_total = 0

                for item in detail_results:
                    s2_total += float(
                        item.get('s2_relevance_score', 0) or 0
                    )

                avg_s2_relevance_score = (
                    s2_total / len(detail_results)
                )

            else:
                avg_s2_relevance_score = 0

            total_image_count = int(
                image_alt.get('total_image_count', 0) or 0
            )

            stuffing_image_count = int(
                image_alt.get('stuffing_image_count', 0) or 0
            )

            if total_image_count > 0:
                stuffing_ratio = (
                    stuffing_image_count / total_image_count
                )
            else:
                stuffing_ratio = 0


            # 1. text_ratio - len_body
            len_body = int(
                text_ratio.get('len_body', 0) or 0
            )

            if len_body < 400:

                weaknesses.append({
                    "id": "weak_1",
                    "condition": "text_ratio.len_body < 400",
                    "data": {
                        "title": "본문 텍스트 분량 부족",
                        "result": "본문 텍스트가 400자 미만으로 매우 적어 검색 엔진이 내용을 파악하기 어렵습니다.",
                        "solution": "상품의 특징, 상세 설명, 활용법 등 400자 이상의 충분한 텍스트 정보를 추가하세요."
                    }
                })

            elif 400 <= len_body < 800:

                warnings.append({
                    "id": "warn_1",
                    "condition": "text_ratio.len_body >= 400 && text_ratio.len_body < 800",
                    "data": {
                        "title": "본문 텍스트 분량 부족 주의",
                        "result": "본문 텍스트가 400자~800자 수준으로, 검색 엔진이 풍부한 정보를 수집하기에는 조금 부족합니다.",
                        "solution": "상품의 상세 설명, 활용 팁, 주의사항 등을 추가하여 800자 이상으로 텍스트를 보완해 보세요."
                    }
                })

            else:

                strengths.append({
                    "id": "strength_1",
                    "condition": "text_ratio.len_body >= 800",
                    "data": {
                        "title": "본문 텍스트 분량 충족",
                        "result": "본문 텍스트가 800자 이상으로 풍부하게 작성되어 검색 엔진이 상품 정보를 정확히 파악할 수 있습니다.",
                        "solution": "현재의 풍부한 텍스트 상태를 유지하고, 신제품 출시 시에도 상세한 정보를 텍스트로 계속 제공하세요."
                    }
                })


            # 2. text_ratio - raw_ratio
            raw_ratio = float(
                text_ratio.get('raw_ratio', 0) or 0
            )

            if raw_ratio < 0.1:

                weaknesses.append({
                    "id": "weak_2",
                    "condition": "text_ratio.raw_ratio < 0.1",
                    "data": {
                        "title": "본문 텍스트 대비 이미지 텍스트 과다",
                        "result": "본문 내 텍스트 비율이 10% 미만으로, 대부분의 정보가 이미지로만 제공되고 있습니다.",
                        "solution": "이미지 속 중요 안내 문구를 텍스트 형태로 직접 작성하여 텍스트 비중을 높이세요."
                    }
                })

            elif 0.1 <= raw_ratio < 0.4:

                warnings.append({
                    "id": "warn_2",
                    "condition": "text_ratio.raw_ratio >= 0.1 && text_ratio.raw_ratio < 0.4",
                    "data": {
                        "title": "본문 텍스트 대비 이미지 텍스트 과다 주의",
                        "result": "본문 내 텍스트 비율이 10%~40% 수준으로, 여전히 이미지 내 정보 비중이 높습니다.",
                        "solution": "이미지 속 주요 텍스트를 본문 텍스트로 일부 변환하면 검색 엔진 인덱싱에 도움이 됩니다."
                    }
                })

            else:

                strengths.append({
                    "id": "strength_2",
                    "condition": "text_ratio.raw_ratio >= 0.4",
                    "data": {
                        "title": "이미지 텍스트 대비 본문 텍스트 분량 적절",
                        "result": "이미지 중심의 정보 전달에 치우치지 않고 충분한 본문 텍스트 비율을 확보하고 있습니다.",
                        "solution": "검색 엔진 인덱싱에 최적화된 상태입니다. 지금처럼 텍스트와 이미지의 균형을 유지하세요."
                    }
                })


            # 3. hybrid_search - avg_cosine_sim_raw
            avg_cosine_sim_raw = float(
                hybrid_search.get('avg_cosine_sim_raw', 0) or 0
            )

            if avg_cosine_sim_raw < 0.25:

                weaknesses.append({
                    "id": "weak_3",
                    "condition": "hybrid_search.avg_cosine_sim_raw < 0.25",
                    "data": {
                        "title": "본문 텍스트와 소비자 질문의 연관도 미비",
                        "result": "본문 텍스트 내용이 소비자의 예상 검색 질의 및 검색 의도와 충분히 맞닿아 있지 않습니다.",
                        "solution": "고객이 주로 검색하는 주요 용어와 자주 묻는 질문 항목을 본문에 자연스럽게 녹여내세요."
                    }
                })

            elif 0.25 <= avg_cosine_sim_raw < 0.5:

                warnings.append({
                    "id": "warn_3",
                    "condition": "hybrid_search.avg_cosine_sim_raw >= 0.25 && hybrid_search.avg_cosine_sim_raw < 0.5",
                    "data": {
                        "title": "본문 텍스트와 소비자 질문의 연관도 미비 주의",
                        "result": "본문 내용과 소비자의 예상 검색 의도 간 연관성이 다소 아쉬운 수준입니다.",
                        "solution": "소비자가 자주 검색하는 의도와 연관된 상품의 용어나 설명 텍스트를 보강해 보세요."
                    }
                })

            else:

                strengths.append({
                    "id": "strength_3",
                    "condition": "hybrid_search.avg_cosine_sim_raw >= 0.5",
                    "data": {
                        "title": "본문 텍스트와 소비자 질문의 연관 존재",
                        "result": "본문 내용이 소비자의 예상 검색 질의 및 의도와 잘 일치하여 상위 노출에 유리합니다.",
                        "solution": "고객이 검색하는 맥락을 잘 반영하고 있으므로 현재의 주요 설명 키워드 구조를 유지하세요."
                    }
                })


            # 4. hybrid_search - avg_lexical_overlap
            avg_lexical_overlap = float(
                hybrid_search.get('avg_lexical_overlap', 0) or 0
            )

            if avg_lexical_overlap < 0.25:

                weaknesses.append({
                    "id": "weak_4",
                    "condition": "hybrid_search.avg_lexical_overlap < 0.25",
                    "data": {
                        "title": "본문 텍스트와 소비자 질문의 키워드 일치도 미비",
                        "result": "소비자가 검색창에 입력하는 핵심 키워드 단어들이 본문 텍스트에 거의 포함되어 있지 않습니다.",
                        "solution": "타겟 검색어와 핵심 명사 키워드를 상세 페이지 텍스트에 포함시켜 인덱싱 확률을 높이세요."
                    }
                })

            elif 0.25 <= avg_lexical_overlap < 0.5:

                warnings.append({
                    "id": "warn_4",
                    "condition": "hybrid_search.avg_lexical_overlap >= 0.25 && hybrid_search.avg_lexical_overlap < 0.5",
                    "data": {
                        "title": "본문 텍스트와 소비자 질문의 키워드 일치도 미비 주의",
                        "result": "소비자가 주로 검색하는 핵심 키워드가 본문 텍스트에 일부만 포함되어 있습니다.",
                        "solution": "타겟 검색 키워드를 본문 내 주요 문맥에 자연스럽게 추가하여 일치도를 높이세요."
                    }
                })

            else:

                strengths.append({
                    "id": "strength_4",
                    "condition": "hybrid_search.avg_lexical_overlap >= 0.5",
                    "data": {
                        "title": "본문 텍스트와 소비자 질문의 키워드 일치",
                        "result": "소비자가 자주 입력하는 핵심 검색 키워드들이 본문 텍스트에 높은 비율로 포함되어 있습니다.",
                        "solution": "타겟 키워드가 적절히 배치되어 있으니, 향후 시즌별 인기 검색어도 지속해서 갱신해 보세요."
                    }
                })


            # 5. keyword_stuffing - dup_penalty
            dup_penalty = float(
                keyword_stuffing.get('dup_penalty', 0) or 0
            )

            if dup_penalty == 0.25:

                weaknesses.append({
                    "id": "weak_5",
                    "condition": "keyword_stuffing.dup_penalty == 0.25",
                    "data": {
                        "title": "상품명에 중복되는 단어 사용",
                        "result": "상품명에 동일하거나 유사한 단어가 반복되어 어뷰징(스팸) 검색어로 오인될 위험이 있습니다.",
                        "solution": "중복된 단어를 제거하고, 핵심 단어 위주로 깔끔하고 명확하게 상품명을 정돈하세요."
                    }
                })

            elif dup_penalty == 0.1:

                warnings.append({
                    "id": "warn_5",
                    "condition": "keyword_stuffing.dup_penalty == 0.1",
                    "data": {
                        "title": "상품명에 중복되는 단어 사용 주의",
                        "result": "상품명에 일부 유사하거나 중복된 단어가 포함되어 가독성을 저해할 수 있습니다.",
                        "solution": "불필요하게 중복된 키워드를 정돈하여 더 간결한 상품명으로 구성하세요."
                    }
                })

            elif dup_penalty == 0:

                strengths.append({
                    "id": "strength_5",
                    "condition": "keyword_stuffing.dup_penalty == 0",
                    "data": {
                        "title": "상품명에 중복되는 단어 없음",
                        "result": "상품명에 무의미하게 중복되거나 반복되는 단어가 없어 명확하고 가독성이 높습니다.",
                        "solution": "어뷰징 위험이 없는 정돈된 상품명입니다. 현재 스타일을 계속 유지해 주세요."
                    }
                })


            # 6. keyword_stuffing - pattern_penalty
            pattern_penalty = float(
                keyword_stuffing.get('pattern_penalty', 0) or 0
            )

            if pattern_penalty == 0.25:

                weaknesses.append({
                    "id": "weak_6",
                    "condition": "keyword_stuffing.pattern_penalty == 0.25",
                    "data": {
                        "title": "특정 접미사 과다 사용",
                        "result": "상품명 끝에 특정 수식어나 동일한 접미사가 무분별하게 반복되어 스팸 분류 위험이 높습니다.",
                        "solution": "불필요한 반복 수식어구를 삭제하고 인위적이지 않은 가독성 높은 상품명으로 수정하세요."
                    }
                })

            elif pattern_penalty == 0.1:

                warnings.append({
                    "id": "warn_6",
                    "condition": "keyword_stuffing.pattern_penalty == 0.1",
                    "data": {
                        "title": "특정 접미사 과다 사용 주의",
                        "result": "상품명 뒤에 특정 수식어나 패턴이 다소 반복되어 어색하게 느껴질 수 있습니다.",
                        "solution": "반복되는 수식어구를 정돈하고 자연스러운 문장형 또는 명사형으로 마무리하세요."
                    }
                })

            elif pattern_penalty == 0:

                strengths.append({
                    "id": "strength_6",
                    "condition": "keyword_stuffing.pattern_penalty == 0",
                    "data": {
                        "title": "특정 접미사를 적절히 사용 또는 사용하지 않음",
                        "result": "특정 수식어나 접미사가 무분별하게 남발되지 않아 자연스럽고 가독성이 우수합니다.",
                        "solution": "검색 스팸 분류 위험이 없는 깔끔한 상품명 형태를 잘 유지하고 있습니다."
                    }
                })


            # 7. keyword_stuffing - len_penalty
            len_penalty = float(
                keyword_stuffing.get('len_penalty', 0) or 0
            )

            if len_penalty == 0.25:

                weaknesses.append({
                    "id": "weak_7",
                    "condition": "keyword_stuffing.len_penalty == 0.25",
                    "data": {
                        "title": "상품명에 사용된 명사 개수 과다",
                        "result": "상품명에 너무 많은 명사가 나열되어 검색 엔진이 제품의 본질을 파악하기 어렵습니다.",
                        "solution": "핵심 명사 1~2개와 꼭 필요한 카테고리 정보만 남기고 상품명을 단축하세요."
                    }
                })

            elif len_penalty == 0.1:

                warnings.append({
                    "id": "warn_7",
                    "condition": "keyword_stuffing.len_penalty == 0.1",
                    "data": {
                        "title": "상품명에 사용된 명사 개수 과다 주의",
                        "result": "상품명에 명사 단어가 다소 많이 나열되어 있어 핵심 제품명이 잘 드러나지 않습니다.",
                        "solution": "가장 중요한 핵심 키워드 위주로 상품명을 조금 더 축약해 보세요."
                    }
                })

            elif len_penalty == 0:

                strengths.append({
                    "id": "strength_7",
                    "condition": "keyword_stuffing.len_penalty == 0",
                    "data": {
                        "title": "상품명에 사용된 명사 개수 적절",
                        "result": "불필요한 명사 나열 없이 핵심 키워드로만 깔끔하게 상품명이 작성되었습니다.",
                        "solution": "검색 엔진이 제품의 본질을 명확히 파악할 수 있는 적절한 상품명 구조입니다."
                    }
                })


            # 8. keyword_stuffing - delim_penalty
            delim_penalty = float(
                keyword_stuffing.get('delim_penalty', 0) or 0
            )

            if delim_penalty == 0.25:

                weaknesses.append({
                    "id": "weak_8",
                    "condition": "keyword_stuffing.delim_penalty == 0.25",
                    "data": {
                        "title": "상품명에 사용된 구분자 개수 과다",
                        "result": "상품명 내 특수문자나 구분자가 과도하게 사용되어 검색 가독성을 저해합니다.",
                        "solution": "불필요한 특수문자나 기호를 제거하고 띄어쓰기 중심으로 간결하게 작성하세요."
                    }
                })

            elif delim_penalty == 0.1:

                warnings.append({
                    "id": "warn_8",
                    "condition": "keyword_stuffing.delim_penalty == 0.1",
                    "data": {
                        "title": "상품명에 사용된 구분자 개수 과다 주의",
                        "result": "상품명 내 특수문자나 구분자가 다소 많이 사용되어 산만해 보일 수 있습니다.",
                        "solution": "불필요한 구분자를 줄이고 띄어쓰기를 활용하여 정돈해 주세요."
                    }
                })

            elif delim_penalty == 0:

                strengths.append({
                    "id": "strength_8",
                    "condition": "keyword_stuffing.delim_penalty == 0",
                    "data": {
                        "title": "상품명에 사용된 구분자 적절",
                        "result": "특수문자나 구분자가 과도하게 사용되지 않아 직관적이고 깔끔한 가독성을 제공합니다.",
                        "solution": "사용자와 검색 엔진 모두에게 가독성이 높으므로 현재의 기호 사용 방식을 유지하세요."
                    }
                })


            # JSON-LD 점수
            parsing_score = float(
                json_ld.get('parsing_score', 0) or 0
            )

            density_score = float(
                json_ld.get('density_score', 0) or 0
            )

            clothing_score = float(
                json_ld.get('clothing_score', 0) or 0
            )

            trust_score = float(
                json_ld.get('trust_score', 0) or 0
            )


            # 9. json_ld - parsing_score
            # weak_9 발생 시 10~12번은 검사하지 않음
            if parsing_score == 0:

                weaknesses.append({
                    "id": "weak_9",
                    "condition": "json_ld.parsing_score == 0",
                    "data": {
                        "title": "json-ld 파싱 에러",
                        "result": "페이지 내 구조화 데이터(JSON-LD) 문법 오류로 인해 검색 엔진이 상세 정보를 읽지 못했습니다.",
                        "solution": "JSON-LD 스크립트의 쉼표, 따옴표, 괄호 등 문법 오류를 수정하여 정상 파싱되도록 고치세요."
                    }
                })

            else:

                if 0.3 <= parsing_score <= 0.7:

                    warnings.append({
                        "id": "warn_9",
                        "condition": "json_ld.parsing_score >= 0.3 && json_ld.parsing_score <= 0.7",
                        "data": {
                            "title": "json-ld 구조화 정도 미비",
                            "result": "구조화 데이터(JSON-LD) 파싱은 성공했으나, 핵심 속성이 일부 누락되어 완성도가 떨어집니다.",
                            "solution": "누락된 JSON-LD 필수 및 권장 속성들을 보완하여 스키마 구조를 개선하세요."
                        }
                    })

                elif parsing_score == 1:

                    strengths.append({
                        "id": "strength_9",
                        "condition": "json_ld.parsing_score == 1",
                        "data": {
                            "title": "json-ld 유효성 충족",
                            "result": "구조화 데이터(JSON-LD) 문법이 올바르게 준수되어 검색 엔진이 페이지 정보를 수집하기 용이합니다.",
                            "solution": "현재의 표준 스키마 구조를 유지하고 추가 상품 등록 시에도 동일한 형식을 적용하세요."
                        }
                    })


                # 10. json_ld - density_score
                if density_score == 0:

                    weaknesses.append({
                        "id": "weak_10",
                        "condition": "json_ld.density_score == 0",
                        "data": {
                            "title": "description 분량 미약",
                            "result": "JSON-LD 메타 데이터 내 description 속성값이 너무 짧거나 작성되지 않았습니다.",
                            "solution": "description 항목에 제품의 핵심 요약 정보를 충분한 길이의 설명글로 보완하세요."
                        }
                    })

                elif 0.35 <= density_score <= 0.7:

                    warnings.append({
                        "id": "warn_10",
                        "condition": "json_ld.density_score >= 0.35 && json_ld.density_score <= 0.7",
                        "data": {
                            "title": "description 분량 미비",
                            "result": "JSON-LD 메타 데이터 내 description 길이가 다소 짧아 검색 엔진 전달력이 약합니다.",
                            "solution": "상품의 핵심 특징을 포함하여 description 설명을 조금 더 상세하게 작성하세요."
                        }
                    })

                elif density_score == 1:

                    strengths.append({
                        "id": "strength_10",
                        "condition": "json_ld.density_score == 1",
                        "data": {
                            "title": "description 분량 충족",
                            "result": "JSON-LD 내 description 속성에 제품의 요약 정보가 충분하고 명확하게 작성되어 있습니다.",
                            "solution": "검색 결과에 풍부한 정보가 전달될 수 있는 상태이므로 현재의 설명 수준을 유지하세요."
                        }
                    })


                # 11. json_ld - clothing_score
                if clothing_score == 0:

                    weaknesses.append({
                        "id": "weak_11",
                        "condition": "json_ld.clothing_score == 0",
                        "data": {
                            "title": "의류 도메인 권장 속성 미포함",
                            "result": "의류 상품 노출을 위한 핵심 권장 속성(color, size, material, brand 등)이 빠져 있습니다.",
                            "solution": "의류 카테고리에 해당하는 표준 JSON-LD 속성 항목을 추가하여 스키마를 보완하세요."
                        }
                    })

                elif 0.25 <= clothing_score <= 0.5:

                    warnings.append({
                        "id": "warn_11",
                        "condition": "json_ld.clothing_score >= 0.25 && json_ld.clothing_score <= 0.5",
                        "data": {
                            "title": "의류 도메인 권장 속성 미비",
                            "result": "의류 카테고리 추천 속성(color, size, brand 등) 중 일부가 빠져 있습니다.",
                            "solution": "빠진 의류 속성값들을 JSON-LD 스키마에 추가 기입하여 노출 확률을 높이세요."
                        }
                    })

                elif clothing_score == 1:

                    strengths.append({
                        "id": "strength_11",
                        "condition": "json_ld.clothing_score == 1",
                        "data": {
                            "title": "의류 도메인 권장 속성 포함",
                            "result": "의류 상품에 필요한 권장 속성(color, size, material, brand 등)이 포함되어 있습니다.",
                            "solution": "카테고리 특화 검색 및 필터링에 유리한 상태이므로 현재 속성 구성을 유지하세요."
                        }
                    })


                # 12. json_ld - trust_score
                if trust_score == 0:

                    weaknesses.append({
                        "id": "weak_12",
                        "condition": "json_ld.trust_score == 0",
                        "data": {
                            "title": "회사 신뢰도 및 연결성 속성 미포함",
                            "result": "판매자 및 브랜드 신뢰도를 증명하는 연관 속성이 누락되었습니다.",
                            "solution": "브랜드 정보 및 공식 판매처 정보를 JSON-LD 스키마에 올바르게 포함시키세요."
                        }
                    })

                elif 0.33 <= trust_score <= 0.67:

                    warnings.append({
                        "id": "warn_12",
                        "condition": "json_ld.trust_score >= 0.33 && json_ld.trust_score <= 0.67",
                        "data": {
                            "title": "회사 신뢰도 및 연결성 속성 미비",
                            "result": "판매처 및 브랜드 연결성을 증명하는 속성 항목이 일부 누락되었습니다.",
                            "solution": "공식 브랜드명 및 관련 정보를 JSON-LD 속성에 추가하여 신뢰도 점수를 올리세요."
                        }
                    })

                elif trust_score == 1:

                    strengths.append({
                        "id": "strength_12",
                        "condition": "json_ld.trust_score == 1",
                        "data": {
                            "title": "회사 신뢰도 및 연결성 속성 포함",
                            "result": "판매자 및 브랜드 출처를 증명하는 연결 속성들이 포함되어 있습니다.",
                            "solution": "검색 엔진에 브랜드 신뢰도를 전달할 수 있는 상태이므로 현재 속성 구성을 지속하세요."
                        }
                    })


            # 13. image_alt - avg_score
            # weak_13 발생 시 14~15번은 검사하지 않음
            avg_score = float(
                image_alt.get('avg_score', 0) or 0
            )

            if avg_score < 0.2:

                weaknesses.append({
                    "id": "weak_13",
                    "condition": "image_alt.avg_score < 0.2",
                    "data": {
                        "title": "alt 속성 미포함",
                        "result": "상세페이지 내 주요 이미지들에 대체 텍스트(alt 속성)가 충분히 설정되어 있지 않습니다.",
                        "solution": "이미지 태그에 해당 이미지가 나타내는 내용을 설명하는 alt 텍스트를 기입하세요."
                    }
                })

            else:

                if 0.5 <= avg_score < 1:

                    warnings.append({
                        "id": "warn_13",
                        "condition": "image_alt.avg_score >= 0.5 && image_alt.avg_score < 1",
                        "data": {
                            "title": "alt 속성 보완 필요",
                            "result": "일부 이미지에 대체 텍스트(alt 속성)가 누락되었거나 내용이 다소 부족합니다.",
                            "solution": "대체 텍스트가 빠진 이미지들을 확인하고 설명을 추가해 보세요."
                        }
                    })

                elif avg_score == 1:

                    strengths.append({
                        "id": "strength_13",
                        "condition": "image_alt.avg_score == 1",
                        "data": {
                            "title": "alt 속성 우수",
                            "result": "모든 이미지에 시각 정보를 정확히 설명하는 alt 속성이 명확하게 작성되어 있습니다.",
                            "solution": "웹 접근성과 이미지 검색 노출을 모두 만족하는 상태이므로 현재 구성을 유지하세요."
                        }
                    })


                # 14. s2_relevance_score 평균
                if avg_s2_relevance_score < 0.1:

                    weaknesses.append({
                        "id": "weak_14",
                        "condition": "image_alt.detail_results.s2_relevance_score_avg < 0.1",
                        "data": {
                            "title": "내용과 관계없는 alt 속성값",
                            "result": "이미지의 실제 시각 내용과 alt 속성에 기입된 대체 텍스트 내용이 부합하지 않습니다.",
                            "solution": "이미지에 보이는 사물, 텍스트, 정보를 있는 그대로 정확하게 설명하는 alt 텍스트로 고치세요."
                        }
                    })

                elif 0.1 <= avg_s2_relevance_score < 0.2:

                    warnings.append({
                        "id": "warn_14",
                        "condition": "image_alt.detail_results.s2_relevance_score_avg >= 0.1 && image_alt.detail_results.s2_relevance_score_avg < 0.2",
                        "data": {
                            "title": "내용과 관계없는 alt 속성값 일부 포함",
                            "result": "몇몇 이미지의 alt 텍스트 내용이 실제 이미지 시각 정보와 잘 맞지 않습니다.",
                            "solution": "이미지 본래의 의미를 직관적으로 설명하는 텍스트로 alt 속성을 수정하세요."
                        }
                    })


                # 15. stuffing_image_count / total_image_count
                if stuffing_ratio >= 0.5:

                    weaknesses.append({
                        "id": "weak_15",
                        "condition": "image_alt.stuffing_image_count / total_image_count >= 0.5",
                        "data": {
                            "title": "alt 속성값 키워드 스터핑",
                            "result": "절반 이상의 이미지 alt 속성에 검색 노출만을 노린 무의미한 키워드가 반복 사용되고 있습니다.",
                            "solution": "스팸성 키워드 반복을 제거하고 각 이미지의 구체적인 설명을 자연스럽게 작성하세요."
                        }
                    })

                elif 0.2 <= stuffing_ratio < 0.5:

                    warnings.append({
                        "id": "warn_15",
                        "condition": "image_alt.stuffing_image_count / total_image_count >= 0.2 && image_alt.stuffing_image_count / total_image_count < 0.5",
                        "data": {
                            "title": "alt 속성값 키워드 스터핑 주의",
                            "result": "일부 이미지 alt 속성에 검색 노출용 반복 키워드가 사용된 정황이 있습니다.",
                            "solution": "인위적인 키워드 반복을 피하고 이미지 자체를 설명하는 자연스러운 문장으로 고치세요."
                        }
                    })

        else:
            print("⚠️ raw_details 값이 비어 있습니다.")

    else:
        print(
            "⚠️ 지정된 page_id에서 "
            "model_ver='3.2.3', is_corrected=0인 "
            "진단 데이터를 찾지 못했습니다."
        )

    # 터미널 확인
    print("=== weaknesses ===")
    print(weaknesses)

    print("=== warnings ===")
    print(warnings)

    print("=== strengths ===")
    print(strengths)

    # 더미 산출물 세팅 주석처리
    # spec_dummy = "소재: 100% 오가닉 코튼 | 치수: 가로 50cm x 세로 70cm"
    # qa_dummy = [
    #     {"q": "세탁기 사용이 가능한가요?", "a": "찬물에 중성세제를 사용하여 울코스 세탁을 권장합니다."},
    #     {"q": "교환 및 반품 기간은 어떻게 되나요?", "a": "수령 후 7일 이내에 신청해 주시면 가능합니다."}
    # ]

    # accordion_html = build_accordion_html(spec_dummy, qa_dummy)

    # json_ld_schema = {
    #     "@context": "https://schema.org/",
    #     "@type": "Product",
    #     "name": product_name,
    #     "description": "100% 국제 인증 오가닉 코튼으로 제작된 친환경 침구 세트입니다.",
    #     "hasMerchantReturnPolicy": {
    #         "@type": "MerchantReturnPolicy",
    #         "applicableCountry": "KR",
    #         "returnPolicyCategory": "https://schema.org/MerchantReturnFiniteReturnWindow",
    #         "merchantReturnDays": 7,
    #         "returnMethod": "https://schema.org/ReturnByMail"
    #     },
    #     "offers": {
    #         "@type": "Offer",
    #         "shippingDetails": {
    #             "@type": "OfferShippingDetails",
    #             "shippingRate": {
    #                 "@type": "MonetaryAmount",
    #                 "value": "3000",
    #                 "currency": "KRW"
    #             }
    #         }
    #     }
    # }

    # [Tab 3]
    # 최적화 산출물
    if correction_result:
        correction_row = correction_result[0]

        # 상품명 추천
        title_dummy = correction_row.get('corrected_title')

        # 제품 상세 설명
        spec_dummy = correction_row.get('corrected_body')

        # Q&A
        qa_raw = correction_row.get('recommended_faq_list')
        if isinstance(qa_raw, str):
            qa_data = json.loads(qa_raw)
        else:
            qa_data = qa_raw
        qa_dummy = []

        for qa in qa_data:
            qa_dummy.append({
                "q": qa["question"],
                "a": qa["answer"]
            })

        # 구조화 데이터
        json_ld_schema = correction_row.get('json_ld_code')

        # 구조화 데이터 - 원본
        original_json_ld = correction_row.get('json_ld_contents')

        json_ld_added = []
        json_ld_improved = []

        original_product = {}
        corrected_product = {}
        original_offer = {}
        corrected_offer = {}

        # 구조화 데이터 비교
        if original_json_ld and json_ld_schema:

            try:
                if isinstance(original_json_ld, str):
                    original_data = json.loads(original_json_ld)
                else:
                    original_data = original_json_ld

                if isinstance(json_ld_schema, str):
                    corrected_data = json.loads(json_ld_schema)
                else:
                    corrected_data = json_ld_schema


                # 원본 Product 찾기
                if isinstance(original_data, list):

                    for item in original_data:

                        if isinstance(item, dict):

                            if item.get("@type") == "Product":
                                original_product = item
                                break

                            elif "@graph" in item:

                                for graph_item in item["@graph"]:

                                    if isinstance(graph_item, dict) and graph_item.get("@type") == "Product":
                                        original_product = graph_item
                                        break

                                if original_product:
                                    break


                elif isinstance(original_data, dict):

                    if original_data.get("@type") == "Product":
                        original_product = original_data

                    elif "@graph" in original_data:

                        for item in original_data["@graph"]:

                            if isinstance(item, dict) and item.get("@type") == "Product":
                                original_product = item
                                break


                # AI 최적화 Product 찾기
                if isinstance(corrected_data, list):

                    for item in corrected_data:

                        if isinstance(item, dict):

                            if item.get("@type") == "Product":
                                corrected_product = item
                                break

                            elif "@graph" in item:

                                for graph_item in item["@graph"]:

                                    if isinstance(graph_item, dict) and graph_item.get("@type") == "Product":
                                        corrected_product = graph_item
                                        break

                                if corrected_product:
                                    break


                elif isinstance(corrected_data, dict):

                    if corrected_data.get("@type") == "Product":
                        corrected_product = corrected_data

                    elif "@graph" in corrected_data:

                        for item in corrected_data["@graph"]:

                            if isinstance(item, dict) and item.get("@type") == "Product":
                                corrected_product = item
                                break


                # 추가 및 보완된 데이터
                compare_fields = [
                    "description",
                    "category",
                    "material",
                    "color",
                    "size"
                ]

                for field in compare_fields:

                    original_value = original_product.get(field)
                    corrected_value = corrected_product.get(field)

                    original_empty = (
                        original_value is None
                        or original_value == ""
                        or original_value == []
                        or original_value == {}
                    )

                    corrected_empty = (
                        corrected_value is None
                        or corrected_value == ""
                        or corrected_value == []
                        or corrected_value == {}
                    )

                    if original_empty and not corrected_empty:

                        if field == "description":
                            json_ld_added.append(f"AI 검색에 최적화된 상세 설명문 작성 ({len(str(corrected_value))}자)")
                        else:
                            json_ld_added.append(f"{field} ({corrected_value})")


                # Product @id 추가
                original_id = original_product.get("@id")
                corrected_id = corrected_product.get("@id")

                if not original_id and corrected_id:
                    json_ld_added.append("Product @id (상품 식별자 추가)")


                # brand 추가
                original_brand = original_product.get("brand")
                corrected_brand = corrected_product.get("brand")

                if not original_brand and corrected_brand:
                    json_ld_added.append("brand (브랜드 정보 추가)")


                # Offer 데이터 가져오기
                original_offers = original_product.get("offers")
                corrected_offers = corrected_product.get("offers")

                if isinstance(original_offers, list):

                    if len(original_offers) > 0:
                        original_offer = original_offers[0]

                elif isinstance(original_offers, dict):
                    original_offer = original_offers


                if isinstance(corrected_offers, list):

                    if len(corrected_offers) > 0:
                        corrected_offer = corrected_offers[0]

                elif isinstance(corrected_offers, dict):
                    corrected_offer = corrected_offers


                # Offer @type 추가
                if not original_offer.get("@type") and corrected_offer.get("@type"):
                    json_ld_added.append("Offer @type (Offer 타입 추가)")


                # seller 추가
                if not original_offer.get("seller") and corrected_offer.get("seller"):
                    json_ld_added.append("seller (판매자 정보 추가)")


                # shippingDetails 추가
                if not original_offer.get("shippingDetails") and corrected_offer.get("shippingDetails"):
                    json_ld_added.append("shippingDetails (배송 정보 추가)")


                # hasMerchantReturnPolicy 추가
                original_return = original_product.get("hasMerchantReturnPolicy")
                corrected_return = corrected_product.get("hasMerchantReturnPolicy")

                if not original_return and corrected_return:
                    json_ld_added.append("hasMerchantReturnPolicy (반품 정책 추가)")


                # 추가 및 보완된 데이터 중복 제거
                added_result = []

                for item in json_ld_added:
                    if item not in added_result:
                        added_result.append(item)

                json_ld_added = added_result


                # 제외 및 개선된 항목
                original_name = original_product.get("name", "")
                corrected_name = corrected_product.get("name", "")


                # 상품명 전체 표현 개선
                if original_name and corrected_name:

                    if original_name != corrected_name:
                        json_ld_improved.append("상품명 표현을 AI 검색에 적합하게 개선")


                # 상품명 홍보성 표현 제거
                marketing_words = [
                    "핏보장",
                    "고퀄",
                    "고퀄리티",
                    "주문폭주",
                    "모델추천",
                    "MD추천",
                    "md추천",
                    "BEST",
                    "강추",
                    "특가",
                    "가격실화",
                    "모델소장"
                ]

                removed_marketing = []

                for word in marketing_words:

                    if word in original_name:

                        if word not in corrected_name:
                            removed_marketing.append(word)

                if len(removed_marketing) > 0:
                    json_ld_improved.append("상품명 홍보성 표현 일부 제외")


                # 상품명 중복 키워드 정리
                original_words = original_name.split()
                corrected_words = corrected_name.split()

                original_duplicate_count = 0
                corrected_duplicate_count = 0

                for word in original_words:

                    if original_words.count(word) > 1:
                        original_duplicate_count += 1

                for word in corrected_words:

                    if corrected_words.count(word) > 1:
                        corrected_duplicate_count += 1

                if original_duplicate_count > corrected_duplicate_count:
                    json_ld_improved.append("상품명 중복 키워드 정리")


                # 상세 설명문 개선
                original_description = original_product.get("description")
                corrected_description = corrected_product.get("description")

                if original_description and corrected_description:

                    if original_description != corrected_description:
                        json_ld_improved.append(
                            f"AI 검색에 최적화된 상세 설명문 개선 ({len(str(corrected_description))}자)"
                        )


                # 제외 및 개선된 항목 중복 제거
                improved_result = []

                for item in json_ld_improved:

                    if item not in improved_result:
                        improved_result.append(item)

                json_ld_improved = improved_result


            except Exception as e:
                print(f"JSON-LD 데이터 비교 error: {e}")


        print("=== 추가 및 보완된 데이터 ===")
        print(json_ld_added)

        print("=== ✂️ 제외 및 개선된 항목 ===")
        print(json_ld_improved)

        # 이미지 alt 추천
        alt_raw = correction_row.get('recommended_alt_list')
        if isinstance(alt_raw, str):
            alt_text = json.loads(alt_raw)
        else:
            alt_text = alt_raw

    else:
        print(
            "⚠️ geo_correction_table에서 "
            "해당 page_id의 교정 결과를 찾지 못했습니다."
        )
    #아코디언에 title_dummy 추가
    accordion_html = build_accordion_html(spec_dummy, qa_dummy, title_dummy)

    response_payload = {
        "status": "success",
        "page_id": HARDCODED_PAGE_ID,
        "product_name": product_name,
        "scores": {
            "before": pre_score,
            "after": post_score
        },
        "trends": {
            "citation_rate": {
                "before": pre_cited_ratio,
                "after": post_cited_ratio,
                "growth_percent": growth_percent
            },
            "citation_rank": {
                "before": pre_cited_rank,
                "after": post_cited_rank,
                "rank_up": rank_up
            }
        },
        "diagnosis": {
        "weaknesses": weaknesses,
        "warnings": warnings,
        "strengths": strengths
        },
        "outputs": {
            "accordion_html": accordion_html,
            # "json_ld": f'<script type="application/ld+json">\n{json.dumps(json_ld_schema, ensure_ascii=False, indent=2)}\n</script>',     더미 산출물 세팅 주석처리
            "json_ld": f'<script type="application/ld+json">\n{json_ld_schema}\n</script>',
            # "alt_text": 'alt="GEO 최적화 오가닉 코튼 침구 세트 - 프리미엄 소재 및 사용 가이드 안내"'      더미 산출물 세팅 주석처리
            "alt_text": alt_text,
            "json_ld_added": json_ld_added,
            "json_ld_improved":json_ld_improved
        }
    }

    return jsonify(response_payload), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)