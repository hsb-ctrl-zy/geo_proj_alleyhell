from scipy import stats
# from google.colab import files
import json
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import numpy as np
import os
import pandas as pd
import pymysql
import re
import seaborn as sns
from IPython.display import display
import sys

# 터미널 창의 입출력 한글 깨짐 방지
sys.stdout.reconfigure(encoding='utf8') #type: ignore
sys.stderr.reconfigure(encoding='utf8') #type: ignore

#MySQL 연결 정보 설정
host_name='211.212.230.86'
user_name='teamuser'
password_val='1638'
db_name='geo_db'
port_num=3306


#데이터베이스 연결
conn=pymysql.connect(
    host=host_name,
    user=user_name,
    password=password_val,
    database=db_name,
    port=port_num,
    charset='utf8'
)

def analyze_detail_compliance(json_str):
    res = {
        "syntax_ok": False,
        "desc_length": 0,
        "desc_valid": False,
        "has_brand": False,
        "has_color": False,
        "has_size": False,
        "has_material": False,
        "has_category": False,
        "apparel_full_compliance": False,
        "has_confidence": False,
        "entity_count": 0,
    }

    if pd.isna(json_str) or not str(json_str).strip():
        return res

    try:
        text = str(json_str).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"[\r\n\t]+", " ", text)

        data = json.loads(text)
        res["syntax_ok"] = True

        target_dict = {}
        if isinstance(data, list) and len(data) > 0:
            target_dict = data[0] if isinstance(data[0], dict) else {}
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                products = [
                    item
                    for item in data["@graph"]
                    if isinstance(item, dict)
                    and item.get("@type") in ["Product", "Clothing"]
                ]
                target_dict = (
                    products[0]
                    if products
                    else (
                        data["@graph"][0]
                        if isinstance(data["@graph"][0], dict)
                        else {}
                    )
                )
                res["entity_count"] = len(data["@graph"])
            else:
                target_dict = data
                res["entity_count"] = len(data.keys())

        if target_dict:
            # 1. Description 분량 진단 (20자 이상 시 규격 이행)
            desc = target_dict.get("description", "")
            if isinstance(desc, str):
                res["desc_length"] = len(desc)
                res["desc_valid"] = len(desc) >= 20

            # 2. 의류 도메인 필수 속성 진단
            res["has_brand"] = "brand" in target_dict
            res["has_color"] = "color" in target_dict
            res["has_size"] = any(
                k in target_dict for k in ["size", "offers", "hasVariant"]
            )
            res["has_material"] = "material" in target_dict
            res["has_category"] = "category" in target_dict

            res["apparel_full_compliance"] = (
                res["has_brand"] and res["has_color"] and res["has_size"] and res["has_material"] and res["has_category"]
            )

            # 3. 신뢰도(Confidence/Certainty) 필드 진단
            res["has_confidence"] = any(
                k in target_dict for k in ["confidence", "certainty"]
            )

    except Exception:
        pass

    return res
def analyze_detail_deep(json_str):
    res = {
        "syntax_ok": False,
        "desc_length": 0,
        "desc_category": "0) JSON 파싱 실패",
        "has_brand": False,
        "has_color": False,
        "has_size": False,
        "has_material": False,
        "has_category": False,
        "missing_attrs": [],
        "has_confidence": False,
        "confidence_val": None,
        "is_graph": False,
        "graph_node_types": [],
        "root_keys_cnt": 0,
    }

    if pd.isna(json_str) or not str(json_str).strip():
        res["desc_category"] = "0) 데이터 없음 (NULL)"
        return res

    # 1. JSON 파싱 및 정제 시도
    try:
        text = str(json_str).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"[\r\n\t]+", " ", text)

        data = json.loads(text)
        res["syntax_ok"] = True

    except Exception:
        return res

    # 2. 파싱 성공 객체 내에서 Target Dict 추출
    target = {}
    if isinstance(data, dict):
        if "@graph" in data and isinstance(data["@graph"], list):
            res["is_graph"] = True
            res["graph_node_types"] = [
                x.get("@type", "Unknown")
                for x in data["@graph"]
                if isinstance(x, dict)
            ]
            prods = [
                x
                for x in data["@graph"]
                if isinstance(x, dict)
                and x.get("@type") in ["Product", "Clothing"]
            ]
            target = (
                prods[0]
                if prods
                else (
                    data["@graph"][0]
                    if isinstance(data["@graph"][0], dict)
                    else {}
                )
            )
        else:
            target = data
            res["root_keys_cnt"] = len(data.keys())
    elif isinstance(data, list) and len(data) > 0:
        target = data[0] if isinstance(data[0], dict) else {}

    # 3. 파싱 성공 객체 대상 Description 및 속성 분석
    if target:
        desc = target.get("description", "")
        desc_str = str(desc).strip() if desc is not None else ""
        d_len = len(desc_str)
        res["desc_length"] = d_len

        if d_len == 0:
            res["desc_category"] = "1) 0자 (순수 누락)"
        elif d_len < 20:
            res["desc_category"] = "2) 1~19자 (단순 요약)"
        elif d_len < 50:
            res["desc_category"] = "3) 20~49자 (보통)"
        else:
            res["desc_category"] = "4) 50자 이상 (충실)"

        res["has_brand"] = "brand" in target
        res["has_color"] = "color" in target
        res["has_size"] = any(
            k in target for k in ["size", "offers", "hasVariant"]
        )
        res["has_material"] = "material" in target
        res["has_category"] = "category" in target

        missing = []
        if not res["has_brand"]:
            missing.append("brand")
        if not res["has_color"]:
            missing.append("color")
        if not res["has_size"]:
            missing.append("size/offers")
        if not res["has_material"]:
            missing.append("material")
        if not res["has_category"]:
            missing.append("category")
        res["missing_attrs"] = (
            " + ".join(missing) if missing else "없음 (전수 충족)"
        )

        if "confidence" in target:
            res["has_confidence"] = True
            res["confidence_val"] = target["confidence"]

    return res
def analyze_detail_properties(json_str):
    res = {
        'entity_count': 0,
        'attribute_count': 0,
        'has_brand': False,
        'has_material': False,
        'has_color': False,
        'has_size': False,
        'has_category': False,
        'apparel_full_compliance': False,
        'has_confidence': False,
        'desc_length': 0,
        'desc_valid': False
    }
    if pd.isna(json_str) or json_str is None:
        return res

    try:
        text = str(json_str).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        data = json.loads(text)

        target_dict = {}
        processed_entities = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    processed_entities.append(item)
                elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict):
                    processed_entities.append(item[0])
            if len(processed_entities) == 1 and '@graph' in processed_entities[0] and isinstance(processed_entities[0]['@graph'], list):
                processed_entities = [node for node in processed_entities[0]['@graph'] if isinstance(node, dict)]
            else:
                processed_entities = [item for item in data if isinstance(item, dict)]

        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                processed_entities = [item for item in data['@graph'] if isinstance(item, dict)]
            else:
                processed_entities = [data]

        res['entity_count'] = len(processed_entities)
        res['attribute_count'] = sum(len(entity.keys()) for entity in processed_entities)

        products = [item for item in processed_entities if item.get("@type") in ["Product", "Clothing"]]
        target_dict = products[0] if products else (processed_entities[0] if processed_entities else {})


        if target_dict:
            desc = target_dict.get("description", "")
            if isinstance(desc, str):
                res['desc_length'] = len(desc)
                res['desc_valid'] = len(desc) >= 20

            res['has_brand'] = 'brand' in target_dict
            res['has_material'] = 'material' in target_dict
            res['has_color'] = 'color' in target_dict
            res['has_size'] = any(k in target_dict for k in ['size', 'offers', 'hasVariant'])
            res['has_category'] = 'category' in target_dict

            res['apparel_full_compliance'] = (
                res['has_brand'] and res['has_color'] and res['has_size'] and res['has_material'] and res['has_category']
            )

            res['has_confidence'] = any(k in target_dict for k in ['confidence', 'conf_score', 'confidence_score', 'certainty'])

    except Exception:
        pass
    return res

def annotate_bars(ax, bars, is_percent=True, suffix=""):
    for bar in bars:
        height = bar.get_height()
        val_str = f'{height:.2f}%' if is_percent else f'{height:.2f}{suffix}'

        if is_percent and f'{height:.2f}' == '100.00':
            y_pos = height - 8
            va = 'top'
        else:
            y_pos = height + 2
            va = 'bottom'

        ax.annotate(val_str,
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha='center', va=va, fontsize=10, color='white')

def check_description_detail(json_str):
    res = {'desc_len': 0, 'desc_over_50': False}
    if pd.isna(json_str) or not json_str:
        return res
    try:
        text = str(json_str).strip()
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        data = json.loads(text)

        nodes = data if isinstance(data, list) else [data]
        all_nodes = []
        for n in nodes:
            if isinstance(n, dict):
                all_nodes.append(n)
                if '@graph' in n and isinstance(n['@graph'], list):
                    all_nodes.extend([g for g in n['@graph'] if isinstance(g, dict)])

        for n in all_nodes:
            desc = n.get('description', '')
            if isinstance(desc, str) and desc:
                length = len(desc)
                if length > res['desc_len']:
                    res['desc_len'] = length
                if length >= 50:
                    res['desc_over_50'] = True
    except Exception:
        pass
    return res

def check_json_ld_compliance(json_str):
    res = {
        'syntax_ok': False,
        'has_context': False,
        'has_type': False,
        'full_compliance': False
    }

    if pd.isna(json_str) or json_str is None:
        return res

    try:
        text = str(json_str).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)

        data = json.loads(text)
        res['syntax_ok'] = True

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict):
            res['has_context'] = '@context' in data
            res['has_type'] = '@type' in data

        res['full_compliance'] = res['syntax_ok'] and res['has_context'] and res['has_type']
    except Exception:
        pass

    return res
def check_type_detail(json_str):
    res = {'root_type': False, 'graph_type': False, 'any_type': False}
    if pd.isna(json_str) or not json_str:
        return res
    try:
        text = str(json_str).strip()
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        data = json.loads(text)

        nodes = data if isinstance(data, list) else [data]
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if '@type' in n:
                res['root_type'] = True
            if '@graph' in n and isinstance(n['@graph'], list):
                for g in n['@graph']:
                    if isinstance(g, dict) and '@type' in g:
                        res['graph_type'] = True
                        break
        res['any_type'] = res['root_type'] or res['graph_type']
    except Exception:
        pass
    return res

def clean_and_validate_json_ld(json_str):
    if pd.isna(json_str) or json_str is None:
        return False

    try:
        text = str(json_str).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)

        text = re.sub(r'[\r\n\t]+', ' ', text)

        data = json.loads(text)

        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
            data = data[0]

        if isinstance(data, list):
            if not data:
                return False
            for item in data:
                if not (isinstance(item, dict) and '@context' in item and '@type' in item):
                    return False
            return True
        elif isinstance(data, dict):
            return '@context' in data and '@type' in data
        else:
            return False

    except Exception:
        return False

def create_single_bar(ax, values, title, ylabel):
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    annotate_bars(ax, bars)

def fetch_data(query):
    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
def get_field_rate(df, candidates):
    """후보 컬럼명 중 존재하는 컬럼의 평균 존재율(%)을 구하는 함수"""
    for col in candidates:
        if col in df.columns:
            return df[col].mean() * 100
    return 0.0
def validate_json_only(json_str):
    try:
        text = str(json_str).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)

        data = json.loads(text)
        if isinstance(data, list) and len(data) == 1:
            data = data[0]

        return isinstance(data, (dict, list))
    except Exception:
        return False

if conn:
    try:
        # 커서를 생성하여 쿼리 실행 준비
        with conn.cursor() as cursor:
            # 간단한 쿼리로 현재 연결된 데이터베이스 이름을 확인
            cursor.execute("SELECT DATABASE();")
            result = cursor.fetchone()
            if result and result[0]:
                print(f"데이터베이스 연결이 성공적으로 확인되었습니다. 현재 데이터베이스: {result[0]}")
            else:
                print("데이터베이스에 연결되었지만, 현재 데이터베이스 이름을 가져올 수 없습니다.")
    except pymysql.Error as e:
        print(f"데이터베이스 연결 확인 중 오류 발생: {e}")
        print("연결에 문제가 있을 수 있습니다.")
else:
    print("데이터베이스 연결 객체(conn)가 생성되지 않았습니다. 연결이 실패했을 수 있습니다.")
try:
    with conn.cursor() as cursor:
        sql_query = "SELECT text_contents FROM raw_data_table;"
        cursor.execute(sql_query)

        all_contents = cursor.fetchall()

        # 결과를 DataFrame으로 변환
        df_contents = pd.DataFrame(all_contents, columns=['text_contents'])

        print(f"총 {len(df_contents)}개의 텍스트를 가져왔습니다.")

except pymysql.Error as e:
    print(f"데이터베이스 쿼리 실행 중 오류 발생: {e}")
except Exception as e:
    print(f"예상치 못한 오류 발생: {e}")
# text_contents의 글자 수 계산 (NaN 값은 0으로 처리하거나 제외)
df_contents['text_length'] = df_contents['text_contents'].apply(lambda x: len(str(x)) if x is not None else 0)

# 각 행의 text_contents와 글자 수 출력
print("각 행의 텍스트 내용과 글자 수:")
display(df_contents[['text_contents', 'text_length']].head())

# 전체 글자 수에 대한 기술 통계량 (원래 코드)
print("\n텍스트 글자 수에 대한 기술 통계량:")
display(df_contents['text_length'].describe())
try:
    with conn.cursor() as cursor:
        sql_query_brand = "SELECT text_contents, brand_type FROM raw_data_table;"
        cursor.execute(sql_query_brand)

        all_data_with_brand = cursor.fetchall()

        # 결과를 DataFrame으로 변환
        df_with_brand = pd.DataFrame(all_data_with_brand, columns=['text_contents', 'brand_type'])

        print(f"총 {len(df_with_brand)}개의 텍스트와 브랜드 타입 데이터를 가져왔습니다.")

except pymysql.Error as e:
    print(f"데이터베이스 쿼리 실행 중 오류 발생: {e}")
except Exception as e:
    print(f"예상치 못한 오류 발생: {e}")
# text_contents의 글자 수 계산 (NaN 값은 0으로 처리하거나 제외)
df_with_brand['text_length'] = df_with_brand['text_contents'].apply(lambda x: len(str(x)) if x is not None else 0)

# brand_type별 평균 글자 수 분석
print("\nBrand Type별 텍스트 글자 수 평균:")
display(df_with_brand.groupby('brand_type')['text_length'].mean().reset_index())

# --- 사용자 제공 폰트 설정 ---
font_path = '/content/Pretendard－M.ttf'

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rc('font', family=font_name)
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    print(f"'{font_name}' 폰트 설정 완료.")
else:
    print(f"경로에 '{font_path}' 폰트 파일을 찾을 수 없습니다. 기본 폰트(나눔고딕 또는 대체 폰트)를 사용합니다.")
    try:
        fm.findfont('NanumGothic', rebuild_if_missing=True)
        plt.rc('font', family='NanumGothic')
        print("NanumGothic 폰트가 설치되어 있어 이를 사용합니다.")
    except Exception:
        plt.rc('font', family='DejaVu Sans')
        print("기본 폰트를 찾을 수 없어 'DejaVu Sans'를 사용합니다.")

# 모든 텍스트 및 축 요소 색상을 흰색으로 설정
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

print("Matplotlib 한글 폰트 및 디자인 설정 완료 (Pretendard 폰트 파일을 업로드해야 제대로 적용됩니다).")# 데이터 행 수(인덱스)와 글자 수를 이용한 산점도 그래프 시각화
fig = plt.figure(figsize=(15, 7))
plt.scatter(df_contents.index, df_contents['text_length'], alpha=0.6, s=10, color='#FF7171')

plt.title('데이터 행 번호에 따른 텍스트 글자 수', fontsize=15)
plt.xlabel('데이터 행 번호', fontsize=12)
plt.ylabel('글자 수', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('본문 텍스트 글자 수(scat).png')

plt.show()
fig = plt.figure(figsize=(15, 7))
ax = sns.histplot(
    data=df_contents,
    x=df_contents.index,
    y='text_length',
    binwidth=(1000, 100),
    cmap=sns.light_palette("#71FFE7", as_cmap=True),
    cbar=True,
    cbar_kws={'label': '빈도수'}
)

plt.title('데이터 행 번호에 따른 텍스트 글자 수 분포 (2D 히스토그램)', fontsize=15)
plt.xlabel('데이터 행 번호', fontsize=12)
plt.ylabel('글자 수', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 컬러바 레이블 및 틱 색상 설정 (rcParams에 의해 자동으로 적용되나 명시적으로 설정)
cbar = ax.collections[0].colorbar
if cbar is not None:
    cbar.set_label('빈도수', color='white')
    cbar.ax.yaxis.set_tick_params(colors='white')

plt.tight_layout()
plt.savefig('본문 텍스트 글자 수(hist)_2d.png')

plt.show()
# brand_type별 평균 글자 수 계산
avg_text_length_by_brand = df_with_brand.groupby('brand_type')['text_length'].mean().reset_index()

# 커스텀 팔레트 설정
color_map = {'대기업': '#FF7171', '소상공인': '#71FFE7'}

# 막대 그래프 시각화
fig = plt.figure(figsize=(8, 5))
ax = sns.barplot(x='brand_type', y='text_length', data=avg_text_length_by_brand, hue='brand_type', palette=color_map, legend=False)

plt.title('Brand Type별 평균 텍스트 글자 수', fontsize=15)
plt.xlabel('Brand Type', fontsize=12)
plt.ylabel('평균 글자 수', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 막대 위에 수치 표시
for p in ax.patches:
    if isinstance(p, Rectangle):  
        height = p.get_height()
        ax.annotate(f'{height:.1f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='white', xytext=(0, 5),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig('brand_type별_평균글자수.png')

plt.show()# 1. DB에서 브랜드명, brand_type, text_contents를 함께 가져오기
try:
    with conn.cursor() as cursor:
        sql_query_brand_detail = "SELECT text_contents, brand_type, brand_name FROM raw_data_table;"
        cursor.execute(sql_query_brand_detail)
        all_data_brand_detail = cursor.fetchall()

        # 새로운 DataFrame 생성
        df_brand_detail = pd.DataFrame(all_data_brand_detail, columns=['text_contents', 'brand_type', 'brand_name'])
        print(f"총 {len(df_brand_detail)}개의 데이터를 성공적으로 가져왔습니다.")
except Exception as e:
    print(f"오류 발생: {e}")

# 2. 글자 수 계산
df_brand_detail['text_length'] = df_brand_detail['text_contents'].apply(
    lambda x: len(str(x)) if x is not None else 0
)
# 1. 브랜드별 평균 글자 수 계산
avg_by_brand_all = df_brand_detail.groupby(['brand_name', 'brand_type'])['text_length'].mean().reset_index()

# 대기업 / 소상공인 데이터 분리 및 정렬
df_major = avg_by_brand_all[avg_by_brand_all['brand_type'] == '대기업'].sort_values(by='text_length', ascending=False)
df_small = avg_by_brand_all[avg_by_brand_all['brand_type'] == '소상공인'].sort_values(by='text_length', ascending=False)# 2. 1행 2열 구조의 서브플롯 생성
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# [좌측] 대기업 차트
sns.barplot(
    data=df_major,
    x='brand_name',
    y='text_length',
    ax=axes[0],
    color='#FF7171'
)
axes[0].set_title(f'대기업 브랜드별 평균 글자 수 (총 {len(df_major)}개)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('브랜드명', fontsize=12)
axes[0].set_ylabel('평균 글자 수', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# 대기업 막대 위 수치 표시
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f'{height:.1f}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=10, fontweight='bold', color='white', xytext=(0, 3),
                         textcoords='offset points')

# [우측] 소상공인 차트
sns.barplot(
    data=df_small,
    x='brand_name',
    y='text_length',
    ax=axes[1],
    color='#71FFE7'
)
axes[1].set_title(f'소상공인 브랜드별 평균 글자 수 (총 {len(df_small)}개)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('브랜드명', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# 소상공인 막대 위 수치 표시
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(f'{height:.1f}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, rotation=0, color='white', xytext=(0, 5),
                         textcoords='offset points')

# 여백 조절
axes[0].set_ylim(0, df_major['text_length'].max() * 1.15)
axes[1].set_ylim(0, df_small['text_length'].max() * 1.25)

plt.tight_layout()

plt.savefig('브랜드별_텍스트_비교.png')
plt.show()# geo_correction_table의 컬럼 목록 확인
try:
    with conn.cursor() as cursor:
        cursor.execute("DESCRIBE geo_correction_table;")
        columns = cursor.fetchall()
        print("=== geo_correction_table 컬럼 목록 ===")
        for col in columns:
            print(f"- {col[0]} ({col[1]})")
except Exception as e:
    print(f"컬럼 정보 조회 실패: {e}")# raw_data_table 컬럼 목록 확인
try:
    with conn.cursor() as cursor:
        cursor.execute("DESCRIBE raw_data_table;")
        columns = cursor.fetchall()
        print("=== raw_data_table 컬럼 목록 ===")
        for col in columns:
            print(f"- {col[0]} ({col[1]})")
except Exception as e:
    print(f"컬럼 정보 조회 실패: {e}")
# 두 테이블의 컬럼명 지정
CORRECTED_COLUMN_NAME = "corrected_body"
RAW_PK_NAME = "page_id"

try:
    with conn.cursor() as cursor:
        sql = f"""
        SELECT
            r.text_contents AS raw_text,
            c.{CORRECTED_COLUMN_NAME} AS corrected_text
        FROM raw_data_table r
        JOIN geo_correction_table c ON r.{RAW_PK_NAME} = c.page_id;
        """
        cursor.execute(sql)
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['raw_text', 'corrected_text'])

    # 글자 수 계산 (None 값은 0 처리)
    df['raw_length'] = df['raw_text'].apply(lambda x: len(str(x)) if x is not None else 0)
    df['corrected_length'] = df['corrected_text'].apply(lambda x: len(str(x)) if x is not None else 0)

    # 글자 수 차이 계산 (교정 후 - 교정 전)
    df['diff'] = df['corrected_length'] - df['raw_length']

    print(f"총 {len(df)}건의 데이터를 성공적으로 로드했습니다.\n")
    display(df[['raw_length', 'corrected_length', 'diff']].describe())

except Exception as e:
    print(f"데이터 로드 및 처리 중 오류 발생: {e}")
# 1. 대응표본 t-검정 수행 (교정 후 - 교정 전)
t_stat, p_value_ttest = stats.ttest_rel(df['corrected_length'], df['raw_length'])

# 2. 효과 크기(Cohen's d) 계산
diff_mean = df['diff'].mean()
diff_std = df['diff'].std(ddof=1)
cohens_d = diff_mean / diff_std if diff_std != 0 else 0

print("=== 1. 대응표본 t-검정 (Paired t-test) 결과 ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value_ttest:.4e}")
print(f"Cohen's d (효과 크기): {cohens_d:.4f}\n")

if p_value_ttest < 0.05:
    print("결과: 귀무가설 기각 (교정 전/후 글자 수 평균에 통계적으로 유의미한 차이가 있습니다.)")
else:
    print("결과: 귀무가설 채택 (교정 전/후 글자 수 평균 차이가 유의미하지 않습니다.)")
# 1. 윌콕슨 부호순위 검정 수행 (zero_method='wilcox'로 변화가 없는 0값 처리)
wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(
    df['corrected_length'],
    df['raw_length'],
    zero_method='wilcox'
)

# 2. 중앙값 계산
raw_median = df['raw_length'].median()
corrected_median = df['corrected_length'].median()
diff_median = df['diff'].median()

print("=== 2. 윌콕슨 부호순위 검정 (비모수) 결과 ===")
print(f"Statistic: {wilcoxon_stat:.4f}")
print(f"p-value: {p_value_wilcoxon:.4e}")
print(f"교정 전 중앙값: {raw_median}, 교정 후 중앙값: {corrected_median} (차이 중앙값: {diff_median})\n")

if p_value_wilcoxon < 0.05: #type: ignore
    print("결과: 귀무가설 기각 (교정 전/후 글자 수 중앙값에 통계적으로 유의미한 차이가 있습니다.)")
else:
    print("결과: 귀무가설 채택 (교정 전/후 글자 수 중앙값 차이가 유의미하지 않습니다.)")# 1. 차이(diff) 데이터의 정규성 검정
# D'Agostino's K-squared test
k2_stat, p_value_dagostino = stats.normaltest(df['diff'])

# Kolmogorov-Smirnov test (표준화 후 비교)
diff_z = (df['diff'] - df['diff'].mean()) / df['diff'].std()
ks_stat, p_value_ks = stats.kstest(diff_z, 'norm')
print("=== 3. 차이값(diff)의 정규성 검정 결과 ===")
print(f"[D'Agostino's Test] Statistic: {k2_stat:.4f}, p-value: {p_value_dagostino:.4e}")
print(f"[K-S Test]          Statistic: {ks_stat:.4f}, p-value: {p_value_ks:.4e}\n")
# 1. 교정 전 글자 수가 0인 데이터 개수 및 비율 계산
zero_count = (df['raw_length'] == 0).sum()
total_count = len(df)
zero_ratio = (zero_count / total_count) * 100

# 2. 왜도(Skewness) 및 첨도(Kurtosis) 계산
raw_skew = df['raw_length'].skew()
diff_skew = df['diff'].skew()

# 3. 데이터 시각화 결과 출력
print("=== [분석 1] 교정 전 텍스트 0자(빈값) 비율 ===")
print(f"- 총 데이터 건수: {total_count:,}건")
print(f"- 교정 전 0자인 데이터 건수: {zero_count:,}건")
print(f"- 0자 데이터 비율: {zero_ratio:.2f}%\n")

print("=== [분석 2] 데이터 왜도(Skewness) 측정 ===")
print(f"- 교정 전 글자 수(raw_length) 왜도: {raw_skew:.4f}")
print(f"- 글자 수 차이(diff) 왜도: {diff_skew:.4f}\n")

print("=== [분석 3] 정규성 검정 및 최종 판단 ===")
k2_stat, p_value = stats.normaltest(df['diff'])

if p_value < 0.05 and (zero_ratio > 10 or abs(diff_skew) > 1): #type: ignore
    print(f"1) 교정 전 데이터 중 0자인 비율이 {zero_ratio:.1f}%로 높습니다.")
    print(f"2) 차이값(diff)의 왜도가 {diff_skew:.2f}로 비대칭성이 강합니다.")
    print(f"3) 정규성 검정 결과(p-value={p_value:.4e}) 정규성이 최종 기각되었습니다.")
    print("-> 따라서 모수 검정(t-test) 대신 비모수 검정(Wilcoxon) 결과를 최종 판단 근거로 채택합니다.")

if p_value_dagostino < 0.05:
    print("[해석] 정규성 가정이 기각되었습니다. (p < 0.05)")
    print("-> 교정 전 글자 수가 0이거나 왜도가 높은 데이터 특성으로 인해 차이값이 정규분포를 따르지 않습니다.")
    print("-> 따라서 모수 검정(t-test)보다는 '2단계: 윌콕슨 부호순위 검정' 결과를 최종 판단 근거로 채택하는 것이 적절합니다.")
else:
    print("[해석] 정규성 가정을 만족합니다. (p >= 0.05)")
    print("-> '1단계: 대응표본 t-검정' 결과를 최종 판단 근거로 활용할 수 있습니다.")

plt.figure(figsize=(12, 7))
sns.kdeplot(x=df['raw_length'], color='#FF7171', fill=True, alpha=0.5, label='교정 전 글자 수')
sns.kdeplot(x=df['corrected_length'], color='#71FFE7', fill=True, alpha=0.5, label='교정 후 글자 수')

plt.title('교정 전 대비 교정 후 본문 텍스트 분량 변화', fontsize=16)
plt.xlabel('글자 수', fontsize=12)
plt.ylabel('밀도', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 4000)
plt.tight_layout()
plt.savefig('교정_전후_글자수_분포.png')
plt.show()
df_lengths = df[['raw_length', 'corrected_length']].melt(var_name='Length_Type', value_name='Length')
df_lengths['Length_Type'] = df_lengths['Length_Type'].map({
    'raw_length': '교정 전 글자 수',
    'corrected_length': '교정 후 글자 수'
})

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 좌측 서브플롯: 교정 전/후 글자 수 분포
sns.violinplot(
    x='Length_Type',
    y='Length',
    data=df_lengths,
    hue='Length_Type',
    palette={'교정 전 글자 수': '#FF7171', '교정 후 글자 수': '#71FFE7'},
    inner='quartile',
    ax=axes[0],
    legend=False
)
axes[0].set_title('교정 전/후 글자 수 분포 (바이올린 플롯)', fontsize=14)
axes[0].set_xlabel('글자 수 유형', fontsize=12)
axes[0].set_ylabel('글자 수', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# 우측 서브플롯: 글자 수 변화량 (diff) 분포
sns.violinplot(
    y=df['diff'],
    color='#FF7171',
    inner='quartile',
    ax=axes[1]
)
axes[1].set_title('글자 수 변화량 (diff) 분포 (바이올린 플롯)', fontsize=14)
axes[1].set_xlabel('변화량', fontsize=12)
axes[1].set_ylabel('글자 수 변화량 (교정 후 - 교정 전)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('글자수_분포_바이올린플롯.png')
plt.show()
fig = plt.figure(figsize=(15, 7))
ax = sns.histplot(
    data=df,
    x=df.index,
    y='diff',
    binwidth=(50, 50),
    cmap=sns.light_palette("#71FFE7", as_cmap=True),
    cbar=True,
    cbar_kws={'label': '빈도수'}
)

plt.title('데이터 행 번호에 따른 글자 수 차이 (diff) 분포 (2D 히스토그램)', fontsize=16)
plt.xlabel('데이터 행 번호', fontsize=12)
plt.ylabel('글자 수 변화량 (교정 후 - 교정 전)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 컬러바 레이블 및 틱 색상 설정
cbar = ax.collections[0].colorbar
if cbar is not None:
    cbar.set_label('빈도수', color='white')
    cbar.ax.yaxis.set_tick_params(colors='white')

plt.tight_layout()
plt.savefig('글자수_차이_분포_2D_히스토그램.png')
plt.show()

if conn:
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DATABASE();")
            result = cursor.fetchone()
            if result and result[0]:
                print(f"데이터베이스 연결이 성공적으로 확인되었습니다. 현재 데이터베이스: {result[0]}")
            else:
                print("데이터베이스에 연결되었지만, 현재 데이터베이스 이름을 가져올 수 없습니다.")
    except pymysql.Error as e:
        print(f"데이터베이스 연결 확인 중 오류 발생: {e}")
        print("연결에 문제가 있을 수 있습니다.")
else:
    print("데이터베이스 연결 객체(conn)가 생성되지 않았습니다. 연결이 실패했을 수 있습니다.")
# 1. DB 데이터 추출 함수 정의
# 2. 3-Way Join 쿼리로 데이터 병합
correct_merge_query = """
SELECT
    e.session_id,
    e.is_corrected,
    e.strictness_level,
    e.eval_sys_prompt_ver,
    l.page_id,
    r.json_ld_contents AS original_json_ld,
    g.json_ld_code AS corrected_json_ld,
    g.cor_sys_prompt_ver
FROM eval_session_table e
JOIN eval_log_table l ON e.session_id = l.session_id
LEFT JOIN raw_data_table r ON l.page_id = r.page_id
LEFT JOIN geo_correction_table g ON l.page_id = g.page_id;
"""

df_ab = fetch_data(correct_merge_query)# 3. 줄바꿈 및 특수문자 정제 후 JSON-LD 유효성 검증하는 전처리 함수
print("데이터 통합 완료! 총 행 수:", len(df_ab))# 1. 원본 및 교정본 유효성 검증 적용
df_ab['is_valid_original'] = df_ab['original_json_ld'].apply(clean_and_validate_json_ld)
df_ab['is_valid_corrected'] = df_ab['corrected_json_ld'].apply(clean_and_validate_json_ld)# 2. 대조군(is_corrected=0)과 실험군(is_corrected=1) 분리
control = df_ab[df_ab['is_corrected'] == 0]
treatment = df_ab[df_ab['is_corrected'] == 1]# 3. 최종 A/B 테스트 리포트 출력
control_rate = control['is_valid_original'].mean() * 100
treatment_orig_rate = treatment['is_valid_original'].mean() * 100
treatment_corr_rate = treatment['is_valid_corrected'].mean() * 100

print("="*45)
print("          A/B 테스트 최종 분석 리포트          ")
print("="*45)
print(f"1. 대조군 (is_corrected=0) [건수: {len(control):,}건]")
print(f"   - 원본 JSON-LD 유효율 : {control_rate:.2f}%\n")
print(f"2. 실험군 (is_corrected=1) [건수: {len(treatment):,}건]")
print(f"   - 교정 전 (원본) 유효율 : {treatment_orig_rate:.2f}%")
print(f"   - 교정 후 (프롬프트) 유효율 : {treatment_corr_rate:.2f}%")
print("-"*45)
print(f"★ 프롬프트 교정 효과 (유효율 변화) : {treatment_corr_rate - treatment_orig_rate:+.2f}%p")
print("="*45)# 교정본 실패 원인 분석
error_list = []

for text in df_ab[df_ab['is_corrected'] == 1]['corrected_json_ld']:
    if pd.isna(text) or text is None:
        error_list.append("NULL / Empty")
        continue

    cleaned = re.sub(r'^```(?:json)?\s*', '', str(text).strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) == 1:
            data = data[0]

        if isinstance(data, dict):
            if '@context' not in data:
                error_list.append("Schema Error: @context 없음")
            elif '@type' not in data:
                error_list.append("Schema Error: @type 없음")
            else:
                error_list.append("정상 (Valid)")
        else:
            error_list.append("Type Error: Dict 형태 아님")
    except json.JSONDecodeError as e:
        error_list.append(f"JSON 문법 오류 ({e.msg})")
    except Exception as e:
        error_list.append(f"기타 오류: {type(e).__name__}")

print("=== 교정본(corrected_json_ld) 파싱 실패 원인 Top 5 ===")
print(pd.Series(error_list).value_counts().head(5))
# 1. 교정 데이터(corrected_json_ld)가 실제로 존재하는 행만 필터링
df_valid_geo = df_ab[df_ab['corrected_json_ld'].notna() & (df_ab['corrected_json_ld'] != '')].copy()

# 2. JSON 파싱 가능 여부 및 스키마 유효성 검증 함수
# 3. 유효성 계산
df_valid_geo['is_valid_orig'] = df_valid_geo['original_json_ld'].apply(validate_json_only)
df_valid_geo['is_valid_corr'] = df_valid_geo['corrected_json_ld'].apply(validate_json_only)

print("=== 실제 교정 실행 데이터(NULL 제외) 대상 최종 리포트 ===")
print(f"- 분석 대상 건수: {len(df_valid_geo):,}건")
print(f"- 원본(교정 전) JSON 파싱 성공률: {df_valid_geo['is_valid_orig'].mean()*100:.2f}%")
print(f"- 교정본(교정 후) JSON 파싱 성공률: {df_valid_geo['is_valid_corr'].mean()*100:.2f}%")
# 1. 교정 데이터가 존재하는 31,846건 대상
df_analysis = df_valid_geo.copy()# 2. 세부 규격 검증 함수 정의
# 3. 교정 전(원본) vs 교정 후 적용
orig_results = pd.DataFrame(df_analysis['original_json_ld'].apply(check_json_ld_compliance).tolist())
corr_results = pd.DataFrame(df_analysis['corrected_json_ld'].apply(check_json_ld_compliance).tolist())# 4. 규격 이행률 비교 리포트 출력
print("="*60)
print("     [1단계] 프롬프트 교정 전/후 JSON-LD 규격 이행률 세부 비교     ")
print("="*60)
print(f"분석 대상 건수: {len(df_analysis):,}건\n")

metrics = [
    ('1. JSON 문법 통과율 (Syntax Check)', 'syntax_ok'),
    ('2. @context 필드 존재율', 'has_context'),
    ('3. @type 필드 존재율', 'has_type'),
    ('★ 최종 완전 규격 이행률 (Full Compliance)', 'full_compliance')
]

for label, col in metrics:
    orig_rate = orig_results[col].mean() * 100
    corr_rate = corr_results[col].mean() * 100
    diff = corr_rate - orig_rate
    print(f"📌 {label}")
    print(f"   - 교정 전 (원본) : {orig_rate:.2f}%")
    print(f"   - 교정 후 (프롬프트): {corr_rate:.2f}%")
    print(f"   - 이행률 변화량  : {diff:+.2f}%p\n")# LLM의 @type 누락 및 키 변형 패턴 분석
type_patterns = []

for item in df_analysis['corrected_json_ld']:
    if pd.isna(item) or item is None:
        continue

    try:
        text = str(item).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)

        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict):
            keys = [str(k).lower() for k in data.keys()]
            if '@type' in data:
                type_patterns.append("정상 (@type 존재)")
            elif 'type' in keys:
                type_patterns.append("키 변형: 'type' (골뱅이 누락)")
            elif '@type' in [k.lower() for k in data.keys()]:
                type_patterns.append("키 변형: 대소문자 혼동 (예: @Type)")
            else:
                type_patterns.append("완전 누락 (@type 관련 키 전혀 없음)")
        else:
            type_patterns.append("구조 오류 (Dict 형태 아님)")

    except Exception:
        type_patterns.append("JSON 문법 파싱 실패 건")

print("="*60)
print("     [2단계] 교정본의 @type 누락 및 변형 패턴 세부 분석     ")
print("="*60)

pattern_series = pd.Series(type_patterns)
for pattern, count in pattern_series.value_counts().items():
    ratio = (count / len(type_patterns)) * 100
    print(f"• {pattern:<35}: {count:,}건 ({ratio:.2f}%)")# @type 완전 누락 건들의 최상위 Key 구조 분석
missing_type_keys = []

for item in df_analysis['corrected_json_ld']:
    if pd.isna(item) or item is None:
        continue
    try:
        text = str(item).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)

        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict) and '@type' not in data:
            k_tuple = tuple(sorted(list(data.keys())))
            missing_type_keys.append(k_tuple)
    except Exception:
        pass
    
print("="*60)
print("   [3단계] @type 누락 데이터에 존재하는 주요 Key 조합 Top 5   ")
print("="*60)
key_series = pd.Series(missing_type_keys)
for keys, count in key_series.value_counts().head(5).items():
    ratio = (count / len(missing_type_keys)) * 100
    print(f"• 존재 Key: {[keys]}")
    print(f"  └ 건수: {count:,}건 ({ratio:.2f}%)\n")# @graph 내부 노드들의 @type 존재 여부 검증
graph_type_results = []

for item in df_analysis['corrected_json_ld']:
    if pd.isna(item) or item is None:
        continue
    try:
        text = str(item).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)

        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict):
            if '@type' in data:
                graph_type_results.append("최상위 @type 존재 (단일 객체)")
            elif '@graph' in data and isinstance(data['@graph'], list):
                has_inner_type = any(
                    isinstance(node, dict) and '@type' in node
                    for node in data['@graph']
                )
                if has_inner_type:
                    graph_type_results.append("@graph 내부 @type 존재 (그래프 구조)")
                else:
                    graph_type_results.append("@graph 내부에도 @type 없음")
            else:
                graph_type_results.append("완전 누락")
    except Exception:
        graph_type_results.append("JSON 문법 오류")

print("="*60)
print("   [4단계] JSON-LD 구조 재해석 후 실제 @type 규격 이행률   ")
print("="*60)

res_series = pd.Series(graph_type_results)
for res_type, count in res_series.value_counts().items():
    ratio = (count / len(graph_type_results)) * 100
    print(f"• {res_type:<35}: {count:,}건 ({ratio:.2f}%)")
# 31,846건 유효 교정 데이터 대상 진단 수행
orig_detail = pd.DataFrame(
    df_valid_geo["original_json_ld"]
    .apply(analyze_detail_compliance)
    .tolist()
)
corr_detail = pd.DataFrame(
    df_valid_geo["corrected_json_ld"]
    .apply(analyze_detail_compliance)
    .tolist()
)

print("=" * 65)
print("  [세부 속성 기준] JSON-LD 규격 이행률 분석 리포트")
print("=" * 65)
print(f"분석 대상 건수: {len(df_valid_geo):,}건\n")

print("📌 1. Description (설명 분량) 이행률")
print(
    f"   - 평균 글자 수       : {orig_detail['desc_length'].mean():.1f}자 ➔ {corr_detail['desc_length'].mean():.1f}자"
)
print(
    f"   - 규격 이행률(>=20자) : {orig_detail['desc_valid'].mean()*100:.2f}% ➔ {corr_detail['desc_valid'].mean()*100:.2f}%\n"
)

print("📌 2. 의류 도메인 필수 속성 이행률")
print(
    f"   - Brand 보유율      : {orig_detail['has_brand'].mean()*100:.2f}% ➔ {corr_detail['has_brand'].mean()*100:.2f}%"
)
print(
    f"   - Color 보유율      : {orig_detail['has_color'].mean()*100:.2f}% ➔ {corr_detail['has_color'].mean()*100:.2f}%"
)
print(
    f"   - Size/Offers 보유율 : {orig_detail['has_size'].mean()*100:.2f}% ➔ {corr_detail['has_size'].mean()*100:.2f}%"
)
print(
    f"   - Material 보유율    : {orig_detail['has_material'].mean()*100:.2f}% ➔ {corr_detail['has_material'].mean()*100:.2f}%"
)
print(
    f"   - Category 보유율    : {orig_detail['has_category'].mean()*100:.2f}% ➔ {corr_detail['has_category'].mean()*100:.2f}%"
)
print(
    f"   - [종합] 필수속성 완수율: {orig_detail['apparel_full_compliance'].mean()*100:.2f}% ➔ {corr_detail['apparel_full_compliance'].mean()*100:.2f}%\n"
)

print("📌 3. 신뢰도(Confidence) 및 엔티티(Entity) 추출 이행 상태")
print(
    f"   - 신뢰도 필드 존재율 : {orig_detail['has_confidence'].mean()*100:.2f}% ➔ {corr_detail['has_confidence'].mean()*100:.2f}%"
)
print(
    f"   - 평균 엔티티/속성 개수: {orig_detail['entity_count'].mean():.1f}개 ➔ {corr_detail['entity_count'].mean():.1f}개\n"
)# 컬럼 존재 여부 확인 후 처리
if 'has_material' in corr_detail.columns:
    material_rate = corr_detail['has_material'].mean()
else:
    print("Warning: 'has_material' 컬럼이 데이터프레임에 존재하지 않습니다.")
orig_deep = pd.DataFrame(
    df_valid_geo["original_json_ld"].apply(analyze_detail_deep).tolist()
)
corr_deep = pd.DataFrame(
    df_valid_geo["corrected_json_ld"].apply(analyze_detail_deep).tolist()
)

print("=" * 70)
print("             [디테일 심화 분석] JSON 규격 및 세부 속성 정밀 진단            ")
print("=" * 70)

# 1. Description 분포 세분화 (JSON 파싱 실패 vs 순수 누락 구분)
print("📌 1. Description 글자 수 구간별 비중 변화")
orig_desc_dist = orig_deep["desc_category"].value_counts(normalize=True) * 100
corr_desc_dist = corr_deep["desc_category"].value_counts(normalize=True) * 100

desc_df = pd.DataFrame(
    {"교정 전(%)": orig_desc_dist, "교정 후(%)": corr_desc_dist}
).fillna(0)
print(desc_df.sort_index().to_string())

# 2. 의류 속성 누락 조합 분석 (파싱 성공건 기준)
print("\n📌 2. 의류 필수 속성 결함 패턴 조합 Top 3 (교정 후)")
valid_corr = corr_deep[corr_deep["syntax_ok"]]
print(valid_corr["missing_attrs"].value_counts().head(3).to_string())

# 3. 신뢰도 및 엔티티 파편화 세부 진단
print("\n📌 3. 신뢰도 필드 및 엔티티 파편화 정밀 상태 (교정 후)")
print(
    f"  - 신뢰도(confidence) 필드 포함률 : {corr_deep['has_confidence'].mean()*100:.2f}%"
)
print(
    f"  - `@graph` 구조 변환 비율       : {corr_deep['is_graph'].mean()*100:.2f}%"
)

all_nodes = [
    node for sublist in corr_deep["graph_node_types"] for node in sublist
]
if all_nodes:
    print("  - `@graph` 내부 추출 엔티티 타입 Top 3:")
    print(pd.Series(all_nodes).value_counts().head(3).to_string())

# --- 폰트 설정 ---
font_path = '/content/Pretendard－M.ttf'
plt.rcParams['axes.unicode_minus'] = False

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rc('font', family=font_name)
    print(f"'{font_name}' 폰트 설정 완료.")
else:
    print(f"경로에 '{font_path}' 폰트 파일을 찾을 수 없습니다. 기본 폰트(나눔고딕 또는 대체 폰트)를 사용합니다.")
    try:
        fm.findfont('NanumGothic', rebuild_if_missing=True)
        plt.rc('font', family='NanumGothic')
        print("NanumGothic 폰트가 설치되어 있어 이를 사용합니다.")
    except Exception:
        plt.rc('font', family='DejaVu Sans')
        print("기본 폰트를 찾을 수 없어 'DejaVu Sans'를 사용합니다.")

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

print("Matplotlib 한글 폰트 및 디자인 설정 완료 (Pretendard 폰트 파일을 업로드해야 제대로 적용됩니다).")# 1. 안전한 컬럼 데이터 추출 함수 정의 (NameError 방지)

# 1. JSON-LD 상세 속성 파싱 함수 (의류 속성 & 신뢰도 포함)
# 2. 데이터프레임 생성
orig_detail = pd.DataFrame(df_valid_geo['original_json_ld'].apply(analyze_detail_properties).tolist())
corr_detail = pd.DataFrame(df_valid_geo['corrected_json_ld'].apply(analyze_detail_properties).tolist())
# 1. 시각화 데이터 준비 (기존 변수값 할당)
labels = ['교정 전', '교정 후']
colors = ['#71FFE7', '#FF7171']

# ① @type 존재율 (%)
# 데이터프레임 적용
orig_type_detail = pd.DataFrame(df_analysis['original_json_ld'].apply(check_type_detail).tolist())
corr_type_detail = pd.DataFrame(df_analysis['corrected_json_ld'].apply(check_type_detail).tolist())

# 교정 전/후 리스트 생성 (루트, @graph 내, 통합 순)
type_rate_orig = [
    orig_type_detail['root_type'].mean() * 100,
    orig_type_detail['graph_type'].mean() * 100,
    orig_type_detail['any_type'].mean() * 100
]

type_rate_corr = [
    corr_type_detail['root_type'].mean() * 100,
    corr_type_detail['graph_type'].mean() * 100,
    corr_type_detail['any_type'].mean() * 100
]

# ② Description 이행률 (분량 및 50자 이상 여부)
# 데이터프레임 적용
orig_desc_df = pd.DataFrame(df_analysis['original_json_ld'].apply(check_description_detail).tolist())
corr_desc_df = pd.DataFrame(df_analysis['corrected_json_ld'].apply(check_description_detail).tolist())

desc_rate_over50 = [
    orig_desc_df['desc_over_50'].mean() * 100,
    corr_desc_df['desc_over_50'].mean() * 100
]
# ③ 의류 도메인 필수 속성 이행률 (%)
apparel_attrs = ['Material', 'Color', 'Size', 'Category']
apparel_cols = ['has_material', 'has_color', 'has_size', 'has_category']

apparel_orig = [orig_detail[col].mean() * 100 for col in apparel_cols]
apparel_corr = [corr_detail[col].mean() * 100 for col in apparel_cols]

# ④ 신뢰도(Confidence) 존재율 (%)
conf_rate = [
    orig_detail['has_confidence'].mean() * 100,
    corr_detail['has_confidence'].mean() * 100
]

# Subplots로 시각화 (2행 2열)
plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('JSON-LD 규격 이행률 및 구조 종합', fontsize=20, fontweight='bold', y=0.98)

# 텍스트 출력 레이블 함수 정의
# [1,1] @type 필드 존재율
ax = axes[0, 0]
categories = ['루트 @type', '@graph 내 @type', '통합 @type']
x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, type_rate_orig, width, label='교정 전', color='#FF7171')
rects2 = ax.bar(x + width/2, type_rate_corr, width, label='교정 후', color='#71FFE7')

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(title_fontsize=12, fontsize=12, loc='upper right')
ax.bar_label(rects1, fmt='%.1f%%', padding=3)
ax.bar_label(rects2, fmt='%.1f%%', padding=3)

# [1,2] Description 규격 이행률
ax2 = axes[0, 1]
bars2 = ax2.bar(['교정 전', '교정 후'], desc_rate_over50, color=['#FF7171','#71FFE7'], width=0.45)
ax2.set_title('50자 이상 작성된 Description 비율', fontsize=15, pad=12, color='white', fontweight='bold')
ax2.set_ylabel('비율 (%)', fontsize=12, color='white')
ax2.set_ylim(0, max(desc_rate_over50 + [10]) * 1.25)
ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
ax2.bar_label(bars2, fmt='%.1f%%', padding=4, color='white', fontsize=10, fontweight='bold')

# [2,1] 의류 도메인 필수 속성 이행률
ax = axes[1, 0]
x_apparel = np.arange(len(apparel_attrs))
width = 0.35
rects_app1 = ax.bar(x_apparel - width/2, apparel_orig, width, label='교정 전', color='#FF7171')
rects_app2 = ax.bar(x_apparel + width/2, apparel_corr, width, label='교정 후', color='#71FFE7')
ax.set_title('의류 도메인 필수 속성 이행률', fontsize=15, pad=12)
ax.set_ylabel('이행률 (%)', fontsize=12)
ax.set_xticks(x_apparel)
ax.set_xticklabels(apparel_attrs, fontsize=12)
ax.set_ylim(0, 125)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.legend(title_fontsize=12, fontsize=12, loc='upper right')

# 바 위에 수치 표기
ax.bar_label(rects_app1, fmt='%.1f%%', padding=4, fontsize=10)
ax.bar_label(rects_app2, fmt='%.1f%%', padding=4, fontsize=10)

# [2,2] 신뢰도(Confidence) 필드 존재율
ax = axes[1, 1]
bars4 = ax.bar(labels, conf_rate, color=colors)
ax.set_title('신뢰도 요소 존재 비율', fontsize=15, pad=12)
ax.set_xlabel('교정 구분', fontsize=12)
ax.set_ylabel('존재율 (%)', fontsize=12)
ax.set_ylim(0, 115)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
annotate_bars(ax, bars4)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig('JSON_LD_종합_이행률.png', dpi=300)
plt.show()
