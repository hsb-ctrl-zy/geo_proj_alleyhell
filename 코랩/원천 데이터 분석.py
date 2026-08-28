from collections import Counter
# from google.colab import files
from kiwipiepy import Kiwi
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Okt
from konlpy.tag import Okt, Kkma, Komoran
import emoji
import json
import math
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
import os
import pandas as pd
import platform
import pymysql
import re
import seaborn as sns
from IPython.display import display

import sys
# 터미널 창의 입출력 한글 깨짐 방지
sys.stdout.reconfigure(encoding='utf8') #type: ignore
sys.stderr.reconfigure(encoding='utf8') #type: ignore

def analyze_product_name_counts(text):
    if not isinstance(text, str) or not text.strip():
        return {
            '명사': 0, '형용사': 0, '동사': 0, '수사': 0, '조사': 0, '어미': 0,
            '유니코드 이모지': 0, '텍스트 이모티콘': 0, '구분자': 0, '기타 특수문자': 0
        }
    # (1) 형태소 분석 (Okt 이용)
    pos_tags = okt.pos(text)
    noun_cnt = sum(1 for _, tag in pos_tags if tag == 'Noun')
    adj_cnt = sum(1 for _, tag in pos_tags if tag == 'Adjective')
    verb_cnt = sum(1 for _, tag in pos_tags if tag == 'Verb')
    num_cnt = sum(1 for _, tag in pos_tags if tag == 'Number')
    josa_cnt = sum(1 for _, tag in pos_tags if tag == 'Josa')
    eomi_cnt = sum(1 for _, tag in pos_tags if tag == 'Eomi')

    # (2) 이모지 및 특수문자 카운팅
    # 유니코드 이모지 (예: ✨, ❤️ 등)
    emoji_cnt = len([c for c in text if c in emoji.EMOJI_DATA])

    # 텍스트 이모티콘 (예: ^^, ㅠㅠ, :) 등)
    emoticon_pattern = r'(\^\^|::-?\(|::-?\)|ㅠㅠ|ㅜㅜ|;\s*;)'
    emoticon_cnt = len(re.findall(emoticon_pattern, text))

    # 구분자 (예: [, ], (, ), /, -, _, ~ 등)
    sep_pattern = r'[\[\]\(\)\/\\_\-~|,]'
    sep_cnt = len(re.findall(sep_pattern, text))

    # 기타 특수문자 (한글, 영문, 숫자, 공백, 위에서 센 특수문자 제외)
    # 일반적인 특수문자 전체 개수에서 구분자/이모티콘 차감
    all_spec_cnt = len(re.findall(r'[^\s\w가-힣]', text))
    etc_spec_cnt = max(0, all_spec_cnt - emoji_cnt - sep_cnt)

    return {
        '명사': noun_cnt,
        '형용사': adj_cnt,
        '동사': verb_cnt,
        '수사': num_cnt,
        '조사': josa_cnt,
        '어미': eomi_cnt,
        '유니코드 이모지': emoji_cnt,
        '텍스트 이모티콘': emoticon_cnt,
        '구분자': sep_cnt,
        '기타 특수문자': etc_spec_cnt
    }
def apply_kiwi_analysis_to_df(df_target):
    # product_name 컬럼에 Kiwi 형태소 분석 함수 적용
    product_name_pos_results_kiwi = df_target['product_name'].apply(get_pos_ratios_kiwi)
    df_target['product_name_kiwi_noun_ratio'] = product_name_pos_results_kiwi.apply(lambda x: x['noun_ratio'])
    df_target['product_name_kiwi_adjective_ratio'] = product_name_pos_results_kiwi.apply(lambda x: x['adjective_ratio'])
    df_target['product_name_kiwi_numeral_ratio'] = product_name_pos_results_kiwi.apply(lambda x: x['numeral_ratio'])
    df_target['product_name_kiwi_verb_ratio'] = product_name_pos_results_kiwi.apply(lambda x: x['verb_ratio'])
    df_target['product_name_kiwi_josa_ratio'] = product_name_pos_results_kiwi.apply(lambda x: x['josa_ratio'])
    df_target['product_name_kiwi_eomi_ratio'] = product_name_pos_results_kiwi.apply(lambda x: x['eomi_ratio'])

    # product_name 컬럼에 특수문자 비율 함수 적용
    product_name_special_char_results_kiwi = df_target['product_name'].apply(get_special_char_ratio_kiwi)
    df_target['product_name_kiwi_special_char_ratio'] = product_name_special_char_results_kiwi.apply(lambda x: x['special_char_ratio'])

    # text_contents 컬럼에 Kiwi 형태소 분석 함수 적용
    text_contents_pos_results_kiwi = df_target['text_contents'].apply(get_pos_ratios_kiwi)
    df_target['text_contents_kiwi_noun_ratio'] = text_contents_pos_results_kiwi.apply(lambda x: x['noun_ratio'])
    df_target['text_contents_kiwi_adjective_ratio'] = text_contents_pos_results_kiwi.apply(lambda x: x['adjective_ratio'])
    df_target['text_contents_kiwi_numeral_ratio'] = text_contents_pos_results_kiwi.apply(lambda x: x['numeral_ratio'])
    df_target['text_contents_kiwi_verb_ratio'] = text_contents_pos_results_kiwi.apply(lambda x: x['verb_ratio'])
    df_target['text_contents_kiwi_josa_ratio'] = text_contents_pos_results_kiwi.apply(lambda x: x['josa_ratio'])
    df_target['text_contents_kiwi_eomi_ratio'] = text_contents_pos_results_kiwi.apply(lambda x: x['eomi_ratio'])

    # text_contents 컬럼에 특수문자 비율 함수 적용
    text_contents_special_char_results_kiwi = df_target['text_contents'].apply(get_special_char_ratio_kiwi)
    df_target['text_contents_kiwi_special_char_ratio'] = text_contents_special_char_results_kiwi.apply(lambda x: x['special_char_ratio'])
    return df_target

def apply_kkma_analysis_to_df(df_target):
    # product_name 컬럼에 Kkma 형태소 분석 함수 적용
    product_name_pos_results_kkma = df_target['product_name'].apply(get_pos_ratios_kkma)
    df_target['product_name_kkma_noun_ratio'] = product_name_pos_results_kkma.apply(lambda x: x['noun_ratio'])
    df_target['product_name_kkma_adjective_ratio'] = product_name_pos_results_kkma.apply(lambda x: x['adjective_ratio'])
    df_target['product_name_kkma_numeral_ratio'] = product_name_pos_results_kkma.apply(lambda x: x['numeral_ratio'])
    df_target['product_name_kkma_verb_ratio'] = product_name_pos_results_kkma.apply(lambda x: x['verb_ratio'])
    df_target['product_name_kkma_josa_ratio'] = product_name_pos_results_kkma.apply(lambda x: x['josa_ratio'])
    df_target['product_name_kkma_eomi_ratio'] = product_name_pos_results_kkma.apply(lambda x: x['eomi_ratio'])

    # product_name 컬럼에 특수문자 비율 함수 적용
    product_name_special_char_results_kkma = df_target['product_name'].apply(get_special_char_ratio_kkma)
    df_target['product_name_kkma_special_char_ratio'] = product_name_special_char_results_kkma.apply(lambda x: x['special_char_ratio'])

    # text_contents 컬럼에 Kkma 형태소 분석 함수 적용
    text_contents_pos_results_kkma = df_target['text_contents'].apply(get_pos_ratios_kkma)
    df_target['text_contents_kkma_noun_ratio'] = text_contents_pos_results_kkma.apply(lambda x: x['noun_ratio'])
    df_target['text_contents_kkma_adjective_ratio'] = text_contents_pos_results_kkma.apply(lambda x: x['adjective_ratio'])
    df_target['text_contents_kkma_numeral_ratio'] = text_contents_pos_results_kkma.apply(lambda x: x['numeral_ratio'])
    df_target['text_contents_kkma_verb_ratio'] = text_contents_pos_results_kkma.apply(lambda x: x['verb_ratio'])
    df_target['text_contents_kkma_josa_ratio'] = text_contents_pos_results_kkma.apply(lambda x: x['josa_ratio'])
    df_target['text_contents_kkma_eomi_ratio'] = text_contents_pos_results_kkma.apply(lambda x: x['eomi_ratio'])

    # text_contents 컬럼에 특수문자 비율 함수 적용
    text_contents_special_char_results_kkma = df_target['text_contents'].apply(get_special_char_ratio_kkma)
    df_target['text_contents_kkma_special_char_ratio'] = text_contents_special_char_results_kkma.apply(lambda x: x['special_char_ratio'])
    return df_target

def apply_komoran_analysis_to_df(df_target):
    # product_name 컬럼에 Komoran 형태소 분석 함수 적용
    product_name_pos_results_komoran = df_target['product_name'].apply(get_pos_ratios_komoran)
    df_target['product_name_komoran_noun_ratio'] = product_name_pos_results_komoran.apply(lambda x: x['noun_ratio'])
    df_target['product_name_komoran_adjective_ratio'] = product_name_pos_results_komoran.apply(lambda x: x['adjective_ratio'])
    df_target['product_name_komoran_numeral_ratio'] = product_name_pos_results_komoran.apply(lambda x: x['numeral_ratio'])
    df_target['product_name_komoran_verb_ratio'] = product_name_pos_results_komoran.apply(lambda x: x['verb_ratio'])
    df_target['product_name_komoran_josa_ratio'] = product_name_pos_results_komoran.apply(lambda x: x['josa_ratio'])
    df_target['product_name_komoran_eomi_ratio'] = product_name_pos_results_komoran.apply(lambda x: x['eomi_ratio'])

    # product_name 컬럼에 특수문자 비율 함수 적용
    product_name_special_char_results_komoran = df_target['product_name'].apply(get_special_char_ratio_komoran)
    df_target['product_name_komoran_special_char_ratio'] = product_name_special_char_results_komoran.apply(lambda x: x['special_char_ratio'])

    # text_contents 컬럼에 Komoran 형태소 분석 함수 적용
    text_contents_pos_results_komoran = df_target['text_contents'].apply(get_pos_ratios_komoran)
    df_target['text_contents_komoran_noun_ratio'] = text_contents_pos_results_komoran.apply(lambda x: x['noun_ratio'])
    df_target['text_contents_komoran_adjective_ratio'] = text_contents_pos_results_komoran.apply(lambda x: x['adjective_ratio'])
    df_target['text_contents_komoran_numeral_ratio'] = text_contents_pos_results_komoran.apply(lambda x: x['numeral_ratio'])
    df_target['text_contents_komoran_verb_ratio'] = text_contents_pos_results_komoran.apply(lambda x: x['verb_ratio'])
    df_target['text_contents_komoran_josa_ratio'] = text_contents_pos_results_komoran.apply(lambda x: x['josa_ratio'])
    df_target['text_contents_komoran_eomi_ratio'] = text_contents_pos_results_komoran.apply(lambda x: x['eomi_ratio'])

    # text_contents 컬럼에 특수문자 비율 함수 적용
    text_contents_special_char_results_komoran = df_target['text_contents'].apply(get_special_char_ratio_komoran)
    df_target['text_contents_komoran_special_char_ratio'] = text_contents_special_char_results_komoran.apply(lambda x: x['special_char_ratio'])
    return df_target

def apply_okt_analysis_to_df(df_target):
    # product_name 컬럼에 형태소 분석 함수 적용
    product_name_pos_results = df_target['product_name'].apply(get_pos_ratios)
    df_target['product_name_noun_ratio'] = product_name_pos_results.apply(lambda x: x['noun_ratio'])
    df_target['product_name_adjective_ratio'] = product_name_pos_results.apply(lambda x: x['adjective_ratio'])
    df_target['product_name_numeral_ratio'] = product_name_pos_results.apply(lambda x: x['numeral_ratio'])
    df_target['product_name_verb_ratio'] = product_name_pos_results.apply(lambda x: x['verb_ratio'])
    df_target['product_name_josa_ratio'] = product_name_pos_results.apply(lambda x: x['josa_ratio'])
    df_target['product_name_eomi_ratio'] = product_name_pos_results.apply(lambda x: x['eomi_ratio'])

    # product_name 컬럼에 특수문자 비율 함수 적용
    product_name_special_char_results = df_target['product_name'].apply(get_special_char_ratio)
    df_target['product_name_special_char_ratio'] = product_name_special_char_results.apply(lambda x: x['special_char_ratio'])

    # text_contents 컬럼에 형태소 분석 함수 적용
    text_contents_pos_results = df_target['text_contents'].apply(get_pos_ratios)
    df_target['text_contents_noun_ratio'] = text_contents_pos_results.apply(lambda x: x['noun_ratio'])
    df_target['text_contents_adjective_ratio'] = text_contents_pos_results.apply(lambda x: x['adjective_ratio'])
    df_target['text_contents_numeral_ratio'] = text_contents_pos_results.apply(lambda x: x['numeral_ratio'])
    df_target['text_contents_verb_ratio'] = text_contents_pos_results.apply(lambda x: x['verb_ratio'])
    df_target['text_contents_josa_ratio'] = text_contents_pos_results.apply(lambda x: x['josa_ratio'])
    df_target['text_contents_eomi_ratio'] = text_contents_pos_results.apply(lambda x: x['eomi_ratio'])

    # text_contents 컬럼에 특수문자 비율 함수 적용
    text_contents_special_char_results = df_target['text_contents'].apply(get_special_char_ratio)
    df_target['text_contents_special_char_ratio'] = text_contents_special_char_results.apply(lambda x: x['special_char_ratio'])
    return df_target

def calculate_average_ratios_by_analyzer(df_target, prefix_base, analyzer_suffix=""):
    columns_to_avg = [
        f'{prefix_base}{analyzer_suffix}_noun_ratio',
        f'{prefix_base}{analyzer_suffix}_adjective_ratio',
        f'{prefix_base}{analyzer_suffix}_numeral_ratio',
        f'{prefix_base}{analyzer_suffix}_verb_ratio',
        f'{prefix_base}{analyzer_suffix}_josa_ratio',
        f'{prefix_base}{analyzer_suffix}_eomi_ratio',
        f'{prefix_base}{analyzer_suffix}_special_char_ratio'
    ]
    avg_ratios = df_target[columns_to_avg].mean().to_dict()
    return {k.replace(f'{prefix_base}{analyzer_suffix}_', ''): v for k, v in avg_ratios.items()}

def count_detailed_special_chars(text):
    if not isinstance(text, str) or not text.strip():
        return {
            'unicode_emoji_count': 0,
            'textual_emoji_count': 0,
            'separator_count': 0,
            'other_symbol_count': 0
        }

    temp_text = text

    # 1. 유니코드 이모지 수
    unicode_emojis = emoji.emoji_list(temp_text)
    unicode_emoji_count = len(unicode_emojis)
    for emo_dict in sorted(unicode_emojis, key=lambda x: x['match_start'], reverse=True):
        temp_text = temp_text[:emo_dict['match_start']] + temp_text[emo_dict['match_end']:]

    # 2. 텍스트 이모지 수
    textual_emoji_matches = re.findall(r':[가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿가-힯ힰ-퟿!-~]+:', temp_text)
    textual_emoji_count = len(textual_emoji_matches)
    for emo in textual_emoji_matches:
        temp_text = temp_text.replace(emo, '', 1)

    # 3. 구분자 (/, |, [], , .)
    separators_matches = re.findall(r'[/\[\](),.|]', temp_text) 
    separator_count = len(separators_matches)
    temp_text = re.sub(r'[/\[\](),.|]', '', temp_text)

    # 4. 다른 기호
    other_symbols_matches = re.findall(r'[^가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿가-힯ힰ-퟿ -~]', temp_text)
    other_symbol_count = len(other_symbols_matches)

    return {
        'unicode_emoji_count': unicode_emoji_count,
        'textual_emoji_count': textual_emoji_count,
        'separator_count': separator_count,
        'other_symbol_count': other_symbol_count
    }

def count_nouns(text, tagger):
    if not isinstance(text, str):
        return []
    try:
        if tagger == okt:
            return [word for word, tag in okt.pos(text) if tag in ['Noun', 'Foreign']]
        elif tagger == kkma:
            return [word for word, tag in kkma.pos(text) if 'N' in tag or 'F' in tag]
        elif tagger == komoran:
            return [word for word, tag in komoran.pos(text) if tag in ['NNG', 'NNP', 'NP', 'NR', 'SL']]
        elif tagger == kiwi:
            result_tokens = kiwi.tokenize(text)
            return [getattr(t, 'form') for t in result_tokens if getattr(t, 'tag').startswith('N')]
        else:
            return []
    except Exception as e:
        print(f"형태소 분석 중 오류 발생 ({tagger.__class__.__name__}): {e} - 텍스트: {text[:50]}...")
        return []

def count_pos_tags_okt(text):
    if not isinstance(text, str) or not text.strip():
        return {'noun_count': 0, 'adjective_count': 0, 'numeral_count': 0, 'verb_count': 0, 'josa_count': 0, 'eomi_count': 0}

    pos_tags = okt.pos(text)

    noun_count = sum(1 for word, tag in pos_tags if tag == 'Noun')
    adjective_count = sum(1 for word, tag in pos_tags if tag == 'Adjective')
    numeral_count = sum(1 for word, tag in pos_tags if tag == 'Number')
    verb_count = sum(1 for word, tag in pos_tags if tag == 'Verb')
    josa_count = sum(1 for word, tag in pos_tags if tag == 'Josa')
    eomi_count = sum(1 for word, tag in pos_tags if tag == 'Eomi')

    return {
        'noun_count': noun_count,
        'adjective_count': adjective_count,
        'numeral_count': numeral_count,
        'verb_count': verb_count,
        'josa_count': josa_count,
        'eomi_count': eomi_count
    }

def extract_nouns(text):
    if not isinstance(text, str) or not text.strip():
        return []
    nouns = [word for word, tag in okt.pos(text) if tag == 'Noun']
    return nouns

def extract_valid_alt_count(raw_details_json):
  try:
    data = json.loads(raw_details_json)
    # raw_details 내부의 detail_results 리스트 추출
    detail_results = (
        data.get("details", {}).get("image_alt", {}).get("detail_results", [])
    )

    # 의미 없는 Alt 텍스트 정의
    meaningless_terms = {"이미지", "상품 사진", "", " "}

    # clean_alt가 존재하고, 공백 제거 후 의미 없는 문자열이 아닌 항목만 필터링
    valid_alts = [
        item.get("clean_alt")
        for item in detail_results
        if item.get("clean_alt")
        and item.get("clean_alt").strip() not in meaningless_terms
    ]
    return len(valid_alts)
  except (json.JSONDecodeError, AttributeError):
    return 0


def get_mean(cnt_col, ratio_col):
    if cnt_col in brand_df.columns:
        return brand_df[cnt_col].mean()
    elif ratio_col in brand_df.columns:
        return brand_df[ratio_col].mean()
    return 0

def get_pos_ratios(text):
    if not isinstance(text, str) or not text.strip():
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    pos_tags = okt.pos(text)
    total_pos_count = len(pos_tags)

    if total_pos_count == 0:
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    noun_count = sum(1 for word, tag in pos_tags if tag == 'Noun')
    adjective_count = sum(1 for word, tag in pos_tags if tag == 'Adjective')
    numeral_count = sum(1 for word, tag in pos_tags if tag == 'Number')
    verb_count = sum(1 for word, tag in pos_tags if tag == 'Verb')
    josa_count = sum(1 for word, tag in pos_tags if tag == 'Josa')
    eomi_count = sum(1 for word, tag in pos_tags if tag == 'Eomi')

    return {
        'noun_ratio': noun_count / total_pos_count,
        'adjective_ratio': adjective_count / total_pos_count,
        'numeral_ratio': numeral_count / total_pos_count,
        'verb_ratio': verb_count / total_pos_count,
        'josa_ratio': josa_count / total_pos_count,
        'eomi_ratio': eomi_count / total_pos_count,
        'total_pos_count': total_pos_count
    }

def get_pos_ratios_kiwi(text):
    if not isinstance(text, str) or not text.strip():
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    result = kiwi.analyze(text)
    if not result or not result[0]:
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    pos_tags = result[0][0]
    total_pos_count = len(pos_tags)

    if total_pos_count == 0:
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    noun_count = sum(1 for token in pos_tags if token.tag.startswith('N'))
    adjective_count = sum(1 for token in pos_tags if token.tag.startswith('VA'))
    numeral_count = sum(1 for token in pos_tags if token.tag.startswith('NR'))
    verb_count = sum(1 for token in pos_tags if token.tag.startswith('V'))
    josa_count = sum(1 for token in pos_tags if token.tag.startswith('J'))
    eomi_count = sum(1 for token in pos_tags if token.tag.startswith('E'))

    return {
        'noun_ratio': noun_count / total_pos_count,
        'adjective_ratio': adjective_count / total_pos_count,
        'numeral_ratio': numeral_count / total_pos_count,
        'verb_ratio': verb_count / total_pos_count,
        'josa_ratio': josa_count / total_pos_count,
        'eomi_ratio': eomi_count / total_pos_count,
        'total_pos_count': total_pos_count
    }

def get_pos_ratios_kkma(text):
    if not isinstance(text, str) or not text.strip():
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    pos_tags = kkma.pos(text)
    total_pos_count = len(pos_tags)

    if total_pos_count == 0:
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('N'))
    adjective_count = sum(1 for word, tag in pos_tags if tag == 'VA')
    numeral_count = sum(1 for word, tag in pos_tags if tag == 'NR')
    verb_count = sum(1 for word, tag in pos_tags if tag == 'VV')
    josa_count = sum(1 for word, tag in pos_tags if tag.startswith('J'))
    eomi_count = sum(1 for word, tag in pos_tags if tag.startswith('E'))

    return {
        'noun_ratio': noun_count / total_pos_count,
        'adjective_ratio': adjective_count / total_pos_count,
        'numeral_ratio': numeral_count / total_pos_count,
        'verb_ratio': verb_count / total_pos_count,
        'josa_ratio': josa_count / total_pos_count,
        'eomi_ratio': eomi_count / total_pos_count,
        'total_pos_count': total_pos_count
    }

def get_pos_ratios_komoran(text):
    if not isinstance(text, str) or not text.strip():
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    pos_tags = komoran.pos(text)
    total_pos_count = len(pos_tags)

    if total_pos_count == 0:
        return {'noun_ratio': 0, 'adjective_ratio': 0, 'numeral_ratio': 0, 'verb_ratio': 0, 'josa_ratio': 0, 'eomi_ratio': 0, 'total_pos_count': 0}

    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('N'))
    adjective_count = sum(1 for word, tag in pos_tags if tag == 'VA')
    numeral_count = sum(1 for word, tag in pos_tags if tag == 'NR')
    verb_count = sum(1 for word, tag in pos_tags if tag == 'VV')
    josa_count = sum(1 for word, tag in pos_tags if tag.startswith('J'))
    eomi_count = sum(1 for word, tag in pos_tags if tag.startswith('E'))

    return {
        'noun_ratio': noun_count / total_pos_count,
        'adjective_ratio': adjective_count / total_pos_count,
        'numeral_ratio': numeral_count / total_pos_count,
        'verb_ratio': verb_count / total_pos_count,
        'josa_ratio': josa_count / total_pos_count,
        'eomi_ratio': eomi_count / total_pos_count,
        'total_pos_count': total_pos_count
    }

def get_special_char_ratio(text):
    if not isinstance(text, str) or not text.strip():
        return {'special_char_ratio': 0, 'total_char_count': 0}

    total_char_count = len(text)
    if total_char_count == 0:
        return {'special_char_ratio': 0, 'total_char_count': 0}

    # 한글, 영어, 숫자를 제외한 모든 문자를 특수문자로 간주
    special_char_count = len(re.findall(r'[^\s\w가-힣]', text))

    return {
        'special_char_ratio': special_char_count / total_char_count,
        'total_char_count': total_char_count
    }

def get_special_char_ratio_kiwi(text):
    return get_special_char_ratio(text)

def get_special_char_ratio_kkma(text):
    return get_special_char_ratio(text)

def get_special_char_ratio_komoran(text):
    return get_special_char_ratio(text)

#데이터베이스 연결
conn=pymysql.connect(
    host=host_name,
    user=user_name,
    password=password_val,
    database=db_name,
    port=port_num,
    charset='utf8'
)
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
# 텍스트 이모티콘 처리를 위한 emoji 라이브러리 설치

try:
    with conn.cursor() as cursor:
        sql = "SELECT product_name, text_contents, brand_type, page_id, brand_name FROM raw_data_table"
        cursor.execute(sql)
        data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['product_name', 'text_contents', 'brand_type', 'page_id', 'brand_name'])
    print("데이터를 성공적으로 로드했습니다. 처음 5개 행을 표시합니다:")
    display(df.head())

    print("\nbrand_type 컬럼의 고유 값과 그 개수:")
    display(df['brand_type'].value_counts())

except pymysql.Error as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    df = pd.DataFrame(columns=['product_name', 'text_contents', 'brand_type', 'page_id', 'brand_name'])
# 'brand_name' 컬럼을 기준으로 데이터 분리 (사전 형태로 저장)
brand_dfs = {}
for brand_name in df['brand_name'].unique():
    brand_dfs[brand_name] = df[df['brand_name'] == brand_name].copy()

print(f"총 {len(brand_dfs)}개의 브랜드 데이터프레임이 생성되었습니다.")

# 예시로 첫 3개 브랜드의 데이터 개수를 출력합니다.
print("\n각 브랜드별 데이터 개수 (상위 3개):")
for i, (brand, brand_df) in enumerate(brand_dfs.items()):
    if i >= 3: # 상위 3개만 출력
        break
    print(f"  - '{brand}' 데이터 개수: {len(brand_df)}개")
okt = Okt()

# --- 특수문자 비율 function ---
# --- Okt ---
print("Okt 형태소 분석기 및 비율 계산 함수 정의 완료.")
kkma = Kkma()

# --- 특수문자 비율 function ---
# --- Kkma ---
print("Kkma 형태소 분석기 및 비율 계산 함수 정의 완료.")
komoran=Komoran()

# --- 특수문자 function ---
# --- Komoran ---
print("Komoran 형태소 분석기 및 비율 계산 함수 정의 완료.")
kiwi = Kiwi()

# --- 특수문자 function ---
# --- Kiwi ---
print("Kiwi 형태소 분석기 및 비율 계산 함수 정의 완료.")
# df_large_enterprise에 Okt 형태소 분석 적용
# brand_dfs를 사용하여 df_large_enterprise와 df_small_business를 재구성
df_large_enterprise_list = []
df_small_business_list = []

for brand_name, brand_df in brand_dfs.items():
    if not brand_df.empty:
        if brand_df['brand_type'].iloc[0] == '대기업':
            df_large_enterprise_list.append(brand_df)
        elif brand_df['brand_type'].iloc[0] == '소상공인':
            df_small_business_list.append(brand_df)

df_large_enterprise = pd.concat(df_large_enterprise_list, ignore_index=True) if df_large_enterprise_list else pd.DataFrame()
df_small_business = pd.concat(df_small_business_list, ignore_index=True) if df_small_business_list else pd.DataFrame()

df_large_enterprise = apply_okt_analysis_to_df(df_large_enterprise.copy())
df_small_business = apply_okt_analysis_to_df(df_small_business.copy())

print("Okt 분석을 이용한 '대기업' 및 '소상공인' 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
# df_large_enterprise에 Kkma 형태소 분석 적용
df_large_enterprise = apply_kkma_analysis_to_df(df_large_enterprise.copy())
df_small_business = apply_kkma_analysis_to_df(df_small_business.copy())

print("Kkma 분석을 이용한 '대기업' 및 '소상공인' 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
# df_large_enterprise에 Komoran 형태소 분석 적용
df_large_enterprise = apply_komoran_analysis_to_df(df_large_enterprise.copy())
df_small_business = apply_komoran_analysis_to_df(df_small_business.copy())

print("Komoran 분석을 이용한 '대기업' 및 '소상공인' 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
# df_large_enterprise에 Kiwi 형태소 분석 적용
df_large_enterprise = apply_kiwi_analysis_to_df(df_large_enterprise.copy())
df_small_business = apply_kiwi_analysis_to_df(df_small_business.copy())

print("Kiwi 분석을 이용한 '대기업' 및 '소상공인' 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
# '대기업' 데이터의 모든 분석 결과 컬럼 통합

# 'product_name' 관련 컬럼
product_name_cols_large = [
    'product_name_noun_ratio', 'product_name_adjective_ratio', 'product_name_numeral_ratio', 'product_name_verb_ratio', 'product_name_josa_ratio', 'product_name_eomi_ratio', 'product_name_special_char_ratio',
    'product_name_kkma_noun_ratio', 'product_name_kkma_adjective_ratio', 'product_name_kkma_numeral_ratio', 'product_name_kkma_verb_ratio', 'product_name_kkma_josa_ratio', 'product_name_kkma_eomi_ratio', 'product_name_kkma_special_char_ratio',
    'product_name_komoran_noun_ratio', 'product_name_komoran_adjective_ratio', 'product_name_komoran_numeral_ratio', 'product_name_komoran_verb_ratio', 'product_name_komoran_josa_ratio', 'product_name_komoran_eomi_ratio', 'product_name_komoran_special_char_ratio',
    'product_name_kiwi_noun_ratio', 'product_name_kiwi_adjective_ratio', 'product_name_kiwi_numeral_ratio', 'product_name_kiwi_verb_ratio', 'product_name_kiwi_josa_ratio', 'product_name_kiwi_eomi_ratio', 'product_name_kiwi_special_char_ratio'
]

# 'text_contents' 관련 컬럼
text_contents_cols_large = [
    'text_contents_noun_ratio', 'text_contents_adjective_ratio', 'text_contents_numeral_ratio', 'text_contents_verb_ratio', 'text_contents_josa_ratio', 'text_contents_eomi_ratio', 'text_contents_special_char_ratio',
    'text_contents_kkma_noun_ratio', 'text_contents_kkma_adjective_ratio', 'text_contents_kkma_numeral_ratio', 'text_contents_kkma_verb_ratio', 'text_contents_kkma_josa_ratio', 'text_contents_kkma_eomi_ratio', 'text_contents_kkma_special_char_ratio',
    'text_contents_komoran_noun_ratio', 'text_contents_komoran_adjective_ratio', 'text_contents_komoran_numeral_ratio', 'text_contents_komoran_verb_ratio', 'text_contents_komoran_josa_ratio', 'text_contents_komoran_eomi_ratio', 'text_contents_komoran_special_char_ratio',
    'text_contents_kiwi_noun_ratio', 'text_contents_kiwi_adjective_ratio', 'text_contents_kiwi_numeral_ratio', 'text_contents_kiwi_verb_ratio', 'text_contents_kiwi_josa_ratio', 'text_contents_kiwi_eomi_ratio', 'text_contents_kiwi_special_char_ratio'
]

# 새로운 DataFrame 생성
df_large_enterprise_display = df_large_enterprise[['product_name', 'text_contents'] + product_name_cols_large + text_contents_cols_large].copy()

# 한글 컬럼명 매핑
column_name_mapping = {
    'product_name': '상품명',
    'text_contents': '본문',

    'product_name_noun_ratio': '상품명_Okt_명사_비율',
    'product_name_adjective_ratio': '상품명_Okt_형용사_비율',
    'product_name_numeral_ratio': '상품명_Okt_수사_비율',
    'product_name_verb_ratio': '상품명_Okt_동사_비율',
    'product_name_josa_ratio': '상품명_Okt_조사_비율',
    'product_name_eomi_ratio': '상품명_Okt_어미_비율',
    'product_name_special_char_ratio': '상품명_Okt_특수문자_비율',

    'product_name_kkma_noun_ratio': '상품명_Kkma_명사_비율',
    'product_name_kkma_adjective_ratio': '상품명_Kkma_형용사_비율',
    'product_name_kkma_numeral_ratio': '상품명_Kkma_수사_비율',
    'product_name_kkma_verb_ratio': '상품명_Kkma_동사_비율',
    'product_name_kkma_josa_ratio': '상품명_Kkma_조사_비율',
    'product_name_kkma_eomi_ratio': '상품명_Kkma_어미_비율',
    'product_name_kkma_special_char_ratio': '상품명_Kkma_특수문자_비율',

    'product_name_komoran_noun_ratio': '상품명_Komoran_명사_비율',
    'product_name_komoran_adjective_ratio': '상품명_Komoran_형용사_비율',
    'product_name_komoran_numeral_ratio': '상품명_Komoran_수사_비율',
    'product_name_komoran_verb_ratio': '상품명_Komoran_동사_비율',
    'product_name_komoran_josa_ratio': '상품명_Komoran_조사_비율',
    'product_name_komoran_eomi_ratio': '상품명_Komoran_어미_비율',
    'product_name_komoran_special_char_ratio': '상품명_Komoran_특수문자_비율',

    'product_name_kiwi_noun_ratio': '상품명_Kiwi_명사_비율',
    'product_name_kiwi_adjective_ratio': '상품명_Kiwi_형용사_비율',
    'product_name_kiwi_numeral_ratio': '상품명_Kiwi_수사_비율',
    'product_name_kiwi_verb_ratio': '상품명_Kiwi_동사_비율',
    'product_name_kiwi_josa_ratio': '상품명_Kiwi_조사_비율',
    'product_name_kiwi_eomi_ratio': '상품명_Kiwi_어미_비율',
    'product_name_kiwi_special_char_ratio': '상품명_Kiwi_특수문자_비율',

    'text_contents_noun_ratio': '본문_Okt_명사_비율',
    'text_contents_adjective_ratio': '본문_Okt_형용사_비율',
    'text_contents_numeral_ratio': '본문_Okt_수사_비율',
    'text_contents_verb_ratio': '본문_Okt_동사_비율',
    'text_contents_josa_ratio': '본문_Okt_조사_비율',
    'text_contents_eomi_ratio': '본문_Okt_어미_비율',
    'text_contents_special_char_ratio': '본문_Okt_특수문자_비율',

    'text_contents_kkma_noun_ratio': '본문_Kkma_명사_비율',
    'text_contents_kkma_adjective_ratio': '본문_Kkma_형용사_비율',
    'text_contents_kkma_numeral_ratio': '본문_Kkma_수사_비율',
    'text_contents_kkma_verb_ratio': '본문_Kkma_동사_비율',
    'text_contents_kkma_josa_ratio': '본문_Kkma_조사_비율',
    'text_contents_kkma_eomi_ratio': '본문_Kkma_어미_비율',
    'text_contents_kkma_special_char_ratio': '본문_Kkma_특수문자_비율',

    'text_contents_komoran_noun_ratio': '본문_Komoran_명사_비율',
    'text_contents_komoran_adjective_ratio': '본문_Komoran_형용사_비율',
    'text_contents_komoran_numeral_ratio': '본문_Komoran_수사_비율',
    'text_contents_komoran_verb_ratio': '본문_Komoran_동사_비율',
    'text_contents_komoran_josa_ratio': '본문_Komoran_조사_비율',
    'text_contents_komoran_eomi_ratio': '본문_Komoran_어미_비율',
    'text_contents_komoran_special_char_ratio': '본문_Komoran_특수문자_비율',

    'text_contents_kiwi_noun_ratio': '본문_Kiwi_명사_비율',
    'text_contents_kiwi_adjective_ratio': '본문_Kiwi_형용사_비율',
    'text_contents_kiwi_numeral_ratio': '본문_Kiwi_수사_비율',
    'text_contents_kiwi_verb_ratio': '본문_Kiwi_동사_비율',
    'text_contents_kiwi_josa_ratio': '본문_Kiwi_조사_비율',
    'text_contents_kiwi_eomi_ratio': '본문_Kiwi_어미_비율',
    'text_contents_kiwi_special_char_ratio': '본문_Kiwi_특수문자_비율'
}

df_large_enterprise_display = df_large_enterprise_display.rename(columns=column_name_mapping)

print("'대기업' 데이터 분석 결과 (상위 3개 행):")
display(df_large_enterprise_display.head(3))
# '소상공인' 데이터의 모든 분석 결과 컬럼 통합

# DataFrame 생성
df_small_business_display = df_small_business[['product_name', 'text_contents'] + product_name_cols_large + text_contents_cols_large].copy()

# 한글 컬럼명 매핑
df_small_business_display = df_small_business_display.rename(columns=column_name_mapping)

print("'소상공인' 데이터 분석 결과 (상위 3개 행):")
display(df_small_business_display.head(3))
# 각 형태소 분석기별 평균 비율 계산 함수
# '대기업' 데이터 평균 비율 계산
avg_large_okt_product_name = calculate_average_ratios_by_analyzer(df_large_enterprise, 'product_name')
avg_large_okt_text_contents = calculate_average_ratios_by_analyzer(df_large_enterprise, 'text_contents')
avg_large_kkma_product_name = calculate_average_ratios_by_analyzer(df_large_enterprise, 'product_name', '_kkma')
avg_large_kkma_text_contents = calculate_average_ratios_by_analyzer(df_large_enterprise, 'text_contents', '_kkma')
avg_large_komoran_product_name = calculate_average_ratios_by_analyzer(df_large_enterprise, 'product_name', '_komoran')
avg_large_komoran_text_contents = calculate_average_ratios_by_analyzer(df_large_enterprise, 'text_contents', '_komoran')
avg_large_kiwi_product_name = calculate_average_ratios_by_analyzer(df_large_enterprise, 'product_name', '_kiwi')
avg_large_kiwi_text_contents = calculate_average_ratios_by_analyzer(df_large_enterprise, 'text_contents', '_kiwi')

# '소상공인' 데이터 평균 비율 계산
avg_small_okt_product_name = calculate_average_ratios_by_analyzer(df_small_business, 'product_name')
avg_small_okt_text_contents = calculate_average_ratios_by_analyzer(df_small_business, 'text_contents')
avg_small_kkma_product_name = calculate_average_ratios_by_analyzer(df_small_business, 'product_name', '_kkma')
avg_small_kkma_text_contents = calculate_average_ratios_by_analyzer(df_small_business, 'text_contents', '_kkma')
avg_small_komoran_product_name = calculate_average_ratios_by_analyzer(df_small_business, 'product_name', '_komoran')
avg_small_komoran_text_contents = calculate_average_ratios_by_analyzer(df_small_business, 'text_contents', '_komoran')
avg_small_kiwi_product_name = calculate_average_ratios_by_analyzer(df_small_business, 'product_name', '_kiwi')
avg_small_kiwi_text_contents = calculate_average_ratios_by_analyzer(df_small_business, 'text_contents', '_kiwi')

print("'대기업' 및 '소상공인' 데이터의 형태소 분석기별 평균 비율 계산 완료.")

# --- 사용자 제공 폰트 설정 ---
font_path = '/content/Pretendard－M.ttf'

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rc('font', family=font_name)
    print(f"'{font_name}' 폰트 설정 완료.")
else:
    print(f"경로에 '{font_path}' 폰트 파일을 찾을 수 없습니다. 기본 폰트(Malgun Gothic 또는 대체 폰트)를 사용합니다.")
    try:
        fm.findfont('Malgun Gothic', rebuild_if_missing=True)
        plt.rc('font', family='Malgun Gothic')
        print("Malgun Gothic 폰트가 설치되어 있어 이를 사용합니다.")
    except Exception:
        plt.rc('font', family='DejaVu Sans')
        print("기본 폰트를 찾을 수 없어 'DejaVu Sans'를 사용합니다.")

# --- 스타일 설정 ---
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'

# 사용자 정의 컬러 팔레트 (빨강, 초록, 노랑)
CUSTOM_PALETTE = ['#FF7171', '#71FFE7', '#FAF761']

# '상품명' 데이터 플로팅을 위한 DataFrame 변환
product_name_ratios_large_df = pd.DataFrame({
    'Okt': avg_large_okt_product_name,
    'Kkma': avg_large_kkma_product_name,
    'Komoran': avg_large_komoran_product_name,
    'Kiwi': avg_large_kiwi_product_name
}).drop('total_pos_count', errors='ignore')
product_name_ratios_small_df = pd.DataFrame({
    'Okt': avg_small_okt_product_name,
    'Kkma': avg_small_kkma_product_name,
    'Komoran': avg_small_komoran_product_name,
    'Kiwi': avg_small_kiwi_product_name
}).drop('total_pos_count', errors='ignore')

# Dataframe '대기업'
product_name_ratios_large_melted = product_name_ratios_large_df.T.reset_index().melt(id_vars='index')
product_name_ratios_large_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
product_name_ratios_large_melted['Group'] = '대기업'

# Dataframe '소상공인'
product_name_ratios_small_melted = product_name_ratios_small_df.T.reset_index().melt(id_vars='index')
product_name_ratios_small_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
product_name_ratios_small_melted['Group'] = '소상공인'

# 두 그룹 병합
product_name_comparison_df = pd.concat([product_name_ratios_large_melted, product_name_ratios_small_melted])

# 한글 품사명으로 변경
pos_type_mapping = {
    'noun_ratio': '명사 비율',
    'adjective_ratio': '형용사 비율',
    'numeral_ratio': '수사 비율',
    'verb_ratio': '동사 비율',
    'josa_ratio': '조사 비율',
    'eomi_ratio': '어미 비율',
    'special_char_ratio': '특수문자 비율'
}
product_name_comparison_df['POS_Type'] = product_name_comparison_df['POS_Type'].replace(pos_type_mapping);

# 요청된 순서대로 정렬
ordered_pos_types = ['명사 비율', '형용사 비율', '수사 비율', '동사 비율', '조사 비율', '어미 비율', '특수문자 비율']
product_name_comparison_df['POS_Type'] = pd.Categorical(product_name_comparison_df['POS_Type'], categories=ordered_pos_types, ordered=True)
product_name_comparison_df = product_name_comparison_df.sort_values('POS_Type')

# 각 형태소 분석기별로 그래프 생성 (Subplots 사용)
analyzers = ['Okt', 'Kkma', 'Komoran', 'Kiwi']

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.flatten()

for i, analyzer in enumerate(analyzers):
    sns.barplot(x='POS_Type', y='Average_Ratio', hue='Group', data=product_name_comparison_df[product_name_comparison_df['Analyzer'] == analyzer], palette=CUSTOM_PALETTE, ax=axes[i])
    axes[i].set_title(f'상품명 품사별 평균 비율 비교: {analyzer} (대기업 vs 소상공인)', fontsize=20, color='white')
    axes[i].set_xlabel('품사 유형', fontsize=15, color='white')
    axes[i].set_ylabel('평균 비율', fontsize=15, color='white')
    axes[i].tick_params(axis='x', rotation=45, colors='white')
    axes[i].tick_params(axis='y', colors='white')
    axes[i].legend(title='그룹', title_fontsize=15, fontsize=15, labelcolor='white', facecolor='none', edgecolor='white')
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # 각 막대 위에 빈도수 값 표시
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt='%.2f', fontsize=10, color='white')

plt.tight_layout()
plt.savefig('대기업vs소상공인_품사_상품명.png', transparent=True)

plt.show()

# '본문' 데이터를 플로팅을 위한 DataFrame으로 변환
text_contents_ratios_large_df = pd.DataFrame({
    'Okt': avg_large_okt_text_contents,
    'Kkma': avg_large_kkma_text_contents,
    'Komoran': avg_large_komoran_text_contents,
    'Kiwi': avg_large_kiwi_text_contents
}).drop('total_pos_count', errors='ignore')
text_contents_ratios_small_df = pd.DataFrame({
    'Okt': avg_small_okt_text_contents,
    'Kkma': avg_small_kkma_text_contents,
    'Komoran': avg_small_komoran_text_contents,
    'Kiwi': avg_small_kiwi_text_contents
}).drop('total_pos_count', errors='ignore')

# Dataframe '대기업'
text_contents_ratios_large_melted = text_contents_ratios_large_df.T.reset_index().melt(id_vars='index')
text_contents_ratios_large_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
text_contents_ratios_large_melted['Group'] = '대기업'

# Dataframe '소상공인'
text_contents_ratios_small_melted = text_contents_ratios_small_df.T.reset_index().melt(id_vars='index')
text_contents_ratios_small_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
text_contents_ratios_small_melted['Group'] = '소상공인'

# 두 그룹 병합
text_contents_comparison_df = pd.concat([text_contents_ratios_large_melted, text_contents_ratios_small_melted])

# 한글 품사명으로 변경
text_contents_comparison_df['POS_Type'] = text_contents_comparison_df['POS_Type'].replace(pos_type_mapping);

text_contents_comparison_df['POS_Type'] = pd.Categorical(text_contents_comparison_df['POS_Type'], categories=ordered_pos_types, ordered=True)
text_contents_comparison_df = text_contents_comparison_df.sort_values('POS_Type')

# 각 형태소 분석기별로 그래프 생성 (Subplots 사용)

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.flatten()

for i, analyzer in enumerate(analyzers):
    sns.barplot(x='POS_Type', y='Average_Ratio', hue='Group', data=text_contents_comparison_df[text_contents_comparison_df['Analyzer'] == analyzer], palette=CUSTOM_PALETTE, ax=axes[i])
    axes[i].set_title(f'본문 품사별 평균 비율 비교: {analyzer} (대기업 vs 소상공인)', fontsize=20, color='white')
    axes[i].set_xlabel('품사 유형', fontsize=15, color='white')
    axes[i].set_ylabel('평균 비율', fontsize=15, color='white')
    axes[i].tick_params(axis='x', rotation=45, colors='white')
    axes[i].tick_params(axis='y', colors='white')
    axes[i].legend(title='그룹', title_fontsize=15,fontsize=15, labelcolor='white', facecolor='none', edgecolor='white')
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # 각 막대 위에 빈도수 값 표시
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt='%.2f', fontsize=10, color='white')

plt.tight_layout()
plt.savefig('대기업vs소상공인_품사_본문.png', transparent=True)

plt.show()

# 품사별 개수 반환 함수 정의
# 특수문자 개수 세분화 함수
print("상품명 개수 카운팅 함수 정의 완료.")
# brand_dfs 내의 각 브랜드 데이터프레임에 상품명 품사 및 특수문자 개수 적용
for brand_name, brand_df in brand_dfs.items():
    # Okt 품사 개수 적용
    product_name_counts = brand_df['product_name'].apply(count_pos_tags_okt)
    for key in product_name_counts.iloc[0].keys():
        brand_df[f'product_name_okt_{key}'] = product_name_counts.apply(lambda x: x[key])

    # 상세 특수문자 개수 적용
    detailed_counts = brand_df['product_name'].apply(count_detailed_special_chars)
    brand_df['product_name_unicode_emoji_count'] = detailed_counts.apply(lambda x: x['unicode_emoji_count'])
    brand_df['product_name_textual_emoji_count'] = detailed_counts.apply(lambda x: x['textual_emoji_count'])
    brand_df['product_name_separator_count'] = detailed_counts.apply(lambda x: x['separator_count'])
    brand_df['product_name_other_symbol_count'] = detailed_counts.apply(lambda x: x['other_symbol_count'])

    # 변경된 DataFrame을 brand_dfs에 업데이트
    brand_dfs[brand_name] = brand_df

print("각 브랜드 데이터프레임에 상품명 품사 및 특수문자 개수 계산 완료.")

if 'CJ' in brand_dfs:
    print("\n'CJ' 브랜드 데이터 상품명 분석 결과 (상위 3개 행):")
    display(brand_dfs['CJ'][['product_name', 'product_name_okt_noun_count', 'product_name_unicode_emoji_count', 'product_name_textual_emoji_count', 'product_name_separator_count', 'product_name_other_symbol_count']].head(3))
else:
    print("\n'CJ' 브랜드 데이터가 없습니다.")
# brand_dfs를 사용하여 df_large_enterprise와 df_small_business를 재구성
df_large_enterprise_list = []
df_small_business_list = []

for brand_name, brand_df in brand_dfs.items():
    if not brand_df.empty:
        if brand_df['brand_type'].iloc[0] == '대기업':
            df_large_enterprise_list.append(brand_df)
        elif brand_df['brand_type'].iloc[0] == '소상공인':
            df_small_business_list.append(brand_df)

df_large_enterprise = pd.concat(df_large_enterprise_list, ignore_index=True) if df_large_enterprise_list else pd.DataFrame()
df_small_business = pd.concat(df_small_business_list, ignore_index=True) if df_small_business_list else pd.DataFrame()

# 평균 개수 계산
avg_counts_large = {
    '명사': df_large_enterprise['product_name_okt_noun_count'].mean() if not df_large_enterprise.empty else 0,
    '형용사': df_large_enterprise['product_name_okt_adjective_count'].mean() if not df_large_enterprise.empty else 0,
    '수사': df_large_enterprise['product_name_okt_numeral_count'].mean() if not df_large_enterprise.empty else 0,
    '동사': df_large_enterprise['product_name_okt_verb_count'].mean() if not df_large_enterprise.empty else 0,
    '조사': df_large_enterprise['product_name_okt_josa_count'].mean() if not df_large_enterprise.empty else 0,
    '어미': df_large_enterprise['product_name_okt_eomi_count'].mean() if not df_large_enterprise.empty else 0,
    '유니코드 이모지': df_large_enterprise['product_name_unicode_emoji_count'].mean() if not df_large_enterprise.empty else 0,
    '텍스트 이모지': df_large_enterprise['product_name_textual_emoji_count'].mean() if not df_large_enterprise.empty else 0,
    '구분자': df_large_enterprise['product_name_separator_count'].mean() if not df_large_enterprise.empty else 0,
    '기타 특수문자': df_large_enterprise['product_name_other_symbol_count'].mean() if not df_large_enterprise.empty else 0
}

avg_counts_small = {
    '명사': df_small_business['product_name_okt_noun_count'].mean() if not df_small_business.empty else 0,
    '형용사': df_small_business['product_name_okt_adjective_count'].mean() if not df_small_business.empty else 0,
    '수사': df_small_business['product_name_okt_numeral_count'].mean() if not df_small_business.empty else 0,
    '동사': df_small_business['product_name_okt_verb_count'].mean() if not df_small_business.empty else 0,
    '조사': df_small_business['product_name_okt_josa_count'].mean() if not df_small_business.empty else 0,
    '어미': df_small_business['product_name_okt_eomi_count'].mean() if not df_small_business.empty else 0,
    '유니코드 이모지': df_small_business['product_name_unicode_emoji_count'].mean() if not df_small_business.empty else 0,
    '텍스트 이모지': df_small_business['product_name_textual_emoji_count'].mean() if not df_small_business.empty else 0,
    '구분자': df_small_business['product_name_separator_count'].mean() if not df_small_business.empty else 0,
    '기타 특수문자': df_small_business['product_name_other_symbol_count'].mean() if not df_small_business.empty else 0
}

# DataFrame으로 변환
df_plot = pd.DataFrame({
    'Category': list(avg_counts_large.keys()) * 2,
    'Average_Count': list(avg_counts_large.values()) + list(avg_counts_small.values()),
    'Group': ['대기업'] * len(avg_counts_large) + ['소상공인'] * len(avg_counts_small)
})

# 시각화
plt.figure(figsize=(14, 8))
sns.barplot(x='Category', y='Average_Count', hue='Group', data=df_plot, palette=CUSTOM_PALETTE)
plt.title('상품명 품사 및 특수문자 평균 개수 비교 (Okt 분석기)', fontsize=16, color='white')
plt.xlabel('카테고리', fontsize=14, color='white')
plt.ylabel('평균 개수', fontsize=14, color='white')
plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.legend(title='그룹', fontsize=12, title_fontsize=12, labelcolor='white', facecolor='none', edgecolor='white')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 각 막대 위에 빈도수 값 표시
for container in plt.gca().containers:
    if isinstance(container, BarContainer):
        plt.bar_label(container, fmt='%.2f', fontsize=10, color='white')

plt.tight_layout()
plt.savefig('대기업vs소상공인_상품명_개수_비교.png', transparent=True)
plt.show()

# 1. 브랜드별 각 항목(품사 및 특수문자) 평균 개수 계산
brand_avg_data = []

for brand_name, brand_df in brand_dfs.items():
    if not brand_df.empty:
        avg_row = {
            'brand_name': brand_name,
            '명사': brand_df['product_name_okt_noun_count'].mean() if 'product_name_okt_noun_count' in brand_df.columns else 0,
            '유니코드 이모지': brand_df['product_name_unicode_emoji_count'].mean() if 'product_name_unicode_emoji_count' in brand_df.columns else 0,
            '텍스트 이모지': brand_df['product_name_textual_emoji_count'].mean() if 'product_name_textual_emoji_count' in brand_df.columns else 0,
            '구분자': brand_df['product_name_separator_count'].mean() if 'product_name_separator_count' in brand_df.columns else 0,
            '기타 특수문자': brand_df['product_name_other_symbol_count'].mean() if 'product_name_other_symbol_count' in brand_df.columns else 0
        }
        brand_avg_data.append(avg_row)

# DataFrame으로 변환
df_brand_avg = pd.DataFrame(brand_avg_data)

# brand_name_to_type 맵핑 생성
brand_name_to_type = df.drop_duplicates(subset=['brand_name'])[['brand_name', 'brand_type']].set_index('brand_name')['brand_type'].to_dict()

# df_brand_avg에 brand_type 컬럼 추가
df_brand_avg['brand_type'] = df_brand_avg['brand_name'].map(brand_name_to_type)

#2. 시각화를 위한 Melt (Long-form 구조로 변환)
df_plot_brand = df_brand_avg.melt(
    id_vars=['brand_name', 'brand_type'],
    var_name='Category',
    value_name='Average_Count'
)

# 대기업 및 소상공인 명사 평균 개수 데이터프레임 생성
df_large_enterprise_noun_avg = df_brand_avg[df_brand_avg['brand_type'] == '대기업'].sort_values(by='명사', ascending=False)
df_small_business_noun_avg = df_brand_avg[df_brand_avg['brand_type'] == '소상공인'].sort_values(by='명사', ascending=False)

# 2. 서브플롯 생성 (1행 3열 구조로 변경)
fig, axes = plt.subplots(1, 3, figsize=(30, 8))

# --- 첫 번째 서브플롯: 카테고리별 브랜드 비교 (기존 ax1, 명사 카테고리 포함) ---
ax1 = sns.barplot(
    data=df_plot_brand,
    x='Category',
    y='Average_Count',
    hue='brand_name',
    palette=CUSTOM_PALETTE,
    ax=axes[0]
)

axes[0].set_title('브랜드별 상품명 품사 및 특수문자 평균 개수 비교', fontsize=15, fontweight='bold', color='white') # 제목 변경
axes[0].set_xlabel('카테고리', fontsize=15, color='white')
axes[0].set_ylabel('평균 개수', fontsize=15, color='white')
axes[0].tick_params(axis='x', rotation=45, colors='white')
axes[0].tick_params(axis='y', colors='white')
axes[0].legend(title='브랜드명', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize=12, fontsize=12, labelcolor='white', facecolor='none', edgecolor='white')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# --- 두 번째 서브플롯: 대기업 브랜드별 상품명 평균 명사 개수 (새로운 ax2) ---
ax2 = sns.barplot(
    data=df_large_enterprise_noun_avg,
    x='brand_name',
    y='명사',
    palette=CUSTOM_PALETTE,
    hue='brand_name',
    legend=False,
    ax=axes[1]
)
axes[1].set_title('대기업 브랜드별 상품명 평균 명사 개수 비교', fontsize=15, fontweight='bold', color='white')
axes[1].set_xlabel('브랜드명', fontsize=15, color='white')
axes[1].set_ylabel('평균 명사 개수', fontsize=15, color='white')
axes[1].tick_params(axis='x', rotation=45, colors='white')
axes[1].tick_params(axis='y', colors='white')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

for container in ax2.containers:
    if isinstance(container, BarContainer):
        ax2.bar_label(container, fmt='%.2f', padding=3, fontsize=9, color='white')

# --- 세 번째 서브플롯: 소상공인 브랜드별 상품명 평균 명사 개수 (새로운 ax3) ---
ax3 = sns.barplot(
    data=df_small_business_noun_avg,
    x='brand_name',
    y='명사',
    palette=CUSTOM_PALETTE,
    hue='brand_name',
    legend=False,
    ax=axes[2]
)
axes[2].set_title('소상공인 브랜드별 상품명 평균 명사 개수 비교', fontsize=15, fontweight='bold', color='white')
axes[2].set_xlabel('브랜드명', fontsize=15, color='white')
axes[2].set_ylabel('평균 명사 개수', fontsize=15, color='white')
axes[2].tick_params(axis='x', rotation=45, colors='white')
axes[2].tick_params(axis='y', colors='white')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

for container in ax3.containers:
    if isinstance(container, BarContainer):
        ax3.bar_label(container, fmt='%.2f', padding=3, fontsize=9, color='white')

plt.tight_layout()

plt.savefig('브랜드별_상품명_개수_비교_세분화.png', transparent=True)
plt.show()
# 상품명에서 명사만 추출하는 함수 (Okt 사용)
# 각 브랜드별 상품명에서 명사 추출 및 빈도 계산
brand_top_nouns = {}

for brand_name, brand_df in brand_dfs.items():
    all_nouns_brand = []
    for product_name in brand_df['product_name']:
        all_nouns_brand.extend(extract_nouns(product_name))

    noun_counts_brand = Counter(all_nouns_brand)
    top_10_nouns_brand = noun_counts_brand.most_common(10)
    brand_top_nouns[brand_name] = top_10_nouns_brand

    print(f"'{brand_name}' 상품명 명사 빈도 Top 10:")
    for noun, count in top_10_nouns_brand:
        print(f"  - {noun}: {count}회")
    print("\n")

print("모든 브랜드에 대한 상품명 명사 빈도 분석 완료.")
# 각 브랜드별 상품명 명사 빈도 시각화 (subplots 통합)

# 브랜드 개수 확인
num_brands = len(brand_top_nouns)

# subplot 그리드 크기 계산 (예: 5x4 그리드)
n_cols = 4
n_rows = math.ceil(num_brands / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for i, (brand_name, top_nouns) in enumerate(brand_top_nouns.items()):
    ax = axes[i]

    if not top_nouns:
        ax.set_title(f"'{brand_name}' 분석할 명사 없음",fontsize=20, color='white')
        ax.axis('off')
        continue

    nouns = [noun for noun, count in top_nouns]
    counts = [count for noun, count in top_nouns]

    bars = sns.barplot(x=nouns, y=counts, palette=CUSTOM_PALETTE, ax=ax, hue=nouns, legend=False)
    ax.set_title(f"'{brand_name}' 상품명 명사 빈도 Top 10", fontsize=20, color='white')
    ax.set_xlabel('명사', fontsize=15, color='white')
    ax.set_ylabel('빈도수', fontsize=15, color='white')
    ax.tick_params(axis='x', rotation=45, labelsize=12, colors='white')
    ax.tick_params(axis='y', labelsize=12, colors='white')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 각 막대 위에 빈도수 값 표시
    for bar in bars.patches:
        if isinstance(bar, Rectangle):
            ax.annotate(f'{int(bar.get_height())}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='white')

# 남은 빈 subplot이 있다면 비활성화
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('모든_브랜드_상품명_명사_빈도_Top10_통합.png', transparent=True)
plt.show()

print("모든 브랜드에 대한 상품명 명사 빈도 통합 시각화 완료.")

# 각 브랜드별 평균 품사 및 특수문자 개수 계산
brand_avg_counts = []

for brand_name, brand_df in brand_dfs.items():
    if brand_df.empty:
        continue

    avg_noun = brand_df['product_name_okt_noun_count'].mean()
    avg_unicode_emoji = brand_df['product_name_unicode_emoji_count'].mean()
    avg_textual_emoji = brand_df['product_name_textual_emoji_count'].mean()
    avg_separator = brand_df['product_name_separator_count'].mean()
    avg_other_symbol = brand_df['product_name_other_symbol_count'].mean()

    brand_avg_counts.append({
        'Brand': brand_name,
        'Category': '명사',
        'Average_Count': avg_noun
    })
    brand_avg_counts.append({
        'Brand': brand_name,
        'Category': '유니코드 이모지',
        'Average_Count': avg_unicode_emoji
    })
    brand_avg_counts.append({
        'Brand': brand_name,
        'Category': '텍스트 이모지',
        'Average_Count': avg_textual_emoji
    })
    brand_avg_counts.append({
        'Brand': brand_name,
        'Category': '구분자',
        'Average_Count': avg_separator
    })
    brand_avg_counts.append({
        'Brand': brand_name,
        'Category': '기타 특수문자',
        'Average_Count': avg_other_symbol
    })

df_brand_avg_counts = pd.DataFrame(brand_avg_counts)

# 시각화를 위한 subplot 그리드 설정
num_brands = df_brand_avg_counts['Brand'].nunique()
n_cols = 4
n_rows = math.ceil(num_brands / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True)
axes = axes.flatten()

# 각 브랜드별로 시각화
for i, brand_name in enumerate(df_brand_avg_counts['Brand'].unique()):
    ax = axes[i]
    brand_data = df_brand_avg_counts[df_brand_avg_counts['Brand'] == brand_name]
    bars = sns.barplot(x='Category', y='Average_Count', data=brand_data, palette=CUSTOM_PALETTE, ax=ax, hue='Category', legend=False)

    ax.set_title(f'{brand_name} 상품명 평균 개수', fontsize=20, color='white')
    ax.set_xlabel('', color='white')
    ax.set_ylabel('평균 개수', fontsize=15, color='white')
    ax.tick_params(axis='x', rotation=45, labelsize=12, colors='white')
    ax.tick_params(axis='y', labelsize=12, colors='white')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 각 막대 위에 빈도수 값 표시
    for bar in bars.patches:
        if isinstance(bar, Rectangle):
            ax.annotate(f'{bar.get_height():.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, color='white')

# 남은 빈 subplot이 있다면 비활성화
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('브랜드별 상품명 평균 품사 및 특수문자 개수 비교', fontsize=20, y=1.02, color='white')
plt.savefig('브랜드별_상품명_명사_특수문자_개수_비교.png', transparent=True)
plt.show()

print("브랜드별 상품명 평균 품사 및 특수문자 개수 시각화 완료.")

#이모지 목록

all_extracted_unicode_emojis = set()

for brand_name, brand_df in brand_dfs.items():
    for product_name in brand_df['product_name'].dropna().astype(str):
        emojis_in_name = [emo['emoji'] for emo in emoji.emoji_list(product_name)]
        for emo in emojis_in_name:
            all_extracted_unicode_emojis.add(emo)

print(f"총 {len(all_extracted_unicode_emojis)}개의 고유 유니코드 이모지가 상품명에서 추출되었습니다.")
print("추출된 유니코드 이모지 목록:")
for emo in sorted(list(all_extracted_unicode_emojis)):
    print(f"- {emo}")

if not all_extracted_unicode_emojis:
    print("상품명에서 유니코드 이모지가 발견되지 않았습니다.")
# 1. 개수 카운팅 함수 정의

# 2. DataFrame 전체 상품명에 카운팅 함수 적용
count_results = df['product_name'].apply(analyze_product_name_counts)
df_counts = pd.DataFrame(list(count_results))

# 원본 DataFrame과 결합
df_analyzed = pd.concat([df[['product_name', 'brand_name']], df_counts], axis=1)

print("=== 상품명별 카운팅 결과 예시 ===")
display(df_analyzed.head(10))
# 3. 시각화 (상위 N개 상품명 비교)
# 상품명이 너무 많으면 시각화가 빽빽해지므로 샘플 15개 추출
sample_df = df_analyzed.head(15).copy()

# 시각화를 위한 상품명 길이 조절 (차트 라벨용)
sample_df['short_name'] = sample_df['product_name'].apply(lambda x: x[:15] + '...' if len(x) > 15 else x)

#상품명을 축약하지 않고 전체 이름 그대로 사용
sample_df['full_name'] = sample_df['product_name']

# (시각화 1) 누적 막대 그래프 (Stacked Bar Chart)
fig, ax = plt.subplots(figsize=(14,8))
categories = ['명사', '형용사', '동사', '수사', '조사', '어미', '유니코드 이모지', '텍스트 이모티콘', '구분자', '기타 특수문자']

sample_df.set_index('short_name')[categories].plot(
    kind='barh',
    stacked=True,
    ax=ax,
    colormap='viridis' # Use a colormap that works well with a dark background or define custom colors
)

plt.title('상품명 하나당 품사 및 특수문자 구성 개수 (상위 15개 샘플)', fontsize=20, fontweight='bold', color='white')
plt.xlabel('개수 카운트', fontsize=15, color='white')
plt.ylabel('상품명', fontsize=15, color='white')
plt.tick_params(axis='x', rotation=45, labelsize=15, colors='white')
plt.tick_params(axis='y', labelsize=15, colors='white')
plt.legend(title='구성 요소', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize=15,fontsize=15, labelcolor='white', facecolor='none', edgecolor='white')
plt.grid(axis='x', linestyle='--', alpha=0.6)

#각 누적 막대 조각 내부에 수치(0보다 큰 값만) 표시
for container in ax.containers:
    if isinstance(container, BarContainer):
        labels = [f'{int(v.get_width())}' if v.get_width() > 0 else '' for v in container]
        ax.bar_label(
            container,
            labels=labels,
            label_type='center',
            fontsize=15,
            color='white',
            fontweight='bold'
                )

plt.tight_layout()
plt.savefig('상품명별 품사&특수분자 구성(상위 15개).png', transparent=True)
plt.show()

# 1. 이모지 또는 이모티콘이 1개 이상 존재하는 상품명 데이터만 필터링
df_emoji_exist = df_analyzed[
    (df_analyzed['유니코드 이모지'] > 0) | (df_analyzed['텍스트 이모티콘'] > 0)
].copy()

print(f"이모지/이모티콘이 포함된 총 상품 수: {len(df_emoji_exist)}개")
# 데이터가 존재할 경우 시각화 진행
if not df_emoji_exist.empty:
    # 상위 15개 샘플 추출 (데이터가 15개 미만이면 전체 사용)
    sample_emoji_df = df_emoji_exist.head(15).copy()

    # y축 라벨용 상품명 길이 정리
    sample_emoji_df['short_name'] = sample_emoji_df['product_name'].apply(
        lambda x: x[:18] + '...' if len(x) > 18 else x
    )

    categories = ['명사', '형용사', '동사', '수사', '조사', '어미', '유니코드 이모지', '텍스트 이모티콘', '구분자', '기타 특수문자']

    # (시각화 1) 누적 막대 그래프 (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    sample_emoji_df.set_index('short_name')[categories].plot(
        kind='barh',
        stacked=True,
        ax=ax,
        colormap='viridis'
    )
    plt.title('이모지/이모티콘 포함 상품명의 구성 개수 분석 (상위 샘플)', fontsize=20, fontweight='bold', color='white')
    plt.xlabel('개수 카운트', fontsize=15, color='white')
    plt.ylabel('상품명', fontsize=15, color='white')
    plt.tick_params(axis='x', rotation=45, labelsize=15, colors='white')
    plt.tick_params(axis='y', labelsize=15, colors='white')
    plt.legend(title='구성 요소', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=15, title_fontsize=15, labelcolor='white', facecolor='none', edgecolor='white')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    #각 누적 막대 조각 내부에 수치(0보다 큰 값만) 표시
    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = [f'{int(v.get_width())}' if v.get_width() > 0 else '' for v in container]
            ax.bar_label(
                container,
                labels=labels,
            label_type='center',
                fontsize=15,
                color='white',
                fontweight='bold'
                    )

    plt.tight_layout()
    plt.savefig('상품명별 품사&특수분자 구성(이모지 포함).png', transparent=True)

    plt.show()

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
# konlpy 설치

# 형태소 분석기 초기화
okt = Okt()
kkma = Kkma()
komoran = Komoran()
kiwi = Kiwi()

print("형태소 분석기 초기화 완료: Okt, Kkma, Komoran, Kiwi")
try:
    # 커서를 생성하여 쿼리 실행 준비
    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        sql_query = "SELECT brand_name, text_contents FROM raw_data_table"
        cursor.execute(sql_query)
        data = cursor.fetchall()
        df = pd.DataFrame(data)
    print("데이터 로드 성공")
    display(df.head(10))
except pymysql.Error as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    df = pd.DataFrame()

# 데이터가 비어있을 경우를 대비
if df.empty:
    print("데이터프레임이 비어있습니다. 데이터를 가져오지 못했거나 테이블에 데이터가 없습니다.")
else:
    # 결측치 확인 및 제거 (text_contents에 결측치가 있으면 분석에 오류 발생 가능)
    initial_rows = len(df)
    df.dropna(subset=['text_contents'], inplace=True)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"{removed_rows}개의 결측치(text_contents가 없는 행)를 제거했습니다.")
    print(f"최종 데이터프레임 크기: {len(df)} 행")
# 각 형태소 분석기로 명사 추출 및 개수 계산
df['okt_nouns'] = df['text_contents'].apply(lambda x: count_nouns(x, okt))
df['okt_noun_count'] = df['okt_nouns'].apply(len)

df['kkma_nouns'] = df['text_contents'].apply(lambda x: count_nouns(x, kkma))
df['kkma_noun_count'] = df['kkma_nouns'].apply(len)

df['komoran_nouns'] = df['text_contents'].apply(lambda x: count_nouns(x, komoran))
df['komoran_noun_count'] = df['komoran_nouns'].apply(len)

df['kiwi_nouns'] = df['text_contents'].apply(lambda x: count_nouns(x, kiwi))
df['kiwi_noun_count'] = df['kiwi_nouns'].apply(len)

display(df[['brand_name', 'text_contents', 'okt_noun_count', 'kkma_noun_count', 'komoran_noun_count']].head())
brand_noun_analysis = df.groupby('brand_name')[['okt_noun_count', 'kkma_noun_count', 'komoran_noun_count','kiwi_noun_count']].mean().reset_index()

print("브랜드별 평균 명사 개수:")
display(brand_noun_analysis.sort_values(by='okt_noun_count', ascending=False))

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# --- 사용자 제공 폰트 설정 ---
font_path = '/content/Pretendard－M.ttf'

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

# 모든 텍스트 및 축 요소 색상을 흰색으로 설정
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

print("Matplotlib 한글 폰트 및 디자인 설정 완료 (Pretendard 폰트 파일을 업로드해야 제대로 적용됩니다).")
# Custom color palette
custom_palette = ['#FF7171', '#71FFE7', '#FAF761', '#A9A9A9']

# 시각화
fig, ax = plt.subplots(figsize=(30, 8))

# 배경 투명 설정
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# 데이터프레임을 복사하여 'brand_name'을 인덱스로 설정하고 플롯
plot_data = brand_noun_analysis.set_index('brand_name')

# 형태소 분석기별 색상 매핑
colors_map = {
    'okt_noun_count': '#FF7171',
    'kkma_noun_count': '#71FFE7',
    'komoran_noun_count': '#FAF761',
    'kiwi_noun_count': '#A9A9A9'
}
# 플롯할 컬럼 순서에 맞춰 색상 리스트 생성
plot_colors = [colors_map[col] for col in plot_data.columns if col in colors_map]

# Bar plot
plot_data.plot(kind='bar', ax=ax, color=plot_colors, width=0.9)

plt.title('브랜드별 평균 명사 개수 (형태소 분석기별)', fontsize=20, color='white')
plt.xlabel('브랜드 이름', fontsize=15, color='white')
plt.ylabel('평균 명사 개수', fontsize=15, color='white')
plt.xticks(rotation=45, ha='right', fontsize=15, color='white')
plt.yticks(fontsize=15, color='white')

# 범례 설정 (배경 투명, 테두리 없음)
legend = plt.legend(title='형태소 분석기', title_fontsize='15', fontsize='15', facecolor='none', edgecolor='none')
# 범례 텍스트 색상 변경
for text in legend.get_texts():
    text.set_color('white')
# 범례 제목 색상 변경
legend.get_title().set_color('white')

plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
plt.tight_layout()

# 차트에 값 수치 표시
for container in ax.containers:
    if isinstance(container,BarContainer):
        ax.bar_label(container, fmt='%.1f', label_type='edge', color='white', fontsize=9)

plt.savefig('브랜드별_명사_수.png')
plt.show()
#MySQL 연결 정보 설정

#데이터베이스 연결
conn=pymysql.connect(
    host=host_name,
    user=user_name,
    password=password_val,
    database=db_name,
    port=port_num,
    charset='utf8'
)
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
        sql = "SELECT product_name, text_contents, brand_type, page_id FROM raw_data_table"
        cursor.execute(sql)
        data = cursor.fetchall()

    # 데이터를 Pandas DataFrame으로 변환
    df = pd.DataFrame(data, columns=['product_name', 'text_contents', 'brand_type', 'page_id'])
    print("데이터를 성공적으로 로드했습니다. 처음 5개 행을 표시합니다:")
    display(df.head())

    print("\nbrand_type 컬럼의 고유 값과 그 개수:")
    display(df['brand_type'].value_counts())

except pymysql.Error as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    df = pd.DataFrame(columns=['product_name', 'text_contents', 'brand_type', 'page_id'])
analyzers = ['Okt', 'Kkma', 'Komoran', 'Kiwi']

# 품사 유형 매핑 (한글 이름으로 변경)
pos_type_mapping = {
    'noun_ratio': '명사 비율',
    'adjective_ratio': '형용사 비율',
    'numeral_ratio': '수사 비율',
    'verb_ratio': '동사 비율',
    'josa_ratio': '조사 비율',
    'eomi_ratio': '어미 비율',
    'special_char_ratio': '특수문자 비율'
}

# 품사 유형의 순서 정의
ordered_pos_types = [
    '명사 비율', '형용사 비율', '수사 비율',
    '동사 비율', '조사 비율', '어미 비율',
    '특수문자 비율'
]

print("형태소 분석기 목록 및 품사 유형 매핑 정의 완료.")
# '대기업'과 '소상공인'으로 데이터 분리
df_large_enterprise = df[df['brand_type'] == '대기업'].copy()
df_small_business = df[df['brand_type'] == '소상공인'].copy()

print("'대기업' 데이터프레임 (df_large_enterprise) 처음 5개 행:")
display(df_large_enterprise.head())

print("\n'소상공인' 데이터프레임 (df_small_business) 처음 5개 행:")
display(df_small_business.head())

print(f"\n'대기업' 데이터 개수: {len(df_large_enterprise)}개")
print(f"'소상공인' 데이터 개수: {len(df_small_business)}개")
okt = Okt()

# --- 특수문자 비율 function ---
# --- Okt ---
print("Okt 형태소 분석기 및 비율 계산 함수 정의 완료.")
kkma = Kkma()

# --- 특수문자 비율 function ---
# --- Kkma ---
print("Kkma 형태소 분석기 및 비율 계산 함수 정의 완료.")
komoran=Komoran()

# --- 특수문자 function ---
# --- Komoran ---
print("Komoran 형태소 분석기 및 비율 계산 함수 정의 완료.")
kiwi = Kiwi()

# --- 특수문자 function ---
# --- Kiwi ---
print("Kiwi 형태소 분석기 및 비율 계산 함수 정의 완료.")
try:
    # 기존 연결이 있다면 닫고 새로 연결
    if 'conn' in globals() and conn.open:
        conn.close()
        print("기존 데이터베이스 연결을 닫고 새로 연결합니다.")

    # 새로운 연결 설정
    conn = pymysql.connect(
        host=host_name,
        user=user_name,
        password=password_val,
        database=db_name,
        port=port_num,
        charset='utf8'
    )
    print("데이터베이스에 다시 연결했습니다.")

    with conn.cursor() as cursor:
        sql = "SELECT * FROM diagnosis_table"
        cursor.execute(sql)
        data_diagnosis = cursor.fetchall()

        # 컬럼 이름 가져오기
        columns = [desc[0] for desc in cursor.description]

    # 데이터를 Pandas DataFrame으로 변환
    df_diagnosis = pd.DataFrame(data_diagnosis, columns=columns)

    # 데이터가 로드되었는지 확인
    if df_diagnosis.empty:
        print("경고: diagnosis_table에서 로드된 데이터가 없습니다. 모든 'keyword_stuffing_score' 그룹 데이터프레임이 비어있을 수 있습니다.")
    else:
        print("diagnosis_table 데이터를 성공적으로 로드했습니다. 처음 5개 행을 표시합니다:")
        display(df_diagnosis.head())

        print("\ndiagnosis_table의 컬럼 목록:")
        print(df_diagnosis.columns.tolist())

        if 'keyword_stuffing_score' in df_diagnosis.columns:
            print("\nkeyword_stuffing_score 컬럼의 통계 요약:")
            display(df_diagnosis['keyword_stuffing_score'].describe())
        else:
            print("경고: 'keyword_stuffing_score' 컬럼이 diagnosis_table에 없습니다. 다른 컬럼을 사용하여 데이터를 분리해야 합니다.")

except pymysql.Error as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    df_diagnosis = pd.DataFrame(columns=['product_name', 'text_contents', 'keyword_stuffing_score'])
    print("데이터 로드 오류로 인해 빈 DataFrame이 생성되었습니다. 컬럼 정보를 확인하려면 쿼리를 수정해야 합니다.")
# raw_data_table(df)와 diagnosis_table(df_diagnosis)를 page_id를 기준으로 병합
df_merged_analysis = pd.merge(
    df[['page_id', 'product_name', 'text_contents']],
    df_diagnosis[['page_id', 'keyword_stuffing_score']],
    on='page_id',
    how='inner'
)

print("raw_data_table과 diagnosis_table 병합 완료. 처음 5개 행을 표시합니다:")
display(df_merged_analysis.head())
print(f"병합된 데이터프레임 크기: {len(df_merged_analysis)}개")

# 병합된 데이터프레임을 0.66을 기준으로 데이터 분리
df_low_stuffing = df_merged_analysis[df_merged_analysis['keyword_stuffing_score'] < 0.66].copy()
df_high_stuffing = df_merged_analysis[df_merged_analysis['keyword_stuffing_score'] >= 0.66].copy()

print("'keyword_stuffing_score' < 0.66 데이터프레임 (df_low_stuffing) 처음 5개 행:")
display(df_low_stuffing.head())

print("\n'keyword_stuffing_score' >= 0.66 데이터프레임 (df_high_stuffing) 처음 5개 행:")
display(df_high_stuffing.head())

print(f"\n'keyword_stuffing_score' < 0.66 데이터 개수: {len(df_low_stuffing)}개")
print(f"'keyword_stuffing_score' >= 0.66 데이터 개수: {len(df_high_stuffing)}개")
# df_low_stuffing 및 df_high_stuffing에 Okt 형태소 분석 적용
df_low_stuffing = apply_okt_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_okt_analysis_to_df(df_high_stuffing.copy())

print("Okt 분석을 이용한 'keyword_stuffing_score' 그룹 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
print(f"df_low_stuffing 컬럼 개수: {len(df_low_stuffing.columns)}")
print(f"df_high_stuffing 컬럼 개수: {len(df_high_stuffing.columns)}")
# df_low_stuffing 및 df_high_stuffing에 Kkma 형태소 분석 적용
df_low_stuffing = apply_kkma_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_kkma_analysis_to_df(df_high_stuffing.copy())

print("Kkma 분석을 이용한 'keyword_stuffing_score' 그룹 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
display(df_low_stuffing.head())
display(df_high_stuffing.head())
# df_low_stuffing 및 df_high_stuffing에 Komoran 형태소 분석 적용
df_low_stuffing = apply_komoran_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_komoran_analysis_to_df(df_high_stuffing.copy())

print("Komoran 분석을 이용한 'keyword_stuffing_score' 그룹 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
display(df_low_stuffing.head())
display(df_high_stuffing.head())
# df_low_stuffing 및 df_high_stuffing에 Kiwi 형태소 분석 적용
df_low_stuffing = apply_kiwi_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_kiwi_analysis_to_df(df_high_stuffing.copy())

print("Kiwi 분석을 이용한 'keyword_stuffing_score' 그룹 데이터의 형태소 분석 및 특수문자 비율 계산 완료.")
display(df_low_stuffing.head())
display(df_high_stuffing.head())
# 'keyword_stuffing_score' < 0.66 데이터 평균 비율 계산

# 원본 df_merged_analysis 에서 다시 분리하여 초기화
df_low_stuffing = df_merged_analysis[df_merged_analysis['keyword_stuffing_score'] < 0.66].copy()
df_high_stuffing = df_merged_analysis[df_merged_analysis['keyword_stuffing_score'] >= 0.66].copy()

# Okt 형태소 분석 적용
df_low_stuffing = apply_okt_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_okt_analysis_to_df(df_high_stuffing.copy())

# Kkma 형태소 분석 적용
df_low_stuffing = apply_kkma_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_kkma_analysis_to_df(df_high_stuffing.copy())

# Komoran 형태소 분석 적용
df_low_stuffing = apply_komoran_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_komoran_analysis_to_df(df_high_stuffing.copy())

# Kiwi 형태소 분석 적용
df_low_stuffing = apply_kiwi_analysis_to_df(df_low_stuffing.copy())
df_high_stuffing = apply_kiwi_analysis_to_df(df_high_stuffing.copy())

# 각 형태소 분석기별 평균 비율 계산 함수
avg_low_okt_product_name = calculate_average_ratios_by_analyzer(df_low_stuffing, 'product_name')
avg_low_okt_text_contents = calculate_average_ratios_by_analyzer(df_low_stuffing, 'text_contents')
avg_low_kkma_product_name = calculate_average_ratios_by_analyzer(df_low_stuffing, 'product_name', '_kkma')
avg_low_kkma_text_contents = calculate_average_ratios_by_analyzer(df_low_stuffing, 'text_contents', '_kkma')
avg_low_komoran_product_name = calculate_average_ratios_by_analyzer(df_low_stuffing, 'product_name', '_komoran')
avg_low_komoran_text_contents = calculate_average_ratios_by_analyzer(df_low_stuffing, 'text_contents', '_komoran')
avg_low_kiwi_product_name = calculate_average_ratios_by_analyzer(df_low_stuffing, 'product_name', '_kiwi')
avg_low_kiwi_text_contents = calculate_average_ratios_by_analyzer(df_low_stuffing, 'text_contents', '_kiwi')

# 'keyword_stuffing_score' >= 0.66 데이터 평균 비율 계산
avg_high_okt_product_name = calculate_average_ratios_by_analyzer(df_high_stuffing, 'product_name')
avg_high_okt_text_contents = calculate_average_ratios_by_analyzer(df_high_stuffing, 'text_contents')
avg_high_kkma_product_name = calculate_average_ratios_by_analyzer(df_high_stuffing, 'product_name', '_kkma')
avg_high_kkma_text_contents = calculate_average_ratios_by_analyzer(df_high_stuffing, 'text_contents', '_kkma')
avg_high_komoran_product_name = calculate_average_ratios_by_analyzer(df_high_stuffing, 'product_name', '_komoran')
avg_high_komoran_text_contents = calculate_average_ratios_by_analyzer(df_high_stuffing, 'text_contents', '_komoran')
avg_high_kiwi_product_name = calculate_average_ratios_by_analyzer(df_high_stuffing, 'product_name', '_kiwi')
avg_high_kiwi_text_contents = calculate_average_ratios_by_analyzer(df_high_stuffing, 'text_contents', '_kiwi')

print("'keyword_stuffing_score' 그룹별 평균 비율 계산 완료.")
# 나눔고딕 폰트 설치

### Matplotlib 한글 폰트 설정 (깨짐 방지)

# 운영체제에 맞는 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

print("Matplotlib 한글 폰트 설정 완료.")
analyzers = ['Okt', 'Kkma', 'Komoran', 'Kiwi']

# 품사 유형 매핑 (한글 이름으로 변경)
pos_type_mapping = {
    'noun_ratio': '명사 비율',
    'adjective_ratio': '형용사 비율',
    'numeral_ratio': '수사 비율',
    'verb_ratio': '동사 비율',
    'josa_ratio': '조사 비율',
    'eomi_ratio': '어미 비율',
    'special_char_ratio': '특수문자 비율'
}

# 품사 유형의 순서 정의
ordered_pos_types = [
    '명사 비율', '형용사 비율', '수사 비율',
    '동사 비율', '조사 비율', '어미 비율',
    '특수문자 비율'
]
# '상품명' 데이터를 플로팅을 위한 DataFrame으로 변환
product_name_ratios_low_stuffing_df = pd.DataFrame({
    'Okt': avg_low_okt_product_name,
    'Kkma': avg_low_kkma_product_name,
    'Komoran': avg_low_komoran_product_name,
    'Kiwi': avg_low_kiwi_product_name
}).drop('total_pos_count', errors='ignore')
product_name_ratios_high_stuffing_df = pd.DataFrame({
    'Okt': avg_high_okt_product_name,
    'Kkma': avg_high_kkma_product_name,
    'Komoran': avg_high_komoran_product_name,
    'Kiwi': avg_high_kiwi_product_name
}).drop('total_pos_count', errors='ignore')

# Dataframe 'low_stuffing'
product_name_ratios_low_stuffing_melted = product_name_ratios_low_stuffing_df.T.reset_index().melt(id_vars='index')
product_name_ratios_low_stuffing_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
product_name_ratios_low_stuffing_melted['Group'] = 'score < 0.66'

# Dataframe 'high_stuffing'
product_name_ratios_high_stuffing_melted = product_name_ratios_high_stuffing_df.T.reset_index().melt(id_vars='index')
product_name_ratios_high_stuffing_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
product_name_ratios_high_stuffing_melted['Group'] = 'score >= 0.66'

# 두 그룹 병합
product_name_comparison_stuffing_df = pd.concat([product_name_ratios_low_stuffing_melted, product_name_ratios_high_stuffing_melted])

# 한글 품사명으로 변경 (pos_type_mapping 재활용)
product_name_comparison_stuffing_df['POS_Type'] = product_name_comparison_stuffing_df['POS_Type'].replace(pos_type_mapping);

# 요청된 순서대로 정렬 (ordered_pos_types 재활용)
product_name_comparison_stuffing_df['POS_Type'] = pd.Categorical(product_name_comparison_stuffing_df['POS_Type'], categories=ordered_pos_types, ordered=True)
product_name_comparison_stuffing_df = product_name_comparison_stuffing_df.sort_values('POS_Type')

# 각 형태소 분석기별로 그래프 생성 (Subplots 사용)

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.flatten()

for i, analyzer in enumerate(analyzers):
    sns.barplot(x='POS_Type', y='Average_Ratio', hue='Group', data=product_name_comparison_stuffing_df[product_name_comparison_stuffing_df['Analyzer'] == analyzer], palette='viridis', ax=axes[i])
    axes[i].set_title(f'상품명 품사별 평균 비율 비교: {analyzer} (score < 0.66 vs score >= 0.66)', fontsize=20)
    axes[i].set_xlabel('품사 유형', fontsize=15)
    axes[i].set_ylabel('평균 비율', fontsize=15)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='그룹', title_fontsize=15, fontsize=15)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('키워드_스터핑_상품명.png')

plt.show()
# '본문' 데이터를 플로팅을 위한 DataFrame으로 변환
text_contents_ratios_low_stuffing_df = pd.DataFrame({
    'Okt': avg_low_okt_text_contents,
    'Kkma': avg_low_kkma_text_contents,
    'Komoran': avg_low_komoran_text_contents,
    'Kiwi': avg_low_kiwi_text_contents
}).drop('total_pos_count', errors='ignore')
text_contents_ratios_high_stuffing_df = pd.DataFrame({
    'Okt': avg_high_okt_text_contents,
    'Kkma': avg_high_kkma_text_contents,
    'Komoran': avg_high_komoran_text_contents,
    'Kiwi': avg_high_kiwi_text_contents
}).drop('total_pos_count', errors='ignore')

# Dataframe 'low_stuffing'
text_contents_ratios_low_stuffing_melted = text_contents_ratios_low_stuffing_df.T.reset_index().melt(id_vars='index')
text_contents_ratios_low_stuffing_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
text_contents_ratios_low_stuffing_melted['Group'] = 'score < 0.66'

# Dataframe 'high_stuffing'
text_contents_ratios_high_stuffing_melted = text_contents_ratios_high_stuffing_df.T.reset_index().melt(id_vars='index')
text_contents_ratios_high_stuffing_melted.columns = ['Analyzer', 'POS_Type', 'Average_Ratio']
text_contents_ratios_high_stuffing_melted['Group'] = 'score >= 0.66'

# 두 그룹 병합
text_contents_comparison_stuffing_df = pd.concat([text_contents_ratios_low_stuffing_melted, text_contents_ratios_high_stuffing_melted])

# 한글 품사명으로 변경
text_contents_comparison_stuffing_df['POS_Type'] = text_contents_comparison_stuffing_df['POS_Type'].replace(pos_type_mapping);

# 요청된 순서대로 정렬
text_contents_comparison_stuffing_df['POS_Type'] = pd.Categorical(text_contents_comparison_stuffing_df['POS_Type'], categories=ordered_pos_types, ordered=True)
text_contents_comparison_stuffing_df = text_contents_comparison_stuffing_df.sort_values('POS_Type')

# 각 형태소 분석기별로 그래프 생성 (Subplots 사용)

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.flatten()

for i, analyzer in enumerate(analyzers):
    sns.barplot(x='POS_Type', y='Average_Ratio', hue='Group', data=text_contents_comparison_stuffing_df[text_contents_comparison_stuffing_df['Analyzer'] == analyzer], palette='viridis', ax=axes[i])
    axes[i].set_title(f'본문 품사별 평균 비율 비교: {analyzer} (score < 0.66 vs score >= 0.66)', fontsize=20)
    axes[i].set_xlabel('품사 유형', fontsize=15)
    axes[i].set_ylabel('평균 비율', fontsize=15)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='그룹', title_fontsize=15, fontsize=15)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.savefig('키워드_스터핑_본문.png')

plt.show()
#데이터베이스 연결
conn=pymysql.connect(
    host=host_name,
    user=user_name,
    password=password_val,
    database=db_name,
    port=port_num,
    charset='utf8'
)
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
        # raw_data_table에서 image_text 컬럼 데이터 가져오기
        cursor.execute("SELECT image_text FROM raw_data_table WHERE image_text IS NOT NULL AND image_text != '';")
        image_text_data = cursor.fetchall()

        # 데이터프레임으로 변환
        df_image_text = pd.DataFrame(image_text_data, columns=['image_text'])

        print(f"raw_data_table에서 로드된 유효한 image_text 행 수: {len(df_image_text)}")
        display(df_image_text.head())

except pymysql.Error as e:
    print(f"raw_data_table에서 데이터 로드 중 오류 발생: {e}")
try:
    with conn.cursor() as cursor:
        # diagnosis_table에서 raw_details 컬럼 데이터 가져오기
        cursor.execute("SELECT raw_details FROM diagnosis_table WHERE raw_details IS NOT NULL AND raw_details != '';")
        raw_detail_data = cursor.fetchall()

        # 데이터프레임으로 변환
        df_raw_detail = pd.DataFrame(raw_detail_data, columns=['raw_details'])

        print(f"diagnosis_table에서 로드된 유효한 raw_details 행 수: {len(df_raw_detail)}")
        display(df_raw_detail.head())

except pymysql.Error as e:
    print(f"diagnosis_table에서 데이터 로드 중 오류 발생: {e}")
print("df_raw_detail의 'raw_details' 컬럼 상위 2개 값:")
for i, row in df_raw_detail.head(2).iterrows():
    print(f"--- Row {i} ---")
    try:
        # JSON 문자열을 파싱하여 가독성 좋게 출력
        parsed_json = json.loads(row['raw_details'])
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(row['raw_details'])
# 1. JSON에서 유효한 clean_alt 개수만 세는 함수 정의
# 2. DB 데이터 로드 및 집계 처리
# conn이 성공적으로 연결된 경우에만 데이터 처리 실행
if conn:
  try:
    with conn.cursor() as cursor:
      # page_id 기준 JOIN 쿼리
      query = """
          SELECT
              d.page_id,
              d.raw_details,
              r.brand_type
          FROM diagnosis_table d
          INNER JOIN raw_data_table r ON d.page_id = r.page_id
          WHERE d.raw_details IS NOT NULL AND d.raw_details != ''
            AND r.brand_type IS NOT NULL AND r.brand_type != '';
          """
      cursor.execute(query)
      data = cursor.fetchall()

      # 데이터프레임 생성
      df = pd.DataFrame(data, columns=["page_id", "raw_details", "brand_type"])

      # 3. 유효한 ALT 속성 개수 계산
      df["valid_alt_count"] = df["raw_details"].apply(extract_valid_alt_count)

      # 4. brand_type별 페이지 수 및 평균 ALT 개수 계산
      result_df = (
          df.groupby("brand_type")["valid_alt_count"]
          .agg(["count", "mean"])
          .reset_index()
      )
      result_df.columns = ["brand_type", "페이지 수", "평균 유효 ALT 개수"]

      # 결과 출력
      display(result_df)

  except pymysql.Error as e:
    print(f"데이터베이스 조회 오류: {e}")
else:
  print("데이터베이스 연결 실패로 데이터 처리 건너뜀.")

# 나눔고딕 폰트 설치

### Matplotlib 한글 폰트 설정 (깨짐 방지)

# 운영체제에 맞는 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Colab 등)
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

print("Matplotlib 한글 폰트 설정 완료.")
#1. 그래프 크기 및 생성
plt.figure(figsize=(7, 5))

#2. 막대그래프 그리기
colors = ['blue','red']
ax = sns.barplot(
    data=result_df,
    x='brand_type',
    y='평균 유효 ALT 개수',
    palette=colors,
    hue='brand_type',
    legend=False,
)

#3. 그래프 세부 디자인 설정
plt.title(
    '기업 유형별 평균 유효 ALT 개수 비교',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.xlabel('기업 유형', fontsize=11, fontweight='bold')
plt.ylabel('평균 유효 ALT 개수 (개)', fontsize=11, fontweight='bold')

#4. 막대 위에 정확한 수치 표시
for bar in ax.patches:
    if isinstance(bar, Rectangle):
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}개',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
                )

plt.tight_layout()
plt.savefig('기업분류별 ALT.png')

plt.show()
#1. SQL 쿼리로 brand_name 추가 조회 및 JOIN
try:
  with conn.cursor() as cursor:
    query = """
        SELECT
            d.page_id,
            d.raw_details,
            r.brand_type,
            r.brand_name
        FROM diagnosis_table d
        INNER JOIN raw_data_table r ON d.page_id = r.page_id
        WHERE d.raw_details IS NOT NULL AND d.raw_details != ''
          AND r.brand_name IS NOT NULL AND r.brand_name != '';
        """
    cursor.execute(query)
    data = cursor.fetchall()

    df = pd.DataFrame(
        data, columns=["page_id", "raw_details", "brand_type", "brand_name"]
    )

    #2. 유효 ALT 개수 계산 (기존 정의된 extract_valid_alt_count 사용)
    df["valid_alt_count"] = df["raw_details"].apply(extract_valid_alt_count)

    #전체 브랜드별 평균 유효 ALT 개수 및 페이지 수 집계
    all_brand_result = (
        df.groupby(["brand_name", "brand_type"])["valid_alt_count"]
        .agg(["count", "mean"])
        .reset_index()
    )
    all_brand_result.columns = [
        "brand_name",
        "brand_type",
        "페이지 수",
        "평균 유효 ALT 개수"
    ]

    #평균 유효 ALT 개수 기준 내림차순 정렬
    brand_result = all_brand_result.sort_values(
        by="평균 유효 ALT 개수", ascending=False
    ).reset_index(drop=True)

    print(f"총 분석된 브랜드 수: {len(all_brand_result)}개")
    display(all_brand_result)

except pymysql.Error as e:
  print(f"DB 조회 중 오류 발생: {e}")
#1. 평균 유효 ALT 개수 소수점 정리 및 내림차순 정렬
brand_result = all_brand_result.sort_values(by="평균 유효 ALT 개수", ascending=False).reset_index(drop=True)
brand_result["평균 유효 ALT 개수"] = brand_result["평균 유효 ALT 개수"].round(2)

#2.전체 브랜드 결과 출력
print(f"=== 전체 브랜드 분석 결과 (총 {len(brand_result)}개 브랜드) ===")
display(brand_result)
# 나눔고딕 폰트 설치

### Matplotlib 한글 폰트 설정 (깨짐 방지)

# 운영체제에 맞는 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

print("Matplotlib 한글 폰트 설정 완료.")
# 3. 전체 브랜드 시각화 (막대 그래프)
plt.figure(figsize=(16, 8))
ax = sns.barplot(
    data=brand_result,
    x="brand_name",
    y="평균 유효 ALT 개수",
    hue="brand_type",
    dodge=False,
    palette={"대기업": "#1f77b4", "소상공인": "#ff7f0e"}
)

# 그래프 세부 설정
plt.title(f"전체 브랜드별 평균 유효 ALT 개수 비교 (총 {len(brand_result)}개 브랜드)", fontsize=25, fontweight="bold", pad=20)
plt.xlabel("브랜드명", fontsize=15, labelpad=10)
plt.ylabel("평균 유효 ALT 개수", fontsize=15, labelpad=10)
plt.yticks(fontsize=15)
plt.xticks(rotation=60, ha="right", fontsize=15)
plt.legend(title="기업 유형", title_fontsize="15", fontsize="15")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# 막대 상단에 수치(값) 표시
for p in ax.patches:
    if isinstance(p,Rectangle):
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=15,
                xytext=(0, 3),
                textcoords="offset points"
                    )

plt.tight_layout()
plt.show()
# 2. 기업 유형별 데이터 분리 및 정렬
df_large = all_brand_result[all_brand_result["brand_type"] == "대기업"].sort_values(
    by="평균 유효 ALT 개수", ascending=False
)
df_small = all_brand_result[all_brand_result["brand_type"] == "소상공인"].sort_values(
    by="평균 유효 ALT 개수", ascending=False
)
# 3. 1행 2열 서브플롯 생성
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# ----- [왼쪽] 대기업 브랜드 차트 -----
sns.barplot(
    data=df_large,
    x="brand_name",
    y="평균 유효 ALT 개수",
    ax=axes[0],
    color="blue"
)
axes[0].set_title(f"대기업 브랜드별 평균 유효 ALT 개수 ({len(df_large)}개)", fontsize=20, fontweight="bold", pad=15)
axes[0].set_xlabel("브랜드명", fontsize=15)
axes[0].set_ylabel("평균 유효 ALT 개수", fontsize=15)
axes[0].tick_params(axis="x", labelsize=15)
axes[0].tick_params(axis="y", labelsize=15)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# 값 수치 표시 (대기업)
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom", fontsize=18, xytext=(0, 3), textcoords="offset points"
        )

# ----- [오른쪽] 소상공인 브랜드 차트 -----
sns.barplot(
    data=df_small,
    x="brand_name",
    y="평균 유효 ALT 개수",
    ax=axes[1],
    color="red"
)
axes[1].set_title(f"소상공인 브랜드별 평균 유효 ALT 개수 ({len(df_small)}개)", fontsize=20, fontweight="bold", pad=15)
axes[1].set_xlabel("브랜드명", fontsize=15)
axes[1].set_ylabel("")
axes[1].tick_params(axis="x", rotation=60,labelsize=15)
axes[1].tick_params(axis="y", labelsize=15)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

# 값 수치 표시 (소상공인)
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom", fontsize=12, xytext=(0, 3), textcoords="offset points"
        )

plt.suptitle("기업 유형별 브랜드 ALT 속성 비교", fontsize=25, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig('브랜드별 유효 ALT.png')

plt.show()
