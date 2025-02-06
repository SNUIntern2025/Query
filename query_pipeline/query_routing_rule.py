from datetime import datetime
import re

# Rule-based query routing
# DB는 아예 제외하였으므로 거기에 맞춰서 코드 수정
def rule_based_routing(query: str) -> str:
    # TODO: 정확한 날짜 확인 필요
    train_data_date = datetime.strptime("2024. 02. 01", "%Y. %m. %d").date()
    # store_db_date = datetime.strptime("2024. 07. 01", "%Y. %m. %d").date()
    
    # 최근 시점의 정보 요구하는지 판별 -> 바로 web으로 routing
    external_keywords = [
        "현재", "최근", "오늘", "어제", "그저께", "요즘", "이번", "저번", "모레", "내일", "글피", "시간", "날씨", "뉴스",
        "업데이트", "실시간", "최신", "변경", "변동", "변화", "이슈", "이벤트", "행사", "올해", "작년", "내년"]
    if any(keyword in query for keyword in external_keywords):
        return "web"
    
    # 정보를 따로 찾을 필요가 없는 경우 판별 -> 바로 none으로 routing
    internal_keywords = [
        "비교", "대조", "요약", "정리", "설명"
    ]
    if any(keyword in query for keyword in internal_keywords):
        return "none"

    # 특정 시점의 정보 요구하는지 파악
    # xxxx년 패턴 파악
    info_date = ""
    ymd = 0b000
    # x년 패턴 파악
    match_year = re.search(r"\d{4}년", query)
    if match_year:
        # xxxx. 으로 변환해서 넣기
        info_date = match_year.group()[:-1] + ". "
        ymd |= (1 << 2)
    # x월 패턴 파악
    match_month = re.search(r"\d{1,2}월", query)
    if match_month:
        info_date += match_month.group()[:-1] + ". "
        ymd |= (1 << 1)
    # x일 패턴 파악
    match_day = re.search(r"\d{1,2}일", query)
    if match_day:
        info_date += match_day.group()[:-1]
        ymd |= (1 << 0)
    # yyyy.mm.dd 패턴 파악
    date_pattern = r"\d{4}\.\d{1,2}\.\d{1,2}"
    if re.match(date_pattern, query):
        info_date = re.search(date_pattern, query).group()
        ymd |= 0b111
    
    # 월이나 일이 빠진 경우 대체
    # 년은 존재하고 월이 빠진 경우 12월로 대체
    if ymd & (1 << 1) == 0 and ymd & (1 << 2) == 1:
        tmp_date = "31" if ymd & (1 << 0) == 0 else match_day.group()[:-1]
        info_date = match_year.group()[:-1] + ". " + "12. " + tmp_date
    # 년, 월이 존재하고 일만 빠진 경우 31일로 대체
    elif ymd & (1 << 0) == 0 and ymd & (0b11 << 1) == 0b11:
        info_date = match_year.group()[:-1] + ". " + match_month.group()[:-1] + ". 31"
    
    # 완성된 날짜를 가지고 비교
    if ymd & (1 << 2) != 0:
        info_date = datetime.strptime(info_date, "%Y. %m. %d").date()
        # 현재 시점보다 나중인지 확인
        now = datetime.now()
        if info_date > now.date():
            return "web"
        
        # == DB가 빠졌으므로 아래 코드도 제외 ==
        # DB에 저장된 날짜보다 나중인지 확인
        # if info_date > store_db_date:
        #     return "web"
        # 훈련 데이터 ~ DB 저장 날짜 사이인지 확인
        # if train_data_date < info_date < store_db_date:
        #     return "db"

        # 훈련 데이터보다 이전 시점인지 확인
        if info_date < train_data_date:
            return "none"
    return "llm"    # 키워드와 날짜로도 알 수 없으면 일단 llm으로 다시 판단.