from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import JSONRetrievalChain
import uvicorn

app = FastAPI()

# RAG 시스템 초기화
rag = JSONRetrievalChain(source_uri="").create_chain()
rag_retriever = rag.retriever
model = rag.model

# 사용자 입력 데이터 모델 정의
class UserData(BaseModel):
    weather: dict
    time: str
    location: str
    activity_type: str
    gender: str
    activity_style: str

# 추천 API 엔드포인트
@app.post("/recommend")
def recommend_outfit(user_data: UserData):
    # Query 생성
    query_keywords = [
        user_data.gender,
        user_data.activity_style,
        user_data.activity_type,
        f"온도 {user_data.weather['temperature']}도",
        f"체감온도 {user_data.weather['feels_like']}도",
        f"바람 {user_data.weather['wind_speed']}m/s",
        user_data.time,
        user_data.location
    ]
    query = " ".join(query_keywords)

    # 카테고리 목록
    categories = ["TOP", "BOTTOM", "SHOES", "ACC", "OUTER"]

    # 각 카테고리별로 최소 1개 이상의 아이템을 검색하여 가져오기
    retrieved_items_per_category = {}

    for category in categories:
        # 카테고리별로 검색 쿼리 생성
        category_query = f"{query} {category}"
        # 문서 검색
        retrieved_docs = rag_retriever.get_relevant_documents(category_query)
        # 해당 카테고리의 아이템 필터링
        category_docs = [doc for doc in retrieved_docs if doc.metadata.get('category', '').upper() == category]
        # 최소 1개 이상의 아이템 확보
        if len(category_docs) < 1:
            # 아이템이 없을 경우 더 많은 결과를 가져옴
            additional_docs = rag_retriever.get_relevant_documents(category_query)
            category_docs.extend([doc for doc in additional_docs if doc.metadata.get('category', '').upper() == category])
            # 중복 제거
            category_docs = list({doc.metadata['id']: doc for doc in category_docs}.values())
        # 가져온 아이템 저장
        retrieved_items_per_category[category] = category_docs
        
    # 프롬프트 생성 함수 정의
    def create_dynamic_prompt(user_data):
        prompt_template = """
        당신은 사용자의 상황에 맞춰 옷차림을 추천해주는 패션 어시스턴트입니다.
        아래 제공된 정보를 활용하여 적절한 옷차림을 추천해 주세요.
        반드시 JSON 형식으로만 답변하고, 다른 텍스트는 포함하지 마세요.

        # 사용자 정보:
        - 성별: {gender}
        - 활동 스타일: {activity_style}
        - 활동 타입: {activity_type}
        - 날씨 정보:
            - 온도: {temperature}℃
            - 체감온도: {feels_like}℃
            - 강수확률: {precipitation_chance}%
            - 습도: {humidity}%
            - 풍속: {wind_speed}m/s
        - 시간: {time}
        - 위치: {location}

        # 추천 아이템:
        {retrieved_items}

        # 요청사항:
        오늘의 날씨와 활동에 어울리는 상의(TOP), 하의(BOTTOM), 신발(SHOES), 겉옷(OUTER) 악세사리(ACC)를 추천해 주세요.
        각 카테고리별로 하나의 아이템을 선택해주세요.
        결과는 다음의 JSON 형식으로만 반환해 주세요:

        {{
            "TOP": id,
            "BOTTOM": id,
            "SHOES": id,
            "OUTER": id,
            "ACC1": id,
            "ACC2": id,
            "Comment": ""
        }}

        # 유의사항:
        - 결과는 반드시 JSON 형식으로만 반환하세요.
        - Comment에는 옷차림에 대한 간단한 코멘트를 적어주세요.
        """
        
        retrieved_items = ""
        for category, items in retrieved_items_per_category.items():
            retrieved_items += f"{category}:\n"
            for item in items:
                retrieved_items += f"- {item.metadata['name']}, id: {item.metadata['id']}\n"

        prompt = prompt_template.format(
            gender=user_data.gender,
            activity_style=user_data.activity_style,
            activity_type=user_data.activity_type,
            temperature=user_data.weather["temperature"],
            feels_like=user_data.weather["feels_like"],
            precipitation_chance=user_data.weather["precipitation_chance"],
            humidity=user_data.weather["humidity"],
            wind_speed=user_data.weather["wind_speed"],
            time=user_data.time,
            location=user_data.location,
            retrieved_items=retrieved_items,
        )
        return prompt

    # 프롬프트 생성
    prompt = create_dynamic_prompt(user_data)

    # 모델에 프롬프트 전달하여 응답 생성
    response = model.predict(prompt)

    # JSON 응답 반환
    try:
        import json
        result = json.loads(response)
        feels_like_temp = user_data.weather['feels_like']
        
        if feels_like_temp > 21:
            result['OUTER'] = None
        else:
            result['ACC2'] = None
            
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="모델의 응답이 올바른 JSON 형식이 아닙니다.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
