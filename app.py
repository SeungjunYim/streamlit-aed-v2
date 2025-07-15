import streamlit as st

# 제목 및 설명
st.title("🚨 새로운 탐구용 AED 시뮬레이터")
st.write("이건 두 번째 탐구용 웹앱입니다. 아래에서 위치를 선택하고 AED 경로를 탐색해보세요.")

# 선택할 수 있는 위치 리스트
locations = [
    "1-1 교실", "1-2 교실", "1-3 교실",
    "2-1 교실", "2-2 교실", "3층 복도",
    "과학실", "보건실", "체육관 입구"
]

# 현재 위치 선택
start = st.selectbox("📍 현재 위치를 선택하세요", locations)

# 버튼 클릭 시 결과 표시
if st.button("🛣️ AED 추천 경로 탐색하기"):
    st.success(f"✅ {start}에서 가장 가까운 AED는 보건실입니다. 복도를 따라 오른쪽으로 이동하세요.")
