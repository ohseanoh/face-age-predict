# face-age-predict
capstone design project
data set download (AI Hub 안면 인식 에이징(aging) 이미지 데이터) (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71415)

data file 정리:

개인별로 구분되어 있는 사진 파일들 한 곳으로 복사하기
개인별로 구분되어 있는 레이블 파일들 한 곳으로 복사하기
나이별로 구분해서 classification 폴더에 넣기: 전 연령을 8개의 구간으로 나누어서

data processing: JSON 파일의 정보로 얼굴 영역만 남도록 잘라주고 모든 이미지 파일을 (224,224)크기로 resize

CNN 모델 설계: VGG16

모델 학습

모델 평가
