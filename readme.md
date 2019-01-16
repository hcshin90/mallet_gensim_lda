## Introduction

Mallet LDA JAVA 코드를 Python gensim Mallet Wrappers로 사용  

#### gensim_mallet.py

Mallet Wrapper를 사용하여 LDA 모델을 생성하고, 결과를 출력함
(Terminal에서 실행시 log 출력을 받아볼 수 있음)  

#### ldamallet_params.py
gensim_mallet.py에서 사용하는 파라미터를 입력받기 위한 파일  

#### pyvis.ipynb
gensim_mallet.py에서 생성한 LDA 모델 결과를
pyldavis 라이브러리를 사용하여 시각화하는 코드  

<br>

## requirements

gensim 3.4.0
pyLDAvis 2.1.1
numpy 1.14.0 

<br>

## Sample data

샘플 데이터는  \n으로 다큐먼트가 구분되며
인풋 토큰 이외에 다른 데이터가 컬럼에 존재한다면,
\t로 구분되면서 인풋 토큰의 컬럼이 마지막에 있는 형태임
모든 Output은 Output 폴더에 생성됨.

