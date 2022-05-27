# Fast API - 1

## REST API

- Representatinoal State Transfer
  - Resource, Method, Representation of Resource로 구성
- 클라이언트: 요청을 하는 플랫폼. 브라우저, 앱등이 해당됨.
- Resource: 유니크한 ID를 가지는 리소스. URI
- Method: 서버에 요청을 보내는 방색. GET, POST, PUT, PATCH, DELETE 등

### URI & URL

- URL: uniform resource Locator. 인터넷 상 자원의 위치
- URI: ... Identifier. 인터넷 상 자원을 식별하기 위한 문자열의 구성. URI는 URL을 포함하는 더 포괄적인 범위임

### Method ~ HTTP

- GET: 정보를 요청. read
- POST: 정보를 입력. create
- PUT: 정보를 업데이트. update
- PATCH: 정보를 업데이트. update
- DELETE: 정보를 삭제 delete

#### GET? POST?

- GET
  - 어떤 정보를 가져와 조회하기 위해 사용.
  - url에 변수를 포함시켜 요청한다
  - 데이터를 헤더에 포함하여 전송하므로 데이터가 url에 노출됨
  - 캐싱가능 빠르다
- POST
  - 데이터를 서버에 제출하여 추가 또는 수정하기 위해 사용
  - url에 변수(데이터)를 노출하지 않고 요청함
  - 데이터는 Body에 포함되어 노출되지 않으므로 기본 보안은 되어있음.
  - 캐싱 불가능 (다만 그 안에 아키텍처로 캐싱 가능)
- 예시
  - GET: localhost:8080/login?id=kyle
  - POST: localhost:8080/login

### Header & Body

- http 통신은 request하고, response를 받을 때 정보를 패킷에 저장한다.
- packet: Header / Body 구조임
  - Header: 보내는 주소, 받는 주소, 시간 등
  - Body: 실제 전달하는 내용

### Status Code

클라이언트 요청에 따라 서버가 어떻게 반응하는지를 알려주는 Code

- 1xx(정보): 요청을 받았고, 프로세스 진행중
- 2xx(요청): 요청을 받았고, 실행함
- 3xx(리다이렉션): 요청 완료를 위한 추가 작업이 필요
- 4xx(클라이언트 오류): 요청 문법이 잘못되거나, 처리 불가능
- 5xx(서버 오류): 서버가 요청에 대해 실패함

### 동기 & 비동기

- 동기(Sync): 서버에 요청을 보냈을 때, 응답이 돌아와야 다음 동작을 수행 할 수 있음. 즉 대기하게 됨.
  - 주피터 노트북이 동기적으로 실행되는 방식임
- 비동기(Async): 요청을 보낼 때 응답 상태와 관련 없이 다음 동작을 수행함. 즉, A, B 작업이 동시에 실행됨.

### IP & Port

- IP
  - 네트워크에 연결된 특정 PC의 주소를 나타내는 체계
  - 4덩이의 숫자로 구성된 IPv4
  - 몇가지 주소는 용도가 정해져있음.
    - localhost: 127.0.0.1
    - 0.0.0.0, 255.255.255.255: broadcast address. 로컬네트워크의 모든 장치와 소통하는 주소
  - 현재는 숫자가 더 많은 IPv6도 나옴
- Port
  - PC에 접속할 수 있는 통로.
  - 사용중인 포트는 중복 불가능함
  - 예를 들어 주피터 노트북은 기본적으로 8888을 사용
  - Port는 0 ~ 65535 까지 존재
  - 그 중 0 ~ 1024는 통신 규약에 정해짐
    - 22: SSH
    - 80: HTTP
    - 443: HTTPS

## Fast API

최근 떠오르는 파이썬 웹 프레임워크다. 플라스크와 장고가 양분한 파이썬 웹 프레임 워크에서 새롭게 등장함.

주요 특지응로는, 높은 퍼포먼스 (Node.js나 go와 대등) 플라스크처럼 쉬운 구조로 마이크로 서비스에 적합하며, Swagger 자동생성(문서), Pydantic을 이용한 Serialization등 생산성도 뛰어나다.

### Fast API vs Flask

- 간결한 문법

```python
@app.route('/', method=["GET"])

@app.get('/')
```

- 비동기 지원

```python
@app.route('/books', methods=['GET'])
def books_table_update():
    title = request.args.get('title', None)
    author = request.args.get('author', None)

@app.get('/books_title/{book_title}/author/{author}')
async def books_table_update(books_title:str, author:str):
    ...code
```

- Bulit-in API Documentation 생성 (Swagger)
- Pydantic을 이용한 데이터 Serialiaztion 및 Validation

아쉬운 점

- 아직까지는 플라스크의 유저가 더 많다! (문서가 많음!)
- ORM등 Database와 관련된 라이브러리가 적음...

### 환경설정

Poetry 가상환경 및 의존성 관리법.

- dependency resolver로 복잡한 의존성들의 버전 충돌을 방지
- virtualenv를 생성하여 격리된 환경에서 빠르게 개발이 가능
- 기존 파이선 패키지 관리 도구에서 지원하지 않는 build, publish가 가능
- pyproject.toml을 기준으로 여러 툴들의 config를 명시적으로 관리

### 실습!

자세한 과정은 pdf 참조! -> 3.1 Fast API - 1.pdf
