# impulse_rag

Requires access to OpenAI API using an access token (https://platform.openai.com/)

0. clone project

```shell
git clone https://github.com/gfoo/impulse_rag.git
cd impulse_rag
```

1. init project

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo 'OPENAI_API_KEY="xxxxxx"' > .env
```

2. download data

```shell
cd pdfs
./download_dir_UNIL.sh
cd ..
```

3. index pdfs

```shell
python indexing.py  ./pdfs ./chroma_storage
```

4. retrieve context from index

```shell
python retrieval.py ./chroma_storage/
```

5. answer question using context and llm

```shell
python generation.py ./chroma_storage/
```
